from __future__ import annotations

from datetime import datetime, timezone
from typing import List
import anyio
import hashlib
from fastapi import BackgroundTasks

from app.services.vector_store import get_vector_store
from app.services.supabase import get_supabase_client
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.models.pantry import PantryItem
from app.core.logging import get_logger

logger = get_logger(__name__)


async def process_embedding_queue_task(batch_size: int = 20) -> None:
    logger.info("Background task started to process embedding queue (batch size: %d)", batch_size)
    try:
        vector_store = await get_vector_store()
        settings = get_settings()
        supabase = get_supabase_client(settings)
        if supabase is None:
            raise RuntimeError("Supabase client is not configured")

        # 1. Claim enqueued jobs using SELECT FOR UPDATE SKIP LOCKED
        response = await anyio.to_thread.run_sync(
            lambda: supabase.rpc("claim_pantry_embedding_jobs", {"p_batch_size": batch_size}).execute()
        )

        jobs = response.data or []
        if not jobs:
            logger.info("No enqueued embedding jobs to process.")
            return

        logger.info("Claimed %d embedding jobs from queue", len(jobs))

        # Map pantry_item_id to job_id
        job_map = {job["pantry_item_id"]: job["job_id"] for job in jobs}
        item_ids = list(job_map.keys())

        # 2. Fetch the corresponding pantry items details
        items_response = await anyio.to_thread.run_sync(
            lambda: supabase.table("pantry_items")
            .select("id, name, category, expiry_date")
            .in_("id", item_ids)
            .execute()
        )

        # Handle enqueued items that were deleted in the meantime
        fetched_ids = {item["id"] for item in (items_response.data or [])}
        orphaned_item_ids = set(item_ids) - fetched_ids
        for item_id in orphaned_item_ids:
            job_id = job_map[str(item_id)]
            logger.warning("Cleaning up enqueued job %d for non-existent pantry item %s", job_id, item_id)
            await anyio.to_thread.run_sync(
                lambda j=job_id: supabase.rpc("complete_pantry_embedding_job", {"p_job_id": j}).execute()
            )

        if not items_response.data:
            return

        embeddings_service = vector_store.embeddings

        # 3. Process each pantry item
        for item in items_response.data:
            item_id = item["id"]
            job_id = job_map[item_id]
            try:
                name = str(item.get("name", "")).strip()
                category = str(item.get("category", "")).strip()
                expiry_date = str(item.get("expiry_date", "")).strip()
                text = f"name: {name}\ncategory: {category}\nexpiry_date: {expiry_date}"

                # Generate embedding
                vectors = await embeddings_service.aembed_documents([text])
                vector = vectors[0]

                generated_at = datetime.now(timezone.utc).isoformat()
                input_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                metadata = {
                    "provider": "google_genai",
                    "model": settings.gemini_embeddings_model,
                    "dimensions": settings.gemini_embeddings_output_dimensionality,
                    "input_sha256": input_sha256,
                    "input_length": len(text),
                    "generated_at": generated_at,
                }

                # Save embedding to the pantry item
                await anyio.to_thread.run_sync(
                    lambda v=vector, g=generated_at, m=metadata, id_=item_id: supabase.table("pantry_items")
                    .update({
                        "embedding": v,
                        "embedding_status": "ready",
                        "embedding_updated_at": g,
                        "embedding_metadata": m,
                        "embedding_error": None,
                    })
                    .eq("id", id_)
                    .execute()
                )

                # Delete enqueued job on completion
                await anyio.to_thread.run_sync(
                    lambda j=job_id: supabase.rpc("complete_pantry_embedding_job", {"p_job_id": j}).execute()
                )
                logger.info("Successfully computed and saved embedding for pantry item %s", item_id)

            except Exception as e:
                logger.error("Failed to compute embedding for pantry item %s: %s", item_id, e)
                # Mark job as failed (handles backoff or DLQ)
                await anyio.to_thread.run_sync(
                    lambda j=job_id, err_msg=str(e): supabase.rpc("fail_pantry_embedding_job", {
                        "p_job_id": j,
                        "p_error_message": err_msg,
                    }).execute()
                )

    except Exception as e:
        logger.error("Error running process_embedding_queue_task: %s", e, exc_info=True)


async def add_pantry_item(pantry_item: PantryItem, background_tasks: BackgroundTasks) -> None:
    logger.info("Adding pantry item synchronously: %s", pantry_item.name)
    settings = get_settings()
    supabase = get_supabase_client(settings)
    if supabase is None:
        raise AppError("Supabase is not configured", status_code=500)

    try:
        vector_store = await get_vector_store()
        embeddings_service = vector_store.embeddings

        name = str(pantry_item.name).strip()
        category = str(pantry_item.category).strip()
        expiry_date = str(pantry_item.expiry_date).strip() if pantry_item.expiry_date else ""
        text = f"name: {name}\ncategory: {category}\nexpiry_date: {expiry_date}"

        # Generate embedding synchronously/directly
        vectors = await embeddings_service.aembed_documents([text])
        vector = vectors[0]

        generated_at = datetime.now(timezone.utc).isoformat()
        input_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = {
            "provider": "google_genai",
            "model": settings.gemini_embeddings_model,
            "dimensions": settings.gemini_embeddings_output_dimensionality,
            "input_sha256": input_sha256,
            "input_length": len(text),
            "generated_at": generated_at,
        }

        # Insert item with status 'ready'. Triggers will NOT enqueue it because status is 'ready'.
        payload = {
            "id": str(pantry_item.id),
            "owner_id": str(pantry_item.owner_id) if pantry_item.owner_id else None,
            "household_id": str(pantry_item.household_id) if pantry_item.household_id else None,
            "name": pantry_item.name,
            "category": pantry_item.category,
            "expiry_date": str(pantry_item.expiry_date) if pantry_item.expiry_date else None,
            "embedding": vector,
            "embedding_status": "ready",
            "embedding_updated_at": generated_at,
            "embedding_metadata": metadata,
        }

        await anyio.to_thread.run_sync(
            lambda: supabase.table("pantry_items").upsert(payload).execute()
        )
        logger.info("Pantry item inserted directly with embedding.")

    except Exception as e:
        logger.error("Failed to add pantry item: %s", e, exc_info=True)
        raise AppError(f"Failed to add pantry item: {str(e)}", status_code=500)


async def add_pantry_item_bulk(pantry_items: List[PantryItem], background_tasks: BackgroundTasks) -> None:
    logger.info("Adding %d pantry items in bulk", len(pantry_items))
    settings = get_settings()
    supabase = get_supabase_client(settings)
    if supabase is None:
        raise AppError("Supabase is not configured", status_code=500)

    try:
        payload = [
            {
                "id": str(item.id),
                "owner_id": str(item.owner_id) if item.owner_id else None,
                "household_id": str(item.household_id) if item.household_id else None,
                "name": item.name,
                "category": item.category,
                "expiry_date": str(item.expiry_date) if item.expiry_date else None,
            }
            for item in pantry_items
        ]

        await anyio.to_thread.run_sync(
            lambda: supabase.table("pantry_items").upsert(payload).execute()
        )
        logger.info("Bulk pantry items inserted. Enqueueing background queue worker processing.")

        # Trigger background queue worker execution
        background_tasks.add_task(process_embedding_queue_task)

    except Exception as e:
        logger.error("Failed to add pantry items in bulk: %s", e, exc_info=True)
        raise AppError(f"Failed to add pantry items in bulk: {str(e)}", status_code=500)
