from __future__ import annotations

from typing import Any, Dict, List, Optional
from functools import lru_cache
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.services.supabase import get_supabase_client
from app.services.embeddings import require_gemini_embeddings
from app.models.pantry import PantryItem

class CompatibleSupabaseVectorStore(SupabaseVectorStore):
    """
    Subclass of SupabaseVectorStore that supports newer postgrest/supabase-py 
    versions where request builders do not have a mutable `.params` attribute,
    and maps default vector store columns to our custom pantry_items schema.
    """
    

    def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        rows: List[Dict[str, Any]] = []
        for idx, embedding in enumerate(vectors):
            doc = documents[idx]
            meta = doc.metadata or {}
            row = {
                "name": doc.page_content,
                "embedding": embedding,
                "embedding_metadata": meta,
                "embedding_status": "ready",
                "owner_id": meta.get("owner_id"),
                "household_id": meta.get("household_id"),
                "category": meta.get("category", "General"),
                "expiry_date": meta.get("expiry_date"),
            }
            if ids is not None:
                row["id"] = ids[idx]
            rows.append(row)

        id_list: List[str] = []
        chunk_size = self.chunk_size
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            # Use upsert to resolve conflicts and guarantee idempotency
            result = self._client.from_(self.table_name).upsert(chunk).execute()

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            ids_returned = [str(r.get("id")) for r in result.data if r.get("id")]
            id_list.extend(ids_returned)

        return id_list

    def add_pantry_item_single(self, pantry_item: PantryItem) -> None:
        self.add_documents(
            documents=[
                Document(
                    page_content=pantry_item.name,
                    metadata={
                        "owner_id": str(pantry_item.owner_id) if pantry_item.owner_id else None,
                        "household_id": str(pantry_item.household_id),
                        "category": pantry_item.category,
                        "expiry_date": str(pantry_item.expiry_date) if pantry_item.expiry_date else None,
                    },
                )
            ],
            ids=[str(pantry_item.id)],
        )

    def add_pantry_items_bulk(self, pantry_items: List[PantryItem], batch_size: int = 20) -> None:
        documents = []
        ids = []
        for item in pantry_items:
            documents.append(
                Document(
                    page_content=item.name,
                    metadata={
                        "owner_id": str(item.owner_id) if item.owner_id else None,
                        "household_id": str(item.household_id),
                        "category": item.category,
                        "expiry_date": str(item.expiry_date) if item.expiry_date else None,
                    },
                )
            )
            ids.append(str(item.id))
        self.add_documents(documents, ids=ids, batch_size=batch_size)


_vector_store_instance: SupabaseVectorStore | None = None

@lru_cache(maxsize=1)
def get_vector_store() -> SupabaseVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        settings = get_settings()
        supabase = get_supabase_client(settings)
        if supabase is None:
            raise AppError("Supabase is not configured for vector store", status_code=500)

        embeddings = require_gemini_embeddings()
        _vector_store_instance = CompatibleSupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="pantry_items",
            query_name="match_pantry_items",
        )
    return _vector_store_instance
