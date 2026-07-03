from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from uuid import UUID

from app.models.pantry import PantryItem
from app.services.pantry import (
    add_pantry_item as add_pantry_item_svc,
    add_pantry_item_bulk as add_pantry_item_bulk_svc,
    process_embedding_queue_task,
)
from app.services.auth import get_current_user_id
from app.core.exceptions import AppError

router = APIRouter(prefix="/pantry", tags=["Pantry"])


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_pantry_item(
    pantry_item: PantryItem,
    background_tasks: BackgroundTasks,
    current_user_id: UUID = Depends(get_current_user_id),
):
    try:
        pantry_item.owner_id = current_user_id
        await add_pantry_item_svc(pantry_item, background_tasks)
        return {"status": "success", "id": str(pantry_item.id)}
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add pantry item: {str(e)}",
        )


@router.post("/add/bulk", status_code=status.HTTP_201_CREATED)
async def add_pantry_items(
    pantry_items: list[PantryItem],
    background_tasks: BackgroundTasks,
    current_user_id: UUID = Depends(get_current_user_id),
):
    try:
        for item in pantry_items:
            item.owner_id = current_user_id
        await add_pantry_item_bulk_svc(pantry_items, background_tasks)
        return {"status": "success", "count": len(pantry_items)}
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add pantry items: {str(e)}",
        )


@router.post("/embed/process", status_code=status.HTTP_202_ACCEPTED)
async def process_embeddings(
    background_tasks: BackgroundTasks,
):
    try:
        background_tasks.add_task(process_embedding_queue_task)
        return {"status": "success", "message": "Embedding queue processor triggered"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger queue processing: {str(e)}",
        )

