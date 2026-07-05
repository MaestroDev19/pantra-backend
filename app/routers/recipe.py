from app.services.auth import get_current_user_id
from fastapi import APIRouter, Depends, HTTPException, status
from app.services.rag import run_rag
from app.core.exceptions import AppError
from uuid import UUID
from pydantic import BaseModel
from typing import Optional, List
import json
import re

router = APIRouter(prefix="/recipe", tags=["Recipe"])

class RecipeRequest(BaseModel):
    query: Optional[str] = "Suggest a recipe based on my pantry"
    preferences: Optional[List[str]] = None


def _safe_int(value: str) -> Optional[int]:
    """Try to parse an int from a string, return None on failure."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _build_pantry_id_map(rag_messages: list) -> dict[str, str]:
    """Build a map of lowercased pantry item name → item ID from the RAG
    tool messages. The retrieval tool now returns dicts with 'id' and 'name'."""
    name_to_id: dict[str, str] = {}
    for msg in rag_messages:
        # Tool messages contain the retriever output
        if getattr(msg, "type", None) != "tool":
            continue
        content = msg.content
        # The tool returns a stringified list of dicts
        if isinstance(content, str):
            try:
                items = json.loads(content.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                continue
        elif isinstance(content, list):
            items = content
        else:
            continue

        for item in items:
            if isinstance(item, dict) and item.get("id") and item.get("name"):
                name_to_id[item["name"].strip().lower()] = str(item["id"])
    return name_to_id


def parse_toon_recipe(text: str, pantry_id_map: Optional[dict[str, str]] = None) -> dict:
    """Parses a recipe in TOON format to the structure expected by the frontend."""
    lines = text.strip().split("\n")
    title = "Generated Recipe"
    gap = None
    tags = []
    ingredients = []
    steps = []
    missing_ingredients = []
    servings = None
    prep_minutes = None
    cook_minutes = None
    matched_pantry_item_ids: list[str] = []

    current_section = None

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Check for headers
        if line_str.startswith("title:"):
            title = line_str[len("title:"):].strip()
            current_section = None
        elif line_str.startswith("servings:"):
            servings = _safe_int(line_str[len("servings:"):])
            current_section = None
        elif line_str.startswith("prep_minutes:"):
            prep_minutes = _safe_int(line_str[len("prep_minutes:"):])
            current_section = None
        elif line_str.startswith("cook_minutes:"):
            cook_minutes = _safe_int(line_str[len("cook_minutes:"):])
            current_section = None
        elif line_str.startswith("gap:"):
            gap_val = line_str[len("gap:"):].strip()
            gap = None if gap_val.lower() == "null" or not gap_val else gap_val
            current_section = None
        elif line_str.startswith("tags["):
            current_section = "tags"
        elif line_str.startswith("ingredients["):
            current_section = "ingredients"
        elif line_str.startswith("instructions["):
            current_section = "instructions"
        elif current_section == "tags" and line.startswith("  "):
            tags.append(line_str)
        elif current_section == "ingredients" and line.startswith("  "):
            # Format: name,source (e.g. lemon,pantry)
            parts = line_str.split(",")
            ing_name = parts[0].strip()
            source = parts[1].strip() if len(parts) > 1 else "buy"
            ingredients.append({"name": ing_name, "source": source})
            if source == "buy":
                missing_ingredients.append(ing_name)
            elif source == "pantry" and pantry_id_map:
                # Try to match this ingredient to a pantry item ID
                item_id = pantry_id_map.get(ing_name.lower())
                if item_id:
                    matched_pantry_item_ids.append(item_id)
        elif current_section == "instructions" and line.startswith("  "):
            steps.append(line_str)

    return {
        "recipe": {
            "title": title,
            "ingredients": [{"name": ing["name"]} for ing in ingredients],
            "steps": steps,
            "tags": tags,
            "servings": servings,
            "prepTimeMinutes": prep_minutes,
            "cookTimeMinutes": cook_minutes,
            "matchedPantryItemIds": matched_pantry_item_ids
        },
        "missingIngredients": missing_ingredients
    }

@router.post("/generate", status_code=status.HTTP_201_CREATED)
async def generate_recipe(
    request: RecipeRequest,
    current_user_id: UUID = Depends(get_current_user_id),
):
    try:
        # Build prompt from query and preferences
        query_text = request.query or ""
        if request.preferences:
            prefs = ", ".join(request.preferences)
            query_text += f" (Preferences: {prefs})"

        recipe_text, rag_messages = run_rag(query_text, current_user_id)

        # Build a mapping of pantry item names → IDs from the retrieval context
        pantry_id_map = _build_pantry_id_map(rag_messages)

        return parse_toon_recipe(recipe_text, pantry_id_map)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recipe: {str(e)}",
        )

