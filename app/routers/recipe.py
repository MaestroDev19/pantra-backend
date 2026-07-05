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


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs often add despite instructions."""
    lines = text.strip().split("\n")
    # Strip leading/trailing ``` lines (with optional language tag)
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines)


def parse_toon_recipe(text: str, pantry_id_map: Optional[dict[str, str]] = None) -> dict:
    import logging
    logger = logging.getLogger(__name__)

    cleaned = _strip_fences(text)
    logger.info("=== RAW TOON TEXT (after fence strip) ===\n%s\n=== END TOON ===", cleaned)

    lines = cleaned.strip().split("\n")
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
    step_marker = re.compile(r'^[\-\*]?\s*\d+[\.\)]\s+')

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        low = line_str.lower()

        if low.startswith("title:"):
            title = line_str[len("title:"):].strip()
            current_section = None
        elif low.startswith("servings:"):
            servings = _safe_int(line_str[len("servings:"):])
            current_section = None
        elif low.startswith("prep_minutes:"):
            prep_minutes = _safe_int(line_str[len("prep_minutes:"):])
            current_section = None
        elif low.startswith("cook_minutes:"):
            cook_minutes = _safe_int(line_str[len("cook_minutes:"):])
            current_section = None
        elif low.startswith("gap:"):
            gap_val = line_str[len("gap:"):].strip()
            gap = None if gap_val.lower() == "null" or not gap_val else gap_val
            current_section = None
        elif low.startswith("tags[") or low == "tags:":
            current_section = "tags"
        elif low.startswith("ingredients[") or low == "ingredients:":
            current_section = "ingredients"
        elif low.startswith("instructions[") or low.startswith("steps[") or low in ("instructions:", "steps:"):
            current_section = "instructions"
        elif current_section == "tags":
            tag = line_str.lstrip("- ").strip()
            if tag:
                tags.append(tag)
        elif current_section == "ingredients":
            entry = line_str.lstrip("- ").strip()
            parts = entry.rsplit(",", 1)  # split on LAST comma — protects names with commas
            ing_name = parts[0].strip()
            source = parts[1].strip().lower() if len(parts) > 1 else "buy"
            if source not in ("pantry", "buy"):
                source = "buy"  # malformed source defaults safe
            if ing_name:
                ingredients.append({"name": ing_name, "source": source})
                if source == "buy":
                    missing_ingredients.append(ing_name)
                elif source == "pantry" and pantry_id_map:
                    item_id = pantry_id_map.get(ing_name.lower())
                    if item_id:
                        matched_pantry_item_ids.append(item_id)
        elif current_section == "instructions":
            m = step_marker.match(line_str)
            step = line_str[m.end():].strip() if m else line_str.lstrip("- ").strip()
            if step:
                steps.append(step)

    logger.info("Parsed: title=%s servings=%s prep=%s cook=%s tags=%d ingredients=%d steps=%d",
                title, servings, prep_minutes, cook_minutes, len(tags), len(ingredients), len(steps))

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

        print(f"\n{'='*60}\nRAW RECIPE TEXT:\n{recipe_text}\n{'='*60}\n")

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

