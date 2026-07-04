from app.services.auth import get_current_user_id
from fastapi import APIRouter, Depends, HTTPException, status
from app.services.rag import run_rag
from app.core.exceptions import AppError
from uuid import UUID
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/recipe", tags=["Recipe"])

class RecipeRequest(BaseModel):
    query: Optional[str] = "Suggest a recipe based on my pantry"
    preferences: Optional[List[str]] = None

def parse_toon_recipe(text: str) -> dict:
    """Parses a recipe in TOON format to the structure expected by the frontend."""
    lines = text.strip().split("\n")
    title = "Generated Recipe"
    gap = None
    tags = []
    ingredients = []
    steps = []
    missing_ingredients = []
    
    current_section = None
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Check for headers
        if line_str.startswith("title:"):
            title = line_str[len("title:"):].strip()
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
        elif current_section == "instructions" and line.startswith("  "):
            steps.append(line_str)
            
    return {
        "recipe": {
            "title": title,
            "ingredients": [{"name": ing["name"]} for ing in ingredients],
            "steps": steps,
            "tags": tags,
            "servings": None,
            "prepTimeMinutes": None,
            "cookTimeMinutes": None,
            "matchedPantryItemIds": []
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
            
        recipe_text = run_rag(query_text, current_user_id)
        return parse_toon_recipe(recipe_text)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recipe: {str(e)}",
        )

