from __future__ import annotations

from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import Settings, get_settings


def get_gemini_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings | None:
    if settings.google_genai_api_key is None:
        return None

    return GoogleGenerativeAIEmbeddings(
        api_key=settings.google_genai_api_key,
        model=settings.gemini_embeddings_model,
        task_type="retrieval_document",
        output_dimensionality=settings.gemini_embeddings_output_dimensionality,
    )


@lru_cache(maxsize=1)
def cached_gemini_embeddings() -> GoogleGenerativeAIEmbeddings | None:
    return get_gemini_embeddings(get_settings())


def require_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    embeddings = cached_gemini_embeddings()
    if embeddings is None:
        raise RuntimeError("Gemini embeddings are not configured")
    return embeddings
