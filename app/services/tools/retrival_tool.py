from typing import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import InjectedToolArg
from app.services.vector_store import get_vector_store

@tool
def retrive_pantry_items(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> list[dict]:
    """Retrieves pantry items from the pantry based on the query.
    Returns a list of dicts with 'content' and 'id' keys."""
    owner_id = str(config["configurable"]["owner_id"])  # set per-request, not by the LLM

    retriever = get_vector_store().as_retriever(
        search_kwargs={"filter": {"owner_id": owner_id}}
    )
    retrieved_docs = retriever.invoke(query)
    return [
        {
            "content": doc.page_content,
            "id": doc.metadata.get("id", None),
            "name": doc.metadata.get("name", doc.page_content),
        }
        for doc in retrieved_docs
    ]

retriver_tool = retrive_pantry_items

if __name__ == "__main__":
    test = retrive_pantry_items.invoke({"query": "is there any cheese in my pantry"})
    print(test)