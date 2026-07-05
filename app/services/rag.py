from __future__ import annotations
from uuid import UUID
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from app.services.tools.retrival_tool import retriver_tool
from app.services.tools.generate_query import generate_query_or_respond
from app.services.tools.other_tools import grade_document, generate_recipe, generate_no_match_recipe, extract_text_content
workflow_builder = StateGraph(MessagesState)

workflow_builder.add_node("generate_query_or_respond", generate_query_or_respond)
workflow_builder.add_node("retrieve", ToolNode([retriver_tool]))
workflow_builder.add_node("generate_recipe", generate_recipe)
workflow_builder.add_node("generate_no_match_recipe", generate_no_match_recipe)

workflow_builder.add_edge(START, "generate_query_or_respond")

def route_on_tool_call(state:MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

workflow_builder.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    route_on_tool_call,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Fork, not a loop: grade_document routes to exactly one of two generation
# nodes, and both terminate at END. No edge anywhere points back to
# generate_query_or_respond or retrieve — structurally cannot cycle.
workflow_builder.add_conditional_edges(
    "retrieve",
    grade_document,
    {
        "generate_recipe": "generate_recipe",
        "generate_no_match_recipe": "generate_no_match_recipe",
    },
)

workflow_builder.add_edge("generate_recipe", END)
workflow_builder.add_edge("generate_no_match_recipe", END)
graph = workflow_builder.compile()

def run_rag(query: str, owner_id: UUID) -> tuple[str, list]:
    """Run the RAG graph and return (recipe_text, all_messages)."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"owner_id": owner_id}}
    )
    
    # Extract the last message (the generated recipe/response)
    final_message = result["messages"][-1]
    recipe_text = extract_text_content(final_message.content)
    return recipe_text, result["messages"]