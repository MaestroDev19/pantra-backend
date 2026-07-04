from __future__ import annotations
from uuid import UUID
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from app.services.tools.retrival_tool import retriver_tool
from app.services.tools.generate_query import generate_query_or_respond
from app.services.tools.other_tools import grade_document, rewrite_question, generate_recipe, extract_text_content
workflow_builder = StateGraph(MessagesState)

workflow_builder.add_node("generate_query_or_respond", generate_query_or_respond)
workflow_builder.add_node("retrieve", ToolNode([retriver_tool]))
workflow_builder.add_node("rewrite_question", rewrite_question)
workflow_builder.add_node("generate_recipe", generate_recipe)

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

workflow_builder.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_document
)

workflow_builder.add_edge("generate_recipe", END)
workflow_builder.add_edge("rewrite_question", "generate_query_or_respond")
graph = workflow_builder.compile()

def run_rag(query: str, owner_id: UUID):
    # Run the graph synchronously to completion
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"owner_id": owner_id}}
    )
    
    # Extract the last message (the generated recipe/response)
    final_message = result["messages"][-1]
    return extract_text_content(final_message.content)