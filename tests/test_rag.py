from __future__ import annotations
from uuid import uuid4
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from app.services.tools.other_tools import grade_document, GradeDocuments
from app.services.rag import graph

def test_grade_document_empty_context():
    state = {
        "messages": [
            HumanMessage(content="Make sushi"),
            ToolMessage(content="[]", name="retrive_pantry_items", tool_call_id="1")
        ]
    }
    
    with patch("app.services.tools.other_tools.grader_model") as mock_grader:
        result = grade_document(state)
        assert result == "generate_no_match_recipe"
        mock_grader.assert_not_called()

def test_grade_document_relevant():
    state = {
        "messages": [
            HumanMessage(content="Make sushi"),
            ToolMessage(content="[{'name': 'salmon', 'id': '123'}]", name="retrive_pantry_items", tool_call_id="1")
        ]
    }
    
    with patch("app.services.tools.other_tools.grader_model") as mock_grader:
        mock_structured = MagicMock()
        mock_grader.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeDocuments(binary_score="yes")
        
        result = grade_document(state)
        assert result == "generate_recipe"
        mock_grader.with_structured_output.assert_called_once_with(GradeDocuments)

def test_grade_document_not_relevant():
    state = {
        "messages": [
            HumanMessage(content="Make sushi"),
            ToolMessage(content="[{'name': 'apple', 'id': '123'}]", name="retrive_pantry_items", tool_call_id="1")
        ]
    }
    
    with patch("app.services.tools.other_tools.grader_model") as mock_grader:
        mock_structured = MagicMock()
        mock_grader.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeDocuments(binary_score="no")
        
        result = grade_document(state)
        assert result == "generate_no_match_recipe"
        mock_grader.with_structured_output.assert_called_once_with(GradeDocuments)

@patch("app.services.tools.generate_query.chat_model")
@patch("app.services.tools.retrival_tool.get_vector_store")
@patch("app.services.tools.other_tools.grader_model")
@patch("app.services.tools.other_tools.recipe_model")
def test_full_rag_graph_pantry_match(mock_recipe, mock_grader, mock_get_vector_store, mock_chat):
    owner_id = uuid4()
    
    # 1. generate_query_or_respond makes tool call
    mock_chat_response = AIMessage(
        content="",
        tool_calls=[{"name": "retrive_pantry_items", "args": {"query": "sushi"}, "id": "call_1"}]
    )
    mock_chat.bind_tools.return_value.invoke.return_value = mock_chat_response
    
    # 2. Mock vector store & retriever returning salmon
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_get_vector_store.return_value = mock_vector_store
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        Document(page_content="salmon", metadata={"id": "123", "name": "salmon"})
    ]
    
    # 3. Grader returns "yes"
    mock_grader_structured = MagicMock()
    mock_grader.with_structured_output.return_value = mock_grader_structured
    mock_grader_structured.invoke.return_value = GradeDocuments(binary_score="yes")
    
    # 4. Recipe model returns the recipe
    mock_recipe.invoke.return_value = AIMessage(content="Pantry Sushi Recipe")
    
    # Run the graph
    result = graph.invoke(
        {"messages": [HumanMessage(content="Make sushi")]},
        config={"configurable": {"owner_id": owner_id}}
    )
    
    # Assertions
    final_message = result["messages"][-1]
    assert final_message.content == "Pantry Sushi Recipe"
    
    mock_recipe.invoke.assert_called_once()
    mock_grader_structured.invoke.assert_called_once()

@patch("app.services.tools.generate_query.chat_model")
@patch("app.services.tools.retrival_tool.get_vector_store")
@patch("app.services.tools.other_tools.grader_model")
@patch("app.services.tools.other_tools.recipe_model")
def test_full_rag_graph_no_match(mock_recipe, mock_grader, mock_get_vector_store, mock_chat):
    owner_id = uuid4()
    
    # 1. generate_query_or_respond makes tool call
    mock_chat_response = AIMessage(
        content="",
        tool_calls=[{"name": "retrive_pantry_items", "args": {"query": "sushi"}, "id": "call_1"}]
    )
    mock_chat.bind_tools.return_value.invoke.return_value = mock_chat_response
    
    # 2. Mock vector store & retriever returning empty list
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_get_vector_store.return_value = mock_vector_store
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = []
    
    # 3. Recipe model returns the buy-list recipe
    mock_recipe.invoke.return_value = AIMessage(content="Buy Sushi Recipe")
    
    # Run the graph
    result = graph.invoke(
        {"messages": [HumanMessage(content="Make sushi")]},
        config={"configurable": {"owner_id": owner_id}}
    )
    
    # Assertions
    final_message = result["messages"][-1]
    assert final_message.content == "Buy Sushi Recipe"
    
    # Grader model should NOT be called because context is empty/NONE
    mock_grader.with_structured_output.assert_not_called()
    mock_recipe.invoke.assert_called_once()
