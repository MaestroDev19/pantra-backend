from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from app.core.config import get_settings
from app.services.tools.retrival_tool import retriver_tool

model = get_settings().gemini_model
chat_model = init_chat_model(
    model,
    model_provider="google_genai",
    api_key=get_settings().google_genai_api_key,
    temperature=get_settings().gemini_temperature,
)

SYSTEM_PROMPT = (
    "You are a smart pantry assistant. "
    "To suggest a recipe, you must search the user's pantry using the `retrive_pantry_items` tool. "
    "Always search the pantry before responding."
)

def generate_query_or_respond(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    model_response = chat_model.bind_tools([retriver_tool]).invoke(messages)
    return {"messages": [model_response]}