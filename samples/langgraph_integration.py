import os
import sys

# Add the mirix module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
mirix_root = os.path.dirname(current_dir)
sys.path.insert(0, mirix_root)

# Load .env file (optional - Mirix now loads .env automatically in mirix/settings.py)
# Kept here for backward compatibility and explicit clarity
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(mirix_root, ".env"))

import logging

from mirix.client import MirixClient
from mirix.schemas.agent import AgentType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from typing import Annotated, List, TypedDict  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

# Initialize LangChain and Mirix
API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=API_KEY,
)

# Define user and organization IDs
user_id = "demo-user"
client_id = "demo-client-app"
org_id = "demo-org"

# Build absolute path to config file (since we're in samples/ subdirectory)
config_path = os.path.join(mirix_root, "mirix/configs/examples/mirix_gemini.yaml")

client = MirixClient(
    api_key=None,  # TODO: add authentication later
    client_id=client_id,
    org_id=org_id,
    debug=True,
)

client.initialize_meta_agent(config_path=config_path, update_agents=True)


class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    user_id: str


graph = StateGraph(State)


def format_memories_for_prompt(memories: dict) -> str:
    """
    Format the memories dictionary returned by retrieve_with_conversation() into a string.

    Args:
        memories: Dictionary containing retrieved memories organized by type

    Returns:
        Formatted string representation of memories
    """
    if not memories or not memories.get("memories"):
        return "No relevant memories found."

    formatted_parts = []

    # ✅ FIX: Format core memory first (most important - contains user's name and persona)
    if "core" in memories["memories"] and memories["memories"]["core"].get("items"):
        formatted_parts.append("\n=== CORE MEMORY ===")
        for block in memories["memories"]["core"]["items"]:
            label = block.get("label", "").upper()
            value = block.get("value", "")
            if value:  # Only show non-empty blocks
                formatted_parts.append(f"{label}: {value}")
        formatted_parts.append("")  # Empty line after core memory

    # Format other memory types
    for memory_type, items in memories["memories"].items():
        if memory_type == "core":  # Already handled above
            continue
        if items and items.get("items"):
            formatted_parts.append(f"\n{memory_type.upper()} MEMORIES:")
            for item in items["items"][:3]:  # Show top 3 items per type
                # Extract the most relevant field based on memory type
                if memory_type == "episodic":
                    text = item.get("details", item.get("summary", ""))
                elif memory_type == "semantic":
                    text = item.get("name", "") + ": " + item.get("summary", "")
                elif memory_type == "procedural":
                    text = item.get("summary", "")
                elif memory_type == "resource":
                    text = item.get("title", "") + ": " + item.get("summary", "")
                elif memory_type == "knowledge":
                    text = item.get("caption", "")
                else:
                    text = str(item)[:100]

                formatted_parts.append(f"  - {text[:150]}")

    return "\n".join(formatted_parts) if formatted_parts else "No relevant memories found."


def create_user_message(text: str) -> List[dict]:
    """
    Create a structured user message for Mirix retrieve_with_conversation() API.

    Args:
        text: The user's message text

    Returns:
        List containing a single user message dictionary
    """
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def create_mirix_messages(user_content: str, assistant_content: str) -> List[dict]:
    """
    Create structured messages for Mirix client.add() API.

    Args:
        user_content: The user's message content
        assistant_content: The assistant's response content

    Returns:
        List of message dictionaries with role and content structure
    """
    return [
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
    ]


def chatbot(state: State):
    messages = state["messages"]
    user_id = state["user_id"]

    try:
        # Create structured message for memory retrieval
        retrieval_messages = create_user_message(messages[-1].content)

        memories = client.retrieve_with_conversation(user_id=user_id, messages=retrieval_messages, limit=10)

        # Format memories into a readable string
        formatted_memories = format_memories_for_prompt(memories)

        system_message = (
            "You are a helpful assistant that can answer questions and help with tasks. "
            "You have the following memories:\n" + formatted_memories
        )

        full_messages = [system_message] + messages

        response = llm.invoke(full_messages)

        # Store the interaction with Mirix
        try:
            mirix_messages = create_mirix_messages(
                user_content=messages[-1].content, assistant_content=response.content
            )
            client.add(user_id=user_id, messages=mirix_messages)
        except Exception as e:
            print(f"Error saving memory: {e}")

        return {"messages": [response]}

    except Exception as e:
        print(f"Error in chatbot: {e}")
        # Fallback response without memory context
        response = llm.invoke(messages)
        return {"messages": [response]}


graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()


def run_conversation(user_input: str, user_name: str):
    # Use the pre-configured user_id from client initialization
    config = {"configurable": {"thread_id": user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "user_id": user_id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            # Check if value is not None and has messages
            if value is not None and value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return


if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")
    user_name = "John Doe"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, user_name)
