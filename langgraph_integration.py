import os

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from mirix import Mirix
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

# Configuration
# OPENAI_API_KEY = 'sk-xxx'  # Replace with your actual OpenAI API key
# MEM0_API_KEY = 'your-mem0-key'  # Replace with your actual Mem0 API key

# Initialize LangChain and Mem0
llm = ChatOpenAI(model="gpt-4o-mini")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

mirix_agent = Mirix(
  api_key=API_KEY,
  model_provider="azure_opena",
  config_path="mirix/configs/mirix_azure_example.yaml",
)

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    user_id: str

graph = StateGraph(State)

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["user_id"]

    try:

        system_message = mirix_agent.construct_system_message(messages[-1].content, user_id=user_id)

        full_messages = [system_message] + messages

        response = llm.invoke(full_messages)

        # Store the interaction with Mirix
        try:
            interaction = f"User: {messages[-1].content}\n\nAssistant: {response.content}"
            mirix_agent.add(interaction, user_id=user_id)
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

def run_conversation(user_input: str, mem0_user_id: str):

    user = mirix_agent.create_user(user_name='alice')

    config = {"configurable": {"thread_id": user.id}}
    state = {"messages": [HumanMessage(content=user_input)], "user_id": user.id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return

if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")
    mem0_user_id = "alice"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, mem0_user_id)

