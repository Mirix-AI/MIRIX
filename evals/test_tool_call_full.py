"""
Test with the FULL MetaMemory system prompt to see if tool selection changes.
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# Read actual MetaMemory prompt
with open("mirix/prompts/system/base/meta_memory_agent.txt", "r") as f:
    full_system_prompt = f.read()

# Full tool set matching what MetaMemory actually gets (BASE_TOOLS + META_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
tools = [
    # BASE_TOOLS
    {
        "type": "function",
        "function": {
            "name": "send_intermediate_message",
            "description": "Sends an intermediate message to the human user. Meanwhile, whenever this function is called, the agent needs to include the `topic` of the current focus. It should NEVER be any questions or requests for the user but only the agent's current progress on the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message contents. All unicode (including emojis) are supported."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": "Search prior conversation history using case-insensitive string matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "page": {"type": "integer", "description": "Page number (0-indexed)."}
                },
                "required": ["query"]
            }
        }
    },
    # META_MEMORY_TOOLS
    {
        "type": "function",
        "function": {
            "name": "trigger_memory_update",
            "description": "Choose which memory to update. This function will trigger another memory agent which is specifically in charge of handling the corresponding memory to update its memory. Trigger all necessary memory updates at once. Put the explanations in the `internal_monologue` field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'The types of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge_vault", "semantic". For instance, [\'episodic\', \'resource\'].'
                    }
                },
                "required": ["memory_types"]
            }
        }
    },
    # UNIVERSAL_MEMORY_TOOLS
    {
        "type": "function",
        "function": {
            "name": "search_in_memory",
            "description": "Search in memory for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "memory_type": {"type": "string", "description": "Type of memory to search."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_memory_update",
            "description": "Complete the memory update process. Must be called after all memory updates are done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Status of the memory update."}
                },
                "required": ["status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_memory_within_timerange",
            "description": "List memory items within a specific time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time in ISO format."},
                    "end_time": {"type": "string", "description": "End time in ISO format."}
                },
                "required": ["start_time", "end_time"]
            }
        }
    }
]

user_message = """[User Message] I had a great dinner with my friend Sarah last night at the new Italian restaurant downtown. She recommended a book called 'Atomic Habits' which I'm going to start reading this weekend."""

messages = [
    {"role": "system", "content": full_system_prompt},
    {"role": "user", "content": user_message}
]

model = "anthropic/claude-haiku-4.5"
print(f"Testing: {model} with FULL MetaMemory prompt")
print(f"System prompt length: {len(full_system_prompt)} chars")
print(f"Number of tools: {len(tools)}")
print()

# Run 3 times to check consistency
for i in range(3):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=512,
    )

    msg = response.choices[0].message
    print(f"Run {i+1}: finish_reason={response.choices[0].finish_reason}")

    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            print(f"  Tool: {tc.function.name}, Args: {json.dumps(args, ensure_ascii=False)}")
    else:
        print(f"  Content: {msg.content[:300] if msg.content else 'None'}")
    print()
