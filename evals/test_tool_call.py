"""
Minimal test: Does Claude Haiku via OpenRouter correctly call tools?
This isolates the tool selection behavior from the MIRIX agent framework.
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY required")

client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# Simulate MetaMemory Agent's tool set
tools = [
    {
        "type": "function",
        "function": {
            "name": "trigger_memory_update",
            "description": "Choose which memory to update. This function will trigger another memory agent to update its memory. Trigger all necessary memory updates at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'The types of memory to update. Choose from: "core", "episodic", "resource", "knowledge_vault", "semantic".'
                    }
                },
                "required": ["memory_types"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_intermediate_message",
            "description": "Sends an intermediate message to the human user about current progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message contents."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_memory_update",
            "description": "Complete the memory update process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status message."
                    }
                },
                "required": ["status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_memory",
            "description": "Search memories for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Simplified MetaMemory system prompt
system_prompt = """You are the Meta Memory Manager. When messages are sent to you, analyze them and save details into corresponding memories by calling `trigger_memory_update`.

Select memory types from: ['core', 'episodic', 'semantic', 'resource', 'knowledge_vault'].

After triggering updates, call `finish_memory_update` to complete."""

user_message = """[User Message] I had a great dinner with my friend Sarah last night at the new Italian restaurant downtown. She recommended a book called 'Atomic Habits' which I'm going to start reading this weekend."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message}
]

# Test with different models
models_to_test = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
]

for model in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")

    try:
        # Test 1: Without strict mode (our fix)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )

        msg = response.choices[0].message
        print(f"finish_reason: {response.choices[0].finish_reason}")
        print(f"has_tool_calls: {bool(msg.tool_calls)}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  Tool: {tc.function.name}")
                print(f"  Args: {tc.function.arguments}")
        else:
            print(f"  Content: {msg.content[:200] if msg.content else 'None'}")

        # Test 2: WITH strict mode (the bug we fixed)
        print(f"\n--- With strict: true ---")
        strict_tools = json.loads(json.dumps(tools))
        for t in strict_tools:
            t["function"]["strict"] = True
            t["function"]["parameters"]["additionalProperties"] = False

        response2 = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=strict_tools,
            tool_choice="auto",
            max_tokens=512,
        )

        msg2 = response2.choices[0].message
        print(f"finish_reason: {response2.choices[0].finish_reason}")
        print(f"has_tool_calls: {bool(msg2.tool_calls)}")

        if msg2.tool_calls:
            for tc in msg2.tool_calls:
                print(f"  Tool: {tc.function.name}")
                print(f"  Args: {tc.function.arguments}")
        else:
            print(f"  Content: {msg2.content[:200] if msg2.content else 'None'}")

    except Exception as e:
        print(f"  ERROR: {e}")
