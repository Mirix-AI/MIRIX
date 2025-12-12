#!/usr/bin/env python3
"""
Claude Code Agent with all standard tools
"""

import asyncio

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, query
from dotenv import load_dotenv

load_dotenv()

# Configuration
KEEP_LAST_N_TURNS = 50  # Keep last N turns in conversation history


def build_system_prompt():
    """Build system prompt

    Returns:
        str: System prompt
    """
    # Base system prompt
    system_prompt = """You are a helpful assistant."""

    return system_prompt.replace("\n", "  ")


def get_agent_options(system_prompt, session_id=None):
    """Get ClaudeAgentOptions with the given system prompt

    Args:
        system_prompt: System prompt to use
        session_id: Optional session ID to resume

    Returns:
        ClaudeAgentOptions: Agent configuration
    """
    allowed_tools = [
        "Task",  # Launch specialized agents
        "Bash",  # Execute shell commands
        "Glob",  # File pattern matching
        "Grep",  # Search in files
        "ExitPlanMode",  # Exit planning mode
        "Read",  # Read files
        "Edit",  # Edit files
        "MultiEdit",  # Multiple edits in one file
        "Write",  # Write files
        "NotebookEdit",  # Edit Jupyter notebooks
        "WebFetch",  # Fetch web content
        "TodoWrite",  # Task management
        "WebSearch",  # Search the web
        "BashOutput",  # Get background bash output
        "KillBash",  # Kill background bash processes
    ]
    if session_id:
        return ClaudeAgentOptions(
            resume=session_id,
            # Include all standard Claude Code tools
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            model="claude-sonnet-4-5",
            max_turns=50,
        )

    else:
        return ClaudeAgentOptions(
            # Include all standard Claude Code tools
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            model="claude-sonnet-4-5",
            max_turns=50,
        )


async def run_agent():

    # Track conversation
    conversation_history = []
    turn_count = 0
    session_id = None

    try:
        while True:
            system_prompt = build_system_prompt()
            options = get_agent_options(system_prompt, session_id)

            try:
                user_input = input("User: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("Agent: ", end="", flush=True)

                assistant_message = None

                async for message in query(prompt=user_input, options=options):
                    # The first message is a system init message with the session ID
                    if hasattr(message, "subtype") and message.subtype == "init":
                        session_id = message.data.get("session_id")
                        print(f"Session started with ID: {session_id}")
                        # You can save this ID for later resumption

                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                print(block.text, end="\n", flush=True)

                    if isinstance(message, AssistantMessage):
                        assistant_message = message

                assistant_response_strs = []
                for block in assistant_message.content:
                    if hasattr(block, "text"):
                        assistant_response_strs.append(block.text)
                assistant_response = "\n".join(assistant_response_strs)

                # Update conversation history
                conversation_history.append((user_input, assistant_response))
                # Keep only last N turns
                if len(conversation_history) > KEEP_LAST_N_TURNS:
                    conversation_history = conversation_history[-KEEP_LAST_N_TURNS:]
                turn_count += 1

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()

    finally:
        pass  # No cleanup needed with query() function


if __name__ == "__main__":
    asyncio.run(run_agent())
