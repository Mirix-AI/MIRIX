#!/usr/bin/env python3
"""
Claude Code Agent with all standard tools plus n8n-mcp server
"""

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from dotenv import load_dotenv

# Load .env file (optional - Mirix now loads .env automatically in mirix/settings.py)
# Kept here for backward compatibility
load_dotenv()


async def create_comprehensive_agent():
    """Create a Claude Code agent with all tools plus n8n-mcp server"""

    # Configure agent with all standard Claude Code tools plus MCP servers
    options = ClaudeAgentOptions(
        # Include all standard Claude Code tools
        allowed_tools=[
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
        ],
        system_prompt="""You are a helpful assistant.""",
        model="claude-4-sonnet-20250514",
        max_turns=50,
        extra_args={"verbose": True},
    )

    return options


async def run_agent():
    """Run the comprehensive Claude Code agent"""
    options = await create_comprehensive_agent()

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("User: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("Agent: ", end="", flush=True)
                await client.query(user_input)
                async for message in client.receive_response():
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                print(block.text, end="\n", flush=True)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_agent())
