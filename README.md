![Mirix Logo](https://github.com/RenKoya1/MIRIX/raw/main/assets/logo.png)

## MIRIX - Multi-Agent Personal Assistant with an Advanced Memory System

Your personal AI that builds memory through screen observation and natural conversation

| 🌐 [Website](https://mirix.io) | 📚 [Documentation](https://docs.mirix.io) | 📄 [Paper](https://arxiv.org/abs/2507.07957) | 💬 [Discord](https://discord.gg/S6CeHNrJ) 
<!-- | [Twitter/X](https://twitter.com/mirix_ai) | [Discord](https://discord.gg/S6CeHNrJ) | -->

---

### Key Features 🔥

- **Multi-Agent Memory System:** Six specialized memory components (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) managed by dedicated agents
- **Screen Activity Tracking:** Continuous visual data capture and intelligent consolidation into structured memories  
- **Privacy-First Design:** All long-term data stored locally with user-controlled privacy settings
- **Advanced Search:** PostgreSQL-native BM25 full-text search with vector similarity support
- **Multi-Modal Input:** Text, images, voice, and screen captures processed seamlessly

### Quick Start
**End-Users**: For end-users who want to build your own memory using MIRIX, please checkout the quick installation guide [here](https://docs.mirix.io/getting-started/installation/#quick-installation-dmg).

**Developers**: For users who want to apply our memory system as the backend, please check out our [Backend Usage](https://docs.mirix.io/user-guide/backend-usage/). Basically, you just need to run:
```
git clone git@github.com:Mirix-AI/MIRIX.git
cd MIRIX

# Create and activate virtual environment
python -m venv mirix_env
source mirix_env/bin/activate  # On Windows: mirix_env\Scripts\activate

pip install -r requirements.txt
```
Then you can run the following python code:
```python
from mirix.agent import AgentWrapper

# Initialize agent with configuration
agent = AgentWrapper("./mirix/configs/mirix.yaml")

# Send basic text information
agent.send_message(
    message="The moon now has a president.",
    memorizing=True,
    force_absorb_content=True
)
```
For more details, please refer to [Backend Usage](https://docs.mirix.io/user-guide/backend-usage/).

## Python SDK (NEW!) 🎉

We've created a simple [Python SDK](https://pypi.org/project/mirix/0.1.5/) that makes it incredibly easy to integrate Mirix's memory capabilities into your applications:

### Installation
https://pypi.org/project/mirix/0.1.5
```bash
pip install mirix
```

### Quick Start with SDK
```python
from mirix import Mirix

# Initialize memory agent (defaults to Google Gemini 2.0 Flash)
memory_agent = Mirix(api_key="your-google-api-key")

# Add memories
memory_agent.add("The moon now has a president")
memory_agent.add("John loves Italian food and is allergic to peanuts")

# Chat with memory context
response = memory_agent.chat("Does the moon have a president?")
print(response)  # "Yes, according to my memory, the moon has a president."

response = memory_agent.chat("What does John like to eat?") 
print(response)  # "John loves Italian food. However, he's allergic to peanuts."
```

## Integration with Claude Agent SDK 🤝

Mirix can be integrated with [Anthropic's Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/python) to give Claude persistent memory across conversations. This allows Claude to remember context, user preferences, and past interactions.

### Basic Setup

Here's a simple example of integrating Mirix with the Claude Agent SDK:

```python
#!/usr/bin/env python3
import os
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage
from mirix import Mirix
from collections import deque
from dotenv import load_dotenv
load_dotenv()

# Configuration
MEMORY_UPDATE_INTERVAL = 3  # Update memory every N turns
REINIT_INTERVAL = 1  # Rebuild system prompt (retrieve from Mirix) every N turns
KEEP_LAST_N_TURNS = 50  # Keep last N turns in memory buffer

def build_system_prompt(mirix_agent=None, user_id=None, conversation_buffer=""):
    """Build system prompt with optional Mirix memory context"""
    system_prompt = """You are a helpful assistant."""
    
    # Add Mirix memory context if available
    if mirix_agent and user_id and conversation_buffer:
        memory_context = mirix_agent.extract_memory_for_system_prompt(
            conversation_buffer, user_id
        )
        if memory_context:
            system_prompt += "\n\nRelevant Memory Context:\n" + memory_context
    
    return system_prompt

async def run_agent():
    """Run Claude Agent SDK with Mirix memory integration"""
    
    # Initialize Mirix memory agent
    mirix_agent = Mirix(
        model_name="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    user = mirix_agent.create_user(user_name="Alice")
    
    # Track conversation for memory updates
    conversation_history = deque(maxlen=KEEP_LAST_N_TURNS)
    turn_count = 0
    turns_since_reinit = 0
    session_id = None
    
    while True:
        # Build system prompt with memory context
        options = ClaudeAgentOptions(
            resume=session_id,
            allowed_tools=["Task", "Bash", "Read", "Edit", "Write", "WebSearch"],
            system_prompt=build_system_prompt(
                mirix_agent, user.id, 
                "\n".join([f"User: {u}\nAssistant: {a}" for u, a in conversation_history])
            ),
            model="claude-sonnet-4-5",
            max_turns=50
        )
        
        user_input = input("User: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        # Get Claude's response
        assistant_response = ""
        async for message in query(prompt=user_input, options=options):
            if hasattr(message, 'subtype') and message.subtype == 'init':
                session_id = message.data.get('session_id')
            
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, 'text'):
                        assistant_response += block.text
                        print(block.text, flush=True)
        
        # Update conversation history
        conversation_history.append((user_input, assistant_response))
        turn_count += 1
        turns_since_reinit += 1
        
        # Periodically update Mirix memory
        if turn_count % MEMORY_UPDATE_INTERVAL == 0:
            combined = "\n".join([
                f"[User] {u}\n[Assistant] {a}" 
                for u, a in conversation_history
            ])
            await asyncio.to_thread(mirix_agent.add, combined, user_id=user.id)
            print("✅ Memory updated!")

if __name__ == "__main__":
    asyncio.run(run_agent())
```

### Key Benefits

- **Persistent Memory:** Mirix helps Claude remember facts, preferences, and context across sessions
- **Intelligent Retrieval:** Mirix automatically retrieves relevant memories for each conversation
- **Scalable:** Works with conversations of any length without token limit issues
- **Flexible Updates:** Configure how often to update memory (e.g., every N turns)

### Example Usage

```bash
cd samples

# Install dependencies
pip install mirix claude-agent-sdk python-dotenv

# Set environment variables
export GEMINI_API_KEY="your-google-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Run the agent
python claude_agent.py
```

## License

Mirix is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact us at `yuwang@mirix.io`

## Join Our Community

Connect with other Mirix users, share your thoughts, and get support:

### 💬 Discord Community
Join our Discord server for real-time discussions, support, and community updates:
**[https://discord.gg/S6CeHNrJ](https://discord.gg/S6CeHNrJ)**

### 🎯 Weekly Discussion Sessions
We host weekly discussion sessions where you can:
- Discuss issues and bugs
- Share ideas about future directions
- Get general consultations and support
- Connect with the development team and community

**📅 Schedule:** Friday nights, 8-9 PM PST  
**🔗 Zoom Link:** [https://ucsd.zoom.us/j/96278791276](https://ucsd.zoom.us/j/96278791276)

### 📱 WeChat Group
If the WeChat QR code expired, you can add the account `ari_asm` so that I can add you to the group chat.
<div align="center">
<img src="frontend/public/wechat-qr.jpg" alt="WeChat QR Code" width="200"/><br/>
<strong>WeChat Group</strong>
</div>

## Acknowledgement
We would like to thank [Letta](https://github.com/letta-ai/letta) for open-sourcing their framework, which served as the foundation for the memory system in this project.
