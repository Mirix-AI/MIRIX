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
**Step 1: Backend & Dashboard (Docker):**
```
docker compose up -d --pull always
```
- Dashboard: http://localhost:5173  
- API: http://localhost:8531  

**Step 2: Create an API key in the dashboard (http://localhost:5173) and set as the environmental variable `MIRIX_API_KEY`.**

**Step 3: Client (Python, `mirix-client`, https://pypi.org/project/mirix-client/):**
```
pip install mirix-client
```

Now you are ready to go! See the example below:
```python
from mirix import MirixClient

client = MirixClient(
    api_key="your-api-key",
    base_url="http://localhost:8531",
)

client.initialize_meta_agent(
    config={
        "llm_config": {
            "model": "gemini-2.0-flash",
            "model_endpoint_type": "google_ai",
            "api_key": "your-api-key-here",
            "model_endpoint": "https://generativelanguage.googleapis.com",
            "context_window": 1_000_000,
        },
        "embedding_config": {
            "embedding_model": "text-embedding-004",
            "embedding_endpoint_type": "google_ai",
            "api_key": "your-api-key-here",
            "embedding_endpoint": "https://generativelanguage.googleapis.com",
            "embedding_dim": 768,
        },
        "meta_agent_config": {
            "agents": [
                {
                    "core_memory_agent": {
                        "blocks": [
                            {"label": "human", "value": ""},
                            {"label": "persona", "value": "I am a helpful assistant."},
                        ]
                    }
                },
                "resource_memory_agent",
                "semantic_memory_agent",
                "episodic_memory_agent",
                "procedural_memory_agent",
                "knowledge_vault_memory_agent",
            ],
        },
    }
)

client.add(
    user_id="demo-user",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "The moon now has a president."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]},
    ],
)

memories = client.retrieve_with_conversation(
    user_id="demo-user",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "What did we discuss on MirixDB in last 4 days?"}]},
    ],
    limit=5,
)
print(memories)
```
For more API examples, see `samples/run_client.py`.

## License

Mirix is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact us at `founders@mirix.io`

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
You can add the account `ari_asm` so that I can add you to the group chat.

## Acknowledgement
We would like to thank [Letta](https://github.com/letta-ai/letta) for open-sourcing their framework, which served as the foundation for the memory system in this project.
