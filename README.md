![Mirix Logo](https://github.com/RenKoya1/MIRIX/raw/main/assets/logo.png)

## MIRIX - Multi-Agent Personal Assistant with an Advanced Memory System

Your personal AI that builds memory through screen observation and natural conversation

| 🌐 [Website](https://mirix.io) | 📚 [Documentation](https://docs.mirix.io) | 📄 [Paper](https://arxiv.org/abs/2507.07957) |
<!-- | [Twitter/X](https://twitter.com/mirix_ai) | [Discord](https://discord.gg/mirix) | -->

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

We've created a simple Python SDK that makes it incredibly easy to integrate Mirix's memory capabilities into your applications:

### Installation
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

## License

Mirix is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact us at `yuwang@mirix.io`

## Join Our Community

Connect with other Mirix users, share your thoughts, and get support:

### 💬 Discord Community
Join our Discord server for real-time discussions, support, and community updates:
**[https://discord.gg/5HWyxJrh](https://discord.gg/5HWyxJrh)**

### 🎯 Weekly Discussion Sessions
We host weekly discussion sessions where you can:
- Discuss issues and bugs
- Share ideas about future directions
- Get general consultations and support
- Connect with the development team and community

**📅 Schedule:** Friday nights, 8-9 PM PST  
**🔗 Zoom Link:** [https://ucsd.zoom.us/j/96278791276](https://ucsd.zoom.us/j/96278791276)

### 📱 WeChat Group
<div align="center">
<img src="frontend/public/wechat-qr.png" alt="WeChat QR Code" width="200"/><br/>
<strong>WeChat Group</strong>
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Mirix-AI/MIRIX&type=Date)](https://star-history.com/#Mirix-AI/MIRIX.&Date)

## Acknowledgement
We would like to thank [Letta](https://github.com/letta-ai/letta) for open-sourcing their framework, which served as the foundation for the memory system in this project.
