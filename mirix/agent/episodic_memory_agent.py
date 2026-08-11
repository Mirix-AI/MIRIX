from mirix.agent import Agent
from mirix.settings import settings


class EpisodicMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # load parent class init
        super().__init__(**kwargs)
        if settings.graph_version in ("v7.23", "v7.24"):
            from mirix.services.ingest_policy_v723 import apply_source_fidelity_prompt

            apply_source_fidelity_prompt(self.agent_state, "episodic")
