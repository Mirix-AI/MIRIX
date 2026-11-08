from mirix.agent import Agent
from mirix.settings import settings


class SemanticMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # load parent class init
        super().__init__(**kwargs)
        if settings.enable_graph_memory:
            from mirix.services.ingest_policy_v723 import apply_source_fidelity_prompt

            apply_source_fidelity_prompt(self.agent_state, "semantic")
