from mirix.agent import Agent


class KnowledgeVaultAgent(Agent):
    def __init__(self, **kwargs):
        # load parent class init
        super().__init__(**kwargs)
