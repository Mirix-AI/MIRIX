from mirix.agent import Agent

class EpisodicMemoryAgent(Agent):
    def __init__(
        self,
        **kwargs
    ):
        # load parent class init 
        super().__init__(**kwargs)
   