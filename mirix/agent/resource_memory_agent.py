import json
from mirix.agent import Agent
from mirix.utils import parse_json

class ResourceMemoryAgent(Agent):
    def __init__(
        self,
        **kwargs
    ):
        # load parent class init 
        super().__init__(**kwargs)
