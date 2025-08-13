class AgentStates:
    """
    Container class to hold all agent state objects.
    Makes it easier to pass around and manage agent states.
    """
    
    def __init__(self):
        self.agent_state = None
        self.episodic_memory_agent_state = None
        self.procedural_memory_agent_state = None
        self.knowledge_vault_agent_state = None
        self.meta_memory_agent_state = None
        self.semantic_memory_agent_state = None
        self.core_memory_agent_state = None
        self.resource_memory_agent_state = None
        self.reflexion_agent_state = None
        self.background_agent_state = None
        
    def set_agent_state(self, name, state):
        """Set an agent state by name."""
        if hasattr(self, name):
            setattr(self, name, state)
        else:
            raise ValueError(f"Unknown agent state name: {name}")
    
    def get_agent_state(self, name):
        """Get an agent state by name."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Unknown agent state name: {name}")
    
    def get_all_states(self):
        """Get all agent states as a dictionary."""
        return {
            'agent_state': self.agent_state,
            'episodic_memory_agent_state': self.episodic_memory_agent_state,
            'procedural_memory_agent_state': self.procedural_memory_agent_state,
            'knowledge_vault_agent_state': self.knowledge_vault_agent_state,
            'meta_memory_agent_state': self.meta_memory_agent_state,
            'semantic_memory_agent_state': self.semantic_memory_agent_state,
            'core_memory_agent_state': self.core_memory_agent_state,
            'resource_memory_agent_state': self.resource_memory_agent_state,
        } 