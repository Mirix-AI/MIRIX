from mirix.schemas.agent import AgentType

# Agent configuration definitions
AGENT_CONFIGS = [
    {
        'name': 'background_agent',
        'agent_type': AgentType.background_agent,
        'attr_name': 'background_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'reflexion_agent',
        'agent_type': AgentType.reflexion_agent,
        'attr_name': 'reflexion_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'episodic_memory_agent',
        'agent_type': AgentType.episodic_memory_agent,
        'attr_name': 'episodic_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'procedural_memory_agent',
        'agent_type': AgentType.procedural_memory_agent,
        'attr_name': 'procedural_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'knowledge_vault_agent',
        'agent_type': AgentType.knowledge_vault_agent,
        'attr_name': 'knowledge_vault_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'meta_memory_agent',
        'agent_type': AgentType.meta_memory_agent,
        'attr_name': 'meta_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'semantic_memory_agent',
        'agent_type': AgentType.semantic_memory_agent,
        'attr_name': 'semantic_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'core_memory_agent',
        'agent_type': AgentType.core_memory_agent,
        'attr_name': 'core_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'resource_memory_agent',
        'agent_type': AgentType.resource_memory_agent,
        'attr_name': 'resource_memory_agent_state',
        'include_base_tools': False
    },
    {
        'name': 'chat_agent',
        'agent_type': None,  # chat_agent doesn't use a specific agent_type
        'attr_name': 'agent_state',
        'include_base_tools': True
    }
] 