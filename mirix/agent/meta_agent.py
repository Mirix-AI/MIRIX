"""
MetaAgent: Orchestrates memory-related sub-agents for memory management operations.

This class manages all memory-related agents (episodic, procedural, semantic, core,
resource, knowledge_vault, reflexion, background, meta_memory) and coordinates
memory operations across them. It does NOT include the chat_agent.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from mirix import EmbeddingConfig, LLMConfig
from mirix.agent.agent import Agent, BaseAgent
from mirix.agent.message_queue import MessageQueue
from mirix.interface import AgentInterface
from mirix.orm import Client, User
from mirix.prompts import gpt_system
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message
from mirix.schemas.usage import MirixUsageStatistics
from mirix.utils import printv

if TYPE_CHECKING:
    from mirix.server.server import SyncServer


class MemoryAgentStates:
    """
    Container class to hold all memory-related agent state objects.
    Does NOT include chat_agent.
    """

    def __init__(self):
        self.episodic_memory_agent_state: Optional[AgentState] = None
        self.procedural_memory_agent_state: Optional[AgentState] = None
        self.knowledge_vault_memory_agent_state: Optional[AgentState] = None
        self.meta_memory_agent_state: Optional[AgentState] = None
        self.semantic_memory_agent_state: Optional[AgentState] = None
        self.core_memory_agent_state: Optional[AgentState] = None
        self.resource_memory_agent_state: Optional[AgentState] = None
        self.reflexion_agent_state: Optional[AgentState] = None
        self.background_agent_state: Optional[AgentState] = None

    def set_agent_state(self, name: str, state: AgentState):
        """Set an agent state by name."""
        if hasattr(self, name):
            setattr(self, name, state)
        else:
            raise ValueError(f"Unknown memory agent state name: {name}")

    def get_agent_state(self, name: str) -> Optional[AgentState]:
        """Get an agent state by name."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Unknown memory agent state name: {name}")

    def get_all_states(self) -> Dict[str, Optional[AgentState]]:
        """Get all memory agent states as a dictionary."""
        return {
            "episodic_memory_agent_state": self.episodic_memory_agent_state,
            "procedural_memory_agent_state": self.procedural_memory_agent_state,
            "knowledge_vault_memory_agent_state": self.knowledge_vault_memory_agent_state,
            "meta_memory_agent_state": self.meta_memory_agent_state,
            "semantic_memory_agent_state": self.semantic_memory_agent_state,
            "core_memory_agent_state": self.core_memory_agent_state,
            "resource_memory_agent_state": self.resource_memory_agent_state,
            "reflexion_agent_state": self.reflexion_agent_state,
            "background_agent_state": self.background_agent_state,
        }

    def get_all_agent_states_list(self) -> List[Optional[AgentState]]:
        """Get all memory agent states as a list."""
        return [
            self.episodic_memory_agent_state,
            self.procedural_memory_agent_state,
            self.knowledge_vault_memory_agent_state,
            self.meta_memory_agent_state,
            self.semantic_memory_agent_state,
            self.core_memory_agent_state,
            self.resource_memory_agent_state,
            self.reflexion_agent_state,
            self.background_agent_state,
        ]


# Memory agent configuration - excludes chat_agent
MEMORY_AGENT_CONFIGS = [
    {
        "name": "episodic_memory_agent",
        "agent_type": AgentType.episodic_memory_agent,
        "attr_name": "episodic_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "procedural_memory_agent",
        "agent_type": AgentType.procedural_memory_agent,
        "attr_name": "procedural_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "knowledge_vault_memory_agent",
        "agent_type": AgentType.knowledge_vault_memory_agent,
        "attr_name": "knowledge_vault_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "meta_memory_agent",
        "agent_type": AgentType.meta_memory_agent,
        "attr_name": "meta_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "semantic_memory_agent",
        "agent_type": AgentType.semantic_memory_agent,
        "attr_name": "semantic_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "core_memory_agent",
        "agent_type": AgentType.core_memory_agent,
        "attr_name": "core_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "resource_memory_agent",
        "agent_type": AgentType.resource_memory_agent,
        "attr_name": "resource_memory_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "reflexion_agent",
        "agent_type": AgentType.reflexion_agent,
        "attr_name": "reflexion_agent_state",
        "include_base_tools": False,
    },
    {
        "name": "background_agent",
        "agent_type": AgentType.background_agent,
        "attr_name": "background_agent_state",
        "include_base_tools": False,
    },
]


class MetaAgent(BaseAgent):
    """
    MetaAgent manages all memory-related sub-agents for coordinated memory operations.

    This agent follows the pattern of Agent in agent.py but is specialized for
    memory management. It orchestrates operations across:
    - Episodic Memory Agent
    - Procedural Memory Agent
    - Knowledge Vault Agent
    - Meta Memory Agent
    - Semantic Memory Agent
    - Core Memory Agent
    - Resource Memory Agent
    - Reflexion Agent
    - Background Agent

    It does NOT include the chat_agent as that is handled separately.
    """

    def __init__(
        self,
        server: "SyncServer",
        user: User,
        actor: Client,
        memory: Memory,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        system_prompts: Optional[Dict[str, str]] = None,
        interface: Optional[AgentInterface] = None,
        filter_tags: Optional[dict] = None,  # Filter tags for memory operations
        use_cache: bool = True,  # Control Redis cache behavior
        client_id: Optional[str] = None,  # Client application identifier
    ):
        """
        Initialize MetaAgent with memory sub-agents.

        Args:
            server: The Mirix server instance
            user: The user associated with this meta agent
            memory: The shared memory object for all sub-agents
            llm_config: LLM configuration for sub-agents
            embedding_config: Embedding configuration for sub-agents
            system_prompts: Pre-loaded system prompts dict (agent_name -> prompt_text)
            filter_tags: Optional dict of tags for filtering and categorization
            use_cache: Control Redis cache behavior (default: True)
            interface: Optional interface for agent interactions
        """
        # Initialize logger
        self.logger = logging.getLogger(f"Mirix.MetaAgent.{user.id}")
        self.logger.setLevel(logging.INFO)

        self.server = server
        self.user = user
        self.actor = actor
        self.memory = memory
        self.interface = interface
        self.system_prompts = system_prompts or {}

        # Store filter_tags as a COPY to prevent mutation across agent instances
        from copy import deepcopy

        # Keep None as None, don't convert to empty dict - they have different meanings
        self.filter_tags = deepcopy(filter_tags) if filter_tags is not None else None
        self.use_cache = use_cache  # Store use_cache for memory operations
        self.client_id = client_id  # Store client_id for multi-tenant isolation

        # Set default configs if not provided
        if llm_config is None:
            llm_config = LLMConfig.default_config("gpt-4o-mini")
        self.llm_config = llm_config

        if embedding_config is None:
            embedding_config = EmbeddingConfig.default_config("text-embedding-004")
        self.embedding_config = embedding_config

        # Initialize container for memory agent states
        self.memory_agent_states = MemoryAgentStates()

        # Initialize message queue for coordinating agent operations
        self.message_queue = MessageQueue()

        # Initialize or load memory sub-agents
        self._initialize_memory_agents()

        # Initialize individual Agent instances for each sub-agent
        self._initialize_agent_instances()

        printv(
            f"[Mirix.Agent.{self.agent_state.name}] INFO: MetaAgent initialized with {len(MEMORY_AGENT_CONFIGS)} memory sub-agents"
        )

    def _initialize_memory_agents(self):
        """
        Initialize all memory-related sub-agents.
        Either loads existing agents from server or creates new ones.
        """
        # Check if agents already exist
        existing_agents = self.server.agent_manager.list_agents(actor=self.actor)

        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Found {len(existing_agents)} existing agents")

        if existing_agents:
            # Load existing agents
            self._load_existing_agents(existing_agents)
        else:
            # Create new agents
            self._create_new_agents()

        # Ensure all agents have correct system prompts and configurations
        self._update_agent_configurations()

    def _load_existing_agents(self, existing_agents: List[AgentState]):
        """Load existing memory agent states from the server."""
        for agent_state in existing_agents:
            # Map agent names to their corresponding attribute names
            if agent_state.name == "episodic_memory_agent":
                self.memory_agent_states.episodic_memory_agent_state = agent_state
            elif agent_state.name == "procedural_memory_agent":
                self.memory_agent_states.procedural_memory_agent_state = agent_state
            elif agent_state.name == "knowledge_vault_memory_agent":
                self.memory_agent_states.knowledge_vault_memory_agent_state = agent_state
            elif agent_state.name == "meta_memory_agent":
                self.memory_agent_states.meta_memory_agent_state = agent_state
            elif agent_state.name == "semantic_memory_agent":
                self.memory_agent_states.semantic_memory_agent_state = agent_state
            elif agent_state.name == "core_memory_agent":
                self.memory_agent_states.core_memory_agent_state = agent_state
            elif agent_state.name == "resource_memory_agent":
                self.memory_agent_states.resource_memory_agent_state = agent_state
            elif agent_state.name == "reflexion_agent":
                self.memory_agent_states.reflexion_agent_state = agent_state
            elif agent_state.name == "background_agent":
                self.memory_agent_states.background_agent_state = agent_state

        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Loaded existing memory agent states")

    def _create_new_agents(self):
        """Create new memory agent states."""
        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Creating new memory agents...")

        # Ensure base tools are available
        self.server.tool_manager.upsert_base_tools(self.actor)

        for config in MEMORY_AGENT_CONFIGS:
            # Get system prompt
            system_prompt = self._get_system_prompt_for_agent(config["name"])

            # Create agent state
            agent_create = CreateAgent(
                name=config["name"],
                agent_type=config["agent_type"],
                system=system_prompt,
                llm_config=self.llm_config,
                embedding_config=self.embedding_config,
                include_base_tools=config["include_base_tools"],
            )
            agent_state = self.server.agent_manager.create_agent(
                agent_create=agent_create,
                actor=self.actor,
            )

            # Store the agent state
            setattr(self.memory_agent_states, config["attr_name"], agent_state)

            printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Created memory agent: {config['name']}")

    def _get_system_prompt_for_agent(self, agent_name: str) -> str:
        """
        Get the system prompt for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            System prompt text
        """
        import os

        # Priority 1: pre-loaded system_prompts dict
        if self.system_prompts and agent_name in self.system_prompts:
            return self.system_prompts[agent_name]

        # Priority 2: custom folder
        if self.system_prompt_folder is not None:
            custom_path = os.path.join(self.system_prompt_folder, f"{agent_name}.txt")
            if os.path.exists(custom_path):
                return gpt_system.get_system_text(os.path.join(self.system_prompt_folder, agent_name))

        # Priority 3: Fallback to base system prompts
        return gpt_system.get_system_text(f"base/{agent_name}")

    def _update_agent_configurations(self):
        """Update all agent configurations with current settings."""
        for agent_state in self.memory_agent_states.get_all_agent_states_list():
            if agent_state is None:
                continue

            # Get system prompt
            system_prompt = self._get_system_prompt_for_agent(agent_state.name)

            # Update agent
            self.server.agent_manager.update_agent_tools_and_system_prompts(
                agent_id=agent_state.id,
                actor=self.actor,
                system_prompt=system_prompt,
            )

            # Update LLM config
            self.server.agent_manager.update_llm_config(
                agent_id=agent_state.id,
                llm_config=self.llm_config,
                actor=self.actor,
            )

        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Updated all memory agent configurations")

    def _initialize_agent_instances(self):
        """
        Initialize Agent instances for each sub-agent to enable direct stepping.
        """
        self.agents: Dict[str, Agent] = {}

        for config in MEMORY_AGENT_CONFIGS:
            agent_state = getattr(self.memory_agent_states, config["attr_name"])
            if agent_state is not None:
                # Create an Agent instance for this sub-agent
                # Pass filter_tags and use_cache from MetaAgent to child agents
                agent_instance = Agent(
                    interface=self.interface,
                    agent_state=agent_state,
                    user=self.user,
                    actor=self.actor,
                    filter_tags=self.filter_tags,
                    use_cache=self.use_cache,
                )
                self.agents[config["name"]] = agent_instance

        printv(
            f"[Mirix.Agent.{self.agent_state.name}] INFO: Initialized {len(self.agents)} Agent instances for sub-agents"
        )

    def step(
        self,
        messages: Union[Message, List[Message]],
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> MirixUsageStatistics:
        """
        Execute a step with a specific memory agent or all agents.

        Args:
            messages: Input message(s) to process
            agent_name: Specific agent to use, or None to use default routing
            **kwargs: Additional arguments passed to the agent's step method

        Returns:
            Usage statistics from the agent step
        """
        if agent_name is not None:
            # Route to specific agent
            if agent_name not in self.agents:
                raise ValueError(f"Unknown memory agent: {agent_name}")

            agent = self.agents[agent_name]
            return agent.step(messages, **kwargs)
        else:
            # Default behavior: route to meta_memory_agent for coordination
            if "meta_memory_agent" in self.agents:
                return self.agents["meta_memory_agent"].step(messages, **kwargs)
            else:
                raise RuntimeError("No meta_memory_agent available for coordination")

    def send_message_to_agent(self, agent_name: str, message: Union[str, dict], **kwargs) -> tuple:
        """
        Send a message to a specific memory agent through the message queue.

        Args:
            agent_name: Name of the agent to send message to
            message: Message content (string or dict)
            **kwargs: Additional arguments for message processing

        Returns:
            Tuple of (response, usage_statistics)
        """
        # Get agent state
        agent_state = self.memory_agent_states.get_agent_state(f"{agent_name}_state")
        if agent_state is None:
            raise ValueError(f"Agent state not found for: {agent_name}")

        # Determine agent type for message queue
        agent_type_map = {
            "episodic_memory_agent": "episodic_memory",
            "procedural_memory_agent": "procedural_memory",
            "knowledge_vault_memory_agent": "knowledge_vault",
            "meta_memory_agent": "meta_memory",
            "semantic_memory_agent": "semantic_memory",
            "core_memory_agent": "core_memory",
            "resource_memory_agent": "resource_memory",
            "reflexion_agent": "reflexion",
            "background_agent": "background",
        }

        agent_type = agent_type_map.get(agent_name, agent_name)

        # Format message
        if isinstance(message, str):
            message_data = {"message": message}
        else:
            message_data = message

        # Send through message queue
        response, usage = self.message_queue.send_message_in_queue(
            client=self.server,  # Pass server directly
            agent_id=agent_state.id,
            message_data=message_data,
            agent_type=agent_type,
            **kwargs,
        )

        return response, usage

    def update_llm_config(self, llm_config: LLMConfig):
        """
        Update the LLM configuration for all memory agents.

        Args:
            llm_config: New LLM configuration
        """
        self.llm_config = llm_config

        for agent_state in self.memory_agent_states.get_all_agent_states_list():
            if agent_state is not None:
                self.server.agent_manager.update_llm_config(
                    agent_id=agent_state.id,
                    llm_config=llm_config,
                    actor=self.actor,
                )

        printv(
            f"[Mirix.Agent.{self.agent_state.name}] INFO: Updated LLM config for all memory agents to model: {llm_config.model}"
        )

    def update_embedding_config(self, embedding_config: EmbeddingConfig):
        """
        Update the embedding configuration for all memory agents.

        Args:
            embedding_config: New embedding configuration
        """
        self.embedding_config = embedding_config

        # Get Client object for actor parameter (needed for write operations)
        actor = None
        if self.client_id:
            actor = self.server.client_manager.get_client_by_id(self.client_id)

        for agent_state in self.memory_agent_states.get_all_agent_states_list():
            if agent_state is not None:
                self.server.agent_manager.update_agent(
                    agent_id=agent_state.id,
                    actor=actor,  # Client for write operations (audit trail)
                    # Note: Would need UpdateAgent schema to include embedding_config
                    # For now, this is a placeholder
                )

        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Updated embedding config for all memory agents")

    def get_agent_state(self, agent_name: str) -> Optional[AgentState]:
        """
        Get the state of a specific memory agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentState object or None if not found
        """
        return self.memory_agent_states.get_agent_state(f"{agent_name}_state")

    def list_memory_agents(self) -> List[str]:
        """
        List all available memory agent names.

        Returns:
            List of memory agent names
        """
        return [config["name"] for config in MEMORY_AGENT_CONFIGS]

    def refresh_agents(self):
        """
        Refresh all agent states from the server.
        Useful after external modifications to agent configurations.
        """
        existing_agents = self.server.agent_manager.list_agents(actor=self.actor)
        self._load_existing_agents(existing_agents)
        self._initialize_agent_instances()
        printv(f"[Mirix.Agent.{self.agent_state.name}] INFO: Refreshed all memory agent states")

    def __repr__(self) -> str:
        agent_count = len([s for s in self.memory_agent_states.get_all_agent_states_list() if s is not None])
        return f"MetaAgent(user={self.user.name}, memory_agents={agent_count}, model={self.llm_config.model})"
