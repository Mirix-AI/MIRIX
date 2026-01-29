"""
Mirix SDK - Simple Python interface for memory-enhanced AI agents
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mirix.local_client import create_client
from mirix.schemas.agent import AgentType, CreateMetaAgent
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import Memory

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration

    Example:
        >>> config = load_config("mirix/configs/mirix.yaml")
        >>> client = MirixClient(org_id="demo-org")
        >>> client.initialize_meta_agent(config=config)
    """
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


class Mirix:
    """
    Simple SDK interface for Mirix memory agent.

    Example:
        from mirix import Mirix

        memory_agent = Mirix(api_key="your-api-key")
        memory_agent.add("The moon now has a president")
        memories = memory_agent.visualize_memories()
    """

    def __init__(
        self,
        api_key: str,
        model_provider: str = "google_ai",
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        load_from: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Mirix memory agent.

        Args:
            api_key: API key for LLM provider (required)
            model_provider: LLM provider name (default: "google_ai")
            model: Model to use (optional). If None, uses default model.
            config_path: Path to custom config file (optional, for loading system prompts)
            load_from: Path to backup directory to restore from (optional)
        """
        if not api_key:
            raise ValueError("api_key is required to initialize Mirix")

        # Set API key environment variable based on provider
        if model_provider.lower() in ["google", "google_ai", "gemini"]:
            os.environ["GEMINI_API_KEY"] = api_key
        elif model_provider.lower() in ["openai", "gpt"]:
            os.environ["OPENAI_API_KEY"] = api_key
        elif model_provider.lower() in ["anthropic", "claude"]:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            # For custom providers, use the provider name as prefix
            os.environ[f"{model_provider.upper()}_API_KEY"] = api_key

        # Force reload of model_settings to pick up new environment variables
        self._reload_model_settings()

        # Load config from file if provided, otherwise use defaults
        system_prompts_folder = None
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                import yaml

                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    system_prompts_folder = config_data.get("system_prompts_folder")

                    # Load llm_config from file
                    if "llm_config" in config_data:
                        llm_config = LLMConfig(**config_data["llm_config"])
                    else:
                        # Fall back to default
                        model = model or "gemini-2.0-flash"
                        llm_config = LLMConfig.default_config(model)

                    # Load embedding_config from file
                    if "embedding_config" in config_data:
                        embedding_config = EmbeddingConfig(**config_data["embedding_config"])
                    else:
                        # Fall back to default
                        embedding_config = EmbeddingConfig.default_config("text-embedding-004")
            else:
                # Config file doesn't exist, use defaults
                model = model or "gemini-2.0-flash"
                llm_config = LLMConfig.default_config(model)
                embedding_config = EmbeddingConfig.default_config("text-embedding-004")
        else:
            # No config file, use defaults
            model = model or "gemini-2.0-flash"
            llm_config = LLMConfig.default_config(model)
            embedding_config = EmbeddingConfig.default_config("text-embedding-004")

        # Initialize client
        self._client = create_client()
        self._client.set_default_llm_config(llm_config)
        self._client.set_default_embedding_config(embedding_config)

        # Check if meta agent already exists
        agents = self._client.list_agents()
        existing_meta_agent = None
        for agent in agents:
            if agent.agent_type == AgentType.meta_memory_agent:
                existing_meta_agent = agent
                break

        if existing_meta_agent:
            self._meta_agent = existing_meta_agent
        else:
            # Create meta agent
            create_request = CreateMetaAgent(
                system_prompts_folder=system_prompts_folder,
                llm_config=llm_config,
                embedding_config=embedding_config,
            )
            self._meta_agent = self._client.create_meta_agent(request=create_request)

    def add(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Add information to memory.

        Args:
            content: Information to memorize
            **kwargs: Additional options (images, metadata, etc.)

        Returns:
            Response from the memory system

        Example:
            memory_agent.add("John likes pizza")
            memory_agent.add("Meeting at 3pm", metadata={"type": "appointment"})
        """
        response = self._client.send_message(agent_id=self._meta_agent.id, role="user", message=content, **kwargs)

        # Extract the response text from MirixResponse
        if hasattr(response, "messages") and response.messages:
            # Get last assistant message
            for msg in reversed(response.messages):
                if msg.role == "assistant":
                    return {"response": msg.text, "success": True}

        return {"response": str(response), "success": True}

    def list_users(self) -> List[Any]:
        """
        List all users in the system.

        Returns:
            List of user objects

        Example:
            users = memory_agent.list_users()
            for user in users:
                logger.debug("User: %s (ID: %s)", user.name, user.id)
        """
        users = self._client.server.user_manager.list_users()
        return users

    def construct_system_message(self, message: str, user_id: str) -> str:
        """
        Construct a system message from a message.
        """
        return self._client.construct_system_message(agent_id=self._meta_agent.id, message=message, user_id=user_id)

    def extract_memory_for_system_prompt(self, message: str, user_id: str) -> str:
        """
        Extract memory for system prompt from a message.
        """
        return self._client.extract_memory_for_system_prompt(
            agent_id=self._meta_agent.id, message=message, user_id=user_id
        )

    def get_user_by_name(self, user_name: str):
        """
        Get a user by their name.

        Args:
            user_name: The name of the user to search for

        Returns:
            User object if found, None if not found

        Example:
            user = memory_agent.get_user_by_name("Alice")
            if user:
                logger.debug("Found user: %s (ID: %s)", user.name, user.id)
            else:
                logger.debug("User not found")
        """
        users = self.list_users()
        for user in users:
            if user.name == user_name:
                return user
        return None

    def clear(self) -> Dict[str, Any]:
        """
        Clear all memories.

        Note: This requires manual database file removal and app restart.

        Returns:
            Dict with warning message and instructions

        Example:
            result = memory_agent.clear()
            logger.debug(result['warning'])
            for step in result['instructions']:
                logger.debug(step)
        """
        return {
            "success": False,
            "warning": "Memory clearing requires manual database reset.",
            "instructions": [
                "1. Stop the Mirix application/process",
                "2. Remove the database file: ~/.mirix/sqlite.db",
                "3. Restart the Mirix application",
                "4. Initialize a new Mirix agent",
            ],
            "manual_command": "rm ~/.mirix/sqlite.db",
            "note": "After removing the database file, you must restart your application and create a new agent instance.",
        }

    def clear_conversation_history(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear conversation history while preserving memories.

        This removes user and assistant messages from the conversation
        history but keeps system messages and all stored memories intact.

        Args:
            user_id: User ID to clear messages for. If None, clears all messages
                    except system messages. If provided, only clears messages for that specific user.

        Returns:
            Dict containing success status, message, and count of deleted messages

        Example:
            # Clear all conversation history
            result = memory_agent.clear_conversation_history()

            # Clear history for specific user
            result = memory_agent.clear_conversation_history(user_id="user_123")

            if result['success']:
                logger.debug("Cleared %s messages", result['messages_deleted'])
            else:
                logger.debug("Failed to clear: %s", result['error'])
        """
        try:
            if user_id is None:
                # Clear all messages except system messages (original behavior)
                # Get agent state first, then get messages
                agent_state = self._client.server.agent_manager.get_agent_by_id(
                    agent_id=self._meta_agent.id,
                    actor=self._client.client,
                )
                current_messages = self._client.server.agent_manager.get_in_context_messages(
                    agent_state=agent_state,
                    actor=self._client.client,
                )
                messages_count = len(current_messages)

                # Clear conversation history using the agent manager reset_messages method
                # actor=Client (for authorization), user_id=None (clear all users' messages)
                self._client.server.agent_manager.reset_messages(
                    agent_id=self._meta_agent.id,
                    actor=self._client.client,
                    user_id=None,  # Clear messages for all users
                    add_default_initial_messages=True,  # Keep system message and initial setup
                )

                return {
                    "success": True,
                    "message": "Successfully cleared conversation history. All user and assistant messages removed (system messages preserved).",
                    "messages_deleted": messages_count,
                }
            else:
                # Get the user object by ID
                target_user = self._client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        "success": False,
                        "error": f"User with ID '{user_id}' not found",
                        "messages_deleted": 0,
                    }

                # Clear messages for specific user (same as FastAPI server implementation)
                # Get current message count for this specific user for reporting
                agent_state = self._client.server.agent_manager.get_agent_by_id(
                    agent_id=self._meta_agent.id,
                    actor=self._client.client,
                )
                current_messages = self._client.server.agent_manager.get_in_context_messages(
                    agent_state=agent_state,
                    actor=self._client.client,
                    user=target_user,
                )
                # Count messages belonging to this user (excluding system messages)
                user_messages_count = len(
                    [msg for msg in current_messages if msg.role != "system" and msg.user_id == target_user.id]
                )

                # Clear conversation history using the agent manager reset_messages method
                # actor=Client (for authorization), user_id=specific user to clear
                self._client.server.agent_manager.reset_messages(
                    agent_id=self._meta_agent.id,
                    actor=self._client.client,
                    user_id=target_user.id,  # Clear messages only for this user
                    add_default_initial_messages=True,  # Keep system message and initial setup
                )

                return {
                    "success": True,
                    "message": f"Successfully cleared conversation history for {target_user.name}. Messages from other users and system messages preserved.",
                    "messages_deleted": user_messages_count,
                }

        except Exception as e:
            return {"success": False, "error": str(e), "messages_deleted": 0}

    def save(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current memory state to disk.

        Note: Save/backup functionality is not yet implemented in the client-based SDK.
        Please use the database backup directly.

        Args:
            path: Save directory path (optional). If not provided, generates
                 timestamp-based directory name.

        Returns:
            Dict containing success status and backup path

        Example:
            result = memory_agent.save("./my_backup")
        """
        return {
            "success": False,
            "error": "Save functionality not yet implemented in client-based SDK. Please backup the database directly.",
            "path": path or "N/A",
        }

    def load(self, path: str) -> Dict[str, Any]:
        """
        Load memory state from a backup directory.

        Note: Load/restore functionality is not yet implemented in the client-based SDK.
        Please restore the database directly.

        Args:
            path: Path to backup directory

        Returns:
            Dict containing success status and any error messages

        Example:
            result = memory_agent.load("./my_backup")
        """
        return {
            "success": False,
            "error": "Load functionality not yet implemented in client-based SDK. Please restore the database directly.",
        }

    def _reload_model_settings(self):
        """
        Force reload of model_settings to pick up new environment variables.

        This is necessary because Pydantic BaseSettings loads environment variables
        at class instantiation time, which happens at import. Since the SDK sets
        environment variables after import, we need to manually update the singleton.
        """
        from mirix.settings import ModelSettings

        # Create a new instance with current environment variables
        new_settings = ModelSettings()

        # Update the global singleton instance with new values
        import mirix.settings

        for field_name in ModelSettings.model_fields:
            setattr(
                mirix.settings.model_settings,
                field_name,
                getattr(new_settings, field_name),
            )

    def create_user(self, user_name: str, timezone: str = "UTC", organization_id: Optional[str] = None) -> Any:
        """
        Create a new user in the system.

        Args:
            user_name: The name for the new user
            timezone: The timezone for the user (default: "UTC")
            organization_id: The organization ID (default: uses default organization)

        Returns:
            User object

        Example:
            user = memory_agent.create_user("Alice")
            logger.debug("Created user: %s", user.name)
        """
        from mirix.schemas.user import UserCreate
        from mirix.services.organization_manager import OrganizationManager

        if organization_id is None:
            organization_id = OrganizationManager.DEFAULT_ORG_ID

        return self._client.server.user_manager.create_user(
            pydantic_user=UserCreate(name=user_name, timezone=timezone, organization_id=organization_id)
        )

    def __call__(self, message: str) -> Dict[str, Any]:
        """
        Allow using the agent as a callable for adding memories.

        Example:
            memory_agent = Mirix(api_key="...")
            response = memory_agent("John likes pizza")
        """
        return self.add(message)

    def insert_tool(
        self,
        name: str,
        source_code: str,
        description: str,
        args_info: Optional[Dict[str, str]] = None,
        returns_info: Optional[str] = None,
        tags: Optional[List[str]] = None,
        apply_to_agents: Union[List[str], str] = "all",
    ) -> Dict[str, Any]:
        """
        Insert a custom tool into the system.

        Args:
            name: The name of the tool function
            source_code: The Python source code for the tool function (without docstring)
            description: Description of what the tool does
            args_info: Optional dict mapping argument names to their descriptions
            returns_info: Optional description of what the function returns
            tags: Optional list of tags for categorization (defaults to ["user_defined"])
            apply_to_all_agents: Whether to add this tool to all existing agents (default: True)

        Returns:
            Dict containing success status, tool data, and any error messages

        Example:
            result = memory_agent.insert_tool(
                name="calculate_sum",
                source_code="def calculate_sum(a: int, b: int) -> int:\n    return a + b",
                description="Calculate the sum of two numbers",
                args_info={"a": "First number", "b": "Second number"},
                returns_info="The sum of a and b",
                tags=["math", "utility"]
            )
        """
        from mirix.schemas.enums import ToolType
        from mirix.schemas.tool import Tool as PydanticTool
        from mirix.services.tool_manager import ToolManager

        # Initialize tool manager
        tool_manager = ToolManager()

        # Check if tool name already exists
        existing_tool = tool_manager.get_tool_by_name(tool_name=name, actor=self._client.client)

        if existing_tool:
            created_tool = existing_tool

        else:
            # Set default tags if not provided
            if tags is None:
                tags = ["user_defined"]

            # Construct complete source code with docstring
            complete_source_code = self._build_complete_source_code(source_code, description, args_info, returns_info)

            # Generate JSON schema from the complete source code
            from mirix.functions.functions import derive_openai_json_schema

            json_schema = derive_openai_json_schema(source_code=complete_source_code, name=name)

            # Create the tool object
            pydantic_tool = PydanticTool(
                name=name,
                source_code=complete_source_code,
                source_type="python",
                tool_type=ToolType.USER_DEFINED,
                tags=tags,
                description=description,
                json_schema=json_schema,
            )

            # Use the tool manager's create_or_update_tool method
            created_tool = tool_manager.create_or_update_tool(pydantic_tool=pydantic_tool, actor=self._client.client)

        # Apply tool to all existing agents if requested
        if apply_to_agents:
            # Get all existing agents
            all_agents = self._client.server.agent_manager.list_agents(
                actor=self._client.client,
                limit=1000,  # Get all agents
            )

            if apply_to_agents != "all":
                all_agents = [agent for agent in all_agents if agent.name in apply_to_agents]

            # Add the tool to each agent
            for agent in all_agents:
                # Get current agent tools
                existing_tools = agent.tools
                existing_tool_ids = [tool.id for tool in existing_tools]

                # Add the new tool if not already present
                if created_tool.id not in existing_tool_ids:
                    new_tool_ids = existing_tool_ids + [created_tool.id]

                    # Update the agent with the new tool
                    from mirix.schemas.agent import UpdateAgent

                    self._client.server.agent_manager.update_agent(
                        agent_id=agent.id,
                        agent_update=UpdateAgent(tool_ids=new_tool_ids),
                        actor=self._client.client,
                    )

        return {
            "success": True,
            "message": f"Tool '{name}' inserted successfully"
            + (" and applied to all existing agents" if apply_to_agents else ""),
            "tool": {
                "id": created_tool.id,
                "name": created_tool.name,
                "description": created_tool.description,
                "tags": created_tool.tags,
                "tool_type": created_tool.tool_type.value if created_tool.tool_type else None,
            },
        }

    def _build_complete_source_code(
        self,
        source_code: str,
        description: str,
        args_info: Optional[Dict[str, str]] = None,
        returns_info: Optional[str] = None,
    ) -> str:
        """
        Build complete source code with proper docstring from user-provided components.

        Args:
            source_code: The bare function code without docstring
            description: Function description
            args_info: Optional dict mapping argument names to descriptions
            returns_info: Optional return value description

        Returns:
            Complete source code with properly formatted docstring
        """
        import re

        # Find the function definition line
        func_match = re.search(r"(def\s+\w+\([^)]*\)\s*(?:->\s*[^:]+)?:)", source_code)
        if not func_match:
            raise ValueError("Invalid function definition in source_code")

        func_def = func_match.group(1)
        func_body = source_code[func_match.end() :].lstrip("\n")

        # Build docstring
        docstring_lines = ['    """', f"    {description}"]

        if args_info:
            docstring_lines.extend(["", "    Args:"])
            for arg_name, arg_desc in args_info.items():
                docstring_lines.append(f"        {arg_name}: {arg_desc}")

        if returns_info:
            docstring_lines.extend(["", "    Returns:", f"        {returns_info}"])

        docstring_lines.append('    """')

        # Combine everything
        complete_code = func_def + "\n" + "\n".join(docstring_lines) + "\n" + func_body

        return complete_code

    def visualize_memories(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize all memories for a specific user.

        Args:
            user_id: User ID to get memories for. If None, uses current active user.

        Returns:
            Dict containing all memory types organized by category

        Example:
            memories = memory_agent.visualize_memories(user_id="user_123")
            logger.debug("Episodic memories: %s", len(memories['episodic']))
            logger.debug("Semantic memories: %s", len(memories['semantic']))
        """
        try:
            # Find the target user
            if user_id:
                target_user = self._client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        "success": False,
                        "error": f"User with ID '{user_id}' not found",
                    }
            else:
                # Find the current active user
                users = self._client.server.user_manager.list_users()
                active_user = next((user for user in users if user.status == "active"), None)
                target_user = active_user if active_user else (users[0] if users else None)

            if not target_user:
                return {"success": False, "error": "No user found"}

            # Get the meta agent state to access memory agents
            meta_agent_state = self._client.get_agent(self._meta_agent.id)

            memories = {}

            # Get episodic memory
            try:
                episodic_manager = self._client.server.episodic_memory_manager
                # Find episodic memory agent from meta agent's children
                episodic_agent = None
                child_agents = self._client.list_agents(parent_id=self._meta_agent.id)
                for agent in child_agents:
                    if agent.agent_type == AgentType.episodic_memory_agent:
                        episodic_agent = agent
                        break

                if episodic_agent:
                    events = episodic_manager.list_episodic_memory(
                        agent_state=episodic_agent,
                        actor=target_user,
                        limit=50,
                        timezone_str=target_user.timezone,
                    )
                else:
                    events = []

                memories["episodic"] = []
                for event in events:
                    memories["episodic"].append(
                        {
                            "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                            "summary": event.summary,
                            "details": event.details,
                            "event_type": event.event_type,
                        }
                    )
            except Exception:
                memories["episodic"] = []

            # Get semantic memory
            try:
                semantic_manager = self._client.server.semantic_memory_manager
                # Find semantic memory agent from meta agent's children
                semantic_agent = None
                for agent in child_agents:
                    if agent.agent_type == AgentType.semantic_memory_agent:
                        semantic_agent = agent
                        break

                if semantic_agent:
                    semantic_items = semantic_manager.list_semantic_items(
                        agent_state=semantic_agent,
                        actor=target_user,
                        limit=50,
                        timezone_str=target_user.timezone,
                    )
                else:
                    semantic_items = []

                memories["semantic"] = []
                for item in semantic_items:
                    memories["semantic"].append(
                        {
                            "title": item.name,
                            "type": "semantic",
                            "summary": item.summary,
                            "details": item.details,
                        }
                    )
            except Exception:
                memories["semantic"] = []

            # Get procedural memory
            try:
                procedural_manager = self._client.server.procedural_memory_manager
                # Find procedural memory agent from meta agent's children
                procedural_agent = None
                for agent in child_agents:
                    if agent.agent_type == AgentType.procedural_memory_agent:
                        procedural_agent = agent
                        break

                if procedural_agent:
                    procedural_items = procedural_manager.list_procedures(
                        agent_state=procedural_agent,
                        actor=target_user,
                        limit=50,
                        timezone_str=target_user.timezone,
                    )
                else:
                    procedural_items = []

                memories["procedural"] = []
                for item in procedural_items:
                    import json

                    # Parse steps if it's a JSON string
                    steps = item.steps
                    if isinstance(steps, str):
                        try:
                            steps = json.loads(steps)
                            # Extract just the instruction text for simpler display
                            if isinstance(steps, list) and steps and isinstance(steps[0], dict):
                                steps = [step.get("instruction", str(step)) for step in steps]
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # If parsing fails, keep as string and split by common delimiters
                            if isinstance(steps, str):
                                steps = [s.strip() for s in steps.replace("\n", "|").split("|") if s.strip()]
                            else:
                                steps = []

                    memories["procedural"].append(
                        {
                            "title": item.entry_type,
                            "type": "procedural",
                            "summary": item.summary,
                            "steps": steps if isinstance(steps, list) else [],
                        }
                    )
            except Exception:
                memories["procedural"] = []

            # Get resource memory
            try:
                resource_manager = self._client.server.resource_memory_manager
                # Find resource memory agent from meta agent's children
                resource_agent = None
                for agent in child_agents:
                    if agent.agent_type == AgentType.resource_memory_agent:
                        resource_agent = agent
                        break

                if resource_agent:
                    resources = resource_manager.list_resources(
                        agent_state=resource_agent,
                        actor=target_user,
                        limit=50,
                        timezone_str=target_user.timezone,
                    )
                else:
                    resources = []

                memories["resources"] = []
                for resource in resources:
                    memories["resources"].append(
                        {
                            "filename": resource.title,
                            "type": resource.resource_type,
                            "summary": resource.summary
                            or (resource.content[:200] + "..." if len(resource.content) > 200 else resource.content),
                            "last_accessed": resource.updated_at.isoformat() if resource.updated_at else None,
                        }
                    )
            except Exception:
                memories["resources"] = []

            # Get core memory
            try:
                core_memory = Memory(
                    blocks=[
                        self._client.server.block_manager.get_block_by_id(block.id, actor=target_user)
                        for block in self._client.server.block_manager.get_blocks(actor=target_user)
                    ]
                )

                memories["core"] = []
                total_characters = 0

                for block in core_memory.blocks:
                    if block.value and block.value.strip() and block.label.lower() != "persona":
                        block_chars = len(block.value)
                        total_characters += block_chars

                        memories["core"].append(
                            {
                                "aspect": block.label,
                                "understanding": block.value,
                                "character_count": block_chars,
                                "total_characters": total_characters,
                                "max_characters": block.limit,
                                "last_updated": None,
                            }
                        )
            except Exception:
                memories["core"] = []

            # Get credentials memory
            try:
                knowledge_vault_manager = self._client.server.knowledge_vault_manager
                # Find knowledge vault agent from meta agent's children
                knowledge_vault_memory_agent = None
                for agent in child_agents:
                    if agent.agent_type == AgentType.knowledge_vault_memory_agent:
                        knowledge_vault_memory_agent = agent
                        break

                if knowledge_vault_memory_agent:
                    vault_items = knowledge_vault_manager.list_knowledge(
                        actor=target_user,
                        agent_state=knowledge_vault_memory_agent,
                        limit=50,
                        timezone_str=target_user.timezone,
                    )
                else:
                    vault_items = []

                memories["credentials"] = []
                for item in vault_items:
                    memories["credentials"].append(
                        {
                            "caption": item.caption,
                            "entry_type": item.entry_type,
                            "source": item.source,
                            "sensitivity": item.sensitivity,
                            "content": "••••••••••••" if item.sensitivity == "high" else item.secret_value,
                        }
                    )
            except Exception:
                memories["credentials"] = []

            return {
                "success": True,
                "user_id": target_user.id,
                "user_name": target_user.name,
                "memories": memories,
                "summary": {
                    "episodic_count": len(memories.get("episodic", [])),
                    "semantic_count": len(memories.get("semantic", [])),
                    "procedural_count": len(memories.get("procedural", [])),
                    "resources_count": len(memories.get("resources", [])),
                    "core_count": len(memories.get("core", [])),
                    "credentials_count": len(memories.get("credentials", [])),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_core_memory(self, label: str, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a specific core memory block with new text.

        Core memory blocks are persistent memory aspects that help the agent
        understand and remember key information about users, preferences, and context.

        Args:
            label: The label/name of the core memory block to update (e.g., "persona", "user_preferences")
            text: The new text content for the memory block
            user_id: User ID to update core memory for. If None, uses the current active user.

        Returns:
            Dict containing success status and message

        Example:
            # Update core memory for current user
            result = memory_agent.update_core_memory(
                label="user_preferences",
                text="User prefers concise responses and technical details"
            )

            # Update core memory for specific user
            result = memory_agent.update_core_memory(
                label="user_preferences",
                text="Alice prefers detailed explanations",
                user_id="user_123"
            )

            if result['success']:
                logger.debug("Core memory updated successfully")
            else:
                logger.debug("Update failed: %s", result['message'])
        """
        try:
            # If user_id is provided, get the specific user
            if user_id:
                target_user = self._client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        "success": False,
                        "message": f"User with ID '{user_id}' not found",
                    }
            else:
                # Use current user
                target_user = self._client.user

            # Update core memory using the client's update_in_context_memory method
            self._client.update_in_context_memory(agent_id=self._meta_agent.id, section=label, value=text)

            return {
                "success": True,
                "message": f"Core memory block '{label}' updated successfully"
                + (f" for user {user_id}" if user_id else ""),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error updating core memory: {str(e)}",
            }
