"""
Mirix SDK - Simple Python interface for memory-enhanced AI agents
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from mirix.agent import AgentWrapper

logger = logging.getLogger(__name__)


class Mirix:
    """
    Simple SDK interface for Mirix memory agent.
    
    Example:
        from mirix import Mirix
        
        memory_agent = Mirix(api_key="your-api-key")
        memory_agent.add("The moon now has a president")
        response = memory_agent.chat("Does moon have a president now?")
    """
    
    def __init__(
        self,
        api_key: str,
        model_provider: str = "google_ai",
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        load_from: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Mirix memory agent.
        
        Args:
            api_key: API key for LLM provider (required)
            model_provider: LLM provider name (default: "google_ai")
            model: Model to use (optional). If None, uses model from config file.
            config_path: Path to custom config file (optional)
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
        
        # Track if config_path was originally provided
        config_path_provided = config_path is not None
        
        # Use default config if not specified
        if not config_path:
            # Try to find config file in order of preference
            import sys
            
            if getattr(sys, 'frozen', False):
                # Running in PyInstaller bundle
                bundle_dir = Path(sys._MEIPASS)
                config_path = bundle_dir / "mirix" / "configs" / "mirix.yaml"
                
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Could not find mirix.yaml config file in PyInstaller bundle at:\n"
                        f"  - {config_path}\n"
                        f"Please ensure config file is properly bundled."
                    )
            else:
                # Running in development - use existing logic
                package_dir = Path(__file__).parent
                
                # 1. Look in package configs directory (for installed package)
                config_path = package_dir / "configs" / "mirix.yaml"
                
                if not config_path.exists():
                    # 2. Look in parent configs directory (for development)
                    config_path = package_dir.parent / "configs" / "mirix.yaml"
                    
                    if not config_path.exists():
                        # 3. Look in current working directory
                        config_path = Path("./mirix/configs/mirix.yaml")
                        
                        if not config_path.exists():
                            raise FileNotFoundError(
                                f"Could not find mirix.yaml config file. Searched in:\n"
                                f"  - {package_dir / 'configs' / 'mirix.yaml'}\n"
                                f"  - {package_dir.parent / 'configs' / 'mirix.yaml'}\n"
                                f"  - {Path('./mirix/configs/mirix.yaml').absolute()}\n"
                                f"Please provide config_path parameter or ensure config file exists."
                            )
        
        # Initialize the underlying agent (with optional backup restore)
        self._agent = AgentWrapper(str(config_path), load_from=load_from)
        
        # Handle model configuration based on parameters:
        # Case 1: model given, config_path None -> load default config then set provided model
        # Case 2: model None, config_path given -> load from config_path and use model from config
        # Case 3: model None, config_path None -> load default config and use default model
        if model is not None:
            # Model explicitly provided - override the config file's model
            self._agent.set_model(model)
            self._agent.set_memory_model(model)
        elif not config_path_provided:
            # No model or config provided - use default model
            default_model = "gemini-2.0-flash"
            self._agent.set_model(default_model)
            self._agent.set_memory_model(default_model)
        # If model is None and config_path was provided, use the model specified in the config file (no override needed)
    
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
        response = self._agent.send_message(
            message=content,
            memorizing=True,
            force_absorb_content=True,
            **kwargs
        )
        return response
    
    def list_users(self) -> Dict[str, Any]:
        """
        List all users in the system.
        
        Returns:
            Dict containing success status, list of users, and any error messages
            
        Example:
            result = memory_agent.list_users()
            if result['success']:
                for user in result['users']:
                    print(f"User: {user['name']} (ID: {user['id']})")
            else:
                print(f"Failed to list users: {result['error']}")
        """
        users = self._agent.client.server.user_manager.list_users()
        return users


    def construct_system_message(self, message: str, user_id: str) -> str:
        """
        Construct a system message from a message.
        """
        return self._agent.construct_system_message(message, user_id)


    def extract_memory_for_system_prompt(self, message: str, user_id: str) -> str:
        """
        Extract memory for system prompt from a message.
        """
        return self._agent.extract_memory_for_system_prompt(message, user_id)

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
                print(f"Found user: {user.name} (ID: {user.id})")
            else:
                print("User not found")
        """
        users = self.list_users()
        for user in users:
            if user.name == user_name:
                return user
        return None

    def chat(self, message: str, **kwargs) -> str:
        """
        Chat with the memory agent.
        
        Args:
            message: Your message/question
            **kwargs: Additional options
            
        Returns:
            Agent's response
            
        Example:
            response = memory_agent.chat("What does John like?")
            # Returns: "According to my memory, John likes pizza."
        """
        response = self._agent.send_message(
            message=message,
            memorizing=False,  # Chat mode, not memorizing by default
            **kwargs
        )
        # Extract text response
        if isinstance(response, dict):
            return response.get("response", response.get("message", str(response)))
        return str(response)
    
    def clear(self) -> Dict[str, Any]:
        """
        Clear all memories.
        
        Note: This requires manual database file removal and app restart.
        
        Returns:
            Dict with warning message and instructions
            
        Example:
            result = memory_agent.clear()
            print(result['warning'])
            for step in result['instructions']:
                print(step)
        """
        return {
            'success': False,
            'warning': 'Memory clearing requires manual database reset.',
            'instructions': [
                '1. Stop the Mirix application/process',
                '2. Remove the database file: ~/.mirix/sqlite.db',
                '3. Restart the Mirix application',
                '4. Initialize a new Mirix agent'
            ],
            'manual_command': 'rm ~/.mirix/sqlite.db',
            'note': 'After removing the database file, you must restart your application and create a new agent instance.'
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
                print(f"Cleared {result['messages_deleted']} messages")
            else:
                print(f"Failed to clear: {result['error']}")
        """
        try:
            if user_id is None:
                # Clear all messages except system messages (original behavior)
                current_messages = self._agent.client.server.agent_manager.get_in_context_messages(
                    agent_id=self._agent.agent_states.agent_state.id,
                    actor=self._agent.client.user
                )
                messages_count = len(current_messages)
                
                # Clear conversation history using the agent manager reset_messages method
                self._agent.client.server.agent_manager.reset_messages(
                    agent_id=self._agent.agent_states.agent_state.id,
                    actor=self._agent.client.user,
                    add_default_initial_messages=True  # Keep system message and initial setup
                )
                
                return {
                    'success': True,
                    'message': f"Successfully cleared conversation history. All user and assistant messages removed (system messages preserved).",
                    'messages_deleted': messages_count
                }
            else:
                # Get the user object by ID
                target_user = self._agent.client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        'success': False,
                        'error': f"User with ID '{user_id}' not found",
                        'messages_deleted': 0
                    }
                
                # Clear messages for specific user (same as FastAPI server implementation)
                # Get current message count for this specific user for reporting
                current_messages = self._agent.client.server.agent_manager.get_in_context_messages(
                    agent_id=self._agent.agent_states.agent_state.id,
                    actor=target_user
                )
                # Count messages belonging to this user (excluding system messages)
                user_messages_count = len([msg for msg in current_messages if msg.role != 'system' and msg.user_id == target_user.id])
                
                # Clear conversation history using the agent manager reset_messages method
                self._agent.client.server.agent_manager.reset_messages(
                    agent_id=self._agent.agent_states.agent_state.id,
                    actor=target_user,
                    add_default_initial_messages=True  # Keep system message and initial setup
                )
                
                return {
                    'success': True,
                    'message': f"Successfully cleared conversation history for {target_user.name}. Messages from other users and system messages preserved.",
                    'messages_deleted': user_messages_count
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'messages_deleted': 0
            }
    
    def save(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current memory state to disk.
        
        Creates a complete backup including agent configuration and database.
        
        Args:
            path: Save directory path (optional). If not provided, generates
                 timestamp-based directory name.
            
        Returns:
            Dict containing success status and backup path
            
        Example:
            result = memory_agent.save("./my_backup")
            if result['success']:
                print(f"Backup saved to: {result['path']}")
            else:
                print(f"Backup failed: {result['error']}")
        """
        from datetime import datetime
        
        if not path:
            path = f"./mirix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            result = self._agent.save_agent(path)
            return {
                'success': True,
                'path': path,
                'message': result.get('message', 'Backup completed successfully')
            }
        except Exception as e:
            return {
                'success': False,
                'path': path,
                'error': str(e)
            }
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load memory state from a backup directory.
        
        Restores both agent configuration and database from backup.
        
        Args:
            path: Path to backup directory
            
        Returns:
            Dict containing success status and any error messages
            
        Example:
            result = memory_agent.load("./my_backup")
            if result['success']:
                print("Memory restored successfully")
            else:
                print(f"Restore failed: {result['error']}")
        """
        try:
            # result = self._agent.load_agent(path)
            config_path = Path(path) / "mirix_config.yaml"
            self._agent = AgentWrapper(str(config_path), load_from=path)
            return {
                    'success': True,
                    'message': 'Memory state loaded successfully'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
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
            setattr(mirix.settings.model_settings, field_name, getattr(new_settings, field_name))
    
    def create_user(self, user_name: str) -> Dict[str, Any]:
        """
        Create a new user in the system.
        
        Args:
            name: The name for the new user
            
        Returns:
            Dict containing success status, message, and user data
            
        Example:
            result = memory_agent.create_user("Alice")
        """
        return self._agent.create_user(name=user_name)['user']
    
    def __call__(self, message: str) -> str:
        """
        Allow using the agent as a callable.
        
        Example:
            memory_agent = Mirix(api_key="...")
            response = memory_agent("What do you remember?")
        """
        return self.chat(message)

    def insert_tool(self, name: str, source_code: str, description: str, args_info: Optional[Dict[str, str]] = None, returns_info: Optional[str] = None, tags: Optional[List[str]] = None, apply_to_agents: Union[List[str], str]='all') -> Dict[str, Any]:
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
        from mirix.services.tool_manager import ToolManager
        from mirix.schemas.tool import Tool as PydanticTool
        from mirix.orm.enums import ToolType
        
        # Initialize tool manager
        tool_manager = ToolManager()
        
        # Check if tool name already exists
        existing_tool = tool_manager.get_tool_by_name(tool_name=name, actor=self._agent.client.user)

        if existing_tool:
            
            created_tool = existing_tool

        else:

            # Set default tags if not provided
            if tags is None:
                tags = ["user_defined"]
            
            # Construct complete source code with docstring
            complete_source_code = self._build_complete_source_code(
                source_code, description, args_info, returns_info
            )
            
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
                json_schema=json_schema
            )
            
            # Use the tool manager's create_or_update_tool method
            created_tool = tool_manager.create_or_update_tool(
                pydantic_tool=pydantic_tool,
                actor=self._agent.client.user
            )

        # Apply tool to all existing agents if requested
        if apply_to_agents:

            # Get all existing agents
            all_agents = self._agent.client.server.agent_manager.list_agents(
                actor=self._agent.client.user,
                limit=1000  # Get all agents
            )

            if apply_to_agents != 'all':
                all_agents = [agent for agent in all_agents if agent.name in all_agents]

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
                    self._agent.client.server.agent_manager.update_agent(
                        agent_id=agent.id,
                        agent_update=UpdateAgent(tool_ids=new_tool_ids),
                        actor=self._agent.client.user
                    )
        
        return {
            'success': True,
            'message': f"Tool '{name}' inserted successfully" + 
                      (" and applied to all existing agents" if apply_to_agents else ""),
            'tool': {
                'id': created_tool.id,
                'name': created_tool.name,
                'description': created_tool.description,
                'tags': created_tool.tags,
                'tool_type': created_tool.tool_type.value if created_tool.tool_type else None
            }
        }
    
    def _build_complete_source_code(self, source_code: str, description: str, args_info: Optional[Dict[str, str]] = None, returns_info: Optional[str] = None) -> str:
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
        func_match = re.search(r'(def\s+\w+\([^)]*\)\s*(?:->\s*[^:]+)?:)', source_code)
        if not func_match:
            raise ValueError("Invalid function definition in source_code")
        
        func_def = func_match.group(1)
        func_body = source_code[func_match.end():].lstrip('\n')
        
        # Build docstring
        docstring_lines = ['    """', f'    {description}']
        
        if args_info:
            docstring_lines.extend(['', '    Args:'])
            for arg_name, arg_desc in args_info.items():
                docstring_lines.append(f'        {arg_name}: {arg_desc}')
        
        if returns_info:
            docstring_lines.extend(['', '    Returns:', f'        {returns_info}'])
        
        docstring_lines.append('    """')
        
        # Combine everything
        complete_code = func_def + '\n' + '\n'.join(docstring_lines) + '\n' + func_body
        
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
            print(f"Episodic memories: {len(memories['episodic'])}")
            print(f"Semantic memories: {len(memories['semantic'])}")
        """
        try:
            # Find the target user
            if user_id:
                target_user = self._agent.client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        'success': False,
                        'error': f"User with ID '{user_id}' not found"
                    }
            else:
                # Find the current active user
                users = self._agent.client.server.user_manager.list_users()
                active_user = next((user for user in users if user.status == 'active'), None)
                target_user = active_user if active_user else (users[0] if users else None)
                
            if not target_user:
                return {
                    'success': False,
                    'error': 'No user found'
                }
            
            memories = {}
            
            # Get episodic memory
            try:
                episodic_manager = self._agent.client.server.episodic_memory_manager
                events = episodic_manager.list_episodic_memory(
                    agent_state=self._agent.agent_states.episodic_memory_agent_state,
                    actor=target_user,
                    limit=50,
                    timezone_str=target_user.timezone
                )
                
                memories['episodic'] = []
                for event in events:
                    memories['episodic'].append({
                        "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                        "summary": event.summary,
                        "details": event.details,
                        "event_type": event.event_type,
                        "tree_path": event.tree_path if hasattr(event, 'tree_path') else [],
                    })
            except Exception as e:
                memories['episodic'] = []
                
            # Get semantic memory
            try:
                semantic_manager = self._agent.client.server.semantic_memory_manager
                semantic_items = semantic_manager.list_semantic_items(
                    agent_state=self._agent.agent_states.semantic_memory_agent_state,
                    actor=target_user,
                    limit=50,
                    timezone_str=target_user.timezone
                )
                
                memories['semantic'] = []
                for item in semantic_items:
                    memories['semantic'].append({
                        "title": item.name,
                        "type": "semantic",
                        "summary": item.summary,
                        "details": item.details,
                        "tree_path": item.tree_path if hasattr(item, 'tree_path') else [],
                    })
            except Exception as e:
                memories['semantic'] = []
                
            # Get procedural memory
            try:
                procedural_manager = self._agent.client.server.procedural_memory_manager
                procedural_items = procedural_manager.list_procedures(
                    agent_state=self._agent.agent_states.procedural_memory_agent_state,
                    actor=target_user,
                    limit=50,
                    timezone_str=target_user.timezone
                )
                
                memories['procedural'] = []
                for item in procedural_items:
                    import json
                    # Parse steps if it's a JSON string
                    steps = item.steps
                    if isinstance(steps, str):
                        try:
                            steps = json.loads(steps)
                            # Extract just the instruction text for simpler display
                            if isinstance(steps, list) and steps and isinstance(steps[0], dict):
                                steps = [step.get('instruction', str(step)) for step in steps]
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # If parsing fails, keep as string and split by common delimiters
                            if isinstance(steps, str):
                                steps = [s.strip() for s in steps.replace('\n', '|').split('|') if s.strip()]
                            else:
                                steps = []
                    
                    memories['procedural'].append({
                        "title": item.entry_type,
                        "type": "procedural", 
                        "summary": item.summary,
                        "steps": steps if isinstance(steps, list) else [],
                        "tree_path": item.tree_path if hasattr(item, 'tree_path') else [],
                    })
            except Exception as e:
                memories['procedural'] = []
                
            # Get resource memory
            try:
                resource_manager = self._agent.client.server.resource_memory_manager
                resources = resource_manager.list_resources(
                    agent_state=self._agent.agent_states.resource_memory_agent_state,
                    actor=target_user,
                    limit=50,
                    timezone_str=target_user.timezone
                )
                
                memories['resources'] = []
                for resource in resources:
                    memories['resources'].append({
                        "filename": resource.title,
                        "type": resource.resource_type,
                        "summary": resource.summary or (resource.content[:200] + "..." if len(resource.content) > 200 else resource.content),
                        "last_accessed": resource.updated_at.isoformat() if resource.updated_at else None,
                        "size": resource.metadata_.get("size") if resource.metadata_ else None,
                        "tree_path": resource.tree_path if hasattr(resource, 'tree_path') else [],
                    })
            except Exception as e:
                memories['resources'] = []
                
            # Get core memory
            try:
                core_memory = self._agent.client.get_in_context_memory(self._agent.agent_states.agent_state.id)
                memories['core'] = []
                total_characters = 0

                for block in core_memory.blocks:
                    if block.value and block.value.strip() and block.label.lower() != "persona":
                        block_chars = len(block.value)
                        total_characters += block_chars

                        memories['core'].append({
                            "aspect": block.label,
                            "understanding": block.value,
                            "character_count": block_chars,
                            "total_characters": total_characters,
                            "max_characters": block.limit,
                            "last_updated": None
                        })
            except Exception as e:
                memories['core'] = []
                
            # Get credentials memory
            try:
                knowledge_vault_manager = self._agent.client.server.knowledge_vault_manager
                vault_items = knowledge_vault_manager.list_knowledge(
                    actor=target_user,
                    agent_state=self._agent.agent_states.knowledge_vault_agent_state,
                    limit=50,
                    timezone_str=target_user.timezone
                )
                
                memories['credentials'] = []
                for item in vault_items:
                    memories['credentials'].append({
                        "caption": item.caption,
                        "entry_type": item.entry_type,
                        "source": item.source,
                        "sensitivity": item.sensitivity,
                        "content": "••••••••••••" if item.sensitivity == 'high' else item.secret_value,
                    })
            except Exception as e:
                memories['credentials'] = []
            
            return {
                'success': True,
                'user_id': target_user.id,
                'user_name': target_user.name,
                'memories': memories,
                'summary': {
                    'episodic_count': len(memories.get('episodic', [])),
                    'semantic_count': len(memories.get('semantic', [])),
                    'procedural_count': len(memories.get('procedural', [])),
                    'resources_count': len(memories.get('resources', [])),
                    'core_count': len(memories.get('core', [])),
                    'credentials_count': len(memories.get('credentials', []))
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def edit_memories(self, user_id: Optional[str] = None, memory_type: str = "all", action: str = "clear") -> Dict[str, Any]:
        """
        Edit memories for a specific user.
        
        Args:
            user_id: User ID to edit memories for. If None, uses current active user.
            memory_type: Type of memory to edit ("episodic", "semantic", "procedural", "resources", "core", "credentials", "conversation", "all")
            action: Action to perform ("clear", "export")
            
        Returns:
            Dict containing success status and results
            
        Example:
            # Clear conversation history for specific user
            result = memory_agent.edit_memories(user_id="user_123", memory_type="conversation", action="clear")
            
            # Export all memories for user
            result = memory_agent.edit_memories(user_id="user_123", action="export")
        """
        try:
            # Find the target user
            if user_id:
                target_user = self._agent.client.server.user_manager.get_user_by_id(user_id)
                if not target_user:
                    return {
                        'success': False,
                        'error': f"User with ID '{user_id}' not found"
                    }
            else:
                # Find the current active user
                users = self._agent.client.server.user_manager.list_users()
                active_user = next((user for user in users if user.status == 'active'), None)
                target_user = active_user if active_user else (users[0] if users else None)
                
            if not target_user:
                return {
                    'success': False,
                    'error': 'No user found'
                }
            
            if action == "clear":
                if memory_type == "conversation":
                    # Clear conversation history for the user
                    current_messages = self._agent.client.server.agent_manager.get_in_context_messages(
                        agent_id=self._agent.agent_states.agent_state.id,
                        actor=target_user
                    )
                    # Count messages belonging to this user (excluding system messages)
                    user_messages_count = len([msg for msg in current_messages if msg.role != 'system' and msg.user_id == target_user.id])
                    
                    # Clear conversation history
                    self._agent.client.server.agent_manager.reset_messages(
                        agent_id=self._agent.agent_states.agent_state.id,
                        actor=target_user,
                        add_default_initial_messages=True
                    )
                    
                    return {
                        'success': True,
                        'message': f"Successfully cleared conversation history for {target_user.name}",
                        'messages_deleted': user_messages_count
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Clear action currently only supports "conversation" memory type'
                    }
                    
            elif action == "export":
                # Export memories to Excel file
                from datetime import datetime
                from pathlib import Path
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"mirix_memories_{target_user.name}_{timestamp}.xlsx"
                
                # Use home directory for export
                file_path = str(Path.home() / filename)
                
                # Determine memory types to export
                if memory_type == "all":
                    memory_types = ["episodic", "semantic", "procedural", "resources", "core", "credentials"]
                else:
                    memory_types = [memory_type]
                
                result = self._agent.export_memories_to_excel(
                    actor=target_user,
                    file_path=file_path,
                    memory_types=memory_types,
                    include_embeddings=False
                )
                
                return result
            else:
                return {
                    'success': False,
                    'error': f'Unsupported action: {action}. Supported actions are "clear" and "export"'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }