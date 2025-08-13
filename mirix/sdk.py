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
    
    def clear_conversation_history(self) -> Dict[str, Any]:
        """
        Clear conversation history while preserving memories.
        
        This removes all user and assistant messages from the conversation
        history but keeps system messages and all stored memories intact.
        
        Returns:
            Dict containing success status, message, and count of deleted messages
            
        Example:
            result = memory_agent.clear_conversation_history()
            if result['success']:
                print(f"Cleared {result['messages_deleted']} messages")
            else:
                print(f"Failed to clear: {result['error']}")
        """
        try:
            # Get current message count for reporting
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
    
    def __call__(self, message: str) -> str:
        """
        Allow using the agent as a callable.
        
        Example:
            memory_agent = Mirix(api_key="...")
            response = memory_agent("What do you remember?")
        """
        return self.chat(message)
