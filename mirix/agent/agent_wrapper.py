import io
import os
import re
import time
import json
import yaml
import uuid
import copy
import pytz
import openai
import base64
import tiktoken
import traceback
import warnings
import threading
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from tqdm import tqdm
from google import genai
from functools import partial
from dotenv import load_dotenv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from mirix.utils import parse_json
from typing import Optional
import queue
from PIL import Image
import logging
from ..voice_utils import process_voice_files, convert_base64_to_audio_segment
from .app_utils import encode_image_from_pil, encode_image

# Import the separated components
from mirix.agent.message_queue import MessageQueue
from mirix.agent.temporary_message_accumulator import TemporaryMessageAccumulator
from mirix.agent.upload_manager import UploadManager
from mirix.agent.agent_states import AgentStates
from mirix.agent.agent_configs import AGENT_CONFIGS
from mirix.agent.app_constants import TEMPORARY_MESSAGE_LIMIT, MAXIMUM_NUM_IMAGES_IN_CLOUD, GEMINI_MODELS, OPENAI_MODELS, WITH_REFLEXION_AGENT, WITH_BACKGROUND_AGENT
from mirix.schemas.mirix_message import MessageType

from mirix import create_client
from mirix import LLMConfig, EmbeddingConfig
from mirix.schemas.agent import AgentType
from mirix.prompts import gpt_system
from mirix.schemas.memory import ChatMemory
from mirix.settings import model_settings

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(levelname)s: %(message)s')

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_image_mime_type(image_path):
    """
    Detect the MIME type of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: MIME type (e.g., 'image/jpeg', 'image/png', etc.)
    """
    try:
        # Use PIL to detect the image format
        with Image.open(image_path) as img:
            format_lower = img.format.lower() if img.format else None
            
            # Map PIL formats to MIME types
            format_to_mime = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg', 
                'png': 'image/png',
                'gif': 'image/gif',
                'bmp': 'image/bmp',
                'webp': 'image/webp',
                'tiff': 'image/tiff',
                'tif': 'image/tiff',
                'ico': 'image/x-icon',
                'svg': 'image/svg+xml'
            }
            
            return format_to_mime.get(format_lower, 'image/jpeg')  # Default to jpeg if unknown
            
    except Exception as e:
        # If PIL fails, try using mimetypes module as fallback
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type.startswith('image/'):
            return mime_type
        
        # Final fallback to jpeg
        return 'image/jpeg'

class AgentWrapper():
    
    def __init__(self, agent_config_file, load_from=None):

        # If load_from is specified, restore the database first before any agent initialization
        if load_from is not None:
            self._restore_database_before_init(load_from)

        with open(agent_config_file, "r") as f:
            agent_config = yaml.safe_load(f)

        self.agent_config = agent_config
        self.agent_name = agent_config['agent_name']
        self.model_name = agent_config['model_name']
        self.model_provider = agent_config.get('model_provider', None)
        self.is_screen_monitor = agent_config.get('is_screen_monitor', False)
        self.chat_agent_standalone = True

        # Initialize logger early
        self.logger = logging.getLogger(f"Mirix.AgentWrapper.{self.agent_name}")
        self.logger.setLevel(logging.INFO)

        self.client = create_client()
        self.client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini")) 
        # self.client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-3-small"))
        self.client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-004"))

        # Initialize agent states container
        self.agent_states = AgentStates()

        if len(self.client.list_agents()) > 0:

            self.client.server.tool_manager.upsert_base_tools(self.client.user)

            all_agent_states = self.client.list_agents()

            for agent_state in all_agent_states:
                if agent_state.name == 'chat_agent':
                    self.agent_states.agent_state = agent_state
                elif agent_state.name == 'episodic_memory_agent':
                    self.agent_states.episodic_memory_agent_state = agent_state
                elif agent_state.name == 'procedural_memory_agent':
                    self.agent_states.procedural_memory_agent_state = agent_state
                elif agent_state.name == 'knowledge_vault_agent':
                    self.agent_states.knowledge_vault_agent_state = agent_state
                elif agent_state.name == 'meta_memory_agent':
                    self.agent_states.meta_memory_agent_state = agent_state
                elif agent_state.name == 'semantic_memory_agent':
                    self.agent_states.semantic_memory_agent_state = agent_state
                elif agent_state.name == 'core_memory_agent': 
                    self.agent_states.core_memory_agent_state = agent_state
                elif agent_state.name == 'resource_memory_agent':
                    self.agent_states.resource_memory_agent_state = agent_state
                elif agent_state.name == 'reflexion_agent':
                    self.agent_states.reflexion_agent_state = agent_state
                elif agent_state.name == 'background_agent':
                    self.agent_states.background_agent_state = agent_state

                system_prompt = gpt_system.get_system_text('base/' + agent_state.name) if not self.is_screen_monitor else gpt_system.get_system_text('screen_monitor/' + agent_state.name)

                self.client.server.agent_manager.update_agent_tools_and_system_prompts(
                    agent_id=agent_state.id,
                    actor=self.client.user,
                    system_prompt=system_prompt
                )
            
            if self.agent_states.reflexion_agent_state is None:
                reflexion_agent_state = self.client.create_agent(
                    name='reflexion_agent',
                    memory=self.agent_states.agent_state.memory,
                    agent_type=AgentType.reflexion_agent,
                    system=gpt_system.get_system_text('base/reflexion_agent'),
                )
                setattr(self.agent_states, 'reflexion_agent_state', reflexion_agent_state)
            
            if self.agent_states.background_agent_state is None:
                background_agent_state = self.client.create_agent(
                    name='background_agent',
                    agent_type=AgentType.background_agent,
                    memory=self.agent_states.agent_state.memory,
                    system=gpt_system.get_system_text('base/background_agent'),
                )
                setattr(self.agent_states, 'background_agent_state', background_agent_state)
            
        else:

            core_memory = ChatMemory(
                human="",
                persona="You are a helpful personal assitant who can help the user remember things."
            )

            # Create agents in a loop using imported configuration
            for config in AGENT_CONFIGS:
                if config['name'] == 'chat_agent':
                    # chat_agent has different parameters
                    agent_state = self.client.create_agent(
                        name=config['name'],
                        memory=core_memory,
                        system = gpt_system.get_system_text('screen_monitor/' + config['name']) if self.is_screen_monitor else gpt_system.get_system_text('base/' + config['name'])
                    )
                else:
                    # All other agents follow the same pattern
                    agent_state = self.client.create_agent(
                        name=config['name'],
                        agent_type=config['agent_type'],
                        memory=core_memory,
                        system = gpt_system.get_system_text('screen_monitor/' + config['name']) if self.is_screen_monitor else gpt_system.get_system_text('base/' + config['name']),
                        include_base_tools=config['include_base_tools'],
                    )
                
                # Set the agent state on the appropriate attribute
                setattr(self.agent_states, config['attr_name'], agent_state)
            
        # for agent_state in all_agent_states:
        #     messages = self.client.server.agent_manager.get_in_context_messages(agent_id=agent_state.id, actor=self.client.user)
        #     print(agent_state.name, len(messages))

        self.set_timezone(self.client.server.user_manager.get_user_by_id(self.client.user_id).timezone)
        self.set_persona("helpful_assistant") # This will now also set self.active_persona_name

        # Initialize screenshot setting (default to True)
        self.include_recent_screenshots = True

        # Initialize components that all mirix models need
        self.message_queue = MessageQueue()
        
        # Track missing API keys for frontend to query
        self.missing_api_keys = []
        
        # Initialize upload manager and URI tracking for file handling
        if self.model_name in GEMINI_MODELS:
            success = self._initialize_gemini_components()
            if not success:
                self.missing_api_keys.append('GEMINI_API_KEY')
        else:
            # For non-GEMINI models, initialize minimal components
            self.google_client = None
            self.existing_files = []
            self.uri_to_create_time = {}
            self.upload_manager = None

        print(f"🔄 Initializing model: {self.model_name}")

        self.set_model(self.model_name)
        self.set_memory_model(self.model_name)
        
        # Initialize temporary message accumulator for ALL mirix models
        self.temp_message_accumulator = TemporaryMessageAccumulator(
            client=self.client,
            google_client=self.google_client,
            timezone=self.timezone,
            upload_manager=self.upload_manager,
            message_queue=self.message_queue,
            model_name=self.model_name,
            temporary_message_limit=TEMPORARY_MESSAGE_LIMIT
        )
        
        # Pass URI tracking to accumulator
        self.temp_message_accumulator.uri_to_create_time = self.uri_to_create_time

        # For GEMINI models, extract all unprocessed images and fill temporary_messages
        if self.model_name in GEMINI_MODELS and self.google_client is not None:
            self._process_existing_uploaded_files()

    def update_chat_agent_system_prompt(self, is_screen_monitoring: bool):
        '''
        Update chat agent system prompt based on screen monitoring status
        '''

        if self.chat_agent_standalone == is_screen_monitoring:

            if self.is_screen_monitor:
                file_name = 'screen_monitor/chat_agent'
            else:
                file_name = 'base/chat_agent'
            
            if is_screen_monitoring:
                file_name = file_name + '_monitor_on'
                self.chat_agent_standalone = False
            else:
                self.chat_agent_standalone = True
                
            self.client.server.agent_manager.update_system_prompt(
                agent_id=self.agent_states.agent_state.id, 
                system_prompt=gpt_system.get_system_text(file_name), 
                actor=self.client.user)

    def _restore_database_before_init(self, folder_path: str):
        """
        Restore database before agent initialization to avoid permission conflicts.
        This is called during __init__ if load_from parameter is provided.
        """
        import subprocess
        import shutil
        import json
        from pathlib import Path
        from mirix.settings import settings
        
        folder = Path(folder_path)
        print(f"🔄 Restoring database from {folder_path} before agent initialization...")
        
        try:
            # Check if folder exists
            if not folder.exists():
                raise ValueError(f'Backup folder {folder_path} does not exist')
            
            # Load agent configuration if available
            config_file = folder / "agent_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    agent_config = json.load(f)
                backup_type = agent_config.get('backup_type', 'sqlite')
            else:
                # Determine backup type from files present
                if (folder / "mirix_database.sql").exists() or (folder / "mirix_database.dump").exists():
                    backup_type = 'postgresql'
                elif (folder / "sqlite.db").exists():
                    backup_type = 'sqlite'
                else:
                    raise ValueError(f'No valid backup files found in {folder_path}')
            
            print(f"📁 Detected backup type: {backup_type}")
            
            # Handle PostgreSQL restoration
            if backup_type == 'postgresql':
                if not settings.mirix_pg_uri_no_default:
                    raise ValueError('Cannot restore PostgreSQL backup: Current setup is using SQLite. Please configure PostgreSQL first.')
                
                # Parse connection details
                if settings.pg_uri:
                    import urllib.parse as urlparse
                    parsed = urlparse.urlparse(settings.pg_uri)
                    db_host = parsed.hostname or 'localhost'
                    db_port = parsed.port or 5432
                    db_user = parsed.username or 'mirix'
                    db_name = parsed.path.lstrip('/') or 'mirix'
                else:
                    db_host = settings.pg_host or 'localhost'
                    db_port = settings.pg_port or 5432
                    db_user = settings.pg_user or 'mirix'
                    db_name = settings.pg_db or 'mirix'
                
                # Use compressed backup if available, otherwise SQL backup
                compressed_backup = folder / "mirix_database.dump"
                sql_backup = folder / "mirix_database.sql"
                
                if compressed_backup.exists():
                    print("📦 Restoring from compressed backup...")
                    restore_cmd = [
                        'pg_restore',
                        '-h', str(db_host),
                        '-p', str(db_port),
                        '-U', db_user,
                        '-d', db_name,
                        '--no-owner',
                        '--no-privileges',
                        '--clean',
                        '--if-exists',
                        '--disable-triggers',
                        str(compressed_backup)
                    ]
                elif sql_backup.exists():
                    print("📝 Restoring from SQL backup...")
                    restore_cmd = [
                        'psql',
                        '-h', str(db_host),
                        '-p', str(db_port),
                        '-U', db_user,
                        '-d', db_name,
                        '-v', 'ON_ERROR_STOP=0',
                        '-f', str(sql_backup)
                    ]
                else:
                    raise ValueError(f'No PostgreSQL backup files found in {folder_path}')
                
                result_proc = subprocess.run(restore_cmd, capture_output=True, text=True, check=False)
                
                if result_proc.returncode != 0:
                    stderr_text = result_proc.stderr or ""
                    # Allow extension permission errors but not other critical errors
                    if "must be owner of extension vector" in stderr_text:
                        print("⚠️ Extension permission warnings detected but continuing...")
                    else:
                        raise RuntimeError(f"PostgreSQL restore failed: {stderr_text}")
                
                print("✅ PostgreSQL database restored successfully!")
                
            # Handle SQLite restoration
            elif backup_type == 'sqlite':
                print("📁 Restoring SQLite database...")
                sqlite_backup = folder / "sqlite.db"
                if not sqlite_backup.exists():
                    raise ValueError(f'SQLite backup file not found in {folder_path}')
                
                sqlite_dest = Path.home() / ".mirix" / "sqlite.db"
                
                # Ensure destination directory exists
                sqlite_dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the database file
                shutil.copyfile(sqlite_backup, sqlite_dest)
                import os
                os.chmod(sqlite_dest, 0o666)  # Make it writable
                
                print("✅ SQLite database restored successfully!")
                
        except Exception as e:
            print(f"❌ Database restoration failed: {str(e)}")
            raise RuntimeError(f"Failed to restore database before agent initialization: {str(e)}")

    def set_include_recent_screenshots(self, include_recent_screenshots: bool):
        self.include_recent_screenshots = include_recent_screenshots

    def _initialize_gemini_components(self) -> bool:
        """
        Initialize Gemini client and related components.
        Returns True if successful, False if API key is missing or initialization fails.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_override_key = self.client.server.provider_manager.get_gemini_override_key()
        gemini_api_key = gemini_override_key or gemini_api_key
        
        if not gemini_api_key:
            self.logger.info("Info: GEMINI_API_KEY not found. Gemini features will be available after API key is provided.")
            self.google_client = None
            self.existing_files = []
            self.uri_to_create_time = {}
            self.upload_manager = None
            return False
            
        try:
            self.google_client = genai.Client(api_key=gemini_api_key)
            
            # self.logger.info("Retrieving existing files from Google Clouds...")
            
            # try:
            #     existing_files = [x for x in self.google_client.files.list()]
            # except Exception as e:
            #     self.logger.error(f"Error retrieving existing files from Google Clouds: {e}")
            #     existing_files = []
            # existing_image_names = set([file.name for file in existing_files])
            
            # self.logger.info(f"# of Existing files in Google Clouds: {len(existing_image_names)}")

            existing_files = []
            existing_image_names = set([file.name for file in existing_files])

            # update the database, delete the files that are in the database but got deleted somehow (potentially due to the calls unrelated to Mirix) in the cloud
            for file_name in self.client.server.cloud_file_mapping_manager.list_all_cloud_file_ids():
                if file_name not in existing_image_names:
                    self.client.server.cloud_file_mapping_manager.delete_mapping(cloud_file_id=file_name)
                else:
                    assert file_name in existing_image_names

            # after this: every file in database, we can find it in the cloud
            # i.e., local database <= cloud

            cloud_file_names_in_database_set = set(self.client.server.cloud_file_mapping_manager.list_all_cloud_file_ids())

            # since there might be images that belong to other projects, we need to delete those in `existing_files`
            remaining_indices = []
            for idx, file in enumerate(existing_files):
                if file.name in cloud_file_names_in_database_set:
                    remaining_indices.append(idx)
            
            # after this, every file in 'existing_files', we can find it in the database

            for file_name in self.client.server.cloud_file_mapping_manager.list_all_cloud_file_ids():
                assert file_name in existing_image_names

            existing_files = [existing_files[i] for i in remaining_indices]
            self.existing_files = existing_files
            self.uri_to_create_time = {file.uri: {'create_time': file.create_time, 'filename': file.name} for file in existing_files}

            self.logger.info(f"# of Existing files in Google Clouds that belong to Mirix: {len(self.uri_to_create_time)}")

            # Initialize upload manager for GEMINI models
            self.upload_manager = UploadManager(self.google_client, self.client, self.existing_files, self.uri_to_create_time)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Warning: Failed to initialize Gemini client: {e}")
            self.logger.warning("Gemini features will be unavailable until a valid API key is provided.")
            self.google_client = None
            self.existing_files = []
            self.uri_to_create_time = {}
            self.upload_manager = None
            return False

    def _process_existing_uploaded_files(self):
        """Process any existing uploaded files for Gemini models."""
        uploaded_mappings = self.client.server.cloud_file_mapping_manager.list_files_with_status(status='uploaded')

        count = 0
        for mapping in uploaded_mappings:
            file_ref = [file for file in self.existing_files if file.name == mapping.cloud_file_id][0]

            self.temp_message_accumulator.temporary_messages.append(
                (mapping.timestamp, {'image_uris': [file_ref],
                                     'audio_segments': None,
                                     'message': None})
            )
            count += 1
            if count == TEMPORARY_MESSAGE_LIMIT:
                self.temp_message_accumulator.absorb_content_into_memory(self.agent_states)
                count = 0

    def delete_files(self, file_names, google_client):
        for file_name in file_names:
            try:
                google_client.files.delete(name=file_name)
                self.client.server.cloud_file_mapping_manager.delete_mapping(cloud_file_id=file_name)
            except:
                continue

    def set_timezone(self, timezone_str):
        """
        timezone_str: Something like "Asia/Shanghai (UTC+8:00)".
        """
        self.logger.info("Setting timezone to: %s", timezone_str)
        self.logger.info(timezone_str.split(" ")[-1])

        self.client.server.user_manager.update_user_timezone(timezone_str, self.client.user.id)
        self.timezone_str = timezone_str
        self.timezone = pytz.timezone(timezone_str.split(" (")[0])

    def set_persona(self, persona_name):
        # Update the persona for the agent using the persona name
        blocks = self.client.server.block_manager.get_blocks(self.client.user)
        persona_block = [block for block in blocks if block.label == 'persona'][0]
        
        # Get the persona text by name
        from mirix.prompts import gpt_persona
        from mirix.schemas.block import BlockUpdate
        persona_text = gpt_persona.get_persona_text(persona_name)
        
        # Update the agent's persona block
        self.client.server.block_manager.update_block(
            block_id=persona_block.id,
            block_update=BlockUpdate(value=persona_text),
            actor=self.client.user
        )
        
        self.active_persona_name = persona_name

    def _determine_model_provider(self, model_name: str, custom_agent_config: dict = None) -> str:
        """
        Determine the effective model provider based on configuration override or model defaults.
        Returns the provider type (e.g., 'openai', 'azure_openai', 'google_ai', 'anthropic').
        """
        # Check for provider override in custom_agent_config first
        if custom_agent_config and custom_agent_config.get('model_provider'):
            return custom_agent_config['model_provider']
        
        # Check for provider override in instance config
        if self.model_provider:
            return self.model_provider
        
        # Fall back to default provider based on model
        if model_name in GEMINI_MODELS:
            return "google_ai"
        elif 'claude' in model_name.lower():
            return "anthropic"
        elif model_name in OPENAI_MODELS:
            return "openai"
        else:
            raise ValueError(f"Invalid model provider: {model_name}")
    
    def _create_llm_config_for_provider(self, model_name: str, provider: str, custom_agent_config: dict = None) -> 'LLMConfig':
        """
        Create LLMConfig based on the specified provider.
        """
        if provider == "azure_openai":
            # Azure OpenAI configuration
            endpoint = (custom_agent_config.get('model_endpoint') if custom_agent_config 
                       else self.agent_config.get('model_endpoint', 'https://your-resource.openai.azure.com/'))
            api_version = (custom_agent_config.get('api_version') if custom_agent_config 
                          else self.agent_config.get('api_version', '2024-10-01-preview'))
            azure_deployment = (custom_agent_config.get('azure_deployment') if custom_agent_config 
                               else self.agent_config.get('azure_deployment', model_name))
            
            # Get custom API key if provided
            api_key = None
            if custom_agent_config and custom_agent_config.get('api_key'):
                api_key = custom_agent_config['api_key']
            elif self.agent_config.get('api_key'):
                api_key = self.agent_config['api_key']
            
            llm_config = LLMConfig(
                model=model_name,
                model_endpoint_type="azure_openai",
                model_endpoint=endpoint,
                context_window=128000,
                # Use the new schema fields instead of dynamic assignment
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=azure_deployment,
                api_key=api_key
            )
                
        elif provider == "google_ai":
            llm_config = LLMConfig(
                model_endpoint_type="google_ai",
                model_endpoint="https://generativelanguage.googleapis.com",
                model=model_name,
                context_window=1000000
            )
            
        elif provider == "anthropic":
            llm_config = LLMConfig(
                model_endpoint_type="anthropic",
                model_endpoint="https://api.anthropic.com/v1",
                model=model_name,
                context_window=200000
            )
            
        elif provider == "openai":
            if model_name == 'gpt-4o-mini' or model_name == 'gpt-4o':
                llm_config = LLMConfig.default_config(model_name)
            else:
                llm_config = LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint="https://api.openai.com/v1",
                    model_wrapper=None,
                    context_window=128000,
                )
        else:
            # Custom provider - use custom config or fallback to OpenAI-compatible
            if custom_agent_config is not None:
                assert 'model_endpoint' in custom_agent_config, "model_endpoint is required for custom models"
                llm_config = LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint=custom_agent_config['model_endpoint'],
                    model_wrapper=None,
                    api_key=custom_agent_config.get('api_key'),
                    **custom_agent_config.get('generation_config', {})
                )
            else:
                assert 'model_endpoint' in self.agent_config, "model_endpoint is required for custom models"
                llm_config = LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint=self.agent_config['model_endpoint'],
                    model_wrapper=None,
                    api_key=self.agent_config.get('api_key'),
                    **self.agent_config.get('generation_config', {})
                )
        
        return llm_config

    def set_model(self, model_name: str, custom_agent_config: dict = None) -> dict:
        """
        Set the model for the agent.
        Returns a dictionary with success status and any missing API keys.
        """
        try:
            self.model_name = model_name
            
            # Determine the effective provider
            provider = self._determine_model_provider(model_name, custom_agent_config)
            
            # Create LLM config based on provider
            llm_config = self._create_llm_config_for_provider(model_name, provider, custom_agent_config)
            
            # Update LLM config for the client
            self.client.set_default_llm_config(llm_config)
            self.client.server.agent_manager.update_llm_config(
                agent_id=self.agent_states.agent_state.id,
                llm_config=llm_config,
                actor=self.client.user
            )
            # set the model for reflexion agent
            self.client.server.agent_manager.update_llm_config(
                agent_id=self.agent_states.reflexion_agent_state.id,
                llm_config=llm_config,
                actor=self.client.user
            )
            
            # Check for missing API keys for the new model
            status = self.check_api_key_status()
            
            result = {
                'success': True, 
                'message': f'Model set to {model_name}',
                'missing_keys': status['missing_keys'],
                'model_requirements': status['model_requirements']
            }
            
            if status['missing_keys']:
                result['message'] += f", but missing API keys: {', '.join(status['missing_keys'])}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error setting model: {e}")
            return {
                'success': False, 
                'message': f'Failed to set model: {str(e)}',
                'missing_keys': [],
                'model_requirements': {}
            }

    def set_memory_model(self, new_model, custom_agent_config: dict = None):
        """Set the model specifically for memory management operations"""
        
        # Define allowed memory models
        ALLOWED_MEMORY_MODELS = GEMINI_MODELS + OPENAI_MODELS

        # Determine the effective provider
        provider = self._determine_model_provider(new_model, custom_agent_config)

        # Validate the model - allow custom models to proceed with validation
        if new_model not in ALLOWED_MEMORY_MODELS and not custom_agent_config and not hasattr(self, 'agent_config'):
            # Invalid model and no custom config
            self.logger.warning(f'Invalid memory model. Only {", ".join(ALLOWED_MEMORY_MODELS)} are supported.')

        llm_config = self._create_llm_config_for_provider(new_model, provider, custom_agent_config)

        # Update only the memory-related agents (all agents except chat_agent)
        memory_agent_names = [
            'episodic_memory_agent',
            'procedural_memory_agent', 
            'knowledge_vault_agent',
            'meta_memory_agent',
            'semantic_memory_agent',
            'core_memory_agent',
            'resource_memory_agent',
            'reflexion_agent',
            'background_agent'
        ]
        
        for agent_state in self.client.list_agents():
            if agent_state.name in memory_agent_names:
                self.client.server.agent_manager.update_llm_config(
                    agent_id=agent_state.id, 
                    llm_config=llm_config, 
                    actor=self.client.user
                )

        # Update the memory model name
        self.memory_model_name = new_model
        
        # Update temp_message_accumulator needs_upload based on the new model
        if hasattr(self, 'temp_message_accumulator'):
            self.temp_message_accumulator.update_model(new_model)
        
        # Determine required keys based on model type
        required_keys = []
        if new_model in GEMINI_MODELS:
            required_keys = ['GEMINI_API_KEY']
        elif new_model in OPENAI_MODELS:
            required_keys = ['OPENAI_API_KEY']
        
        return {
            'success': True,
            'message': f'Memory model set to {new_model} successfully.',
            'missing_keys': [],
            'model_requirements': {
                'current_model': new_model,
                'required_keys': required_keys
            }
        }
    
    def get_current_model(self) -> str:
        """
        Get the current model name being used by the chat agent.
        Returns the model name string.
        """
        return self.model_name

    def get_current_memory_model(self) -> str:
        """
        Get the current model name being used by the memory manager.
        Returns the model name string.
        """
        return getattr(self, 'memory_model_name', self.model_name)  # Fallback to chat model if not set

    def get_persona_details(self) -> dict[str, str]:
        """
        Retrieves all persona names and their corresponding prompt texts by scanning the personas directory.
        """
        import os
        from mirix.prompts import gpt_persona
        
        persona_details = {}
        personas_dir = os.path.join(os.path.dirname(gpt_persona.__file__), "personas")
        
        try:
            # Scan the personas directory for .txt files
            if os.path.exists(personas_dir):
                for filename in os.listdir(personas_dir):
                    if filename.endswith('.txt'):
                        persona_name = filename[:-4]  # Remove .txt extension
                        try:
                            persona_text = gpt_persona.get_persona_text(persona_name)
                            persona_details[persona_name] = persona_text
                        except FileNotFoundError:
                            # Skip files that can't be read
                            continue
        except Exception as e:
            self.logger.error(f"Error scanning personas directory: {e}")
            # Fallback to empty dict if scanning fails
            
        return persona_details

    def get_core_memory_persona(self) -> str:
        """
        Get the current persona text from the agent's core memory
        """
        blocks = self.client.server.block_manager.get_blocks(self.client.user)
        persona_block = [block for block in blocks if block.label == 'persona'][0]
        return persona_block.value

    def update_core_memory_persona(self, text: str):
        """
        Update the persona text in the agent's core memory
        """
        self.client.update_agent_memory_block(
            agent_id=self.agent_states.agent_state.id,
            label='persona',
            value=text
        )
    
    def update_core_memory(self, text: str, label: str):
        """
        Update the core memory with the given text and label
        """
        self.client.update_agent_memory_block(
            agent_id=self.agent_states.agent_state.id,
            label=label,
            value=text
        )

    def apply_persona_template(self, persona_name: str):
        """
        Apply a persona template to the agent's core memory
        """
        # Get the persona template text
        from mirix.prompts import gpt_persona
        persona_text = gpt_persona.get_persona_text(persona_name)
        
        # Update the core memory with this text
        self.update_core_memory_persona(persona_text)
        
        # Update the active persona name
        self.active_persona_name = persona_name

    def clear_old_screenshots(self):
        
        queue_length = self.message_queue.get_queue_length()
            
        if queue_length > 0:
            return # do not clear if there are messages in the queue

        if len(self.uri_to_create_time) > MAXIMUM_NUM_IMAGES_IN_CLOUD:
            # find the oldest files according to self.uri_to_create_time
            files_to_delete = sorted(self.uri_to_create_time.items(), key=lambda x: x[1]['create_time'])[:len(self.uri_to_create_time) - MAXIMUM_NUM_IMAGES_IN_CLOUD]
            file_names_to_delete = [x[1]['filename'] for x in files_to_delete]

            for x in files_to_delete:
                del self.uri_to_create_time[x[0]]

            self.logger.info(f"Deleting files: {file_names_to_delete}")
            # Only attempt to delete if google_client is initialized
            if self.google_client is not None:
                threading.Thread(target=self.delete_files, args=([x for x in file_names_to_delete], self.google_client)).start()
            else:
                self.logger.warning("Warning: Cannot delete files from Google Cloud - Gemini client not initialized")

    def reflextion_on_memory(self):
        """
        Run the reflexion process with comprehensive memory analysis:
        1. Call specific agents to remove redundancy in each memory type (episodic, semantic, core, resource, procedural, knowledge vault)
        2. Call reflexion agent to identify and resolve potential conflicts between memories
        3. Call agents to identify patterns and analyze new memories
        """
        
        self.logger.info("Starting comprehensive reflexion on memory...")
        
        # Step 1: Call specific memory agents to remove redundancy
        self.logger.info("Step 1: Calling memory agents to remove redundancy...")
        redundancy_results = self._call_agents_to_remove_redundancy()
        
        # Step 2: Call reflexion agent to identify and resolve conflicts
        self.logger.info("Step 2: Calling reflexion agent to resolve conflicts...")
        conflict_results = self._call_reflexion_agent_for_conflicts()
        
        # Step 3: Call agents to connect memories. 
        # connect("epi_id", "sem_id")
        

        # Step 4: Call agents to identify patterns and create new memories
        self.logger.info("Step 3: Calling agents to analyze patterns and create insights...")
        pattern_results = self._call_agents_for_pattern_analysis()
        
        # Final summary
        final_summary = {
            'redundancy_actions': redundancy_results,
            'conflict_resolutions': conflict_results,
            'pattern_insights': pattern_results
        }
        
        self.logger.info("Reflexion process completed with actual agent actions.")
        return final_summary

    def _call_agents_to_remove_redundancy(self):
        """Call specific memory agents to actually remove redundancy in their respective memory types"""
        redundancy_results = {}
        
        # Call episodic memory agent to remove redundancy
        self.logger.info("Calling episodic memory agent to remove redundancy...")
        try:
            message = "Please review your episodic memories and remove any redundant or duplicate entries. Look for similar events, overlapping timeframes, or repeated information. Merge similar memories where appropriate and delete exact duplicates. Focus on maintaining the most informative and comprehensive version of each memory."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.episodic_memory_agent_state.id,
                {'message': message},
                agent_type='episodic_memory',
            )
            redundancy_results['episodic'] = response
        except Exception as e:
            self.logger.error(f"Error calling episodic memory agent: {e}")
            redundancy_results['episodic'] = f"Error: {e}"
        
        # Call semantic memory agent to remove redundancy
        self.logger.info("Calling semantic memory agent to remove redundancy...")
        try:
            message = "Please review your semantic memories and eliminate redundancy. Look for duplicate concepts, overlapping knowledge entries, or repetitive information. Consolidate similar semantic items and remove exact duplicates while preserving the most complete and accurate information."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.semantic_memory_agent_state.id,
                {'message': message},
                agent_type='semantic_memory',
            )
            redundancy_results['semantic'] = response
        except Exception as e:
            self.logger.error(f"Error calling semantic memory agent: {e}")
            redundancy_results['semantic'] = f"Error: {e}"
        
        # Call core memory agent to remove redundancy
        self.logger.info("Calling core memory agent to remove redundancy...")
        try:
            message = "Please review the core memory blocks and remove any redundant or overlapping content. Look for duplicate information across different blocks, consolidate related content, and ensure each block contains unique and essential information without unnecessary repetition."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.core_memory_agent_state.id,
                {'message': message},
                agent_type='core_memory',
            )
            redundancy_results['core'] = response
        except Exception as e:
            self.logger.error(f"Error calling core memory agent: {e}")
            redundancy_results['core'] = f"Error: {e}"
        
        # Call resource memory agent to remove redundancy
        self.logger.info("Calling resource memory agent to remove redundancy...")
        try:
            message = "Please review your resource memories and remove redundant entries. Look for duplicate files, similar documents, or repeated resource information. Consolidate similar resources and remove exact duplicates while maintaining the most useful and comprehensive versions."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.resource_memory_agent_state.id,
                {'message': message},
                agent_type='resource_memory',
            )
            redundancy_results['resource'] = response
        except Exception as e:
            self.logger.error(f"Error calling resource memory agent: {e}")
            redundancy_results['resource'] = f"Error: {e}"
        
        # Call procedural memory agent to remove redundancy
        self.logger.info("Calling procedural memory agent to remove redundancy...")
        try:
            message = "Please review your procedural memories and eliminate redundancy. Look for duplicate procedures, overlapping step sequences, or repetitive process information. Merge similar procedures and remove exact duplicates while preserving the most accurate and complete procedural knowledge."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.procedural_memory_agent_state.id,
                {'message': message},
                agent_type='procedural_memory',
            )
            redundancy_results['procedural'] = response
        except Exception as e:
            self.logger.error(f"Error calling procedural memory agent: {e}")
            redundancy_results['procedural'] = f"Error: {e}"
        
        # Call knowledge vault agent to remove redundancy
        self.logger.info("Calling knowledge vault agent to remove redundancy...")
        try:
            message = "Please review your knowledge vault entries and remove redundant information. Look for duplicate credentials, repeated sensitive information, or overlapping security-related data. Consolidate similar entries and remove exact duplicates while maintaining security and completeness."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.knowledge_vault_agent_state.id,
                {'message': message},
                agent_type='knowledge_vault',
            )
            redundancy_results['knowledge_vault'] = response
        except Exception as e:
            self.logger.error(f"Error calling knowledge vault agent: {e}")
            redundancy_results['knowledge_vault'] = f"Error: {e}"
        
        return redundancy_results

    def _call_reflexion_agent_for_conflicts(self):
        """Call reflexion agent to identify and resolve conflicts between memories"""
        self.logger.info("Calling reflexion agent to resolve memory conflicts...")
        
        try:
            message = """Please analyze all memories across different memory types (episodic, semantic, core, resource, procedural, knowledge vault) and identify potential conflicts:

1. IDENTIFY CONFLICTS:
   - Look for contradictory information between different memory types
   - Find temporal inconsistencies in episodic memories
   - Detect conflicting facts or procedures
   - Identify outdated information that conflicts with newer data

2. RESOLVE CONFLICTS:
   - Update outdated information with more recent, accurate data
   - Flag unresolvable conflicts for human review
   - Merge conflicting memories where appropriate
   - Create clarifying notes for ambiguous situations

3. REPORT ACTIONS:
   - Provide a detailed summary of conflicts found
   - List all resolutions and updates made
   - Highlight any conflicts requiring human intervention

Please perform these actions and provide a comprehensive report of all conflict resolutions made."""
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.reflexion_agent_state.id,
                {'message': message},
                agent_type='reflexion',
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling reflexion agent for conflicts: {e}")
            return f"Error: {e}"

    def _call_agents_for_pattern_analysis(self):
        """Call agents to identify patterns and create new insights"""
        pattern_results = {}
        
        # Call reflexion agent for overall pattern analysis
        self.logger.info("Calling reflexion agent for pattern analysis...")
        try:
            message = """Please analyze patterns across all memory types and generate new insights:

1. IDENTIFY PATTERNS:
   - Find recurring themes across episodic and semantic memories
   - Detect temporal patterns in memory formation
   - Identify relationship patterns between different memories
   - Discover usage patterns in procedural and resource memories

2. GENERATE INSIGHTS:
   - Create new semantic connections based on identified patterns
   - Generate meta-knowledge about user preferences and behaviors
   - Identify gaps in knowledge that should be filled
   - Suggest optimizations for memory organization

3. CREATE NEW MEMORIES:
   - Add new semantic memories based on discovered patterns
   - Update core memory with new insights about the user
   - Create procedural memories for frequently repeated patterns
   - Generate summary memories for related episodic events

Please perform this analysis and create new memories as appropriate. Provide a detailed report of patterns found and new memories created."""
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.reflexion_agent_state.id,
                {'message': message},
                agent_type='reflexion',
            )
            pattern_results['reflexion_patterns'] = response
            
        except Exception as e:
            self.logger.error(f"Error calling reflexion agent for patterns: {e}")
            pattern_results['reflexion_patterns'] = f"Error: {e}"
        
        # Call semantic memory agent for new connections
        self.logger.info("Calling semantic memory agent for new connections...")
        try:
            message = "Based on recent episodic memories and existing semantic knowledge, please identify new semantic connections and create new semantic memories that capture emerging patterns, relationships, or insights. Look for connections between concepts that weren't previously linked."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.semantic_memory_agent_state.id,
                {'message': message},
                agent_type='semantic_memory',
            )
            pattern_results['semantic_connections'] = response
            
        except Exception as e:
            self.logger.error(f"Error calling semantic memory agent for patterns: {e}")
            pattern_results['semantic_connections'] = f"Error: {e}"
        
        # Call meta memory agent for high-level insights
        self.logger.info("Calling meta memory agent for high-level insights...")
        try:
            message = "Please analyze the overall memory system and generate meta-insights about memory usage patterns, knowledge gaps, and opportunities for memory optimization. Create new meta-memories that capture these high-level observations about the memory system itself."
            
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.meta_memory_agent_state.id,
                {'message': message},
                agent_type='meta_memory',
            )
            pattern_results['meta_insights'] = response
            
        except Exception as e:
            self.logger.error(f"Error calling meta memory agent: {e}")
            pattern_results['meta_insights'] = f"Error: {e}"
        
        return pattern_results

    def _remove_memory_redundancy(self):
        """Remove redundancy in each memory type"""
        redundancy_results = {}
        
        try:
            # Analyze episodic memory redundancy
            episodic_memories = self.client.server.episodic_memory_manager.list_episodic_memory(
                self.agent_states.episodic_memory_agent_state, limit=None
            )
            redundancy_results['episodic'] = self._analyze_redundancy(episodic_memories, 'episodic')
            
            # Analyze semantic memory redundancy
            semantic_memories = self.client.server.semantic_memory_manager.list_semantic_items(
                self.agent_states.semantic_memory_agent_state, limit=None
            )
            redundancy_results['semantic'] = self._analyze_redundancy(semantic_memories, 'semantic')
            
            # Analyze core memory redundancy
            core_blocks = self.client.server.block_manager.get_blocks(self.client.user)
            redundancy_results['core'] = self._analyze_core_redundancy(core_blocks)
            
            # Analyze resource memory redundancy
            resource_memories = self.client.server.resource_memory_manager.list_resources(
                self.agent_states.resource_memory_agent_state, limit=None
            )
            redundancy_results['resource'] = self._analyze_redundancy(resource_memories, 'resource')
            
            # Analyze procedural memory redundancy
            procedural_memories = self.client.server.procedural_memory_manager.list_procedures(
                self.agent_states.procedural_memory_agent_state, limit=None
            )
            redundancy_results['procedural'] = self._analyze_redundancy(procedural_memories, 'procedural')
            
            # Analyze knowledge vault redundancy
            knowledge_vault_items = self.client.server.knowledge_vault_manager.list_knowledge(
                self.agent_states.knowledge_vault_agent_state, limit=None
            )
            redundancy_results['knowledge_vault'] = self._analyze_redundancy(knowledge_vault_items, 'knowledge_vault')
            
        except Exception as e:
            self.logger.error(f"Error analyzing redundancy: {e}")
            redundancy_results['error'] = str(e)
        
        return redundancy_results

    def _analyze_redundancy(self, memories, memory_type):
        """Analyze redundancy within a specific memory type"""
        if not memories:
            return f"No {memory_type} memories found."
        
        redundancy_info = {
            'total_count': len(memories),
            'potential_duplicates': [],
            'similar_items': [],
            'recommendations': []
        }
        
        # Compare memories for similarity
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories[i+1:], i+1):
                similarity_score = self._calculate_similarity(memory1, memory2, memory_type)
                
                if similarity_score > 0.9:  # Very high similarity - potential duplicate
                    redundancy_info['potential_duplicates'].append({
                        'memory1_id': getattr(memory1, 'id', i),
                        'memory2_id': getattr(memory2, 'id', j),
                        'similarity': similarity_score,
                        'content1': self._get_memory_content(memory1, memory_type),
                        'content2': self._get_memory_content(memory2, memory_type)
                    })
                elif similarity_score > 0.7:  # High similarity - could be merged
                    redundancy_info['similar_items'].append({
                        'memory1_id': getattr(memory1, 'id', i),
                        'memory2_id': getattr(memory2, 'id', j),
                        'similarity': similarity_score,
                        'content1': self._get_memory_content(memory1, memory_type),
                        'content2': self._get_memory_content(memory2, memory_type)
                    })
        
        # Generate recommendations
        if redundancy_info['potential_duplicates']:
            redundancy_info['recommendations'].append(f"Found {len(redundancy_info['potential_duplicates'])} potential duplicates that should be reviewed for removal.")
        
        if redundancy_info['similar_items']:
            redundancy_info['recommendations'].append(f"Found {len(redundancy_info['similar_items'])} similar items that could be merged or consolidated.")
        
        return redundancy_info

    def _analyze_core_redundancy(self, core_blocks):
        """Analyze redundancy in core memory blocks"""
        if not core_blocks:
            return "No core memory blocks found."
        
        redundancy_info = {
            'total_blocks': len(core_blocks),
            'overlapping_content': [],
            'recommendations': []
        }
        
        # Check for overlapping content between core blocks
        for i, block1 in enumerate(core_blocks):
            for j, block2 in enumerate(core_blocks[i+1:], i+1):
                if self._check_core_overlap(block1, block2):
                    redundancy_info['overlapping_content'].append({
                        'block1_label': block1.label,
                        'block2_label': block2.label,
                        'overlap_description': f"Potential content overlap between {block1.label} and {block2.label}"
                    })
        
        if redundancy_info['overlapping_content']:
            redundancy_info['recommendations'].append("Review overlapping core memory blocks for consolidation.")
        
        return redundancy_info

    def _identify_memory_conflicts(self):
        """Identify potential conflicts between memories across all types"""
        conflict_results = {
            'cross_type_conflicts': [],
            'temporal_conflicts': [],
            'content_conflicts': [],
            'recommendations': []
        }
        
        try:
            # Get all memory types
            all_memories = {
                'episodic': self.client.server.episodic_memory_manager.list_episodic_memory(
                    self.agent_states.episodic_memory_agent_state, limit=None
                ) or [],
                'semantic': self.client.server.semantic_memory_manager.list_semantic_items(
                    self.agent_states.semantic_memory_agent_state, limit=None
                ) or [],
                'procedural': self.client.server.procedural_memory_manager.list_procedures(
                    self.agent_states.procedural_memory_agent_state, limit=None
                ) or [],
                'resource': self.client.server.resource_memory_manager.list_resources(
                    self.agent_states.resource_memory_agent_state, limit=None
                ) or [],
                'knowledge_vault': self.client.server.knowledge_vault_manager.list_knowledge(
                    self.agent_states.knowledge_vault_agent_state, limit=None
                ) or []
            }
            
            # Check for conflicts between different memory types
            conflict_results['cross_type_conflicts'] = self._find_cross_type_conflicts(all_memories)
            
            # Check for temporal conflicts in episodic memories
            conflict_results['temporal_conflicts'] = self._find_temporal_conflicts(all_memories['episodic'])
            
            # Check for content conflicts across all memories
            conflict_results['content_conflicts'] = self._find_content_conflicts(all_memories)
            
            # Generate recommendations
            total_conflicts = (len(conflict_results['cross_type_conflicts']) + 
                             len(conflict_results['temporal_conflicts']) + 
                             len(conflict_results['content_conflicts']))
            
            if total_conflicts > 0:
                conflict_results['recommendations'].append(f"Found {total_conflicts} potential conflicts that need resolution.")
            else:
                conflict_results['recommendations'].append("No significant conflicts detected between memories.")
                
        except Exception as e:
            self.logger.error(f"Error identifying conflicts: {e}")
            conflict_results['error'] = str(e)
        
        return conflict_results

    def _analyze_memory_patterns(self):
        """Identify patterns and analyze new memories"""
        pattern_results = {
            'recurring_themes': [],
            'temporal_patterns': [],
            'relationship_patterns': [],
            'new_insights': [],
            'recommendations': []
        }
        
        try:
            # Get all memories for pattern analysis
            all_memories = self._get_all_memories_for_analysis()
            
            # Identify recurring themes
            pattern_results['recurring_themes'] = self._identify_recurring_themes(all_memories)
            
            # Identify temporal patterns
            pattern_results['temporal_patterns'] = self._identify_temporal_patterns(all_memories)
            
            # Identify relationship patterns
            pattern_results['relationship_patterns'] = self._identify_relationship_patterns(all_memories)
            
            # Generate new insights based on patterns
            pattern_results['new_insights'] = self._generate_insights_from_patterns(pattern_results)
            
            # Generate recommendations
            pattern_results['recommendations'] = self._generate_pattern_recommendations(pattern_results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            pattern_results['error'] = str(e)
        
        return pattern_results

    def _calculate_similarity(self, memory1, memory2, memory_type):
        """Calculate similarity between two memories (simplified implementation)"""
        try:
            content1 = self._get_memory_content(memory1, memory_type).lower()
            content2 = self._get_memory_content(memory2, memory_type).lower()
            
            # Simple similarity based on common words (in real implementation, would use embeddings)
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            if not words1 and not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    def _get_memory_content(self, memory, memory_type):
        """Extract relevant content from memory based on type"""
        try:
            if memory_type == 'episodic':
                return f"{getattr(memory, 'summary', '')} {getattr(memory, 'details', '')}"
            elif memory_type == 'semantic':
                return f"{getattr(memory, 'name', '')} {getattr(memory, 'summary', '')} {getattr(memory, 'details', '')}"
            elif memory_type == 'procedural':
                return f"{getattr(memory, 'summary', '')} {' '.join(getattr(memory, 'steps', []))}"
            elif memory_type == 'resource':
                return f"{getattr(memory, 'title', '')} {getattr(memory, 'summary', '')} {getattr(memory, 'content', '')}"
            elif memory_type == 'knowledge_vault':
                return f"{getattr(memory, 'caption', '')} {getattr(memory, 'secret_value', '')}"
            else:
                return str(memory)
        except Exception:
            return ""

    def _check_core_overlap(self, block1, block2):
        """Check if two core memory blocks have overlapping content"""
        try:
            content1 = getattr(block1, 'value', '').lower()
            content2 = getattr(block2, 'value', '').lower()
            
            if not content1 or not content2:
                return False
            
            # Simple overlap check (could be enhanced with more sophisticated analysis)
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            overlap_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2))
            return overlap_ratio > 0.3  # 30% overlap threshold
            
        except Exception:
            return False

    def _find_cross_type_conflicts(self, all_memories):
        """Find conflicts between different memory types"""
        conflicts = []
        # Implementation would compare memories across types for conflicting information
        # This is a placeholder for the complex logic
        return conflicts

    def _find_temporal_conflicts(self, episodic_memories):
        """Find temporal conflicts in episodic memories"""
        conflicts = []
        # Implementation would check for chronologically impossible events
        return conflicts

    def _find_content_conflicts(self, all_memories):
        """Find content conflicts across all memories"""
        conflicts = []
        # Implementation would identify contradictory information
        return conflicts

    def _get_all_memories_for_analysis(self):
        """Get all memories organized for pattern analysis"""
        return {
            'episodic': self.client.server.episodic_memory_manager.list_episodic_memory(
                self.agent_states.episodic_memory_agent_state, limit=None
            ) or [],
            'semantic': self.client.server.semantic_memory_manager.list_semantic_items(
                self.agent_states.semantic_memory_agent_state, limit=None
            ) or [],
            'procedural': self.client.server.procedural_memory_manager.list_procedures(
                self.agent_states.procedural_memory_agent_state, limit=None
            ) or [],
            'resource': self.client.server.resource_memory_manager.list_resources(
                self.agent_states.resource_memory_agent_state, limit=None
            ) or [],
            'knowledge_vault': self.client.server.knowledge_vault_manager.list_knowledge(
                self.agent_states.knowledge_vault_agent_state, limit=None
            ) or []
        }

    def _identify_recurring_themes(self, all_memories):
        """Identify recurring themes across memories"""
        themes = []
        # Implementation would analyze content for recurring topics
        return themes

    def _identify_temporal_patterns(self, all_memories):
        """Identify temporal patterns in memories"""
        patterns = []
        # Implementation would analyze time-based patterns
        return patterns

    def _identify_relationship_patterns(self, all_memories):
        """Identify relationship patterns between memories"""
        patterns = []
        # Implementation would find connections between memories
        return patterns

    def _generate_insights_from_patterns(self, pattern_results):
        """Generate new insights based on identified patterns"""
        insights = []
        # Implementation would create new knowledge from patterns
        return insights

    def _generate_pattern_recommendations(self, pattern_results):
        """Generate recommendations based on pattern analysis"""
        recommendations = []
        # Implementation would suggest actions based on patterns
        return recommendations

    def _compile_reflexion_summary(self, redundancy_results, conflict_results, pattern_results):
        """Compile a comprehensive summary of the reflexion analysis"""
        summary = {
            'redundancy_summary': redundancy_results,
            'conflict_summary': conflict_results,
            'pattern_summary': pattern_results,
            'overall_recommendations': []
        }
        
        # Add overall recommendations
        if any('error' not in results for results in [redundancy_results, conflict_results, pattern_results]):
            summary['overall_recommendations'].append("Complete comprehensive memory optimization based on analysis.")
        
        return summary

    def send_message(self, 
                      message=None, 
                      images=None, 
                      image_uris=None, 
                      sources=None,
                      voice_files=None,
                      memorizing=False, 
                      delete_after_upload=True, 
                      specific_timestamps=None,
                      display_intermediate_message=None,
                      force_absorb_content=False,
                      async_upload=True):

        # Check if Gemini features are required but not available
        if self.model_name in GEMINI_MODELS and not self.is_gemini_client_initialized():
            if images is not None or image_uris is not None or voice_files is not None:
                self.logger.warning("Warning: Gemini API key not configured. Image and voice features are unavailable.")
                self.logger.warning("Please provide a Gemini API key through the frontend to enable these features.")
                # For now, proceed with text-only message if available
                if message is None:
                    return "Error: Gemini API key required for image/voice features. Please configure it in the settings."
        
        if memorizing:
            
            # Validate that at least some content is provided for memorization
            if message is None and images is None and image_uris is None and voice_files is None:
                self.logger.warning("Warning: memorizing=True but no content (message, images, image_uris, or voice_files) provided. Skipping memorization.")
                return None
            
            # Get timestamp for this memorization event
            if specific_timestamps is not None and len(specific_timestamps) > 0:
                timestamp = specific_timestamps[0]
            else:
                timestamp = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')

            # Process images
            if image_uris is None and images is not None:
                image_uris = []
                for image in images:
                    filename = f'./tmp/image_{uuid.uuid4()}.png'
                    image.save(filename)
                    image_uris.append(filename)

            self.temp_message_accumulator.add_message(
                {
                    'message': message,
                    'image_uris': image_uris,
                    'sources': sources,
                    'voice_files': voice_files,
                },
                timestamp,
                delete_after_upload=delete_after_upload,
                async_upload=async_upload
            )
            
            # Check if we should trigger memory absorption
            ready_messages = self.temp_message_accumulator.should_absorb_content()
            if force_absorb_content or ready_messages:
                t1 = time.time()
                # Pass the ready messages to absorb_content_into_memory if available
                if ready_messages:
                    self.temp_message_accumulator.absorb_content_into_memory(self.agent_states, ready_messages)
                else:
                    # Force absorb with whatever is available
                    self.temp_message_accumulator.absorb_content_into_memory(self.agent_states)
                t2 = time.time()
                self.logger.info(f"Time taken to absorb content into memory: {t2 - t1} seconds")
                self.clear_old_screenshots()

        else:

            if image_uris is not None:
                if isinstance(message, str):
                    message = [{'type': 'text', 'text': message}]
                for image_uri in image_uris:
                    mime_type = get_image_mime_type(image_uri)
                    message.append({'type': 'image_data', 'image_data': {'data': f"data:{mime_type};base64,{encode_image(image_uri)}", 'detail': 'auto'}})

            # Only get recent images for chat context if user has enabled this feature
            if self.include_recent_screenshots:

                if isinstance(message, str):
                    message = [{'type': 'text', 'text': message}]

                extra_messages = []

                most_recent_images = self.temp_message_accumulator.get_recent_images_for_chat(current_timestamp=datetime.now(self.timezone))

                if len(most_recent_images) > 0:

                    extra_messages.append({
                        'type': 'text',
                        'text': f"Additional images (screenshots) from the system start here:"
                    })

                    for idx, (timestamp, file_ref, source) in enumerate(most_recent_images):
                        
                        if hasattr(file_ref, 'uri'):
                            source_text = f"; Screenshot from App: {source}" if source else ""
                            extra_messages.append({
                                'type': 'text',
                                'text': f"Timestamp: {timestamp}; Image Index {idx}" + source_text
                            })
                            extra_messages.append({
                                'type': 'google_cloud_file_uri',
                                'google_cloud_file_uri': file_ref.uri
                            })
                        else:
                            # For non-GEMINI models, convert local file paths to base64
                            try:
                                source_text = f"; Screenshot from App: {source_text}" if source else ""
                                extra_messages.append({
                                    'type': 'text',
                                    'text': f"Timestamp: {timestamp}; Image Index {idx}" + source_text
                                })
                                mime_type = get_image_mime_type(file_ref)
                                base64_data = encode_image(file_ref)
                                extra_messages.append({
                                    'type': 'image_data',
                                    'image_data': {
                                        'data': f"data:{mime_type};base64,{base64_data}",
                                        'detail': 'auto'
                                    }
                                })
                            except Exception as e:
                                self.logger.error(f"Failed to encode image for chat context: {file_ref}, error: {e}")
                                # Skip this image if encoding fails
                                continue
                    
                    extra_messages.append({
                        'type': 'text',
                        'text': f"Additional images (screenshots) from the system end here."
                    })

                extra_messages = None if len(extra_messages) == 0 else extra_messages

            else:
                extra_messages = None

            # get the response according to the message
            response, _ = self.message_queue.send_message_in_queue(
                self.client,
                self.agent_states.agent_state.id,
                {
                    'message': message,
                    'display_intermediate_message': display_intermediate_message,
                    'force_response': True,
                    'existing_file_uris': set(list(self.uri_to_create_time.keys())),
                    'extra_messages': extra_messages,
                }, 
                agent_type='chat',
            )

            # Check if response is an error string
            if response == "ERROR":
                return "ERROR_RESPONSE_FAILED"
            
            # Check if response has the expected structure
            if not hasattr(response, 'messages') or len(response.messages) < 2:
                return "ERROR_INVALID_RESPONSE_STRUCTURE"
            
            try:

                # find how many tools are called
                num_tools_called = 0
                for message in response.messages[::-1]:
                    if message.message_type == MessageType.tool_return_message:
                        num_tools_called += 1
                    else:
                        break

                # Check if the message has tool_call attribute
                # 1->3; 2->5
                if not hasattr(response.messages[-(num_tools_called * 2 + 1)], 'tool_call'):
                    return "ERROR_NO_TOOL_CALL"
                
                tool_call = response.messages[-(num_tools_called * 2 + 1)].tool_call
                
                parsed_args = parse_json(tool_call.arguments)
                
                if 'message' not in parsed_args:
                    return "ERROR_NO_MESSAGE_IN_ARGS"
                    
                response_text = parsed_args['message']
                
            except (AttributeError, KeyError, IndexError, json.JSONDecodeError) as e:
                return "ERROR_PARSING_EXCEPTION"
            
            # Add conversation to accumulator
            self.temp_message_accumulator.add_user_conversation(message, response_text)
            
            return response_text

    def cleanup_upload_workers(self):
        """Delegate to UploadManager for cleanup."""
        if hasattr(self, 'upload_manager') and self.upload_manager is not None:
            self.upload_manager.cleanup_upload_workers()

    def is_gemini_client_initialized(self) -> bool:
        """Check if the Gemini client is properly initialized."""
        return self.google_client is not None
            
    def check_api_key_status(self) -> dict:
        """
        Check the status of required API keys for the frontend.
        Returns a dictionary with information about missing and available API keys.
        """
        status = {
            'missing_keys': [],
            'available_keys': [],
            'model_requirements': {}
        }
        
        # Check what keys are needed for current model
        if self.model_name in GEMINI_MODELS:
            status['model_requirements']['current_model'] = self.model_name
            status['model_requirements']['required_keys'] = ['GEMINI_API_KEY']
            
            # Check database first for API key, then model_settings
            gemini_override_key = self.client.server.provider_manager.get_gemini_override_key()
            has_gemini_key = gemini_override_key or model_settings.gemini_api_key
            
            if not self.is_gemini_client_initialized() or not has_gemini_key:
                if 'GEMINI_API_KEY' not in status['missing_keys']:
                    status['missing_keys'].append('GEMINI_API_KEY')
            else:
                status['available_keys'].append('GEMINI_API_KEY')
                
        elif self.model_name in OPENAI_MODELS:
            status['model_requirements']['current_model'] = self.model_name
            status['model_requirements']['required_keys'] = ['OPENAI_API_KEY']
            
            # Check database first for API key, then model_settings
            openai_override_key = self.client.server.provider_manager.get_openai_override_key()
            has_openai_key = openai_override_key or model_settings.openai_api_key
            
            if not has_openai_key:
                if 'OPENAI_API_KEY' not in status['missing_keys']:
                    status['missing_keys'].append('OPENAI_API_KEY')
            else:
                status['available_keys'].append('OPENAI_API_KEY')
                
        elif 'claude' in self.model_name.lower():
            status['model_requirements']['current_model'] = self.model_name
            status['model_requirements']['required_keys'] = ['ANTHROPIC_API_KEY']
            
            claude_override_key = self.client.server.provider_manager.get_anthropic_override_key()
            has_claude_key = claude_override_key or model_settings.anthropic_api_key

            if not has_claude_key:
                if 'ANTHROPIC_API_KEY' not in status['missing_keys']:
                    status['missing_keys'].append('ANTHROPIC_API_KEY')
            else:
                status['available_keys'].append('ANTHROPIC_API_KEY')
        
        # Update the internal missing_api_keys list to match what we found
        self.missing_api_keys = status['missing_keys'].copy()
            
        return status
    
    def provide_api_key(self, key_name: str, api_key: str) -> dict:
        """
        Provide an API key for a specific service.
        Saves the key to database for persistence across sessions.
        Returns a dictionary with success status and any error messages.
        """

        result = {'success': False, 'message': ''}
        
        # Handle specific initialization for different services
        if key_name == 'GEMINI_API_KEY':
            # Save to database using provider_manager
            try:
                # Create or update the Google AI provider in the database
                self.client.server.provider_manager.upsert_provider(
                    name="google_ai",
                    api_key=api_key,
                    organization_id=self.client.user.organization_id,
                    actor=self.client.user
                )
                result['success'] = True
                result['message'] = 'Gemini API key successfully saved to database!'
            except Exception as e:
                result['message'] = f'Failed to save Gemini API key to database: {str(e)}'
                return result
                
            if self.model_name not in GEMINI_MODELS:
                result['message'] = f"Gemini API key saved but not needed for current model: {self.model_name}"
                return result
                
            # Try to initialize Gemini client with the provided key
            try:
                self.google_client = genai.Client(api_key=api_key)
                
                # Complete the initialization
                success = self._complete_gemini_initialization()
                if success:
                    # Remove from missing keys list
                    if 'GEMINI_API_KEY' in self.missing_api_keys:
                        self.missing_api_keys.remove('GEMINI_API_KEY')
                    
                    result['message'] = 'Gemini API key successfully saved to database and Gemini client initialized!'
                else:
                    result['message'] = 'Gemini API key saved to database but failed to complete initialization'
                    
            except Exception as e:
                result['message'] = f'Gemini API key saved to database but validation failed: {str(e)}'
                self.google_client = None
                
        elif key_name == 'OPENAI_API_KEY':
            # Save to database using provider_manager
            try:
                # Create or update the OpenAI provider in the database
                self.client.server.provider_manager.upsert_provider(
                    name="openai",
                    api_key=api_key,
                    organization_id=self.client.user.organization_id,
                    actor=self.client.user
                )
                result['success'] = True
                result['message'] = 'OpenAI API key successfully saved to database!'
            except Exception as e:
                result['message'] = f'Failed to save OpenAI API key to database: {str(e)}'
                return result

            # Remove from missing keys list
            if 'OPENAI_API_KEY' in self.missing_api_keys:
                self.missing_api_keys.remove('OPENAI_API_KEY')
            
        elif key_name == 'ANTHROPIC_API_KEY':
            # Save to database using provider_manager
            try:
                # Create or update the Anthropic provider in the database
                self.client.server.provider_manager.upsert_provider(
                    name="anthropic",
                    api_key=api_key,
                    organization_id=self.client.user.organization_id,
                    actor=self.client.user
                )
                result['success'] = True
                result['message'] = 'Anthropic API key successfully saved to database!'
            except Exception as e:
                result['message'] = f'Failed to save Anthropic API key to database: {str(e)}'
                return result
            
            # Remove from missing keys list
            if 'ANTHROPIC_API_KEY' in self.missing_api_keys:
                self.missing_api_keys.remove('ANTHROPIC_API_KEY')
            
        else:
            # For any other API key, just confirm it was provided
            result['success'] = True
            result['message'] = f'{key_name} successfully configured!'
            
        return result
    


    def _complete_gemini_initialization(self) -> bool:
        """Complete Gemini initialization after API key is provided."""
        try:
            
            # Get existing files
            self.logger.info("Getting existing files...")
            existing_files = []
            existing_image_names = set([file.name for file in existing_files])
            self.logger.info(f"# of Existing files in Google Clouds: {len(existing_image_names)}")

            # Sync database with cloud files
            for file_name in self.client.server.cloud_file_mapping_manager.list_all_cloud_file_ids():
                if file_name not in existing_image_names:
                    self.client.server.cloud_file_mapping_manager.delete_mapping(cloud_file_id=file_name)

            cloud_file_names_in_database_set = set(self.client.server.cloud_file_mapping_manager.list_all_cloud_file_ids())

            # Filter files that belong to this project
            remaining_indices = []
            for idx, file in enumerate(existing_files):
                if file.name in cloud_file_names_in_database_set:
                    remaining_indices.append(idx)

            existing_files = [existing_files[i] for i in remaining_indices]
            self.existing_files = existing_files
            self.uri_to_create_time = {file.uri: {'create_time': file.create_time, 'filename': file.name} for file in existing_files}

            self.logger.info(f"# of Existing files in Google Clouds that belong to Mirix: {len(self.uri_to_create_time)}")

            # Initialize upload manager
            self.upload_manager = UploadManager(self.google_client, self.client, self.existing_files, self.uri_to_create_time)
            
            # Update temporary message accumulator
            self.temp_message_accumulator.google_client = self.google_client
            self.temp_message_accumulator.upload_manager = self.upload_manager
            self.temp_message_accumulator.uri_to_create_time = self.uri_to_create_time
            
            # Process existing uploaded files
            self._process_existing_uploaded_files()
            
            self.logger.info("Gemini initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete Gemini initialization: {e}")
            return False

    def save_agent(self, folder_path: str) -> dict:
        """
        Save the current agent state to a directory.
        For PostgreSQL: Creates database dumps and saves configuration.
        For SQLite: Copies the database file.
        
        Args:
            folder_path: Directory path where agent state will be saved
            
        Returns:
            Dictionary with success status and message
        """
        import subprocess
        import shutil
        from pathlib import Path
        from mirix.settings import settings
        
        result = {'success': False, 'message': ''}
        
        try:
            # Create directory if it doesn't exist
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            
            # Check if using PostgreSQL or SQLite
            if settings.mirix_pg_uri_no_default:
                # PostgreSQL backup
                self.logger.info(f"Creating PostgreSQL backup for agent in {folder_path}")
                
                # Parse connection details from settings
                if settings.pg_uri:
                    # Parse full URI if available
                    import urllib.parse as urlparse
                    parsed = urlparse.urlparse(settings.pg_uri)
                    db_host = parsed.hostname or 'localhost'
                    db_port = parsed.port or 5432
                    db_user = parsed.username or 'mirix'
                    db_name = parsed.path.lstrip('/') or 'mirix'
                else:
                    # Use individual settings
                    db_host = settings.pg_host or 'localhost'
                    db_port = settings.pg_port or 5432
                    db_user = settings.pg_user or 'mirix'
                    db_name = settings.pg_db or 'mirix'
                
                # Create SQL dump backup
                backup_file = Path(folder_path) / "mirix_database.sql"
                cmd = [
                    'pg_dump',
                    '-h', str(db_host),
                    '-p', str(db_port),
                    '-U', db_user,
                    '-d', db_name,
                    '-f', str(backup_file)
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self.logger.info(f"✅ PostgreSQL backup created: {backup_file}")
                    
                    # Also create a compressed backup for efficiency
                    compressed_backup = Path(folder_path) / "mirix_database.dump"
                    cmd_compressed = [
                        'pg_dump',
                        '-h', str(db_host),
                        '-p', str(db_port),
                        '-U', db_user,
                        '-d', db_name,
                        '-Fc',
                        '-f', str(compressed_backup)
                    ]
                    
                    subprocess.run(cmd_compressed, capture_output=True, text=True, check=True)
                    self.logger.info(f"✅ Compressed PostgreSQL backup created: {compressed_backup}")
                    
                    # Save connection details and agent configuration for restoration
                    agent_config = {
                        'agent_name': self.agent_name,
                        'model_name': self.model_name,
                        'memory_model_name': getattr(self, 'memory_model_name', self.model_name),
                        'timezone_str': getattr(self, 'timezone_str', 'UTC'),
                        'active_persona_name': getattr(self, 'active_persona_name', 'helpful_assistant'),
                        'include_recent_screenshots': getattr(self, 'include_recent_screenshots', True),
                        'is_screen_monitor': getattr(self, 'is_screen_monitor', False),
                        'backup_type': 'postgresql',
                        'backup_timestamp': datetime.now().isoformat(),
                        'connection_info': {
                            'db_host': db_host,
                            'db_port': db_port,
                            'db_user': db_user,
                            'db_name': db_name
                        }
                    }
                    
                    with open(Path(folder_path) / "agent_config.json", "w") as f:
                        json.dump(agent_config, f, indent=2)
                    
                    # Save the agent configuration as YAML
                    config_dest = Path(folder_path) / "mirix_config.yaml"
                    with open(config_dest, "w") as f:
                        yaml.dump(self.agent_config, f, default_flow_style=False, indent=2)
                    self.logger.info(f"✅ Agent configuration saved: {config_dest}")
                    
                    result['success'] = True
                    result['message'] = f'Agent state saved successfully to {folder_path}'
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"PostgreSQL backup failed: {e.stderr if e.stderr else str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    result['message'] = error_msg
                    return result
                    
            else:
                # SQLite backup (original behavior)
                self.logger.info(f"Creating SQLite backup for agent in {folder_path}")
                sqlite_source = Path.home() / ".mirix" / "sqlite.db"
                sqlite_dest = Path(folder_path) / "sqlite.db"
                
                if sqlite_source.exists():
                    shutil.copyfile(sqlite_source, sqlite_dest)
                    self.logger.info(f"✅ SQLite backup created: {sqlite_dest}")
                    
                    # Save agent configuration
                    agent_config = {
                        'agent_name': self.agent_name,
                        'model_name': self.model_name,
                        'memory_model_name': getattr(self, 'memory_model_name', self.model_name),
                        'timezone_str': getattr(self, 'timezone_str', 'UTC'),
                        'active_persona_name': getattr(self, 'active_persona_name', 'helpful_assistant'),
                        'include_recent_screenshots': getattr(self, 'include_recent_screenshots', True),
                        'is_screen_monitor': getattr(self, 'is_screen_monitor', False),
                        'backup_type': 'sqlite',
                        'backup_timestamp': datetime.now().isoformat()
                    }
                    
                    with open(Path(folder_path) / "agent_config.json", "w") as f:
                        json.dump(agent_config, f, indent=2)
                    
                    # Save the agent configuration as YAML
                    config_dest = Path(folder_path) / "mirix_config.yaml"
                    with open(config_dest, "w") as f:
                        yaml.dump(self.agent_config, f, default_flow_style=False, indent=2)
                    self.logger.info(f"✅ Agent configuration saved: {config_dest}")
                    
                    result['success'] = True
                    result['message'] = f'Agent state saved successfully to {folder_path}'
                else:
                    result['message'] = f'SQLite database not found at {sqlite_source}'
                    return result
                    
        except Exception as e:
            error_msg = f"Failed to save agent state: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            result['message'] = error_msg
            
        return result

    def get_database_info(self) -> dict:
        """
        Get information about the current database setup and content.
        
        Returns:
            Dictionary with database information
        """
        from pathlib import Path
        from mirix.settings import settings
        
        info = {
            'database_type': 'postgresql' if settings.mirix_pg_uri_no_default else 'sqlite',
            'tables': {},
            'total_records': 0
        }
        
        try:
            if settings.mirix_pg_uri_no_default:
                # PostgreSQL info
                info['connection_details'] = {
                    'host': settings.pg_host or 'localhost',
                    'port': settings.pg_port or 5432,
                    'user': settings.pg_user or 'mirix',
                    'database': settings.pg_db or 'mirix'
                }
                
                # Get table information using the client
                try:
                    # Use the existing managers to get record counts
                    if hasattr(self.client, 'server'):
                        managers = {
                            'agents': len(self.client.list_agents()),
                            'episodic_memory': len(self.client.server.episodic_memory_manager.list_episodic_memory(self.agent_states.episodic_memory_agent_state, limit=None) or []),
                            'semantic_memory': len(self.client.server.semantic_memory_manager.list_semantic_items(self.agent_states.semantic_memory_agent_state, limit=None) or []),
                            'procedural_memory': len(self.client.server.procedural_memory_manager.list_procedures(self.agent_states.procedural_memory_agent_state, limit=None) or []),
                            'knowledge_vault': len(self.client.server.knowledge_vault_manager.list_knowledge(self.agent_states.knowledge_vault_agent_state, limit=None) or []),
                            'resource_memory': len(self.client.server.resource_memory_manager.list_resources(self.agent_states.resource_memory_agent_state, limit=None) or []),
                        }
                        
                        info['tables'] = managers
                        info['total_records'] = sum(managers.values())
                        
                except Exception as e:
                    info['error'] = f"Could not get detailed table information: {str(e)}"
                    
            else:
                # SQLite info
                sqlite_path = Path.home() / ".mirix" / "sqlite.db"
                info['database_path'] = str(sqlite_path)
                info['database_exists'] = sqlite_path.exists()
                
                if sqlite_path.exists():
                    info['database_size'] = sqlite_path.stat().st_size
                    
                    # Get table information using the client
                    try:
                        if hasattr(self.client, 'server'):
                            managers = {
                                'agents': len(self.client.list_agents()),
                                'episodic_memory': len(self.client.server.episodic_memory_manager.list_episodic_memory(self.agent_states.episodic_memory_agent_state, limit=None) or []),
                                'semantic_memory': len(self.client.server.semantic_memory_manager.list_semantic_items(self.agent_states.semantic_memory_agent_state, limit=None) or []),
                                'procedural_memory': len(self.client.server.procedural_memory_manager.list_procedures(self.agent_states.procedural_memory_agent_state, limit=None) or []),
                                'knowledge_vault': len(self.client.server.knowledge_vault_manager.list_knowledge(self.agent_states.knowledge_vault_agent_state, limit=None) or []),
                                'resource_memory': len(self.client.server.resource_memory_manager.list_resources(self.agent_states.resource_memory_agent_state, limit=None) or []),
                            }
                            
                            info['tables'] = managers
                            info['total_records'] = sum(managers.values())
                            
                    except Exception as e:
                        info['error'] = f"Could not get detailed table information: {str(e)}"
                        
        except Exception as e:
            info['error'] = f"Failed to get database info: {str(e)}"
            
        return info

    def export_memories_to_csv(self, csv_file_path: str, include_embeddings: bool = False) -> dict:
        """
        Export all memories from all memory types to a CSV file.
        
        Args:
            csv_file_path: Path where the CSV file will be saved
            include_embeddings: Whether to include embedding vectors in the CSV (default: False)
            
        Returns:
            Dictionary with export status and statistics
        """
        import pandas as pd
        from pathlib import Path
        import json
        
        result = {
            'success': False,
            'message': '',
            'exported_counts': {},
            'total_exported': 0,
            'file_path': csv_file_path
        }
        
        try:
            # Ensure the output directory exists
            Path(csv_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            all_memories = []
            memory_counts = {}
            
            # 1. Export Episodic Memory
            try:
                episodic_memories = self.client.server.episodic_memory_manager.list_episodic_memory(
                    agent_state=self.agent_states.episodic_memory_agent_state,
                    limit=None
                )
                memory_counts['episodic'] = len(episodic_memories)
                
                for memory in episodic_memories:
                    row = {
                        'memory_type': 'episodic',
                        'id': memory.id,
                        'created_at': memory.created_at,
                        'occurred_at': getattr(memory, 'occurred_at', None),
                        'event_type': getattr(memory, 'event_type', None),
                        'actor': getattr(memory, 'actor', None),
                        'summary': getattr(memory, 'summary', None),
                        'details': getattr(memory, 'details', None),
                        'organization_id': memory.organization_id,
                        'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                        'metadata': json.dumps(memory.metadata_),
                        'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                    }
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                        row['details_embedding'] = json.dumps(getattr(memory, 'details_embedding', None))
                    
                    all_memories.append(row)
                    
                self.logger.info(f"Exported {len(episodic_memories)} episodic memories")
                
            except Exception as e:
                self.logger.error(f"Error exporting episodic memories: {e}")
                memory_counts['episodic'] = 0
            
            # 2. Export Semantic Memory
            try:
                semantic_memories = self.client.server.semantic_memory_manager.list_semantic_items(
                    agent_state=self.agent_states.semantic_memory_agent_state,
                    limit=None
                )
                memory_counts['semantic'] = len(semantic_memories)
                
                for memory in semantic_memories:
                    row = {
                        'memory_type': 'semantic',
                        'id': memory.id,
                        'created_at': memory.created_at,
                        'occurred_at': None,  # Semantic memories don't have occurred_at
                        'event_type': None,
                        'actor': None,
                        'summary': getattr(memory, 'summary', None),
                        'details': getattr(memory, 'details', None),
                        'name': getattr(memory, 'name', None),
                        'source': getattr(memory, 'source', None),
                        'organization_id': memory.organization_id,
                        'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                        'metadata': json.dumps(memory.metadata_),
                        'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                    }
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        row['name_embedding'] = json.dumps(getattr(memory, 'name_embedding', None))
                        row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                        row['details_embedding'] = json.dumps(getattr(memory, 'details_embedding', None))
                    
                    all_memories.append(row)
                    
                self.logger.info(f"Exported {len(semantic_memories)} semantic memories")
                
            except Exception as e:
                self.logger.error(f"Error exporting semantic memories: {e}")
                memory_counts['semantic'] = 0
            
            # 3. Export Procedural Memory
            try:
                procedural_memories = self.client.server.procedural_memory_manager.list_procedures(
                    agent_state=self.agent_states.procedural_memory_agent_state,
                    limit=None
                )
                memory_counts['procedural'] = len(procedural_memories)
                
                for memory in procedural_memories:
                    row = {
                        'memory_type': 'procedural',
                        'id': memory.id,
                        'created_at': memory.created_at,
                        'occurred_at': None,
                        'event_type': None,
                        'actor': None,
                        'summary': getattr(memory, 'summary', None),
                        'details': None,
                        'entry_type': getattr(memory, 'entry_type', None),
                        'steps': getattr(memory, 'steps', None),
                        'organization_id': memory.organization_id,
                        'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                        'metadata': json.dumps(memory.metadata_),
                        'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                    }
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                        row['steps_embedding'] = json.dumps(getattr(memory, 'steps_embedding', None))
                    
                    all_memories.append(row)
                    
                self.logger.info(f"Exported {len(procedural_memories)} procedural memories")
                
            except Exception as e:
                self.logger.error(f"Error exporting procedural memories: {e}")
                memory_counts['procedural'] = 0
            
            # 4. Export Resource Memory
            try:
                resource_memories = self.client.server.resource_memory_manager.list_resources(
                    agent_state=self.agent_states.resource_memory_agent_state,
                    limit=None
                )
                memory_counts['resource'] = len(resource_memories)
                
                for memory in resource_memories:
                    row = {
                        'memory_type': 'resource',
                        'id': memory.id,
                        'created_at': memory.created_at,
                        'occurred_at': None,
                        'event_type': None,
                        'actor': None,
                        'summary': getattr(memory, 'summary', None),
                        'details': None,
                        'title': getattr(memory, 'title', None),
                        'content': getattr(memory, 'content', None),
                        'resource_type': getattr(memory, 'resource_type', None),
                        'organization_id': memory.organization_id,
                        'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                        'metadata': json.dumps(memory.metadata_),
                        'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                    }
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                    
                    all_memories.append(row)
                    
                self.logger.info(f"Exported {len(resource_memories)} resource memories")
                
            except Exception as e:
                self.logger.error(f"Error exporting resource memories: {e}")
                memory_counts['resource'] = 0
            
            # 5. Export Knowledge Vault
            try:
                knowledge_vault_items = self.client.server.knowledge_vault_manager.list_knowledge(
                    agent_state=self.agent_states.knowledge_vault_agent_state,
                    limit=None
                )
                memory_counts['knowledge_vault'] = len(knowledge_vault_items)
                
                for memory in knowledge_vault_items:
                    row = {
                        'memory_type': 'knowledge_vault',
                        'id': memory.id,
                        'created_at': memory.created_at,
                        'occurred_at': None,
                        'event_type': None,
                        'actor': None,
                        'summary': None,
                        'details': None,
                        'entry_type': getattr(memory, 'entry_type', None),
                        'source': getattr(memory, 'source', None),
                        'sensitivity': getattr(memory, 'sensitivity', None),
                        'secret_value': getattr(memory, 'secret_value', None),
                        'caption': getattr(memory, 'caption', None),
                        'organization_id': memory.organization_id,
                        'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                        'metadata': json.dumps(memory.metadata_),
                        'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                    }
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        row['caption_embedding'] = json.dumps(getattr(memory, 'caption_embedding', None))
                    
                    all_memories.append(row)
                    
                self.logger.info(f"Exported {len(knowledge_vault_items)} knowledge vault items")
                
            except Exception as e:
                self.logger.error(f"Error exporting knowledge vault items: {e}")
                memory_counts['knowledge_vault'] = 0
            
            # Create DataFrame and export to CSV
            if all_memories:
                df = pd.DataFrame(all_memories)
                
                # Ensure columns are not treated as categorical to avoid sorting issues
                df['memory_type'] = df['memory_type'].astype(str)
                if 'created_at' in df.columns:
                    # Convert created_at to string to avoid timezone/datetime issues
                    df['created_at'] = df['created_at'].astype(str)
                
                # Sort by memory type and creation date
                df = df.sort_values(['memory_type', 'created_at'], ascending=[True, False])
                
                # Export to CSV
                df.to_csv(csv_file_path, index=False, encoding='utf-8')
                
                result['success'] = True
                result['exported_counts'] = memory_counts
                result['total_exported'] = sum(memory_counts.values())
                result['message'] = f'Successfully exported {result["total_exported"]} memories to {csv_file_path}'
                result['columns'] = list(df.columns)
                
                self.logger.info(f"✅ Memory export completed: {result['message']}")
                
            else:
                result['message'] = 'No memories found to export'
                result['exported_counts'] = memory_counts
                result['total_exported'] = 0
                self.logger.warning("⚠️ No memories found to export")
                
        except Exception as e:
            error_msg = f"Failed to export memories to CSV: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            result['message'] = error_msg
            
        return result
            
    def export_memories_to_excel(self, file_path: str, memory_types: list = None, include_embeddings: bool = False) -> dict:
        """
        Export selected memory types to an Excel file with separate sheets for each memory type.
        
        Args:
            file_path: Path where the Excel file will be saved
            memory_types: List of memory types to export. If None, exports all types.
            include_embeddings: Whether to include embedding vectors in the export (default: False)
            
        Returns:
            Dictionary with export status and statistics
        """
        import pandas as pd
        from pathlib import Path
        import json
        
        # Default to all memory types if none specified
        if memory_types is None:
            memory_types = ['episodic', 'semantic', 'procedural', 'resource']
        
        result = {
            'success': False,
            'message': '',
            'exported_counts': {},
            'total_exported': 0,
            'file_path': file_path
        }
        
        try:
            # Ensure the output directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                total_exported = 0
                
                # Export each memory type to its own sheet
                for memory_type in memory_types:
                    try:
                        if memory_type == 'episodic':
                            memories, count = self._export_episodic_memories(include_embeddings)
                        elif memory_type == 'semantic':
                            memories, count = self._export_semantic_memories(include_embeddings)
                        elif memory_type == 'procedural':
                            memories, count = self._export_procedural_memories(include_embeddings)
                        elif memory_type == 'resource':
                            memories, count = self._export_resource_memories(include_embeddings)
                        else:
                            self.logger.warning(f"Unknown memory type: {memory_type}")
                            continue
                        
                        result['exported_counts'][memory_type] = count
                        total_exported += count
                        
                        if memories:
                            df = pd.DataFrame(memories)
                            
                            # Sort by creation date if available
                            if 'created_at' in df.columns:
                                df['created_at'] = df['created_at'].astype(str)
                                df = df.sort_values('created_at', ascending=False)
                            
                            # Write to Excel sheet
                            sheet_name = memory_type.capitalize()
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            self.logger.info(f"Exported {count} {memory_type} memories to '{sheet_name}' sheet")
                        else:
                            # Create empty sheet with headers
                            self.logger.info(f"No {memory_type} memories found, creating empty sheet")
                            
                    except Exception as e:
                        self.logger.error(f"Error exporting {memory_type} memories: {e}")
                        result['exported_counts'][memory_type] = 0
                
                result['total_exported'] = total_exported
                
                if total_exported > 0:
                    result['success'] = True
                    result['message'] = f'Successfully exported {total_exported} memories to {file_path} with {len(memory_types)} sheets'
                else:
                    result['success'] = True  # Still success even if no memories
                    result['message'] = f'No memories found to export, created empty Excel file at {file_path}'
                
                self.logger.info(f"✅ Memory export completed: {result['message']}")
                
        except Exception as e:
            error_msg = f"Failed to export memories to Excel: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            result['message'] = error_msg
            
        return result
    
    def _export_episodic_memories(self, include_embeddings: bool = False) -> tuple:
        """Export episodic memories and return (memories_list, count)"""
        try:
            episodic_memories = self.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=self.agent_states.episodic_memory_agent_state,
                limit=None
            )
            
            memories = []
            for memory in episodic_memories:
                row = {
                    'id': memory.id,
                    'created_at': memory.created_at,
                    'occurred_at': getattr(memory, 'occurred_at', None),
                    'event_type': getattr(memory, 'event_type', None),
                    'actor': getattr(memory, 'actor', None),
                    'summary': getattr(memory, 'summary', None),
                    'details': getattr(memory, 'details', None),
                    'organization_id': memory.organization_id,
                    'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                    'metadata': json.dumps(memory.metadata_),
                    'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                }
                
                if include_embeddings:
                    row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                    row['details_embedding'] = json.dumps(getattr(memory, 'details_embedding', None))
                
                memories.append(row)
            
            return memories, len(episodic_memories)
            
        except Exception as e:
            self.logger.error(f"Error exporting episodic memories: {e}")
            return [], 0
    
    def _export_semantic_memories(self, include_embeddings: bool = False) -> tuple:
        """Export semantic memories and return (memories_list, count)"""
        try:
            semantic_memories = self.client.server.semantic_memory_manager.list_semantic_items(
                agent_state=self.agent_states.semantic_memory_agent_state,
                limit=None
            )
            
            memories = []
            for memory in semantic_memories:
                row = {
                    'id': memory.id,
                    'created_at': memory.created_at,
                    'name': getattr(memory, 'name', None),
                    'summary': getattr(memory, 'summary', None),
                    'details': getattr(memory, 'details', None),
                    'source': getattr(memory, 'source', None),
                    'organization_id': memory.organization_id,
                    'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                    'metadata': json.dumps(memory.metadata_),
                    'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                }
                
                if include_embeddings:
                    row['name_embedding'] = json.dumps(getattr(memory, 'name_embedding', None))
                    row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                    row['details_embedding'] = json.dumps(getattr(memory, 'details_embedding', None))
                
                memories.append(row)
            
            return memories, len(semantic_memories)
            
        except Exception as e:
            self.logger.error(f"Error exporting semantic memories: {e}")
            return [], 0
    
    def _export_procedural_memories(self, include_embeddings: bool = False) -> tuple:
        """Export procedural memories and return (memories_list, count)"""
        try:
            procedural_memories = self.client.server.procedural_memory_manager.list_procedures(
                agent_state=self.agent_states.procedural_memory_agent_state,
                limit=None
            )
            
            memories = []
            for memory in procedural_memories:
                row = {
                    'id': memory.id,
                    'created_at': memory.created_at,
                    'entry_type': getattr(memory, 'entry_type', None),
                    'summary': getattr(memory, 'summary', None),
                    'steps': getattr(memory, 'steps', None),
                    'organization_id': memory.organization_id,
                    'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                    'metadata': json.dumps(memory.metadata_),
                    'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                }
                
                if include_embeddings:
                    row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                    row['steps_embedding'] = json.dumps(getattr(memory, 'steps_embedding', None))
                
                memories.append(row)
            
            return memories, len(procedural_memories)
            
        except Exception as e:
            self.logger.error(f"Error exporting procedural memories: {e}")
            return [], 0
    
    def _export_resource_memories(self, include_embeddings: bool = False) -> tuple:
        """Export resource memories and return (memories_list, count)"""
        try:
            resource_memories = self.client.server.resource_memory_manager.list_resources(
                agent_state=self.agent_states.resource_memory_agent_state,
                limit=None
            )
            
            memories = []
            for memory in resource_memories:
                row = {
                    'id': memory.id,
                    'created_at': memory.created_at,
                    'title': getattr(memory, 'title', None),
                    'summary': getattr(memory, 'summary', None),
                    'content': getattr(memory, 'content', None),
                    'resource_type': getattr(memory, 'resource_type', None),
                    'organization_id': memory.organization_id,
                    'tree_path': json.dumps(getattr(memory, 'tree_path', [])),
                    'metadata': json.dumps(memory.metadata_),
                    'last_modify': json.dumps(getattr(memory, 'last_modify', {}))
                }
                
                if include_embeddings:
                    row['summary_embedding'] = json.dumps(getattr(memory, 'summary_embedding', None))
                
                memories.append(row)
            
            return memories, len(resource_memories)
            
        except Exception as e:
            self.logger.error(f"Error exporting resource memories: {e}")
            return [], 0
            