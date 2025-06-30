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
from mirix.agent.app_constants import TEMPORARY_MESSAGE_LIMIT, MAXIMUM_NUM_IMAGES_IN_CLOUD, GEMINI_MODELS, OPENAI_MODELS

from mirix import create_client
from mirix import LLMConfig, EmbeddingConfig
from mirix.schemas.agent import AgentType
from mirix.prompts import gpt_system
from mirix.schemas.memory import ChatMemory
from mirix.settings import model_settings

class AgentWrapper():
    
    def __init__(self, agent_config_file, load_from=None):

        # If load_from is specified, restore the database first before any agent initialization
        if load_from is not None:
            self._restore_database_before_init(load_from)

        with open(agent_config_file, "r") as f:
            agent_config = yaml.safe_load(f)

        self.agent_name = agent_config['agent_name']
        self.model_name = agent_config['model_name']
        self.is_screen_monitor = agent_config.get('is_screen_monitor', False)

        # Initialize logger early
        self.logger = logging.getLogger(f"Mirix.AgentWrapper.{self.agent_name}")
        self.logger.setLevel(logging.INFO)

        self.client = create_client()
        self.client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini")) 
        self.client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-3-small")) # FOR LOCOMO evaluation
        # self.client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-004"))

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

                if agent_state.name == 'chat_agent':
                    system_prompt = gpt_system.get_system_text(agent_state.name) if not self.is_screen_monitor else gpt_system.get_system_text(agent_state.name + '_screen_monitor')
                else:
                    system_prompt = gpt_system.get_system_text(agent_state.name) if not self.is_screen_monitor else gpt_system.get_system_text(agent_state.name + '_screen_monitor')

                self.client.server.agent_manager.update_agent_tools_and_system_prompts(
                    agent_id=agent_state.id,
                    actor=self.client.user,
                    system_prompt=system_prompt
                )
                
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
                        system=gpt_system.get_system_text(config['name']),
                    )
                else:
                    # All other agents follow the same pattern
                    agent_state = self.client.create_agent(
                        name=config['name'],
                        agent_type=config['agent_type'],
                        memory=core_memory,
                        system=gpt_system.get_system_text(config['name']) if not self.is_screen_monitor else gpt_system.get_system_text(config['name']) + gpt_system.get_system_text(config['name'] + '_screen_monitor'),
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
        
        if not gemini_api_key:
            self.logger.info("Info: GEMINI_API_KEY not found. Gemini features will be available after API key is provided.")
            self.google_client = None
            self.existing_files = []
            self.uri_to_create_time = {}
            self.upload_manager = None
            return False
            
        try:
            self.google_client = genai.Client(api_key=gemini_api_key)
            self.logger.info("Retrieving existing files from Google Clouds...")
            
            try:
                existing_files = [x for x in self.google_client.files.list()]
            except Exception as e:
                self.logger.error(f"Error retrieving existing files from Google Clouds: {e}")
                existing_files = []
            existing_image_names = set([file.name for file in existing_files])
            
            self.logger.info(f"# of Existing files in Google Clouds: {len(existing_image_names)}")

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

    def set_model(self, model_name: str) -> dict:
        """
        Set the model for the agent.
        Returns a dictionary with success status and any missing API keys.
        """
        try:
            self.model_name = model_name
            
            # Create LLM config manually to ensure it picks up updated API keys from model_settings
            if model_name == 'gpt-4o-mini' or model_name == 'gpt-4o':
                llm_config = LLMConfig.default_config(model_name)

            elif model_name in GEMINI_MODELS:
                llm_config = LLMConfig(
                    model_endpoint_type="google_ai",
                    model_endpoint="https://generativelanguage.googleapis.com",
                    model=model_name,
                    context_window=1000000
                )

            elif 'claude' in model_name.lower():
                llm_config = LLMConfig(
                    model_endpoint_type="anthropic",
                    model_endpoint="https://api.anthropic.com/v1",
                    model=model_name,
                    context_window=200000
                )

            elif model_name in OPENAI_MODELS:
                llm_config = LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint="https://api.openai.com/v1",
                    model_wrapper=None,
                    context_window=128000,
                )
            else:
                # Fallback to default_config for unsupported models
                llm_config = LLMConfig.default_config(model_name)
            
            # Update LLM config for the client
            self.client.set_default_llm_config(llm_config)
            self.client.server.agent_manager.update_llm_config(
                agent_id=self.agent_states.agent_state.id,
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

    def set_memory_model(self, new_model):
        """Set the model specifically for memory management operations"""
        
        # Define allowed memory models
        ALLOWED_MEMORY_MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17', 'gemini-2.5-flash']

        # Validate the model
        if new_model not in ALLOWED_MEMORY_MODELS:
            # warnings.warn(f'Invalid memory model. Only {", ".join(ALLOWED_MEMORY_MODELS)} are supported.')
            # self.logger.warning(f'Invalid memory model. Only {", ".join(ALLOWED_MEMORY_MODELS)} are supported.')

            if new_model in OPENAI_MODELS:
                llm_config = LLMConfig(
                    model=new_model,
                    model_endpoint_type="openai",
                    model_endpoint="https://api.openai.com/v1",
                    model_wrapper=None,
                    context_window=128000,
                )
            else:
                raise ValueError(f"Invalid memory model: {new_model}")
        
        else:
            
            # All allowed memory models are Gemini models
            llm_config = LLMConfig(
                model_endpoint_type="google_ai",
                model_endpoint="https://generativelanguage.googleapis.com",
                model=new_model,
                context_window=100000
            )
            
            # Check for API key availability
            if not self.is_gemini_client_initialized():
                return {
                    'success': False,
                    'message': f'Memory model set to {new_model}, but Gemini API key is required.',
                    'missing_keys': ['GEMINI_API_KEY'],
                    'model_requirements': {
                        'current_model': new_model,
                        'required_keys': ['GEMINI_API_KEY']
                    }
                }
        
        # Update only the memory-related agents (all agents except chat_agent)
        memory_agent_names = [
            'episodic_memory_agent',
            'procedural_memory_agent', 
            'knowledge_vault_agent',
            'meta_memory_agent',
            'semantic_memory_agent',
            'core_memory_agent',
            'resource_memory_agent'
        ]
        
        for agent_state in self.client.list_agents():
            if agent_state.name in memory_agent_names:
                self.client.server.agent_manager.update_llm_config(
                    agent_id=agent_state.id, 
                    llm_config=llm_config, 
                    actor=self.client.user
                )

        self.memory_model_name = new_model
        
        return {
            'success': True,
            'message': f'Memory model set to {new_model} successfully.',
            'missing_keys': [],
            'model_requirements': {
                'current_model': new_model,
                'required_keys': ['GEMINI_API_KEY']
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

    def send_message(self, 
                      message=None, 
                      images=None, 
                      image_uris=None, 
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

            # Only get recent images for chat context if user has enabled this feature
            if self.include_recent_screenshots:

                if isinstance(message, str):
                    message = [{'type': 'text', 'text': message}]

                extra_messages = []

                most_recent_images = self.temp_message_accumulator.get_recent_images_for_chat()

                if len(most_recent_images) > 0:

                    extra_messages.append({
                        'type': 'text',
                        'text': f"Additional images (screenshots) from the system start here:"
                    })

                    for idx, (timestamp, file_ref) in enumerate(most_recent_images):
                        
                        if hasattr(file_ref, 'uri'):
                            extra_messages.append({
                                'type': 'text',
                                'text': f"Timestamp: {timestamp} Image Index {idx}:"
                            })
                            extra_messages.append({
                                'type': 'google_cloud_file_uri',
                                'google_cloud_file_uri': file_ref.uri
                            })
                        else:
                            raise NotImplementedError("Local file paths are not supported for chat context")
                    
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
                return "ERROR"
            
            # Check if response has the expected structure
            if not hasattr(response, 'messages') or len(response.messages) < 2:
                return "ERROR"
            
            try:

                # Check if the message has tool_call attribute
                if not hasattr(response.messages[-3], 'tool_call'):
                    return "ERROR"
                
                tool_call = response.messages[-3].tool_call
                
                parsed_args = parse_json(tool_call.arguments)
                
                if 'message' not in parsed_args:
                    return "ERROR"
                    
                response_text = parsed_args['message']
                
            except (AttributeError, KeyError, IndexError, json.JSONDecodeError) as e:
                return "ERROR"
            
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
            
            if not self.is_gemini_client_initialized() or not model_settings.gemini_api_key:
                if 'GEMINI_API_KEY' not in status['missing_keys']:
                    status['missing_keys'].append('GEMINI_API_KEY')
            else:
                status['available_keys'].append('GEMINI_API_KEY')
                
        elif self.model_name in OPENAI_MODELS:
            status['model_requirements']['current_model'] = self.model_name
            status['model_requirements']['required_keys'] = ['OPENAI_API_KEY']
            
            if not model_settings.openai_api_key:
                if 'OPENAI_API_KEY' not in status['missing_keys']:
                    status['missing_keys'].append('OPENAI_API_KEY')
            else:
                status['available_keys'].append('OPENAI_API_KEY')
                
        elif 'claude' in self.model_name.lower():
            status['model_requirements']['current_model'] = self.model_name
            status['model_requirements']['required_keys'] = ['ANTHROPIC_API_KEY']
            
            if not model_settings.anthropic_api_key:
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
        Saves the key to .env file for persistence across sessions.
        Returns a dictionary with success status and any error messages.
        """
        result = {'success': False, 'message': ''}
        
        # Save to .env file first for all key types
        try:
            self._save_api_key_to_env_file(key_name, api_key)
            
            # Reload the global model_settings to pick up the new API key
            model_settings.__init__()
            
            # Update the environment variable for this session
            os.environ[key_name] = api_key
            
            result['success'] = True
            result['message'] = f'{key_name} successfully configured and saved to .env file!'
            
        except Exception as e:
            result['message'] = f'Failed to save {key_name}: {str(e)}'
            return result
        
        # Handle specific initialization for different services
        if key_name == 'GEMINI_API_KEY':
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
                    
                    result['message'] = 'Gemini API key successfully configured and Gemini client initialized!'
                else:
                    result['message'] = 'Gemini API key saved but failed to complete initialization'
                    
            except Exception as e:
                result['message'] = f'Gemini API key saved but validation failed: {str(e)}'
                self.google_client = None
                
        elif key_name == 'OPENAI_API_KEY':
            # For OpenAI, the key will be picked up automatically by the client
            # when model_settings is reloaded and used in the next request
            if 'OPENAI_API_KEY' in self.missing_api_keys:
                self.missing_api_keys.remove('OPENAI_API_KEY')
            result['message'] = 'OpenAI API key successfully configured!'
            
        elif key_name == 'ANTHROPIC_API_KEY':
            # For Anthropic, the key will be picked up automatically by the client
            # when model_settings is reloaded and used in the next request
            if 'ANTHROPIC_API_KEY' in self.missing_api_keys:
                self.missing_api_keys.remove('ANTHROPIC_API_KEY')
            result['message'] = 'Anthropic API key successfully configured!'
            
        else:
            # For any other API key, just confirm it was saved
            result['message'] = f'{key_name} successfully configured!'
            
        return result
    
    def _save_api_key_to_env_file(self, key_name: str, api_key: str):
        """
        Save an API key to the .env file for persistence across sessions.
        """
        from pathlib import Path
        
        # Find the .env file (look in current directory and parent directories)
        env_file_path = None
        current_path = Path.cwd()
        
        # Check current directory and up to 3 parent directories
        for _ in range(4):
            potential_env_path = current_path / '.env'
            if potential_env_path.exists():
                env_file_path = potential_env_path
                break
            current_path = current_path.parent
        
        # If no .env file found, create one in the current working directory
        if env_file_path is None:
            env_file_path = Path.cwd() / '.env'
        
        # Read existing .env file content
        env_content = {}
        if env_file_path.exists():
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_content[key.strip()] = value.strip()
        
        # Update the API key
        env_content[key_name] = api_key
        
        # Write back to .env file
        with open(env_file_path, 'w') as f:
            for key, value in env_content.items():
                f.write(f"{key}={value}\n")
        
        self.logger.info(f"API key {key_name} saved to {env_file_path}")

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
            