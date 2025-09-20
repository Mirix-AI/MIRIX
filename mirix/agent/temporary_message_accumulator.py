import os
import time
import uuid
import threading
import copy
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from mirix.agent.app_constants import TEMPORARY_MESSAGE_LIMIT, GEMINI_MODELS, SKIP_META_MEMORY_MANAGER
from mirix.constants import CHAINING_FOR_MEMORY_UPDATE
from mirix.voice_utils import process_voice_files, convert_base64_to_audio_segment
from mirix.agent.app_utils import encode_image

def get_image_mime_type(image_path):
    """Get MIME type for image files."""
    if image_path.lower().endswith(('.png', '.PNG')):
        return 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        return 'image/jpeg'
    elif image_path.lower().endswith(('.gif', '.GIF')):
        return 'image/gif'
    elif image_path.lower().endswith(('.webp', '.WEBP')):
        return 'image/webp'
    else:
        return 'image/png'  # Default fallback

class TemporaryMessageAccumulator:
    """
    Handles accumulation and processing of temporary messages (screenshots, voice, text)
    for memory absorption into different agent types.
    """
    
    def __init__(self, client, google_client, timezone, upload_manager, message_queue, 
                 model_name, temporary_message_limit=TEMPORARY_MESSAGE_LIMIT):
        self.client = client
        self.google_client = google_client
        self.timezone = timezone
        self.upload_manager = upload_manager
        self.message_queue = message_queue
        self.model_name = model_name
        self.temporary_message_limit = temporary_message_limit
        
        # Initialize logger
        self.logger = logging.getLogger(f"Mirix.TemporaryMessageAccumulator.{model_name}")
        self.logger.setLevel(logging.INFO)
        
        # Determine if this model needs file uploads
        self.needs_upload = model_name in GEMINI_MODELS
        
        # Initialize locks for thread safety
        self._temporary_messages_lock = threading.Lock()

        # Initialize temporary message storage
        self.temporary_messages = []  # Flat list of (timestamp, item) tuples
        self.temporary_user_messages = [[]]  # List of batches
        
        # URI tracking for cloud files
        self.uri_to_create_time = {}
        
        # Upload tracking for cleanup
        self.upload_start_times = {}  # Track when uploads started for cleanup purposes
    
    def add_message(self, full_message, timestamp, delete_after_upload=True, async_upload=True):
        """Add a message to temporary storage."""
        if self.needs_upload and self.upload_manager is not None:
            if 'image_uris' in full_message and full_message['image_uris']:
                # Handle image uploads with optional sources information
                if async_upload:
                    image_file_ref_placeholders = [self.upload_manager.upload_file_async(image_uri, timestamp) for image_uri in full_message['image_uris']]
                else:
                    image_file_ref_placeholders = [self.upload_manager.upload_file(image_uri, timestamp) for image_uri in full_message['image_uris']]
                # Track upload start times for timeout detection
                current_time = time.time()
                for placeholder in image_file_ref_placeholders:
                    if isinstance(placeholder, dict) and placeholder.get('pending'):
                        placeholder_id = id(placeholder)  # Use object ID as unique identifier
                        self.upload_start_times[placeholder_id] = current_time
            else:
                image_file_ref_placeholders = None
                
            if 'voice_files' in full_message and full_message['voice_files']:
                audio_segment = []
                for i, voice_file in enumerate(full_message['voice_files']):
                    converted_segment = convert_base64_to_audio_segment(voice_file)
                    if converted_segment is not None:
                        audio_segment.append(converted_segment)
                    else:
                        self.logger.error(f"❌ Error converting voice chunk {i+1}/{len(full_message['voice_files'])} to AudioSegment")
                        continue
                audio_segment = None if len(audio_segment) == 0 else audio_segment
                if audio_segment:
                    self.logger.info(f"✅ Successfully processed {len(audio_segment)} voice segments")
                else:
                    self.logger.info("❌ No voice segments were successfully processed")
            else:
                audio_segment = None

            with self._temporary_messages_lock:
                sources = full_message.get('sources')
                self.temporary_messages.append(
                    (timestamp, {'image_uris': image_file_ref_placeholders,
                                 'sources': sources,
                                 'audio_segments': audio_segment,
                                 'message': full_message['message']})
                )
                
                # Print accumulation statistics
                total_messages = len(self.temporary_messages)
                total_images = sum(len(item.get('image_uris', []) or []) for _, item in self.temporary_messages)
                total_voice_segments = sum(len(item.get('audio_segments', []) or []) for _, item in self.temporary_messages)

            if delete_after_upload and full_message['image_uris']:
                threading.Thread(
                    target=self._cleanup_file_after_upload, 
                    args=(full_message['image_uris'], image_file_ref_placeholders), 
                    daemon=True
                ).start()

        else:
            
            image_uris = full_message.get('image_uris', [])
            if image_uris is None:
                image_uris = []
            image_count = len(image_uris)
            voice_files = full_message.get('voice_files', [])
            if voice_files is None:
                voice_files = []
            voice_count = len(voice_files)
            
            with self._temporary_messages_lock:
                sources = full_message.get('sources')
                image_uris = full_message.get('image_uris', [])
                self.temporary_messages.append(
                    (timestamp, {
                        'image_uris': image_uris,
                        'sources': sources,
                        'audio_segments': full_message.get('voice_files', []),
                        'message': full_message['message'],
                        'delete_after_upload': delete_after_upload  # Store delete flag for OpenAI models
                    })
                )
                
                # # Print accumulation statistics
                # total_messages = len(self.temporary_messages)
                # total_images = sum(len(item.get('image_uris', []) or []) for _, item in self.temporary_messages)
                # total_voice_files = sum(len(item.get('audio_segments', []) or []) for _, item in self.temporary_messages)
        
    def add_user_conversation(self, user_message, assistant_response):
        """Add user conversation to temporary storage."""
        self.temporary_user_messages[-1].extend([
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': assistant_response}
        ])
    

    
    def should_absorb_content(self):
        """Check if content should be absorbed into memory and return ready messages."""
        
        if self.needs_upload:
            with self._temporary_messages_lock:
                ready_messages = []
                
                # Process messages in temporal order
                for i, (timestamp, item) in enumerate(self.temporary_messages):
                    item_copy = copy.deepcopy(item)
                    has_pending_uploads = False
                    
                    # Check if this message has any pending uploads
                    if 'image_uris' in item and item['image_uris']:
                        processed_image_uris = []
                        pending_count = 0
                        completed_count = 0
                        
                        for j, file_ref in enumerate(item['image_uris']):
                            if isinstance(file_ref, dict) and file_ref.get('pending'):
                                placeholder_id = id(file_ref)
                                
                                # Get upload status
                                upload_status = self.upload_manager.get_upload_status(file_ref)
                                
                                if upload_status['status'] == 'completed':
                                    # Upload completed, use the resolved reference
                                    processed_image_uris.append(upload_status['result'])
                                    completed_count += 1
                                    # Note: Don't clean up here, this is just a check
                                elif upload_status['status'] == 'failed':
                                    # Note: Don't clean up here, this is just a check
                                    continue
                                elif upload_status['status'] == 'unknown':
                                    # Upload was cleaned up, treat as failed
                                    continue
                                else:
                                    # Still pending
                                    has_pending_uploads = True
                                    pending_count += 1
                                    break
                            else:
                                # Already uploaded file reference
                                processed_image_uris.append(file_ref)
                                completed_count += 1
                        
                        if has_pending_uploads:
                            # Found a pending message - we must stop here to maintain temporal order
                            # We cannot process any messages beyond this point
                            break
                        else:
                            # Update the copy with resolved image URIs
                            item_copy['image_uris'] = processed_image_uris
                            ready_messages.append((timestamp, item_copy))
                    else:
                        # No images or already processed, add to ready list
                        ready_messages.append((timestamp, item_copy))

                # Check if we have enough ready messages to process
                if len(ready_messages) >= self.temporary_message_limit:
                    return ready_messages
                else:
                    return []
        else:
            # For non-GEMINI models: no uploads needed, just check message count
            with self._temporary_messages_lock:
                # Since there are no pending uploads to wait for, all messages are ready
                if len(self.temporary_messages) >= self.temporary_message_limit:
                    # Return all messages as ready for processing
                    ready_messages = []
                    for timestamp, item in self.temporary_messages:
                        item_copy = copy.deepcopy(item)
                        ready_messages.append((timestamp, item_copy))
                    return ready_messages
                else:
                    return []
    
    def get_recent_images_for_chat(self, current_timestamp):
        """Get the most recent images for chat context (non-blocking).
        
        Returns:
            List of tuples: (timestamp, file_ref, sources) where sources may be None
        """
        with self._temporary_messages_lock:
            # Get the most recent content
            recent_limit = min(self.temporary_message_limit, len(self.temporary_messages))
            most_recent_content = self.temporary_messages[-recent_limit:] if recent_limit > 0 else []
            
            # Calculate timestamp cutoff (1 minute ago)
            cutoff_time = current_timestamp - timedelta(minutes=1)
            
            # Extract only images for the current message context
            most_recent_images = []
            for timestamp, item in most_recent_content:
                # Handle different timestamp formats that might be used
                if isinstance(timestamp, str):
                    # Try to parse timestamp string and make it timezone-aware
                    timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    # If timezone-naive, localize it to match the cutoff_time timezone awareness
                    if timestamp_dt.tzinfo is None:
                        timestamp_dt = self.timezone.localize(timestamp_dt)
                elif isinstance(timestamp, datetime):
                    timestamp_dt = timestamp
                    # If timezone-naive, localize it to match the cutoff_time timezone awareness
                    if timestamp_dt.tzinfo is None:
                        timestamp_dt = self.timezone.localize(timestamp_dt)
                elif isinstance(timestamp, (int, float)):
                    # Unix timestamp - make it timezone-aware
                    timestamp_dt = datetime.fromtimestamp(timestamp, tz=self.timezone)
                else:
                    # Skip if we can't parse the timestamp
                    continue
                
                # Check if timestamp is within the past 1 minute
                if timestamp_dt < cutoff_time:
                    continue
                
                # Check if this item has images
                if 'image_uris' in item and item['image_uris']:
                    for j, file_ref in enumerate(item['image_uris']):
                        if self.needs_upload and self.upload_manager is not None:
                            # For GEMINI models: Resolve pending uploads for immediate use (non-blocking check)
                            if isinstance(file_ref, dict) and file_ref.get('pending'):
                                placeholder_id = id(file_ref)
                                
                                # Get upload status
                                upload_status = self.upload_manager.get_upload_status(file_ref)
                                
                                if upload_status['status'] == 'completed':
                                    original_placeholder = file_ref  # Store original before modifying
                                    file_ref = upload_status['result']
                                    # Note: Don't clean up here, this is just for chat context
                                elif upload_status['status'] == 'failed':
                                    # Upload failed, skip this image
                                    # Note: Don't clean up here, this is just for chat context
                                    continue
                                elif upload_status['status'] == 'unknown':
                                    # Upload was cleaned up, treat as failed
                                    # Note: Don't clean up here, this is just for chat context
                                    continue
                                else:
                                    continue  # Still pending, skip
                                    
                        # For non-GEMINI models: file_ref is already the image URI, use as-is
                        # Include sources information if available
                        sources = item.get('sources')
                        most_recent_images.append((timestamp, file_ref, sources[j] if sources else None))
            
            return most_recent_images
    
    def absorb_content_into_memory(self, agent_states, ready_messages=None, user_id=None):
        """Process accumulated content and send to memory agents."""

        if ready_messages is not None:
            # Use the pre-processed ready messages
            ready_to_process = ready_messages
            
            # Remove the processed messages from temporary_messages and clean up placeholders
            with self._temporary_messages_lock:
                # Remove processed messages from the beginning (they were processed in temporal order)
                num_to_remove = len(ready_messages)
                
                # Clean up placeholders from the messages being removed
                if self.needs_upload and self.upload_manager is not None:
                    for i in range(min(num_to_remove, len(self.temporary_messages))):
                        timestamp, item = self.temporary_messages[i]
                        if 'image_uris' in item and item['image_uris']:
                            for file_ref in item['image_uris']:
                                if isinstance(file_ref, dict) and file_ref.get('pending'):
                                    placeholder_id = id(file_ref)
                                    # Clean up upload manager status and local tracking
                                    self.upload_manager.cleanup_resolved_upload(file_ref)
                                    self.upload_start_times.pop(placeholder_id, None)
                
                self.temporary_messages = self.temporary_messages[num_to_remove:]
        else:
            # Use the existing logic to separate and process messages
            with self._temporary_messages_lock:
                # Separate uploaded images, pending images, and text content
                ready_to_process = []  # Items that are ready to be processed
                pending_items = []     # Items that need to stay for next cycle
                
                for timestamp, item in self.temporary_messages:
                    item_copy = copy.deepcopy(item)
                    has_pending_uploads = False
                    
                    # Process image URIs if they exist
                    if 'image_uris' in item and item['image_uris']:
                        processed_image_uris = []
                        for file_ref in item['image_uris']:
                            if self.needs_upload and self.upload_manager is not None:
                                # For GEMINI models: Check if this is a pending placeholder
                                if isinstance(file_ref, dict) and file_ref.get('pending'):
                                    placeholder_id = id(file_ref)
                                    # Get upload status
                                    upload_status = self.upload_manager.get_upload_status(file_ref)
                                    
                                    if upload_status['status'] == 'completed':
                                        # Upload completed, use the result
                                        processed_image_uris.append(upload_status['result'])
                                        # Clean up both upload manager and local tracking
                                        self.upload_manager.cleanup_resolved_upload(file_ref)
                                        self.upload_start_times.pop(placeholder_id, None)
                                    elif upload_status['status'] == 'failed':
                                        # Upload failed, skip this image but continue processing
                                        # Clean up both upload manager and local tracking
                                        self.upload_manager.cleanup_resolved_upload(file_ref)
                                        self.upload_start_times.pop(placeholder_id, None)
                                        continue
                                    elif upload_status['status'] == 'unknown':
                                        # Upload was cleaned up, treat as failed
                                        print(f"Skipping unknown/cleaned upload in absorb_content_into_memory")
                                        # Only clean up local tracking since upload manager already cleaned up
                                        self.upload_start_times.pop(placeholder_id, None)
                                        continue
                                    else:
                                        # Still pending, keep original for next cycle
                                        has_pending_uploads = True
                                        break
                                else:
                                    # Already uploaded file reference
                                    processed_image_uris.append(file_ref)
                            else:
                                # For non-GEMINI models: store the image URI directly for base64 conversion later
                                processed_image_uris.append(file_ref)
                        
                        if has_pending_uploads:
                            # Keep for next cycle if any uploads are still pending
                            pending_items.append((timestamp, item))
                        else:
                            # All uploads completed, update the item
                            item_copy['image_uris'] = processed_image_uris
                            ready_to_process.append((timestamp, item_copy))
                    else:
                        # No images or already processed, add to ready list
                        ready_to_process.append((timestamp, item_copy))

                # Keep only items that are still pending (for GEMINI models) or clear all (for non-GEMINI models)
                self.temporary_messages = pending_items

        # Extract voice content from ready_to_process messages
        voice_content = []
        for _, item in ready_to_process:
            if 'audio_segments' in item and item['audio_segments'] is not None:
                # audio_segments can be a list of audio segments that can be directly combined
                voice_content.extend(item['audio_segments'])

        # Save voice content to folder if any exists
        if voice_content:

            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            voice_folder = f"tmp_voice_content_{current_timestamp}"
            
            try:
                os.makedirs(voice_folder, exist_ok=True)
                self.logger.info(f"Created voice content folder: {voice_folder}")
                
                for i, audio_segment in enumerate(voice_content):
                    try:
                        # Save audio segment to file
                        if hasattr(audio_segment, 'export'):
                            # AudioSegment object
                            filename = f"voice_segment_{i+1:03d}.wav"
                            filepath = os.path.join(voice_folder, filename)
                            audio_segment.export(filepath, format="wav")
                            self.logger.info(f"Saved voice segment {i+1} to {filepath}")
                        else:
                            # Handle other audio formats (e.g., raw bytes)
                            filename = f"voice_segment_{i+1:03d}.dat"
                            filepath = os.path.join(voice_folder, filename)
                            with open(filepath, 'wb') as f:
                                if isinstance(audio_segment, bytes):
                                    f.write(audio_segment)
                                else:
                                    # Convert to bytes if needed
                                    f.write(str(audio_segment).encode())
                            self.logger.info(f"Saved voice data {i+1} to {filepath}")
                    except Exception as e:
                        self.logger.error(f"Failed to save voice segment {i+1}: {e}")
                        
                self.logger.info(f"Successfully saved {len(voice_content)} voice segments to {voice_folder}")
            except Exception as e:
                self.logger.error(f"Failed to create voice content folder {voice_folder}: {e}")

        # Process content and build message
        message = self._build_memory_message(ready_to_process, voice_content)
        
        # Handle user conversation if exists
        message, user_message_added = self._add_user_conversation_to_message(message)
       
        if SKIP_META_MEMORY_MANAGER:
            # Add system instruction
            if user_message_added:
                system_message = "[System Message] Interpret the provided content and the conversations between the user and the chat agent, according to what the user is doing, trigger the appropriate memory update."
            else:
                system_message = "[System Message] Interpret the provided content, according to what the user is doing, extract the important information matching your memory type and save it into the memory."
        else:
            # Add system instruction for meta memory manager
            if user_message_added:
                system_message = "[System Message] As the meta memory manager, analyze the provided content and the conversations between the user and the chat agent. Based on what the user is doing, determine which memory should be updated (episodic, procedural, knowledge vault, semantic, core, and resource)."
            else:
                system_message = "[System Message] As the meta memory manager, analyze the provided content. Based on the content, determine what memories need to be updated (episodic, procedural, knowledge vault, semantic, core, and resource)"
            
        message.append({
            'type': 'text',
            'text': system_message
        })

        t1 = time.time()
        if SKIP_META_MEMORY_MANAGER:
            # Send to memory agents in parallel
            self._send_to_memory_agents_separately(message, set(list(self.uri_to_create_time.keys())), agent_states, user_id=user_id)
        else:
            # Send to meta memory agent
            response, agent_type = self._send_to_meta_memory_agent(message, set(list(self.uri_to_create_time.keys())), agent_states, user_id=user_id)

        t2 = time.time()
        self.logger.info(f"Time taken to send to memory agents: {t2 - t1} seconds")

        # # write the logic to send the message to all the agents one by one
        # payloads = {
        #     'message': message,
        #     'chaining': CHAINING_FOR_MEMORY_UPDATE
        # }
        
        # for agent_type in ['episodic_memory', 'procedural_memory', 'knowledge_vault', 
        #                  'semantic_memory', 'core_memory', 'resource_memory']:
        #     self.message_queue.send_message_in_queue(
        #         self.client,
        #         agent_states,
        #         payloads,
        #         agent_type
        #     )
        
        # Clean up processed content
        self._cleanup_processed_content(ready_to_process, user_message_added)
    
    def _build_memory_message(self, ready_to_process, voice_content):
        """Build the message content for memory agents."""

        # Collect content organized by source
        images_by_source = {}  # source_name -> [(timestamp, file_refs)]
        text_content = []
        audio_content = []

        for timestamp, item in ready_to_process:
            # Handle images with sources
            if 'image_uris' in item and item['image_uris']:
                sources = item.get('sources', [])
                image_uris = item['image_uris']
                
                # If we have sources, group images by source
                if sources and len(sources) == len(image_uris):
                    for source, file_ref in zip(sources, image_uris):
                        if source not in images_by_source:
                            images_by_source[source] = []
                        images_by_source[source].append((timestamp, file_ref))
                else:
                    # Fallback: if no sources or mismatch, group under generic name
                    generic_source = "Screenshots"
                    if generic_source not in images_by_source:
                        images_by_source[generic_source] = []
                    for file_ref in image_uris:
                        images_by_source[generic_source].append((timestamp, file_ref))
            
            # Handle text messages
            if 'message' in item and item['message']:
                text_content.append((timestamp, item['message']))
            
            # Handle audio segments
            if 'audio_segments' in item and item['audio_segments']:
                audio_content.extend(item['audio_segments'])

        # Process voice files from both sources (voice_content and audio_segments)
        all_voice_content = voice_content.copy() if voice_content else []
        all_voice_content.extend(audio_content)
        
        voice_transcription = ""
        if all_voice_content:
            t1 = time.time()
            voice_transcription = process_voice_files(all_voice_content)
            t2 = time.time()

        # Build the structured message for memory agents
        message_parts = []
        
        # Add screenshots grouped by source
        if images_by_source:
            # Add general introductory text
            message_parts.append({
                'type': 'text',
                'text': 'The following are the screenshots taken from the computer of the user:'
            })
            
            # Group by source application
            for source_name, source_images in images_by_source.items():
                # Add source-specific header
                message_parts.append({
                    'type': 'text',
                    'text': f"These are the screenshots from {source_name}:"
                })
                
                # Add each image with its timestamp
                for timestamp, file_ref in source_images:
                    message_parts.append({
                        'type': 'text',
                        'text': f"Timestamp: {timestamp}"
                    })
                    
                    # Handle different types of file references
                    if hasattr(file_ref, 'uri'):
                        # GEMINI models: use Google Cloud file URI
                        message_parts.append({
                            'type': 'google_cloud_file_uri',
                            'google_cloud_file_uri': file_ref.uri
                        })
                    else:
                        # OpenAI models: convert to base64
                        try:
                            mime_type = get_image_mime_type(file_ref)
                            base64_data = encode_image(file_ref)
                            message_parts.append({
                                'type': 'image_data',
                                'image_data': {
                                    'data': f"data:{mime_type};base64,{base64_data}",
                                    'detail': 'auto'
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Failed to encode image {file_ref}: {e}")
                            # Add a text message indicating the image couldn't be processed
                            message_parts.append({
                                'type': 'text',
                                'text': f"[Image at {file_ref} could not be processed]"
                            })
        
        # Add voice transcription if any
        if voice_transcription:
            message_parts.append({
                'type': 'text',
                'text': f'The following are the voice recordings and their transcriptions:\n{voice_transcription}'
            })
        
        # Add text content if any
        if text_content:
            message_parts.append({
                'type': 'text',
                'text': 'The following are text messages from the user:'
            })
            
            for idx, (timestamp, text) in enumerate(text_content):
                message_parts.append({
                    'type': 'text',
                    'text': f"Timestamp: {timestamp} Text:\n{text}"
                })

        return message_parts
    
    def _add_user_conversation_to_message(self, message):
        """Add user conversation to the message if it exists."""
        user_message_added = False
        if len(self.temporary_user_messages[-1]) > 0:
            user_conversation = 'The following are the conversations between the user and the Chat Agent while capturing this content:\n'
            for idx, user_message in enumerate(self.temporary_user_messages[-1]):
                user_conversation += f"role: {user_message['role']}; content: {user_message['content']}\n"
            user_conversation = user_conversation.strip()
            
            message.append({
                'type': 'text',
                'text': user_conversation
            })
            
            self.temporary_user_messages.append([])
            user_message_added = True
        return message, user_message_added
    
    def _send_to_meta_memory_agent(self, message, existing_file_uris, agent_states, user_id=None):
        """Send the processed content to the meta memory agent."""
        
        payloads = {
            'message': message,
            'existing_file_uris': existing_file_uris,
            'chaining': CHAINING_FOR_MEMORY_UPDATE,
            'message_queue': self.message_queue,
            'user_id': user_id
        }

        response, agent_type = self.message_queue.send_message_in_queue(
            self.client, agent_states.meta_memory_agent_state.id, payloads, 'meta_memory'
        )
        return response, agent_type

    def _send_to_memory_agents_separately(self, message, existing_file_uris, agent_states, user_id=None):
        """Send the processed content to all memory agents in parallel."""
        import time
        import threading
        
        payloads = {
            'message': message,
            'existing_file_uris': existing_file_uris,
            'chaining': CHAINING_FOR_MEMORY_UPDATE,
            'user_id': user_id
        }
        
        responses = []
        memory_agent_types = ['episodic_memory', 'procedural_memory', 'knowledge_vault', 
                             'semantic_memory', 'core_memory', 'resource_memory']
        
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(self.message_queue.send_message_in_queue, 
                           self.client, self.message_queue._get_agent_id_for_type(agent_states, agent_type), payloads, agent_type) 
                for agent_type in memory_agent_types
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                response, agent_type = future.result()
                responses.append(response)
        
        overall_end = time.time()
  
    def _cleanup_processed_content(self, ready_to_process, user_message_added):
        """Clean up processed content and mark files as processed."""
        # Mark processed files as processed in database and cleanup upload results (only for GEMINI models)
        if self.needs_upload and self.upload_manager is not None:
            for timestamp, item in ready_to_process:
                if 'image_uris' in item and item['image_uris']:
                    for file_ref in item['image_uris']:
                        if hasattr(file_ref, 'name'):
                            try:
                                self.client.server.cloud_file_mapping_manager.set_processed(cloud_file_id=file_ref.name)
                            except Exception as e:
                                pass
            
            # Clean up upload results from memory now that they've been processed
            # We need to track which placeholders were originally used to get these file_refs
            # Since we don't have direct access to the original placeholders, we'll rely on
            # the cleanup happening in the upload manager's periodic cleanup or
            # when the same placeholder is accessed again
        else:
            # For OpenAI models: Clean up image files if delete_after_upload is True
            for timestamp, item in ready_to_process:
                # Check if this item should have its files deleted
                should_delete = item.get('delete_after_upload', True)  # Default to True for backward compatibility
                
                if should_delete and 'image_uris' in item and item['image_uris']:
                    for image_uri in item['image_uris']:
                        # Only delete if it's a local file path (string)
                        if isinstance(image_uri, str):
                            self._delete_local_image_file(image_uri)
        
        # Clean up user messages if added
        if user_message_added:
            if len(self.temporary_user_messages) > 1:
                self.temporary_user_messages.pop(0)
    
    def _delete_local_image_file(self, image_path):
        """Delete a local image file with retry logic."""
        try:
            max_retries = 10
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        self.logger.debug(f"Deleted processed image file: {image_path}")
                        if not os.path.exists(image_path):
                            break
                    else:
                        break  # File doesn't exist, nothing to do
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(0.1)
                    else:
                        self.logger.warning(f"Failed to delete image file {image_path} after {max_retries} attempts: {e}")
        except Exception as e:
            self.logger.error(f"Error while trying to delete image file {image_path}: {e}")

    def _cleanup_file_after_upload(self, filenames, placeholders):
        """Clean up local file after upload completes."""

        if self.upload_manager is None:
            return  # No upload manager for non-GEMINI models
        
        for filename, placeholder in zip(filenames, placeholders):
            placeholder_id = id(placeholder) if isinstance(placeholder, dict) else None
            
            try:
                # Wait for upload to complete with timeout
                upload_successful = self.upload_manager.wait_for_upload(placeholder, timeout=60)
                
                if upload_successful:
                    # Clean up tracking
                    if placeholder_id:
                        self.upload_start_times.pop(placeholder_id, None)
                else:
                    # Don't clean up tracking here, let the timeout detection handle it
                    pass
                
                # Remove file after upload attempt (successful or not)
                max_retries = 10
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                            # self.logger.info(f"Removed file: {filename}")
                            if not os.path.exists(filename):
                                break
                            else:
                                pass
                        else:
                            break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(0.1)
                        else:
                            pass
                        
            except Exception as e:
                # Still try to remove the local file
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except Exception as cleanup_error:
                    pass
    
    def get_message_count(self):
        """Get the current count of temporary messages."""
        with self._temporary_messages_lock:
            return len(self.temporary_messages)
    
    def get_upload_status_summary(self):
        """Get a summary of current upload statuses for debugging."""
        summary = {
            'total_messages': len(self.temporary_messages),
        }
        
        # Get upload manager status if available
        if self.upload_manager and hasattr(self.upload_manager, 'get_upload_status_summary'):
            summary['upload_manager_status'] = self.upload_manager.get_upload_status_summary()
        
        return summary
    
    def update_model(self, new_model_name):
        """Update the model name and related settings."""
        self.model_name = new_model_name
        self.needs_upload = new_model_name in GEMINI_MODELS
        self.logger = logging.getLogger(f"Mirix.TemporaryMessageAccumulator.{new_model_name}")
        self.logger.setLevel(logging.INFO) 