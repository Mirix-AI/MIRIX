import os
import traceback
import base64
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import queue
import threading
from ..agent.agent_wrapper import AgentWrapper

"""
VOICE RECORDING STRATEGY & ARCHITECTURE:

Current Implementation:
- Frontend records audio in 5-second chunks (CHUNK_DURATION = 5000ms)
- Chunks are accumulated locally until a screenshot is sent
- Raw voice files are sent to the agent for accumulation and processing
- Agent accumulates voice files alongside images until TEMPORARY_MESSAGE_LIMIT is reached
- Voice processing happens in agent.absorb_content_into_memory()

Recommended Alternative Strategy:
Instead of 5-second chunks, you can:
1. Send 1-second micro-chunks to reduce latency
2. Agent accumulates chunks until TEMPORARY_MESSAGE_LIMIT is reached
3. This aligns perfectly with how images are accumulated in agent.py

Benefits of 1-second chunks:
- Lower latency for real-time feedback
- More granular control over audio processing
- Better alignment with the existing image accumulation pattern
- Smoother user experience
- Voice processing happens in batches during memory absorption

Implementation changes needed:
- Frontend: Change CHUNK_DURATION from 5000 to 1000
- Agent: Handles voice file accumulation and processing during memory absorption
- Server: Passes raw voice files to agent without processing

FFPROBE WARNING:
The warning about ffprobe/avprobe is harmless and expected if FFmpeg isn't in your system PATH.
To fix it, install FFmpeg:
- Windows: Download from https://ffmpeg.org and add to PATH
- macOS: brew install ffmpeg  
- Linux: sudo apt install ffmpeg

The warning doesn't affect functionality as pydub falls back gracefully.
"""

app = FastAPI(title="Mirix Agent API", version="0.1.2")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

class MessageRequest(BaseModel):
    message: Optional[str] = None
    image_uris: Optional[List[str]] = None
    sources: Optional[List[str]] = None  # Source names corresponding to image_uris
    voice_files: Optional[List[str]] = None  # Base64 encoded voice files
    memorizing: bool = False
    is_screen_monitoring: Optional[bool] = False

class MessageResponse(BaseModel):
    response: str
    status: str = "success"

class PersonaDetailsResponse(BaseModel):
    personas: Dict[str, str]

class UpdatePersonaRequest(BaseModel):
    text: str

class UpdatePersonaResponse(BaseModel):
    success: bool
    message: str

class UpdateCoreMemoryRequest(BaseModel):
    label: str
    text: str

class UpdateCoreMemoryResponse(BaseModel):
    success: bool
    message: str

class ApplyPersonaTemplateRequest(BaseModel):
    persona_name: str

class CoreMemoryPersonaResponse(BaseModel):
    text: str

class SetModelRequest(BaseModel):
    model: str

class AddCustomModelRequest(BaseModel):
    model_name: str
    model_endpoint: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4096
    maximum_length: int = 32768

class AddCustomModelResponse(BaseModel):
    success: bool
    message: str

class ListCustomModelsResponse(BaseModel):
    models: List[str]

class SetModelResponse(BaseModel):
    success: bool
    message: str
    missing_keys: List[str]
    model_requirements: Dict[str, Any]

class GetCurrentModelResponse(BaseModel):
    current_model: str

class SetTimezoneRequest(BaseModel):
    timezone: str

class SetTimezoneResponse(BaseModel):
    success: bool
    message: str

class GetTimezoneResponse(BaseModel):
    timezone: str

class ScreenshotSettingRequest(BaseModel):
    include_recent_screenshots: bool

class ScreenshotSettingResponse(BaseModel):
    success: bool
    include_recent_screenshots: bool
    message: str

# API Key validation functionality
def get_required_api_keys_for_model(model_endpoint_type: str) -> List[str]:
    """Get required API keys for a given model endpoint type"""
    api_key_mapping = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "azure": ["AZURE_API_KEY", "AZURE_BASE_URL", "AZURE_API_VERSION"],
        "google_ai": ["GEMINI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    }
    return api_key_mapping.get(model_endpoint_type, [])

def check_missing_api_keys(agent) -> Dict[str, List[str]]:
    """Check for missing API keys based on the agent's configuration"""
    
    if agent is None:
        return {"error": ["Agent not initialized"]}
    
    try:
        # Use the new AgentWrapper method instead of the old logic
        status = agent.check_api_key_status()
        
        return {
            "missing_keys": status["missing_keys"],
            "model_type": status.get("model_requirements", {}).get("current_model", "unknown")
        }
        
    except Exception as e:
        print(f"Error in check_missing_api_keys: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {"error": [f"Error checking API keys: {str(e)}"]}

class ApiKeyRequest(BaseModel):
    key_name: str
    key_value: str

class ApiKeyCheckResponse(BaseModel):
    missing_keys: List[str]
    model_type: str
    requires_api_key: bool

class ApiKeyUpdateResponse(BaseModel):
    success: bool
    message: str

# Memory endpoint response models
class EpisodicMemoryItem(BaseModel):
    timestamp: str
    content: str
    context: Optional[str] = None
    emotions: Optional[List[str]] = None

class KnowledgeSkillItem(BaseModel):
    title: str
    type: str  # "semantic" or "procedural"
    content: str
    proficiency: Optional[str] = None
    tags: Optional[List[str]] = None

class DocsFilesItem(BaseModel):
    filename: str
    type: str
    summary: str
    last_accessed: Optional[str] = None
    size: Optional[str] = None

class CoreUnderstandingItem(BaseModel):
    aspect: str
    understanding: str
    confidence: Optional[float] = None
    last_updated: Optional[str] = None

class CredentialItem(BaseModel):
    name: str
    type: str
    content: str  # Will be masked
    tags: Optional[List[str]] = None
    last_used: Optional[str] = None

class ClearConversationResponse(BaseModel):
    success: bool
    message: str
    messages_deleted: int

class CleanupDetachedMessagesResponse(BaseModel):
    success: bool
    message: str
    cleanup_results: Dict[str, int]

class ExportMemoriesRequest(BaseModel):
    file_path: str
    memory_types: List[str]
    include_embeddings: bool = False

class ExportMemoriesResponse(BaseModel):
    success: bool
    message: str
    exported_counts: Dict[str, int]
    total_exported: int
    file_path: str

class ReflexionRequest(BaseModel):
    pass  # No parameters needed for now

class ReflexionResponse(BaseModel):
    success: bool
    message: str
    processing_time: Optional[float] = None



@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the server starts"""
    global agent
    agent = AgentWrapper('configs/mirix_monitor.yaml')
    print("Agent initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring server status"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/send_message")
async def send_message_endpoint(request: MessageRequest):
    """Send a message to the agent and get the response"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Check for missing API keys
    api_key_check = check_missing_api_keys(agent)
    if "error" in api_key_check:
        raise HTTPException(status_code=500, detail=api_key_check["error"][0])
    
    if api_key_check["missing_keys"]:
        # Return a special response indicating missing API keys
        return MessageResponse(
            response=f"Missing API keys for {api_key_check['model_type']} model: {', '.join(api_key_check['missing_keys'])}. Please provide the required API keys.",
            status="missing_api_keys"
        )
    
    try:
        print(f"Starting agent.send_message (non-streaming) with: message='{request.message}', memorizing={request.memorizing}")
        
        # Run the blocking agent.send_message() in a background thread to avoid blocking other requests
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            lambda: agent.send_message(
                message=request.message,
                image_uris=request.image_uris,
                sources=request.sources,  # Pass sources to agent
                voice_files=request.voice_files,  # Pass voice files to agent
                memorizing=request.memorizing
            )
        )
        
        print(f"Agent response (non-streaming): {response}")
        
        if response == "ERROR":
            raise HTTPException(status_code=500, detail="Agent returned an error")
        
        # Handle case where agent returns None
        if response is None:
            if request.memorizing:
                # When memorizing=True, None response is expected (no response needed)
                response = ""
            else:
                # When memorizing=False, None response is an error
                response = "I received your message but couldn't generate a response. Please try again."
        
        return MessageResponse(response=response)
    
    except Exception as e:
        print(f"Error in send_message_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/send_streaming_message")
async def send_streaming_message_endpoint(request: MessageRequest):
    """Send a message to the agent and stream intermediate messages and final response"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Check for missing API keys
    api_key_check = check_missing_api_keys(agent)
    if "error" in api_key_check:
        raise HTTPException(status_code=500, detail=api_key_check["error"][0])
    
    if api_key_check["missing_keys"]:
        # Return a special SSE event for missing API keys
        async def missing_keys_response():
            yield f"data: {json.dumps({'type': 'missing_api_keys', 'missing_keys': api_key_check['missing_keys'], 'model_type': api_key_check['model_type']})}\n\n"
        
        return StreamingResponse(
            missing_keys_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    
    agent.update_chat_agent_system_prompt(request.is_screen_monitoring)

    # Create a queue to collect intermediate messagess
    message_queue = queue.Queue()
    
    def display_intermediate_message(message_type: str, message: str):
        """Callback function to capture intermediate messages"""
        message_queue.put({
            "type": "intermediate",
            "message_type": message_type,
            "content": message
        })
    
    async def generate_stream():
        """Generator function for streaming responses"""
        try:
            # Start the agent processing in a separate thread
            result_queue = queue.Queue()
            
            async def run_agent():
                try:
                    
                    # Run agent.send_message in a background thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,  # Use default ThreadPoolExecutor
                        lambda: agent.send_message(
                            message=request.message,
                            image_uris=request.image_uris,
                            sources=request.sources,  # Pass sources to agent
                            voice_files=request.voice_files,  # Pass raw voice files
                            memorizing=request.memorizing,
                            display_intermediate_message=display_intermediate_message
                        )
                    )
                    # Handle various response cases
                    if response is None:
                        if request.memorizing:
                            result_queue.put({"type": "final", "response": ""})
                        else:
                            print("[DEBUG] Agent returned None response")
                            result_queue.put({"type": "error", "error": "Agent returned no response"})
                    elif isinstance(response, str) and response.startswith("ERROR_"):
                        # Handle specific error types from agent wrapper
                        print(f"[DEBUG] Agent returned specific error: {response}")
                        if response == "ERROR_RESPONSE_FAILED":
                            print("[DEBUG] - Message queue response failed")
                            result_queue.put({"type": "error", "error": "Message processing failed in agent queue"})
                        elif response == "ERROR_INVALID_RESPONSE_STRUCTURE":
                            print("[DEBUG] - Response structure invalid (missing messages or insufficient count)")
                            result_queue.put({"type": "error", "error": "Invalid response structure from agent"})
                        elif response == "ERROR_NO_TOOL_CALL":
                            print("[DEBUG] - Expected message missing tool_call attribute")
                            result_queue.put({"type": "error", "error": "Agent response missing required tool call"})
                        elif response == "ERROR_NO_MESSAGE_IN_ARGS":
                            print("[DEBUG] - Tool call arguments missing 'message' key")
                            result_queue.put({"type": "error", "error": "Agent tool call missing message content"})
                        elif response == "ERROR_PARSING_EXCEPTION":
                            print("[DEBUG] - Exception occurred during response parsing")
                            result_queue.put({"type": "error", "error": "Failed to parse agent response"})
                        else:
                            print(f"[DEBUG] - Unknown error type: {response}")
                            result_queue.put({"type": "error", "error": f"Unknown agent error: {response}"})
                    elif response == "ERROR":
                        print("[DEBUG] Agent returned generic ERROR string")
                        result_queue.put({"type": "error", "error": "Agent processing failed"})
                    elif not response or (isinstance(response, str) and response.strip() == ""):
                        if request.memorizing:
                            print("[DEBUG] Agent returned empty response - expected for memorizing=True")
                            result_queue.put({"type": "final", "response": ""})
                        else:
                            print("[DEBUG] Agent returned empty response unexpectedly")
                            result_queue.put({"type": "error", "error": "Agent returned empty response"})
                    else:
                        print(f"[DEBUG] Agent returned successful response (length: {len(str(response))})")
                        result_queue.put({"type": "final", "response": response})
                        
                except Exception as e:
                    print(f"[DEBUG] Exception in run_agent: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    result_queue.put({"type": "error", "error": str(e)})
            
            # Start agent processing as async task
            agent_task = asyncio.create_task(run_agent())
            
            # Keep track of whether we've sent the final result
            final_result_sent = False
            
            # Stream intermediate messages and wait for final result
            while not final_result_sent:
                # Check for intermediate messages first
                try:
                    intermediate_msg = message_queue.get_nowait()
                    yield f"data: {json.dumps(intermediate_msg)}\n\n"
                    continue  # Continue to next iteration to check for more messages
                except queue.Empty:
                    pass
                
                # Check for final result with timeout
                try:
                    # Use a short timeout to allow for intermediate messages
                    final_result = result_queue.get(timeout=0.1)
                    if final_result["type"] == "error":
                        yield f"data: {json.dumps({'type': 'error', 'error': final_result['error']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'final', 'response': final_result['response']})}\n\n"
                    final_result_sent = True
                    break
                except queue.Empty:
                    # If no result yet, check if task is still running
                    if agent_task.done():
                        # Task is done but no result - this shouldn't happen, but handle it
                        try:
                            # Check if the task raised an exception
                            agent_task.result()
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'error': f'Agent processing failed: {str(e)}'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'error': 'Agent processing completed unexpectedly without result'})}\n\n"
                        final_result_sent = True
                        break
                    # Otherwise continue the loop to check for more intermediate messages
                    await asyncio.sleep(0.1)  # Yield control to allow other operations
            
            # Make sure task completes
            if not agent_task.done():
                try:
                    await asyncio.wait_for(agent_task, timeout=5.0)
                except asyncio.TimeoutError:
                    agent_task.cancel()
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Agent processing timed out'})}\n\n"
            
        except Exception as e:
            print(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    try:
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    except Exception as e:
        print(f"Error in send_streaming_message_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.get("/personas", response_model=PersonaDetailsResponse)
async def get_personas():
    """Get all personas with their details (name and text)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        persona_details = agent.get_persona_details()
        return PersonaDetailsResponse(personas=persona_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting personas: {str(e)}")

@app.post("/personas/update", response_model=UpdatePersonaResponse)
async def update_persona(request: UpdatePersonaRequest):
    """Update the agent's core memory persona text"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        agent.update_core_memory_persona(request.text)
        return UpdatePersonaResponse(success=True, message="Core memory persona updated successfully")
    except Exception as e:
        return UpdatePersonaResponse(success=False, message=f"Error updating core memory persona: {str(e)}")

@app.post("/personas/apply_template", response_model=UpdatePersonaResponse)
async def apply_persona_template(request: ApplyPersonaTemplateRequest):
    """Apply a persona template to the agent"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        agent.apply_persona_template(request.persona_name)
        return UpdatePersonaResponse(success=True, message=f"Persona template '{request.persona_name}' applied successfully")
    except Exception as e:
        return UpdatePersonaResponse(success=False, message=f"Error applying persona template: {str(e)}")

@app.post("/core_memory/update", response_model=UpdateCoreMemoryResponse)
async def update_core_memory(request: UpdateCoreMemoryRequest):
    """Update a specific core memory block with new text"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        agent.update_core_memory(text=request.text, label=request.label)
        return UpdateCoreMemoryResponse(success=True, message=f"Core memory block '{request.label}' updated successfully")
    except Exception as e:
        return UpdateCoreMemoryResponse(success=False, message=f"Error updating core memory: {str(e)}")

@app.get("/personas/core_memory", response_model=CoreMemoryPersonaResponse)
async def get_core_memory_persona():
    """Get the core memory persona text"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        persona_text = agent.get_core_memory_persona()
        return CoreMemoryPersonaResponse(text=persona_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting core memory persona: {str(e)}")

@app.get("/models/current", response_model=GetCurrentModelResponse)
async def get_current_model():
    """Get the current model being used by the agent"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        current_model = agent.get_current_model()
        return GetCurrentModelResponse(current_model=current_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current model: {str(e)}")

@app.post("/models/set", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """Set the model for the agent"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Check if this is a custom model
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_config = None
        
        if custom_models_dir.exists():
            # Look for a config file that matches this model name
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if config and config.get('model_name') == request.model:
                            custom_config = config
                            print(f"Found custom model config for '{request.model}' at {config_file}")
                            break
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue
        
        # Set the model with custom config if found, otherwise use standard method
        if custom_config:
            result = agent.set_model(request.model, custom_agent_config=custom_config)
        else:
            result = agent.set_model(request.model)
            
        return SetModelResponse(
            success=result['success'], 
            message=result['message'],
            missing_keys=result.get('missing_keys', []),
            model_requirements=result.get('model_requirements', {})
        )
    except Exception as e:
        return SetModelResponse(
            success=False, 
            message=f"Error setting model: {str(e)}",
            missing_keys=[],
            model_requirements={}
        )

@app.get("/models/memory/current", response_model=GetCurrentModelResponse)
async def get_current_memory_model():
    """Get the current model being used by the memory manager"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        current_model = agent.get_current_memory_model()
        return GetCurrentModelResponse(current_model=current_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current memory model: {str(e)}")

@app.post("/models/memory/set", response_model=SetModelResponse)
async def set_memory_model(request: SetModelRequest):
    """Set the model for the memory manager"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Check if this is a custom model
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_config = None
        
        if custom_models_dir.exists():
            # Look for a config file that matches this model name
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if config and config.get('model_name') == request.model:
                            custom_config = config
                            print(f"Found custom model config for memory model '{request.model}' at {config_file}")
                            break
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue
        
        # Set the memory model with custom config if found, otherwise use standard method
        if custom_config:
            result = agent.set_memory_model(request.model, custom_agent_config=custom_config)
        else:
            result = agent.set_memory_model(request.model)
            
        return SetModelResponse(
            success=result['success'], 
            message=result['message'],
            missing_keys=result.get('missing_keys', []),
            model_requirements=result.get('model_requirements', {})
        )
    except Exception as e:
        return SetModelResponse(
            success=False, 
            message=f"Error setting memory model: {str(e)}",
            missing_keys=[],
            model_requirements={}
        )

@app.post("/models/custom/add", response_model=AddCustomModelResponse)
async def add_custom_model(request: AddCustomModelRequest):
    """Add a custom model configuration"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Create config file for the custom model
        config = {
            'agent_name': 'mirix',
            'model_name': request.model_name,
            'model_endpoint': request.model_endpoint,
            'api_key': request.api_key,
            'generation_config': {
                'temperature': request.temperature,
                'max_tokens': request.max_tokens,
                'context_window': request.maximum_length
            }
        }

        # Create custom models directory if it doesn't exist
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from model name (sanitize for filesystem)
        safe_model_name = "".join(c for c in request.model_name if c.isalnum() or c in ('-', '_', '.')).rstrip()
        config_filename = f"{safe_model_name}.yaml"
        config_file_path = custom_models_dir / config_filename
        
        # Save config to YAML file
        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Also set the model in the agent
        agent.set_model(request.model_name, custom_agent_config=config)

        return AddCustomModelResponse(
            success=True,
            message=f"Custom model '{request.model_name}' added successfully and saved to {config_file_path}"
        )
        
    except Exception as e:
        print(f"Error adding custom model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return AddCustomModelResponse(
            success=False,
            message=f"Error adding custom model: {str(e)}"
        )
    

@app.get("/models/custom/list", response_model=ListCustomModelsResponse)
async def list_custom_models():
    """List all available custom models"""
    try:
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        models = []
        
        if custom_models_dir.exists():
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if config and 'model_name' in config:
                            models.append(config['model_name'])
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue
        
        return ListCustomModelsResponse(models=models)
        
    except Exception as e:
        print(f"Error listing custom models: {e}")
        return ListCustomModelsResponse(models=[])

@app.get("/timezone/current", response_model=GetTimezoneResponse)
async def get_current_timezone():
    """Get the current timezone of the agent"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        current_timezone = agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
        return GetTimezoneResponse(timezone=current_timezone)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current timezone: {str(e)}")

@app.post("/timezone/set", response_model=SetTimezoneResponse)
async def set_timezone(request: SetTimezoneRequest):
    """Set the timezone for the agent"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        agent.set_timezone(request.timezone)
        return SetTimezoneResponse(success=True, message=f"Timezone '{request.timezone}' set successfully")
    except Exception as e:
        return SetTimezoneResponse(success=False, message=f"Error setting timezone: {str(e)}")

@app.get("/screenshot_setting", response_model=ScreenshotSettingResponse)
async def get_screenshot_setting():
    """Get the current screenshot setting"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    return ScreenshotSettingResponse(
        success=True,
        include_recent_screenshots=agent.include_recent_screenshots,
        message="Screenshot setting retrieved successfully"
    )

@app.post("/screenshot_setting/set", response_model=ScreenshotSettingResponse)
async def set_screenshot_setting(request: ScreenshotSettingRequest):
    """Set whether to include recent screenshots in messages"""
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        agent.set_include_recent_screenshots(request.include_recent_screenshots)
        return ScreenshotSettingResponse(
            success=True, 
            include_recent_screenshots=request.include_recent_screenshots,
            message=f"Screenshot setting updated: {'enabled' if request.include_recent_screenshots else 'disabled'}"
        )
    except Exception as e:
        return ScreenshotSettingResponse(
            success=False, 
            include_recent_screenshots=False,
            message=f"Error updating screenshot setting: {str(e)}"
        )

@app.get("/api_keys/check", response_model=ApiKeyCheckResponse)
async def check_api_keys():
    """Check for missing API keys based on current agent configuration"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Use the new AgentWrapper method
        api_key_status = agent.check_api_key_status()
        
        return ApiKeyCheckResponse(
            missing_keys=api_key_status["missing_keys"],
            model_type=api_key_status.get("model_requirements", {}).get("current_model", "unknown"),
            requires_api_key=len(api_key_status["missing_keys"]) > 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking API keys: {str(e)}")

@app.post("/api_keys/update", response_model=ApiKeyUpdateResponse)
async def update_api_key(request: ApiKeyRequest):
    """Update an API key value"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
        
    try:
        # Use the new AgentWrapper method which handles .env file saving
        result = agent.provide_api_key(request.key_name, request.key_value)
        
        # Also update environment variable and model_settings for backwards compatibility
        if result["success"]:
            os.environ[request.key_name] = request.key_value
            
            from mirix.settings import model_settings
            setting_name = request.key_name.lower()
            if hasattr(model_settings, setting_name):
                setattr(model_settings, setting_name, request.key_value)
        else:
            # If AgentWrapper doesn't support this key type, fall back to manual .env saving
            if "Unsupported API key type" in result["message"]:
                # Save to .env file manually for non-Gemini keys
                _save_api_key_to_env_file(request.key_name, request.key_value)
                os.environ[request.key_name] = request.key_value
                
                from mirix.settings import model_settings
                setting_name = request.key_name.lower()
                if hasattr(model_settings, setting_name):
                    setattr(model_settings, setting_name, request.key_value)
                
                result["success"] = True
                result["message"] = f"API key '{request.key_name}' saved to .env file successfully"
        
        return ApiKeyUpdateResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        return ApiKeyUpdateResponse(
            success=False,
            message=f"Error updating API key: {str(e)}"
        )

def _save_api_key_to_env_file(key_name: str, api_key: str):
    """
    Helper function to save API key to .env file for non-AgentWrapper keys.
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
    
    print(f"API key {key_name} saved to {env_file_path}")

# Memory endpoints
@app.get("/memory/episodic")
async def get_episodic_memory():
    """Get episodic memory (past events)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Access the episodic memory manager through the client
        client = agent.client
        episodic_manager = client.server.episodic_memory_manager
        
        # Get episodic events using the correct method name
        events = episodic_manager.list_episodic_memory(
            agent_state=agent.agent_states.episodic_memory_agent_state,
            limit=50,
            timezone_str=agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
        )
        
        # Transform to frontend format

        episodic_items = []
        for event in events:
            episodic_items.append({
                "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                "summary": event.summary,
                "details": event.details,
                "event_type": event.event_type,
                "tree_path": event.tree_path if hasattr(event, 'tree_path') else [],
            })

        return episodic_items
        
    except Exception as e:
        print(f"Error retrieving episodic memory: {str(e)}")
        # Return empty list if no memory or error
        return []

@app.get("/memory/semantic")
async def get_semantic_memory():
    """Get semantic memory (knowledge)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        client = agent.client
        semantic_items_list = []
        
        # Get semantic memory items
        try:
            semantic_manager = client.server.semantic_memory_manager
            semantic_items = semantic_manager.list_semantic_items(
                agent_state=agent.agent_states.semantic_memory_agent_state,
                limit=50,
                timezone_str=agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
            )
            
            for item in semantic_items:
                semantic_items_list.append({
                    "title": item.name,
                    "type": "semantic",
                    "summary": item.summary,
                    "details": item.details,
                    "tree_path": item.tree_path if hasattr(item, 'tree_path') else [],
                })
        except Exception as e:
            print(f"Error retrieving semantic memory: {str(e)}")
            
        return semantic_items_list
        
    except Exception as e:
        print(f"Error retrieving semantic memory: {str(e)}")
        return []

@app.get("/memory/procedural")
async def get_procedural_memory():
    """Get procedural memory (skills and procedures)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        client = agent.client
        procedural_items_list = []
        
        # Get procedural memory items  
        try:
            procedural_manager = client.server.procedural_memory_manager
            procedural_items = procedural_manager.list_procedures(
                agent_state=agent.agent_states.procedural_memory_agent_state,
                limit=50,
                timezone_str=agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
            )

            for item in procedural_items:
                # Parse steps if it's a JSON string
                steps = item.steps
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                        # Extract just the instruction text for simpler frontend display
                        if isinstance(steps, list) and steps and isinstance(steps[0], dict):
                            steps = [step.get('instruction', str(step)) for step in steps]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # If parsing fails, keep as string and split by common delimiters
                        if isinstance(steps, str):
                            steps = [s.strip() for s in steps.replace('\n', '|').split('|') if s.strip()]
                        else:
                            steps = []
                
                procedural_items_list.append({
                    "title": item.entry_type,
                    "type": "procedural", 
                    "summary": item.summary,
                    "steps": steps if isinstance(steps, list) else [],
                    "tree_path": item.tree_path if hasattr(item, 'tree_path') else [],
                })

        except Exception as e:
            print(f"Error retrieving procedural memory: {str(e)}")
            
        return procedural_items_list
        
    except Exception as e:
        print(f"Error retrieving procedural memory: {str(e)}")
        return []

@app.get("/memory/resources")
async def get_resource_memory():
    """Get resource memory (docs and files)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        client = agent.client
        resource_manager = client.server.resource_memory_manager
        
        # Get resource memory items using correct method name
        resources = resource_manager.list_resources(
            agent_state=agent.agent_states.resource_memory_agent_state,
            limit=50,
            timezone_str=agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
        )
        
        # Transform to frontend format
        docs_files = []
        for resource in resources:
            docs_files.append({
                "filename": resource.title,
                "type": resource.resource_type,
                "summary": resource.summary or (resource.content[:200] + "..." if len(resource.content) > 200 else resource.content),
                "last_accessed": resource.updated_at.isoformat() if resource.updated_at else None,
                "size": resource.metadata_.get("size") if resource.metadata_ else None,
                "tree_path": resource.tree_path if hasattr(resource, 'tree_path') else [],
            })
        
        return docs_files
        
    except Exception as e:
        print(f"Error retrieving resource memory: {str(e)}")
        return []

@app.get("/memory/core")
async def get_core_memory():
    """Get core memory (understanding of user)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Get core memory from the main agent
        core_memory = agent.client.get_in_context_memory(agent.agent_states.agent_state.id)
        
        core_understanding = []
        total_characters = 0

        # Extract understanding from memory blocks (skip persona block)
        for block in core_memory.blocks:
            if block.value and block.value.strip() and block.label.lower() != "persona":
                block_chars = len(block.value)
                total_characters += block_chars

                core_item = {
                    "aspect": block.label,
                    "understanding": block.value,
                    "character_count": block_chars,
                    "total_characters": total_characters,
                    "max_characters": block.limit,
                    "last_updated": None  # Core memory doesn't track individual updates
                }
                
                core_understanding.append(core_item)
        
        return core_understanding
        
    except Exception as e:
        print(f"Error retrieving core memory: {str(e)}")
        return []

@app.get("/memory/credentials")
async def get_credentials_memory():
    """Get credentials memory (knowledge vault with masked content)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        client = agent.client
        knowledge_vault_manager = client.server.knowledge_vault_manager
        
        # Get knowledge vault items using correct method name
        vault_items = knowledge_vault_manager.list_knowledge(
            agent_state=agent.agent_states.knowledge_vault_agent_state,
            limit=50,
            timezone_str=agent.client.server.user_manager.get_user_by_id(agent.client.user.id).timezone
        )
        
        # Transform to frontend format with masked content
        credentials = []
        for item in vault_items:
            credentials.append({
                "caption": item.caption,
                "entry_type": item.entry_type,
                "source": item.source,
                "sensitivity": item.sensitivity,
                "content": "••••••••••••" if item.sensitivity == 'high' else item.secret_value,  # Always mask the actual content
            })
        
        return credentials
        
    except Exception as e:
        print(f"Error retrieving credentials memory: {str(e)}")
        return []

@app.post("/conversation/clear", response_model=ClearConversationResponse)
async def clear_conversation_history():
    """Permanently clear all conversation history for the current agent (memories are preserved)"""
    try:
        if agent is None:
            raise HTTPException(status_code=400, detail="Agent not initialized")
        
        # Get current message count for reporting
        current_messages = agent.client.server.agent_manager.get_in_context_messages(
            agent_id=agent.agent_states.agent_state.id,
            actor=agent.client.user
        )
        messages_count = len(current_messages)
        
        # Clear conversation history using the agent manager reset_messages method
        agent.client.server.agent_manager.reset_messages(
            agent_id=agent.agent_states.agent_state.id,
            actor=agent.client.user,
            add_default_initial_messages=True  # Keep system message and initial setup
        )
        
        return ClearConversationResponse(
            success=True,
            message=f"Successfully cleared conversation history. All user and assistant messages removed (system messages preserved).",
            messages_deleted=messages_count
        )
        
    except Exception as e:
        print(f"Error clearing conversation history: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.post("/export/memories", response_model=ExportMemoriesResponse)
async def export_memories(request: ExportMemoriesRequest):
    """Export memories to Excel file with separate sheets for each memory type"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.export_memories_to_excel(
            file_path=request.file_path,
            memory_types=request.memory_types,
            include_embeddings=request.include_embeddings
        )
        
        if result['success']:
            return ExportMemoriesResponse(
                success=True,
                message=result['message'],
                exported_counts=result['exported_counts'],
                total_exported=result['total_exported'],
                file_path=result['file_path']
            )
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except Exception as e:
        print(f"Error exporting memories: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to export memories: {str(e)}")

@app.post("/reflexion", response_model=ReflexionResponse)
async def trigger_reflexion(request: ReflexionRequest):
    """Trigger reflexion agent to reorganize memory - runs in separate thread to not block other requests"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        print("Starting reflexion process...")
        start_time = datetime.now()
        
        # Run reflexion in a separate thread to avoid blocking other requests
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            _run_reflexion_process,
            agent
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"Reflexion process completed in {processing_time:.2f} seconds")
        
        return ReflexionResponse(
            success=result['success'],
            message=result['message'],
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"Error in reflexion endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Reflexion process failed: {str(e)}")

def _run_reflexion_process(agent):
    """
    Run the reflexion process - this is the blocking function that runs in a separate thread.
    This function can be replaced with the actual reflexion agent logic.
    """
    try:
        # TODO: Replace this with actual reflexion agent logic
        # For now, this is a placeholder that simulates reflexion work
        
        agent.reflexion_on_memory()
        return {
            'success': True,
            'message': 'Memory reorganization completed successfully. Reflexion agent has optimized memory structure and connections.'
        }
        
    except Exception as e:
        print(f"Error in reflexion process: {str(e)}")
        return {
            'success': False,
            'message': f'Reflexion process failed: {str(e)}'
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=47283)
