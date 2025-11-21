Raw Memory Data Flow & Processing Logic
This document outlines the data flow for screenshot capture, raw memory creation, and agent processing in the MIRIX system.

1. Screenshot Capture (Frontend)
Component: 
ScreenshotMonitor.js
 (React) / 
electron.js
 (Main Process)
Action:
ScreenshotMonitor
 triggers window.electronAPI.takeScreenshot().
electron.js
 captures the screen and saves it to /Volumes/Lexar/mirix/images/screenshot_<timestamp>.png.
electron.js
 returns the local file path to the frontend.
ScreenshotMonitor
 sends the path to the backend via POST /send_streaming_message.
2. Message Reception (Backend)
Component: 
fastapi_server.py
 -> Client.send_message
Action:
send_streaming_message_endpoint
 receives the request.
It calls agent.send_message (which delegates to client.send_message via 
AgentWrapper
).
client.send_message calls TemporaryMessageAccumulator.add_message.
3. Accumulation & Upload (Backend)
Component: TemporaryMessageAccumulator.add_message
Action:
Gemini Models:
upload_manager uploads the image to Google Cloud.
CRITICAL: original_local_paths are stored alongside the upload placeholders.
Non-Gemini Models:
Images are stored as local paths.
original_local_paths are populated from image_uris.
4. Memory Creation (Backend)
Component: TemporaryMessageAccumulator.absorb_content_into_memory -> 
_build_memory_message
Action:
Iterates through accumulated messages.
Resolution:
Retrieves local_file_path from original_local_paths.
Fallback: If missing, tries to use image_uri (if it's a string).
OCR:
Runs OCRUrlExtractor.extract_urls_and_text(local_file_path).
Failure Point: If local_file_path is invalid or missing, OCR fails.
Database Insertion:
Calls RawMemoryManager.insert_raw_memory.
Stores screenshot_path, ocr_text, captured_at, etc.
Failure Point: If this fails, no raw memory ID is generated.
5. Agent Processing (Backend)
Component: 
_send_to_memory_agents_separately
 -> EpisodicMemoryAgent
Action:
Constructs a message containing text, image URIs (Cloud or Base64), and Raw Memory IDs.
Sends this message to episodic_memory_agent (and others).
Episodic Memory Agent:
Receives the message.
Processes it (likely using episodic_memory_merge or similar tools).
Error: "function return was over limit" suggests a tool call returned too much data, potentially causing the agent to fail or truncate critical info.
Investigation Points
Stagnation: If 
insert_raw_memory
 fails (Step 4), no IDs are passed to Step 5. The logs should show "❌ Failed to store screenshot".
Agent Error: The "over limit" error in Step 5 might be a separate issue or a side effect of trying to process too much context, but it shouldn't stop Step 4 from happening first.