import base64
import mimetypes
from typing import List


def _preprocess_ollama_messages(messages, put_inner_thoughts_in_kwargs=False):
    """
    Preprocess messages for Ollama compatibility.
    Converts image_id references to base64-encoded images that Ollama can understand.
    """
    from mirix.services.file_manager import FileManager
    
    file_manager = FileManager()
    processed_messages = []
    
    for m in messages:
        msg_dict = m.to_openai_dict(
            put_inner_thoughts_in_kwargs=put_inner_thoughts_in_kwargs
        )
        
        # Fix ImageContent format for Ollama compatibility
        if msg_dict.get("content") and isinstance(msg_dict["content"], list):
            new_content = []
            for item in msg_dict["content"]:
                if isinstance(item, dict):
                    # Check for image_url with image_id (needs conversion to base64)
                    if item.get("type") == "image_url" and "image_id" in item:
                        try:
                            # Get file metadata from database
                            file = file_manager.get_file_metadata_by_id(item["image_id"])
                            
                            if file.source_url is not None:
                                # Use URL directly if available
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": file.source_url
                                    }
                                })
                            elif file.file_path is not None:
                                # Convert local file to base64
                                mime_type, _ = mimetypes.guess_type(file.file_path)
                                if mime_type is None or not mime_type.startswith("image/"):
                                    mime_type = "image/jpeg"  # Default fallback
                                
                                with open(file.file_path, "rb") as img_file:
                                    base64_data = base64.b64encode(img_file.read()).decode("utf-8")
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_data}"
                                        }
                                    })
                            else:
                                # Fallback to text placeholder if no file path available
                                new_content.append({
                                    "type": "text",
                                    "text": f"[Image ID: {item['image_id']}] (Image file not found)"
                                })
                        except Exception as e:
                            # If we can't load the image, use a text placeholder
                            new_content.append({
                                "type": "text",
                                "text": f"[Image ID: {item['image_id']}] (Error loading image: {str(e)})"
                            })
                    
                    elif item.get("type") == "image_url" and "image_url" not in item:
                        # Fallback for any other malformed image_url
                        new_content.append({
                            "type": "text",
                            "text": "[Image: malformed content]"
                        })
                    else:
                        # Keep other content types as-is
                        new_content.append(item)
                else:
                    new_content.append(item)
            
            msg_dict["content"] = new_content
        
        processed_messages.append(msg_dict)
    
    return processed_messages
