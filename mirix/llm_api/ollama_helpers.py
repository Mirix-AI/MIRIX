
def _preprocess_ollama_messages(messages, put_inner_thoughts_in_kwargs=False):
    processed_messages = []
    for m in messages:
        msg_dict = m.to_openai_dict(
            put_inner_thoughts_in_kwargs=put_inner_thoughts_in_kwargs
        )
        # Fix ImageContent format for OpenAI/Ollama compatibility
        if msg_dict.get("content") and isinstance(msg_dict["content"], list):
            new_content = []
            for item in msg_dict["content"]:
                if isinstance(item, dict):
                    # Check for invalid image_id format and convert to valid image_url
                    if item.get("type") == "image_url" and "image_id" in item:
                        # Construct a valid OpenAI image_url object
                        # Since we don't have easy access to the image file here to base64 encode it,
                        # and passing a file path usually doesn't work for remote APIs (unless Ollama runs locally),
                        # we will try to provide a text description if we can't properly link it,
                        # OR if we assume Ollama is local, maybe we can pass the path if we had it.
                        # But we only have image_id.
                        # However, for now, to prevent 400 errors, we must fix the structure.
                        # We will convert it to a text "placeholder" to avoid crashing until we can implement full image resolution.
                        # Using a text placeholder is safer than sending invalid JSON that crashes the request.
                        new_content.append({
                            "type": "text", 
                            "text": f"[Image ID: {item['image_id']}] (Image skipped due to resolution limit)"
                        })
                    elif item.get("type") == "image_url" and "image_url" not in item:
                         # Fallback for any other malformed image_url
                        new_content.append({
                            "type": "text", 
                            "text": "[Image: malformed content]"
                        })
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            msg_dict["content"] = new_content
        processed_messages.append(msg_dict)
    return processed_messages
