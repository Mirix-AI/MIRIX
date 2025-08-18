import os
import sys

import json
import torch
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
from copy import deepcopy
from google import genai
from google.genai import types

sys.path.append("../")

from mirix.agent import AgentWrapper as mirixAgent

# Define a function to encode images as base64 strings
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class AgentWrapper():

    def __init__(self, agent_name, system=None, load_agent_from=None, num_images_to_accumulate=False, model_name=None, config_path=None):

        self.agent_name = agent_name
        self.num_images_to_accumulate = num_images_to_accumulate # how many images to accumulate before querying the model
        if self.num_images_to_accumulate is not None:
            self.accumulated_image_uris = []
            self.message = ''

        if self.agent_name == 'gpt-long-context':
            self.model_name = model_name
            self.maximum_allowed_images = 250
            load_dotenv()
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.context = []

            if load_agent_from is not None:
                self.load_agent(load_agent_from)

        elif self.agent_name == 'gemini-long-context':
            self.model_name = model_name if model_name is not None else "gemini-2.5-flash"
            load_dotenv()
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required for gemini-long-context agent")
            
            self.context = []
            self.maximum_allowed_images = 3600

            if load_agent_from is not None:
                self.load_agent(load_agent_from)

        elif self.agent_name == 'siglip':
            from transformers import AutoProcessor, AutoModel
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").cuda()
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.client = OpenAI()
            self.image_embeddings = []
            self.text_descriptions = []
            self.text_embeddings = []
            self.image_paths = []
            self.topk = 50

            if load_agent_from is not None:
                self.load_agent(load_agent_from)

        elif self.agent_name == 'mirix':
            if load_agent_from is None:
                # Use provided config_path or default to the original hardcoded path
                config_file = config_path if config_path is not None else "../mirix/configs/mirix_gpt4.yaml"
                self.agent = mirixAgent(config_file)
            else:
                self.load_agent(load_agent_from, config_path)

    def prepare_before_asking_questions(self):
        if self.agent_name == 'gpt-long-context':
            return

        elif self.agent_name == 'gemini-long-context':
            return

        elif self.agent_name == 'mirix':
            self.agent.update_core_memory_persona("Is a helpful assistant that answers questions with extreme conciseness.\nIs persistent and tries to find the answerr using different queries and different search methods. Never uses unnecessary words or repeats the question in the answer. Always provides the shortest answer possible and tries to utter the fewest words possible.")

    def get_answer_from_retrieved_images_and_texts(self, retrieved_images=None, retrieved_texts=None, message=None, model_name='gpt-4o-mini', retrieved_captions=None):

        if model_name in ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4.1']:

            message = [
                {'type': 'text', "text": message},
            ]

            if retrieved_captions is not None:
                message.append({'type': 'text', "text": "Here are the retrieved image captions and texts from the memory base. Please answer my questions based on the retrieved images and texts."},)
            else:
                message.append({'type': 'text', "text": "Here are the retrieved images and texts from the memory base. Please answer my questions based on the retrieved images and texts."})

            if retrieved_images is not None:

                for idx, (image, text) in enumerate(zip(retrieved_images, retrieved_texts)):
                    if text != '':
                        message.extend([
                            {'type': 'text', "text": f"Result {idx}:\nTEXT: " + text + "\nIMAGE: "},
                            {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode_image(image)}", 'detail': 'auto'}}
                        ])
                    else:
                        message.extend([
                            {'type': 'text', "text": f"Result {idx}:\nIMAGE: "},
                            {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode_image(image)}", 'detail': 'auto'}}
                        ])
            else:

                assert retrieved_captions is not None
                for idx, (caption, text) in enumerate(zip(retrieved_captions, retrieved_texts)):
                    if text != '':
                        message.extend([
                            {'type': 'text', "text": f"Result {idx}:\nTEXT: " + text + "\nIMAGE CAPTION: "},
                            {'type': 'text', "text": caption}
                        ])
                    else:
                        message.extend([
                            {'type': 'text', "text": f"Result {idx}:\nIMAGE CAPTION: "},
                            {'type': 'text', "text": caption}
                        ])
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You will be given a list of images and texts. There exists a retriever that can retrieve the relevant images and texts from the memory base. You need to answer the questions based on the retrieved images and texts."},
                    {"role": "user", "content": message}
                ]
            )

            return response.choices[0].message.content

        elif model_name in ['gemini-2.0-flash']:
            
            parts = [
                {"text": message},
            ]

            if retrieved_images is not None:
                parts.append({"text": "Here are the retrieved images and texts from the memory base. Please answer my questions based on the retrieved images and texts."})
            
            if retrieved_captions is not None:
                parts.append({"text": "Here are the retrieved image captions and texts from the memory base. Please answer my questions based on the retrieved images and texts."})

            if retrieved_images is not None:

                for idx, (image, text) in enumerate(zip(retrieved_images, retrieved_texts)):
                    if text != '':
                        parts.append({"text": f"Result {idx}:\nTEXT: " + text + "\nIMAGE: "})
                    else:
                        parts.append({"text": f"Result {idx}:\nIMAGE: "})

                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",        # adjust if PNG or other format
                            "data": encode_image(image)       # base64 string of the image bytes
                        }
                    })
            
            else:

                assert retrieved_captions is not None
                for idx, (caption, text) in enumerate(zip(retrieved_captions, retrieved_texts)):
                    if text != '':
                        parts.append({"text": f"Result {idx}:\nTEXT: " + text + "\nIMAGE CAPTION: "})
                    else:
                        parts.append({"text": f"Result {idx}:\nIMAGE CAPTION: "})

                    parts.append({'text': caption})

            payload = {
                "contents": [
                    {
                        'role': 'user',
                        'parts': [{'text': 'You will be given a list of images and texts. There exists a retriever that can retrieve the relevant images and texts from the memory base. You need to answer the questions based on the retrieved images and texts.'}]   
                    },
                    {
                        'role': 'user',
                        'parts': parts
                    }
                ]
            }

            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent"
            params = {"key": "AIzaSyAPdbLNkHSDcD8gd6wgB0KqjXp3yG5YRGs" }  # API key as query param
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, params=params, headers=headers, json=payload)
            # Check the response
            if response.status_code == 200:
                result = response.json()
                # The API returns a JSON with a 'candidates' list. 
                # The generated caption text is typically in result["candidates"][0]["content"]["parts"][0]["text"]
                try:
                    response = result["candidates"][0]["content"]["parts"][0]["text"]
                except:
                    response = 'error'
            else:
                # Handle error
                print(f"Error: {response.status_code}")
                print(response.text)
                response = None
            
            return response
        else:
            raise NotImplementedError("Only gpt-4o-mini is supported for now.")


    def send_message(self, message=None, image_uris=None, memorizing=False, timestamp=None):
        
        if self.agent_name == 'gpt-long-context':
            if memorizing:
                # Store message and images in context for later use
                context_entry = {
                    'message': message,
                    'image_uris': image_uris if image_uris else []
                }
                self.context.append(context_entry)
            else:
                # Collect all image URIs from context and current request
                all_image_uris = []
                all_texts = []
                
                # Add images and texts from context (in chronological order)
                for ctx_entry in self.context:
                    for img_uri in ctx_entry['image_uris']:
                        all_image_uris.append(img_uri)
                        all_texts.append(ctx_entry['message'] if ctx_entry['message'] else '')
                
                # Add current images
                if image_uris:
                    for img_uri in image_uris:
                        all_image_uris.append(img_uri)
                        all_texts.append('')  # No associated text for current images
                
                # Take only the latest images up to the maximum limit
                if len(all_image_uris) > self.maximum_allowed_images:
                    selected_image_uris = all_image_uris[-self.maximum_allowed_images:]
                    selected_texts = all_texts[-self.maximum_allowed_images:]
                    print(f"Warning: Using latest {self.maximum_allowed_images} images out of {len(all_image_uris)} total images.")
                else:
                    selected_image_uris = all_image_uris
                    selected_texts = all_texts
                
                # Build message content for OpenAI API
                message_content = [
                    {'type': 'text', 'text': message if message else "Please analyze the provided images and texts."}
                ]
                
                # Add context information if we have images
                if selected_image_uris:
                    message_content.append({
                        'type': 'text', 
                        'text': "Here are the images and texts from the conversation history. Please answer my questions based on these images and texts."
                    })
                    
                    # Add images and associated texts
                    for idx, (image_uri, text) in enumerate(zip(selected_image_uris, selected_texts)):
                        if text != '':
                            message_content.append({
                                'type': 'text', 
                                'text': f"Context {idx}:\nTEXT: {text}\nIMAGE: "
                            })
                        else:
                            message_content.append({
                                'type': 'text', 
                                'text': f"Context {idx}:\nIMAGE: "
                            })
                        
                        # Add the image
                        message_content.append({
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/png;base64,{encode_image(image_uri)}",
                                'detail': 'auto'
                            }
                        })
                
                # Add text-only context messages
                text_context = []
                for ctx_entry in self.context:
                    if ctx_entry['message'] and not ctx_entry['image_uris']:
                        text_context.append(ctx_entry['message'])
                
                if text_context:
                    context_text = "\n\n".join(text_context)
                    if selected_image_uris:
                        message_content.append({
                            'type': 'text',
                            'text': f"\nAdditional text context:\n{context_text}"
                        })
                    else:
                        # If no images, include text context at the beginning
                        message_content.insert(1, {
                            'type': 'text',
                            'text': f"Previous conversation context:\n{context_text}\n\nCurrent question: "
                        })
                
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant that can answer questions related to the texts and images.'},
                    {'role': 'user', 'content': message_content}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                return response.choices[0].message.content

        elif self.agent_name == 'gemini-long-context':
            if memorizing:
                # Store message and images in context for later use
                context_entry = {
                    'message': message,
                    'image_uris': image_uris if image_uris else []
                }
                self.context.append(context_entry)
            else:
                # Collect all image URIs from context and current request
                all_image_uris = []
                all_texts = []
                
                # Add images and texts from context (in chronological order)
                for ctx_entry in self.context:
                    for img_uri in ctx_entry['image_uris']:
                        all_image_uris.append(img_uri)
                        all_texts.append(ctx_entry['message'] if ctx_entry['message'] else '')
                
                # Add current images
                if image_uris:
                    for img_uri in image_uris:
                        all_image_uris.append(img_uri)
                        all_texts.append('')  # No associated text for current images
                
                # Take only the latest images up to the maximum limit
                if len(all_image_uris) > self.maximum_allowed_images:
                    selected_image_uris = all_image_uris[-self.maximum_allowed_images:]
                    selected_texts = all_texts[-self.maximum_allowed_images:]
                    print(f"Warning: Using latest {self.maximum_allowed_images} images out of {len(all_image_uris)} total images.")
                else:
                    selected_image_uris = all_image_uris
                    selected_texts = all_texts
                
                # Use the same logic as get_answer_from_retrieved_images_and_texts for gemini-2.0-flash
                parts = [
                    {"text": message if message else "Please analyze the provided images and texts."},
                ]
                
                if selected_image_uris:
                    parts.append({"text": "Here are the images and texts from the conversation history. Please answer my questions based on these images and texts."})
                
                    for idx, (image, text) in enumerate(zip(selected_image_uris, selected_texts)):
                        if text != '':
                            parts.append({"text": f"Context {idx}:\nTEXT: " + text + "\nIMAGE: "})
                        else:
                            parts.append({"text": f"Context {idx}:\nIMAGE: "})

                        try:
                            # Resize image to 256x256 without overwriting original
                            pil_image = Image.open(image)
                            resized_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
                        except:
                            continue

                        # Convert resized image to base64
                        import io
                        img_buffer = io.BytesIO()
                        # Save as JPEG to reduce size
                        resized_image.convert('RGB').save(img_buffer, format='JPEG', quality=85)
                        img_buffer.seek(0)
                        resized_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": resized_base64
                            }
                        })

                payload = {
                    "contents": [
                        {
                            'role': 'user',
                            'parts': [{'text': 'You are a helpful assistant that can answer questions related to the texts and images.'}]   
                        },
                        {
                            'role': 'user',
                            'parts': parts
                        }
                    ]
                }

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                params = {"key": self.api_key}  # Use the stored API key
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, params=params, headers=headers, json=payload)
                
                # Check the response
                if response.status_code == 200:
                    result = response.json()
                    # The API returns a JSON with a 'candidates' list. 
                    # The generated caption text is typically in result["candidates"][0]["content"]["parts"][0]["text"]
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return response_text
                else:
                    # Handle error
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    return f"Error: {response.status_code} - {response.text}"

        elif self.agent_name == 'mirix':
            
            if memorizing:

                if message is None:
                    # ScreenshotVQA does not need to absorb content immediately
                    force_absorb_content = False
                else:
                    # LOCOMO needs to absorb content immediately
                    force_absorb_content = True

                response = self.agent.send_message(
                    message=message,
                    image_uris=image_uris,
                    memorizing=True,
                    force_absorb_content=force_absorb_content,
                    delete_after_upload=False,
                    async_upload=False,
                    specific_timestamps=timestamp
                )

                return response

                # updated_episodic_memory = self.agent.client.server.episodic_memory_manager.get_most_recently_updated_event()
                # updated_semantic_memory = self.agent.client.server.semantic_memory_manager.get_most_recently_updated_item()
                # if updated_episodic_memory is None or updated_semantic_memory is None:
                #     response = None
                # else:
                #     response = {
                #         'updated_episodic_memory': {
                #             'summary': updated_episodic_memory[0].summary,
                #             'details': updated_episodic_memory[0].details,
                #             'occurred_at': updated_episodic_memory[0].occurred_at.strftime('%Y-%m-%d %H:%M:%S'),
                #         },
                #         'updated_semantic_memory': {
                #             'name': updated_semantic_memory[0].name,
                #             'summary': updated_semantic_memory[0].summary,
                #             'details': updated_semantic_memory[0].details,
                #         }
                #     }
                # return response
            else:
                response = self.agent.send_message(
                    message=message,
                    memorizing=memorizing,
                    delete_after_upload=False,
                )
                return response

        elif self.agent_name == 'siglip':

            if memorizing: 
                try:
                    for image_uri in image_uris:
                        image = Image.open(image_uri)
                        inputs = self.processor(images=image, return_tensors="pt")
                        inputs['pixel_values'] = inputs['pixel_values'].cuda()
                        with torch.no_grad():
                            image_features = self.model.get_image_features(**inputs)
                        if isinstance(self.image_embeddings, list):
                            self.image_embeddings.append(image_features.cpu())
                        else:
                            self.image_embeddings = torch.cat([self.image_embeddings, image_features.cpu()], dim=0)
                        self.image_paths.append(image_uri)
                except:
                    print("Error when processing images:", image_uris)
            
            else:

                text_inputs = self.processor(message, return_tensors="pt")
                text_inputs['input_ids'] = text_inputs['input_ids'].cuda()
                if len(text_inputs['input_ids'][0]) > 64:
                    text_inputs['input_ids'] = text_inputs['input_ids'][:, :64]
                    
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                    sims = (text_features.cpu() @ self.image_embeddings.transpose(0, 1))
                    indices = torch.topk(sims, k=self.topk).indices
                
                retrieved_images = [self.image_paths[x] for x in indices[0].tolist()]
                # retrieved_texts = [self.messages[x] for x in indices[0].tolist()]
                retrieved_texts = [''] * len(retrieved_images)

                response = self.get_answer_from_retrieved_images_and_texts(retrieved_images=retrieved_images, retrieved_texts=retrieved_texts, message=message, model_name='gemini-2.0-flash')
                return response

    def save_agent(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.agent_name == 'gpt-long-context':
            with open(f"{folder}/context.json", "w") as f:
                json.dump(self.context, f, indent=2)

        elif self.agent_name == 'gemini-long-context':
            with open(f"{folder}/context.json", "w") as f:
                json.dump(self.context, f, indent=2)

        elif self.agent_name == 'mirix':

            from mirix.settings import settings
            if settings.mirix_pg_uri_no_default:
                self.agent.save_agent(folder)
            else:
                import shutil
                shutil.copyfile(os.path.expanduser("~/.mirix/sqlite.db"), f"{folder}/sqlite.db")
            
        elif self.agent_name == 'siglip':
            if isinstance(self.image_embeddings, list):
                self.image_embeddings = torch.cat(self.image_embeddings, dim=0)
            torch.save(self.image_embeddings, f"{folder}/image_embeddings.pt")
            with open(f"{folder}/image_paths.json", "w") as f:
                json.dump(self.image_paths, f, indent=2)
        
    def load_agent(self, folder, config_path=None):
        if self.agent_name == 'gpt-long-context':
            with open(f"{folder}/context.json", "r") as f:
                self.context = json.load(f)

        elif self.agent_name == 'gemini-long-context':
            with open(f"{folder}/context.json", "r") as f:
                self.context = json.load(f)

        elif self.agent_name == 'mirix':

            from mirix.settings import settings
            if settings.mirix_pg_uri_no_default:
                config_file = config_path if config_path is not None else "../configs/mirix_gpt4.yaml"
                self.agent = mirixAgent(config_file, load_from=folder)

            else:
                import shutil
                import time
                
                # Close any existing agent connection first
                if hasattr(self, 'agent') and self.agent is not None:
                    try:
                        # Try to properly close any existing connections
                        if hasattr(self.agent, 'client') and hasattr(self.agent.client, 'server'):
                            # Close any open database sessions
                            del self.agent.client.server
                        del self.agent
                    except:
                        pass
                
                # Wait a moment for any file handles to be released
                time.sleep(0.1)
                
                # Copy the database file with retry mechanism
                target_db = os.path.expanduser("~/.mirix/sqlite.db")
                source_db = f"{folder}/sqlite.db"

                shutil.copyfile(source_db, target_db)
                # set the database to writable
                os.chmod(target_db, 0o666)
                
                # Now create the mirix agent with the provided config or default
                config_file = config_path if config_path is not None else "../configs/mirix_gpt4.yaml"
                self.agent = mirixAgent(config_file)
        
        elif self.agent_name == 'siglip':
            self.image_embeddings = torch.load(f"{folder}/image_embeddings.pt")
            self.image_paths = json.load(open(f"{folder}/image_paths.json"))