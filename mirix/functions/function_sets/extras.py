import os
import uuid
from typing import Optional

from mirix.constants import MESSAGE_CHATGPT_FUNCTION_MODEL, MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE
from mirix.llm_api.llm_api_tools import create
from mirix.schemas.message import Message
from mirix.utils import json_dumps, json_loads


# def message_chatgpt(self, message: str):
#     """
#     Send a message to a more basic AI, ChatGPT. A useful resource for asking questions. ChatGPT does not retain memory of previous interactions.

#     Args:
#         message (str): Message to send ChatGPT. Phrase your message as a full English sentence.

#     Returns:
#         str: Reply message from ChatGPT
#     """
#     dummy_user_id = uuid.uuid4()
#     dummy_agent_id = uuid.uuid4()
#     message_sequence = [
#         Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="system", text=MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE),
#         Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="user", text=str(message)),
#     ]
#     # TODO: this will error without an LLMConfig
#     response = create(
#         model=MESSAGE_CHATGPT_FUNCTION_MODEL,
#         messages=message_sequence,
#     )

#     reply = response.choices[0].message.content
#     return reply


# def read_from_text_file(self, filename: str, line_start: int, num_lines: Optional[int] = 1):
#     """
#     Read lines from a text file.

#     Args:
#         filename (str): The name of the file to read.
#         line_start (int): Line to start reading from.
#         num_lines (Optional[int]): How many lines to read (defaults to 1).

#     Returns:
#         str: Text read from the file
#     """
#     max_chars = 500
#     trunc_message = True
#     if not os.path.exists(filename):
#         raise FileNotFoundError(f"The file '{filename}' does not exist.")

#     if line_start < 1 or num_lines < 1:
#         raise ValueError("Both line_start and num_lines must be positive integers.")

#     lines = []
#     chars_read = 0
#     with open(filename, "r", encoding="utf-8") as file:
#         for current_line_number, line in enumerate(file, start=1):
#             if line_start <= current_line_number < line_start + num_lines:
#                 chars_to_add = len(line)
#                 if max_chars is not None and chars_read + chars_to_add > max_chars:
#                     # If adding this line exceeds MAX_CHARS, truncate the line if needed and stop reading further.
#                     excess_chars = (chars_read + chars_to_add) - max_chars
#                     lines.append(line[:-excess_chars].rstrip("\n"))
#                     if trunc_message:
#                         lines.append(f"[SYSTEM ALERT - max chars ({max_chars}) reached during file read]")
#                     break
#                 else:
#                     lines.append(line.rstrip("\n"))
#                     chars_read += chars_to_add
#             if current_line_number >= line_start + num_lines - 1:
#                 break

#     return "\n".join(lines)


# def append_to_text_file(self, filename: str, content: str):
#     """
#     Append to a text file.

#     Args:
#         filename (str): The name of the file to append to.
#         content (str): Content to append to the file.

#     Returns:
#         Optional[str]: None is always returned as this function does not produce a response.
#     """
#     if not os.path.exists(filename):
#         raise FileNotFoundError(f"The file '{filename}' does not exist.")

#     with open(filename, "a", encoding="utf-8") as file:
#         file.write(content + "\n")


# def http_request(self, method: str, url: str, payload_json: Optional[str] = None):
#     """
#     Generates an HTTP request and returns the response.

#     Args:
#         method (str): The HTTP method (e.g., 'GET', 'POST').
#         url (str): The URL for the request.
#         payload_json (Optional[str]): A JSON string representing the request payload.

#     Returns:
#         dict: The response from the HTTP request.
#     """
#     try:
#         headers = {"Content-Type": "application/json"}

#         # For GET requests, ignore the payload
#         if method.upper() == "GET":
#             print(f"[HTTP] launching GET request to {url}")
#             response = requests.get(url, headers=headers)
#         else:
#             # Validate and convert the payload for other types of requests
#             if payload_json:
#                 payload = json_loads(payload_json)
#             else:
#                 payload = {}
#             print(f"[HTTP] launching {method} request to {url}, payload=\n{json_dumps(payload, indent=2)}")
#             response = requests.request(method, url, json=payload, headers=headers)

#         return {"status_code": response.status_code, "headers": dict(response.headers), "body": response.text}
#     except Exception as e:
#         return {"error": str(e)}


def fetch_and_read_pdf(self, url: str, max_pages: Optional[int] = 10):
    """
    Fetch and read a PDF file from a URL, extracting its text content.
    
    Args:
        url (str): The URL of the PDF file to fetch and read.
        max_pages (Optional[int]): Maximum number of pages to read (defaults to 10).
    
    Returns:
        str: Extracted text content from the PDF file.
    """
    try:
        print(f"[PDF_READER] Fetching PDF from: {url}")
        
        # Import libraries for PDF processing
        try:
            import requests
            import io
            from PyPDF2 import PdfReader
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "required library"
            return f"PDF processing library not available. Please install '{missing_lib}' to enable PDF reading functionality."
        
        # Fetch the PDF file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return f"Failed to fetch PDF: HTTP status {response.status_code}"
        
        # Check if the response is actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            return f"The URL does not appear to point to a PDF file. Content-Type: {content_type}"
        
        print(f"[PDF_READER] PDF fetched successfully, size: {len(response.content)} bytes")
        
        # Read the PDF content
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        
        num_pages = len(pdf_reader.pages)
        max_pages = min(max_pages or 10, num_pages)
        
        print(f"[PDF_READER] PDF has {num_pages} pages, reading first {max_pages} pages")
        
        extracted_text = []
        for page_num in range(max_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    extracted_text.append(f"--- Page {page_num + 1} ---")
                    extracted_text.append(text.strip())
                    extracted_text.append("")  # Empty line for separation
            except Exception as page_error:
                extracted_text.append(f"--- Page {page_num + 1} (Error reading page) ---")
                extracted_text.append(f"Error: {str(page_error)}")
                extracted_text.append("")
        
        if not extracted_text:
            return f"PDF was fetched successfully but no readable text content was found. The PDF may contain only images or be encrypted."
        
        full_text = "\n".join(extracted_text)
        
        # Limit output length for practical use
        if len(full_text) > 10000:
            full_text = full_text[:10000] + f"\n\n[Text truncated - showing first 10,000 characters of {len(full_text)} total]"
        
        print(f"[PDF_READER] Successfully extracted text from {max_pages} pages")
        return full_text
        
    except Exception as e:
        print(f"[PDF_READER] Error: {e}")
        return f"Error reading PDF: {str(e)}"


def web_search(self, query: str, num_results: Optional[int] = 5):
    """
    Search the web for information using DuckDuckGo comprehensive search.
    
    Args:
        query (str): The search query to look for on the web.
        num_results (Optional[int]): Maximum number of results to return (defaults to 5, max 10).
    
    Returns:
        str: Search results formatted as text with titles, URLs, and descriptions.
    """
    try:
        # Limit num_results to reasonable bounds
        num_results = min(max(1, num_results or 5), 10)
        
        print(f"[WEB_SEARCH] Searching for: {query}")
        
        # Import ddgs here to avoid import errors if not available
        try:
            from ddgs import DDGS
        except ImportError:
            return f"Web search library not available. Please install 'ddgs' library to enable web search functionality."
        
        # Use DDGS for comprehensive web search results
        ddgs = DDGS()
        search_results = ddgs.text(query, max_results=num_results)
        
        if not search_results:
            return f"No results found for '{query}'. Try rephrasing your search terms or using more specific keywords."
        
        # Format results for display
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            # Limit description length for readability
            if len(body) > 200:
                body = body[:197] + "..."
            
            formatted_results.append(f"{i}. {title}")
            if body:
                formatted_results.append(f"   {body}")
            if href:
                formatted_results.append(f"   URL: {href}")
            formatted_results.append("")  # Empty line for separation
        
        # Remove the last empty line
        if formatted_results and formatted_results[-1] == "":
            formatted_results.pop()
        
        result_text = "\n".join(formatted_results)
        print(f"[WEB_SEARCH] Found {len(search_results)} results")
        
        return result_text
        
    except Exception as e:
        print(f"[WEB_SEARCH] Error: {e}")
        return f"Search error: {str(e)}. Try rephrasing your search query."
