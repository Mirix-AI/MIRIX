"""
Gmail MCP Client - Integrates Gmail functionality with MCP system
Based on the original Gmail implementation and adapted for MCP compatibility
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import base64
import mimetypes

from .base_client import BaseMCPClient
from .types import BaseServerConfig, MCPTool, MCPServerType
# Import authenticate_gmail locally to avoid circular import
def authenticate_gmail_local(client_id: str, client_secret: str, token_file: str = None) -> dict:
    """
    Authenticate with Gmail using OAuth2 with a local server to catch the callback.
    Local copy to avoid circular imports.
    """
    import os
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    
    # Gmail API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
    
    try:
        creds = None
        token_path = token_file or os.path.expanduser("~/.mirix/gmail_token.json")
        
        # Load existing token if available
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except Exception:
                pass
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            
            if not creds:
                # Create the client config
                client_config = {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "redirect_uris": [
                            "http://localhost:8080/",
                            "http://localhost:8081/",
                            "http://localhost:8082/"
                        ]
                    }
                }
                
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                
                # Try specific ports that match redirect URIs
                for port in [8080, 8081, 8082]:
                    try:
                        # Request offline access to get refresh token
                        creds = flow.run_local_server(port=port, open_browser=True, access_type='offline', prompt='consent')
                        break
                    except OSError:
                        if port == 8082:
                            # If all ports fail, use automatic port selection
                            creds = flow.run_local_server(port=0, open_browser=True, access_type='offline', prompt='consent')
            
            # Save the credentials for the next run
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        return {
            "success": True,
            "message": "Gmail authenticated successfully",
            "token_path": token_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

logger = logging.getLogger(__name__)

class GmailMCPClient(BaseMCPClient):
    """Gmail MCP Client that provides Gmail functionality through MCP interface"""
    
    def __init__(self, server_config):  # Use generic type to avoid circular import
        super().__init__(server_config)
        self.gmail_service = None
        self.credentials = None
        
    def _initialize_connection(self, server_config, timeout: float) -> bool:
        """Initialize Gmail connection using OAuth"""
        try:
            token_file = server_config.token_file or os.path.expanduser("~/.mirix/gmail_token.json")
            
            # Authenticate with Gmail
            auth_result = authenticate_gmail_local(
                server_config.client_id,
                server_config.client_secret,
                token_file
            )
            
            if not auth_result["success"]:
                logger.error(f"Gmail authentication failed: {auth_result.get('error', 'Unknown error')}")
                return False
            
            # Load credentials and build service
            if os.path.exists(token_file):
                try:
                    self.credentials = Credentials.from_authorized_user_file(
                        token_file,
                        ['https://www.googleapis.com/auth/gmail.readonly',
                         'https://www.googleapis.com/auth/gmail.send',
                         'https://www.googleapis.com/auth/gmail.modify']
                    )
                    
                    # Check if credentials have refresh token
                    if not hasattr(self.credentials, 'refresh_token') or not self.credentials.refresh_token:
                        logger.warning("Gmail token missing refresh_token. Removing invalid token file.")
                        os.remove(token_file)
                        # Retry authentication with fresh flow
                        auth_result = authenticate_gmail_local(
                            server_config.client_id,
                            server_config.client_secret,
                            token_file
                        )
                        if auth_result["success"] and os.path.exists(token_file):
                            self.credentials = Credentials.from_authorized_user_file(
                                token_file,
                                ['https://www.googleapis.com/auth/gmail.readonly',
                                 'https://www.googleapis.com/auth/gmail.send',
                                 'https://www.googleapis.com/auth/gmail.modify']
                            )
                        else:
                            logger.error("Failed to re-authenticate after removing invalid token")
                            return False
                    
                    self.gmail_service = build('gmail', 'v1', credentials=self.credentials)
                    logger.info("Gmail service initialized successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to load Gmail credentials: {str(e)}")
                    return False
            else:
                logger.error("Gmail token file not found after authentication")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gmail connection: {str(e)}")
            return False
    
    def list_tools(self, timeout: float = 10.0) -> List[MCPTool]:
        """Return list of available Gmail tools"""
        return [
            MCPTool(
                name="gmail_send_email",
                description="Send an email using Gmail",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body (plain text)"
                        },
                        "cc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "CC recipients (optional)"
                        },
                        "bcc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "BCC recipients (optional)"
                        },
                        "html_body": {
                            "type": "string",
                            "description": "HTML version of email body (optional)"
                        },
                        "attachments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to attach (optional)"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            ),
            MCPTool(
                name="gmail_read_emails",
                description="Read emails from Gmail inbox",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gmail search query (optional, e.g., 'is:unread')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of emails to retrieve (default: 10)",
                            "default": 10
                        }
                    }
                }
            ),
            MCPTool(
                name="gmail_get_email",
                description="Get a specific email by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Gmail message ID"
                        }
                    },
                    "required": ["email_id"]
                }
            )
        ]
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], timeout: float = 60.0) -> Tuple[str, bool]:
        """Execute a Gmail tool"""
        self._check_initialized()
        
        # Ensure Gmail service is available before executing tools
        if not self._ensure_gmail_service():
            return "Gmail authentication required. Please run the Gmail connection process.", True
        
        try:
            if tool_name == "gmail_send_email":
                return self._send_email(tool_args)
            elif tool_name == "gmail_read_emails":
                return self._read_emails(tool_args)
            elif tool_name == "gmail_get_email":
                return self._get_email(tool_args)
            else:
                return f"Unknown tool: {tool_name}", True
        except Exception as e:
            logger.error(f"Error executing Gmail tool {tool_name}: {str(e)}")
            return f"Error executing tool: {str(e)}", True
    
    def _ensure_gmail_service(self) -> bool:
        """Ensure Gmail service is available, try to authenticate if not"""
        if self.gmail_service is not None:
            return True
            
        try:
            # Try to initialize connection using the stored config
            success = self._initialize_connection(self.server_config, timeout=30.0)
            if success and self.gmail_service is not None:
                logger.info("Gmail service established successfully")
                return True
            else:
                logger.warning("Gmail service initialization failed")
                return False
        except Exception as e:
            logger.error(f"Failed to ensure Gmail service: {str(e)}")
            return False
    
    def _send_email(self, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Send an email using Gmail API"""
        try:
            to = args["to"]
            subject = args["subject"]
            body = args["body"]
            cc = args.get("cc", [])
            bcc = args.get("bcc", [])
            html_body = args.get("html_body")
            attachments = args.get("attachments", [])
            
            # Create email message
            message = self._create_message(to, subject, body, cc, bcc, attachments, html_body)
            
            # Send the message
            result = self.gmail_service.users().messages().send(userId='me', body=message).execute()
            
            return f"Email sent successfully! Message ID: {result['id']}", False
            
        except Exception as e:
            return f"Failed to send email: {str(e)}", True
    
    def _read_emails(self, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Read emails from Gmail"""
        try:
            query = args.get("query", "")
            max_results = args.get("max_results", 10)
            
            # Get list of messages
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return "No emails found", False
            
            # Get details for each message
            email_details = []
            for message in messages:
                msg = self.gmail_service.users().messages().get(
                    userId='me',
                    id=message['id']
                ).execute()
                
                headers = msg['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                
                email_details.append({
                    'id': message['id'],
                    'subject': subject,
                    'from': sender,
                    'date': date
                })
            
            return json.dumps(email_details, indent=2), False
            
        except Exception as e:
            return f"Failed to read emails: {str(e)}", True
    
    def _get_email(self, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Get a specific email by ID"""
        try:
            email_id = args["email_id"]
            
            # Get the message
            message = self.gmail_service.users().messages().get(
                userId='me',
                id=email_id
            ).execute()
            
            # Extract email content
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
            
            # Get body
            body = self._extract_message_body(message['payload'])
            
            email_data = {
                'id': email_id,
                'subject': subject,
                'from': sender,
                'date': date,
                'body': body
            }
            
            return json.dumps(email_data, indent=2), False
            
        except Exception as e:
            return f"Failed to get email: {str(e)}", True
    
    def _create_message(self, to: str, subject: str, body: str, 
                       cc: List[str] = None, bcc: List[str] = None,
                       attachments: List[str] = None, html_body: str = None) -> dict:
        """Create an email message (based on original implementation)"""
        # Create message container
        if html_body or attachments:
            message = MIMEMultipart('alternative' if html_body else 'mixed')
        else:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
        
        message['to'] = to
        message['subject'] = subject
        
        if cc:
            message['cc'] = ', '.join(cc)
        if bcc:
            message['bcc'] = ', '.join(bcc)
        
        # Add text body
        text_part = MIMEText(body, 'plain')
        message.attach(text_part)
        
        # Add HTML body if provided
        if html_body:
            html_part = MIMEText(html_body, 'html')
            message.attach(html_part)
        
        # Add attachments if provided
        if attachments:
            for file_path in attachments:
                if os.path.isfile(file_path):
                    content_type, encoding = mimetypes.guess_type(file_path)
                    if content_type is None or encoding is not None:
                        content_type = 'application/octet-stream'
                    
                    main_type, sub_type = content_type.split('/', 1)
                    
                    with open(file_path, 'rb') as fp:
                        attachment = MIMEBase(main_type, sub_type)
                        attachment.set_payload(fp.read())
                    
                    encoders.encode_base64(attachment)
                    attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename={os.path.basename(file_path)}'
                    )
                    message.attach(attachment)
                else:
                    logger.warning(f"Attachment file '{file_path}' not found")
        
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    
    def _extract_message_body(self, payload):
        """Extract message body from Gmail API payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body