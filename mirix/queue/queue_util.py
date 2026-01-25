import logging
from typing import List, Optional

from mirix.schemas.client import Client
from mirix.schemas.message import MessageCreate
from mirix.schemas.message import MessageCreate as PydanticMessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.mirix_message_content import TextContent

from mirix.queue.message_pb2 import User as ProtoUser
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
import mirix.queue as queue

from mirix.observability import add_trace_to_queue_message

logger = logging.getLogger(__name__)

def put_messages(
        actor: Client,
        agent_id: str,
        input_messages: List[MessageCreate],
        chaining: Optional[bool] = True,
        user_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        occurred_at: Optional[str] = None,
    ):
        """
        Create QueueMessage protobuf and send to queue.
        
        Args:
            actor: The Client performing the action (for auth/write operations)
                   Client ID is derived from actor.id
            agent_id: ID of the agent to send message to
            input_messages: List of messages to send
            chaining: Enable/disable chaining
            user_id: Optional user ID (end-user ID)
            verbose: Enable verbose logging
            filter_tags: Filter tags dictionary
            use_cache: Control Redis cache behavior
            occurred_at: Optional ISO 8601 timestamp string for episodic memory
        """
        logger.debug("Creating queue message for agent_id=%s, actor=%s (client_id derived from actor)", agent_id, actor.id)
        
        if not actor or not actor.id:
            raise ValueError(
                f"Cannot queue message: actor is None or has no ID. "
                f"actor={actor}, actor.id={actor.id if actor else 'N/A'}"
            )
        
        # Convert Pydantic Client to protobuf User (protobuf schema still uses "User")
        proto_user = ProtoUser()
        proto_user.id = actor.id
        proto_user.organization_id = actor.organization_id or ""
        proto_user.name = actor.name
        proto_user.status = actor.status
        # Client doesn't have timezone - use default "UTC"
        proto_user.timezone = getattr(actor, 'timezone', 'UTC')
        if actor.created_at:
            proto_user.created_at.FromDatetime(actor.created_at)
        if actor.updated_at:
            proto_user.updated_at.FromDatetime(actor.updated_at)
        proto_user.is_deleted = actor.is_deleted
        
        # Convert Pydantic MessageCreate list to protobuf MessageCreate list
        proto_input_messages = []
        for msg in input_messages:
            proto_msg = ProtoMessageCreate()
            # Map role
            if msg.role == MessageRole.user:
                proto_msg.role = ProtoMessageCreate.ROLE_USER
            elif msg.role == MessageRole.system:
                proto_msg.role = ProtoMessageCreate.ROLE_SYSTEM
            else:
                proto_msg.role = ProtoMessageCreate.ROLE_UNSPECIFIED
            
            # Handle content (can be string or list)
            if isinstance(msg.content, str):
                proto_msg.text_content = msg.content
            # For list content, we'd need to convert to structured_content
            # but for now, just convert to string representation
            elif isinstance(msg.content, list):
                # Convert list of content to string for now
                text_parts = []
                for content_part in msg.content:
                    if isinstance(content_part, TextContent):
                        text_parts.append(content_part.text)
                proto_msg.text_content = "\n".join(text_parts)
            
            # Optional fields
            if msg.name:
                proto_msg.name = msg.name
            if msg.otid:
                proto_msg.otid = msg.otid
            if msg.sender_id:
                proto_msg.sender_id = msg.sender_id
            if msg.group_id:
                proto_msg.group_id = msg.group_id
            
            proto_input_messages.append(proto_msg)
        
        # Build the QueueMessage
        queue_msg = QueueMessage()
        
        queue_msg.actor.CopyFrom(proto_user)
            
        queue_msg.agent_id = agent_id
        queue_msg.input_messages.extend(proto_input_messages)
        
        # Optional fields
        if chaining is not None:
            queue_msg.chaining = chaining
        if user_id:
            queue_msg.user_id = user_id
        if verbose is not None:
            queue_msg.verbose = verbose
        
        # Convert dict to Struct for filter_tags
        if filter_tags:
            queue_msg.filter_tags.update(filter_tags)
        
        # Set use_cache
        queue_msg.use_cache = use_cache
        
        # Set occurred_at if provided
        if occurred_at is not None:
            queue_msg.occurred_at = occurred_at
        
        # Add LangFuse trace context for distributed tracing
        queue_msg = add_trace_to_queue_message(queue_msg)
        
        # Send to queue
        logger.debug("Sending message to queue: agent_id=%s, input_messages_count=%s, occurred_at=%s", 
                    agent_id, len(input_messages), occurred_at)
        queue.save(queue_msg)
        logger.debug("Message successfully sent to queue")
