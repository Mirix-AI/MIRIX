"""
Background worker that consumes messages from the queue
Runs in a daemon thread and processes messages through the server
"""
import threading
from typing import TYPE_CHECKING, Optional, Any
from datetime import datetime

from mirix.queue.message_pb2 import QueueMessage
from mirix.log import get_logger
from mirix.services.user_manager import UserManager

if TYPE_CHECKING:
    from .queue_interface import QueueInterface
    from mirix.schemas.user import User
    from mirix.schemas.client import Client
    from mirix.schemas.message import MessageCreate

logger = get_logger(__name__)  # Use Mirix logger for proper configuration

class QueueWorker:
    """Background worker that processes messages from the queue"""
    
    def __init__(self, queue: 'QueueInterface', server: Optional[Any] = None):
        """
        Initialize the queue worker
        
        Args:
            queue: Queue implementation to consume from
            server: Optional server instance to invoke APIs on
        """
        logger.debug("Initializing queue worker with server=%s", 'provided' if server else 'None')
        
        self.queue = queue
        self._server = server
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
    
    def _convert_proto_user_to_pydantic(self, proto_user) -> 'Client':
        """
        Convert protobuf User to Pydantic Client
        
        Note: The protobuf schema still uses "User" for historical reasons,
        but it actually represents a Client in the new architecture.
        
        Args:
            proto_user: Protobuf User message (actually represents a Client)
            
        Returns:
            Pydantic Client object
        """
        # Lazy import to avoid circular dependency
        from mirix.schemas.client import Client
        
        return Client(
            id=proto_user.id,
            organization_id=proto_user.organization_id if proto_user.organization_id else None,
            name=proto_user.name,
            status=proto_user.status,
            scope="",  # Default scope
            # Client doesn't have timezone field - it's only for User
            created_at=proto_user.created_at.ToDatetime() if proto_user.HasField('created_at') else datetime.now(),
            updated_at=proto_user.updated_at.ToDatetime() if proto_user.HasField('updated_at') else datetime.now(),
            is_deleted=proto_user.is_deleted,
            api_key_hash=None  # No API key hash from protobuf
        )
    
    def _convert_proto_message_to_pydantic(self, proto_msg) -> 'MessageCreate':
        """
        Convert protobuf MessageCreate to Pydantic MessageCreate
        
        Args:
            proto_msg: Protobuf MessageCreate message
            
        Returns:
            Pydantic MessageCreate object
        """
        # Lazy import to avoid circular dependency
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.enums import MessageRole
        
        # Map role
        if proto_msg.role == proto_msg.ROLE_USER:
            role = MessageRole.user
        elif proto_msg.role == proto_msg.ROLE_SYSTEM:
            role = MessageRole.system
        else:
            role = MessageRole.user  # Default
        
        # Get content (currently only supporting text_content)
        content = proto_msg.text_content if proto_msg.HasField('text_content') else ""
        
        return MessageCreate(
            role=role,
            content=content,
            name=proto_msg.name if proto_msg.HasField('name') else None,
            otid=proto_msg.otid if proto_msg.HasField('otid') else None,
            sender_id=proto_msg.sender_id if proto_msg.HasField('sender_id') else None,
            group_id=proto_msg.group_id if proto_msg.HasField('group_id') else None,
            filter_tags=None  # Filter tags not stored in protobuf message
        )
    
    def set_server(self, server: Any) -> None:
        """
        Set or update the server instance.
        
        Args:
            server: Server instance to invoke APIs on
        """
        with self._lock:
            self._server = server
            logger.info("Updated worker server instance")
    
    def _process_message(self, message: QueueMessage) -> None:
        """
        Process a queue message by calling server.send_messages()
        
        Args:
            message: QueueMessage protobuf to process
        """
        # Check if server is available
        with self._lock:
            server = self._server
        
        if server is None:
            logger.warning(
                "No server available - skipping message: agent_id=%s, input_messages_count=%s",
                message.agent_id,
                len(message.input_messages)
            )
            return
        
        try:
            # Validate actor exists before processing
            if not message.HasField('actor') or not message.actor.id:
                raise ValueError(
                    f"Queue message for agent {message.agent_id} missing required actor information"
                )
            
            # Convert protobuf to Pydantic Client (actor for write operations)
            actor = self._convert_proto_user_to_pydantic(message.actor)
            
            input_messages = [
                self._convert_proto_message_to_pydantic(msg) 
                for msg in message.input_messages
            ]
            
            # Extract optional parameters
            chaining = message.chaining if message.HasField('chaining') else True
            
            # Extract user_id from queue message
            user_id = message.user_id if message.HasField('user_id') else None

            # Initialize user from user_id
            user = None
            if user_id:
                user_manager = UserManager()
                try:
                    user = user_manager.get_user_by_id(user_id)
                except Exception as e:
                    logger.warning(
                        "Failed to load user with id=%s, falling back to default user: %s",
                        user_id,
                        e
                    )
                    user = user_manager.get_default_user()
            else:
                # If no user_id provided, use default user
                user_manager = UserManager()
                user = user_manager.get_default_user()            
            
            # Extract filter_tags from protobuf Struct
            filter_tags = None
            if message.HasField('filter_tags') and message.filter_tags:
                filter_tags = dict(message.filter_tags)
            
            # Extract use_cache
            use_cache = message.use_cache if message.HasField('use_cache') else True
            
            # Log the processing
            logger.info(
                "Processing message via server: agent_id=%s, client_id=%s (from actor), user_id=%s, input_messages_count=%s, use_cache=%s, filter_tags=%s",
                message.agent_id,
                actor.id,
                user_id,
                len(input_messages),
                use_cache,
                filter_tags
            )
            
            # Call server.send_messages() with actor (Client) + client_id + user_id
            usage = server.send_messages(
                actor=actor,  # Client for authentication + write operations
                agent_id=message.agent_id,
                input_messages=input_messages,
                chaining=chaining,
                user=user,      # End-user user
                filter_tags=filter_tags,
                use_cache=use_cache,
            )
            
            # Log successful processing
            logger.debug(
                "Successfully processed message: agent_id=%s, usage=%s",
                message.agent_id,
                usage.model_dump() if usage else 'None'
            )
            
        except Exception as e:
            logger.error(
                "Error processing message for agent_id=%s: %s",
                message.agent_id,
                e,
                exc_info=True
            )
    
    def _consume_messages(self) -> None:
        """
        Main worker loop - continuously consume and process messages
        Runs in a separate thread
        """
        logger.info("🎯 Queue worker thread ENTERED _consume_messages()")
        logger.info("   _running=%s, _server=%s", self._running, self._server is not None)
        
        while self._running:
            try:
                # Get message from queue (with timeout to allow graceful shutdown)
                message: QueueMessage = self.queue.get(timeout=1.0)
                
                # Log receipt of message
                logger.debug(
                    "Received message: agent_id=%s, user_id=%s, input_messages_count=%s (client_id will be derived from actor)",
                    message.agent_id,
                    message.user_id if message.HasField('user_id') else 'None',
                    len(message.input_messages)
                )
                
                # Process the message through the server
                self._process_message(message)
                
            except Exception as e:
                # Handle timeout and other exceptions
                # For queue.Empty or StopIteration, just continue
                if type(e).__name__ in ['Empty', 'StopIteration']:
                    continue
                else:
                    logger.error("Error in message consumption loop: %s", e, exc_info=True)
        
        logger.info("Queue worker stopped")
    
    def start(self) -> None:
        """Start the background worker thread"""
        if self._running:
            logger.warning("⚠️ Queue worker already running")
            return  # Already running
        
        logger.info("🚀 Starting queue worker thread...")
        self._running = True
        
        # Create and start daemon thread
        # Daemon threads automatically stop when the main program exits
        logger.info("📝 Creating daemon thread for message consumption...")
        self._thread = threading.Thread(target=self._consume_messages, daemon=True)
        logger.info("✅ Thread created: %s", self._thread)
        
        logger.info("▶️  Starting thread...")
        self._thread.start()
        logger.info("✅ Thread.start() called, is_alive=%s", self._thread.is_alive())
        
        logger.info("✅ Queue worker thread started successfully")
    
    def stop(self) -> None:
        """Stop the background worker thread"""
        if not self._running:
            logger.warning("Queue worker not running, nothing to stop")
            return  # Not running
        
        logger.debug("Stopping queue worker")
        self._running = False
        
        # Wait for thread to finish
        if self._thread:
            logger.debug("Waiting for worker thread to finish")
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Worker thread did not finish within timeout")
            else:
                logger.debug("Worker thread finished successfully")
        
        # Close queue resources
        logger.debug("Closing queue resources")
        self.queue.close()
        
        logger.debug("Queue worker stopped")


