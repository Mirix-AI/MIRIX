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
from mirix.observability import restore_trace_from_queue_message
from mirix.observability.context import clear_trace_context

if TYPE_CHECKING:
    from .queue_interface import QueueInterface
    from mirix.schemas.user import User
    from mirix.schemas.client import Client
    from mirix.schemas.message import MessageCreate

logger = get_logger(__name__)  # Use Mirix logger for proper configuration


class QueueWorker:
    """Background worker that processes messages from the queue"""
    
    def __init__(
        self,
        queue: 'QueueInterface',
        server: Optional[Any] = None,
        partition_id: Optional[int] = None
    ):
        """
        Initialize the queue worker
        
        Args:
            queue: Queue implementation to consume from
            server: Optional server instance to invoke APIs on
            partition_id: Optional partition ID for partitioned queues.
                         If set, worker will only consume from this partition.
                         If None, uses default queue.get() behavior.
        """
        logger.debug(
            "Initializing queue worker: server=%s, partition_id=%s",
            'provided' if server else 'None',
            partition_id
        )
        
        self.queue = queue
        self._server = server
        self._partition_id = partition_id
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
            is_deleted=proto_user.is_deleted
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
        try:
            # Restore LangFuse trace context from queue message for distributed tracing
            trace_restored = restore_trace_from_queue_message(message)
            if trace_restored:
                logger.debug("Restored trace context from queue message for processing")
            
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
                except Exception:
                    # User doesn't exist - auto-create it using the client's organization
                    logger.info(
                        "User with id=%s not found, auto-creating with organization_id=%s",
                        user_id,
                        actor.organization_id
                    )
                    
                    from mirix.schemas.user import User as PydanticUser
                    
                    try:
                        # Create user with provided user_id and client's organization
                        user = user_manager.create_user(
                            pydantic_user=PydanticUser(
                                id=user_id,
                                name=user_id,  # Use user_id as default name
                                organization_id=actor.organization_id,
                                timezone=user_manager.DEFAULT_TIME_ZONE,
                                status="active",
                                is_deleted=False
                            )
                        )
                        logger.info("✓ Auto-created user: %s in organization: %s", user_id, actor.organization_id)
                    except Exception as create_error:
                        # If auto-creation fails, fall back to admin user
                        logger.error(
                            "Failed to auto-create user with id=%s: %s. Falling back to admin user.",
                            user_id,
                            create_error
                        )
                        user = user_manager.get_admin_user()
            else:
                # If no user_id provided, use admin user
                user_manager = UserManager()
                user = user_manager.get_admin_user()            
            
            # Extract filter_tags from protobuf Struct
            filter_tags = None
            if message.HasField('filter_tags') and message.filter_tags:
                filter_tags = dict(message.filter_tags)
            
            # Extract use_cache
            use_cache = message.use_cache if message.HasField('use_cache') else True
            
            # Extract occurred_at
            occurred_at = message.occurred_at if message.HasField('occurred_at') else None
            
            # Log the processing
            logger.info(
                "Processing message via server: agent_id=%s, client_id=%s (from actor), user_id=%s, input_messages_count=%s, use_cache=%s, filter_tags=%s, occurred_at=%s",
                message.agent_id,
                actor.id,
                user_id,
                len(input_messages),
                use_cache,
                filter_tags,
                occurred_at
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
                occurred_at=occurred_at,  # Optional timestamp for episodic memory
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
        finally:
            # Clear trace context to prevent leaking between messages
            clear_trace_context()
    
    def _consume_messages(self) -> None:
        """
        Main worker loop - continuously consume and process messages
        Runs in a separate thread
        """
        partition_info = f", partition={self._partition_id}" if self._partition_id is not None else ""
        logger.info("🎯 Queue worker thread ENTERED _consume_messages()%s", partition_info)
        logger.info("   _running=%s, _server=%s", self._running, self._server is not None)
        
        while self._running:
            try:
                # Get message from queue (with timeout to allow graceful shutdown)
                # Use partition-specific get if partition_id is set and queue supports it
                if self._partition_id is not None and hasattr(self.queue, 'get_from_partition'):
                    message: QueueMessage = self.queue.get_from_partition(
                        self._partition_id, timeout=1.0
                    )
                else:
                    message: QueueMessage = self.queue.get(timeout=1.0)
                
                # Log receipt of message
                logger.debug(
                    "Received message%s: agent_id=%s, user_id=%s, input_messages_count=%s (client_id will be derived from actor)",
                    partition_info,
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
        
        # Note: No logging here to avoid errors during shutdown
        # when logging system may already be closed
    
    def start(self) -> None:
        """Start the background worker thread"""
        if self._running:
            logger.warning("⚠️ Queue worker already running")
            return  # Already running
        
        partition_info = f" (partition {self._partition_id})" if self._partition_id is not None else ""
        logger.info("🚀 Starting queue worker thread%s...", partition_info)
        self._running = True
        
        # Create and start daemon thread
        # Daemon threads automatically stop when the main program exits
        thread_name = f"QueueWorker-{self._partition_id}" if self._partition_id is not None else "QueueWorker"
        logger.info("📝 Creating daemon thread for message consumption%s...", partition_info)
        self._thread = threading.Thread(
            target=self._consume_messages,
            daemon=True,
            name=thread_name
        )
        logger.info("✅ Thread created: %s", self._thread)
        
        logger.info("▶️  Starting thread...")
        self._thread.start()
        logger.info("✅ Thread.start() called, is_alive=%s", self._thread.is_alive())
        
        logger.info("✅ Queue worker thread%s started successfully", partition_info)
    
    def stop(self, close_queue: bool = True) -> None:
        """
        Stop the background worker thread
        
        Args:
            close_queue: Whether to close the queue resources. Set to False
                        when multiple workers share the same queue.
        
        Note: No logging during stop to avoid errors when called during shutdown,
        as the logging system may have already closed its file handlers.
        """
        if not self._running:
            return  # Not running
        
        self._running = False
        
        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Close queue resources only if requested
        # (shared queues should be closed by the manager, not individual workers)
        if close_queue:
            self.queue.close()


