from datetime import datetime
from typing import Dict, List, Optional

from mirix.orm.errors import NoResultFound
from mirix.orm.message import Message as MessageModel
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.message import MessageUpdate
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types
from mirix.services.utils import update_timezone


class MessageManager:
    """Manager class to handle business logic related to Messages."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @update_timezone
    @enforce_types
    def get_message_by_id(self, message_id: str, actor: PydanticUser) -> Optional[PydanticMessage]:
        """Fetch a message by ID."""
        with self.session_maker() as session:
            try:
                message = MessageModel.read(db_session=session, identifier=message_id, actor=actor)
                return message.to_pydantic()
            except NoResultFound:
                return None

    @update_timezone
    @enforce_types
    def get_messages_by_ids(self, message_ids: List[str], actor: PydanticUser) -> List[PydanticMessage]:
        """Fetch messages by ID and return them in the requested order."""
        with self.session_maker() as session:
            results = MessageModel.list(db_session=session, id=message_ids, organization_id=actor.organization_id, limit=len(message_ids))

            if len(results) != len(message_ids):
                raise NoResultFound(
                    f"Expected {len(message_ids)} messages, but found {len(results)}. Missing ids={set(message_ids) - set([r.id for r in results])}"
                )

            # Sort results directly based on message_ids
            result_dict = {msg.id: msg.to_pydantic() for msg in results}
            return [result_dict[msg_id] for msg_id in message_ids]

    @enforce_types
    def create_message(self, pydantic_msg: PydanticMessage, actor: PydanticUser) -> PydanticMessage:
        """Create a new message."""
        with self.session_maker() as session:
            # Set the organization id and user id of the Pydantic message
            pydantic_msg.organization_id = actor.organization_id
            pydantic_msg.user_id = actor.id
            msg_data = pydantic_msg.model_dump()
            msg = MessageModel(**msg_data)
            msg.create(session, actor=actor)  # Persist to database
            return msg.to_pydantic()

    @enforce_types
    def create_many_messages(self, pydantic_msgs: List[PydanticMessage], actor: PydanticUser) -> List[PydanticMessage]:
        """Create multiple messages."""
        return [self.create_message(m, actor=actor) for m in pydantic_msgs]

    @enforce_types
    def update_message_by_id(self, message_id: str, message_update: MessageUpdate, actor: PydanticUser) -> PydanticMessage:
        """
        Updates an existing record in the database with values from the provided record object.
        """
        with self.session_maker() as session:
            # Fetch existing message from database
            message = MessageModel.read(
                db_session=session,
                identifier=message_id,
                actor=actor,
            )

            # Some safety checks specific to messages
            if message_update.tool_calls and message.role != MessageRole.assistant:
                raise ValueError(
                    f"Tool calls {message_update.tool_calls} can only be added to assistant messages. Message {message_id} has role {message.role}."
                )
            if message_update.tool_call_id and message.role != MessageRole.tool:
                raise ValueError(
                    f"Tool call IDs {message_update.tool_call_id} can only be added to tool messages. Message {message_id} has role {message.role}."
                )

            # get update dictionary
            update_data = message_update.model_dump(exclude_unset=True, exclude_none=True)
            # Remove redundant update fields
            update_data = {key: value for key, value in update_data.items() if getattr(message, key) != value}

            for key, value in update_data.items():
                setattr(message, key, value)
            message.update(db_session=session, actor=actor)

            return message.to_pydantic()

    @enforce_types
    def delete_message_by_id(self, message_id: str, actor: PydanticUser) -> bool:
        """Delete a message."""
        with self.session_maker() as session:
            try:
                msg = MessageModel.read(
                    db_session=session,
                    identifier=message_id,
                    actor=actor,
                )
                msg.hard_delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Message with id {message_id} not found.")

    @enforce_types
    def size(
        self,
        actor: PydanticUser,
        role: Optional[MessageRole] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Get the total count of messages with optional filters.

        Args:
            actor: The user requesting the count
            role: The role of the message
        """
        with self.session_maker() as session:
            return MessageModel.size(db_session=session, actor=actor, role=role, agent_id=agent_id)

    @update_timezone
    @enforce_types
    def list_user_messages_for_agent(
        self,
        agent_id: str,
        actor: Optional[PydanticUser] = None,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
        ascending: bool = True,
    ) -> List[PydanticMessage]:
        """List user messages with flexible filtering and pagination options.

        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content

        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        message_filters = {"role": "user"}
        if filters:
            message_filters.update(filters)

        return self.list_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
            cursor=cursor,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            filters=message_filters,
            query_text=query_text,
            ascending=ascending,
        )

    @update_timezone
    @enforce_types
    def list_messages_for_agent(
        self,
        agent_id: str,
        actor: Optional[PydanticUser] = None,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
        ascending: bool = True,
    ) -> List[PydanticMessage]:
        """List messages with flexible filtering and pagination options.

        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content

        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        with self.session_maker() as session:
            # Start with base filters
            message_filters = {"agent_id": agent_id}
            if actor:
                message_filters.update({"organization_id": actor.organization_id})
            if filters:
                message_filters.update(filters)

            results = MessageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                query_text=query_text,
                ascending=ascending,
                **message_filters,
            )

            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def delete_detached_messages_for_agent(self, agent_id: str, actor: PydanticUser) -> int:
        """
        Delete messages that belong to an agent but are not in the agent's current message_ids list.
        
        This is useful for cleaning up messages that were removed from context during 
        context window management but still exist in the database.
        
        Args:
            agent_id: The ID of the agent to clean up messages for
            actor: The user performing this action
            
        Returns:
            int: Number of messages deleted
        """
        with self.session_maker() as session:
            # First, get the agent to access its current message_ids
            from mirix.orm.agent import Agent as AgentModel
            try:
                agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            except NoResultFound:
                raise ValueError(f"Agent with id {agent_id} not found.")
            
            # Get current message_ids (messages that should be kept)
            current_message_ids = set(agent.message_ids or [])
            
            # Find all messages for this agent
            all_messages = MessageModel.list(
                db_session=session, 
                agent_id=agent_id,
                organization_id=actor.organization_id,
                limit=None  # Get all messages
            )
            
            # Identify detached messages (not in current message_ids)
            detached_messages = [
                msg for msg in all_messages 
                if msg.id not in current_message_ids
            ]
            
            # Delete detached messages
            deleted_count = 0
            for msg in detached_messages:
                msg.hard_delete(session, actor=actor)
                deleted_count += 1
            
            session.commit()
            return deleted_count

    @enforce_types
    def cleanup_all_detached_messages(self, actor: PydanticUser) -> Dict[str, int]:
        """
        Cleanup detached messages for all agents in the organization.
        
        Args:
            actor: The user performing this action
            
        Returns:
            Dict[str, int]: Dictionary mapping agent_id to number of messages deleted
        """
        from mirix.orm.agent import Agent as AgentModel
        
        with self.session_maker() as session:
            # Get all agents for this organization
            agents = AgentModel.list(
                db_session=session,
                organization_id=actor.organization_id,
                limit=None
            )
            
            cleanup_results = {}
            total_deleted = 0
            
            for agent in agents:
                # Get current message_ids for this agent
                current_message_ids = set(agent.message_ids or [])
                
                # Find all messages for this agent
                all_messages = MessageModel.list(
                    db_session=session,
                    agent_id=agent.id,
                    organization_id=actor.organization_id,
                    limit=None
                )
                
                # Identify and delete detached messages
                detached_messages = [
                    msg for msg in all_messages 
                    if msg.id not in current_message_ids
                ]
                
                deleted_count = 0
                for msg in detached_messages:
                    msg.hard_delete(session, actor=actor)
                    deleted_count += 1
                
                cleanup_results[agent.id] = deleted_count
                total_deleted += deleted_count
            
            session.commit()
            cleanup_results['total'] = total_deleted
            return cleanup_results
