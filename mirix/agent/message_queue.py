import uuid
import time
import threading
import traceback


class MessageQueue:
    """
    Handles queueing and ordering of messages to different agent types.
    Ensures that messages of the same type are processed in order.
    """
    
    def __init__(self):
        self.message_queue = {}
        self._message_queue_lock = threading.Lock()
    
    def send_message_in_queue(self, client, agent_id, kwargs, agent_type='chat'):
        """
        Queue a message to be sent to a specific agent type.
        
        Args:
            client: The mirix client instance
            agent_id: The ID of the agent to send the message to
            kwargs: Arguments to pass to client.send_message
            agent_type: Type of agent to send message to
            
        Returns:
            Tuple of (response, agent_type)
        """
        message_uuid = uuid.uuid4()

        with self._message_queue_lock:
            self.message_queue[message_uuid] = {
                'kwargs': kwargs,
                'started': False,
                'finished': False,
                'type': agent_type,
            }

        # Wait for earlier requests of the same type to finish
        while not self._check_if_earlier_requests_are_finished(message_uuid):
            time.sleep(0.1)

        with self._message_queue_lock:
            self.message_queue[message_uuid]['started'] = True

        try:
            response = client.send_message(
                agent_id=agent_id,
                role='user',
                **self.message_queue[message_uuid]['kwargs']
            )
        except Exception as e:
            print(f"Error sending message: {e}")
            print(traceback.format_exc())
            print("agent_type: ", agent_type, "gets error. agent_id: ", agent_id, "ERROR")
            response = "ERROR"

        with self._message_queue_lock:
            self.message_queue[message_uuid]['finished'] = True
            del self.message_queue[message_uuid]
        
        return response, agent_type
    
    def _check_if_earlier_requests_are_finished(self, message_uuid):
        """Check if all earlier requests of the same type have finished."""
        with self._message_queue_lock:
            if message_uuid not in self.message_queue:
                raise ValueError("Message not found in the queue.")
            
            # Get current message type
            current_message_type = self.message_queue[message_uuid]['type']
            
            # Find index of current message
            message_keys = list(self.message_queue.keys())
            idx = message_keys.index(message_uuid)
            
            # Check earlier messages of the same type
            for i in range(idx):
                earlier_message = self.message_queue[message_keys[i]]
                if earlier_message['type'] == current_message_type:
                    if not earlier_message['finished']:
                        return False
            
            return True
    
    def _get_agent_id_for_type(self, agent_states, agent_type):
        """Get the agent ID for the specified agent type."""
        agent_type_to_state_mapping = {
            'chat': 'agent_state',
            'episodic_memory': 'episodic_memory_agent_state',
            'procedural_memory': 'procedural_memory_agent_state',
            'knowledge_vault': 'knowledge_vault_agent_state',
            'meta_memory': 'meta_memory_agent_state',
            'semantic_memory': 'semantic_memory_agent_state',
            'core_memory': 'core_memory_agent_state',
            'resource_memory': 'resource_memory_agent_state',
            'meta_memory_agent': 'meta_memory_agent_state',  # Alias
        }
        
        state_name = agent_type_to_state_mapping.get(agent_type)
        if not state_name:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if not hasattr(agent_states, state_name):
            raise ValueError(f"Agent state {state_name} not found")

        return getattr(agent_states, state_name).id
    
    def get_queue_length(self):
        """Get the current length of the message queue."""
        with self._message_queue_lock:
            return len(self.message_queue)
