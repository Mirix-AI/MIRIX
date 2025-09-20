from typing import List, Optional
from mirix.agent import Agent, AgentState
from datetime import datetime
from mirix.utils import convert_timezone_to_utc

def send_message(self: "Agent",  agent_state: "AgentState", message: str, topic: str = None) -> Optional[str]:
    """
    Sends a message to the human user. Meanwhile, whenever this function is called, the agent needs to include the `topic` of the current focus. It can be the same as before, it can also be updated when the agent is focusing on something different.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.
        topic (str): The focus of the agent right now. It is used to track the most recent topic in the conversation and will be used to retrieve the relevant memories from each memory component. 

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    agent_state.topic = topic
    return None

def send_intermediate_message(self: "Agent",  agent_state: "AgentState", message: str, topic: str = None) -> Optional[str]:
    """
    Sends an intermediate message to the human user. Meanwhile, whenever this function is called, the agent needs to include the `topic` of the current focus. It should NEVER be any questions or requests for the user but only the agent's current progress on the task.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.
        topic (str): The focus of the agent right now. It is used to track the most recent topic in the conversation and will be used to retrieve the relevant memories from each memory component. 

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    agent_state.topic = topic
    return None

def conversation_search(self: "Agent", query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """

    import math

    from mirix.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from mirix.utils import json_dumps

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    # TODO: add paging by page number. currently cursor only works with strings.
    # original: start=page * count
    messages = self.message_manager.list_user_messages_for_agent(
        agent_id=self.agent_state.id,
        actor=self.user,
        query_text=query,
        limit=count,
    )
    total = len(messages)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(messages) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(messages)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [message.text for message in messages]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


def search_in_memory(self: "Agent", memory_type: str, query: str, search_field: str, search_method: str, timezone_str: str) -> Optional[str]:
    """
    Choose which memory to search. All memory types support multiple search methods with different performance characteristics. Most of the time, you should use search over 'details' for episodic memory and semantic memory, 'content' for resource memory (but for resource memory, `embedding` is not supported for content field so you have to use other search methods), 'description' for procedural memory. This is because these fields have the richest information and is more likely to contain the keywords/query. You can always start from a thorough search over the whole memory by setting memory_type as 'all' and search_field as 'null', and then narrow down to specific fields and specific memories.
    
    Args:
        memory_type: The type of memory to search in. It should be chosen from the following: "episodic", "resource", "procedural", "knowledge_vault", "semantic", "all". Here "all" means searching in all the memories. 
        query: The keywords/query used to search in the memory.        
        search_field: The field to search in the memory. It should be chosen from the attributes of the corresponding memory. For "episodic" memory, it can be 'summary', 'details'; for "resource" memory, it can be 'summary', 'content'; for "procedural" memory, it can be 'summary', 'steps'; for "knowledge_vault", it can be 'secret_value', 'caption'; for semantic memory, it can be 'name', 'summary', 'details'. For "all", it should also be "null" as the system will search all memories with default fields. 
        search_method: The method to search in the memory. Choose from:
            - 'bm25': BM25 ranking-based full-text search (fast and effective for keyword-based searches)
            - 'embedding': Vector similarity search using embeddings (most powerful, good for conceptual matches)
    
    Returns:
        str: Query result string
    """

    if memory_type == 'resource' and search_field == 'content' and search_method == 'embedding':
        raise ValueError("embedding is not supported for resource memory's 'content' field.")
    if memory_type == 'knowledge_vault' and search_field == 'secret_value' and search_method == 'embedding':
        raise ValueError("embedding is not supported for knowledge_vault memory's 'secret_value' field.")
    
    if memory_type == 'all':
        search_field = 'null'

    if memory_type == 'core':
        # It means the model is an idiot, but we still return the results:
        return self.agent_state.memory.compile(), len(self.agent_state.memory.list_block_labels())

    if memory_type == 'episodic' or memory_type == 'all':
        episodic_memory = self.episodic_memory_manager.list_episodic_memory(
            actor=self.user,
            agent_state=self.agent_state,
            query=query,
            search_field=search_field if search_field != 'null' else 'summary',
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        formatted_results_from_episodic = [{'memory_type': 'episodic', 'id': x.id, 'timestamp': x.occurred_at, 'event_type': x.event_type, 'actor': x.actor, 'summary': x.summary, 'details': x.details} for x in episodic_memory]
        if memory_type == 'episodic':
            return formatted_results_from_episodic, len(formatted_results_from_episodic)

    if memory_type == 'resource' or memory_type == 'all':
        resource_memories = self.resource_memory_manager.list_resources(
            actor=self.user,
            agent_state=self.agent_state,
            query=query,
            search_field=search_field if search_field != 'null' else ('summary' if search_method == 'embedding' else 'content'),
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        formatted_results_resource = [{'memory_type': 'resource', 'id': x.id, 'resource_type': x.resource_type, 'summary': x.summary, 'content': x.content} for x in resource_memories]
        if memory_type == 'resource':
            return formatted_results_resource, len(formatted_results_resource)
    
    if memory_type == 'procedural' or memory_type == 'all':
        procedural_memories = self.procedural_memory_manager.list_procedures(
            actor=self.user,
            agent_state=self.agent_state,
            query=query,
            search_field=search_field if search_field != 'null' else 'summary',
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        formatted_results_procedural = [{'memory_type': 'procedural', 'id': x.id, 'entry_type': x.entry_type, 'summary': x.summary, 'steps': x.steps} for x in procedural_memories]
        if memory_type == 'procedural':
            return formatted_results_procedural, len(formatted_results_procedural)
    
    if memory_type == 'knowledge_vault' or memory_type == 'all':
        knowledge_vault_memories = self.knowledge_vault_manager.list_knowledge(
            actor=self.user,
            agent_state=self.agent_state,
            query=query,
            search_field=search_field if search_field != 'null' else 'caption',
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        formatted_results_knowledge_vault = [{'memory_type': 'knowledge_vault', 'id': x.id, 'entry_type': x.entry_type, 'source': x.source, 'sensitivity': x.sensitivity, 'secret_value': x.secret_value, 'caption': x.caption} for x in knowledge_vault_memories]
        if memory_type == 'knowledge_vault':
            return formatted_results_knowledge_vault, len(formatted_results_knowledge_vault)

    if memory_type == 'semantic' or memory_type == 'all':
        semantic_memories = self.semantic_memory_manager.list_semantic_items(
            actor=self.user,
            agent_state=self.agent_state,
            query=query,
            search_field=search_field if search_field != 'null' else 'summary',
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        # title, summary, details, source
        formatted_results_semantic = [{"memory_type": 'semantic', 'id': x.id, 'name': x.name, 'summary': x.summary, 'details': x.details, 'source': x.source} for x in semantic_memories]
        if memory_type == 'semantic':
            return formatted_results_semantic, len(formatted_results_semantic)

    else:
        raise ValueError(f"Memory type '{memory_type}' is not supported. Please choose from 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'.")
    return formatted_results_from_episodic + formatted_results_resource + formatted_results_procedural + formatted_results_knowledge_vault + formatted_results_semantic, len(formatted_results_from_episodic) + len(formatted_results_resource) + len(formatted_results_procedural) + len(formatted_results_knowledge_vault) + len(formatted_results_semantic)

def list_memory_within_timerange(self: "Agent", memory_type: str, start_time: str, end_time: str, timezone_str: str) -> Optional[str]:
    """
    List memories around a specific timestamp
    Args:
        memory_type (str): The type of memory to search in. It should be chosen from the following: "episodic", "resource", "procedural", "knowledge_vault", "semantic", "all". Here "all" means searching in all the memories. 
        start_time (str): The start time of the time range. It has to be in the form of "%Y-%m-%d %H:%M:%S"
        end_time (str): The end time of the time range. It has to be in the form of "%Y-%m-%d %H:%M:%S"
    """

    start_time = convert_timezone_to_utc(start_time, timezone_str)
    end_time = convert_timezone_to_utc(end_time, timezone_str)

    if memory_type == 'episodic' or memory_type == 'all':
        episodic_memory = self.episodic_memory_manager.list_episodic_memory_around_timestamp(
            actor=self.user,
            agent_state=self.agent_state,
            start_time=start_time,
            end_time=end_time,
            timezone_str=timezone_str,
        )
        formatted_results_from_episodic = [{'memory_type': 'episodic', 'id': x.id, 'timestamp': x.occurred_at, 'event_type': x.event_type, 'actor': x.actor, 'summary': x.summary} for x in episodic_memory]
        if memory_type == 'episodic':
            if len(formatted_results_from_episodic) == 0:
                return f"No results found."
            elif len(formatted_results_from_episodic) > 50:
                return "Too many results found. Please narrow down your search."
            else:
                return formatted_results_from_episodic, len(formatted_results_from_episodic)
    
    # currently only episodic memory is supported
    return None
