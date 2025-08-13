import re
from typing import List, Optional

from mirix.agent import Agent, AgentState
from mirix.schemas.knowledge_vault import KnowledgeVaultItemBase
from mirix.schemas.episodic_memory import EpisodicEventForLLM
from mirix.schemas.resource_memory import ResourceMemoryItemBase
from mirix.schemas.procedural_memory import ProceduralMemoryItemBase
from mirix.schemas.semantic_memory import SemanticMemoryItemBase

def core_memory_append(self: "Agent", agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
    """
    Append to the contents of core memory. The content will be appended to the end of the block with the given label. If you hit the limit, you can use `core_memory_rewrite` to rewrite the entire block to shorten the content. Note that "Line n:" is only for your visualization of the memory, and you should not include it in the content.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # check if the content starts with something like "Line n:" (here n is a number) using regex
    if re.match(r"^Line \d+:", content):
        raise ValueError("You should not include 'Line n:' (here n is a number) in the content.")

    current_value = str(agent_state.memory.get_block(label).value)
    new_value = (current_value + "\n" + str(content)).strip()
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None

def core_memory_rewrite(self: "Agent", agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
    """
    Rewrite the entire content of block <label> in core memory. The entire content in that block will be replaced with the new content. If the old content is full, and you have to rewrite the entire content, make sure to be extremely concise and make it shorter than 20% of the limit.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    new_value = content.strip()
    if current_value != new_value:
        agent_state.memory.update_block_value(label=label, value=new_value)
    return None

def core_memory_replace(self: "Agent", agent_state: "AgentState", label: str, old_line_numbers: str, new_content: str) -> Optional[str]:  # type: ignore
    """
    Replace the contents of core memory. `old_line_numbers` can be a single line number or a range. For instance, it should be "Line 5" to update line 5, or "Line 5-8" to update/delete lines 5 through 8 in the memory. To delete memories, use an empty string for `new_content`. Otherwise, the content will be replaced with the new content. When replacing a range of lines, the new_content will replace all the specified lines.
     
    Args:
        label (str): Section of the memory to be edited (persona or human).
        old_line_numbers (str): The line number(s) of the content to be updated. Can be "Line 5" for single line or "Line 5-8" for a range.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported. Can contain newlines to create multiple lines.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    current_value = current_value.split("\n")
    
    # Parse line numbers
    line_part = old_line_numbers.split(" ")[-1]  # Extract the number part after "Line "
    
    if "-" in line_part:
        # Range of lines
        start_line, end_line = line_part.split("-")
        start_idx = int(start_line) - 1  # Convert to 0-based index
        end_idx = int(end_line) - 1      # Convert to 0-based index
    else:
        # Single line
        start_idx = end_idx = int(line_part) - 1
    
    if new_content != "":
        # Replace the range with new content
        new_lines = new_content.split("\n")
        current_value = current_value[:start_idx] + new_lines + current_value[end_idx + 1:]
    else:
        # Delete the lines
        current_value = current_value[:start_idx] + current_value[end_idx + 1:]
    
    agent_state.memory.update_block_value(label=label, value="\n".join(current_value))
    return None

def episodic_memory_insert(self: "Agent", items: List[EpisodicEventForLLM]):
    """
    The tool to update episodic memory. The item being inserted into the episodic memory is an event either happened on the user or the assistant.

    Args:
        items (array): List of episodic memory items to insert.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    for item in items:
        self.episodic_memory_manager.insert_event(
            agent_state=self.agent_state,
            timestamp=item['occurred_at'],
            event_type=item['event_type'],
            actor=item['actor'],
            summary=item['summary'],
            details=item['details'],
            organization_id=self.user.organization_id,
            tree_path=item.get('tree_path')
        )
    response = "Events inserted! Now you need to check if there are repeated events shown in the system prompt."
    return response

def episodic_memory_merge(self: "Agent", event_id: str, combined_summary: str = None, combined_details: str = None):
    """
    The tool to merge the new episodic event into the selected episodic event by event_id, should be used when the user is continuing doing the same thing with more details. The combined_summary and combined_details will overwrite the old summary and details of the selected episodic event. Thus DO NOT use "User continues xxx" as the combined_summary because the old one WILL BE OVERWRITTEN and then we can only see "User continus xxx" without the old event.
        
    Args:
        event_id (str): This is the id of which episodic event to append to. 
        combined_summary (str): The updated summary. Note that it will overwrite the old summary so make sure to include the information from the old summary. The new summary needs to be only slightly different from the old summary.
        combined_details (str): The new details to add into the details of the selected episodic event.
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    episodic_memory = self.episodic_memory_manager.update_event(
        event_id=event_id,
        new_summary=combined_summary,
        new_details=combined_details
    )
    response = "These are the `summary` and the `details` of the updated event:\n", str({'event_id': episodic_memory.id, 'summary': episodic_memory.summary, 'details': episodic_memory.details}) + "\nIf the `details` are too verbose, or the `summary` cannot cover the information in the `details`, call episodic_memory_replace to update this event."
    return response

def episodic_memory_replace(self: "Agent", event_ids: List[str], new_items: List[EpisodicEventForLLM]):
    """
    The tool to replace or delete items in the episodic memory. To replace the memory, set the event_ids to be the ids of the events that needs to be replaced and new_items as the updated events. Note that the number of new items does not need to be the same as the number of event_ids as it is not a one-to-one mapping. To delete the memory, set the event_ids to be the ids of the events that needs to be deleted and new_items as an empty list. To insert new events, use episodic_memory_insert function.

    Args:
        event_ids (str): The ids of the episodic events to be deleted (or replaced).
        new_items (array): List of new episodic memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """

    for event_id in event_ids:
        # It will raise an error if the event_id is not found in the episodic memory.
        self.episodic_memory_manager.get_episodic_memory_by_id(event_id)

    for event_id in event_ids:
        self.episodic_memory_manager.delete_event_by_id(event_id)

    for new_item in new_items:
        self.episodic_memory_manager.insert_event(
            agent_state=self.agent_state,
            timestamp=new_item['occurred_at'],
            event_type=new_item['event_type'],
            actor=new_item['actor'],
            summary=new_item['summary'],
            details=new_item['details'],
            organization_id=self.user.organization_id,
            tree_path=new_item.get('tree_path')
        )

def check_episodic_memory(self: "Agent", event_ids: List[str], timezone_str: str) -> List[EpisodicEventForLLM]:
    """
    The tool to check the episodic memory. This function will return the episodic events with the given event_ids.

    Args:
        event_ids (str): The ids of the episodic events to be checked.
    
    Returns:
        List[EpisodicEventForLLM]: List of episodic events with the given event_ids.
    """
    episodic_memory = [
        self.episodic_memory_manager.get_episodic_memory_by_id(event_id, timezone_str=timezone_str) for event_id in event_ids
    ]

    formatted_results = [{'event_id': x.id, 'timestamp': x.occurred_at, 'event_type': x.event_type, 'actor': x.actor, 'summary': x.summary, 'details': x.details, 'tree_path': x.tree_path} for x in episodic_memory]

    return formatted_results

def resource_memory_insert(self: "Agent", items: List[ResourceMemoryItemBase]):
    """
    The tool to insert new items into resource memory.

    Args:
        items (array): List of resource memory items to insert.
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    for item in items:
        self.resource_memory_manager.insert_resource(
            agent_state=self.agent_state,
            title=item['title'],
            summary=item['summary'],
            resource_type=item['resource_type'],
            content=item['content'],
            organization_id=self.user.organization_id,
            tree_path=item.get('tree_path')
        )

def resource_memory_update(self: "Agent", old_ids: List[str], new_items: List[ResourceMemoryItemBase]):
    """
    The tool to update and delete items in the resource memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new resource memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    
    for old_id in old_ids:
        self.resource_memory_manager.delete_resource_by_id(
            resource_id=old_id
        )
    
    for item in new_items:
        self.resource_memory_manager.insert_resource(
            agent_state=self.agent_state,
            title=item['title'],
            summary=item['summary'],
            resource_type=item['resource_type'],
            content=item['content'],
            organization_id=self.user.organization_id,
            tree_path=item.get('tree_path')
        )

def procedural_memory_insert(self: "Agent", items: List[ProceduralMemoryItemBase]):
    """
    The tool to insert new procedures into procedural memory. Note that the `summary` should not be a general term such as "guide" or "workflow" but rather a more informative description of the procedure.

    Args:
        items (array): List of procedural memory items to insert.
        
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    for item in items:
        self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            entry_type=item['entry_type'],
            summary=item['summary'],
            steps=item['steps'],
            organization_id=self.user.organization_id,
            tree_path=item.get('tree_path')
        )

def procedural_memory_update(self: "Agent", old_ids: List[str], new_items: List[ProceduralMemoryItemBase]):
    """
    The tool to update/delete items in the procedural memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.
    
    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new procedural memory items to insert. If this is an empty list, then it means that the items are being deleted.
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    for old_id in old_ids:
        self.procedural_memory_manager.delete_procedure_by_id(
            procedure_id=old_id
        )
    
    for item in new_items:
        self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            entry_type=item['entry_type'],
            summary=item['summary'],
            steps=item['steps'],
            organization_id=self.user.organization_id,
            tree_path=item.get('tree_path')
        )

def check_semantic_memory(self: "Agent", semantic_item_ids: List[str], timezone_str: str) -> List[SemanticMemoryItemBase]:
    """
    The tool to check the semantic memory. This function will return the semantic memory items with the given ids.

    Args:
        semantic_item_ids (str): The ids of the semantic memory items to be checked.
    
    Returns:
        List[SemanticMemoryItemBase]: List of semantic memory items with the given ids.
    """
    semantic_memory = [
        self.semantic_memory_manager.get_semantic_item_by_id(semantic_memory_id=id, timezone_str=timezone_str) for id in semantic_item_ids
    ]

    formatted_results = [{'semantic_item_id': x.id, 'name': x.name, 'summary': x.summary, 'details': x.details, 'source': x.source, 'tree_path': x.tree_path} for x in semantic_memory]

    return formatted_results

def semantic_memory_insert(self: "Agent", items: List[SemanticMemoryItemBase]):
    """
    The tool to insert items into semantic memory. 

    Args:
        items (array): List of semantic memory items to insert.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    for item in items:
        self.semantic_memory_manager.insert_semantic_item(
            agent_state=self.agent_state,
            name=item['name'],
            summary=item['summary'],
            details=item['details'],
            source=item['source'],
            tree_path=item['tree_path'],
            organization_id=self.user.organization_id
        )

def semantic_memory_update(self: "Agent", old_semantic_item_ids: List[str], new_items: List[SemanticMemoryItemBase]):
    """
    The tool to update/delete items in the semantic memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_semantic_item_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new semantic memory items to insert. If this is an empty list, then it means that the items are being deleted.
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    for old_id in old_semantic_item_ids:
        self.semantic_memory_manager.delete_semantic_item_by_id(
            semantic_memory_id=old_id
        )
    
    new_ids = []
    for item in new_items:
        inserted_item = self.semantic_memory_manager.insert_semantic_item(
            agent_state=self.agent_state,
            name=item['name'],
            summary=item['summary'],
            details=item['details'],
            source=item['source'],
            tree_path=item['tree_path'],
            organization_id=self.user.organization_id
        )
        new_ids.append(inserted_item.id)
    
    message_to_return = "Semantic memory with the following ids have been deleted: " + str(old_semantic_item_ids) + f". New semantic memory items are created: {str(new_ids)}"
    return message_to_return

def knowledge_vault_insert(self: "Agent", items: List[KnowledgeVaultItemBase]):
    """
    The tool to update knowledge vault.

    Args:
        items (array): List of knowledge vault items to insert.
        
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    for item in items:
        self.knowledge_vault_manager.insert_knowledge(
            agent_state=self.agent_state,
            entry_type=item['entry_type'],
            source=item['source'],
            sensitivity=item['sensitivity'],
            secret_value=item['secret_value'],
            caption=item['caption'],
            organization_id=self.user.organization_id
        )

def knowledge_vault_update(self: "Agent", old_ids: List[str], new_items: List[KnowledgeVaultItemBase]):
    """
    The tool to update/delete items in the knowledge vault. To update the knowledge_vault, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.
    
    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new knowledge vault items to insert. If this is an empty list, then it means that the items are being deleted.
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response
    """
    for old_id in old_ids:
        self.knowledge_vault_manager.delete_knowledge_by_id(
            knowledge_vault_item_id=old_id
        )
    
    for item in new_items:
        self.knowledge_vault_manager.insert_knowledge(
            agent_state=self.agent_state,
            entry_type=item['entry_type'],
            source=item['source'],
            sensitivity=item['sensitivity'],
            secret_value=item['secret_value'],
            caption=item['caption'],
            organization_id=self.user.organization_id
        )

def trigger_memory_update_with_instruction(self: "Agent", user_message: object, instruction: str, memory_type: str) -> Optional[str]:
    """
    Choose which memory to update. The function will trigger one specific memory agent with the instruction telling the agent what to do.

    Args:
        instruction (str): The instruction to the memory agent.
        memory_type (str): The type of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge_vault", "semantic". For instance, ['episodic', 'resource'].
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix import create_client

    client = create_client()
    agents = client.list_agents()

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}")

    # Fallback to sequential processing for backward compatibility
    response = ''

    if memory_type == "core":
        agent_type = "core_memory_agent"
    elif memory_type == "episodic":
        agent_type = "episodic_memory_agent"
    elif memory_type == "resource":
        agent_type = "resource_memory_agent"
    elif memory_type == "procedural":
        agent_type = "procedural_memory_agent"
    elif memory_type == "knowledge_vault":
        agent_type = "knowledge_vault_agent"
    elif memory_type == 'semantic':
        agent_type = "semantic_memory_agent"
    else:
        raise ValueError(f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'.")

    matching_agent = None
    for agent in agents:
        if agent.agent_type == agent_type:
            matching_agent = agent
            break
    
    if matching_agent is None:
        raise ValueError(f"No agent found with type '{agent_type}'")
    
    client.send_message(agent_id=matching_agent.id, 
        role='user', 
        message="[Message from Chat Agent (Now you are allowed to make multiple function calls sequentially)] " +instruction, 
        existing_file_uris=user_message['existing_file_uris'],
        retrieved_memories=user_message.get('retrieved_memories', None)
    )
    response += '[System Message] Agent ' + matching_agent.name + ' has been triggered to update the memory.\n'

    return response.strip()

def trigger_memory_update(self: "Agent", user_message: object, memory_types: List[str]) -> Optional[str]:
    """
    Choose which memory to update. This function will trigger another memory agent which is specifically in charge of handling the corresponding memory to update its memory. Trigger all necessary memory updates at once. Put the explanations in the `internal_monologue` field.

    Args:
        memory_types (List[str]): The types of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge_vault", "semantic". For instance, ['episodic', 'resource'].
        
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix import create_client

    client = create_client()
    agents = client.list_agents()

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}")

    if 'message_queue' in user_message:
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import time

        # Use multi-processing approach similar to _send_to_memory_agents_separately
        message_queue = user_message['message_queue']
        
        # Map memory types to agent types
        memory_type_to_agent_type = {
            "core": "core_memory_agent",
            "episodic": "episodic_memory_agent", 
            "resource": "resource_memory_agent",
            "procedural": "procedural_memory_agent",
            "knowledge_vault": "knowledge_vault_agent",
            "semantic": "semantic_memory_agent"
        }
        
        # Filter to only supported memory types
        valid_agent_types = []
        for memory_type in memory_types:
            if memory_type in memory_type_to_agent_type:
                valid_agent_types.append(memory_type_to_agent_type[memory_type])
            else:
                raise ValueError(f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'.")
        
        # Prepare payloads for message queue
        payloads = {
            'message': user_message['message'],
            'existing_file_uris': user_message.get('existing_file_uris', set()),
            'chaining': user_message.get('chaining', False),
            'message_queue': message_queue,
            'retrieved_memories': user_message.get('retrieved_memories', None)
        }

        responses = []
        overall_start = time.time()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=len(valid_agent_types)) as pool:
            futures = []
            for agent_type in valid_agent_types:
                matching_agents = [agent for agent in agents if agent.agent_type == agent_type]
                if not matching_agents:
                    raise ValueError(f"No agent found with type '{agent_type}'")
                futures.append(
                    pool.submit(message_queue.send_message_in_queue, 
                               client, matching_agents[0].id, payloads, agent_type)
                )
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                response, agent_type = future.result()
                responses.append(response)

        overall_end = time.time()
        response_message = f'[System Message] {len(valid_agent_types)} memory agents have been triggered in parallel to update the memory. Total time: {overall_end - overall_start:.2f} seconds.'
        return response_message

    else:
        
        # Fallback to sequential processing for backward compatibility
        response = ''

        for memory_type in memory_types:

            if memory_type == "core":
                agent_type = "core_memory_agent"
            elif memory_type == "episodic":
                agent_type = "episodic_memory_agent"
            elif memory_type == "resource":
                agent_type = "resource_memory_agent"
            elif memory_type == "procedural":
                agent_type = "procedural_memory_agent"
            elif memory_type == "knowledge_vault":
                agent_type = "knowledge_vault_agent"
            elif memory_type == 'semantic':
                agent_type = "semantic_memory_agent"
            else:
                raise ValueError(f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'.")

            matching_agent = None
            for agent in agents:
                if agent.agent_type == agent_type:
                    matching_agent = agent
                    break
            
            if matching_agent is None:
                raise ValueError(f"No agent found with type '{agent_type}'")
            
            client.send_message(agent_id=matching_agent.id, 
                role='user', 
                message=user_message['message'], 
                existing_file_uris=user_message['existing_file_uris'],
                retrieved_memories=user_message.get('retrieved_memories', None)
            )
            response += '[System Message] Agent ' + matching_agent.name + ' has been triggered to update the memory.\n'
        
        return response.strip()

def finish_memory_update(self: "Agent"):
    """
    Finish the memory update process. This function should be called after the Memory is updated.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None