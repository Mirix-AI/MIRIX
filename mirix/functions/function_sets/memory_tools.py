import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Optional

from mirix.agent import Agent, AgentState
from mirix.log import get_logger
from mirix.observability.context import (
    clear_trace_context,
    get_trace_context,
    mark_observation_as_child,
    set_trace_context,
)
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.episodic_memory import EpisodicEventForLLM
from mirix.schemas.knowledge_vault import KnowledgeVaultItemBase
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.procedural_memory import ProceduralMemoryItemBase
from mirix.schemas.resource_memory import ResourceMemoryItemBase
from mirix.schemas.semantic_memory import SemanticMemoryItemBase

logger = get_logger(__name__)


def core_memory_append(
    self: "Agent", agent_state: "AgentState", label: str, content: str
) -> Optional[str]:  # type: ignore
    """
    Append to the contents of core memory. The content will be appended to the end of the block with the given label. If you hit the limit, you can use `core_memory_rewrite` to rewrite the entire block to shorten the content. Note that "Line n:" is only for your visualization of the memory, and you should not include it in the content.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is returned on success, or an error message string if the operation would exceed the limit.
    """
    # check if the content starts with something like "Line n:" (here n is a number) using regex
    if re.match(r"^Line \d+:", content):
        raise ValueError("You should not include 'Line n:' (here n is a number) in the content.")

    # Get the current block and its limit
    current_block = agent_state.memory.get_block(label)
    current_value = str(current_block.value)
    limit = current_block.limit

    # Calculate the new value and its length
    new_value = (current_value + "\n" + str(content)).strip()
    new_length = len(new_value)

    # Check if the new value would exceed the limit
    if new_length > limit:
        # Return a descriptive error message instead of raising an exception
        # This allows the agent to see the error and adapt its behavior
        error_msg = (
            f"ERROR: Cannot append - would exceed {limit} character limit "
            f"(current: {len(current_value)}, adding: {len(content)}, "
            f"total would be: {new_length}). "
            f"Use core_memory_rewrite to condense the '{label}' block first, "
            f"targeting around {int(limit * 0.5)} characters (~50% capacity)."
        )
        return error_msg

    # If within limit, perform the append
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None


def core_memory_rewrite(
    self: "Agent", agent_state: "AgentState", label: str, content: str
) -> Optional[str]:  # type: ignore
    """
    Rewrite the entire content of block <label> in core memory. The entire content in that block will be replaced with the new content. If the old content is full, and you have to rewrite the entire content, make sure to be extremely concise and make it shorter than 20% of the limit.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.
    Returns:
        Optional[str]: None is returned on success, or an error message string if the new content exceeds the limit.
    """
    # Get the current block and its limit
    current_block = agent_state.memory.get_block(label)
    current_value = str(current_block.value)
    limit = current_block.limit
    new_value = content.strip()
    new_length = len(new_value)

    # Check if the new value exceeds the limit
    if new_length > limit:
        error_msg = (
            f"ERROR: Rewrite failed - new content exceeds {limit} character limit "
            f"(provided: {new_length} characters). "
            f"Please condense further, targeting around {int(limit * 0.5)} characters (~50% capacity)."
        )
        return error_msg

    # Only update if the content actually changed
    if current_value != new_value:
        agent_state.memory.update_block_value(label=label, value=new_value)
        # Provide feedback on the operation
        percentage = int((new_length / limit) * 100)
        return f"Successfully rewrote '{label}' block: {new_length}/{limit} characters ({percentage}% full)."

    return None


def episodic_memory_insert(self: "Agent", items: List[EpisodicEventForLLM]):
    """
    The tool to update episodic memory. The item being inserted into the episodic memory is an event either happened on the user or the assistant.

    Args:
        items (array): List of episodic memory items to insert.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)
    occurred_at_override = getattr(self, "occurred_at", None)  # Optional timestamp override from API

    for item in items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = occurred_at_override if occurred_at_override else item["occurred_at"]

        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        self.episodic_memory_manager.insert_event(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            timestamp=timestamp,  # Use potentially overridden timestamp
            event_type=item["event_type"],
            event_actor=item["actor"],
            summary=item["summary"],
            details=item["details"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
    response = "Events inserted! Now you need to check if there are repeated events shown in the system prompt."
    return response


def episodic_memory_merge(
    self: "Agent",
    event_id: str,
    combined_summary: str = None,
    combined_details: str = None,
):
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
        new_details=combined_details,
        actor=self.actor,
        agent_state=self.agent_state,
        update_mode="replace",
    )
    response = (
        "These are the `summary` and the `details` of the updated event:\n",
        str(
            {
                "event_id": episodic_memory.id,
                "summary": episodic_memory.summary,
                "details": episodic_memory.details,
            }
        )
        + "\nIf the `details` are too verbose, or the `summary` cannot cover the information in the `details`, call episodic_memory_replace to update this event.",
    )
    return response


def episodic_memory_replace(self: "Agent", event_ids: List[str], new_items: List[EpisodicEventForLLM]):
    """
    The tool to replace or delete items in the episodic memory. To replace the memory, set the event_ids to be the ids of the events that needs to be replaced and new_items as the updated events. Note that the number of new items does not need to be the same as the number of event_ids as it is not a one-to-one mapping. To delete the memory, set the event_ids to be the ids of the events that needs to be deleted and new_items as an empty list. To insert new events, use episodic_memory_insert function.

    Args:
        event_ids (str): The ids of the episodic events to be deleted (or replaced).
        new_items (array): List of new episodic memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)
    occurred_at_override = getattr(self, "occurred_at", None)  # Optional timestamp override from API

    if self.user is None:
        raise ValueError("User is required to access episodic memory")

    if self.actor.organization_id is None:
        raise ValueError("Organization ID is required to access episodic memory")

    for event_id in event_ids:
        # It will raise an error if the event_id is not found in the episodic memory.
        self.episodic_memory_manager.get_episodic_memory_by_id(event_id, user=self.user)

    for event_id in event_ids:
        self.episodic_memory_manager.delete_event_by_id(event_id, actor=self.actor)

    for new_item in new_items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = occurred_at_override if occurred_at_override else new_item["occurred_at"]

        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        self.episodic_memory_manager.insert_event(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            timestamp=timestamp,  # Use potentially overridden timestamp
            event_type=new_item["event_type"],
            event_actor=new_item["actor"],
            summary=new_item["summary"],
            details=new_item["details"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def check_episodic_memory(self: "Agent", event_ids: List[str], timezone_str: str) -> List[EpisodicEventForLLM]:
    """
    The tool to check the episodic memory. This function will return the episodic events with the given event_ids.

    Args:
        event_ids (str): The ids of the episodic events to be checked.

    Returns:
        List[EpisodicEventForLLM]: List of episodic events with the given event_ids.
    """
    if self.user is None:
        raise ValueError("User is required to check episodic memory")

    episodic_memory = [
        self.episodic_memory_manager.get_episodic_memory_by_id(event_id, user=self.user, timezone_str=timezone_str)
        for event_id in event_ids
    ]

    formatted_results = [
        {
            "event_id": x.id,
            "timestamp": x.occurred_at,
            "event_type": x.event_type,
            "actor": x.actor,
            "summary": x.summary,
            "details": x.details,
        }
        for x in episodic_memory
    ]

    return formatted_results


def resource_memory_insert(self: "Agent", items: List[ResourceMemoryItemBase]):
    """
    The tool to insert new items into resource memory.

    Args:
        items (array): List of resource memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    # No imports needed - using agent instance attributes

    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    inserted_count = 0
    skipped_count = 0
    skipped_titles = []

    for item in items:
        # Check for existing similar resources (by title, summary, and filter_tags)
        existing_resources = self.resource_memory_manager.list_resources(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all resources
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )

        # Check if this resource already exists
        is_duplicate = False
        for existing in existing_resources:
            if (
                existing.title == item["title"]
                and existing.summary == item["summary"]
                and existing.content == item["content"]
            ):
                is_duplicate = True
                skipped_count += 1
                skipped_titles.append(item["title"])
                break

        if not is_duplicate:
            self.resource_memory_manager.insert_resource(
                actor=self.actor,
                agent_state=self.agent_state,
                agent_id=agent_id,
                title=item["title"],
                summary=item["summary"],
                resource_type=item["resource_type"],
                content=item["content"],
                organization_id=self.actor.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1

    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{t}'" for t in skipped_titles[:3])
        if len(skipped_titles) > 3:
            skipped_list += f" and {len(skipped_titles) - 3} more"
        return f"Inserted {inserted_count} new resource(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new resource(s)."
    else:
        return "No resources were inserted."


def resource_memory_update(self: "Agent", old_ids: List[str], new_items: List[ResourceMemoryItemBase]):
    """
    The tool to update and delete items in the resource memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new resource memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        self.resource_memory_manager.delete_resource_by_id(resource_id=old_id, actor=self.actor)

    for item in new_items:
        self.resource_memory_manager.insert_resource(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            title=item["title"],
            summary=item["summary"],
            resource_type=item["resource_type"],
            content=item["content"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def procedural_memory_insert(self: "Agent", items: List[ProceduralMemoryItemBase]):
    """
    The tool to insert new procedures into procedural memory. Note that the `summary` should not be a general term such as "guide" or "workflow" but rather a more informative description of the procedure.

    Args:
        items (array): List of procedural memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    inserted_count = 0
    skipped_count = 0
    skipped_summaries = []

    for item in items:
        # Check for existing similar procedures (by summary and filter_tags)
        existing_procedures = self.procedural_memory_manager.list_procedures(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all procedures
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )

        # Check if this procedure already exists
        is_duplicate = False
        for existing in existing_procedures:
            if existing.summary == item["summary"] and existing.steps == item["steps"]:
                is_duplicate = True
                skipped_count += 1
                skipped_summaries.append(item["summary"])
                break

        if not is_duplicate:
            self.procedural_memory_manager.insert_procedure(
                agent_state=self.agent_state,
                agent_id=agent_id,
                entry_type=item["entry_type"],
                summary=item["summary"],
                steps=item["steps"],
                actor=self.actor,
                organization_id=self.user.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1

    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{s}'" for s in skipped_summaries[:3])
        if len(skipped_summaries) > 3:
            skipped_list += f" and {len(skipped_summaries) - 3} more"
        return f"Inserted {inserted_count} new procedure(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new procedure(s)."
    else:
        return "No procedures were inserted."


def procedural_memory_update(self: "Agent", old_ids: List[str], new_items: List[ProceduralMemoryItemBase]):
    """
    The tool to update/delete items in the procedural memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new procedural memory items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        self.procedural_memory_manager.delete_procedure_by_id(procedure_id=old_id, actor=self.actor)

    for item in new_items:
        self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            agent_id=agent_id,
            entry_type=item["entry_type"],
            summary=item["summary"],
            steps=item["steps"],
            actor=self.actor,
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def check_semantic_memory(
    self: "Agent", semantic_item_ids: List[str], timezone_str: str
) -> List[SemanticMemoryItemBase]:
    """
    The tool to check the semantic memory. This function will return the semantic memory items with the given ids.

    Args:
        semantic_item_ids (str): The ids of the semantic memory items to be checked.

    Returns:
        List[SemanticMemoryItemBase]: List of semantic memory items with the given ids.
    """
    semantic_memory = [
        self.semantic_memory_manager.get_semantic_item_by_id(
            semantic_memory_id=id, user=self.user, timezone_str=timezone_str
        )
        for id in semantic_item_ids
    ]

    formatted_results = [
        {
            "semantic_item_id": x.id,
            "name": x.name,
            "summary": x.summary,
            "details": x.details,
            "source": x.source,
        }
        for x in semantic_memory
    ]

    return formatted_results


def semantic_memory_insert(self: "Agent", items: List[SemanticMemoryItemBase]):
    """
    The tool to insert items into semantic memory.

    Args:
        items (array): List of semantic memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    inserted_count = 0
    skipped_count = 0
    skipped_names = []

    for item in items:
        # Check for existing similar semantic items (by name, summary, and filter_tags)
        existing_items = self.semantic_memory_manager.list_semantic_items(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all items
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )

        # Check if this semantic item already exists
        is_duplicate = False
        for existing in existing_items:
            if (
                existing.name == item["name"]
                and existing.summary == item["summary"]
                and existing.details == item["details"]
            ):
                is_duplicate = True
                skipped_count += 1
                skipped_names.append(item["name"])
                break

        if not is_duplicate:
            self.semantic_memory_manager.insert_semantic_item(
                agent_state=self.agent_state,
                agent_id=agent_id,
                name=item["name"],
                summary=item["summary"],
                details=item["details"],
                source=item["source"],
                organization_id=self.actor.organization_id,
                actor=self.actor,  # Client for write operations
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1

    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{n}'" for n in skipped_names[:3])
        if len(skipped_names) > 3:
            skipped_list += f" and {len(skipped_names) - 3} more"
        return f"Inserted {inserted_count} new semantic item(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new semantic item(s)."
    else:
        return "No semantic items were inserted."


def semantic_memory_update(
    self: "Agent",
    old_semantic_item_ids: List[str],
    new_items: List[SemanticMemoryItemBase],
):
    """
    The tool to update/delete items in the semantic memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_semantic_item_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new semantic memory items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_semantic_item_ids:
        self.semantic_memory_manager.delete_semantic_item_by_id(semantic_memory_id=old_id, actor=self.actor)

    new_ids = []
    for item in new_items:
        inserted_item = self.semantic_memory_manager.insert_semantic_item(
            agent_state=self.agent_state,
            agent_id=agent_id,
            name=item["name"],
            summary=item["summary"],
            details=item["details"],
            source=item["source"],
            actor=self.actor,
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
        new_ids.append(inserted_item.id)

    message_to_return = (
        "Semantic memory with the following ids have been deleted: "
        + str(old_semantic_item_ids)
        + f". New semantic memory items are created: {str(new_ids)}"
    )
    return message_to_return


def knowledge_vault_insert(self: "Agent", items: List[KnowledgeVaultItemBase]):
    """
    The tool to update knowledge vault.

    Args:
        items (array): List of knowledge vault items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    inserted_count = 0
    skipped_count = 0
    skipped_captions = []

    for item in items:
        # Check for existing similar knowledge vault items (by caption, source, and filter_tags)
        existing_items = self.knowledge_vault_manager.list_knowledge(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all items
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )

        # Check if this knowledge vault item already exists
        is_duplicate = False
        for existing in existing_items:
            if (
                existing.caption == item["caption"]
                and existing.source == item["source"]
                and existing.secret_value == item["secret_value"]
            ):
                is_duplicate = True
                skipped_count += 1
                skipped_captions.append(item["caption"])
                break

        if not is_duplicate:
            self.knowledge_vault_manager.insert_knowledge(
                actor=self.actor,
                agent_state=self.agent_state,
                agent_id=agent_id,
                entry_type=item["entry_type"],
                source=item["source"],
                sensitivity=item["sensitivity"],
                secret_value=item["secret_value"],
                caption=item["caption"],
                organization_id=self.actor.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1

    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{c}'" for c in skipped_captions[:3])
        if len(skipped_captions) > 3:
            skipped_list += f" and {len(skipped_captions) - 3} more"
        return f"Inserted {inserted_count} new knowledge vault item(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new knowledge vault item(s)."
    else:
        return "No knowledge vault items were inserted."


def knowledge_vault_update(self: "Agent", old_ids: List[str], new_items: List[KnowledgeVaultItemBase]):
    """
    The tool to update/delete items in the knowledge vault. To update the knowledge_vault, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new knowledge vault items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        self.knowledge_vault_manager.delete_knowledge_by_id(knowledge_vault_item_id=old_id, actor=self.actor)

    for item in new_items:
        self.knowledge_vault_manager.insert_knowledge(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            entry_type=item["entry_type"],
            source=item["source"],
            sensitivity=item["sensitivity"],
            secret_value=item["secret_value"],
            caption=item["caption"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def trigger_memory_update_with_instruction(
    self: "Agent", user_message: object, instruction: str, memory_type: str
) -> Optional[str]:
    """
    Choose which memory to update. The function will trigger one specific memory agent with the instruction telling the agent what to do.

    Args:
        instruction (str): The instruction to the memory agent.
        memory_type (str): The type of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge_vault", "semantic". For instance, ['episodic', 'resource'].

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix.local_client import create_client

    client = create_client()
    agents = client.list_agents()

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}")

    # Fallback to sequential processing for backward compatibility
    response = ""

    if memory_type == "core":
        agent_type = "core_memory_agent"
    elif memory_type == "episodic":
        agent_type = "episodic_memory_agent"
    elif memory_type == "resource":
        agent_type = "resource_memory_agent"
    elif memory_type == "procedural":
        agent_type = "procedural_memory_agent"
    elif memory_type == "knowledge_vault":
        agent_type = "knowledge_vault_memory_agent"
    elif memory_type == "semantic":
        agent_type = "semantic_memory_agent"
    else:
        raise ValueError(
            f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'."
        )

    matching_agent = None
    for agent in agents:
        if agent.agent_type == agent_type:
            matching_agent = agent
            break

    if matching_agent is None:
        raise ValueError(f"No agent found with type '{agent_type}'")

    client.send_message(
        role="user",
        user_id=self.user.id,
        agent_id=matching_agent.id,
        message="[Message from Chat Agent (Now you are allowed to make multiple function calls sequentially)] "
        + instruction,
        existing_file_uris=user_message["existing_file_uris"],
        retrieved_memories=user_message.get("retrieved_memories", None),
    )
    response += "[System Message] Agent " + matching_agent.name + " has been triggered to update the memory.\n"

    return response.strip()


def trigger_memory_update(self: "Agent", user_message: object, memory_types: List[str]) -> Optional[str]:
    """
    Choose which memory to update. This function will trigger another memory agent which is specifically in charge of handling the corresponding memory to update its memory. Trigger all necessary memory updates at once. Put the explanations in the `internal_monologue` field.

    Args:
        memory_types (List[str]): The types of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge_vault", "semantic". For instance, ['episodic', 'resource'].

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix.agent import (
        CoreMemoryAgent,
        EpisodicMemoryAgent,
        KnowledgeVaultAgent,
        ProceduralMemoryAgent,
        ResourceMemoryAgent,
        SemanticMemoryAgent,
    )

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}")

    # De-duplicate memory types while preserving order.
    # The MetaMemoryAgent (LLM) can occasionally emit duplicates (e.g., ["semantic", "semantic"]).
    # Running duplicates in parallel can cause races (e.g., double-deletes).
    original_memory_types = list(memory_types)
    memory_types = list(dict.fromkeys(memory_types))
    if len(memory_types) != len(original_memory_types):
        logger.debug(
            "De-duplicated trigger_memory_update memory_types: %s -> %s",
            original_memory_types,
            memory_types,
        )

    # Map memory types to agent classes
    memory_type_to_agent_class = {
        "core": CoreMemoryAgent,
        "episodic": EpisodicMemoryAgent,
        "resource": ResourceMemoryAgent,
        "procedural": ProceduralMemoryAgent,
        "knowledge_vault": KnowledgeVaultAgent,
        "semantic": SemanticMemoryAgent,
    }

    # Validate memory types
    for memory_type in memory_types:
        if memory_type not in memory_type_to_agent_class:
            raise ValueError(
                f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'."
            )

    # Get child agents
    child_agent_states = self.agent_manager.list_agents(parent_id=self.agent_state.id, actor=self.actor)

    # Map agent types to agent states
    agent_type_to_state = {agent_state.agent_type: agent_state for agent_state in child_agent_states}

    # Capture trace context from main thread BEFORE spawning worker threads
    # ContextVars don't automatically propagate to ThreadPoolExecutor workers
    parent_trace_context = get_trace_context()
    langfuse = get_langfuse_client()

    def _run_single_memory_update(memory_type: str) -> str:
        # ThreadPoolExecutor reuses worker threads; ContextVars can leak between runs unless cleared.
        try:
            # Restore trace context in worker thread (ContextVars are thread-local)
            # This is for LangFuse hierarchical tracing
            if parent_trace_context.get("trace_id"):
                set_trace_context(
                    trace_id=parent_trace_context.get("trace_id"),
                    observation_id=parent_trace_context.get("observation_id"),
                    user_id=parent_trace_context.get("user_id"),
                    session_id=parent_trace_context.get("session_id"),
                )

            agent_class = memory_type_to_agent_class[memory_type]
            agent_type_str = f"{memory_type}_memory_agent"

            agent_state = agent_type_to_state.get(agent_type_str)
            if agent_state is None:
                raise ValueError(f"No agent found with type '{agent_type_str}'")

            # Get filter_tags, use_cache, client_id, user_id, and occurred_at from parent agent instance
            # Deep copy filter_tags to ensure complete isolation between child agents
            parent_filter_tags = getattr(self, "filter_tags", None)
            # Don't use 'or {}' because empty dict {} is valid and different from None
            filter_tags = deepcopy(parent_filter_tags) if parent_filter_tags is not None else None
            use_cache = getattr(self, "use_cache", True)
            actor = getattr(self, "actor", None)
            user = getattr(self, "user", None)
            occurred_at = getattr(self, "occurred_at", None)  # Get occurred_at from parent agent

            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Creating {memory_type} agent with filter_tags={filter_tags}, client_id={actor.id if actor else None}, user_id={user.id if user else None}, occurred_at={occurred_at}"
            )

            memory_agent = agent_class(
                agent_state=agent_state,
                interface=self.interface,
                actor=actor,
                user=user,
                filter_tags=filter_tags,
                use_cache=use_cache,
            )

            # Set occurred_at on the child agent so it can use it during memory operations
            if occurred_at is not None:
                memory_agent.occurred_at = occurred_at

            # Work on a copy of the user message so parallel updates do not interfere
            if "message" not in user_message:
                raise KeyError("user_message must contain a 'message' field")

            if hasattr(user_message["message"], "model_copy"):
                message_copy = user_message["message"].model_copy(deep=True)  # type: ignore[attr-defined]
            else:
                message_copy = deepcopy(user_message["message"])

            system_msg = TextContent(
                text=(
                    "[System Message] According to the instructions, the retrieved memories "
                    "and the above content, update the corresponding memory."
                )
            )

            if isinstance(message_copy.content, str):
                message_copy.content = [
                    TextContent(text=message_copy.content),
                    system_msg,
                ]
            elif isinstance(message_copy.content, list):
                message_copy.content = list(message_copy.content) + [system_msg]
            else:
                message_copy.content = [system_msg]

            # Extract topics and retrieved_memories from parent agent to pass to sub-agents
            # This ensures sub-agents use the same keywords for memory retrieval
            retrieved_memories = user_message.get("retrieved_memories", None)
            topics = None
            if retrieved_memories and isinstance(retrieved_memories, dict):
                topics = retrieved_memories.get("key_words", None)

            # Pass actor (Client) and user (User) to memory agent
            # actor is needed for write operations, user is needed for read operations

            # Wrap child agent execution in a LangFuse span for hierarchical tracing
            trace_id = parent_trace_context.get("trace_id") if parent_trace_context else None
            parent_span_id = parent_trace_context.get("observation_id") if parent_trace_context else None
            if langfuse and trace_id:
                from typing import cast

                from langfuse.types import TraceContext

                # Format span name as Title Case (e.g., "Episodic Memory Agent")
                span_name = agent_type_str.replace("_", " ").title()

                # Build trace_context with parent_span_id (per Langfuse v3 docs)
                trace_context_dict: dict = {"trace_id": trace_id}
                if parent_span_id:
                    trace_context_dict["parent_span_id"] = parent_span_id

                # Use context manager for proper OTel context propagation
                with langfuse.start_as_current_observation(
                    name=span_name,
                    as_type="agent",
                    trace_context=cast(TraceContext, trace_context_dict),
                    metadata={
                        "memory_type": memory_type,
                        "agent_name": agent_state.name,
                    },
                ) as span:
                    mark_observation_as_child(span)

                    # Get this span's ID for child operations
                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        logger.debug(
                            f"Child agent span created: agent={agent_type_str}, "
                            f"span_id={span_observation_id}, parent={parent_span_id}"
                        )
                        # Update ContextVar so child LLM calls use this span as parent
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=parent_trace_context.get("user_id"),
                            session_id=parent_trace_context.get("session_id"),
                        )

                    memory_agent.step(
                        input_messages=message_copy,
                        chaining=user_message.get("chaining", False),
                        actor=actor,
                        user=user,
                        topics=topics,
                        retrieved_memories=retrieved_memories,
                    )
            else:
                # No tracing available, run directly
                memory_agent.step(
                    input_messages=message_copy,
                    chaining=user_message.get("chaining", False),
                    actor=actor,
                    user=user,
                    topics=topics,
                    retrieved_memories=retrieved_memories,
                )

            return f"[System Message] Agent {agent_state.name} has been triggered to update the memory.\n"
        finally:
            clear_trace_context()

    max_workers = min(len(memory_types), max(os.cpu_count() or 1, 1))
    responses: dict[int, str] = {}

    if not memory_types:
        return ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_run_single_memory_update, memory_type): index
            for index, memory_type in enumerate(memory_types)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            memory_type = memory_types[index]
            try:
                responses[index] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to trigger memory update for '{memory_type}'") from exc

    ordered_responses = [responses[i] for i in range(len(memory_types)) if i in responses]
    return "".join(ordered_responses).strip()


def finish_memory_update(self: "Agent"):
    """
    Finish the memory update process. This function should be called after the Memory is updated.

    Note: This function takes no parameters. Call it without any arguments.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None
