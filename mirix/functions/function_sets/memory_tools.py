import asyncio
import functools
import os
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple

from mirix.agent import Agent, AgentState

if TYPE_CHECKING:
    from mirix.schemas.memory import Memory
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
from mirix.schemas.procedural_memory import SkillCitationForLLM, SkillExampleForLLM

from mirix.schemas.resource_memory import ResourceMemoryItemBase
from mirix.schemas.semantic_memory import SemanticMemoryItemBase

logger = get_logger(__name__)


async def core_memory_append(
    self: "Agent", blocks_in_memory: "Memory", label: str, content: str
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
        raise ValueError(
            "You should not include 'Line n:' (here n is a number) in the content."
        )

    # Get the current block and its limit
    current_block = blocks_in_memory.get_block(label)
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
    blocks_in_memory.update_block_value(label=label, value=new_value)
    return None


async def core_memory_rewrite(
    self: "Agent", blocks_in_memory: "Memory", label: str, content: str
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
    current_block = blocks_in_memory.get_block(label)
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
        blocks_in_memory.update_block_value(label=label, value=new_value)
        # Provide feedback on the operation
        percentage = int((new_length / limit) * 100)
        return f"Successfully rewrote '{label}' block: {new_length}/{limit} characters ({percentage}% full)."

    return None


async def episodic_memory_insert(self: "Agent", items: List[EpisodicEventForLLM]):
    """
    The tool to update episodic memory. The item being inserted into the episodic memory is an event either happened on the user or the assistant.

    Args:
        items (array): List of episodic memory items to insert.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)
    occurred_at_override = getattr(
        self, "occurred_at", None
    )  # Optional timestamp override from API

    for item in items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = (
            occurred_at_override if occurred_at_override else item["occurred_at"]
        )

        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        try:
            await self.episodic_memory_manager.insert_event(
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
        except Exception as e:
            print(
                f"[episodic_memory_insert] insert_event FAILED for item {item!r}: {e}"
            )
            traceback.print_exc()
            raise

    response = "Events inserted! Now you need to check if there are repeated events shown in the system prompt."
    return response


async def episodic_memory_merge(
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

    try:
        episodic_memory = await self.episodic_memory_manager.update_event(
            event_id=event_id,
            new_summary=combined_summary,
            new_details=combined_details,
            actor=self.actor,
            agent_state=self.agent_state,
            update_mode="replace",
        )
    except Exception as e:
        print(
            f"[episodic_memory_merge] update_event FAILED for event_id "
            f"{event_id!r}: {e}"
        )
        traceback.print_exc()
        raise
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


async def episodic_memory_replace(
    self: "Agent", event_ids: List[str], new_items: List[EpisodicEventForLLM]
):
    """
    The tool to replace or delete items in the episodic memory. To replace the memory, set the event_ids to be the ids of the events that needs to be replaced and new_items as the updated events. Note that the number of new items does not need to be the same as the number of event_ids as it is not a one-to-one mapping. To delete the memory, set the event_ids to be the ids of the events that needs to be deleted and new_items as an empty list. To insert new events, use episodic_memory_insert function.

    Args:
        event_ids (str): The ids of the episodic events to be deleted (or replaced).
        new_items (array): List of new episodic memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)
    occurred_at_override = getattr(
        self, "occurred_at", None
    )  # Optional timestamp override from API

    if self.user is None:
        raise ValueError("User is required to access episodic memory")

    if self.actor.organization_id is None:
        raise ValueError("Organization ID is required to access episodic memory")

    for event_id in event_ids:
        # It will raise an error if the event_id is not found in the episodic memory.
        await self.episodic_memory_manager.get_episodic_memory_by_id(
            event_id, user=self.user
        )

    for event_id in event_ids:
        try:
            await self.episodic_memory_manager.delete_event_by_id(
                event_id, actor=self.actor
            )
        except Exception as e:
            print(
                f"[episodic_memory_replace] delete_event_by_id FAILED for "
                f"event_id {event_id!r}: {e}"
            )
            traceback.print_exc()
            raise

    for new_item in new_items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = (
            occurred_at_override if occurred_at_override else new_item["occurred_at"]
        )

        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        try:
            await self.episodic_memory_manager.insert_event(
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
        except Exception as e:
            print(
                f"[episodic_memory_replace] insert_event FAILED for item "
                f"{new_item!r}: {e}"
            )
            traceback.print_exc()
            raise


async def check_episodic_memory(
    self: "Agent", event_ids: List[str], timezone_str: str
) -> List[EpisodicEventForLLM]:
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
        await self.episodic_memory_manager.get_episodic_memory_by_id(
            event_id, user=self.user, timezone_str=timezone_str
        )
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


async def resource_memory_insert(self: "Agent", items: List[ResourceMemoryItemBase]):
    """
    The tool to insert new items into resource memory.

    Args:
        items (array): List of resource memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    # No imports needed - using agent instance attributes

    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

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
        existing_resources = await self.resource_memory_manager.list_resources(
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
            try:
                await self.resource_memory_manager.insert_resource(
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
            except Exception as e:
                print(
                    f"[resource_memory_insert] insert_resource FAILED for item "
                    f"{item!r}: {e}"
                )
                traceback.print_exc()
                raise
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


async def resource_memory_update(
    self: "Agent", old_ids: List[str], new_items: List[ResourceMemoryItemBase]
):
    """
    The tool to update and delete items in the resource memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new resource memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        await self.resource_memory_manager.delete_resource_by_id(
            resource_id=old_id, actor=self.actor
        )

    for item in new_items:
        await self.resource_memory_manager.insert_resource(
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


def _bump_patch_version(version: str) -> str:
    """Increment patch version: '0.1.0' -> '0.1.1'.

    Args:
        version (str): Semver version string to bump.
    """
    try:
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return "0.1.1"


# ---------------------------------------------------------------------------
# Skill tools.
#
# Inside a skill learning run the curator attaches a CuratorBatchState to the
# agent instance (``self.curator_batch``, see
# mirix.services.skill_curator_state): the edit budget of the batch (one
# create/edit per experience), the run's delete limit (one soft delete per
# learning run), the write guard (the run must still be active) and the output
# cap. Every validation — duplicate names included — runs BEFORE a write,
# and only a write that went through costs budget: a rejected call is free.
# Outside a run (the in-flow procedural agent) there is no budget and no delete
# limit. Every tool works inside the agent's scope (``filter_tags["scope"]``):
# a skill of another scope is "not found" by id and by name.
# ---------------------------------------------------------------------------

_SKILL_TEXT_FIELDS = ("name", "description", "instructions")
_SKILL_JSON_FIELDS = ("triggers", "examples")
_SKILL_EDITABLE_FIELDS = _SKILL_TEXT_FIELDS + _SKILL_JSON_FIELDS + ("entry_type",)


def _skill_provenance(self: "Agent") -> dict:
    """Who is changing a skill, for its version record.

    Inside a skill learning run the curator sets ``skill_run_id`` on the agent
    instance (the curator batch id); the change is then a ``curator`` change of
    that run. Outside a run (the in-flow procedural agent) it is an ``agent``
    change with no run.
    """
    from mirix.services.procedural_memory_manager import ACTOR_AGENT, ACTOR_CURATOR

    run_id = getattr(self, "skill_run_id", None)
    return {
        "run_id": run_id,
        "actor_kind": ACTOR_CURATOR if run_id else ACTOR_AGENT,
    }


def _curator_batch(self: "Agent"):
    """The curator batch this agent is working on, or None outside a run."""
    return getattr(self, "curator_batch", None)


def _skill_scopes(self: "Agent") -> Optional[List[str]]:
    """The scope the skill tools work in: the agent's
    ``filter_tags["scope"]`` (the client's write scope, stamped by the REST
    layer and by the curator), or None — no scope restriction — for an agent
    without one."""
    filter_tags = getattr(self, "filter_tags", None)
    scope = filter_tags.get("scope") if isinstance(filter_tags, dict) else None
    return [str(scope)] if scope else None


def _skill_owner_id(self: "Agent") -> str:
    user = getattr(self, "user", None)
    if user is not None and getattr(user, "id", None):
        return user.id
    user_id = getattr(self, "user_id", None)
    if user_id:
        return user_id
    from mirix.services.user_manager import UserManager

    return UserManager.ADMIN_USER_ID


def _skill_write_options(self: "Agent", *, cited: Sequence[str] = ()) -> dict:
    """Keyword arguments of every skill write: provenance, the scope, and —
    in a learning run — the run's write guard, the batch lineage and the
    experiences the write cites."""
    options = dict(_skill_provenance(self))
    options["scopes"] = _skill_scopes(self)
    batch = _curator_batch(self)
    if batch is not None:
        options["write_guard"] = batch.write_guard
        options["lineage_experience_ids"] = list(batch.experience_ids)
        if cited:
            options["cited_experience_ids"] = list(cited)
    return options


# What a rejected call of each skill tool left unchanged.
_NOTHING_DONE = {
    "skill_create": "Nothing was created",
    "skill_edit": "Nothing was changed",
    "skill_delete": "Nothing was deleted",
    "experience_triage": "Nothing was recorded",
}

# At most this many ids are named in one tool answer.
_IDS_SHOWN = 50


def _experience_ids_arg(value) -> Tuple[List[str], Optional[str]]:
    """``(ids, problem)`` of an ``experience_ids``-style argument: None or ""
    is no id; a JSON list given as a string is decoded; a bare id string is
    one id; a list of strings is stripped, its empty entries (a strict-mode
    filler such as ``[""]``) dropped and repeats removed, in order. Anything
    else is a problem."""
    import json as _json

    problem = "must be a list of experience ids."
    if value is None:
        return [], None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [], None
        if text.startswith("["):
            try:
                value = _json.loads(text)
            except (TypeError, ValueError):
                return [], problem
        else:
            return [text], None
    if not isinstance(value, (list, tuple)):
        return [], problem
    ids: List[str] = []
    for item in value:
        if item is None:
            continue
        if not isinstance(item, str):
            return [], problem
        item = item.strip()
        if item and item not in ids:
            ids.append(item)
    return ids, None


def _named(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_IDS_SHOWN])
    if len(ids) > _IDS_SHOWN:
        shown += f" … and {len(ids) - _IDS_SHOWN} more"
    return shown


def _citations_arg(value) -> Tuple[List[str], Dict[str, List[str]], Optional[str]]:
    """``(ids, quotes, problem)`` of a skill write's ``experience_ids``:
    a list of ``{"id", "quote"}`` objects — or, as before, of bare ids (a
    citation without a quote) — in order, repeats merged (their quotes kept);
    a JSON list given as a string is decoded and a bare id string is one id.
    An entry whose id is empty (a strict-mode filler) is dropped. ``quotes``
    maps an id to the quotes given for it. Anything else is a problem."""
    import json as _json

    problem = 'must be a list of objects with an "id" and a "quote".'
    if value is None:
        return [], {}, None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [], {}, None
        if not text.startswith("["):
            return [text], {}, None
        try:
            value = _json.loads(text)
        except (TypeError, ValueError):
            return [], {}, problem
    if not isinstance(value, (list, tuple)):
        return [], {}, problem
    ids: List[str] = []
    quotes: Dict[str, List[str]] = {}
    for item in value:
        if item is None:
            continue
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, str):
            experience_id, quote = item.strip(), None
        elif isinstance(item, dict):
            raw_id = item.get("id", item.get("experience_id"))
            raw_quote = item.get("quote")
            if raw_id is None:
                raw_id = ""
            if not isinstance(raw_id, str) or not (
                raw_quote is None or isinstance(raw_quote, str)
            ):
                return [], {}, problem
            experience_id, quote = raw_id.strip(), raw_quote
        else:
            return [], {}, problem
        if not experience_id:
            continue
        if experience_id not in ids:
            ids.append(experience_id)
        if quote is not None and quote.strip():
            quotes.setdefault(experience_id, []).append(quote)
    return ids, quotes, None


def _citation_problem(
    self: "Agent", tool: str, experience_ids
) -> Tuple[List[str], Dict[str, List[str]], Optional[str]]:
    """The ids a skill write cites and the quotes given for them, or the
    rejection message (recorded; it costs nothing) when the
    argument is malformed or names an id this batch did not show. Outside a
    curation batch the argument is ignored: ``(ids, quotes, None)``, and the
    caller notes it when ids were given."""
    from mirix.services.skill_curator_state import REJECT_INVALID

    ids, quotes, problem = _citations_arg(experience_ids)
    batch = _curator_batch(self)
    if batch is None:
        return ids, quotes, None
    nothing = _NOTHING_DONE.get(tool, "Nothing was changed")
    if problem:
        return [], {}, _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            f"Rejected: 'experience_ids' {problem} {nothing} and this call cost no budget.",
        )
    unknown = batch.unknown_ids(ids)
    if unknown:
        return [], {}, _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            f"Rejected: experience_ids names {', '.join(unknown[:10])}"
            f"{' …' if len(unknown) > 10 else ''}, which this batch did not show. "
            "Cite only the ids of this batch's experiences and earlier lessons. "
            f"{nothing} and this call cost no budget.",
        )
    return ids, quotes, None


def _verified_citations(
    cited: Sequence[str],
    quotes: Dict[str, List[str]],
    new_text: str,
    old_text: str = "",
) -> Tuple[List[str], Dict[str, str]]:
    """``(verified, unverified)`` of a write's citations: a citation
    counts when one of its quotes occurs in the text the write added or
    changed (``new_text``) and is not text the skill already had
    (``old_text``) — :func:`skill_text_checks.citation_quote_problem`;
    ``unverified`` maps every other cited id to why not."""
    from mirix.services.skill_text_checks import QUOTE_MISSING, citation_quote_problem

    verified: List[str] = []
    unverified: Dict[str, str] = {}
    for experience_id in cited:
        reason = QUOTE_MISSING
        for quote in quotes.get(experience_id, ()):
            reason = citation_quote_problem(quote, new_text, old_text)
            if reason is None:
                break
        if reason is None:
            verified.append(experience_id)
        else:
            unverified[experience_id] = reason
    return verified, unverified


def _unverified_note(self: "Agent", unverified: Dict[str, str], skill_id: str) -> str:
    """The answer's note on citations whose quote does not show where the
    write applies them: they were not recorded and stay pending — an
    experience of the batch is not consumed by it unless a change cites it
    with a quote or the triage lists it, and an earlier lesson stays in the
    pool as it was."""
    from mirix.services.skill_text_checks import CITATION_QUOTE_MIN_CHARS

    if not unverified:
        return ""
    batch = _curator_batch(self)
    if batch is not None:
        batch.record_unverified_citations(unverified, skill_id)
    listed = "; ".join(
        f"{experience_id} ({reason})"
        for experience_id, reason in list(unverified.items())[:_IDS_SHOWN]
    )
    more = len(unverified) - _IDS_SHOWN
    if more > 0:
        listed += f" … and {more} more"
    return (
        f" NOT counted as applied, because no quote of the text this change "
        f"adds shows where it applies them: {listed}. They stay pending: cite "
        "them on a change with a quote copied from the words it adds that apply "
        f"their lesson (at least {CITATION_QUOTE_MIN_CHARS} letters), or list "
        "them with experience_triage."
    )


def _citation_note(
    self: "Agent",
    cited: Sequence[str],
    skill_id: str,
    *,
    delete: bool = False,
    unverified: Optional[Dict[str, str]] = None,
) -> str:
    """Record a write's (``delete``: a soft delete's) citations in its batch
    and say what it cited; ``unverified`` are citations it gave without a
    quote of its new text that applies them: not recorded."""
    batch = _curator_batch(self)
    if batch is None:
        if cited or unverified:
            return " (experience_ids are ignored outside a skill curation batch.)"
        return ""
    if cited or not unverified:
        # A write whose citations all lacked a verifiable quote is counted
        # there (citations_unverified), not also as an uncited write.
        batch.record_citations(list(cited), skill_id, delete=delete)
    note = _unverified_note(self, unverified or {}, skill_id)
    if cited:
        return f" Cited: {', '.join(cited)}." + note
    if unverified:
        return note
    return (
        " It cites no experience: name the ones it addresses in experience_ids "
        "next time, or list them in experience_triage."
    )


def _skill_output_cap(self: "Agent") -> int:
    from mirix.services.skill_curator_state import skill_output_char_cap

    batch = _curator_batch(self)
    if batch is not None:
        return batch.output_char_cap
    return skill_output_char_cap(self.agent_state.llm_config)


def _skill_reject(self: "Agent", tool: str, kind: str, message: str) -> str:
    """A rejected skill tool call: nothing changed and nothing is charged. The
    agent counts it (``declined_tool_calls``): a terminal call the step runs
    after this one in the same turn waits until the model read the answer."""
    try:
        self.declined_tool_calls = getattr(self, "declined_tool_calls", 0) + 1
    except AttributeError:  # an agent object that takes no attributes
        pass
    batch = _curator_batch(self)
    if batch is not None:
        return batch.reject(tool, kind, message)
    return message


def _budget_note(self: "Agent") -> str:
    batch = _curator_batch(self)
    if batch is None:
        return ""
    return f" Edit budget left in this batch: {batch.remaining} of {batch.budget}."


def _instructions_too_long(self: "Agent", tool: str, length: int) -> str:
    from mirix.constants import SKILL_MAX_INSTRUCTIONS_CHARS
    from mirix.services.skill_curator_state import REJECT_TOO_LONG

    return _skill_reject(
        self,
        tool,
        REJECT_TOO_LONG,
        f"Rejected: the instructions would be {length} characters, over the "
        f"{SKILL_MAX_INSTRUCTIONS_CHARS}-character limit of a skill. Please "
        "condense them (remove repetition, keep the essential steps and caveats) "
        "or move a separate topic into its own skill. Nothing was changed and "
        "this call cost no budget.",
    )


@functools.lru_cache(maxsize=1)
def _skill_param_docs() -> dict:
    """skill_create's parameter descriptions by name (parsed once): the tool's
    own vocabulary, which a schema echo repeats (skill_text_checks)."""
    from docstring_parser import parse

    return {
        param.arg_name: param.description or ""
        for param in parse(skill_create.__doc__ or "").params
    }


def _content_problem(
    field: str, value, *, min_words: int = 0, doc: str = ""
) -> Optional[str]:
    """Why ``value`` is a placeholder or an echo of the tool's schema rather
    than content for ``field`` (skill_text_checks), or None."""
    from mirix.services.skill_text_checks import placeholder_problem

    return placeholder_problem(
        field, value, _skill_param_docs().get(doc or field, ""), min_words=min_words
    )


def _validate_skill_fields(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    entry_type: Optional[str] = None,
    triggers=None,
    examples=None,
    check: tuple = (),
) -> Optional[str]:
    """The first problem with the given skill fields, or None.

    Beyond types and the schema's limits, every text must be content: not a
    placeholder ("...", "string"), not the words of the tool's own parameter
    description ("kebab-case name", "detailed plain text"), not a stub that
    says it is one, and a description / instructions of at least a few words
    (skill_text_checks). This holds for every caller of the skill tools (a
    curation batch, the in-flow procedural agent, the auto-dream agent)."""
    from mirix.schemas.procedural_memory import (
        SKILL_ENTRY_TYPES,
        SKILL_MAX_DESCRIPTION_LEN,
        SKILL_MAX_EXAMPLES_COUNT,
        SKILL_MAX_NAME_LEN,
        SKILL_MAX_TRIGGER_LEN,
        skill_nul_problem,
    )
    from mirix.services.skill_text_checks import (
        SKILL_MIN_DESCRIPTION_WORDS,
        SKILL_MIN_INSTRUCTIONS_WORDS,
    )

    # A NUL character (U+0000) cannot be stored: a
    # rejection the model can act on, before any embedding or write.
    given = {
        "name": name,
        "description": description,
        "instructions": instructions,
        "triggers": triggers,
        "examples": examples,
    }
    problem = skill_nul_problem({f: v for f, v in given.items() if f in check})
    if problem:
        return problem
    if "name" in check:
        if not isinstance(name, str) or not name.strip():
            return "'name' must be a non-empty string."
        if len(name) > SKILL_MAX_NAME_LEN:
            return f"'name' exceeds {SKILL_MAX_NAME_LEN} characters."
        problem = _content_problem("name", name)
        if problem:
            return problem
    if "description" in check:
        if not isinstance(description, str) or not description.strip():
            return "'description' must be a non-empty string."
        if len(description) > SKILL_MAX_DESCRIPTION_LEN:
            return f"'description' exceeds {SKILL_MAX_DESCRIPTION_LEN} characters."
        problem = _content_problem(
            "description", description, min_words=SKILL_MIN_DESCRIPTION_WORDS
        )
        if problem:
            return problem
    if "instructions" in check:
        if not isinstance(instructions, str) or not instructions.strip():
            return "'instructions' must be a non-empty string."
        problem = _content_problem(
            "instructions", instructions, min_words=SKILL_MIN_INSTRUCTIONS_WORDS
        )
        if problem:
            return problem
    if "entry_type" in check and entry_type not in SKILL_ENTRY_TYPES:
        return f"'entry_type' must be one of: {', '.join(sorted(SKILL_ENTRY_TYPES))}."
    # The schema's limits (mirix/schemas/procedural_memory.py) are checked here,
    # before the write: past them, the write's own validation would fail only
    # after the retrieval vector was computed, as a tool error instead of a
    # rejection the model can act on.
    if "triggers" in check and triggers is not None:
        if not isinstance(triggers, list) or not all(isinstance(t, str) for t in triggers):
            return "'triggers' must be a list of strings."
        for i, trigger in enumerate(triggers):
            if len(trigger) > SKILL_MAX_TRIGGER_LEN:
                return (
                    f"'triggers[{i}]' exceeds {SKILL_MAX_TRIGGER_LEN} characters: a "
                    "trigger is one short condition."
                )
            if not _TWO_LETTERS.search(trigger):
                return (
                    f"'triggers[{i}]' ({trigger.strip()[:40]!r}) has no word: a "
                    "trigger is a short condition in words."
                )
            problem = _content_problem(f"triggers[{i}]", trigger, doc="triggers")
            if problem:
                return problem
    if "examples" in check and examples is not None:
        if not isinstance(examples, list) or not all(isinstance(e, dict) for e in examples):
            return "'examples' must be a list of objects."
        if len(examples) > SKILL_MAX_EXAMPLES_COUNT:
            return (
                f"'examples' has {len(examples)} entries; at most "
                f"{SKILL_MAX_EXAMPLES_COUNT} are allowed."
            )
        for i, example in enumerate(examples):
            if set(example) != {"input", "output"} or not all(
                isinstance(example[k], str) for k in example
            ):
                return (
                    f"'examples[{i}]' must be an object with exactly two strings, "
                    '"input" and "output"; got keys '
                    f"{', '.join(sorted(map(str, example))) or 'none'}."
                )
            for key in ("input", "output"):
                problem = _content_problem(
                    f"examples[{i}].{key}", example[key], doc="examples"
                )
                if problem:
                    return problem
    return None


# A trigger needs a word: two letters in a row (any script).
_TWO_LETTERS = re.compile(r"[^\W\d_]{2}")


def _without_list_fillers(triggers, examples):
    """skill_create's ``triggers`` and ``examples`` without their blank
    entries. A strict tool schema makes a model send both lists even when it
    has none, and a blank entry (``[""]``, an example whose texts are all
    empty) is that filler, as for ``experience_ids``: it carries nothing, so
    it is dropped rather than stored or rejected. Any other entry is left to
    the validation (half an example is checked, not dropped)."""

    def blank(value) -> bool:
        return isinstance(value, str) and not value.strip()

    if isinstance(triggers, list):
        triggers = [t for t in triggers if not blank(t)]
    if isinstance(examples, list):
        examples = [
            e
            for e in examples
            if not (isinstance(e, dict) and all(blank(v) for v in e.values()))
        ]
    return triggers, examples


def _format_skill_line(skill) -> str:
    from mirix import constants

    line = (
        f"[ID: {skill.id}] {skill.name} (v{skill.version}) - "
        f"{skill.entry_type}: {skill.description}"
    )
    length = len(getattr(skill, "instructions", None) or "")
    if length > constants.SKILL_TARGET_INSTRUCTIONS_CHARS:
        line += f" [instructions {length} chars: over target]"
    return line


async def _skill_usage_records(self: "Agent", skill_ids: Sequence[str]) -> dict:
    """The usage records of ``skill_ids`` for the skills' owner:
    ``{skill_id: SkillUsageRecord}``. Rows are keyed like the distiller writes
    them: under the client's organization, else the default one
    (``owner_org``) — never the user's, which the distiller never uses."""
    from mirix.constants import DEFAULT_ORG_ID
    from mirix.services.skill_session_feedback import SkillSessionFeedbackManager

    actor = getattr(self, "actor", None)
    organization_id = getattr(actor, "organization_id", None) or DEFAULT_ORG_ID
    manager = SkillSessionFeedbackManager(
        session_maker=getattr(self.procedural_memory_manager, "session_maker", None)
    )
    return await manager.usage_records(
        organization_id=organization_id,
        user_id=_skill_owner_id(self),
        skill_ids=list(skill_ids),
    )


def _size_note(length: int) -> str:
    """Advice when a skill's instructions are over their soft target,
    or "": a retrieved skill is injected whole, so it should stay compact."""
    from mirix import constants

    target = constants.SKILL_TARGET_INSTRUCTIONS_CHARS
    if length <= target:
        return ""
    return (
        f" Its instructions are now {length} characters, over the {target}-character "
        "target of a skill (a retrieved skill is placed whole in the agent's "
        "context): in a later change condense them (merge overlapping rules, drop "
        "details that belong to one case) or move a separate kind of task into its "
        "own skill."
    )


def _retrieval_text_note(self: "Agent", skill_id: str) -> str:
    """After an instructions edit: retrieval never reads the
    instructions, so a change of what the skill covers needs its retrieval
    text changed too; in a batch that follow-up costs no budget."""
    note = (
        " Retrieval finds a skill only by its name, description and triggers, and "
        "this change left them as they were: if it changed which tasks the skill "
        "covers or when it applies, update the description or triggers to match."
    )
    batch = _curator_batch(self)
    if batch is not None:
        from mirix.services.skill_curator_state import RETRIEVAL_TEXT_FIELDS

        free = [f for f in RETRIEVAL_TEXT_FIELDS if batch.is_free_followup(skill_id, f)]
        if free:
            note += (
                f" In this batch one follow-up change of its {_or_list(free)} "
                "costs no edit budget (once per field)."
            )
    return note


def _or_list(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


async def skill_list(self: "Agent", query: str = "", offset: int = 0) -> str:
    """
    List your skills, sorted by name, one page at a time.

    Args:
        query (str): Optional search terms. Empty lists every skill; otherwise only skills whose name, description or triggers contain every term.
        offset (int): Where the page starts (0 for the first page). The answer names the offset of the next page when there is one.

    Returns:
        str: One line per skill (id, name, version, entry type, description; marked when its instructions are over their target size) followed by an indented line of its usage record (sessions it was served in, how often it was followed, its effects, those sessions' outcomes, how many sessions it was built from), the total number of skills, and the offset of the next page.
    """
    from mirix.services.skill_session_feedback import render_usage_line

    from mirix.services.skill_curator_state import (
        REJECT_INVALID,
        SKILL_LIST_PAGE_SIZE,
    )

    try:
        offset = int(offset or 0)
    except (TypeError, ValueError):
        offset = -1
    if offset < 0:
        return _skill_reject(
            self, "skill_list", REJECT_INVALID, "'offset' must be a whole number >= 0."
        )
    query = query if isinstance(query, str) else ""
    skills, total = await self.procedural_memory_manager.list_skills_page(
        user=self.user,
        scopes=_skill_scopes(self),
        query=query,
        offset=offset,
        limit=SKILL_LIST_PAGE_SIZE,
    )
    matching = f' matching "{query.strip()}"' if query.strip() else ""
    if total == 0:
        return f"No skills found{matching}."
    if not skills:
        return (
            f"No skills at offset {offset}: there are {total} skill(s){matching}. "
            "Call skill_list with a smaller offset."
        )

    records = await _skill_usage_records(self, [skill.id for skill in skills])
    cap = _skill_output_cap(self)
    # The page is also cut to the output cap: keep whole entries only.
    reserve = 300  # header + footer
    lines: List[str] = []
    shown = 0
    used = 0
    for skill in skills:
        entry = (
            _format_skill_line(skill)
            + "\n    "
            + render_usage_line(records.get(skill.id))
        )
        if shown and used + len(entry) + 1 > cap - reserve:
            break
        lines.append(entry)
        shown += 1
        used += len(entry) + 1
    shown_end = offset + shown
    header = f"Skills {offset + 1}-{shown_end} of {total}{matching} (sorted by name):"
    if shown_end < total:
        quoted = query.replace('"', '\\"')
        footer = (
            f'Next page: skill_list(query="{quoted}", offset={shown_end}) '
            f"({total - shown_end} more)."
        )
    else:
        footer = "End of list."
    return "\n".join([header, *lines, footer])


async def skill_read(self: "Agent", name_or_id: str) -> str:
    """
    Read the complete content of a skill by name or ID.

    Args:
        name_or_id (str): The skill's exact name or its ID, as skill_list shows them.

    Returns:
        str: Full skill content including all fields, then the skill's usage record (sessions it was served in, how often it was followed, its effects, those sessions' outcomes, per version served, and how many sessions it was built from).
    """
    import json as _json

    from mirix.services.skill_session_feedback import render_usage_record

    from mirix.orm.errors import NoResultFound
    from mirix.services.skill_curator_state import REJECT_NOT_FOUND, head_tail_chars

    key = (name_or_id or "").strip() if isinstance(name_or_id, str) else ""
    scopes = _skill_scopes(self)
    skill = None
    if key.startswith("proc"):
        try:
            skill = await self.procedural_memory_manager.get_item_by_id(
                item_id=key,
                user=self.user,
                timezone_str="UTC",
                actor=self.actor,
                scopes=scopes,
            )
        except NoResultFound:
            skill = None
    if skill is None and key:
        skill = await self.procedural_memory_manager.get_live_skill_by_name(
            user=self.user, name=key, scopes=scopes
        )
    if skill is None:
        return _skill_reject(
            self,
            "skill_read",
            REJECT_NOT_FOUND,
            f"Skill '{name_or_id}' not found. Use skill_list() to see your skills.",
        )

    from mirix import constants

    instructions = skill.instructions or ""
    size = _size_note(len(instructions)).strip()
    text = "\n".join(
        [
            f"ID: {skill.id}",
            f"Name: {skill.name}",
            f"Version: {skill.version}",
            f"Entry Type: {skill.entry_type}",
            f"Description: {skill.description}",
            # The fields a retrieval matches (never the instructions).
            "Retrieval matches only: name, description, triggers.",
            *([size] if size else []),
            f"Instructions ({len(instructions)} characters; target "
            f"{constants.SKILL_TARGET_INSTRUCTIONS_CHARS}):\n{instructions}",
            f"Triggers: {_json.dumps(skill.triggers or [], ensure_ascii=False)}",
            f"Examples: {_json.dumps(skill.examples or [], ensure_ascii=False)}",
            # What the distilled sessions say about it (evidence only).
            render_usage_record(
                (await _skill_usage_records(self, [skill.id])).get(skill.id)
            ),
        ]
    )
    capped, omitted = head_tail_chars(text, _skill_output_cap(self))
    if omitted:
        batch = _curator_batch(self)
        if batch is not None:
            batch.record_capped_output(omitted)
    return capped


async def skill_create(
    self: "Agent",
    name: str,
    description: str,
    instructions: str,
    entry_type: Literal["workflow", "guide", "script"],
    triggers: List[str] = None,
    examples: List[SkillExampleForLLM] = None,
    experience_ids: List[SkillCitationForLLM] = None,
) -> str:
    """
    Create a new skill. Fails if a skill with the same name already exists.

    Args:
        name (str): Lowercase words joined by hyphens that name the task and its approach; unique among your skills.
        description (str): One or two sentences: what this skill does and when it is useful.
        instructions (str): Detailed instructions as plain text: the steps, conditions and caveats a reader needs to follow them without other context.
        entry_type (str): Category — "workflow", "guide", or "script".
        triggers (array): Short conditions, in plain words, that signal this skill is relevant; an empty list for none.
        examples (array): Input/output examples, each an object with an "input" text (the situation or request) and an "output" text (what was done or answered); an empty list for none.
        experience_ids (array): The experiences and earlier lessons of this skill curation batch that this skill applies (every one it addresses), each an object with its "id" and a "quote": a phrase of at least 12 letters (e.g. "check the label first") copied exactly from this skill's description, instructions, triggers or examples that applies that experience's lesson. Only ids shown in the batch are accepted; a citation whose quote is not in that text (or is shorter) does not count and its experience stays pending. Outside a curation batch pass an empty list.

    Returns:
        str: Confirmation with the created skill's ID.
    """
    from mirix.constants import SKILL_MAX_INSTRUCTIONS_CHARS
    from mirix.services import skill_history as history
    from mirix.services.procedural_memory_manager import SkillNameConflictError
    from mirix.services.skill_curator_state import (
        REJECT_BUDGET,
        REJECT_DUPLICATE,
        REJECT_INVALID,
    )

    tool = "skill_create"
    triggers, examples = _without_list_fillers(triggers, examples)
    # 1) Every validation first: a rejected call costs nothing.
    problem = _validate_skill_fields(
        name=name,
        description=description,
        instructions=instructions,
        entry_type=entry_type,
        triggers=triggers,
        examples=examples,
        check=("name", "description", "instructions", "entry_type", "triggers", "examples"),
    )
    if problem:
        return _skill_reject(self, tool, REJECT_INVALID, f"Rejected: {problem} Nothing was created.")
    if len(instructions) > SKILL_MAX_INSTRUCTIONS_CHARS:
        return _instructions_too_long(self, tool, len(instructions))
    # The citations are checked before anything is looked up or embedded.
    cited, quotes, rejection = _citation_problem(self, tool, experience_ids)
    if rejection:
        return rejection

    scopes = _skill_scopes(self)
    # Names are unique per user across scopes (the live-name index).
    existing = await self.procedural_memory_manager.get_live_skill_by_name(
        user=self.user, name=name
    )
    if existing is not None:
        tags = existing.filter_tags if isinstance(existing.filter_tags, dict) else {}
        if scopes is None or tags.get("scope") in scopes:
            message = (
                f"Skill '{name}' already exists (ID: {existing.id}). Use skill_edit to "
                "change it, or choose another name. Nothing was created."
            )
        else:
            message = (
                f"The name '{name}' is already taken by a skill you cannot access. "
                "Choose another name. Nothing was created."
            )
        return _skill_reject(self, tool, REJECT_DUPLICATE, message)

    batch = _curator_batch(self)
    if batch is not None and batch.remaining <= 0:
        return _skill_reject(self, tool, REJECT_BUDGET, batch.budget_message())

    # A citation counts only when its quote is text of the new skill.
    recorded, unverified = list(cited), {}
    if batch is not None:
        recorded, unverified = _verified_citations(
            cited,
            quotes,
            _create_text(description, instructions, triggers, examples),
        )

    # 2) The write (versioned, locked, guarded by the learning run).
    agent_state = self.agent_state
    agent_id = agent_state.parent_id if agent_state.parent_id is not None else agent_state.id
    filter_tags = getattr(self, "filter_tags", None)
    try:
        result = await self.procedural_memory_manager.apply_skill_change(
            history.CHANGE_CREATE,
            actor=self.actor,
            user_id=_skill_owner_id(self),
            fields={
                "name": name,
                "entry_type": entry_type,
                "description": description,
                "instructions": instructions,
                "triggers": triggers or [],
                "examples": examples or [],
                "version": "0.1.0",
                "agent_id": agent_id,
                "client_id": getattr(self.actor, "id", None),
                "organization_id": self.user.organization_id,
                "filter_tags": filter_tags if filter_tags else None,
            },
            agent_state=agent_state,
            **_skill_write_options(self, cited=recorded),
        )
    except SkillNameConflictError:
        # Race: a concurrent create took the name after the check above.
        return _skill_reject(
            self,
            tool,
            REJECT_DUPLICATE,
            f"Skill '{name}' already exists. Use skill_read / skill_edit on the "
            "existing skill, or choose another name. Nothing was created.",
        )

    # 3) It went through: charge it.
    if batch is not None:
        batch.charge("create", result.skill.id)
        batch.record_write(
            _write_entry(
                "skill_create",
                result.skill.id,
                "*",
                _added_words_of_create(description, instructions, triggers, examples),
                0,
                cited,
                recorded=bool(recorded) or not cited,
                unverified=unverified,
            )
        )
    return (
        f"Skill '{name}' created successfully (ID: {result.skill.id}, version: 0.1.0)."
        + _citation_note(self, recorded, result.skill.id, unverified=unverified)
        + _numbering_note(instructions)
        + _size_note(len(instructions))
        + _budget_note(self)
    )


async def skill_edit(
    self: "Agent",
    skill_id: str,
    field: str,
    old_text: str = None,
    new_text: str = None,
    value: str = None,
    experience_ids: List[SkillCitationForLLM] = None,
) -> str:
    """
    Edit an existing skill. For a text field (name, description, instructions) either patch it with old_text/new_text (find-and-replace), or replace the whole field with value (a full rewrite). For the other fields (triggers, examples, entry_type) replace the entire field with value; for triggers and examples value must be a JSON string. Pass an empty string for the parameters you do not use: value when you patch, old_text and new_text when you replace a whole field.

    Args:
        skill_id (str): The ID of the skill to edit.
        field (str): The field to modify — "name", "description", "instructions", "entry_type", "triggers", or "examples".
        old_text (str): For a text-field patch: the text to find and replace. An empty string when you replace a whole field.
        new_text (str): For a text-field patch: the replacement text (an empty string deletes the found text). An empty string when you replace a whole field.
        value (str): The new value for the entire field; an empty string when you patch.
            - name / description / instructions: the complete new text (a rewrite of the whole field).
            - entry_type: one of "workflow", "guide", "script".
            - triggers:   a JSON-encoded list of strings, each a short condition in plain words.
            - examples:   a JSON-encoded list of objects, each with exactly an "input" and an "output" string.
        experience_ids (array): The experiences and earlier lessons of this skill curation batch that this change applies (every one it addresses), each an object with its "id" and a "quote": a phrase of at least 12 letters (e.g. "check the label first") copied exactly from the text this change adds (the new_text of a patch, or the new parts of a whole new value; not text the field already had) that applies that experience's lesson. Only ids shown in the batch are accepted; a citation whose quote is not in the added text (or is shorter) does not count and its experience stays pending. Outside a curation batch pass an empty list.

    Returns:
        str: Confirmation of the edit with the new version number.
    """
    from mirix.orm.errors import NoResultFound
    from mirix.services import skill_history as history
    from mirix.services.procedural_memory_manager import (
        SkillChangedMeanwhile,
        SkillNameConflictError,
    )
    from mirix.services.skill_curator_state import (
        REJECT_BUDGET,
        REJECT_DUPLICATE,
        REJECT_INVALID,
        REJECT_NO_CHANGE,
        REJECT_NOT_FOUND,
    )
    from mirix.services.skill_text_checks import SKILL_CITATION_MIN_ADDED_WORDS

    tool = "skill_edit"
    if field not in _SKILL_EDITABLE_FIELDS:
        return _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            f"Invalid field '{field}'. Must be one of: "
            f"{', '.join(sorted(_SKILL_EDITABLE_FIELDS))}.",
        )

    # A strict tool schema makes every parameter required, so a model fills
    # the ones it does not use with "": an empty old_text or value
    # counts as not given. new_text="" still deletes the text a patch finds;
    # without an old_text it is only a filler too.
    old_text = _given(old_text)
    value = _given(value)
    if old_text is None and new_text == "":
        new_text = None
    cited, quotes, rejection = _citation_problem(self, tool, experience_ids)
    if rejection:
        return rejection

    is_patch = old_text is not None
    scopes = _skill_scopes(self)
    # A read-modify-write: the new value is computed from the skill as read
    # here, and the write is refused (SkillChangedMeanwhile) when that field
    # changed meanwhile. A patch is then applied again to the fresh value; a
    # whole-field rewrite is not (it was written against what the model saw).
    for _attempt in range(_SKILL_EDIT_ATTEMPTS):
        try:
            skill = await self.procedural_memory_manager.get_item_by_id(
                item_id=skill_id,
                user=self.user,
                timezone_str="UTC",
                actor=self.actor,
                scopes=scopes,
            )
        except NoResultFound:
            skill = None
        if skill is None:
            return _skill_reject(
                self, tool, REJECT_NOT_FOUND, f"Skill '{skill_id}' not found. Nothing was changed."
            )

        current = getattr(skill, field)
        planned = await _plan_skill_edit(
            self, tool, skill, field, current, old_text, new_text, value
        )
        if isinstance(planned, _SkillEditRejected):
            return planned.message
        new_value = planned

        if new_value == current:
            return _skill_reject(
                self,
                tool,
                REJECT_NO_CHANGE,
                f"The new value of '{field}' is the same as the current one: nothing to "
                "change, and no budget was used.",
            )

        batch = _curator_batch(self)
        # Keeping the retrieval text in step with an instructions edit of
        # this batch never competes with a lesson for the budget.
        free = batch is not None and batch.is_free_followup(skill.id, field)
        if batch is not None and not free and batch.remaining <= 0:
            return _skill_reject(self, tool, REJECT_BUDGET, batch.budget_message())

        # A citation says the change carries the cited lesson: only a change
        # that adds text can (a deletion, a renumbering or a reflow adds none,
        # and a rename or an entry_type change carries no lesson). Such an
        # edit goes through, but in a batch its citations are not recorded.
        # Outside a batch nothing reads the counts, so they are not computed.
        added, removed = (
            _edit_word_counts(field, current, new_value)
            if batch is not None
            else (0, 0)
        )
        no_lesson = (
            batch is not None and bool(cited) and added < SKILL_CITATION_MIN_ADDED_WORDS
        )
        # A free follow-up keeps the retrieval text in step with the
        # instructions edit that carried the lesson: it carries none of its
        # own, so its citations are not recorded (a lesson it cited would be
        # used at no budget cost), and citing nothing is right.
        recorded = [] if (no_lesson or free) else list(cited)
        unverified: Dict[str, str] = {}
        if batch is not None and recorded:
            # A citation counts only when its quote is text this edit adds.
            added_text, replaced_text = _edit_texts(
                field, current, new_value, old_text, new_text
            )
            recorded, unverified = _verified_citations(
                recorded, quotes, added_text, replaced_text
            )

        try:
            result = await self.procedural_memory_manager.apply_skill_change(
                history.CHANGE_EDIT,
                actor=self.actor,
                user_id=_skill_owner_id(self),
                skill_id=skill.id,
                fields={field: new_value},
                expected_fields={field: current},
                bump_version=True,
                agent_state=self.agent_state,
                **_skill_write_options(self, cited=recorded),
            )
        except NoResultFound:
            # Deleted (or moved out of this scope) since it was read above.
            return _skill_reject(
                self, tool, REJECT_NOT_FOUND, f"Skill '{skill_id}' not found. Nothing was changed."
            )
        except SkillNameConflictError:
            return _skill_reject(
                self,
                tool,
                REJECT_DUPLICATE,
                f"Another skill is already named '{new_value}'. Choose another name. "
                "Nothing was changed.",
            )
        except SkillChangedMeanwhile:
            if is_patch:
                continue  # read the new value and apply the patch to it
            return _changed_meanwhile(self, tool, skill_id, field)

        if batch is not None:
            if free:
                batch.record_free_followup(result.skill.id, field)
            else:
                batch.charge("edit", result.skill.id)
            batch.record_write(
                _write_entry(
                    tool,
                    result.skill.id,
                    field,
                    added,
                    removed,
                    cited,
                    recorded=bool(recorded) or not cited,
                    unverified=unverified,
                )
            )
        if free:
            # Neither a citation nor an uncited write: see above.
            already = set(batch.citations) | set(batch.precited_ids)
            new_ids = [i for i in cited if i not in already]
            note = ""
            if new_ids:
                batch.citations_dropped += 1
                note = (
                    f" Its citations ({_named(new_ids)}) were NOT recorded: a "
                    "follow-up of the retrieval text costs no budget, so it carries "
                    "no lesson of its own. Cite them on a change that carries their "
                    "lesson, or list them with experience_triage."
                )
        elif no_lesson:
            batch.citations_dropped += 1
            why = _NO_LESSON_FIELDS.get(
                field,
                "this change adds no new words (a deletion, a renumbering or a "
                "reflow adds none)",
            )
            note = (
                f" Its citations ({_named(cited)}) were NOT recorded: {why}, so it "
                "cannot carry their lesson. Cite them on the change that adds their "
                "lesson, or list them with experience_triage (as covered when the "
                "removal itself was the lesson)."
            )
        else:
            note = _citation_note(
                self, recorded, result.skill.id, unverified=unverified
            )
        if field in ("instructions", "description") and (
            value is not None or _patch_rewrites(current, old_text)
        ):
            # A patch whose old_text is (most of) the whole field is a
            # rewrite too, whatever the value parameter held.
            note += _rewrite_note(current or "", new_value)
        if field == "instructions":
            note += _numbering_note(new_value)
            note += _size_note(len(new_value or ""))
            note += _retrieval_text_note(self, result.skill.id)
        if free:
            note += (
                " This change of its retrieval text follows this batch's change of "
                "its instructions, so it cost no edit budget and needs no citation."
            )
        return (
            f"Skill '{result.skill.name}' updated (field: {field}, new version: "
            f"{result.skill.version})." + note + _budget_note(self)
        )
    return _changed_meanwhile(self, tool, skill_id, field)


def _write_entry(
    tool: str,
    skill_id: str,
    field: str,
    added: int,
    removed: int,
    cited: Sequence[str],
    *,
    recorded: bool,
    unverified: Optional[Dict[str, str]] = None,
) -> dict:
    """A write's ``CuratorBatchState.write_log`` entry (``unverified``: the
    cited ids whose quote did not show the write applying them, and
    why)."""
    return {
        "tool": tool,
        "skill_id": skill_id,
        "field": field,
        "added_words": added,
        "removed_words": removed,
        "cited": list(cited),
        "citations_recorded": bool(recorded),
        "citations_unverified": dict(unverified or {}),
    }


def _create_text(description, instructions, triggers, examples) -> str:
    """The texts of a new skill that can apply a lesson, as one text (a
    citation's quote must be part of it). Its name is left out: a quote of
    the name would show any lesson applied."""
    from mirix.services.skill_text_checks import list_text

    return "\n".join(
        str(part or "")
        for part in (
            description,
            instructions,
            list_text(triggers or []),
            list_text(examples or []),
        )
    )


def _edit_texts(field: str, current, new_value, old_text, new_text) -> Tuple[str, str]:
    """``(added, had)`` text of a skill_edit: a patch adds its
    ``new_text``; a whole new value its value (a list field as its words).
    ``had`` is the field's whole current value: a quote of text the field
    already had anywhere — moved or repeated by the patch — shows nothing
    new."""
    from mirix.services.skill_text_checks import list_text

    if field in _SKILL_JSON_FIELDS:
        return list_text(new_value or []), list_text(current or [])
    if old_text is not None:
        return str(new_text or ""), str(current or "")
    return str(new_value or ""), str(current or "")


#: A text-field patch whose old_text is at least this share of the field
#: (whitespace collapsed) replaces most of it: its answer reports what it
#: dropped like a rewrite's.
PATCH_REWRITE_SHARE = 0.5


def _patch_rewrites(current, old_text) -> bool:
    """Whether a patch with ``old_text`` replaces (most of) the whole field
    ``current``."""
    if old_text is None:
        return False
    whole = " ".join(str(current or "").split())
    found = " ".join(str(old_text).split())
    return bool(whole) and len(found) >= PATCH_REWRITE_SHARE * len(whole)


def _added_words_of_create(description, instructions, triggers, examples) -> int:
    from mirix.services.skill_text_checks import content_words, list_text

    return sum(
        len(content_words(text))
        for text in (
            description,
            instructions,
            list_text(triggers or []),
            list_text(examples or []),
        )
    )


# Fields whose edit never carries a lesson (it counts 0 added words), and why.
_NO_LESSON_FIELDS = {
    "name": "a rename carries no lesson",
    "entry_type": "an entry_type change carries no lesson",
}


def _edit_word_counts(field: str, current, new_value) -> Tuple[int, int]:
    """``(added, removed)`` content words of an edit of ``field``
    (skill_text_checks.added_words); a rename or an entry_type change counts
    0 added: it carries no lesson."""
    from mirix.services.skill_text_checks import added_words, list_text

    if field in ("description", "instructions"):
        return added_words(current or "", new_value or "")
    if field in _SKILL_JSON_FIELDS:
        return added_words(list_text(current or []), list_text(new_value or []))
    return 0, 0


# Sentences a rewrite dropped that its answer names at most, and their length.
_REWRITE_DROPPED_SHOWN = 5
_REWRITE_SENTENCE_CHARS = 160


def _rewrite_note(old: str, new: str) -> str:
    """What a whole-field rewrite dropped from the old text (advice only:
    the rewrite went through)."""
    from mirix.services.skill_text_checks import dropped_sentences

    dropped, total = dropped_sentences(old, new)
    if not dropped:
        return ""
    shown = []
    for i, sentence in enumerate(dropped[:_REWRITE_DROPPED_SHOWN], 1):
        text = " ".join(sentence.split())
        if len(text) > _REWRITE_SENTENCE_CHARS:
            text = text[: _REWRITE_SENTENCE_CHARS - 1] + "…"
        shown.append(f"{i}) {text}")
    more = len(dropped) - len(shown)
    return (
        f" This rewrite dropped {len(dropped)} of {total} sentences of the old text "
        f"(not found, even reworded): {' '.join(shown)}"
        + (f" … and {more} more." if more > 0 else "")
        + " Restore with a patch any of these lessons that is still right and that"
        " the new text no longer says (one it keeps in other words is kept)."
    )


def _numbering_note(instructions: str) -> str:
    """A warning about the numbered steps of ``instructions`` (a repeated or
    skipped number), or ""."""
    from mirix.services.skill_text_checks import numbering_problems

    problems = numbering_problems(instructions or "")
    if not problems:
        return ""
    return (
        f" Warning: the numbered steps are out of order ({'; '.join(problems)}). "
        "Renumber them in order with a patch."
    )


def _given(value):
    """A skill_edit argument, or None when it was not given: missing, or the
    empty string a strict tool schema makes the model send for a parameter it
    does not use."""
    if value is None or (isinstance(value, str) and value == ""):
        return None
    return value


# Reads of a skill_edit patch whose field keeps changing under it.
_SKILL_EDIT_ATTEMPTS = 3


@dataclass
class _SkillEditRejected:
    """A skill_edit call refused while its new value was worked out (the
    rejection is already recorded)."""

    message: str


def _changed_meanwhile(self: "Agent", tool: str, skill_id: str, field: str) -> str:
    from mirix.services.skill_curator_state import REJECT_CHANGED

    return _skill_reject(
        self,
        tool,
        REJECT_CHANGED,
        f"Skill '{skill_id}' changed while this edit was being made (its "
        f"'{field}' is no longer the value the edit was computed from). Nothing "
        "was changed and no budget was used: read the skill again "
        "(skill_read) and redo the edit on its current content.",
    )


async def _plan_skill_edit(
    self: "Agent",
    tool: str,
    skill,
    field: str,
    current,
    old_text,
    new_text,
    value,
):
    """The new value of ``field`` for a skill_edit call on ``skill`` as read,
    or a :class:`_SkillEditRejected` — every check that needs no write."""
    import json as _json

    from mirix.constants import SKILL_MAX_INSTRUCTIONS_CHARS
    from mirix.services.skill_curator_state import (
        REJECT_DUPLICATE,
        REJECT_INVALID,
        head_tail_chars,
    )

    def reject(kind: str, message: str) -> _SkillEditRejected:
        return _SkillEditRejected(_skill_reject(self, tool, kind, message))

    if field in _SKILL_TEXT_FIELDS:
        # old_text / value are None when not given (an empty string included,
        # see skill_edit): a patch has old_text, a rewrite has value.
        current_text = current or ""
        if old_text is not None and value is not None:
            return reject(
                REJECT_INVALID,
                "Give either a patch (old_text and new_text, with value empty) or "
                "a whole new field (value, with old_text and new_text empty), not "
                "both. Nothing was changed.",
            )
        if old_text is not None:
            if new_text is None or not str(old_text).strip():
                # An old_text of whitespace only anchors nothing: the empty
                # pattern would match at the very start of the field.
                return reject(
                    REJECT_INVALID,
                    f"For a patch of '{field}' both old_text (the text to find, not "
                    "only whitespace) and new_text are required, with value empty "
                    "(or give value, with old_text and new_text empty, to replace "
                    "the whole field). Nothing was changed.",
                )
            # Whitespace-insensitive match of old_text.
            normalized_current = " ".join(current_text.split())
            normalized_old = " ".join(str(old_text).split())
            if normalized_old not in normalized_current:
                shown, _ = head_tail_chars(current_text, max(2000, _skill_output_cap(self) // 4))
                return reject(
                    REJECT_INVALID,
                    f"old_text not found in field '{field}'. Nothing was changed. "
                    f"Current value:\n{shown}",
                )
            # Any run of whitespace in old_text matches any run in the field
            # (the same rule as the check above; a pattern built from the
            # escaped text would need one whitespace character per original one).
            pattern = r"\s+".join(re.escape(token) for token in str(old_text).split())
            new_value = re.sub(pattern, lambda _m: str(new_text), current_text, count=1)
        elif value is not None:
            new_value = str(value)
        else:
            return reject(
                REJECT_INVALID,
                f"For a patch of text field '{field}' give old_text (the text to "
                "find) and new_text, with value empty; to replace the whole field "
                "give value, with old_text and new_text empty. Nothing was changed.",
            )
        problem = _validate_skill_fields(check=(field,), **{field: new_value})
        if problem:
            return reject(REJECT_INVALID, f"Rejected: {problem} Nothing was changed.")
        if field == "instructions" and len(new_value) > SKILL_MAX_INSTRUCTIONS_CHARS:
            return _SkillEditRejected(_instructions_too_long(self, tool, len(new_value)))
        if field == "name" and new_value != current_text:
            holder = await self.procedural_memory_manager.get_live_skill_by_name(
                user=self.user, name=new_value
            )
            if holder is not None and holder.id != skill.id:
                return reject(
                    REJECT_DUPLICATE,
                    f"Another skill is already named '{new_value}'. Choose another "
                    "name. Nothing was changed.",
                )
        return new_value
    if field in _SKILL_JSON_FIELDS:
        if value is None:
            return reject(
                REJECT_INVALID,
                f"For field '{field}', the value parameter is required (JSON string).",
            )
        # triggers/examples are typed List[str] / List[dict] in the schema. The
        # agent delivers them as JSON strings (or, from some models, already
        # decoded), so decode + validate here.
        if isinstance(value, str):
            try:
                decoded = _json.loads(value)
            except (TypeError, _json.JSONDecodeError) as exc:
                return reject(
                    REJECT_INVALID,
                    f"Invalid JSON for field '{field}': {exc}. "
                    f'Pass a JSON-encoded list, e.g. \'["trigger-a", "trigger-b"]\'.',
                )
        else:
            decoded = value
        if not isinstance(decoded, list):
            return reject(
                REJECT_INVALID,
                f"Field '{field}' must be a JSON array, got {type(decoded).__name__}.",
            )
        problem = _validate_skill_fields(check=(field,), **{field: decoded})
        if problem:
            return reject(REJECT_INVALID, f"Field {problem}")
        return decoded
    # entry_type
    if value is None:
        return reject(
            REJECT_INVALID,
            f"For field '{field}', the value parameter is required.",
        )
    problem = _validate_skill_fields(check=("entry_type",), entry_type=value)
    if problem:
        return reject(REJECT_INVALID, f"Rejected: {problem}")
    return value


async def skill_delete(
    self: "Agent",
    skill_id: str,
    reason: str,
    experience_ids: List[SkillCitationForLLM] = None,
) -> str:
    """
    Delete a skill (a soft delete: it leaves retrieval and can be restored from its version history).

    In a skill learning run at most one skill may be deleted per run; deleting does not use the create/edit budget.

    Args:
        skill_id (str): The ID of the skill to delete.
        reason (str): Why the skill must go (e.g. it is wrong, harmful, or fully merged into another skill). Required; recorded in the skill's history.
        experience_ids (array): The experiences and earlier lessons of this skill curation batch that this deletion comes from (every one it addresses), each an object with its "id" and a "quote" (empty: a deletion adds no text). Only ids shown in the batch are accepted. Outside a curation batch pass an empty list.

    Returns:
        str: Confirmation message.
    """
    from mirix.orm.errors import NoResultFound
    from mirix.services import skill_history as history
    from mirix.services.skill_curator_state import (
        REJECT_DELETE_LIMIT,
        REJECT_INVALID,
        REJECT_NOT_FOUND,
    )

    tool = "skill_delete"
    if not isinstance(reason, str) or not reason.strip():
        return _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            "A deletion needs a reason: say why the skill must go. Nothing was deleted.",
        )
    # A deletion adds no text for a quote to point at: its citations count
    # as given (the quote check reads the text a create or an edit adds).
    cited, _quotes, rejection = _citation_problem(self, tool, experience_ids)
    if rejection:
        return rejection
    try:
        skill = await self.procedural_memory_manager.get_item_by_id(
            item_id=skill_id,
            user=self.user,
            timezone_str="UTC",
            actor=self.actor,
            scopes=_skill_scopes(self),
        )
    except NoResultFound:
        skill = None
    if skill is None:
        return _skill_reject(
            self, tool, REJECT_NOT_FOUND, f"Skill '{skill_id}' not found. Nothing was deleted."
        )

    batch = _curator_batch(self)
    if batch is not None and batch.run_limits.deletes_left <= 0:
        return _skill_reject(
            self,
            tool,
            REJECT_DELETE_LIMIT,
            f"Rejected: at most {batch.run_limits.max_deletes} skill deletion(s) are "
            "allowed per learning run, and this run already used it. Improve the "
            "skill with skill_edit instead, or leave it. Nothing was deleted.",
        )

    try:
        # Always a soft delete through the skill write path:
        # never the hard delete, which erases the history.
        await self.procedural_memory_manager.apply_skill_change(
            history.CHANGE_DELETE,
            actor=self.actor,
            user_id=_skill_owner_id(self),
            skill_id=skill.id,
            reason=reason.strip(),
            **_skill_write_options(self, cited=cited),
        )
    except NoResultFound:
        return _skill_reject(
            self, tool, REJECT_NOT_FOUND, f"Skill '{skill_id}' not found. Nothing was deleted."
        )

    if batch is not None:
        batch.record_delete(skill.id)
    return (
        f"Skill '{skill.id}' soft-deleted (excluded from retrieval; restorable from its "
        "history). Reason recorded." + _citation_note(self, cited, skill.id, delete=True)
    )


async def experience_triage(
    self: "Agent",
    covered_ids: List[str],
    not_worth_ids: List[str],
    deferred_ids: List[str],
) -> str:
    """
    In a skill curation batch: say what becomes of the experiences and earlier lessons that no change of yours cites. Call it in a turn of its own and read its answer; then call finish_memory_update() alone. A later call overrides an id's earlier list. When an experience of the batch is neither cited nor triaged, finish_memory_update() refuses and lists it.

    Args:
        covered_ids (array): Ids an existing skill already covers: there is nothing to add.
        not_worth_ids (array): Ids not worth a skill: too narrow, wrong, or contradicted.
        deferred_ids (array): Ids that may matter but lack the evidence for a skill yet. A new experience listed here stays for later batches; an earlier lesson listed here counts as ignored, like one not triaged at all, and one ignored in three batches without coming up again leaves the pool.

    Returns:
        str: What was recorded, and the ids still untriaged.
    """
    from mirix.services.skill_curator_state import REJECT_INVALID, REJECT_NO_CHANGE

    tool = "experience_triage"
    batch = _curator_batch(self)
    if batch is None:
        return _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            "experience_triage works only inside a skill curation batch. Nothing was recorded.",
        )
    lists = {}
    for name, value in (
        ("covered_ids", covered_ids),
        ("not_worth_ids", not_worth_ids),
        ("deferred_ids", deferred_ids),
    ):
        ids, problem = _experience_ids_arg(value)
        if problem:
            return _skill_reject(
                self,
                tool,
                REJECT_INVALID,
                f"Rejected: '{name}' {problem} Nothing was recorded.",
            )
        lists[name] = ids
    seen = {}
    twice = []
    for name, ids in lists.items():
        for experience_id in ids:
            if experience_id in seen and experience_id not in twice:
                twice.append(experience_id)
            seen.setdefault(experience_id, name)
    if twice:
        return _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            f"Rejected: {_named(twice)} appear(s) in more than one list; give each id "
            "one list. Nothing was recorded.",
        )
    unknown = batch.unknown_ids(list(seen))
    if unknown:
        return _skill_reject(
            self,
            tool,
            REJECT_INVALID,
            f"Rejected: {', '.join(unknown[:10])}{' …' if len(unknown) > 10 else ''} "
            "was not shown in this batch. Triage only the ids of this batch's "
            "experiences and earlier lessons. Nothing was recorded.",
        )
    if not seen:
        # Three empty lists decide nothing: not a triage.
        unaddressed = batch.unaddressed_experiences()
        if unaddressed:
            return _skill_reject(
                self,
                tool,
                REJECT_NO_CHANGE,
                "Rejected: all three lists are empty, so nothing was recorded. "
                f"Still neither cited nor triaged: {_named(unaddressed)}. Put each "
                "of them in covered_ids, not_worth_ids or deferred_ids.",
            )
        return (
            "Nothing recorded: all three lists are empty, and every experience of "
            "this batch is already cited or triaged."
        )
    batch.record_triage(
        lists["covered_ids"], lists["not_worth_ids"], lists["deferred_ids"]
    )
    batch.triage_calls += 1
    parts = []
    for label, name in (
        ("covered", "covered_ids"),
        ("not worth a skill", "not_worth_ids"),
        ("deferred", "deferred_ids"),
    ):
        ids = lists[name]
        parts.append(f"{label} {len(ids)}" + (f" ({_named(ids)})" if ids else ""))
    answer = "Triage recorded: " + "; ".join(parts) + "."
    cited = [i for i in seen if batch.is_cited(i)]
    if cited:
        answer += (
            f" {_named(cited)} are cited by a change of this batch, so they count as "
            "used whatever list they are in."
        )
    untriaged = batch.untriaged()
    if untriaged:
        answer += (
            f" Neither cited nor triaged yet (they count as deferred): "
            f"{_named(untriaged)}."
        )
    else:
        answer += " Every experience and earlier lesson of this batch is cited or triaged."
    return answer


async def check_semantic_memory(
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
        await self.semantic_memory_manager.get_semantic_item_by_id(
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


async def semantic_memory_insert(self: "Agent", items: List[SemanticMemoryItemBase]):
    """
    The tool to insert items into semantic memory.

    Args:
        items (array): List of semantic memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

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
        existing_items = await self.semantic_memory_manager.list_semantic_items(
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
            try:
                await self.semantic_memory_manager.insert_semantic_item(
                    agent_state=self.agent_state,
                    agent_id=agent_id,
                    name=item["name"],
                    summary=item["summary"],
                    details=item["details"],
                    source=item.get("source", ""),
                    organization_id=self.actor.organization_id,
                    actor=self.actor,
                    filter_tags=filter_tags if filter_tags else None,
                    use_cache=use_cache,
                    user_id=user_id,
                )
            except Exception as e:
                print(
                    f"[semantic_memory_insert] insert_semantic_item FAILED for "
                    f"item {item!r}: {e}"
                )
                traceback.print_exc()
                raise
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


async def semantic_memory_update(
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
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_semantic_item_ids:
        try:
            await self.semantic_memory_manager.delete_semantic_item_by_id(
                semantic_memory_id=old_id, actor=self.actor
            )
        except Exception as e:
            print(
                f"[semantic_memory_update] delete_semantic_item_by_id FAILED for "
                f"old_id {old_id!r}: {e}"
            )
            traceback.print_exc()
            raise

    new_ids = []
    for item in new_items:
        try:
            inserted_item = await self.semantic_memory_manager.insert_semantic_item(
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
        except Exception as e:
            print(
                f"[semantic_memory_update] insert_semantic_item FAILED for "
                f"item {item!r}: {e}"
            )
            traceback.print_exc()
            raise
        new_ids.append(inserted_item.id)

    message_to_return = (
        "Semantic memory with the following ids have been deleted: "
        + str(old_semantic_item_ids)
        + f". New semantic memory items are created: {str(new_ids)}"
    )
    return message_to_return


async def knowledge_vault_insert(self: "Agent", items: List[KnowledgeVaultItemBase]):
    """
    The tool to update knowledge vault.

    Args:
        items (array): List of knowledge vault items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

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
        existing_items = await self.knowledge_vault_manager.list_knowledge(
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
            try:
                await self.knowledge_vault_manager.insert_knowledge(
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
            except Exception as e:
                print(
                    f"[knowledge_vault_insert] insert_knowledge FAILED for "
                    f"item {item!r}: {e}"
                )
                traceback.print_exc()
                raise
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


async def knowledge_vault_update(
    self: "Agent", old_ids: List[str], new_items: List[KnowledgeVaultItemBase]
):
    """
    The tool to update/delete items in the knowledge vault. To update the knowledge_vault, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new knowledge vault items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )

    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        try:
            await self.knowledge_vault_manager.delete_knowledge_by_id(
                knowledge_vault_item_id=old_id, actor=self.actor
            )
        except Exception as e:
            print(
                f"[knowledge_vault_update] delete_knowledge_by_id FAILED for "
                f"old_id {old_id!r}: {e}"
            )
            traceback.print_exc()
            raise

    for item in new_items:
        try:
            await self.knowledge_vault_manager.insert_knowledge(
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
        except Exception as e:
            print(
                f"[knowledge_vault_update] insert_knowledge FAILED for item "
                f"{item!r}: {e}"
            )
            traceback.print_exc()
            raise


async def trigger_memory_update_with_instruction(
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

    if not isinstance(user_message, dict):
        raise TypeError(
            f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}"
        )

    if memory_type == "core":
        agent_type = "core_memory_agent"
    elif memory_type == "episodic":
        agent_type = "episodic_memory_agent"
    elif memory_type == "resource":
        agent_type = "resource_memory_agent"
    elif memory_type == "procedural":
        # NOTE: this is the EXPLICIT, user-driven memory-correction path (the chat
        # agent invokes this tool only when the user explicitly asks to insert /
        # update / delete a specific memory). It is intentionally NOT subject to the
        # "procedural has exactly one producer" gate applied to trigger_memory_update,
        # which governs only AUTOMATIC extraction from a conversation transcript.
        # Skill *learning* still flows solely through the distillation path; a direct
        # user command to edit procedural memory is a different operation and is kept.
        agent_type = "procedural_memory_agent"
    elif memory_type == "knowledge_vault":
        agent_type = "knowledge_vault_memory_agent"
    elif memory_type == "semantic":
        agent_type = "semantic_memory_agent"
    else:
        raise ValueError(
            f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge_vault', 'semantic'."
        )

    client = await create_client()
    agents = await client.list_agents()
    matching_agent = None
    for agent in agents:
        if agent.agent_type == agent_type:
            matching_agent = agent
            break
    if matching_agent is None:
        raise ValueError(f"No agent found with type '{agent_type}'")
    await client.send_message(
        role="user",
        user_id=self.user.id,
        agent_id=matching_agent.id,
        message="[Message from Chat Agent (Now you are allowed to make multiple function calls sequentially)] "
        + instruction,
        existing_file_uris=user_message["existing_file_uris"],
        retrieved_memories=user_message.get("retrieved_memories", None),
        block_filter_tags=getattr(self, "block_filter_tags", None),
        block_filter_tags_update_mode=getattr(
            self, "block_filter_tags_update_mode", "merge"
        ),
    )
    result = (
        "[System Message] Agent "
        + matching_agent.name
        + " has been triggered to update the memory.\n"
    )
    return result.strip()


async def trigger_memory_update(
    self: "Agent", user_message: object, memory_types: List[str]
) -> Optional[str]:
    """
    Choose which memory to update. This function will trigger another memory agent which is specifically in charge of handling the corresponding memory to update its memory. Trigger all necessary memory updates at once. Put the explanations in the `internal_monologue` field.

    Args:
        memory_types (List[str]): The types of memory to update. It should be chosen from the following: "core", "episodic", "resource", "knowledge_vault", "semantic". For instance, ['episodic', 'resource']. ("procedural" is still accepted for backward compatibility but is a no-op here: procedural/skill memory is produced solely by the conversation-store distillation path, not by this dispatch.)

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
        raise TypeError(
            f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}"
        )

    # Procedural memory has EXACTLY ONE producer: skill learning runs over the
    # Conversation Message Store, triggered deterministically outside this tool
    # (after task-session writes, explicit session closes and the background
    # sweeper — see mirix/services/skill_learning_trigger.py). The meta agent
    # must never dispatch the procedural_memory_agent during normal extraction,
    # so "procedural" is filtered out of memory_types here. It stays an accepted
    # input value (a harmless no-op) so existing callers / LLM emissions don't
    # error; it simply no longer dispatches.
    #
    # Task sessions never reach this tool: /memory/add(_sync) stores them in the
    # conversation store only and the queue worker drops any that is queued,
    # so there is no per-tag branch here.
    filtered_memory_types = [mt for mt in memory_types if mt != "procedural"]
    if len(filtered_memory_types) != len(memory_types):
        logger.debug(
            "Filtered 'procedural' out of trigger_memory_update dispatch "
            "(procedural memory is produced solely by skill learning runs): "
            "%s -> %s",
            list(memory_types),
            filtered_memory_types,
        )
    memory_types = filtered_memory_types

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
    child_agent_states = await self.agent_manager.list_agents(
        parent_id=self.agent_state.id, actor=self.actor
    )

    # Map agent types to agent states (key by string so lookup works for enum or deserialized string)
    def _agent_type_key(at):
        return at.value if hasattr(at, "value") else str(at)

    agent_type_to_state = {
        _agent_type_key(agent_state.agent_type): agent_state
        for agent_state in child_agent_states
    }

    if not child_agent_states:
        raise ValueError(
            "No child memory agents found for the meta agent. "
            "Ensure the meta agent has been initialized (e.g. via POST /agents/meta/initialize)."
        )

    # Capture trace context before spawning concurrent tasks
    parent_trace_context = get_trace_context()
    langfuse = get_langfuse_client()

    async def _run_single_memory_update(memory_type: str) -> str:
        try:
            # Restore trace context in each coroutine for LangFuse hierarchical tracing
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

            # Get filter_tags, block_filter_tags, use_cache, client_id, user_id, and occurred_at from parent agent instance
            # Deep copy filter_tags and block_filter_tags to ensure complete isolation between child agents
            parent_filter_tags = getattr(self, "filter_tags", None)
            # Don't use 'or {}' because empty dict {} is valid and different from None
            filter_tags = (
                deepcopy(parent_filter_tags) if parent_filter_tags is not None else None
            )
            parent_block_filter_tags = getattr(self, "block_filter_tags", None)
            block_filter_tags = (
                deepcopy(parent_block_filter_tags)
                if parent_block_filter_tags is not None
                else None
            )
            block_filter_tags_update_mode = getattr(
                self, "block_filter_tags_update_mode", "merge"
            )
            use_cache = getattr(self, "use_cache", True)
            actor = getattr(self, "actor", None)
            user = getattr(self, "user", None)
            occurred_at = getattr(
                self, "occurred_at", None
            )  # Get occurred_at from parent agent

            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Creating {memory_type} agent with filter_tags={filter_tags}, block_filter_tags={block_filter_tags}, client_id={actor.id if actor else None}, user_id={user.id if user else None}, occurred_at={occurred_at}"
            )

            memory_agent = agent_class(
                agent_state=agent_state,
                interface=self.interface,
                actor=actor,
                user=user,
                filter_tags=filter_tags,
                block_filter_tags=block_filter_tags,
                block_filter_tags_update_mode=block_filter_tags_update_mode,
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
                    "[System] Update the memory based on the instructions. Do not miss any important information."
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

            chaining = user_message.get("chaining", False)
            # Procedural agent needs chaining for multi-step CLI workflow
            if memory_type == "procedural":
                chaining = True

            # Extract topics and retrieved_memories from parent agent to pass to sub-agents
            # This ensures sub-agents use the same keywords for memory retrieval
            retrieved_memories = user_message.get("retrieved_memories", None)
            topics = None
            if retrieved_memories and isinstance(retrieved_memories, dict):
                topics = retrieved_memories.get("key_words", None)

            # Pass actor (Client) and user (User) to memory agent
            # actor is needed for write operations, user is needed for read operations

            # Wrap child agent execution in a LangFuse span for hierarchical tracing
            trace_id = (
                parent_trace_context.get("trace_id") if parent_trace_context else None
            )
            parent_span_id = (
                parent_trace_context.get("observation_id")
                if parent_trace_context
                else None
            )
            if langfuse and trace_id:
                from typing import cast

                from langfuse.types import TraceContext

                # Format span name as Title Case (e.g., "Episodic Memory Agent")
                span_name = agent_type_str.replace("_", " ").title()

                # Build trace_context with parent_span_id (per Langfuse v3 docs)
                trace_context_dict: dict = {"trace_id": trace_id}
                if parent_span_id:
                    trace_context_dict["parent_span_id"] = parent_span_id

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

                    await memory_agent.step(
                        input_messages=message_copy,
                        chaining=chaining,
                        actor=actor,
                        user=user,
                        topics=topics,
                        retrieved_memories=retrieved_memories,
                    )
            else:
                # No tracing available, run directly
                await memory_agent.step(
                    input_messages=message_copy,
                    chaining=chaining,
                    actor=actor,
                    user=user,
                    topics=topics,
                    retrieved_memories=retrieved_memories,
                )

            return f"[System Message] Agent {agent_state.name} has been triggered to update the memory.\n"
        finally:
            clear_trace_context()

    responses: dict[int, str] = {}

    if not memory_types:
        return ""

    tasks = [_run_single_memory_update(memory_type) for memory_type in memory_types]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise RuntimeError(
                f"Failed to trigger memory update for '{memory_types[i]}'"
            ) from result
        responses[i] = result

    ordered_responses = [
        responses[i] for i in range(len(memory_types)) if i in responses
    ]
    return "".join(ordered_responses).strip()


async def finish_memory_update(self: "Agent") -> str:
    """
    Finish the memory update process. This function should be called after the Memory is updated.

    Note: This function takes no parameters. Call it without any arguments.

    In a skill curation batch, when an experience of the batch is neither
    cited by a change nor triaged, it refuses to end the batch and lists
    those experiences.

    Returns:
        str: Empty string (no response content).
    """
    batch = _curator_batch(self)
    if batch is None:
        return ""
    from mirix.constants import SKILL_CURATOR_FINISH_REMINDERS
    from mirix.errors import ToolCallRefused
    from mirix.services.skill_curator_state import REJECT_UNADDRESSED

    unaddressed = batch.unaddressed_experiences()
    if unaddressed and batch.finish_refusals < SKILL_CURATOR_FINISH_REMINDERS:
        batch.finish_refusals += 1
        # The refusal says what to do, never how to get past it.
        unquoted = [i for i in unaddressed if i in batch.unverified_citations]
        note = (
            f" Cited without a quote of the text the change added that applies "
            f"them, so not counted: {_named(unquoted)}."
            if unquoted
            else ""
        )
        raise ToolCallRefused(
            batch.reject(
                "finish_memory_update",
                REJECT_UNADDRESSED,
                f"Not finished: {len(unaddressed)} experience(s) of this batch are "
                f"neither cited by a change nor triaged: {_named(unaddressed)}."
                f"{note} Cite each one in the experience_ids of the change that "
                "carries its lesson, with a quote of the words that apply it, or "
                "list it with experience_triage(covered_ids, not_worth_ids, "
                "deferred_ids); then call finish_memory_update() alone.",
            )
        )
    return ""
