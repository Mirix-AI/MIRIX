"""Shared runtime helpers for procedural skill evolution."""

from __future__ import annotations

from typing import Dict, List

from mirix.helpers.keyed_locks import KeyedLocks

# Procedural evolution resets a persistent agent's in-context messages before
# and after a step. Serializing per procedural agent prevents concurrent runs
# from deleting each other's transient chain messages. Self-evicting registry;
# use `async with _evolve_locks.acquire(agent_id):`.
_evolve_locks = KeyedLocks()


def _diff_skills(before: List, after: List) -> Dict[str, List[str]]:
    """Compute created, edited, and deleted skill ids between two snapshots."""
    before_map = {_attr(s, "id"): s for s in (before or [])}
    after_map = {_attr(s, "id"): s for s in (after or [])}
    before_ids = set(before_map)
    after_ids = set(after_map)

    created = sorted(after_ids - before_ids)
    deleted = sorted(before_ids - after_ids)
    edited = []
    for sid in before_ids & after_ids:
        b = before_map[sid]
        a = after_map[sid]
        if (
            _attr(b, "version") != _attr(a, "version")
            or _attr(b, "instructions") != _attr(a, "instructions")
            or _attr(b, "description") != _attr(a, "description")
        ):
            edited.append(sid)
    return {"created": created, "edited": sorted(edited), "deleted": deleted}


async def reset_agent_in_context_to_system(server, agent_id: str, actor) -> None:
    """Force an agent's in-context history back to system-message-only.

    Keeps index 0 (the system message) and deletes detached conversation, tool,
    and heartbeat messages so the DB does not accumulate them. This is
    idempotent when the context is already system-only or empty.
    """
    refreshed = await server.agent_manager.get_agent_by_id(
        agent_id=agent_id, actor=actor
    )
    msg_ids = refreshed.message_ids or []
    if len(msg_ids) <= 1:
        return
    await server.agent_manager.set_in_context_messages(
        agent_id=agent_id, message_ids=[msg_ids[0]], actor=actor
    )
    await server.message_manager.delete_detached_messages_for_agent(
        agent_id=agent_id, actor=actor
    )


def _attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
