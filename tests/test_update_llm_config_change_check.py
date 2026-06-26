"""update_llm_config is a no-op when the target config is unchanged (VEPAGE-1228).

Previously update_llm_config unconditionally issued an update_agent write even
when the persisted llm_config already matched. Callers on the agent
registration / initialize-meta-agent path invoke it per agent, so an unchanged
config produced needless writes (and read-backs). This makes it skip the write
when the current config already equals the requested one.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager


def _actor():
    return Client(id="client-1", name="c", organization_id="org-1")


@pytest.mark.asyncio
async def test_noop_when_config_unchanged():
    am = AgentManager()
    cfg = LLMConfig.default_config("gpt-4o-mini")
    current = AsyncMock()
    current.llm_config = cfg

    with (
        patch.object(am, "get_agent_by_id", new=AsyncMock(return_value=current)),
        patch.object(am, "update_agent", new=AsyncMock()) as upd,
    ):
        out = await am.update_llm_config(agent_id="agent-1", llm_config=cfg, actor=_actor())

    upd.assert_not_awaited()
    assert out is current


@pytest.mark.asyncio
async def test_writes_when_config_changed():
    am = AgentManager()
    current = AsyncMock()
    current.llm_config = LLMConfig.default_config("gpt-4o-mini")
    new_cfg = LLMConfig.default_config("gpt-4o")

    with (
        patch.object(am, "get_agent_by_id", new=AsyncMock(return_value=current)),
        patch.object(am, "update_agent", new=AsyncMock(return_value="updated")) as upd,
    ):
        out = await am.update_llm_config(agent_id="agent-1", llm_config=new_cfg, actor=_actor())

    upd.assert_awaited_once()
    assert out == "updated"
