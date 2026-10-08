"""What WebShop's full-history conversation must leave unchanged, pinned as
fingerprints:

* WebShop's step-1 request, byte for byte (the rollout's request fingerprint, which the
  record keeps as ``first_prompt_sha256``), with and without memory: the conversation
  only appends to it;
* ALFWorld's actor requests on every step of a scripted episode, with and without memory,
  and ALFWorld's agent context: the change is WebShop's alone, so ALFWorld's prompts
  stay byte-identical.
"""

from __future__ import annotations

import hashlib

import pytest

from evals.skill_bench.agent import rollout as ro
from evals.skill_bench.tests.test_env_alfworld import Games, act, make_env
from evals.skill_bench.tests.webshop_support import make_shop

#: A memory text with braces, quotes and lines (``test_agent_context.MEMORY``).
MEMORY = "## Skills\n- check the {receptacle} first & 'quote' \"x\"\n- then act"

#: ``(split, index) -> {memory: fingerprint}`` of the step-1 request on the scripted shop.
WEBSHOP_STEP_1 = {
    ("test", 0): {
        "": "d8e088505ce8f0b4725a13dd422bed589a6f24be8b00be0021f590cc940c4b47",
        MEMORY: "4434d53c28959089f1cad4d5f040f538c804c798b3831eff290d636846d6ce74",
    },
    ("train", 0): {
        "": "cd53df354a242b522a474ed0b79117265e53479af62a96edcb992843a5f4ba18",
        MEMORY: "14ce03a33bf943121b9c140466e2b7a54c7634da023bab23a49840fda836cc7c",
    },
}
#: sha256 over the fingerprints of the five requests of the scripted ALFWorld episode.
ALFWORLD_REQUESTS = {
    "": "9da4ec89e36f73368599091a76a3cc5b07d43f20f7413a7f3a7817e83bc1db42",
    MEMORY: "6393e20e7483325b49d6a2bb64d3929e395ad7fff5072dfbe118c94db34e1465",
}
ALFWORLD_AGENT_CONTEXT = (
    "16c9a1b4efc2309df446ec401012bc3abbcd7be12ee2ddcdcf7e7040a2b21be2"
)


def _fingerprint(messages: list[dict[str, str]]) -> str:
    return ro._Request(messages).fingerprint()


@pytest.mark.asyncio
@pytest.mark.parametrize(("split", "index"), sorted(WEBSHOP_STEP_1))
async def test_webshops_step_1_request_is_byte_identical(tmp_path, split, index):
    env = make_shop(tmp_path).env()
    task = env.load_tasks(split)[index]
    async with env.episode(task) as episode:
        fields = (await episode.reset()).fields
    for memory, expected in WEBSHOP_STEP_1[(split, index)].items():
        messages = env.actor_messages(fields, memory)
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert _fingerprint(messages) == expected, repr(memory)


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", sorted(ALFWORLD_REQUESTS))
async def test_alfworlds_actor_requests_are_byte_identical(tmp_path, memory):
    env = make_env(Games(), tmp_path)
    task = env.load_tasks("train")[0]
    digest = hashlib.sha256()
    async with env.episode(task) as episode:
        fields = (await episode.reset()).fields
        for action in ["go to countertop 1", "look", "inventory", "look", "win"]:
            digest.update(_fingerprint(env.actor_messages(fields, memory)).encode())
            fields = (await episode.step(act(action))).fields
    assert digest.hexdigest() == ALFWORLD_REQUESTS[memory]


def test_alfworlds_agent_context_is_byte_identical(tmp_path):
    text = make_env(Games(), tmp_path).agent_context()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == ALFWORLD_AGENT_CONTEXT
