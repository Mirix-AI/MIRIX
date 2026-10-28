"""Offline end to end: task sessions -> learning runs -> skills -> retrieval.

Run through the fresh-database runner (MIRIX_PG_URI is read at import time):

    scripts/run_pg_integration_tests.sh tests/test_skill_learning_e2e_pg.py

MIRIX runs the way the eval harness runs it: ``scripts/start_server.py``
in a process of its own — uvicorn and the real FastAPI lifespan (table creation, the
session sweeper and lease recovery, the re-embedding pass, the retrieval-log purge), the
real managers on the runner's Postgres — with a throwaway HOME, no provider key,
``MIRIX_LLM_MAX_RETRIES=0`` and a network guard that refuses every connection that is not
to loopback. The tests talk to it over HTTP only, the way
``evals/skill_bench/methods/mirix.py`` does (``X-Client-ID`` / ``X-Org-ID`` headers, one
message per ``/memory/add``), and read Postgres directly only for what the API does not
show.

Every model call — the distiller, the curator, topic extraction, embeddings — goes to the
offline fake model server (``tests/fakes/fake_llm_server.py``) in this process. Its
responder (:class:`ScriptedModel`) plays the model from the request itself: a distiller
request gets two experiences about its session's topic, a curator batch gets the tool
calls ``skill_create`` / ``skill_edit`` / ``finish_memory_update`` for the experiences in
its payload, a topic extraction gets the topic named in the conversation. Embeddings put
each topic word on an axis of its own, so a skill's cosine to a query is 1.0 when they
share their topic and 0.0 otherwise. Nothing reaches a model provider.

``test_task_sessions_become_retrievable_skills`` — six task sessions (one with images,
one long enough to be distilled in chunks under a 21k-token window), closed one by one:
the fifth close fires a learning run, the flush takes the sixth; the
distiller gets the images in place and the chunks; the curator works in batches and
writes skills with version history; ``/memory/retrieve/conversation`` with
``memory_types=['procedural']`` returns the matching skill through the gate with its
scores, ``retrieval_id`` and a readable retrieval log row, and MIRIX's renderer turns it
into the injected text; the learning runs carry their metrics and usage. The
real app's OpenAPI document has every endpoint and field the harness checks for.

``test_a_restart_mid_run_releases_and_relearns_its_sessions`` — the server is killed
(SIGKILL) while a flush run is distilling; after the lease has run out, the restarted
server's startup recovery fails the run with ``lease_expired`` and releases its sessions,
and the next flush learns them.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import text

from curator_harness import words
from fakes.fake_llm_server import ChatResponse, FakeLLMServer
from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
START_SERVER = REPO / "scripts" / "start_server.py"

#: The small window that makes the long session chunk and the curator batch;
#: the output cap keeps the planning reserve at 1,600 tokens. The curator step
#: that edits the long topic's skill (its long triggers in the call) must still
#: fit under floor(window x 0.8) - 1,600.
CONTEXT_WINDOW = 21_000
MAX_OUTPUT_TOKENS = 2_000
EMBEDDING_MODEL = "google/gemini-embedding-001"  # the harness's
EMBEDDING_DIM = 3_072
RUN_TIMEOUT_S = 240.0

#: Topic words: each is an embedding axis of its own (see :func:`topic_embedding`).
TOPICS = (
    "spreadsheet",
    "invoice",
    "warehouse",
    "calendar",
    "thermostat",
    "passport",
    "telescope",
    "aquarium",
)
#: The long session's topic; its skill gets long triggers, so re-embedding it is chunked.
LONG_TOPIC = "warehouse"
#: The edited triggers' words next to a dash (``reconcile<topic>``) and CJK
#: word, searched by BM25.
NON_ASCII_WORD = "reconcile"
CJK_WORD = "对账"
#: The chunk cap of gemini-embedding-001 (2,048 x 0.88).
EMBEDDING_CHUNK_CAP = 1_800
DISTILLER_PROMPT_HEAD = "You are the Experience Distiller"
EXTRACTION_TOOL = "update_topic_and_time"
CURATOR_FINISH_TOOL = "finish_memory_update"
EXPERIENCE_CHARS = 2_400


def _harness_contract() -> Optional[Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]]:
    """What ``evals/skill_bench/methods/mirix.py`` requires of MIRIX's OpenAPI document
    (its ``REQUIRED_ENDPOINTS`` / ``REQUIRED_RETRIEVE_FIELDS``), imported from
    the harness itself so the two lists can never drift apart. None when the
    harness is not in this checkout (it lives on its own branch)."""
    try:
        from evals.skill_bench.methods.mirix import (
            REQUIRED_ENDPOINTS,
            REQUIRED_RETRIEVE_FIELDS,
        )
    except ImportError:
        return None

    return tuple(REQUIRED_ENDPOINTS), tuple(REQUIRED_RETRIEVE_FIELDS)


# The MIRIX server process: start_server.py, with the tokenizer vocabulary never
# downloaded (counts use the 3-chars-per-token estimate, as in the unit suite) and every
# connection that is not to loopback refused (no model provider can be reached).
LAUNCHER = r"""
import runpy
import sys

tests_dir, start_server, port = sys.argv[1:4]
sys.path.insert(0, tests_dir)

import pytest
import tiktoken.load
from db_isolation import block_non_loopback_network


def _no_download(path, *args, **kwargs):
    raise OSError(f"offline e2e: the tokenizer vocabulary is never downloaded ({path})")


tiktoken.load.read_file = _no_download
block_non_loopback_network(pytest.MonkeyPatch())
sys.argv = [start_server, "--host", "127.0.0.1", "--port", port, "--log-level", "warning"]
runpy.run_path(start_server, run_name="__main__")
"""


# --------------------------------------------------------------------------- #
# The MIRIX server process                                                     #
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MirixServer:
    """``scripts/start_server.py`` in its own process (and process group)."""

    def __init__(self, home: Path, log_path: Path) -> None:
        self.home = home
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> Dict[str, str]:
        """Only what MIRIX needs: no provider key can reach it."""
        env = {
            key: os.environ[key]
            for key in ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE")
            if key in os.environ
        }
        tiktoken_dir = self.home / "tiktoken-empty"
        tiktoken_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "HOME": str(self.home),
                "MIRIX_PG_URI": str(settings.mirix_pg_uri_no_default),
                "MIRIX_REDIS_ENABLED": "false",
                "MIRIX_LLM_MAX_RETRIES": "0",
                "MIRIX_LLM_TIMEOUT_SECONDS": "120",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TIKTOKEN_CACHE_DIR": str(tiktoken_dir),
            }
        )
        return env

    async def start(self, timeout: float = 120.0) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.port = _free_port()
        with open(self.log_path, "ab") as log:
            log.write(f"\n==== start on port {self.port} ====\n".encode())
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    LAUNCHER,
                    str(TESTS),
                    str(START_SERVER),
                    str(self.port),
                ],
                cwd=REPO,
                env=self.env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(trust_env=False) as http:
            while True:
                code = self.proc.poll()
                if code is not None:
                    raise AssertionError(
                        f"MIRIX exited with {code} during startup:\n{self.tail()}"
                    )
                try:
                    response = await http.get(f"{self.base_url}/health", timeout=2.0)
                    # The harness's readiness check (infra/mirix_server.py).
                    if (
                        response.status_code == 200
                        and response.json().get("status") == "healthy"
                    ):
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"MIRIX did not answer /health within {timeout:g} s:\n{self.tail()}"
                    )
                await asyncio.sleep(0.2)

    def kill(self) -> None:
        """A crash: SIGKILL, nothing of MIRIX's shutdown runs."""
        assert self.proc is not None
        os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=30)

    async def stop(self) -> None:
        """A graceful stop (SIGTERM), SIGKILL if it does not end in 30 s."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            await asyncio.to_thread(proc.wait, 30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.to_thread(proc.wait, 30)
        except ProcessLookupError:
            pass

    def errors(self) -> List[str]:
        """Logged errors and tracebacks: a run that "succeeded" while something
        failed on the way (a prune, a checkpoint, a tool call) shows up here."""
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return []
        return [
            line
            for line in content
            if " - ERROR - " in line or "] ERROR:" in line or "Traceback" in line
        ]

    def tail(self, lines: int = 120) -> str:
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return "(no server log)"
        return "\n".join(content[-lines:])


class Api:
    """MIRIX's REST API for one client, called like the harness's ``MirixApi``."""

    def __init__(self, base_url: str, *, client_id: str, org_id: str) -> None:
        self.client_id = client_id
        self.org_id = org_id
        self.http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Client-ID": client_id, "X-Org-ID": org_id},
            timeout=httpx.Timeout(180.0, connect=30.0),
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: Optional[Dict[str, Any]] = None,
        expect: Iterable[int] = (200,),
    ) -> Any:
        response = await self.http.request(method, path, json=json_body, params=params)
        assert response.status_code in tuple(expect), (
            f"{method} {path} -> {response.status_code}: {response.text[:3000]}"
        )
        return response.json()


# --------------------------------------------------------------------------- #
# The model                                                                    #
# --------------------------------------------------------------------------- #


def topic_embedding(text_value: str, dim: int) -> List[float]:
    """One axis per topic word, weighted by how often the text names it (a last
    axis when it names none), normalized: cosine 1.0 for texts on the same
    single topic, 0.0 for different topics, and a text naming two topics 5:4
    has cosine 5/sqrt(41) = 0.781 and 4/sqrt(41) = 0.625 to their skills (the
    relative gate's case). Chunks of one topic's text average to the same
    vector."""
    low = text_value.lower()
    counts = [low.count(topic) for topic in TOPICS]
    if not any(counts):
        counts.append(1)  # the "no topic" axis
    norm = math.sqrt(sum(c * c for c in counts))
    vector = [0.0] * dim
    for i, count in enumerate(counts):
        vector[i] = count / norm
    return vector


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _parts(content: Any) -> List[tuple]:
    """``[("text", text) | ("image", url)]`` in order."""
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for part in content or []:
        if part.get("type") == "text":
            out.append(("text", part.get("text", "")))
        elif part.get("type") == "image_url":
            out.append(("image", part["image_url"]["url"]))
    return out


def _tool_names(body: Dict[str, Any]) -> set:
    return {
        tool.get("function", {}).get("name")
        for tool in body.get("tools") or []
        if isinstance(tool, dict)
    }


def skill_name(topic: str) -> str:
    return f"{topic}-workflow"


def skill_fields(topic: str) -> Dict[str, Any]:
    return {
        "name": skill_name(topic),
        "description": f"Reusable steps for {topic} tasks.",
        "instructions": (
            f"1. Restate the {topic} goal.\n2. Apply the approach the user confirmed."
            "\n3. Verify the result before answering."
        ),
        "entry_type": "workflow",
        "triggers": [f"a task about a {topic}"],
        "examples": [{"input": f"a {topic} request", "output": "done, verified"}],
    }


class ScriptedModel:
    """Plays the model for MIRIX's three LLM roles, telling them apart by request.

    Thread-safe: the fake server answers from its own threads. Every request it
    could not answer is recorded in :attr:`errors` (the fake then returns a 500).
    """

    def __init__(self, session_topics: Dict[str, str]) -> None:
        self.session_topics = dict(session_topics)
        self.lock = threading.Lock()
        self.skill_ids: Dict[str, str] = {}
        self.plans: Dict[str, List[tuple]] = {}
        self.batches: List[Dict[str, Any]] = []
        self.edits = 0
        self.distill_requests: List[Dict[str, Any]] = []
        self.extraction_requests: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        # Holding: distiller requests of these sessions wait for ``release``.
        self.hold_sessions: set = set()
        self.held: set = set()
        self.release = threading.Event()
        # Holding the curator: its first request waits for ``release`` (and
        # is then answered into a dead socket, never curated).
        self.hold_curator = False
        self.curator_held = threading.Event()
        #: topic -> the triggers of its last edit.
        self.last_triggers: Dict[str, List[str]] = {}
        #: batch payload -> topic -> the ids of the batch's experiences on it,
        #: which the topic's create / edit cites (a finish with an experience
        #: neither cited nor triaged is refused once).
        self.topic_ids: Dict[str, Dict[str, List[str]]] = {}

    def __call__(self, body: Dict[str, Any]) -> ChatResponse:
        try:
            tools = _tool_names(body)
            if EXTRACTION_TOOL in tools:
                return self._extract(body)
            if CURATOR_FINISH_TOOL in tools:
                with self.lock:
                    hold = self.hold_curator
                if hold:
                    self.curator_held.set()
                    self.release.wait(timeout=300)
                    return ChatResponse(content="(the server is gone)")
                return self._curate(body)
            system = _text_of((body.get("messages") or [{}])[0].get("content"))
            if system.startswith(DISTILLER_PROMPT_HEAD):
                return self._distill(body)
            raise AssertionError(f"unexpected model request: tools={sorted(tools)}")
        except Exception as exc:
            with self.lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            raise

    # -- distiller ----------------------------------------------------------- #

    def _distill(self, body: Dict[str, Any]) -> ChatResponse:
        user = body["messages"][1]
        transcript = _text_of(user["content"])
        session_id = re.search(r"^session_id: (\S+)$", transcript, re.M).group(1)
        chunk = re.search(r"chunk (\d+)/(\d+) covering rounds (\d+)-(\d+)", transcript)
        record = {
            "session_id": session_id,
            "chunk": tuple(int(g) for g in chunk.groups()) if chunk else None,
            "parts": _parts(user["content"]),
            "max_completion_tokens": body.get("max_completion_tokens"),
        }
        with self.lock:
            self.distill_requests.append(record)
            hold = session_id in self.hold_sessions
            if hold:
                self.held.add(session_id)
        if hold:
            self.release.wait(timeout=300)
        topic = self.session_topics[session_id]
        where = f"chunk {chunk.group(1)}" if chunk else "whole session"
        experiences = [
            {
                "experience_type": kind,
                "title": f"{topic}: {label} ({where})",
                "content": f"When a {topic} task comes up, {label}. "
                + words(
                    EXPERIENCE_CHARS,
                    seed=zlib.crc32(f"{session_id}|{where}|{kind}".encode()),
                ),
                "importance": 0.8,
                "credibility": 0.9,
                "evidence": {"quote": "Great, that worked.", "signal_type": signal},
            }
            for kind, label, signal in (
                ("worth_learning", "check the totals first", "user_confirmation"),
                ("worth_avoiding", "never skip the final review", "user_critique"),
            )
        ]
        return ChatResponse(content=json.dumps(experiences))

    # -- curator ------------------------------------------------------------- #

    def _plan(self, payload: str) -> List[tuple]:
        """Per batch: create each new topic's skill or edit the existing one, then
        edit each skill created here that has a second experience in the batch
        (within the edit budget = the batch's experience count), then finish."""
        budget = int(re.search(r"Edit budget for this batch: (\d+)", payload).group(1))
        titles = re.findall(r"^### \[(?:LEARN|AVOID)\] (.+)$", payload, re.M)
        topics = [title.split(":", 1)[0].strip() for title in titles]
        ids = self.topic_ids.setdefault(payload, {})
        for experience_id, title in re.findall(
            r'^<experience id="([^"]+)">\n### \[(?:LEARN|AVOID)\] (.+)$', payload, re.M
        ):
            ids.setdefault(title.split(":", 1)[0].strip(), []).append(experience_id)
        counts = Counter(topics)
        plan: List[tuple] = []
        created = []
        for topic in dict.fromkeys(topics):
            if skill_name(topic) in self.skill_ids:
                plan.append(("edit", topic))
            else:
                plan.append(("create", topic))
                created.append(topic)
        room = budget - len(plan)
        for topic in created:
            if room > 0 and counts[topic] >= 2:
                plan.append(("edit", topic))
                room -= 1
        plan.append(("finish", None))
        self.batches.append(
            {
                "budget": budget,
                "experiences": len(titles),
                "topics": topics,
                "plan": plan,
            }
        )
        return plan

    def _edited_triggers(self, topic: str) -> List[str]:
        self.edits += 1
        triggers = [
            f"a task about a {topic}",
            f"a follow-up {topic} request (rev {self.edits})",
            # Words next to curly quotes and a dash, and a CJK word: stored
            # in a JSON column that escapes them, and still indexed as words
            # (found only with non-ASCII triggers).
            f"“{topic}” review—{NON_ASCII_WORD}{topic} {CJK_WORD}",
        ]
        if topic == LONG_TOPIC:
            # The retrieval text passes the 1,800-token chunk cap of
            # gemini-embedding-001, so re-embedding it is chunked; every
            # trigger stays within SKILL_MAX_TRIGGER_LEN (512).
            triggers += [
                f"{topic} case {i}: " + words(460, seed=100 + i) for i in range(12)
            ]
        self.last_triggers[topic] = triggers
        return triggers

    def _curate(self, body: Dict[str, Any]) -> ChatResponse:
        messages = body["messages"]
        start = max(
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user"
            and "## Skill curation — batch" in _text_of(m.get("content"))
        )
        payload = _text_of(messages[start]["content"])
        later = messages[start + 1 :]
        step = sum(1 for m in later if m.get("role") == "assistant")
        with self.lock:
            for m in later:
                if m.get("role") == "tool":
                    for name, skill_id in re.findall(
                        r"Skill '([^']+)' created successfully \(ID: ([^,)]+)",
                        _text_of(m.get("content")),
                    ):
                        self.skill_ids[name] = skill_id
            if payload not in self.plans:
                self.plans[payload] = self._plan(payload)
            action, topic = self.plans[payload][step]
            cited = list(self.topic_ids.get(payload, {}).get(topic, []))
            if action == "create":
                # Each citation quotes the text that applies it.
                quote = f"Restate the {topic} goal"
                call = {
                    "name": "skill_create",
                    "arguments": {
                        **skill_fields(topic),
                        "experience_ids": [{"id": i, "quote": quote} for i in cited],
                    },
                }
            elif action == "edit":
                triggers = self._edited_triggers(topic)
                call = {
                    "name": "skill_edit",
                    "arguments": {
                        "skill_id": self.skill_ids[skill_name(topic)],
                        "field": "triggers",
                        "value": json.dumps(triggers),
                        "experience_ids": [
                            {"id": i, "quote": triggers[1]} for i in cited
                        ],
                    },
                }
            else:
                call = {"name": CURATOR_FINISH_TOOL, "arguments": {}}
        return ChatResponse(tool_calls=[call])

    # -- topic extraction ---------------------------------------------------- #

    def _extract(self, body: Dict[str, Any]) -> ChatResponse:
        user_messages = [m for m in body["messages"] if m.get("role") == "user"]
        conversation = " ".join(_text_of(m.get("content")) for m in user_messages)
        images = [
            url
            for m in user_messages
            for kind, url in _parts(m.get("content"))
            if kind == "image"
        ]
        found = [topic for topic in TOPICS if topic in conversation.lower()]
        topic = f"{found[0]} export; totals" if found else "gardening; tomato seedlings"
        with self.lock:
            self.extraction_requests.append(
                {"conversation": conversation, "images": images, "topic": topic}
            )
        return ChatResponse(
            tool_calls=[{"name": EXTRACTION_TOOL, "arguments": {"topic": topic}}]
        )


# --------------------------------------------------------------------------- #
# Sessions                                                                     #
# --------------------------------------------------------------------------- #


def _png(color: str, size=(12, 9)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


def short_session(topic: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": f"Task: prepare the {topic} summary for the team."},
        {
            "role": "assistant",
            "content": f"I will open the {topic} and check its fields.",
        },
        {"role": "user", "content": "The totals are wrong, please fix them."},
        {
            "role": "assistant",
            "content": f"Fixed the totals of the {topic} and re-checked.",
        },
        {"role": "user", "content": "Great, that worked."},
    ]


def image_session(topic: str, pages: List[bytes]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here is the scanned {topic}, page one:"},
                {"type": "image_url", "image_url": {"url": _data_url(pages[0])}},
                {"type": "text", "text": "and page two:"},
                {"type": "image_url", "image_url": {"url": _data_url(pages[1])}},
                {"type": "text", "text": "Read the total due."},
            ],
        },
        {"role": "assistant", "content": f"The {topic} total due is 1,250.00."},
        {"role": "user", "content": "Great, that worked."},
    ]


def tool_session(topic: str) -> List[Dict[str, Any]]:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": f"{topic}_lookup", "arguments": '{"day": "monday"}'},
    }
    retry = {**call, "id": "call_2"}
    return [
        {"role": "user", "content": f"Task: find a free {topic} slot on Monday."},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "error: locked, retry"},
        {"role": "assistant", "content": None, "tool_calls": [retry]},
        {"role": "tool", "tool_call_id": "call_2", "content": "free: 10:00-11:00"},
        {"role": "assistant", "content": f"The {topic} has 10:00-11:00 free."},
        {"role": "user", "content": "Great, that worked."},
    ]


LONG_TASK_STATEMENT = f"Task: reorganize the {LONG_TOPIC} shelves by demand."


def long_session(
    topic: str, rounds: int = 24, chars: int = 1_300
) -> List[Dict[str, Any]]:
    """Short first and last rounds (the chunk frame) around long ones."""
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": LONG_TASK_STATEMENT},
        {"role": "assistant", "content": f"I will list the {topic} aisles first."},
    ]
    for i in range(rounds):
        messages.append(
            {"role": "user", "content": f"Aisle {i} report: " + words(chars, seed=i)}
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Moved the aisle {i} items: "
                + words(chars, seed=1_000 + i),
            }
        )
    messages += [
        {"role": "user", "content": "Is every aisle done?"},
        {"role": "assistant", "content": "Yes, every aisle is reorganized."},
        {"role": "user", "content": "Great, that worked."},
    ]
    return messages


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


async def _query(sql: str, **params) -> List[Dict[str, Any]]:
    from mirix.server.server import engine

    async with engine.connect() as conn:
        return [dict(row) for row in (await conn.execute(text(sql), params)).mappings()]


async def _execute(sql: str, **params) -> int:
    from mirix.server.server import engine

    async with engine.begin() as conn:
        return (await conn.execute(text(sql), params)).rowcount


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def _engine():
    """This module's pool on this module's loop; closed at the end."""
    from mirix.server import server

    await server.engine.dispose(close=False)
    yield
    await server.engine.dispose()


def _fake(model: ScriptedModel) -> FakeLLMServer:
    return FakeLLMServer(
        embedding_dim=EMBEDDING_DIM,
        embedding_fn=topic_embedding,
        default_responder=model,
    )


def _meta_agent_config(fake: FakeLLMServer) -> Dict[str, Any]:
    """The harness's ``meta_agent_config``, pointed at the fake."""
    return {
        "llm_config": {
            "model_endpoint_type": "openai",
            "model": "fake-model",
            "model_endpoint": fake.base_url,
            "context_window": CONTEXT_WINDOW,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "api_key": "fake-key",
        },
        "embedding_config": {
            "embedding_endpoint_type": "openrouter",
            "embedding_dim": EMBEDDING_DIM,
            "embedding_endpoint": fake.base_url,
            "embedding_model": EMBEDDING_MODEL,
            "api_key": "fake-key",
        },
    }


async def _set_up_tenant(server: MirixServer, fake: FakeLLMServer, tag: str):
    """Organization, client, meta agent and user, as the harness's
    ``_set_up_learning`` makes them. Returns ``(api, user_id, meta_agent_id)``."""
    suffix = uuid.uuid4().hex[:8]
    org_id, client_id = f"e2e-org-{tag}-{suffix}", f"e2e-client-{tag}-{suffix}"
    user_id, scope = f"e2e-user-{tag}-{suffix}", f"e2e_{tag}"
    api = Api(server.base_url, client_id=client_id, org_id=org_id)
    await api.call(
        "POST",
        "/organizations/create_or_get",
        json_body={"org_id": org_id, "name": org_id},
    )
    await api.call(
        "POST",
        "/clients/create_or_get",
        json_body={
            "client_id": client_id,
            "name": client_id,
            "org_id": org_id,
            "write_scope": scope,
            "read_scopes": [scope],
            "status": "active",
        },
        expect=(200, 201),
    )
    agent = await api.call(
        "POST",
        "/agents/meta/initialize",
        json_body={"config": _meta_agent_config(fake), "update_agents": False},
    )
    user = await api.call(
        "POST", "/users/create_or_get", json_body={"user_id": user_id, "name": user_id}
    )
    assert user["id"] == user_id
    return api, user_id, agent["id"]


async def _add_session(
    api: Api, user_id: str, agent_id: str, session_id: str, messages
):
    """``on_message`` of the harness: one message per ``/memory/add``."""
    for message in messages:
        response = await api.call(
            "POST",
            "/memory/add",
            json_body={
                "user_id": user_id,
                "meta_agent_id": agent_id,
                "messages": [message],
                "chaining": True,
                "verbose": False,
                "session_id": session_id,
                "session_tag": "task",
            },
        )
        assert response["status"] == "stored", response
        assert response["warnings"] == [] and response["learning_run_id"] is None


async def _close(api: Api, user_id: str, session_id: str) -> Dict[str, Any]:
    response = await api.call(
        "POST", f"/memory/sessions/{session_id}/close", json_body={"user_id": user_id}
    )
    assert response["already_closed"] is False and response["closed_at"], response
    return response


async def _wait_for_run(api: Api, run_id: str, timeout: float = RUN_TIMEOUT_S) -> Dict:
    deadline = time.monotonic() + timeout
    while True:
        run = await api.call("GET", f"/memory/learning_runs/{run_id}")
        if run["status"] in ("succeeded", "failed"):
            return run
        assert time.monotonic() < deadline, f"run {run_id} still {run['status']}"
        await asyncio.sleep(0.25)


async def _wait_until(predicate, what: str, timeout: float = RUN_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not await predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.2)


async def _skills(api: Api, user_id: str) -> Dict[str, Dict[str, Any]]:
    """The harness's ``list_skills``: every procedural item, by name."""
    listing = await api.call(
        "GET",
        "/memory/components",
        params={"user_id": user_id, "memory_type": "procedural", "limit": 1000},
    )
    procedural = listing["memories"]["procedural"]
    assert procedural["total_count"] == len(procedural["items"])
    return {item["name"]: item for item in procedural["items"]}


def _check_run_basics(run: Dict[str, Any], *, trigger: str, sessions: List[str]):
    assert run["status"] == "succeeded", run.get("error")
    assert run["error"] is None and run["mode"] == "procedural"
    assert run["trigger"] == trigger
    assert sorted(run["session_ids"]) == sorted(sessions)
    assert sorted(run["session_watermarks"]) == sorted(sessions)
    for key in ("heartbeat_at", "started_at", "finished_at"):
        assert run[key], key
    assert run["retry_events"] == [], run["retry_events"]


# --------------------------------------------------------------------------- #
# The learning flow                                                            #
# --------------------------------------------------------------------------- #


async def test_task_sessions_become_retrievable_skills(tmp_path):
    sessions = {
        f"e2e-task-{i}-{uuid.uuid4().hex[:6]}": t for i, t in enumerate(TOPICS[:6])
    }
    ids = list(sessions)
    model = ScriptedModel(sessions)
    pages = [_png("red"), _png("navy", size=(10, 14))]
    transcripts = {
        ids[0]: short_session(TOPICS[0]),
        ids[1]: image_session(TOPICS[1], pages),
        ids[2]: long_session(TOPICS[2]),
        ids[3]: tool_session(TOPICS[3]),
        ids[4]: short_session(TOPICS[4]),
        ids[5]: short_session(TOPICS[5]),
    }
    server = MirixServer(tmp_path / "home", tmp_path / "mirix.log")
    with _fake(model) as fake:
        await server.start()
        api = None
        try:
            await _check_openapi(server)
            api, user_id, agent_id = await _set_up_tenant(server, fake, "flow")
            await _learning_flow(
                api, fake, model, user_id, agent_id, ids, transcripts, pages
            )
            assert server.errors() == []
        except BaseException:
            print(f"---- MIRIX server log ----\n{server.tail(400)}")
            raise
        finally:
            if api is not None:
                await api.aclose()
            await server.stop()


async def _learning_flow(api, fake, model, user_id, agent_id, ids, transcripts, pages):
    for session_id in ids:
        await _add_session(api, user_id, agent_id, session_id, transcripts[session_id])

    # -- the trigger: nothing before the fifth close; the fifth fires --------
    for session_id in ids[:4]:
        assert (await _close(api, user_id, session_id))["learning_run_id"] is None
    first_run_id = (await _close(api, user_id, ids[4]))["learning_run_id"]
    assert first_run_id
    # The sixth end alone is one pending session, not five: no run.
    assert (await _close(api, user_id, ids[5]))["learning_run_id"] is None
    first = await _wait_for_run(api, first_run_id)
    _check_run_basics(first, trigger="close", sessions=ids[:5])

    # -- the flush takes the sixth, however few -------------------------------
    flush = await api.call(
        "POST",
        "/memory/skills/flush",
        json_body={
            "user_id": user_id,
            "close_open_sessions": False,
            "meta_agent_id": agent_id,
        },
    )
    assert flush["session_count"] == 1 and flush["learning_run_id"]
    second = await _wait_for_run(api, flush["learning_run_id"])
    _check_run_basics(second, trigger="flush", sessions=ids[5:])
    empty = await api.call(
        "POST", "/memory/skills/flush", json_body={"user_id": user_id}
    )
    assert empty == {
        "learning_run_id": None,
        "session_count": 0,
        "closed_session_ids": [],
        "pending_experiences": 0,
    }

    # -- the harness's idle check: every run listed, none active ---------------------
    listed = (
        await api.call(
            "GET",
            "/memory/learning_runs",
            params={"user_id": user_id, "limit": 500, "meta_agent_id": agent_id},
        )
    )["learning_runs"]
    assert [r["id"] for r in listed] == [second["id"], first["id"]]
    active = await api.call(
        "GET", "/memory/learning_runs", params={"user_id": user_id, "status": "active"}
    )
    assert active["learning_runs"] == []

    assert model.errors == [] and fake.unscripted_requests == 0
    await _check_distillation(first, second, model, ids, transcripts, pages)
    await _check_curation(first, second, model, ids)
    skills = await _check_skills(api, user_id, first, second)
    await _check_retrieval(api, model, user_id, skills, pages)
    await _check_store(api, user_id, ids)


async def _check_openapi(server: MirixServer) -> None:
    """The harness refuses a MIRIX whose OpenAPI document lacks what it calls."""
    contract = _harness_contract()
    if contract is None:
        return
    async with httpx.AsyncClient(base_url=server.base_url, trust_env=False) as http:
        document = (await http.get("/openapi.json")).json()
    endpoints, retrieve_fields = contract
    paths = document["paths"]
    missing = [(m, p) for m, p in endpoints if m not in paths.get(p, {})]
    assert missing == []
    body = paths["/memory/retrieve/conversation"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[1]
    fields = document["components"]["schemas"][ref]["properties"]
    assert set(retrieve_fields) <= set(fields)


async def _check_distillation(first, second, model, ids, transcripts, pages):
    """Every session distilled; the images in place; the long one in chunks."""
    metrics = {e["session_id"]: e for e in first["distill_metrics"]}
    metrics.update({e["session_id"]: e for e in second["distill_metrics"]})
    assert sorted(metrics) == sorted(ids)
    for session_id in ids:
        entry = metrics[session_id]
        assert entry["status"] == "ok", entry
        assert entry["messages"] == len(transcripts[session_id]), entry
        assert entry["llm_usage"]["total_tokens"] > 0, entry
    requests = {}
    for record in model.distill_requests:
        requests.setdefault(record["session_id"], []).append(record)

    # The image session: one multimodal message, both pages in their place.
    image_id = ids[1]
    assert metrics[image_id]["images"] == 2 and metrics[image_id]["chunks"] == 1
    [request] = requests[image_id]
    kinds = [kind for kind, _ in request["parts"]]
    assert kinds.count("image") == 2, kinds
    sent = [
        base64.b64decode(value.split(",", 1)[1])
        for kind, value in request["parts"]
        if kind == "image"
    ]
    assert sent == pages
    positions = {
        "page one": next(
            i
            for i, (k, v) in enumerate(request["parts"])
            if k == "text" and "page one" in v
        ),
        "page two": next(
            i
            for i, (k, v) in enumerate(request["parts"])
            if k == "text" and "page two" in v
        ),
    }
    image_positions = [i for i, (k, _) in enumerate(request["parts"]) if k == "image"]
    assert positions["page one"] < image_positions[0] < positions["page two"]
    assert positions["page two"] < image_positions[1]

    # The long session: chunked on round boundaries, each chunk framed.
    long_id = ids[2]
    entry = metrics[long_id]
    assert entry["chunks"] >= 2, entry
    chunk_requests = requests[long_id]
    assert sorted(r["chunk"][0] for r in chunk_requests) == list(
        range(1, entry["chunks"] + 1)
    )
    assert {r["chunk"][1] for r in chunk_requests} == {entry["chunks"]}
    for record in chunk_requests:
        assert LONG_TASK_STATEMENT in "".join(
            v for k, v in record["parts"] if k == "text"
        )
    assert entry["experiences"] == 2 * entry["chunks"]
    for session_id in (s for s in ids if s != long_id):
        assert metrics[session_id]["chunks"] == 1
        assert metrics[session_id]["experiences"] == 2
    # Every call's output cap is computed from its input, within 80%.
    for record in model.distill_requests:
        assert 1_024 <= record["max_completion_tokens"] <= int(MAX_OUTPUT_TOKENS * 0.8)

    # The run's usage sums its stages.
    for run in (first, second):
        usage = run["llm_usage"]
        assert usage["total_tokens"] > 0
        assert set(usage["by_stage"]) >= {"distill", "curate"}, usage


async def _check_curation(first, second, model, ids):
    """Batches in order, each finished and consumed, budget = experiences."""
    batches = first["curator_metrics"]
    assert len(batches) >= 2, batches
    assert [b["batch_id"] for b in batches] == [
        f"{first['id']}:b{i}" for i in range(len(batches))
    ]
    long_chunks = next(
        e["chunks"] for e in first["distill_metrics"] if e["session_id"] == ids[2]
    )
    assert sum(b["experiences"] for b in batches) == 2 * 4 + 2 * long_chunks
    for run in (first, second):
        for batch in run["curator_metrics"]:
            assert batch["status"] == "settled" and batch["outcome"] == "consume", batch
            assert batch["stop_reason"] == "finish", batch
            assert batch["budget"] == batch["experiences"] == batch["consumed"]
            assert batch["payload_tokens"] <= batch["capacity_tokens"], batch
            assert batch["rejected_tool_calls"] == 0, batch
            assert batch["writes"] == batch["creates"] + batch["edits"] >= 1, batch
            assert batch["steps"] == batch["writes"] + 1, batch
    [only] = second["curator_metrics"]
    assert only["creates"] == 1 and only["edits"] == 1
    # The model saw the batches in this order, each within its edit budget.
    assert len(model.batches) == len(batches) + 1
    for planned, batch in zip(model.batches, batches + [only]):
        assert planned["budget"] == batch["experiences"]
    # The long topic's re-embedding was chunked, counted on the run.
    # Whether a text passes the cap depends on the token counter: the server
    # runs on the 3-characters-per-token estimate (LAUNCHER: the vocabulary is
    # never downloaded), so the expectation is computed with that estimate —
    # never assumed (with the o200k vocabulary the same text may fit).
    long_text = _retrieval_text(
        skill_fields(LONG_TOPIC), model.last_triggers[LONG_TOPIC]
    )
    assert _estimated_chunks(long_text) >= 2, "the long skill must need chunks"
    for topic, triggers in model.last_triggers.items():
        if topic != LONG_TOPIC:
            assert (
                _estimated_chunks(_retrieval_text(skill_fields(topic), triggers)) == 1
            )
    assert first["embedding_chunked_calls"] >= 1
    assert second["embedding_chunked_calls"] == 0


def _retrieval_text(fields: Dict[str, Any], triggers: List[str]) -> str:
    from mirix.services.skill_history import skill_retrieval_text

    return skill_retrieval_text({**fields, "triggers": triggers})


def _estimated_chunks(text_value: str) -> int:
    """The chunks the server's guard makes of ``text_value`` on the
    characters-per-token estimate it runs on."""
    from mirix.embedding_guard import split_text_for_embedding
    from mirix.services.context_budget import FALLBACK_CHARS_PER_TOKEN

    def count(value: str) -> int:
        return math.ceil(len(value) / FALLBACK_CHARS_PER_TOKEN)

    if count(text_value) <= EMBEDDING_CHUNK_CAP:
        return 1
    return len(split_text_for_embedding(text_value, EMBEDDING_CHUNK_CAP, count))


async def _check_skills(api, user_id, first, second) -> Dict[str, Dict[str, Any]]:
    """One skill per topic, each with its version history."""
    skills = await _skills(api, user_id)
    assert sorted(skills) == sorted(skill_name(t) for t in TOPICS[:6])
    batch_ids = {b["batch_id"] for b in first["curator_metrics"]} | {
        b["batch_id"] for b in second["curator_metrics"]
    }
    for name, skill in skills.items():
        # The harness's snapshot call: no user_id.
        history = await api.call("GET", f"/memory/skills/{skill['id']}/history")
        versions = history["versions"]
        assert history["is_deleted"] is False
        assert [v["version_no"] for v in versions] == list(range(1, len(versions) + 1))
        assert versions[0]["change_type"] == "create"
        assert len(versions) >= 2, (name, versions)
        assert {v["change_type"] for v in versions[1:]} == {"edit"}
        assert all(v["run_id"] in batch_ids for v in versions), versions
        assert all(v["changed_fields"] == ["triggers"] for v in versions[1:])
        # The first version rebuilt from the content store.
        rebuilt = await api.call("GET", f"/memory/skills/{skill['id']}/versions/1")
        topic = name.removesuffix("-workflow")
        original = skill_fields(topic)
        for field in ("name", "description", "instructions", "entry_type", "triggers"):
            assert rebuilt["content"][field] == original[field], field
        assert skill["triggers"] != original["triggers"]  # the edit is live
        assert skill["description"] == original["description"]
    return skills


async def _check_retrieval(api, model, user_id, skills, pages):
    """Gated hybrid retrieval, scores, retrieval_id, renderer, retrieval log."""
    from mirix.client.render import render_retrieved_memories

    probe_session = f"e2e-probe-{uuid.uuid4().hex[:6]}"
    body = {
        "user_id": user_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "New task: fill in this spreadsheet."},
                    {"type": "image_url", "image_url": {"url": _data_url(pages[0])}},
                ],
            }
        ],
        "limit": 10,
        "local_model_for_retrieval": None,
        "memory_types": ["procedural"],
        "session_id": probe_session,
    }
    response = await api.call("POST", "/memory/retrieve/conversation", json_body=body)
    assert response["success"] is True and response["errors"] == {}
    assert response["extraction_status"] == "ok"
    assert response["topics"] == "spreadsheet export; totals"
    assert set(response["memories"]) == {"procedural"}
    [item] = response["memories"]["procedural"]["items"]
    expected = skills[skill_name("spreadsheet")]
    assert item["id"] == expected["id"]
    scores = item["retrieval_scores"]
    assert set(scores) == {"bm25", "cosine", "rrf"}
    assert scores["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert scores["bm25"] > 0 and scores["rrf"] > 0
    retrieval_id = response["retrieval_id"]
    assert retrieval_id
    # The image went to the extraction model with the text.
    extraction = model.extraction_requests[-1]
    assert len(extraction["images"]) == 1

    # MIRIX's own renderer (the harness's injection).
    rendered = render_retrieved_memories(response)
    assert rendered == "\n".join(
        [
            "## Relevant skills from memory",
            f"### {expected['name']}",
            expected["description"],
            "When to use: " + "; ".join(expected["triggers"]),
            "Instructions:",
            expected["instructions"],
            # One "- key: value; ..." line per example.
            "Examples:",
            *(
                "- " + "; ".join(f"{key}: {value}" for key, value in example.items())
                for example in expected["examples"]
            ),
        ]
    )

    # The gate can return nothing: an unrelated task gets no skill, renders "".
    unrelated = await api.call(
        "POST",
        "/memory/retrieve/conversation",
        json_body={
            **body,
            "messages": [{"role": "user", "content": "Plan the vegetable garden."}],
        },
    )
    assert unrelated["memories"]["procedural"]["items"] == []
    assert render_retrieved_memories(unrelated) == ""

    await _check_relative_gate(api, user_id, skills)
    await _check_non_ascii_keywords(api, user_id, skills)

    # One log row per retrieval, readable by the client that retrieved.
    logs = (
        await api.call(
            "GET", "/memory/retrieval_logs", params={"session_id": probe_session}
        )
    )["retrieval_logs"]
    assert [log["id"] for log in logs] == [unrelated["retrieval_id"], retrieval_id]
    log = logs[1]
    assert log["user_id"] == user_id and log["client_id"] == api.client_id
    assert log["session_id"] == probe_session
    assert log["memory_types"] == ["procedural"]
    assert log["extraction_status"] == "ok" and log["embedding_status"] == "ok"
    assert log["extracted_topics"] == "spreadsheet export; totals"
    trace = log["per_type"]["procedural"]
    assert trace["backend"] == "postgresql" and trace["method"] == "hybrid"
    assert trace["returned_ids"] == [expected["id"]]
    assert (
        trace["candidate_count"] == len(skills) and trace["missing_vector_count"] == 0
    )
    assert trace["gate"]["applied"] is True and trace["gate"]["passed"] == 1
    ranked = {c["id"]: c for c in trace["candidates"]}
    assert set(ranked) == {s["id"] for s in skills.values()}
    assert ranked[expected["id"]]["returned"] is True
    assert log["llm_usage"]["topic_extraction"]["calls"] == 1
    assert log["llm_usage"]["topic_extraction"]["input"]["images"] == 1
    assert log["latency_ms"] >= 0
    assert logs[0]["per_type"]["procedural"]["returned_ids"] == []


async def _check_relative_gate(api, user_id, skills) -> None:
    """The relative gate decides: a query naming spreadsheets 5 times
    and invoices 4 times has cosine 0.781 to the spreadsheet skill and 0.625
    to the invoice one — both above the calibrated 0.62 floor, but the
    invoice skill is below 0.9 x 0.781 = 0.703, so the gate keeps the
    spreadsheet skill only."""
    topic = " ".join(["spreadsheet"] * 5 + ["invoice"] * 4)
    response = await api.call(
        "GET",
        "/memory/retrieve/topic",
        params={"user_id": user_id, "topic": topic, "limit": 10},
    )
    assert response["success"] is True, response
    [item] = response["memories"]["procedural"]["items"]
    assert item["id"] == skills[skill_name("spreadsheet")]["id"]
    assert item["retrieval_scores"]["cosine"] == pytest.approx(5 / math.sqrt(41))
    logs = (
        await api.call("GET", "/memory/retrieval_logs", params={"user_id": user_id})
    )["retrieval_logs"]
    [log] = [entry for entry in logs if entry["id"] == response["retrieval_id"]]
    trace = log["per_type"]["procedural"]
    gate = trace["gate"]
    assert gate["applied"] is True and gate["passed"] == 1
    assert gate["min_similarity"] == pytest.approx(0.62)
    assert gate["threshold"] == pytest.approx(0.9 * 5 / math.sqrt(41))
    by_id = {c["id"]: c for c in trace["candidates"]}
    invoice = by_id[skills[skill_name("invoice")]["id"]]
    assert invoice["cosine"] == pytest.approx(4 / math.sqrt(41))
    assert gate["min_similarity"] < invoice["cosine"] < gate["threshold"]
    assert invoice["gate_passed"] is False and invoice["returned"] is False


async def _check_non_ascii_keywords(api, user_id, skills) -> None:
    """A trigger word next to a dash, and a CJK word, are BM25 terms:
    each skill's edited trigger is ``“<topic>” review—reconcile<topic> 对账``."""
    target = skills[skill_name("invoice")]["id"]
    for query, expected in (
        (f"{NON_ASCII_WORD}invoice", {target}),
        (CJK_WORD, {skill["id"] for skill in skills.values()}),
    ):
        found = await api.call(
            "GET",
            "/memory/search",
            params={
                "user_id": user_id,
                "query": query,
                "memory_type": "procedural",
                "search_method": "bm25",
                "limit": 10,
            },
        )
        assert found["success"] is True, found
        rows = [r for r in found["results"] if r["memory_type"] == "procedural"]
        assert {r["id"] for r in rows} == expected, (query, rows)
        assert all(r["retrieval_scores"]["bm25"] > 0 for r in rows), rows


async def _check_store(api: Api, user_id: str, ids: List[str]) -> None:
    """What only the database shows: sessions closed and distilled, the image
    session's parts stored by reference (never base64), experiences consumed,
    skills stamped with the client's scope, each run's dream checkpoint in the
    learning user's own memory."""
    sessions = await _query(
        "SELECT session_id, closed_at, close_reason, message_count "
        "FROM conversation_session WHERE user_id = :u",
        u=user_id,
    )
    assert sorted(s["session_id"] for s in sessions) == sorted(ids)
    assert {s["close_reason"] for s in sessions} == {"explicit"}
    rows = await _query(
        "SELECT session_id, content, content_parts, distilled_at "
        "FROM conversation_message WHERE user_id = :u",
        u=user_id,
    )
    assert rows and all(r["distilled_at"] is not None for r in rows)
    image_rows = [r for r in rows if r["content_parts"]]
    assert [r["session_id"] for r in image_rows] == [ids[1]]
    parts = image_rows[0]["content_parts"]
    assert [p["type"] for p in parts].count("image_ref") == 2
    assert "base64" not in image_rows[0]["content"]
    assert "[image: file-" in image_rows[0]["content"]
    experiences = await _query(
        "SELECT status, consumed_by, curation_attempts FROM skill_experience "
        "WHERE user_id = :u",
        u=user_id,
    )
    assert experiences and {e["status"] for e in experiences} == {"consumed"}
    assert all(":b" in e["consumed_by"] for e in experiences)
    assert {e["curation_attempts"] for e in experiences} == {0}
    scopes = await _query(
        "SELECT DISTINCT filter_tags::text AS tags FROM procedural_memory "
        "WHERE user_id = :u",
        u=user_id,
    )
    assert [json.loads(s["tags"]) for s in scopes] == [{"scope": "e2e_flow"}]
    checkpoints = await _query(
        "SELECT user_id, client_id FROM episodic_memory "
        "WHERE organization_id = :o AND event_type = 'auto_dream_checkpoint'",
        o=api.org_id,
    )
    assert checkpoints == [{"user_id": user_id, "client_id": api.client_id}] * 2


# --------------------------------------------------------------------------- #
# A restart mid-run                                                            #
# --------------------------------------------------------------------------- #


async def test_a_restart_mid_run_releases_and_relearns_its_sessions(tmp_path):
    sessions = {
        f"e2e-restart-{i}-{uuid.uuid4().hex[:6]}": topic
        for i, topic in enumerate(TOPICS[6:])
    }
    ids = list(sessions)
    model = ScriptedModel(sessions)
    home = tmp_path / "home"
    crashed = MirixServer(home, tmp_path / "mirix-before.log")
    restarted = MirixServer(home, tmp_path / "mirix-after.log")
    with _fake(model) as fake:
        await crashed.start()
        api = None
        try:
            api, user_id, agent_id = await _set_up_tenant(crashed, fake, "restart")
            await _restart_flow(
                api, fake, model, crashed, restarted, user_id, agent_id, ids, sessions
            )
        except BaseException:
            print(f"---- before the crash ----\n{crashed.tail(200)}")
            print(f"---- after the restart ----\n{restarted.tail(400)}")
            raise
        finally:
            model.release.set()
            if api is not None:
                await api.aclose()
            if crashed.proc is not None and crashed.proc.poll() is None:
                crashed.kill()
            await restarted.stop()


async def _restart_flow(
    api, fake, model, crashed, restarted, user_id, agent_id, ids, topics
):
    for session_id in ids:
        await _add_session(
            api, user_id, agent_id, session_id, short_session(topics[session_id])
        )
        assert (await _close(api, user_id, session_id))["learning_run_id"] is None

    # A flush run starts distilling; the model holds both sessions' calls.
    model.hold_sessions = set(ids)
    flush = await api.call(
        "POST", "/memory/skills/flush", json_body={"user_id": user_id}
    )
    run_id = flush["learning_run_id"]
    assert flush["session_count"] == 2 and run_id

    async def in_flight() -> bool:
        run = await api.call("GET", f"/memory/learning_runs/{run_id}")
        with model.lock:
            held = set(model.held)
        return run["status"] == "running" and held == set(ids)

    await _wait_until(in_flight, "the run to be distilling both sessions")

    # The crash: nothing of MIRIX's shutdown runs, the run stays 'running'.
    crashed.kill()
    [row] = await _query(
        "SELECT status, heartbeat_at FROM skill_learning_run WHERE id = :i", i=run_id
    )
    assert row["status"] == "running"
    # The lease runs out (MIRIX_LEARNING_RUN_LEASE_SECONDS, 300 s): the last heartbeat
    # is an hour old by the database clock the lease is measured on.
    assert (
        await _execute(
            "UPDATE skill_learning_run SET heartbeat_at = clock_timestamp() - "
            "interval '1 hour' WHERE id = :i",
            i=run_id,
        )
        == 1
    )
    model.release.set()  # the held answers go to a dead socket
    model.hold_sessions = set()

    # The restart: the startup lease recovery fails the run and releases its sessions.
    await api.aclose()
    await restarted.start()
    api = Api(restarted.base_url, client_id=api.client_id, org_id=api.org_id)
    try:
        await _after_restart(api, fake, model, user_id, run_id, ids, topics)
    finally:
        await api.aclose()
    assert crashed.errors() == [] and restarted.errors() == []


async def _after_restart(api, fake, model, user_id, run_id, ids, topics):
    assert await _query(
        "SELECT count(*) AS n FROM skill_experience WHERE user_id = :u", u=user_id
    ) == [{"n": 0}]

    async def reclaimed() -> bool:
        run = await api.call("GET", f"/memory/learning_runs/{run_id}")
        return run["status"] == "failed"

    await _wait_until(reclaimed, "the startup lease recovery", timeout=60)
    dead = await api.call("GET", f"/memory/learning_runs/{run_id}")
    assert dead["error"] == "lease_expired" and dead["finished_at"]
    # Killed mid-distill: nothing of the run was recorded, no session committed.
    assert not dead["distill_metrics"] and not dead["curator_metrics"]
    rows = await _query(
        "SELECT count(*) AS n, count(distilled_at) AS distilled "
        "FROM conversation_message WHERE user_id = :u",
        u=user_id,
    )
    assert rows[0]["n"] > 0 and rows[0]["distilled"] == 0

    # Released: the next flush claims exactly those sessions and learns them.
    again = await api.call(
        "POST", "/memory/skills/flush", json_body={"user_id": user_id}
    )
    assert again["session_count"] == 2 and again["learning_run_id"] != run_id
    relearned = await _wait_for_run(api, again["learning_run_id"])
    _check_run_basics(relearned, trigger="flush", sessions=ids)
    assert sorted(dead["session_ids"]) == sorted(relearned["session_ids"])
    assert {e["status"] for e in relearned["distill_metrics"]} == {"ok"}
    assert [b["stop_reason"] for b in relearned["curator_metrics"]] == ["finish"] * len(
        relearned["curator_metrics"]
    )
    skills = await _skills(api, user_id)
    assert sorted(skills) == sorted(skill_name(t) for t in topics.values())
    rows = await _query(
        "SELECT count(*) AS n, count(distilled_at) AS distilled "
        "FROM conversation_message WHERE user_id = :u",
        u=user_id,
    )
    assert rows[0]["n"] == rows[0]["distilled"] > 0
    assert model.errors == [] and fake.unscripted_requests == 0


# --------------------------------------------------------------------------- #
# A crash after distilling: the flush curates what was left                   #
# --------------------------------------------------------------------------- #


async def test_a_crash_after_distilling_leaves_experiences_a_flush_curates(tmp_path):
    """The run distilled its sessions (committed) and died in the curator: the
    sessions are learned, their experiences are pending. The flush has no
    session to claim, yet it starts a run that curates those experiences."""
    sessions = {
        f"e2e-leftover-{i}-{uuid.uuid4().hex[:6]}": topic
        for i, topic in enumerate(TOPICS[6:])
    }
    ids = list(sessions)
    model = ScriptedModel(sessions)
    home = tmp_path / "home"
    crashed = MirixServer(home, tmp_path / "mirix-before.log")
    restarted = MirixServer(home, tmp_path / "mirix-after.log")
    with _fake(model) as fake:
        await crashed.start()
        api = None
        try:
            api, user_id, agent_id = await _set_up_tenant(crashed, fake, "leftover")
            for session_id in ids:
                await _add_session(
                    api,
                    user_id,
                    agent_id,
                    session_id,
                    short_session(sessions[session_id]),
                )
                assert (await _close(api, user_id, session_id))[
                    "learning_run_id"
                ] is None
            model.hold_curator = True
            flush = await api.call(
                "POST", "/memory/skills/flush", json_body={"user_id": user_id}
            )
            run_id = flush["learning_run_id"]
            assert flush["session_count"] == 2 and run_id

            async def curating() -> bool:
                return model.curator_held.is_set()

            await _wait_until(curating, "the run to reach the curator")
            crashed.kill()
            pending = await _query(
                "SELECT count(*) AS n FROM skill_experience "
                "WHERE user_id = :u AND status = 'pending'",
                u=user_id,
            )
            assert pending[0]["n"] == 4  # two per session, all waiting
            assert (
                await _execute(
                    "UPDATE skill_learning_run SET heartbeat_at = clock_timestamp() "
                    "- interval '1 hour' WHERE id = :i",
                    i=run_id,
                )
                == 1
            )
            with model.lock:
                model.hold_curator = False
            model.release.set()

            await api.aclose()
            await restarted.start()
            api = Api(restarted.base_url, client_id=api.client_id, org_id=api.org_id)

            async def reclaimed() -> bool:
                run = await api.call("GET", f"/memory/learning_runs/{run_id}")
                return run["status"] == "failed"

            await _wait_until(reclaimed, "the startup lease recovery", timeout=60)
            rows = await _query(
                "SELECT count(*) AS n, count(distilled_at) AS distilled "
                "FROM conversation_message WHERE user_id = :u",
                u=user_id,
            )
            assert rows[0]["n"] == rows[0]["distilled"] > 0  # nothing to claim

            again = await api.call(
                "POST", "/memory/skills/flush", json_body={"user_id": user_id}
            )
            assert again["session_count"] == 0
            assert again["pending_experiences"] == 4
            assert again["learning_run_id"] and again["learning_run_id"] != run_id
            curated = await _wait_for_run(api, again["learning_run_id"])
            assert curated["status"] == "succeeded", curated
            assert curated["session_ids"] == [] and not curated["distill_metrics"]
            batches = curated["curator_metrics"]
            assert sum(b.get("consumed", 0) for b in batches) == 4, batches
            skills = await _skills(api, user_id)
            assert sorted(skills) == sorted(skill_name(t) for t in sessions.values())
            assert await _query(
                "SELECT count(*) AS n FROM skill_experience "
                "WHERE user_id = :u AND status = 'pending'",
                u=user_id,
            ) == [{"n": 0}]
            # Nothing left: the next flush starts no run.
            idle = await api.call(
                "POST", "/memory/skills/flush", json_body={"user_id": user_id}
            )
            assert idle["learning_run_id"] is None
            assert idle["pending_experiences"] == 0
            assert model.errors == [] and fake.unscripted_requests == 0
            assert crashed.errors() == [] and restarted.errors() == []
        except BaseException:
            print(f"---- before the crash ----\n{crashed.tail(200)}")
            print(f"---- after the restart ----\n{restarted.tail(400)}")
            raise
        finally:
            model.release.set()
            if api is not None:
                await api.aclose()
            if crashed.proc is not None and crashed.proc.poll() is None:
                crashed.kill()
            await restarted.stop()
