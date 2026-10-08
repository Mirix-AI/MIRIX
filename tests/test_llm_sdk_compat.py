"""The model SDKs MIRIX installs accept what MIRIX hands them.

Two failures used to hide behind fully mocked SDKs:

* ``requirements.txt`` / ``pyproject.toml`` left ``anthropic`` unpinned
  while ``uv.lock`` pinned 0.94.0, so ``pip install`` brought anthropic 1.x:
  it runs on ``httpx2`` and its ``messages.create`` no longer takes
  ``temperature``, which MIRIX always sends — every Anthropic call failed;
* ``llm_timeout()`` returned an ``httpx.Timeout``, which the
  ``httpx2``-based SDK rejects; a test passed only because importing
  ``openai`` first relabels ``httpx.Timeout``.

So these tests use the INSTALLED SDKs for real, against local stubs (no
network, no provider, no key): each SDK client is constructed with MIRIX's
transport options, and the Anthropic client sends a real request — built by
MIRIX, sent by the SDK — to a local Messages-API stub, also in a fresh
interpreter that never imported ``openai``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

from mirix import constants
from mirix.llm_api import helpers
from mirix.schemas.enums import MessageRole
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent

REPO = Path(__file__).resolve().parent.parent

TOOLS = [
    {
        "name": "open_drawer",
        "description": "Open a drawer by its number.",
        "parameters": {
            "type": "object",
            "properties": {"number": {"type": "integer", "description": "which one"}},
            "required": ["number"],
        },
    }
]


@pytest.fixture(autouse=True)
def direct_to_the_stub(monkeypatch):
    """Never through a system proxy: the stubs are on loopback."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


# --------------------------------------------------------------------------- #
# A local Anthropic Messages API stub                                          #
# --------------------------------------------------------------------------- #

_TOOL_USE_ANSWER = {
    "id": "msg_stub_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-stub",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_stub_1",
            "name": "open_drawer",
            "input": {"number": 3},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7},
}


class AnthropicStub:
    """``POST /v1/messages`` answered with one tool call; requests recorded."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002 - stdlib signature
                pass

            def do_POST(self):  # noqa: N802 - stdlib naming
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append(
                    {
                        "path": self.path.split("?", 1)[0],
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "json": body,
                    }
                )
                payload = json.dumps(_TOOL_USE_ANSWER).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "AnthropicStub":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def anthropic_stub():
    with AnthropicStub() as stub:
        yield stub


def _messages() -> List[Message]:
    return [
        Message(
            role=MessageRole.system, content=[TextContent(text="You are an agent.")]
        ),
        Message(role=MessageRole.user, content=[TextContent(text="Open drawer 3.")]),
    ]


# --------------------------------------------------------------------------- #
# The installed Anthropic SDK is the one MIRIX's client speaks                 #
# --------------------------------------------------------------------------- #


def _version_tuple(text: str):
    return tuple(int(p) for p in text.split(".")[:3] if p.isdigit())


class TestPinnedAnthropic:
    def test_the_installed_sdk_is_within_the_pin(self):
        import anthropic

        version = _version_tuple(anthropic.__version__)
        assert (0, 94, 0) <= version < (1, 0, 0), anthropic.__version__

    def test_every_dependency_list_pins_the_same_range(self):
        spec = "anthropic>=0.94.0,<1.0.0"
        assert f'"{spec}"' in (REPO / "pyproject.toml").read_text()
        assert spec in (REPO / "requirements.txt").read_text().splitlines()
        assert f'"{spec}"' in (REPO / "samples" / "pyproject.toml").read_text()
        lock = (REPO / "uv.lock").read_text()
        assert '{ name = "anthropic", specifier = ">=0.94.0,<1.0.0" }' in lock
        # The locked version itself is inside the range.
        locked = lock.split('name = "anthropic"\nversion = "', 1)[1].split('"', 1)[0]
        assert (0, 94, 0) <= _version_tuple(locked) < (1, 0, 0)

    @pytest.mark.asyncio
    async def test_mirix_builds_and_sends_a_request_the_sdk_accepts(
        self, anthropic_stub, monkeypatch
    ):
        # End to end on the installed SDK: MIRIX builds the request (with
        # temperature, the budgeted max_tokens, tools and tool_choice), the
        # SDK sends it, MIRIX parses the answer.
        from mirix.llm_api.anthropic_client import AnthropicClient
        from mirix.services.provider_manager import ProviderManager

        async def key(self):
            return "sk-ant-local-stub"

        monkeypatch.setattr(ProviderManager, "get_anthropic_override_key", key)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", anthropic_stub.base_url)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        client = AnthropicClient(
            LLMConfig(
                model="claude-stub",
                model_endpoint_type="anthropic",
                context_window=200_000,
                max_output_tokens=8_000,
                put_inner_thoughts_in_kwargs=True,
            )
        )
        response = await client.send_llm_request(messages=_messages(), tools=TOOLS)

        [call] = response.choices[0].message.tool_calls
        assert call.function.name == "open_drawer"
        assert json.loads(call.function.arguments) == {"number": 3}
        assert response.usage.total_tokens == 19
        [sent] = anthropic_stub.requests
        assert sent["path"] == "/v1/messages"
        assert sent["headers"]["x-api-key"] == "sk-ant-local-stub"
        assert "tools-2024-04-04" in sent["headers"]["anthropic-beta"]
        body = sent["json"]
        assert body["model"] == "claude-stub"
        assert "temperature" in body  # what anthropic 1.x no longer accepts
        assert body["max_tokens"] == 8_000 * 4 // 5  # a small input
        assert body["tools"][0]["name"] == "open_drawer"
        assert body["system"] == "You are an agent."

    def test_in_a_fresh_interpreter_without_openai(self, anthropic_stub):
        # Importing openai relabels httpx.Timeout, which masked the
        # anthropic 1.x type check. In an interpreter that never imports
        # openai, MIRIX's transport options still build a working client.
        script = textwrap.dedent(
            f"""
            import asyncio, sys
            import anthropic
            from mirix.llm_api.helpers import llm_transport_options

            assert "openai" not in sys.modules, "openai would mask the httpx.Timeout bug"

            async def main():
                options = llm_transport_options(anthropic)
                assert type(options["timeout"]) is anthropic.Timeout
                async with anthropic.AsyncAnthropic(
                    api_key="sk-ant-local-stub",
                    base_url={anthropic_stub.base_url!r},
                    **options,
                ) as client:
                    message = await client.beta.messages.create(
                        model="claude-stub",
                        max_tokens=64,
                        temperature=0.7,
                        messages=[{{"role": "user", "content": "hi"}}],
                        betas=["tools-2024-04-04"],
                    )
                assert message.content[0].name == "open_drawer", message
                assert "openai" not in sys.modules
                print("OK", anthropic.__version__, client.timeout)

            asyncio.run(main())
            """
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "OPENROUTER_API_KEY",
                "GOOGLE_API_KEY",
            )
        }
        env.update(
            {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        )
        done = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr[-3000:]
        assert "OK" in done.stdout
        assert len(anthropic_stub.requests) == 1

    @pytest.mark.asyncio
    async def test_an_unreachable_host_is_a_connection_error_not_a_type_error(
        self, monkeypatch
    ):
        # The symptom: with a rejected timeout type every call died inside
        # the SDK as TypeError('float' + 'Timeout'), wrapped as a connection
        # error — an endless, silent infrastructure backoff. A closed local
        # port must fail as a real connection error.
        import anthropic

        from mirix.services.retry_policy import ErrorKind, classify_error

        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        async with anthropic.AsyncAnthropic(
            api_key="sk-ant-local-stub",
            base_url="http://127.0.0.1:9",
            **helpers.llm_transport_options(anthropic),
        ) as client:
            with pytest.raises(anthropic.APIConnectionError) as raised:
                await client.messages.create(
                    model="claude-stub",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hi"}],
                )
        cause = raised.value.__cause__
        assert not isinstance(cause, TypeError), repr(cause)
        assert isinstance(cause, httpx.ConnectError), repr(cause)
        assert classify_error(raised.value) is ErrorKind.INFRA


# --------------------------------------------------------------------------- #
# A timeout every SDK accepts                                                  #
# --------------------------------------------------------------------------- #


class TestSDKTimeouts:
    @pytest.mark.asyncio
    async def test_each_sdk_client_is_constructed_with_mirix_options(self, monkeypatch):
        # Real SDK clients, constructed and closed; nothing is sent.
        import anthropic
        import openai

        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 42.0)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 1)
        clients = [
            openai.AsyncOpenAI(
                api_key="fake",
                base_url="http://127.0.0.1:9/v1",
                **helpers.llm_transport_options(openai),
            ),
            openai.AsyncAzureOpenAI(
                api_key="fake",
                api_version="2024-10-01-preview",
                azure_endpoint="http://127.0.0.1:9",
                **helpers.llm_transport_options(openai),
            ),
            anthropic.AsyncAnthropic(
                api_key="fake", **helpers.llm_transport_options(anthropic)
            ),
        ]
        for client in clients:
            assert client.timeout.read == 42.0 and client.timeout.connect == 5.0
            assert client.max_retries == 1
            await client.close()
        # MIRIX's own httpx calls (the Google client) get an httpx.Timeout.
        options = helpers.llm_transport_options()
        assert isinstance(options["timeout"], httpx.Timeout)
        async with httpx.AsyncClient(timeout=options["timeout"]) as raw:
            assert raw.timeout.read == 42.0 and raw.timeout.connect == 5.0

    @pytest.mark.asyncio
    async def test_mirix_clients_hand_their_sdk_its_own_timeout_type(self, monkeypatch):
        import anthropic
        import openai

        from mirix.llm_api.anthropic_client import AnthropicClient
        from mirix.llm_api.azure_openai_client import AzureOpenAIClient
        from mirix.llm_api.openai_client import OpenAIClient
        from mirix.services.provider_manager import ProviderManager

        async def key(self):
            return "sk-ant-fake"

        monkeypatch.setattr(ProviderManager, "get_anthropic_override_key", key)
        spied = []
        real = helpers.llm_timeout

        def spy(sdk=None):
            spied.append(sdk)
            return real(sdk)

        monkeypatch.setattr(helpers, "llm_timeout", spy)

        sdk_client = await AnthropicClient(
            LLMConfig(
                model="claude-x", model_endpoint_type="anthropic", context_window=8000
            )
        )._get_anthropic_client(async_client=True)
        assert type(sdk_client.timeout) is anthropic.Timeout
        await sdk_client.close()

        kwargs = await OpenAIClient(
            LLMConfig(
                model="m",
                model_endpoint_type="openai",
                model_endpoint="http://127.0.0.1:9/v1",
                context_window=8000,
                api_key="fake",
            )
        )._prepare_client_kwargs()
        assert type(kwargs["timeout"]) is openai.Timeout
        await openai.AsyncOpenAI(**kwargs).close()

        kwargs = await AzureOpenAIClient(
            LLMConfig(
                model="gpt",
                model_endpoint_type="azure_openai",
                context_window=8000,
                api_key="k",
                azure_endpoint="http://127.0.0.1:9",
                api_version="2024-10-01-preview",
            )
        )._prepare_client_kwargs()
        assert type(kwargs["timeout"]) is openai.Timeout
        await openai.AsyncAzureOpenAI(**kwargs).close()
        assert spied == [anthropic, openai, openai]

    def test_an_sdk_on_another_http_stack_gets_its_own_timeout_type(self, monkeypatch):
        # anthropic 1.x exports httpx2.Timeout, not httpx.Timeout.
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 30.0)

        class OtherStackTimeout:
            def __init__(self, timeout, *, connect):
                self.read, self.connect = timeout, connect

        timeout = helpers.llm_timeout(SimpleNamespace(Timeout=OtherStackTimeout))
        assert isinstance(timeout, OtherStackTimeout)
        assert (timeout.read, timeout.connect) == (30.0, 5.0)

    def test_an_sdk_without_a_usable_timeout_type_gets_float_seconds(self, monkeypatch):
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 30.0)

        class Positional:
            def __init__(self, a, b, c, d):
                pass

        assert helpers.llm_timeout(SimpleNamespace(Timeout=Positional)) == 30.0
        assert helpers.llm_timeout(SimpleNamespace()) == 30.0
        assert helpers.llm_timeout(SimpleNamespace(Timeout="not a type")) == 30.0

    def test_the_connect_timeout_never_exceeds_the_budget(self, monkeypatch):
        import anthropic

        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 2.0)
        assert helpers.llm_timeout().connect == 2.0
        assert helpers.llm_timeout(anthropic).connect == 2.0


class TestConnectTimeoutDetection:
    def test_httpx_and_same_named_classes_of_other_stacks(self):
        from mirix.services.retry_policy import is_connect_timeout

        request = httpx.Request("POST", "http://127.0.0.1:9")
        assert is_connect_timeout(httpx.ConnectTimeout("x", request=request))
        assert not is_connect_timeout(httpx.ReadTimeout("x", request=request))
        assert not is_connect_timeout(None)

        class TimeoutException(Exception):  # e.g. httpx2's hierarchy
            pass

        class ConnectTimeout(TimeoutException):
            pass

        assert is_connect_timeout(ConnectTimeout("no route"))

    def test_an_sdk_timeout_caused_by_a_connect_timeout_is_infrastructure(self):
        import anthropic
        import openai

        from mirix.errors import LLMConnectionError, LLMTimeoutError
        from mirix.llm_api.anthropic_client import AnthropicClient
        from mirix.llm_api.openai_client import OpenAIClient
        from mirix.services.retry_policy import ErrorKind, classify_error

        class ConnectTimeout(Exception):  # another stack's connect timeout
            pass

        request = httpx.Request("POST", "http://127.0.0.1:9")
        config = LLMConfig(model="m", model_endpoint_type="openai", context_window=8000)
        for sdk, client in (
            (anthropic, AnthropicClient(config)),
            (openai, OpenAIClient(config)),
        ):
            try:
                raise sdk.APITimeoutError(request=request) from ConnectTimeout("x")
            except sdk.APITimeoutError as exc:
                mapped = client.handle_llm_error(exc)
                assert isinstance(mapped, LLMConnectionError)
                assert not isinstance(mapped, LLMTimeoutError)
                assert classify_error(exc) is ErrorKind.INFRA
