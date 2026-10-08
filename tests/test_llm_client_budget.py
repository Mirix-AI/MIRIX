"""LLM calls follow the context budget.

Against the offline fake OpenAI-compatible server (tests/fakes):

* with ``max_output_tokens`` set, the ``max_completion_tokens`` actually SENT is
  ``output_cap(estimated input)`` — the model's output ceiling for small
  inputs, what is left of 80% of the window for large ones; without it the
  fixed ``max_tokens`` is sent as before;
* an input that would leave < 1,024 output tokens raises
  ``ContextBudgetExceeded`` before ANY request reaches the server;
* ``MIRIX_LLM_TIMEOUT_SECONDS`` / ``MIRIX_LLM_MAX_RETRIES`` are applied to the
  SDK clients explicitly (and behave: a retry after a 5xx, none with 0, a
  timed-out call is an ``LLMTimeoutError``);
* the Anthropic and Google clients size their ``max_tokens`` /
  ``max_output_tokens`` the same way; the Google client's HTTP call has its own
  transport retry loop (Retry-After honored, asyncio.sleep only);
* endpoint type ``openrouter`` is the OpenAI-compatible client with
  OpenRouter's key (never the OpenAI key), and an ``openrouter`` embedding
  config works through the OpenAI-compatible ``/embeddings`` path.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from fakes.fake_llm_server import Fault
from mirix import constants
from mirix.errors import (
    ContextBudgetExceeded,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from mirix.llm_api import helpers
from mirix.llm_api.llm_client import LLMClient
from mirix.llm_api.openai_client import OPENROUTER_DEFAULT_BASE_URL, OpenAIClient
from mirix.schemas.enums import MessageRole
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.services import context_budget
from mirix.services.context_budget import (
    MIN_OUTPUT_TOKENS,
    ModelLimits,
    get_token_counter,
)
from mirix.settings import model_settings

pytestmark = pytest.mark.asyncio

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
def direct_to_the_fake(monkeypatch):
    """Talk to the fake server directly: httpx otherwise honors a system proxy
    (e.g. macOS network settings), which would turn a dropped connection into
    the proxy's 502."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture(autouse=True)
def fallback_counter():
    """Count with the deterministic 3-chars-per-token fallback (no vocabulary)."""
    context_budget._reset_tokenizer_state_for_tests()
    yield
    context_budget._reset_tokenizer_state_for_tests()


def _messages(user_text="Put a mug in the sink."):
    return [
        Message(
            role=MessageRole.system, content=[TextContent(text="You are an agent.")]
        ),
        Message(role=MessageRole.user, content=[TextContent(text=user_text)]),
    ]


def _config(base_url, **overrides):
    values = dict(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=base_url,
        context_window=200_000,
        max_output_tokens=32_000,
        api_key="fake-key",
    )
    values.update(overrides)
    return LLMConfig(**values)


def _expected_cap(sent_body, config):
    counted = {k: v for k, v in sent_body.items() if k != "max_completion_tokens"}
    estimated = get_token_counter().count_request(counted)
    return ModelLimits.from_llm_config(config).output_cap(estimated), estimated


# --------------------------------------------------------------------------- #
# OpenAI-compatible client against the fake server                             #
# --------------------------------------------------------------------------- #


class TestOpenAICompatibleCap:
    async def test_small_input_gets_the_output_ceiling(self, fake_llm_server):
        fake_llm_server.queue_text("done")
        config = _config(fake_llm_server.base_url)
        client = LLMClient.create(config)

        response = await client.send_llm_request(messages=_messages())

        assert response.choices[0].message.content == "done"
        sent = fake_llm_server.last_chat_request().json
        expected, _ = _expected_cap(sent, config)
        assert sent["max_completion_tokens"] == expected == 32_000 * 4 // 5

    async def test_large_input_gets_what_is_left_of_the_window(self, fake_llm_server):
        fake_llm_server.queue_tool_call("open_drawer", {"number": 1})
        config = _config(fake_llm_server.base_url, context_window=20_000)
        client = LLMClient.create(config)
        long_text = "observation " * 3_500  # ~14k tokens at 3 chars/token

        await client.send_llm_request(messages=_messages(long_text), tools=TOOLS)

        sent = fake_llm_server.last_chat_request().json
        expected, estimated = _expected_cap(sent, config)
        assert sent["max_completion_tokens"] == expected
        assert expected == 16_000 - estimated  # the window binds, not the ceiling
        assert 1_024 <= expected < 25_600
        assert sent["tools"]  # tool schemas are part of the estimate

    async def test_without_max_output_tokens_the_request_is_unchanged(
        self, fake_llm_server
    ):
        """The budget is opt-in. Without max_output_tokens the request is
        what it was before this change: ChatCompletionRequest has no
        max_completion_tokens field, so the configured max_tokens never reached
        the wire and the provider's default applies — kept as is, so existing
        deployments do not suddenly get a 4,096-token output cap."""
        fake_llm_server.queue_text("ok")
        config = _config(
            fake_llm_server.base_url, max_output_tokens=None, max_tokens=4096
        )
        await LLMClient.create(config).send_llm_request(messages=_messages())
        sent = fake_llm_server.last_chat_request().json
        assert "max_completion_tokens" not in sent and "max_tokens" not in sent

    async def test_oversize_input_raises_before_any_request(self, fake_llm_server):
        config = _config(
            fake_llm_server.base_url, context_window=8_000, max_output_tokens=8_000
        )
        client = LLMClient.create(config)
        huge = "x" * 20_000  # ~6,667 tokens: 6,400 usable - input < 1,024

        with pytest.raises(ContextBudgetExceeded) as exc:
            await client.send_llm_request(messages=_messages(huge))

        assert exc.value.output_cap < 1_024
        assert fake_llm_server.chat_requests() == []
        assert fake_llm_server.requests == []

    async def test_the_cap_follows_the_luna_table_on_the_wire(self, fake_llm_server):
        fake_llm_server.queue_text("ok")
        config = _config(
            fake_llm_server.base_url,
            context_window=1_050_000,
            max_output_tokens=128_000,
        )
        await LLMClient.create(config).send_llm_request(messages=_messages())
        assert (
            fake_llm_server.last_chat_request().json["max_completion_tokens"] == 102_400
        )


class TestTransportSettings:
    async def test_defaults_are_explicit(self, fake_llm_server):
        assert constants.LLM_TIMEOUT_SECONDS == 600.0
        assert constants.LLM_MAX_RETRIES == 2
        kwargs = await OpenAIClient(
            _config(fake_llm_server.base_url)
        )._prepare_client_kwargs()
        timeout = kwargs["timeout"]
        assert timeout.read == timeout.write == timeout.pool == 600.0
        assert kwargs["max_retries"] == 2
        # Connecting never waits the whole call budget — an unreachable
        # provider is an infrastructure error in seconds (the SDKs' default).
        assert timeout.connect == helpers.LLM_CONNECT_TIMEOUT_SECONDS == 5.0

    async def test_settings_reach_the_sdk_client(self, fake_llm_server, monkeypatch):
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 12.5)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        kwargs = await OpenAIClient(
            _config(fake_llm_server.base_url)
        )._prepare_client_kwargs()
        assert kwargs["timeout"].read == 12.5 and kwargs["max_retries"] == 0
        assert kwargs["timeout"].connect == 5.0
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 2.0)
        assert helpers.llm_timeout().connect == 2.0  # never above the budget

    def test_settings_mirror_the_effective_values(self, monkeypatch):
        from mirix.settings import settings

        assert settings.llm_timeout_seconds == 600.0
        assert settings.llm_max_retries == 2
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 12.5)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        assert settings.llm_timeout_seconds == 12.5 and settings.llm_max_retries == 0

    def test_the_harness_settings_probe_reads_the_environment(self, tmp_path):
        """The eval harness (evals/skill_bench/infra/mirix_server.py) imports
        mirix.settings in a fresh interpreter under the cell's environment and
        refuses to start MIRIX unless settings.llm_max_retries /
        settings.llm_timeout_seconds equal MIRIX_LLM_MAX_RETRIES /
        MIRIX_LLM_TIMEOUT_SECONDS."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        code = (
            "import json\n"
            "from mirix.settings import settings\n"
            "print('PROBE' + json.dumps({name: getattr(settings, name) "
            "for name in ('llm_max_retries', 'llm_timeout_seconds')}))\n"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("_API_KEY") and key != "MIRIX_PG_URI"
        }
        env.update(
            HOME=str(tmp_path),
            MIRIX_LLM_MAX_RETRIES="0",
            MIRIX_LLM_TIMEOUT_SECONDS="660",
            PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        [line] = [line for line in out.splitlines() if line.startswith("PROBE")]
        assert json.loads(line[len("PROBE") :]) == {
            "llm_max_retries": 0,
            "llm_timeout_seconds": 660.0,
        }

    async def test_a_5xx_is_retried_by_the_transport(
        self, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 2)
        fake_llm_server.inject_fault(Fault.rate_limit(retry_after=0))
        fake_llm_server.queue_text("after the retry")
        response = await LLMClient.create(
            _config(fake_llm_server.base_url)
        ).send_llm_request(messages=_messages())
        assert response.choices[0].message.content == "after the retry"
        assert [r.outcome for r in fake_llm_server.chat_requests()] == [
            "fault:429",
            "scripted",
        ]

    async def test_zero_retries_means_one_attempt(self, fake_llm_server, monkeypatch):
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        fake_llm_server.inject_fault(Fault.http_error(503))
        with pytest.raises(LLMServerError):
            await LLMClient.create(_config(fake_llm_server.base_url)).send_llm_request(
                messages=_messages()
            )
        assert len(fake_llm_server.chat_requests()) == 1

    async def test_a_timed_out_call_is_an_llm_timeout(
        self, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 0.3)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        fake_llm_server.inject_fault(Fault.slow(3))
        fake_llm_server.queue_text("too late")
        with pytest.raises(LLMTimeoutError):
            await LLMClient.create(_config(fake_llm_server.base_url)).send_llm_request(
                messages=_messages()
            )

    async def test_a_trickling_answer_is_cut_at_the_wall_clock_deadline(
        self, fake_llm_server, monkeypatch
    ):
        # The HTTP timeouts bound each read, not the exchange. A provider
        # that keeps trickling bytes never trips them; the per-call wall-clock
        # deadline (MIRIX_LLM_TIMEOUT_SECONDS) does — retries included.
        import time

        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 0.6)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 2)
        fake_llm_server.inject_fault(Fault.trickling(5.0, interval=0.05), times=3)
        started = time.monotonic()
        with pytest.raises(LLMTimeoutError, match="deadline"):
            await LLMClient.create(_config(fake_llm_server.base_url)).send_llm_request(
                messages=_messages()
            )
        assert time.monotonic() - started < 2.0  # not 3 x 5 s
        assert len(fake_llm_server.chat_requests()) == 1

    async def test_transport_retries_are_recorded_with_the_stage_recorder(
        self, fake_llm_server, monkeypatch
    ):
        # The SDK retries inside one call; each retry is recorded
        # with the enclosing recording_retries block's recorder.
        from mirix.services.retry_policy import RetryEventRecorder, recording_retries

        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 2)
        fake_llm_server.inject_fault(Fault.http_error(503))
        fake_llm_server.inject_fault(Fault.rate_limit(retry_after=0))
        fake_llm_server.queue_text("third time lucky")
        recorder = RetryEventRecorder()
        client = LLMClient.create(_config(fake_llm_server.base_url))
        with recording_retries(recorder, stage="curate", session_id="s-1"):
            response = await client.send_llm_request(messages=_messages())
        assert response.choices[0].message.content == "third time lucky"
        assert [
            (e["kind"], e["attempt"], e["stage"], e["session_id"])
            for e in recorder.events
        ] == [("infra", 1, "transport", "s-1"), ("infra", 2, "transport", "s-1")]
        # Outside a block nothing is recorded (and nothing fails).
        fake_llm_server.inject_fault(Fault.http_error(503))
        fake_llm_server.queue_text("again")
        await client.send_llm_request(messages=_messages())
        assert len(recorder.events) == 2

    async def test_a_dropped_connection_stays_a_connection_error(
        self, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        fake_llm_server.inject_fault(Fault.drop_connection())
        with pytest.raises(LLMConnectionError) as exc:
            await LLMClient.create(_config(fake_llm_server.base_url)).send_llm_request(
                messages=_messages()
            )
        assert not isinstance(exc.value, LLMTimeoutError)

    @pytest.mark.parametrize("outcome", ["answer", "error"])
    async def test_every_request_closes_its_sdk_client(
        self, fake_llm_server, monkeypatch, outcome
    ):
        # A client left to the garbage collector closes its pooled sockets
        # behind the event loop's back; a later connection that reuses such a
        # descriptor can hang until the connect timeout (seen on Postgres
        # integration runs). Each request closes its client, success or not.
        import mirix.llm_api.openai_client as openai_module

        made = []
        real = openai_module.recording_sdk_class

        def spying(sdk_class):
            cls = real(sdk_class)

            class Spy(cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    made.append(self)

            return Spy

        monkeypatch.setattr(openai_module, "recording_sdk_class", spying)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        client = LLMClient.create(_config(fake_llm_server.base_url))
        if outcome == "answer":
            fake_llm_server.queue_text("done")
            await client.send_llm_request(messages=_messages())
        else:
            fake_llm_server.inject_fault(Fault.http_error(503))
            with pytest.raises(LLMError):
                await client.send_llm_request(messages=_messages())
        assert len(made) == 1 and made[0].is_closed()

    async def test_azure_gets_the_same_settings(self, monkeypatch):
        from mirix.llm_api.azure_openai_client import AzureOpenAIClient

        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 42.0)
        config = LLMConfig(
            model="gpt",
            model_endpoint_type="azure_openai",
            context_window=10_000,
            api_key="k",
            azure_endpoint="https://example.invalid",
            api_version="2024-10-01-preview",
        )
        kwargs = await AzureOpenAIClient(config)._prepare_client_kwargs()
        assert kwargs["timeout"].read == 42.0 and kwargs["max_retries"] == 2
        assert kwargs["timeout"].connect == 5.0


# --------------------------------------------------------------------------- #
# MIRIX's own failures around a call                                           #
# --------------------------------------------------------------------------- #


class TestRequestBuildErrors:
    """A failure before anything was sent is MIRIX's, never the model's: it
    must not look like an unusable answer (ValueError / KeyError /
    AssertionError, which a strict agent step reports as PARSE) nor like a
    provider error (LLMError)."""

    @pytest.mark.parametrize(
        "bug", [KeyError("tool_rules"), AssertionError("x"), ValueError("state")]
    )
    async def test_a_build_failure_is_raised_as_a_request_build_error(
        self, fake_llm_server, monkeypatch, bug
    ):
        from mirix.errors import LLMRequestBuildError

        client = LLMClient.create(_config(fake_llm_server.base_url))

        async def broken(*args, **kwargs):
            raise bug

        monkeypatch.setattr(client, "build_request_data", broken)
        with pytest.raises(LLMRequestBuildError) as raised:
            await client.send_llm_request(messages=_messages())
        assert raised.value.stage == "build"
        assert raised.value.__cause__ is bug
        assert not isinstance(
            raised.value, (LLMError, ValueError, KeyError, AssertionError)
        )
        # The retry policy's verdict is in the text (what a learning run records).
        assert raised.value.kind == "internal"
        assert str(raised.value).startswith(
            f"building the {type(client).__name__} request failed [kind=internal] "
            f"({type(bug).__name__}: "
        )
        assert fake_llm_server.requests == []

    async def test_a_network_failure_deeper_in_the_build_is_recorded_as_infra(
        self, fake_llm_server, monkeypatch
    ):
        """What building the request met on the way (a download whose connection
        reset, re-raised by the code that fetched it) is recorded with the retry
        policy's verdict on the whole chain: the text names only the outer error."""
        from mirix.errors import LLMRequestBuildError
        from mirix.services.retry_policy import ErrorKind, classify_error

        client = LLMClient.create(_config(fake_llm_server.base_url))

        async def fetch_fails(*args, **kwargs):
            try:
                raise ConnectionResetError(54, "Connection reset by peer")
            except ConnectionResetError as exc:
                raise RuntimeError("fetching the image failed") from exc

        monkeypatch.setattr(client, "build_request_data", fetch_fails)
        with pytest.raises(LLMRequestBuildError) as raised:
            await client.send_llm_request(messages=_messages())
        assert classify_error(raised.value) is ErrorKind.INFRA
        assert raised.value.kind == "infra"
        assert "[kind=infra] (RuntimeError: fetching the image failed)" in str(
            raised.value
        )
        assert "ConnectionResetError" not in str(raised.value)
        assert fake_llm_server.requests == []

    async def test_mirix_errors_raised_while_building_keep_their_kind(
        self, fake_llm_server, monkeypatch
    ):
        from mirix.errors import MirixConfigurationError

        client = LLMClient.create(_config(fake_llm_server.base_url))
        for error in (
            ContextBudgetExceeded("too big"),
            MirixConfigurationError("no key"),
            LLMBadRequestError("image refused"),
        ):

            async def raising(*args, error=error, **kwargs):
                raise error

            monkeypatch.setattr(client, "build_request_data", raising)
            with pytest.raises(type(error)) as raised:
                await client.send_llm_request(messages=_messages())
            assert raised.value is error

    async def test_a_programming_error_calling_the_sdk_is_mirixs(
        self, fake_llm_server, monkeypatch
    ):
        from mirix.errors import LLMRequestBuildError

        client = LLMClient.create(_config(fake_llm_server.base_url))

        async def refuse(request_data):
            raise TypeError("create() got an unexpected keyword argument 'temperature'")

        monkeypatch.setattr(client, "request", refuse)
        with pytest.raises(LLMRequestBuildError) as raised:
            await client.send_llm_request(messages=_messages())
        assert raised.value.stage == "sdk_call"
        assert isinstance(raised.value.__cause__, TypeError)
        assert raised.value.kind == "internal"
        assert str(raised.value).startswith(
            f"calling the {type(client).__name__} SDK failed [kind=internal] (TypeError: "
        )

    async def test_a_provider_failure_is_still_an_llm_error(
        self, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        fake_llm_server.inject_fault(Fault.http_error(400, "bad tool schema"))
        with pytest.raises(LLMBadRequestError):
            await LLMClient.create(_config(fake_llm_server.base_url)).send_llm_request(
                messages=_messages()
            )


# --------------------------------------------------------------------------- #
# Anthropic                                                                    #
# --------------------------------------------------------------------------- #


class TestAnthropic:
    def _client(self, **overrides):
        from mirix.llm_api.anthropic_client import AnthropicClient

        values = dict(
            model="claude-x",
            model_endpoint_type="anthropic",
            context_window=200_000,
            max_output_tokens=64_000,
            put_inner_thoughts_in_kwargs=True,
        )
        values.update(overrides)
        return AnthropicClient(LLMConfig(**values))

    async def test_max_tokens_is_the_budgeted_cap(self):
        client = self._client()
        data = await client.build_request_data(
            _messages(), client.llm_config, tools=TOOLS
        )
        counted = {k: v for k, v in data.items() if k != "max_tokens"}
        estimated = get_token_counter().count_request(counted)
        assert data["max_tokens"] == min(64_000 * 4 // 5, 160_000 - estimated)
        assert data["max_tokens"] == 51_200

    async def test_oversize_input_raises_before_the_request(self, monkeypatch):
        client = self._client(context_window=4_000, max_output_tokens=4_000)

        async def never(request_data):
            raise AssertionError("no request may be sent")

        monkeypatch.setattr(client, "request", never)
        with pytest.raises(ContextBudgetExceeded):
            await client.send_llm_request(messages=_messages("y" * 12_000))

    async def test_thinking_fits_under_a_large_cap_unchanged(self):
        client = self._client(enable_reasoner=True, max_reasoning_tokens=8_000)
        data = await client.build_request_data(
            _messages(), client.llm_config, tools=TOOLS
        )
        assert data["max_tokens"] == 51_200
        assert data["thinking"]["budget_tokens"] == 8_000

    async def test_thinking_shrinks_when_the_cap_is_small(self):
        # Anthropic requires budget_tokens < max_tokens. A large input
        # leaves a small cap; the thinking budget shrinks so the answer keeps
        # at least MIN_OUTPUT_TOKENS.
        client = self._client(
            context_window=20_000,
            max_output_tokens=16_000,
            enable_reasoner=True,
            max_reasoning_tokens=12_000,
        )
        data = await client.build_request_data(
            _messages("z" * 36_000), client.llm_config, tools=TOOLS
        )
        cap = data["max_tokens"]
        assert cap < 12_000 + MIN_OUTPUT_TOKENS  # the configured budget cannot fit
        assert data["thinking"]["budget_tokens"] == cap - MIN_OUTPUT_TOKENS
        assert 1024 <= data["thinking"]["budget_tokens"] < cap

    def test_a_cap_too_small_for_thinking_and_an_answer_raises(self):
        from mirix.llm_api.anthropic_client import fit_thinking_budget

        data = {
            "max_tokens": 2_000,
            "thinking": {"type": "enabled", "budget_tokens": 4_000},
        }
        with pytest.raises(ContextBudgetExceeded):
            fit_thinking_budget(data)
        fit_thinking_budget(no_thinking := {"max_tokens": 1_500})
        assert no_thinking == {"max_tokens": 1_500}
        edge = {
            "max_tokens": 2_048,
            "thinking": {"type": "enabled", "budget_tokens": 9},
        }
        fit_thinking_budget(edge)  # 9 + 1024 <= 2048: nothing to do
        assert edge["thinking"]["budget_tokens"] == 9
        tight = {
            "max_tokens": 2_048,
            "thinking": {"type": "enabled", "budget_tokens": 4_000},
        }
        fit_thinking_budget(tight)
        assert tight["thinking"]["budget_tokens"] == 1_024

    async def test_the_request_closes_its_sdk_client(self, monkeypatch):
        client = self._client()
        events = []

        class FakeSDK:
            def __init__(self):
                self.beta = SimpleNamespace(
                    messages=SimpleNamespace(create=self.create)
                )

            async def create(self, **kwargs):
                events.append("create")
                return SimpleNamespace(model_dump=lambda: {"id": "msg"})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                events.append("closed")

        async def fake_sdk(async_client=False):
            return FakeSDK()

        monkeypatch.setattr(client, "_get_anthropic_client", fake_sdk)
        assert await client.request({"model": "claude-x"}) == {"id": "msg"}
        assert events == ["create", "closed"]

    async def test_sdk_client_gets_timeout_and_retries(self, monkeypatch):
        from mirix.services.provider_manager import ProviderManager

        async def key(self):
            return "sk-ant-test"

        monkeypatch.setattr(ProviderManager, "get_anthropic_override_key", key)
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 33.0)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 1)
        sdk = await self._client()._get_anthropic_client(async_client=True)
        assert sdk.timeout.read == 33.0 and sdk.timeout.connect == 5.0
        assert sdk.max_retries == 1
        assert sdk.api_key == "sk-ant-test"

    async def test_timeouts_map_to_llm_timeout(self):
        import anthropic

        client = self._client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        try:
            raise anthropic.APITimeoutError(request=request) from httpx.ReadTimeout(
                "slow", request=request
            )
        except anthropic.APITimeoutError as exc:
            assert isinstance(client.handle_llm_error(exc), LLMTimeoutError)
        try:
            raise anthropic.APITimeoutError(request=request) from httpx.ConnectTimeout(
                "no route", request=request
            )
        except anthropic.APITimeoutError as exc:
            mapped = client.handle_llm_error(exc)
            assert isinstance(mapped, LLMConnectionError)
            assert not isinstance(mapped, LLMTimeoutError)


# --------------------------------------------------------------------------- #
# Google                                                                       #
# --------------------------------------------------------------------------- #


class TestGoogle:
    def _client(self, **overrides):
        from mirix.llm_api.google_ai_client import GoogleAIClient

        values = dict(
            model="gemini-2.0-flash",
            model_endpoint_type="google_ai",
            model_endpoint="https://generativelanguage.example",
            context_window=1_000_000,
            max_output_tokens=65_536,
        )
        values.update(overrides)
        return GoogleAIClient(LLMConfig(**values))

    async def test_max_output_tokens_is_the_budgeted_cap(self):
        client = self._client()
        data = await client.build_request_data(
            _messages(), client.llm_config, tools=TOOLS
        )
        counted = {k: v for k, v in data.items() if k != "generation_config"}
        estimated = get_token_counter().count_request(counted)
        cap = data["generation_config"]["max_output_tokens"]
        assert cap == min(65_536 * 4 // 5, 800_000 - estimated) == 52_428

    async def test_oversize_input_raises(self):
        client = self._client(context_window=4_000, max_output_tokens=4_000)
        with pytest.raises(ContextBudgetExceeded):
            await client.build_request_data(
                _messages("z" * 12_000), client.llm_config, tools=None
            )

    async def test_request_uses_the_transport_settings(self, monkeypatch):
        from mirix.llm_api import google_ai_client
        from mirix.services.provider_manager import ProviderManager

        async def key(self):
            return "g-key"

        seen = {}

        async def fake_post(url, headers, data, *, timeout, max_retries):
            seen.update(url=url, headers=headers, timeout=timeout, retries=max_retries)
            return {"ok": True}

        monkeypatch.setattr(ProviderManager, "get_gemini_override_key", key)
        monkeypatch.setattr(google_ai_client, "post_json_with_retries", fake_post)
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 99.0)
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 3)
        assert await self._client().request({"contents": []}) == {"ok": True}
        assert seen["timeout"].read == 99.0 and seen["timeout"].connect == 5.0
        assert seen["retries"] == 3
        assert seen["headers"]["x-goog-api-key"] == "g-key"

    @pytest.mark.parametrize(
        "status, expected",
        [
            (429, LLMRateLimitError),
            (500, LLMServerError),
            (503, LLMServerError),
            (408, LLMServerError),
            (401, LLMAuthenticationError),
            (403, LLMPermissionDeniedError),
            (404, LLMNotFoundError),
            (400, LLMBadRequestError),
        ],
    )
    async def test_http_errors_map_to_llm_errors(self, status, expected):
        request = httpx.Request(
            "POST", "https://g.example/v1beta/models/m:generateContent"
        )
        response = httpx.Response(status, request=request)
        error = httpx.HTTPStatusError("boom", request=request, response=response)
        assert isinstance(self._client().handle_llm_error(error), expected)

    async def test_transport_errors_map_to_llm_errors(self):
        request = httpx.Request("POST", "https://g.example")
        client = self._client()
        assert isinstance(
            client.handle_llm_error(httpx.ReadTimeout("slow", request=request)),
            LLMTimeoutError,
        )
        connect = client.handle_llm_error(httpx.ConnectTimeout("x", request=request))
        assert isinstance(connect, LLMConnectionError)
        assert not isinstance(connect, LLMTimeoutError)
        assert isinstance(
            client.handle_llm_error(httpx.ConnectError("refused", request=request)),
            LLMConnectionError,
        )


class TestPostJsonWithRetries:
    def _transport(self, answers, seen):
        def handler(request):
            seen.append(json.loads(request.content))
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        return httpx.MockTransport(handler)

    async def test_retries_5xx_and_429_honoring_retry_after(self):
        seen, sleeps = [], []

        async def sleep(seconds):
            sleeps.append(seconds)

        from mirix.services.retry_policy import RetryEventRecorder, recording_retries

        answers = [
            httpx.Response(503),
            httpx.Response(429, headers={"retry-after": "2"}),
            httpx.Response(200, json={"candidates": []}),
        ]
        recorder = RetryEventRecorder()
        with recording_retries(recorder, stage="distill"):
            body = await helpers.post_json_with_retries(
                "https://g.example",
                {},
                {"q": 1},
                timeout=5,
                max_retries=2,
                transport=self._transport(answers, seen),
                sleep=sleep,
            )
        assert body == {"candidates": []}
        assert len(seen) == 3
        assert 0.375 <= sleeps[0] <= 0.5  # 0.5 s with jitter
        assert sleeps[1] == 2.0  # the server's Retry-After
        # Every transport retry is recorded, with the wait it took.
        assert [(e["kind"], e["attempt"], e["stage"]) for e in recorder.events] == [
            ("infra", 1, "transport"),
            ("infra", 2, "transport"),
        ]
        assert recorder.events[1]["waited_s"] == 2.0

    async def test_connection_errors_are_retried(self):
        seen, sleeps = [], []

        async def sleep(seconds):
            sleeps.append(seconds)

        request = httpx.Request("POST", "https://g.example")
        answers = [
            httpx.ConnectError("refused", request=request),
            httpx.Response(200, json={"ok": 1}),
        ]
        body = await helpers.post_json_with_retries(
            "https://g.example",
            {},
            {},
            timeout=5,
            max_retries=1,
            transport=self._transport(answers, seen),
            sleep=sleep,
        )
        assert body == {"ok": 1} and len(sleeps) == 1

    async def test_retries_are_bounded(self):
        seen = []

        async def sleep(seconds):
            pass

        answers = [httpx.Response(500, text="down")] * 3
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await helpers.post_json_with_retries(
                "https://g.example?key=secret",
                {},
                {},
                timeout=5,
                max_retries=2,
                transport=self._transport(list(answers), seen),
                sleep=sleep,
            )
        assert len(seen) == 3
        assert "Status code: 500" in str(exc.value) and "down" in str(exc.value)
        assert "secret" not in str(exc.value)  # the query (API key) is not echoed

    async def test_client_errors_are_not_retried(self):
        seen = []
        with pytest.raises(httpx.HTTPStatusError):
            await helpers.post_json_with_retries(
                "https://g.example",
                {},
                {},
                timeout=5,
                max_retries=5,
                transport=self._transport([httpx.Response(400)], seen),
            )
        assert len(seen) == 1

    async def test_zero_retries(self):
        seen = []
        with pytest.raises(httpx.HTTPStatusError):
            await helpers.post_json_with_retries(
                "https://g.example",
                {},
                {},
                timeout=5,
                max_retries=0,
                transport=self._transport([httpx.Response(503)], seen),
            )
        assert len(seen) == 1

    @pytest.mark.parametrize(
        "header, expected",
        [("3", 3.0), ("0", 0.0), ("600", None), ("soon", None), (None, None)],
    )
    async def test_retry_after_parsing(self, header, expected):
        assert helpers._retry_after_seconds(header) == expected


# --------------------------------------------------------------------------- #
# OpenRouter                                                                   #
# --------------------------------------------------------------------------- #


class TestOpenRouter:
    def _config(self, base_url, **overrides):
        values = dict(
            model="example/chat-model",
            model_endpoint_type="openrouter",
            model_endpoint=base_url,
            context_window=1_310_720,
            max_output_tokens=943_717,
        )
        values.update(overrides)
        return LLMConfig(**values)

    async def test_openrouter_is_the_openai_compatible_client(self):
        assert type(LLMClient.create(self._config(None))) is OpenAIClient

    async def test_a_call_goes_through_with_openrouters_key(
        self, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(model_settings, "openrouter_api_key", "sk-or-test")
        monkeypatch.setattr(model_settings, "openai_api_key", "sk-openai-never")
        fake_llm_server.queue_text("hi from openrouter")
        client = LLMClient.create(self._config(fake_llm_server.base_url))

        response = await client.send_llm_request(messages=_messages())

        assert response.choices[0].message.content == "hi from openrouter"
        request = fake_llm_server.last_chat_request()
        assert request.headers["authorization"] == "Bearer sk-or-test"
        assert request.json["model"] == "example/chat-model"
        assert request.json["max_completion_tokens"] == 754_973  # floor(943,717 * 0.8)

    async def test_the_openai_key_is_never_sent_to_openrouter(self, monkeypatch):
        monkeypatch.setattr(model_settings, "openrouter_api_key", None)
        monkeypatch.setattr(model_settings, "openai_api_key", "sk-openai-never")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-never")
        kwargs = await OpenAIClient(self._config(None))._prepare_client_kwargs()
        assert kwargs["api_key"] == "DUMMY_API_KEY"
        assert kwargs["base_url"] == OPENROUTER_DEFAULT_BASE_URL

    async def test_the_config_key_wins(self, monkeypatch):
        monkeypatch.setattr(model_settings, "openrouter_api_key", "sk-or-env")
        kwargs = await OpenAIClient(
            self._config("https://gateway.example/v1", api_key="sk-cfg")
        )._prepare_client_kwargs()
        assert kwargs["api_key"] == "sk-cfg"
        assert kwargs["base_url"] == "https://gateway.example/v1"

    async def test_openrouter_embeddings(self, fake_llm_server, monkeypatch):
        from mirix.embeddings import embedding_model
        from mirix.schemas.embedding_config import EmbeddingConfig

        monkeypatch.setattr(model_settings, "openrouter_api_key", "sk-or-embed")
        config = EmbeddingConfig(
            embedding_endpoint_type="openrouter",
            embedding_endpoint=fake_llm_server.base_url,
            embedding_model="google/gemini-embedding-001",
            embedding_dim=3072,
        )
        model = await embedding_model(config)

        vector = await model.get_text_embedding("put a mug in the sink")

        # The provider's 3,072 dimensions, zero-padded to MAX_EMBEDDING_DIM by
        # the length guard every factory model sits behind.
        expected = fake_llm_server.embed("put a mug in the sink")
        assert len(vector) == constants.MAX_EMBEDDING_DIM
        assert vector[: len(expected)] == pytest.approx(expected)
        assert not any(vector[len(expected) :])
        [request] = fake_llm_server.embedding_requests()
        assert request.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer sk-or-embed"
        assert request.json["model"] == "google/gemini-embedding-001"

    async def test_openrouter_embedding_defaults(self, monkeypatch):
        from mirix.embeddings import embedding_model
        from mirix.schemas.embedding_config import EmbeddingConfig

        monkeypatch.setattr(model_settings, "openrouter_api_key", "sk-or-env")
        monkeypatch.setattr(model_settings, "openai_api_key", "sk-openai")
        model = await embedding_model(
            EmbeddingConfig(
                embedding_endpoint_type="openrouter",
                embedding_model="m",
                embedding_dim=8,
                api_key=None,
            )
        )
        assert model._base_url == OPENROUTER_DEFAULT_BASE_URL
        assert model._api_key == "sk-or-env"

    async def test_the_openai_key_never_goes_to_openrouter_by_default(
        self, monkeypatch
    ):
        # With no OpenRouter key, the default endpoint (openrouter.ai)
        # is refused — a configuration error, never the OpenAI key.
        from mirix.embeddings import embedding_model
        from mirix.errors import MirixConfigurationError
        from mirix.schemas.embedding_config import EmbeddingConfig
        from mirix.services.retry_policy import ErrorKind, classify_error

        monkeypatch.setattr(model_settings, "openrouter_api_key", None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(model_settings, "openai_api_key", "sk-openai")
        config = EmbeddingConfig(
            embedding_endpoint_type="openrouter",
            embedding_model="m",
            embedding_dim=8,
            api_key=None,
        )
        with pytest.raises(MirixConfigurationError) as refused:
            await embedding_model(config)
        assert "sk-openai" not in str(refused.value)
        assert classify_error(refused.value, operation="embedding") is ErrorKind.CONFIG

        # An endpoint the operator set explicitly may take the OpenAI key
        # (the historical setup: an OpenRouter key stored as OPENAI_API_KEY).
        config.embedding_endpoint = "https://gateway.example/v1"
        model = await embedding_model(config)
        assert model._api_key == "sk-openai"
        assert model._base_url == "https://gateway.example/v1"
