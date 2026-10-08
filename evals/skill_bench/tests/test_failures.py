"""Failure taxonomy and H8 recovery rules, including real gateway error bodies."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.infra import failures as f
from evals.skill_bench.infra.failures import Failure, FailureKind, Phase, Recovery
from evals.skill_bench.infra.mirix_server import MirixServerError
from evals.skill_bench.infra.processes import OwnedProcessError
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, running_gateway


def chat(model: str = "openai/gpt-6-luna") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


async def gateway_error(
    tmp_path, fake: FakeUpstream, role: str = "actor", body: dict | None = None, **cfg
):
    async with running_gateway(tmp_path, fake, **cfg) as (_, client):
        endpoint = "embeddings" if role == "embedding" else "chat/completions"
        response = await client.post(
            f"/runs/f/{role}/v1/{endpoint}", json=body or chat()
        )
    return response.status_code, response.json()


@pytest.mark.asyncio
async def test_actor_call_that_alone_hits_the_deadline_is_a_model_failure(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(delay_s=3))
    status, body = await gateway_error(tmp_path, fake, deadline_s=1.0)
    error = body["error"]
    assert error["reason"] == "deadline_during_attempt" and error["retries"] == 0
    assert error["attempt_budget_s"] >= 1.0 - f.model_timeout_allowance(1.0)
    assert error["queue_wait_s"] < 0.25 and error["pre_attempt_s"] < 0.25
    failure = f.classify_gateway_error(status, body, role="actor")
    assert (failure.kind, failure.code) == (FailureKind.MODEL, f.MODEL_TIMEOUT)
    assert f.recovery_for(Phase.LEARNING, failure) is Recovery.SCORE_AND_CONTINUE
    learner = f.classify_gateway_error(status, body, role="learner")
    assert learner.kind is FailureKind.MODEL
    assert f.recovery_for(Phase.LEARNING, learner) is Recovery.COUNT_AND_CONTINUE
    embedding = f.classify_gateway_error(status, body, role="embedding")
    assert embedding.kind is FailureKind.INFRA


@pytest.mark.asyncio
async def test_deadline_spent_in_the_gateway_queue_is_infra_not_the_models(tmp_path):
    """W7/W26: call B queues behind A for one slot, then its only attempt times out."""
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(delay_s=1.5), Reply(delay_s=5))
    async with running_gateway(tmp_path, fake, deadline_s=2.0, max_concurrency=1) as (
        _,
        client,
    ):
        first = asyncio.ensure_future(
            client.post("/runs/q/actor/v1/chat/completions", json=chat())
        )
        await asyncio.sleep(0.2)  # A holds the only slot
        second = await client.post("/runs/q/actor/v1/chat/completions", json=chat())
        assert (await first).status_code == 200
    error = second.json()["error"]
    assert second.status_code == 504 and error["reason"] == "deadline_during_attempt"
    assert error["retries"] == 0 and error["upstream_attempts"] == 1
    assert error["queue_wait_s"] > 1.0 and error["attempt_budget_s"] < 1.0
    failure = f.classify_gateway_error(504, second.json(), role="actor")
    assert (failure.kind, failure.code) == (FailureKind.INFRA, f.DEADLINE_AFTER_QUEUE)
    assert not f.is_scored_as_model_failure(failure)
    assert f.recovery_for(Phase.TEST, failure) is Recovery.RERUN_ITEM


@pytest.mark.asyncio
async def test_deadline_spent_on_a_stalled_balance_check_is_infra(
    tmp_path, monkeypatch
):
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(delay_s=5))
    async with running_gateway(tmp_path, fake, deadline_s=2.0) as (gateway, client):
        original = gateway.refresh_balance

        async def stalled(force: bool = False) -> None:
            await asyncio.sleep(1.0)  # /key and /credits hanging
            await original(force)

        monkeypatch.setattr(gateway, "refresh_balance", stalled)
        response = await client.post("/runs/b/actor/v1/chat/completions", json=chat())
    error = response.json()["error"]
    assert response.status_code == 504 and error["pre_attempt_s"] >= 1.0
    failure = f.classify_gateway_error(504, response.json(), role="actor")
    assert (failure.kind, failure.code) == (FailureKind.INFRA, f.DEADLINE_AFTER_QUEUE)


def test_deadline_blame_needs_the_attempt_budget():
    def body(**extra):
        return {
            "error": {
                "source": "gateway",
                "type": "deadline_exceeded",
                "reason": "deadline_during_attempt",
                "retries": 0,
                "upstream_attempts": 1,
                "deadline_s": 600.0,
                **extra,
            }
        }

    blame = lambda b: f.classify_gateway_error(504, b, "actor").code  # noqa: E731
    assert blame(body(attempt_budget_s=599.2)) == f.MODEL_TIMEOUT
    assert blame(body(attempt_budget_s=595.0)) == f.MODEL_TIMEOUT  # 5 s allowance
    assert blame(body(attempt_budget_s=594.9)) == f.DEADLINE_AFTER_QUEUE
    assert blame(body(attempt_budget_s=200.0)) == f.DEADLINE_AFTER_QUEUE
    assert blame(body()) == f.DEADLINE_AFTER_QUEUE  # no evidence: never the model
    assert blame(body(attempt_budget_s="x")) == f.DEADLINE_AFTER_QUEUE
    assert f.model_timeout_allowance(600.0) == 5.0
    assert f.model_timeout_allowance(2.0) == 0.5


@pytest.mark.asyncio
async def test_upstream_504s_after_retries_keep_their_own_code(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(504) for _ in range(10)])
    status, body = await gateway_error(tmp_path, fake, backoff_base_s=0.001)
    assert status == 504 and body["error"]["type"] == "upstream_infra_failure"
    failure = f.classify_gateway_error(status, body, "actor")
    assert (failure.kind, failure.code) == (FailureKind.INFRA, f.UPSTREAM_INFRA)
    # A bare 504 (no gateway body, e.g. a proxy) is infrastructure too.
    assert f.classify_gateway_error(504, None).code == f.UPSTREAM_INFRA


@pytest.mark.asyncio
async def test_deadline_spent_on_retries_is_infra(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(503) for _ in range(10)])
    status, body = await gateway_error(
        tmp_path, fake, deadline_s=0.5, backoff_base_s=0.2
    )
    failure = f.classify_gateway_error(status, body, role="actor")
    assert (failure.kind, failure.code) == (FailureKind.INFRA, f.DEADLINE_AFTER_RETRIES)
    assert failure.retries >= 1
    assert f.recovery_for(Phase.LEARNING, failure) is Recovery.RESTART_CELL
    assert f.recovery_for(Phase.TEST, failure) is Recovery.RERUN_ITEM
    assert not f.is_scored_as_model_failure(failure)


@pytest.mark.asyncio
async def test_retries_exhausted_and_budget_pause_are_infra(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(502) for _ in range(10)])
    status, body = await gateway_error(tmp_path, fake, backoff_base_s=0.001)
    assert f.classify_gateway_error(status, body).code == f.UPSTREAM_INFRA
    status, body = await gateway_error(
        tmp_path / "b", FakeUpstream(limit_remaining=1.0), balance_floor_usd=5.0
    )
    assert status == 402
    assert f.classify_gateway_error(status, body).kind is FailureKind.INFRA


@pytest.mark.asyncio
async def test_gateway_refusals_are_harness_failures(tmp_path, monkeypatch):
    status, body = await gateway_error(
        tmp_path, FakeUpstream(), body=chat("openai/gpt-4o")
    )
    assert status == 403
    assert f.classify_gateway_error(status, body).kind is FailureKind.HARNESS
    from evals.skill_bench.infra import tokens

    monkeypatch.setattr(tokens, "count_text", lambda text: 10**7)
    status, body = await gateway_error(tmp_path / "b", FakeUpstream())
    assert status == 400 and body["error"]["type"] == "context_budget_exceeded"
    failure = f.classify_gateway_error(status, body, role="actor")
    assert (failure.kind, failure.code) == (FailureKind.HARNESS, f.GATEWAY_REJECTED)
    assert f.recovery_for(Phase.TEST, failure) is Recovery.RERUN_ITEM


@pytest.mark.asyncio
async def test_upstream_rejections(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(400, {"error": {"code": 400, "message": "invalid tool schema"}}),
    )
    status, body = await gateway_error(tmp_path, fake)
    assert f.classify_gateway_error(status, body).code == f.UPSTREAM_REJECTED
    moderated = {
        "error": {
            "source": "upstream",
            "code": 403,
            "message": "Input flagged by moderation",
        }
    }
    assert f.classify_gateway_error(403, moderated, "actor").code == f.CONTENT_FILTER
    assert (
        f.classify_gateway_error(401, {"error": {"source": "upstream"}}).kind
        is FailureKind.INFRA
    )


def test_classify_completion():
    ok = {"choices": [{"finish_reason": "stop"}]}
    assert f.classify_completion(ok) is None
    truncated = f.classify_completion(
        {"choices": [{"finish_reason": "length"}]}, role="actor"
    )
    assert truncated is not None and (truncated.kind, truncated.code) == (
        FailureKind.MODEL,
        f.TRUNCATED,
    )


def test_classify_exception():
    import asyncpg

    request = httpx.Request("POST", "http://127.0.0.1:1")
    cases = {
        httpx.ConnectError("refused", request=request): (
            FailureKind.INFRA,
            f.GATEWAY_UNREACHABLE,
        ),
        httpx.ReadTimeout("slow", request=request): (
            FailureKind.INFRA,
            f.GATEWAY_CLIENT_TIMEOUT,
        ),
        MirixServerError("died"): (FailureKind.INFRA, f.MIRIX_SERVER),
        OwnedProcessError("cannot clean up"): (FailureKind.INFRA, f.OWNED_PROCESS),
        asyncpg.exceptions.ConnectionDoesNotExistError("gone"): (
            FailureKind.INFRA,
            f.DATABASE,
        ),
        asyncio.TimeoutError(): (FailureKind.INFRA, f.IDLE_WAIT_TIMEOUT),
        protocol.ProtocolDriftError("drift"): (FailureKind.HARNESS, f.PROTOCOL_DRIFT),
        KeyError("bug"): (FailureKind.HARNESS, f.UNEXPECTED_EXCEPTION),
    }
    for exc, expected in cases.items():
        failure = f.classify_exception(exc, role="actor")
        assert (failure.kind, failure.code) == expected, type(exc).__name__
    wrapped = Failure(FailureKind.MODEL, f.FORMAT_ERROR, "actor")
    assert f.classify_exception(f.GatewayCallError(wrapped)) is wrapped
    assert f.classify_exception(f.LearningAttemptInvalid(wrapped)) is wrapped


@pytest.mark.parametrize(
    ("phase", "kind", "role", "expected"),
    [
        (Phase.LEARNING, FailureKind.INFRA, "actor", Recovery.RESTART_CELL),
        (Phase.LEARNING, FailureKind.HARNESS, "learner", Recovery.RESTART_CELL),
        (Phase.LEARNING, FailureKind.INFRA, "mirix", Recovery.RESTART_CELL),
        (Phase.TEST, FailureKind.INFRA, "actor", Recovery.RERUN_ITEM),
        (Phase.TEST, FailureKind.HARNESS, None, Recovery.RERUN_ITEM),
        (Phase.TEST, FailureKind.MODEL, "actor", Recovery.SCORE_AND_CONTINUE),
        (Phase.LEARNING, FailureKind.MODEL, "actor", Recovery.SCORE_AND_CONTINUE),
        (Phase.LEARNING, FailureKind.MODEL, "judge", Recovery.COUNT_AND_CONTINUE),
    ],
)
def test_recovery_rules(phase, kind, role, expected):
    assert f.recovery_for(phase, Failure(kind, "x", role)) is expected
    assert f.recovery_for(phase.value, Failure(kind, "x", role)) is expected


def test_failure_log_and_counters_round_trip(tmp_path):
    log = f.FailureLog(tmp_path / "cell" / "failures.jsonl")
    assert log.record(
        "learning",
        Failure(FailureKind.INFRA, f.UPSTREAM_INFRA, "actor", 503),
        item="t1",
    ) is (Recovery.RESTART_CELL)
    log.record(
        Phase.TEST,
        Failure(FailureKind.MODEL, f.FORMAT_ERROR, "actor"),
        item="t2",
        attempt=1,
    )
    log.record(
        Phase.TEST,
        Failure(FailureKind.MODEL, f.FORMAT_ERROR, "actor"),
        item="t3",
        attempt=1,
    )
    rows = log.rows()
    assert [r["recovery"] for r in rows] == [
        "restart_cell",
        "score_and_continue",
        "score_and_continue",
    ]
    counters = log.counters()
    assert counters.total(phase="test", kind=FailureKind.MODEL) == 2
    assert counters.total(phase=Phase.LEARNING, kind=FailureKind.INFRA) == 1
    assert counters.total() == 3
    data = counters.to_dict()
    assert data["totals"]["test"]["model"] == 2
    assert f.FailureCounters.from_dict(data).to_dict() == data
    assert Failure.from_dict(rows[0]) == Failure(
        FailureKind.INFRA, f.UPSTREAM_INFRA, "actor", 503
    )


def test_pending_test_items_resume_only_unscored():
    items = ["t1", "t2", "t3", "t4", "t5"]
    results = [
        {"raw_id": "t1", "score": {"primary": 1.0}},
        {"raw_id": "t2", "score": None, "failure": {"kind": "infra"}},
        {"raw_id": "t3", "score": {"primary": 0.0}, "failure": {"kind": "model"}},
        {"raw_id": "t4", "score": None},
        {"raw_id": "t4", "score": {"primary": 0.0}},  # a later successful re-run wins
        {"raw_id": "t1", "score": None},  # ... and so does a later failure
    ]
    assert f.pending_test_items(items, results) == ["t1", "t2", "t5"]


# --- X13: infrastructure failures a caller swallowed in the learning phase --------------------


def _line(call_id: str, *, status: str = "ok", http: int = 200, **kw: object) -> dict:
    row: dict = {
        "event": "call",
        "call_id": call_id,
        "role": "mirix",
        "run_phase": "learning",
        "phase": "learning",
        "status": status,
        "http_status": http,
    }
    if status != "ok":
        row["error"] = {"type": "upstream_timeout", "message": "504 from upstream"}
    row.update(kw)
    return row


def test_an_infra_failure_no_retry_recovered_is_reported():
    lines = [
        _line("a", status="error", http=504),  # MIRIX deferred the session: lost
        _line("b", status="error", http=504),
        _line("c", repeat_of="b"),  # M3.5 retry recovered b
        _line("d", status="error", http=504),
        _line("e", status="error", http=504, repeat_of="d"),
        _line("f", repeat_of="e"),  # a chain: f recovers e and d
        _line("g", status="error", http=400),  # not infrastructure (reported elsewhere)
        _line("h", status="error", http=504, role="actor"),  # another role
        _line("i", status="error", http=504, run_phase="test", phase="test"),
        {"event": "call_started", "call_id": "j", "role": "mirix"},
    ]
    lost = f.unrecovered_infra_calls(
        lines, roles=("mirix", "embedding"), phases=("setup", "learning")
    )
    assert [line["call_id"] for line in lost] == ["a"]
    assert f.ledger_line_failure(lines[0]).kind is FailureKind.INFRA
    assert f.ledger_line_failure(lines[2]) is None


def test_the_run_phase_decides_which_calls_are_learning_calls():
    """A line tagged 'learning' by its caller while the run was in its test phase is a
    test-phase line (X25); lines from before run_phase existed fall back to their tag."""
    tagged = _line("a", status="error", http=504, run_phase="test", phase="learning")
    old = _line("b", status="error", http=504)
    del old["run_phase"]
    assert f.line_phase(tagged) == "test" and f.line_phase(old) == "learning"
    lost = f.unrecovered_infra_calls(
        [tagged, old], roles=("mirix",), phases=("learning",)
    )
    assert [line["call_id"] for line in lost] == ["b"]


# --- upstream refusals: recorded, stopped only when they repeat ------------------------------


def _upstream(status: int, message: str) -> Reply:
    return Reply(status, {"error": {"code": status, "message": message}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (413, "Request entity too large", f.REFUSAL_PAYLOAD),
        (400, "Provider returned error", f.REFUSAL_PAYLOAD),
        # A 400 is a payload refusal only when it names the request's size or images, or
        # is OpenRouter's generic one above; any other 400 is a request the harness
        # built wrong and stays a stop.
        (400, "Request payload too large", f.REFUSAL_PAYLOAD),
        (400, "Image exceeds the maximum allowed size", f.REFUSAL_PAYLOAD),
        (
            400,
            "This endpoint's maximum context length is 131072 tokens. However, you "
            "requested about 140000 tokens. Please reduce the length of the messages.",
            f.REFUSAL_PAYLOAD,
        ),
        (400, "max_tokens: must be less than or equal to 131072", None),
        # A 400 naming the request's own parameters stops, even when it
        # also says "too large", "image" or "context length".
        (
            400,
            "max_tokens is too large: 200000. This model supports at most 131072 "
            "completion tokens, whereas you provided 200000.",
            None,
        ),
        (
            400,
            "This model's maximum context length is 131072 tokens. However, you "
            "requested 201000 tokens (1000 in the messages, 200000 in the completion).",
            None,
        ),
        (
            400,
            "This model's maximum context length is 131072 tokens. However, you "
            "requested 150000 tokens (140000 in the messages, 10000 in the completion).",
            f.REFUSAL_PAYLOAD,
        ),
        (400, "Invalid image URL: unsupported MIME type image/x-foo", None),
        (
            400,
            "Invalid 'messages[3]': tool_call_id not found in previous messages",
            None,
        ),
        (400, "Unsupported parameter: reasoning.effort", None),
        (400, "invalid tool schema", None),
        (413, "Request Entity Too Large", f.REFUSAL_PAYLOAD),
        (
            404,
            "No endpoints found that support tool use. Try disabling tools.",
            f.REFUSAL_NO_ENDPOINT,
        ),
        (404, "No endpoints found that support image input", f.REFUSAL_NO_ENDPOINT),
        (
            404,
            "No endpoints found matching your data policy (Free model publication).",
            None,
        ),
        (404, "Not Found", None),
        (403, "Forbidden", None),
    ],
)
async def test_upstream_refusals_by_status_and_message(
    tmp_path, status, message, expected
):
    """Through the real gateway: what reaches the harness is classified by its class."""
    fake = FakeUpstream()
    fake.queue("chat/completions", _upstream(status, message))
    got_status, body = await gateway_error(tmp_path, fake, backoff_base_s=0.001)
    assert got_status == status
    failure = f.classify_gateway_error(got_status, body, "actor")
    assert (failure.kind, failure.code) == (FailureKind.HARNESS, f.UPSTREAM_REJECTED)
    assert f.upstream_refusal(failure) == expected
    assert f.refusal_class(failure) == expected
    if expected is None:
        assert f.refusal_signature(failure) is None
    else:
        assert f.refusal_signature(failure) == f"actor/{expected}/{status}"


@pytest.mark.asyncio
async def test_gateway_refusals_and_infrastructure_are_not_upstream_refusals(
    tmp_path, monkeypatch
):
    from evals.skill_bench.infra import tokens

    monkeypatch.setattr(tokens, "count_text", lambda text: 10**7)
    status, body = await gateway_error(tmp_path, FakeUpstream())
    budget = f.classify_gateway_error(status, body, "mirix")
    assert (status, budget.code) == (400, f.GATEWAY_REJECTED)
    assert f.upstream_refusal(budget) is None
    assert f.refusal_class(budget) == f.REFUSAL_GATEWAY
    assert f.refusal_signature(budget) == "mirix/gateway/400"
    monkeypatch.undo()
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(502) for _ in range(10)])
    status, body = await gateway_error(tmp_path / "b", fake, backoff_base_s=0.001)
    exhausted = f.classify_gateway_error(status, body, "actor")
    assert exhausted.kind is FailureKind.INFRA
    assert f.upstream_refusal(exhausted) is None and f.refusal_class(exhausted) is None
    model = Failure(FailureKind.MODEL, f.MODEL_TIMEOUT, "actor", 504)
    assert f.refusal_class(model) is None


def test_no_endpoint_markers_match_the_gateway():
    from evals.skill_bench.infra import gateway

    assert f.NO_ENDPOINT_MARKERS == gateway.NO_ENDPOINT_PATTERNS
    for marker in f.ACCOUNT_REFUSAL_MARKERS:
        assert marker in gateway.NO_ENDPOINT_FINAL_PATTERNS
    assert set(f.REFUSAL_CLASSES) == {f.REFUSAL_PAYLOAD, f.REFUSAL_NO_ENDPOINT}


def test_protocol_rejection_types_match_the_audit():
    """The gateway refusals MIRIX's learning never counts past are exactly
    those the audit fails a cell for (H10.3)."""
    from evals.skill_bench import audit

    assert f.PROTOCOL_REJECTION_TYPES == audit.PROTOCOL_REJECTIONS
    assert f.REFUSAL_GATEWAY not in f.REFUSAL_CLASSES
    assert (
        f.REFUSAL_STREAK_ITEMS,
        f.PROVIDER_UNSERVED_TRIES,
        f.MIRIX_BATCH_ROLLBACKS,
    ) == (3, 2, 3)
    assert f.TOLERATED_REFUSAL_ROLES == {"actor", "learner", "judge"}
    assert f.Recovery.ACCEPT_FAILED_BATCH.value == "accept_failed_batch"


def test_a_provider_unserved_failure_has_the_class_of_its_status():
    unserved = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 404, "x")
    assert f.refusal_class(unserved) == f.REFUSAL_NO_ENDPOINT
    assert f.refusal_signature(unserved) == "actor/no_endpoint/404"
    for status in (400, 413):
        payload = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", status)
        assert f.refusal_signature(payload) == f"actor/payload/{status}"
    # The same refusal before and after the rollout converted it has one signature.
    refused = Failure(
        FailureKind.HARNESS,
        f.UPSTREAM_REJECTED,
        "actor",
        404,
        "No endpoints found that support tool use",
    )
    assert f.refusal_signature(refused) == f.refusal_signature(unserved)


def _refused(role: str, status: int = 413, detail: str = "too large") -> Failure:
    return Failure(FailureKind.HARNESS, f.UPSTREAM_REJECTED, role, status, detail, 0)


@pytest.mark.parametrize(
    ("phase", "failure", "expected"),
    [
        (Phase.LEARNING, _refused("learner"), Recovery.COUNT_AND_CONTINUE),
        (Phase.LEARNING, _refused("judge", 400), Recovery.COUNT_AND_CONTINUE),
        # A 400 that names no size or image is the harness's bug: a stop.
        (
            Phase.LEARNING,
            _refused(
                "learner", 400, "max_tokens: must be less than or equal to 131072"
            ),
            Recovery.RESTART_CELL,
        ),
        (
            Phase.LEARNING,
            _refused("actor", 400, "tool_call_id not found in previous messages"),
            Recovery.RESTART_CELL,
        ),
        (
            Phase.LEARNING,
            _refused("actor", 404, "No endpoints found that support tool use"),
            Recovery.COUNT_AND_CONTINUE,
        ),
        (
            Phase.LEARNING,
            _refused("mirix", 404, "No endpoints found that support image input"),
            Recovery.RESTART_CELL,
        ),
        (Phase.LEARNING, _refused("embedding", 400), Recovery.RESTART_CELL),
        (
            Phase.TEST,
            _refused("actor", 404, "No endpoints found that support tool use"),
            Recovery.RERUN_ITEM,
        ),
        (
            Phase.TEST,
            Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 404),
            Recovery.SCORE_AND_CONTINUE,
        ),
        (
            Phase.LEARNING,
            Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 413),
            Recovery.SCORE_AND_CONTINUE,
        ),
        (
            Phase.LEARNING,
            Failure(FailureKind.HARNESS, f.REFUSAL_STREAK, "learner", 413),
            Recovery.RESTART_CELL,
        ),
        (
            Phase.LEARNING,
            _refused("actor", 404, "No endpoints found matching your data policy"),
            Recovery.RESTART_CELL,
        ),
        (
            Phase.LEARNING,
            Failure(FailureKind.HARNESS, f.GATEWAY_REJECTED, "learner", 400),
            Recovery.RESTART_CELL,
        ),
    ],
)
def test_recovery_of_refusals(phase, failure, expected):
    assert f.recovery_for(phase, failure) is expected
    assert not f.is_scored_as_model_failure(failure)


def test_refusal_streaks_stop_on_the_third_consecutive_refusal_of_one_role():
    streaks = f.RefusalStreaks()
    judge = _refused("judge", 400, "Provider returned error")
    learner = _refused("learner", 413)
    # Two roles interleaved: each keeps its own streak.
    assert streaks.observe("judge", judge) is None
    assert streaks.observe("learner", learner) is None
    assert streaks.observe("judge", judge) is None
    assert (streaks.count("judge"), streaks.count("learner")) == (2, 1)
    # A served call resets its role only.
    assert streaks.observe("judge", None) is None
    assert streaks.count("judge") == 0 and streaks.count("learner") == 1
    # A failure that is not a refusal resets too.
    infra = Failure(FailureKind.INFRA, f.UPSTREAM_INFRA, "learner", 503)
    assert streaks.observe("learner", infra) is None and streaks.count("learner") == 0
    # A different signature restarts at 1.
    assert streaks.observe("learner", learner) is None
    assert streaks.observe("learner", _refused("learner", 400)) is None
    assert streaks.count("learner") == 1
    assert streaks.observe("learner", _refused("learner", 400)) is None
    recorded = streaks.to_dict()
    assert recorded == {"learner": {"signature": "learner/payload/400", "count": 2}}
    # The recorded state round-trips (a learning checkpoint) and the third one stops.
    again = f.RefusalStreaks(json.loads(json.dumps(recorded)))
    assert again.to_dict() == recorded
    stop = again.observe("learner", _refused("learner", 400, "Provider returned error"))
    assert stop is not None
    assert (stop.kind, stop.code, stop.role, stop.status) == (
        FailureKind.HARNESS,
        f.REFUSAL_STREAK,
        "learner",
        400,
    )
    assert "learner/payload/400" in stop.detail and "3 consecutive" in stop.detail
    assert "Provider returned error" in stop.detail and len(stop.detail) <= 500
    assert f.recovery_for(Phase.LEARNING, stop) is Recovery.RESTART_CELL
    # An explicit signature (MIRIX: several problems per idle check) is honoured.
    mirix = f.RefusalStreaks()
    gw = Failure(FailureKind.HARNESS, f.GATEWAY_REJECTED, "mirix", 409, "phase")
    assert mirix.observe("mirix", gw, signature="s") is None
    assert mirix.to_dict() == {"mirix": {"signature": "s", "count": 1}}
    # Unreadable recorded rows are ignored.
    assert (
        f.RefusalStreaks(
            {
                "a": {"signature": "x", "count": "2"},
                "b": {"count": 1},
                "c": [1],
                "d": {"signature": "x", "count": True},
                "e": {"signature": "x", "count": 0},
            }
        ).to_dict()
        == {}
    )
    assert f.RefusalStreaks(["not", "a", "mapping"]).to_dict() == {}  # type: ignore[arg-type]


def _result(
    raw_id: str, *, score: float | None, failure: Failure | None = None
) -> dict:
    return {
        "raw_id": raw_id,
        "score": None if score is None else {"primary": score},
        "failure": failure.to_dict() if failure else None,
    }


def test_unserved_retries_need_the_items_last_try_to_be_a_no_endpoint_refusal():
    no_endpoint = _refused("actor", 404, "No endpoints found that support tool use")
    payload = _refused("actor", 413)
    infra = Failure(FailureKind.INFRA, f.UPSTREAM_INFRA, "actor", 503)
    learner = _refused("learner", 404, "No endpoints found that support tool use")
    rows = [
        _result(
            "a", score=None, failure=no_endpoint
        ),  # a: refused once -> next is scored
        _result("b", score=None, failure=no_endpoint),
        _result("b", score=None, failure=infra),  # b: an INFRA try in between resets
        _result(
            "c", score=None, failure=payload
        ),  # c: a payload refusal is not counted
        _result("d", score=None, failure=learner),  # d: not the actor
        _result("e", score=None, failure=no_endpoint),
        _result("e", score=0.0),  # e: scored: nothing pending
        _result("f", score=None, failure=infra),
        _result("f", score=None, failure=no_endpoint),  # f: its last try was refused
        {"raw_id": "g", "score": None, "failure": {"kind": "bogus", "code": "x"}},
        {"raw_id": "h", "score": None, "failure": "not a mapping"},
    ]
    assert f.unserved_retries(rows) == frozenset({"a", "f"})
    assert f.unserved_retries([]) == frozenset()
    # A scored provider_unserved row (the item's second try) is final: not listed.
    scored = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 404)
    rows.append(_result("a", score=0.0, failure=scored))
    assert f.unserved_retries(rows) == frozenset({"f"})


def _learning(
    position: int, failure: Failure | None, *, score: float | None = 0.0
) -> dict:
    return {
        "position": position,
        "score": None if score is None else {"primary": score},
        "failure": failure.to_dict() if failure else None,
    }


def test_learning_refusal_streak_reads_the_last_three_positions():
    unserved = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 413, "large")
    other = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 404, "none")
    rows = [_learning(0, None), _learning(1, unserved), _learning(2, unserved)]
    assert f.learning_refusal_streak(rows) is None  # only two
    rows.append(_learning(3, unserved))
    stop = f.learning_refusal_streak(rows)
    assert stop is not None
    assert (stop.kind, stop.code, stop.role, stop.status) == (
        FailureKind.HARNESS,
        f.REFUSAL_STREAK,
        "actor",
        413,
    )
    assert "actor/payload/413" in stop.detail
    # A gap in the positions breaks the run.
    gap = [_learning(1, unserved), _learning(2, unserved), _learning(4, unserved)]
    assert f.learning_refusal_streak(gap) is None
    # Mixed signatures never stop.
    mixed = [_learning(1, unserved), _learning(2, other), _learning(3, unserved)]
    assert f.learning_refusal_streak(mixed) is None
    # A replayed position keeps its latest row.
    replayed = [*rows, _learning(3, None, score=1.0)]
    assert f.learning_refusal_streak(replayed) is None
    back = [*replayed, _learning(3, unserved)]
    assert f.learning_refusal_streak(back) is not None
    # An unscored row (a resumed infrastructure failure) is not a refusal.
    unscored = [*rows[:-1], _learning(3, unserved, score=None)]
    assert f.learning_refusal_streak(unscored) is None
    # A model failure, or rows without positions, never count.
    model = Failure(FailureKind.MODEL, f.MODEL_TIMEOUT, "actor", 504)
    assert (
        f.learning_refusal_streak(
            [_learning(p, model) for p in range(3)] + [{"score": None}]
        )
        is None
    )
    assert f.learning_refusal_streak([]) is None
    assert f.learning_refusal_streak(rows, items=2) is not None


def test_a_test_item_scored_unanswered_is_logged_score_and_continue(tmp_path):
    """The failures.jsonl rows run.py writes for a test item refused no-endpoint twice:
    the first try re-runs the item, the scored second try continues (S1 (a))."""
    log = f.FailureLog(tmp_path / "failures.jsonl")
    refused = _refused("actor", 404, "No endpoints found that support tool use")
    assert log.record("test", refused, item="u", attempt=1) is Recovery.RERUN_ITEM
    unserved = Failure(FailureKind.HARNESS, f.PROVIDER_UNSERVED, "actor", 404, "x")
    recovery = log.record("test", unserved, item="u", attempt=1, stage="actor")
    assert recovery is Recovery.SCORE_AND_CONTINUE
    first, second = log.rows()
    assert "stage" not in first  # rows without a stage keep today's shape
    assert (second["stage"], second["code"], second["recovery"]) == (
        "actor",
        "provider_unserved",
        "score_and_continue",
    )
    assert log.counters().total(phase="test", kind=FailureKind.HARNESS) == 2
