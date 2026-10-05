"""tools/preflight.py: every readiness check, offline.

OpenRouter is an ``httpx.MockTransport`` that serves ``/key``, ``/credits`` and the saved
model metadata of ``fixtures/openrouter_models_subset.json`` (the payloads that build the
committed ``config/models.json``); the environments and MIRIX's preflight are fakes,
except in the last tests, which run the real ones on this machine's data and this
checkout's MIRIX (the network checks stay off). The key never reaches the output.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.tools import preflight as P

FIXTURE = Path(__file__).parent / "fixtures" / "openrouter_models_subset.json"
#: A saved public endpoint list of z-ai/glm-5.3-flash (trimmed).
ENDPOINTS = Path(__file__).parent / "fixtures" / "openrouter_glm_endpoints.json"
KEY = "sk-or-v1-preflight-test-key-never-printed"


@pytest.fixture
def provider_policy(monkeypatch):
    """A test-only provider policy on the endpoint list above (the protocol names no
    model): z-ai/glm-5.3-flash ignores open-inference and wafer."""
    monkeypatch.setattr(
        protocol,
        "PROVIDER_POLICY",
        {"z-ai/glm-5.3-flash": {"ignore": ("open-inference", "wafer")}},
    )
    monkeypatch.setattr(
        protocol,
        "PROVIDER_SLUG_NAMES",
        {"open-inference": "OpenInference", "wafer": "Wafer"},
    )


class OpenRouter:
    """``httpx.MockTransport`` handler for the endpoints the preflight reads."""

    def __init__(self, *, remaining: float | None = 50.0) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.chat = fixture["chat"]
        self.embeddings = fixture["embeddings"]
        self.remaining = remaining
        self.endpoints = json.loads(ENDPOINTS.read_text())
        self.credits: tuple[float, float] | None = None
        self.key_status = 200
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        auth = request.headers.get("authorization")
        if path == "/key":
            if auth != f"Bearer {KEY}" or self.key_status != 200:
                return httpx.Response(
                    self.key_status if self.key_status != 200 else 401,
                    json={"error": {"message": "No auth credentials found"}},
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "label": "sk-or-v1-pre...",
                        "limit": 100,
                        "limit_remaining": self.remaining,
                        "usage": 12.5,
                        "is_free_tier": False,
                    }
                },
            )
        if path == "/credits":
            if self.credits is None:
                return httpx.Response(403, json={"error": {"message": "forbidden"}})
            total, used = self.credits
            return httpx.Response(
                200, json={"data": {"total_credits": total, "total_usage": used}}
            )
        if path == "/models":
            assert auth is None  # public metadata: no key sent
            return httpx.Response(200, json=self.chat)
        if path == "/embeddings/models":
            assert auth is None
            return httpx.Response(200, json=self.embeddings)
        if path == f"/models/{self.endpoints['data']['id']}/endpoints":
            assert auth is None  # public: no key sent
            return httpx.Response(200, json=self.endpoints)
        return httpx.Response(404)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


class FakeEnv:
    def __init__(self, benchmark: str, broken: str | None = None) -> None:
        self.benchmark = benchmark
        self.broken = broken

    def load_tasks(self, split: str) -> list[int]:
        if self.broken == "load":
            raise FileNotFoundError(f"data/{self.benchmark} is not prepared")
        sizes = {
            **protocol.SPLIT_SIZES[self.benchmark],
            **protocol.EXTRA_SPLIT_SIZES.get(self.benchmark, {}),
        }
        return list(range(sizes[split]))


class Fakes:
    """Deps whose environments and MIRIX answer as configured."""

    def __init__(self, router: OpenRouter | None = None) -> None:
        self.router = router or OpenRouter()
        self.broken: dict[str, str] = {}
        self.mirix_error: Exception | None = None
        self.webshop = False  # the repository's WebShop manifest is provisional
        self.drift: Mapping[str, Any] = {
            "drifted": False,
            "current": "f" * 64,
            "frozen": "f" * 64,
        }
        self.env_calls: list[tuple[str, tuple[str, ...]]] = []
        self.mirix_dirs: list[Path] = []
        self.dirty = False
        self.resources: dict[str, Any] = {
            "mem_total_gb": 32.0,
            "mem_available_gb": 12.5,
            "swap_used_gb": 0.5,
            "disk_free_gb": 120.0,
            "errors": [],
        }
        self.code_calls: list[bool] = []
        self.machine_id: str | None = "0123abcd" * 4

    def deps(self) -> P.Deps:
        async def env_preflight(env: FakeEnv, benchmark: str, splits: Sequence[str]):
            self.env_calls.append((benchmark, tuple(splits)))
            if self.broken.get(benchmark) == "preflight":
                raise RuntimeError(f"{benchmark} game files are not the pinned ones")
            return {"splits": list(splits)}

        async def mirix(cell_dir: Path):
            self.mirix_dirs.append(cell_dir)
            assert cell_dir.is_dir()
            if self.mirix_error is not None:
                raise self.mirix_error
            return {
                "postgres": "127.0.0.1:55432",
                "template": "skill_bench_template",
                "llm_contract": {
                    "llm_max_retries": 0,
                    "llm_timeout_seconds": 630.0,
                    "embedding_timeout_seconds": 630.0,
                },
            }

        def code_fingerprint(*, mirix: bool = False) -> dict[str, Any]:
            self.code_calls.append(mirix)
            code: dict[str, Any] = {"harness": "a" * 64, "harness_files": 120}
            if mirix:
                code.update(mirix="b" * 64, mirix_files=900)
            return code

        return P.Deps(
            transport=httpx.MockTransport(self.router),
            env_factory=lambda b: FakeEnv(b, self.broken.get(b)),
            env_preflight=env_preflight,
            mirix=mirix,
            protocol_drift=lambda: self.drift,
            webshop_prepared=lambda: self.webshop,
            code_fingerprint=code_fingerprint,
            git=lambda: {"commit": "c" * 40, "dirty": self.dirty, "branch": "main"},
            host=lambda: {
                "hostname": "mac-1",
                "machine_id": self.machine_id,
                "system": "Darwin",
                "release": "25.4.0",
                "machine": "arm64",
                "python": "3.11.13",
            },
            resources=lambda where: dict(self.resources, at=str(where)),
        )


def env_file(tmp_path: Path, key: str | None = KEY) -> Path:
    path = tmp_path / "keys.env"
    path.write_text(f"OPENROUTER_API_KEY={key}\nOTHER=1\n" if key else "OTHER=1\n")
    return path


def run(fakes: Fakes, capsys, *argv: str) -> tuple[int, dict[str, dict[str, Any]], str]:
    code = P.main([*argv, "--json"], deps=fakes.deps())
    out = capsys.readouterr()
    assert KEY not in out.out and KEY not in out.err
    return code, {c["name"]: c for c in json.loads(out.out)}, out.out


# --- all good ------------------------------------------------------------------------------


def test_everything_ready(tmp_path, capsys):
    fakes = Fakes()
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0
    assert {n: c["status"] for n, c in checks.items()} == {
        "key": P.OK,
        "balance": P.OK,
        "models": P.OK,
        "provider_policy": P.OK,
        "protocol": P.OK,
        "code": P.OK,
        "host": P.OK,
        "budget": P.SKIP,
        "resources": P.OK,
        "data:alfworld": P.OK,
        "data:webshop": P.SKIP,
        "mirix": P.OK,
    }
    assert checks["key"]["detail"].startswith(
        "OPENROUTER_API_KEY present in --env-file"
    )
    assert checks["balance"]["detail"] == (
        "$50.00 left (key limit); needs $10.00 (floor $10.00 + budget $0.00)"
    )
    assert checks["balance"]["info"]["usage_usd"] == 12.5
    assert "matches the live metadata" in checks["models"]["detail"]
    # The line names the model each short key runs, before anything is paid.
    assert checks["models"]["detail"].startswith(
        f"luna={protocol.MODELS['luna']} luna56={protocol.MODELS['luna56']} "
        f"embedding={protocol.EMBEDDING_MODEL}; "
    )
    assert checks["models"]["info"]["resolved"] == {
        "luna": "openai/gpt-6-luna",
        "luna56": "openai/gpt-5.6-luna",
        "embedding": "google/gemini-embedding-001",
    }
    assert checks["data:alfworld"]["detail"] == (
        "items load (train 39, val 18); environment preflight passed"
    )
    # Split mode smoke reads train and val, never the test split.
    assert fakes.env_calls == [("alfworld", ("train", "val"))]
    assert (
        "MIRIX retries 0, timeout 630.0 s, embedding timeout 630.0 s"
        in checks["mirix"]["detail"]
    )
    assert not fakes.mirix_dirs[0].exists()  # its scratch cell directory is removed
    assert fakes.router.paths() == [
        "/api/v1/key",
        "/api/v1/models",
        "/api/v1/embeddings/models",
    ]


def test_text_output(tmp_path, capsys):
    fakes = Fakes()
    assert P.main(["--env-file", str(env_file(tmp_path))], deps=fakes.deps()) == 0
    out = capsys.readouterr().out
    assert KEY not in out
    assert "[ok  ] key" in out and "[skip] data:webshop" in out
    assert out.rstrip().endswith("preflight: ready (no check failed)")


# --- the key and the balance -------------------------------------------------------------------


def test_a_missing_key_fails_and_nothing_is_asked_with_it(
    tmp_path, capsys, monkeypatch
):
    fakes = Fakes()
    code, checks, text = run(fakes, capsys, "--env-file", str(env_file(tmp_path, None)))
    assert code == 1
    assert checks["key"]["status"] == P.FAIL
    assert checks["key"]["detail"].startswith(
        "OPENROUTER_API_KEY not found in --env-file"
    )
    assert checks["balance"]["status"] == P.SKIP
    assert "/api/v1/key" not in fakes.router.paths()
    code, checks, _ = run(fakes, capsys, "--env-file", str(tmp_path / "nope.env"))
    assert "(no such file)" in checks["key"]["detail"]
    # Without --env-file the key comes from the environment, as the gateway reads it.
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    code, checks, _ = run(fakes, capsys)
    assert code == 0 and checks["key"]["detail"].startswith(
        "OPENROUTER_API_KEY present in $OPENROUTER_API_KEY"
    )


def test_a_rejected_key(tmp_path, capsys):
    fakes = Fakes()
    fakes.router.key_status = 401
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert checks["balance"] == {
        "name": "balance",
        "status": P.FAIL,
        "detail": "OpenRouter rejected the key (HTTP 401)",
        "info": {},
    }


@pytest.mark.parametrize(
    ("remaining", "argv", "status", "detail"),
    [
        (9.99, (), P.FAIL, "$9.99 left (key limit); needs $10.00"),
        (25.0, ("--budget", "20"), P.FAIL, "$25.00 left (key limit); needs $30.00"),
        (25.0, ("--budget", "10"), P.OK, "$25.00 left (key limit); needs $20.00"),
        (3.0, ("--balance-floor", "0", "--budget", "2.5"), P.OK, "needs $2.50"),
    ],
)
def test_the_balance_must_cover_the_floor_and_the_budget(
    tmp_path, capsys, remaining, argv, status, detail
):
    fakes = Fakes(OpenRouter(remaining=remaining))
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)), *argv)
    assert checks["balance"]["status"] == status
    assert detail in checks["balance"]["detail"]
    assert code == (1 if status == P.FAIL else 0)


def test_a_key_without_a_limit_uses_the_account_credits(tmp_path, capsys):
    fakes = Fakes(OpenRouter(remaining=None))
    fakes.router.credits = (120.0, 100.0)
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0
    assert checks["balance"]["detail"].startswith("$20.00 left (account credits)")
    fakes.router.credits = None  # credits unreadable: the balance is unknown
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert checks["balance"]["detail"].startswith("balance unknown")


def test_openrouter_unreachable(tmp_path, capsys):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    fakes = Fakes()
    deps = fakes.deps()
    deps.transport = httpx.MockTransport(down)
    code = P.main(["--env-file", str(env_file(tmp_path)), "--json"], deps=deps)
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)}
    assert code == 1
    assert checks["balance"]["detail"] == "OpenRouter unreachable: ConnectError"
    assert checks["models"]["detail"].endswith(
        "; model metadata unavailable: ConnectError"
    )


def test_the_key_is_scrubbed_from_every_line(tmp_path, capsys):
    """Even a check whose error quotes the key never prints it."""
    fakes = Fakes()
    fakes.mirix_error = RuntimeError(f"refused: Authorization Bearer {KEY}")
    code, checks, text = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert (
        checks["mirix"]["detail"]
        == "RuntimeError: refused: Authorization Bearer <redacted>"
    )


# --- models.json and the protocol ------------------------------------------------------------


def test_models_json_drift_names_what_changed(tmp_path, capsys):
    """A change to what the protocol reads is recorded with its
    frozen and its live value, as a warning: preflight passes (exit 0)."""
    router = OpenRouter()
    router.chat = copy.deepcopy(router.chat)
    entry = next(m for m in router.chat["data"] if m["id"] == protocol.MODELS["luna"])
    frozen_window = entry["context_length"]
    entry["context_length"] += 1
    fakes = Fakes(router)
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0
    assert checks["models"]["status"] == P.WARN
    luna = f"models.{protocol.MODELS['luna']}"
    # The window is what the protocol reads (one token more leaves luna's floored
    # limits as they were); the raw entry's hash and the notes derived from it (a
    # window note now names luna) are not.
    assert checks["models"]["info"]["differences"] == [f"{luna}.context_length"]
    assert checks["models"]["info"]["unread_differences"] == [
        "notes",
        f"{luna}.raw_sha256",
    ]
    assert checks["models"]["info"]["changes"] == [
        {
            "field": f"{luna}.context_length",
            "frozen": frozen_window,
            "live": frozen_window + 1,
        }
    ]
    detail = checks["models"]["detail"]
    assert f"{luna}.context_length {frozen_window} -> {frozen_window + 1}." in detail
    assert "Recorded, not blocking: cells keep using config/models.json" in detail
    assert "re-freeze the protocol, before any paid cell" not in detail
    router.chat = {"data": [m for m in router.chat["data"] if m is not entry]}
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert checks["models"]["detail"].endswith(
        f"; the live metadata is unusable: {protocol.MODELS['luna']} not found at "
        "https://openrouter.ai/api/v1/models"
    )


def test_drift_the_protocol_does_not_read_is_only_a_warning(tmp_path, capsys):
    """OpenRouter edits a field of its raw entry that is not copied (here the
    description), which changes ``raw_sha256``, and the pricing. Nothing the protocol
    reads changed, so preflight passes with a warning and asks for no re-freeze."""
    router = OpenRouter()
    router.chat = copy.deepcopy(router.chat)
    entry = next(m for m in router.chat["data"] if m["id"] == protocol.MODELS["luna"])
    entry["description"] = str(entry.get("description") or "") + " (edited)"
    entry["pricing"] = {**entry["pricing"], "prompt": "0.000009"}
    fakes = Fakes(router)
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    luna = f"models.{protocol.MODELS['luna']}"
    assert code == 0
    assert checks["models"]["status"] == P.WARN
    assert "differences" not in checks["models"]["info"]
    assert checks["models"]["info"]["unread_differences"] == [
        f"{luna}.pricing.prompt",
        f"{luna}.raw_sha256",
    ]
    assert "no re-freeze" in checks["models"]["detail"]


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (
            lambda e: e["supported_parameters"].remove("seed"),
            "supported_parameters.seed",
        ),
        (
            lambda e: e["top_provider"].update(max_completion_tokens=1000),
            "top_provider.max_completion_tokens",
        ),
        (
            lambda e: e["architecture"].update(input_modalities=["text"]),
            "input_modalities",
        ),
        (
            lambda e: e["reasoning"].update(supported_efforts=["high"]),
            "reasoning.supported_efforts",
        ),
    ],
)
def test_drift_of_what_the_protocol_reads_is_a_warning_with_both_values(
    tmp_path, capsys, edit, field
):
    from evals.skill_bench.tools import fetch_models

    frozen = json.loads(protocol.MODELS_JSON.read_text(encoding="utf-8"))
    luna = protocol.MODELS["luna"]
    router = OpenRouter()
    router.chat = copy.deepcopy(router.chat)
    entry = next(m for m in router.chat["data"] if m["id"] == luna)
    edit(entry)
    code, checks, _ = run(Fakes(router), capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0 and checks["models"]["status"] == P.WARN
    info = checks["models"]["info"]
    assert f"models.{luna}.{field}" in info["differences"]
    change = next(c for c in info["changes"] if c["field"] == f"models.{luna}.{field}")
    old_value = fetch_models.protocol_fields(frozen["models"][luna])[field]
    assert change["frozen"] == old_value and change["live"] != old_value
    # Every protocol difference has its values; nothing else is listed as a change.
    assert [c["field"] for c in info["changes"]] == info["differences"]
    assert "Recorded, not blocking" in checks["models"]["detail"]


def test_field_changes_list_added_and_removed_models_and_top_level_keys():
    frozen = {
        "schema_version": 1,
        "fetched_at": "then",
        "models": {"a/one": {"context_length": 10}, "b/two": {"context_length": 5}},
    }
    live = {
        "schema_version": 2,
        "fetched_at": "now",
        "models": {"a/one": {"context_length": 12}, "c/three": {"context_length": 7}},
    }
    assert P._field_changes(frozen, live) == [
        {"field": "schema_version", "frozen": 1, "live": 2},
        {"field": "models.a/one.context_length", "frozen": 10, "live": 12},
        {"field": "models.b/two", "frozen": "present", "live": None},
        {"field": "models.c/three", "frozen": None, "live": "present"},
    ]


def test_a_long_value_is_cut_in_the_detail_only():
    value = ["x" * 50] * 10
    shown = P._shown(value)
    assert len(shown) == P.SHOWN_VALUE_CHARS and shown.endswith("...")
    assert P._shown(1310720) == "1310720"


def test_unusable_model_metadata_still_fails(tmp_path, capsys):
    """No metadata at all cannot be recorded as a change: it stays a failure."""
    router = OpenRouter()
    router.chat = copy.deepcopy(router.chat)
    router.chat = {
        "data": [m for m in router.chat["data"] if m["id"] != protocol.MODELS["luna"]]
    }
    code, checks, _ = run(Fakes(router), capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1 and checks["models"]["status"] == P.FAIL
    assert "the live metadata is unusable" in checks["models"]["detail"]


def test_fetch_models_check_follows_the_same_rule(tmp_path, capsys):
    from evals.skill_bench.tools import fetch_models

    fixture = json.loads(FIXTURE.read_text())
    chat, embeddings = copy.deepcopy(fixture["chat"]), fixture["embeddings"]
    (tmp_path / "e.json").write_text(json.dumps(embeddings))
    entry = next(m for m in chat["data"] if m["id"] == protocol.MODELS["luna"])
    entry["description"] = "edited"
    (tmp_path / "c.json").write_text(json.dumps(chat))
    argv = ["--check", "--chat-json", str(tmp_path / "c.json")]
    argv += ["--embeddings-json", str(tmp_path / "e.json")]
    assert fetch_models.main(argv) == 0
    assert "needs no re-freeze" in capsys.readouterr().out
    entry["context_length"] += 1
    (tmp_path / "c.json").write_text(json.dumps(chat))
    assert fetch_models.main(argv) == 1
    assert "context_length" in capsys.readouterr().err


def test_protocol_drift_is_a_warning(tmp_path, capsys):
    """run.py auto-freezes a drifted tree at the next attempt or
    resume, so preflight records the drift as a warning and passes."""
    fakes = Fakes()
    fakes.drift = {
        "drifted": True,
        "current": "a" * 64,
        "frozen": "b" * 64,
        "constants_changed": True,
        "constants_changed_keys": ["actor_max_turns"],
        "changed": ["envs/templates/alfworld/actor.txt"],
        "added": [],
        "removed": [],
    }
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0
    check = checks["protocol"]
    assert check["status"] == P.WARN
    assert "changed: envs/templates/alfworld/actor.txt" in check["detail"]
    assert "freezes this tree automatically" in check["detail"]
    assert check["detail"].endswith(P.AUTO_FREEZE_TEXT)
    # The manual re-freeze hint is replaced by the auto-freeze text.
    assert "re-freeze with" not in check["detail"]
    assert check["info"]["current"] == "a" * 64
    assert check["info"]["frozen"] == "b" * 64
    assert check["info"]["changed"] == ["envs/templates/alfworld/actor.txt"]
    assert check["info"]["constants_changed_keys"] == ["actor_max_turns"]


def test_a_missing_or_corrupt_frozen_record_is_a_warning(tmp_path):
    """A missing or corrupt record is recorded, never recreated: a warning that
    points at tools/freeze_protocol."""
    missing = tmp_path / "missing.json"
    check = P.check_protocol(lambda: protocol.protocol_drift(path=missing))
    assert check.status == P.WARN
    assert "missing" in check.detail and check.detail.endswith(P.AUTO_FREEZE_TEXT)
    assert "repair with tools/freeze_protocol" in check.detail
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    check = P.check_protocol(lambda: protocol.protocol_drift(path=corrupt))
    assert check.status == P.WARN
    assert check.info["error"].startswith("JSONDecodeError")
    assert check.detail.endswith(P.AUTO_FREEZE_TEXT)


def test_a_removed_method_is_refused_with_the_decision(
    tmp_path, capsys, removed_method
):
    """A method the study removed is refused before any check runs (exit 2), with the
    decision, not "invalid choice"."""
    fakes = Fakes()
    with pytest.raises(SystemExit) as exc:
        P.main(
            ["--env-file", str(env_file(tmp_path)), "--method", removed_method],
            deps=fakes.deps(),
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "example removal" in err and "invalid choice" not in err


def test_the_real_protocol_check_passes_on_this_tree():
    check = P.check_protocol(protocol.protocol_drift)
    assert check.status != P.FAIL, check.detail
    assert check.status == P.OK or check.detail.endswith(P.AUTO_FREEZE_TEXT)


# --- data, MIRIX ----------------------------------------------------------------------------


def test_missing_or_unpinned_data_fails_that_benchmark(tmp_path, capsys):
    fakes = Fakes()
    fakes.webshop = True
    fakes.broken = {"webshop": "load", "alfworld": "preflight"}
    code, checks, _ = run(fakes, capsys, "--offline")
    assert code == 1
    assert checks["data:webshop"]["detail"] == (
        "FileNotFoundError: data/webshop is not prepared"
    )
    assert checks["data:alfworld"]["detail"] == (
        "RuntimeError: alfworld game files are not the pinned ones"
    )


def test_webshop_is_checked_when_its_manifest_is_final_or_named(tmp_path, capsys):
    """A provisional WebShop manifest skips its data check (the server and the real data
    may be on another host), a final one runs it, and naming WebShop runs it whatever
    the manifest says."""
    fakes = Fakes()
    code, checks, out = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert checks["data:webshop"]["status"] == P.SKIP and code == 0
    assert "prepare_webshop freeze" in checks["data:webshop"]["detail"]
    assert all(b != "webshop" for b, _ in fakes.env_calls)
    fakes.webshop = True
    code, checks, _ = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert checks["data:webshop"]["status"] == P.OK and code == 0
    assert ("webshop", ("train", "val")) in fakes.env_calls
    fakes.webshop = False
    fakes.broken = {"webshop": "preflight"}
    code, checks, _ = run(
        fakes,
        capsys,
        "--offline",
        "--env-file",
        str(env_file(tmp_path)),
        "--benchmark",
        "webshop",
    )
    assert set(checks) >= {"data:webshop"} and "data:alfworld" not in checks
    assert checks["data:webshop"]["status"] == P.FAIL and code == 1


def test_the_default_webshop_readiness_reads_the_manifest_status():
    from evals.skill_bench.envs.webshop import load_split_manifest

    assert P._webshop_prepared() is (load_split_manifest()["status"] == "final")


def test_protocol_mode_reads_the_test_split(tmp_path, capsys):
    fakes = Fakes()
    run(
        fakes,
        capsys,
        "--offline",
        "--split-mode",
        "protocol",
        "--benchmark",
        "alfworld",
    )
    assert fakes.env_calls == [("alfworld", ("train", "test"))]


def test_a_study_checks_its_benchmarks_methods_and_split(tmp_path, capsys):
    """``--study`` checks what the scheduler's ``--study``
    runs (study.STUDIES), narrowed by ``--benchmark`` and ``--method``."""
    fakes = Fakes()
    fakes.webshop = True
    run(fakes, capsys, "--offline", "--study", "phase16")
    assert fakes.env_calls == [
        ("alfworld", ("train", "test")),
        ("webshop", ("train", "test")),
    ]
    fakes.env_calls.clear()
    run(fakes, capsys, "--offline", "--study", "phase16-pilot")
    assert fakes.env_calls == [("webshop", ("pilot",))]
    fakes.env_calls.clear()
    run(fakes, capsys, "--offline", "--study", "phase16-dev", "--benchmark", "alfworld")
    assert fakes.env_calls == [("alfworld", ("train", "val"))]
    for bad in (
        ("--study", "phase16-pilot", "--benchmark", "alfworld"),
        ("--study", "phase16", "--split-mode", "smoke"),
        ("--study", "no-such-study"),
    ):
        with pytest.raises(SystemExit):
            P.main(["--offline", *bad], deps=fakes.deps())
    err = capsys.readouterr().err
    assert "outside --study phase16-pilot" in err
    assert "unknown study 'no-such-study'" in err


def test_mirix_is_checked_only_for_mirix_cells(tmp_path, capsys):
    fakes = Fakes()
    fakes.mirix_error = RuntimeError(
        "MIRIX cells need the compose Postgres on 127.0.0.1:55432"
    )
    code, checks, _ = run(
        fakes,
        capsys,
        "--offline",
        "--env-file",
        str(env_file(tmp_path)),
        "--method",
        "none",
        "awm",
    )
    assert checks["mirix"]["status"] == P.SKIP and code == 0
    code, checks, _ = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert checks["mirix"]["status"] == P.FAIL and code == 1
    assert "compose Postgres" in checks["mirix"]["detail"]


def test_offline_sends_nothing(tmp_path, capsys, provider_policy):
    fakes = Fakes()
    code, checks, _ = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert code == 0
    assert (checks["balance"]["status"], checks["models"]["status"]) == (P.SKIP, P.SKIP)
    assert checks["provider_policy"]["status"] == P.SKIP
    assert checks["provider_policy"]["detail"] == (
        "--offline: z-ai/glm-5.3-flash ignores open-inference, wafer not compared with "
        "the live endpoints"
    )
    # Offline too, the line names the models this tree would run.
    assert checks["models"]["detail"].startswith(f"luna={protocol.MODELS['luna']} ")
    assert fakes.router.requests == []


# --- the provider policy ------------------------------------------------------------------


def test_the_provider_policy_names_real_endpoints(tmp_path, capsys, provider_policy):
    fakes = Fakes()
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    policy = checks["provider_policy"]
    assert code == 0 and policy["status"] == P.OK
    assert policy["detail"] == (
        "z-ai/glm-5.3-flash ignores open-inference (1 endpoints), wafer (1 endpoints); "
        "32 other endpoints, 32 up"
    )
    assert policy["info"]["z-ai/glm-5.3-flash"]["ignored_endpoints"] == {
        "open-inference": ["open-inference/fp4"],
        "wafer": ["wafer"],
    }


def _endpoints(fakes: Fakes) -> list[dict[str, Any]]:
    fakes.router.endpoints = copy.deepcopy(fakes.router.endpoints)
    return fakes.router.endpoints["data"]["endpoints"]


def test_an_ignored_slug_that_names_no_endpoint_warns(
    tmp_path, capsys, provider_policy
):
    """A provider OpenRouter delisted between two hosts' preflights makes its ignore
    entry match nothing, which excludes nothing: a warning, not a failure whose obvious
    "fix" (editing the policy, re-freezing) would split the protocol across hosts
    mid-run."""
    fakes = Fakes()
    endpoints = _endpoints(fakes)
    endpoints[:] = [e for e in endpoints if not e["tag"].startswith("wafer")]
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0 and checks["provider_policy"]["status"] == P.WARN
    assert (
        "ignored slug 'wafer' names no endpoint now"
        in checks["provider_policy"]["detail"]
    )
    assert "leave the policy as frozen" in checks["provider_policy"]["detail"]
    # Renamed under its old name, the provider is still caught (a failure).
    renamed = next(e for e in _endpoints(fakes) if e["tag"] == "relace")
    renamed["provider_name"] = "Wafer"
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1 and checks["provider_policy"]["status"] == P.FAIL
    assert (
        "would flag allowed providers as ignored: Wafer"
        in (checks["provider_policy"]["detail"])
    )


def test_a_provider_name_the_audit_would_not_recognise_fails(
    tmp_path, capsys, provider_policy
):
    """The ledger records the provider's name: the audit must match it to the slug."""
    fakes = Fakes()
    wafer = next(e for e in _endpoints(fakes) if e["tag"] == "wafer")
    wafer["provider_name"] = "Wafer Systems"
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert (
        "the audit would not recognise 'Wafer Systems' (tag wafer)"
        in (checks["provider_policy"]["detail"])
    )


def test_an_allowed_provider_the_audit_would_flag_fails(
    tmp_path, capsys, provider_policy
):
    fakes = Fakes()
    other = next(e for e in _endpoints(fakes) if e["tag"] == "relace")
    other["provider_name"] = "Open-Inference"
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert (
        "would flag allowed providers as ignored: Open-Inference"
        in (checks["provider_policy"]["detail"])
    )


def test_too_few_endpoints_left_up_is_a_warning(tmp_path, capsys, provider_policy):
    fakes = Fakes()
    for endpoint in _endpoints(fakes)[:30]:
        endpoint["status"] = -5
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 0 and checks["provider_policy"]["status"] == P.WARN
    assert (
        "only 3 other endpoints are up (status 0), fewer than 5"
        in (checks["provider_policy"]["detail"])
    )


def test_an_unavailable_endpoint_list_fails(tmp_path, capsys, provider_policy):
    fakes = Fakes()
    fakes.router.endpoints = {"data": {"id": "somebody/else", "endpoints": []}}
    code, checks, _ = run(fakes, capsys, "--env-file", str(env_file(tmp_path)))
    assert code == 1
    assert checks["provider_policy"]["detail"] == (
        "z-ai/glm-5.3-flash: endpoint list unavailable (HTTPStatusError)"
    )


# --- code, host, budget and resources -----------------------------------------------------


def test_the_code_and_host_lines(tmp_path, capsys):
    fakes = Fakes()
    code, checks, _ = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert checks["code"]["detail"] == (
        "commit cccccccccccc (main); harness aaaaaaaaaaaaaaaa (120 files), "
        "mirix bbbbbbbbbbbbbbbb (900 files)"
    )
    assert checks["host"]["detail"] == (
        "mac-1 (Darwin 25.4.0 arm64), Python 3.11.13, machine id 0123abcd"
    )
    assert checks["host"]["status"] == P.OK
    assert fakes.code_calls == [True]
    # Without a MIRIX cell MIRIX's digest is not computed; a dirty tree warns.
    fakes.dirty = True
    code, checks, _ = run(
        fakes,
        capsys,
        "--offline",
        "--env-file",
        str(env_file(tmp_path)),
        "--method",
        "none",
    )
    assert fakes.code_calls[-1] is False
    assert checks["code"]["status"] == P.WARN and code == 0
    assert "mirix" not in checks["code"]["detail"].split(";")[1]
    assert "the tracked files differ from the commit" in checks["code"]["detail"]


def _budget(runs: Path, record: Any) -> None:
    path = runs / "_gateway" / "budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record if isinstance(record, str) else json.dumps(record))


def test_the_budget_line_reads_the_shared_start(tmp_path, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    fakes = Fakes()
    args = ("--env-file", str(env_file(tmp_path)), "--runs-dir", str(runs))
    _, checks, _ = run(fakes, capsys, *args)
    assert checks["budget"]["status"] == P.OK
    assert "no budget record" in checks["budget"]["detail"]
    assert "copies the first host's budget.json" in checks["budget"]["detail"]
    record = {
        "start_usage_usd": 10.0,
        "started_at": "2026-09-29T05:00:00+00:00",
        "budgets": [{"budget_usd": 200.0}],
    }
    _budget(runs, record)
    _, checks, _ = run(fakes, capsys, *args, "--budget", "25")
    assert checks["budget"]["status"] == P.OK
    assert checks["budget"]["detail"] == (
        "start $10.00 at 2026-09-29T05:00:00+00:00 (last budget $200.00); $2.50 spent "
        "since, $22.50 left of --budget $25.00"
    )
    assert checks["budget"]["info"]["left_usd"] == 22.5
    # Spent (the key's usage is 12.5), another key, an unreadable record.
    code, checks, _ = run(fakes, capsys, *args, "--budget", "2.5")
    assert code == 1 and "(budget_spent)" in checks["budget"]["detail"]
    _budget(runs, {**record, "start_usage_usd": 20.0})
    code, checks, _ = run(fakes, capsys, *args)
    assert code == 1 and "another key?" in checks["budget"]["detail"]
    _budget(runs, "{")
    code, checks, _ = run(fakes, capsys, *args)
    assert code == 1 and "(budget_unknown)" in checks["budget"]["detail"]
    # Offline: the record only.
    _budget(runs, record)
    code, checks, _ = run(fakes, capsys, "--offline", *args)
    assert checks["budget"]["detail"].endswith("the key's usage is unknown here")
    # A merged view of two hosts is never a place to run cells.
    (runs / "_merged.json").write_text("{}")
    code, checks, _ = run(fakes, capsys, "--offline", *args)
    assert code == 1 and "merged view" in checks["budget"]["detail"]


def test_the_budget_line_reads_the_replicate_start_it_will_copy(tmp_path, capsys):
    """``schedule.py --budget-start-from runs/phase16-r1`` copies r1's
    record at r2's first launch; before that, r2's preflight reads r1's."""
    r1, r2 = tmp_path / "phase16-r1", tmp_path / "phase16-r2"
    r2.mkdir()
    fakes = Fakes()
    args = (
        "--env-file",
        str(env_file(tmp_path)),
        "--runs-dir",
        str(r2),
        "--budget-start-from",
        str(r1),
    )
    code, checks, _ = run(fakes, capsys, *args, "--budget", "25")
    assert checks["budget"]["status"] == P.WARN and code == 0
    assert "start that runs directory first" in checks["budget"]["detail"]
    _budget(r1, {"start_usage_usd": 10.0, "started_at": "t1", "budgets": []})
    _, checks, _ = run(fakes, capsys, *args, "--budget", "25")
    assert checks["budget"]["status"] == P.OK
    detail = checks["budget"]["detail"]
    assert detail.startswith("start $10.00 at t1, shared from ")
    assert "$2.50 spent since, $22.50 left of --budget $25.00" in detail
    assert checks["budget"]["info"]["start_from"].endswith(
        "phase16-r1/_gateway/budget.json"
    )
    # Once r2 has a record of its own, that one is read.
    _budget(r2, {"start_usage_usd": 11.0, "started_at": "t2", "budgets": []})
    _, checks, _ = run(fakes, capsys, *args, "--budget", "25")
    assert checks["budget"]["detail"].startswith("start $11.00 at t2;")


def test_the_balance_needs_only_what_the_shared_budget_has_left(tmp_path, capsys):
    """A host's preflight may run after another host spent part of the shared budget;
    the key must hold the floor plus what is left of it."""
    runs = tmp_path / "runs"
    runs.mkdir()
    # The key's usage is 12.5 (the fake): $6.50 spent since the shared start at $6.
    _budget(runs, {"start_usage_usd": 6.0, "budgets": [{"budget_usd": 200.0}]})
    fakes = Fakes(OpenRouter(remaining=204.0))
    args = ("--env-file", str(env_file(tmp_path)), "--runs-dir", str(runs))
    code, checks, _ = run(fakes, capsys, *args, "--budget", "200")
    assert checks["balance"]["status"] == P.OK, checks["balance"]
    assert checks["balance"]["detail"] == (
        "$204.00 left (key limit); needs $203.50 (floor $10.00 + budget $200.00 - "
        "$6.50 already spent)"
    )
    assert checks["balance"]["info"]["needed_usd"] == 203.5
    # Without a shared start, the whole budget is still needed.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    code, checks, _ = run(
        fakes,
        capsys,
        "--env-file",
        str(env_file(tmp_path)),
        "--runs-dir",
        str(fresh),
        "--budget",
        "200",
    )
    assert checks["balance"]["status"] == P.FAIL and code == 1
    assert (
        "needs $210.00 (floor $10.00 + budget $200.00)" in checks["balance"]["detail"]
    )


def test_the_resources_line_warns_under_the_gates(tmp_path, capsys):
    fakes = Fakes()
    args = (
        "--offline",
        "--env-file",
        str(env_file(tmp_path)),
        "--runs-dir",
        str(tmp_path),
    )
    _, checks, _ = run(fakes, capsys, *args)
    assert checks["resources"]["status"] == P.OK
    assert checks["resources"]["detail"] == (
        f"memory 12.5 GB available of 32.0 GB, swap used 0.5 GB; disk 120.0 GB free "
        f"under {tmp_path}"
    )
    code, checks, _ = run(
        fakes, capsys, *args, "--min-free-memory-gb", "16", "--min-free-disk-gb", "200"
    )
    assert code == 0 and checks["resources"]["status"] == P.WARN
    detail = checks["resources"]["detail"]
    assert "available memory below --min-free-memory-gb 16" in detail
    assert "free disk below --min-free-disk-gb 200" in detail
    fakes.resources = {
        "mem_available_gb": None,
        "disk_free_gb": 1.0,
        "errors": ["vm_stat"],
    }
    _, checks, _ = run(fakes, capsys, *args)
    assert checks["resources"]["status"] == P.WARN
    assert "unreadable: vm_stat" in checks["resources"]["detail"]


def test_the_real_host_code_and_resources_lines(tmp_path, capsys):
    """This machine's own readers: no network, no key."""
    fakes = Fakes()
    deps = fakes.deps()
    deps.code_fingerprint = protocol.code_fingerprint
    deps.git = P._default_git
    deps.host = P._default_host
    deps.resources = P._default_resources
    P.main(
        ["--offline", "--method", "none", "--runs-dir", str(tmp_path), "--json"],
        deps=deps,
    )
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)}
    assert checks["host"]["status"] == P.OK and checks["host"]["info"]["hostname"]
    assert len(checks["host"]["info"]["machine_id"]) == 32
    assert checks["code"]["info"]["code"]["harness_files"] > 0
    assert checks["resources"]["info"]["disk_free_gb"] > 0


def test_a_host_without_a_machine_id_is_warned_about(tmp_path, capsys):
    """Without a machine id, resumes compare the hostname, which a Mac
    without a scutil HostName changes with its network."""
    fakes = Fakes()
    fakes.machine_id = None
    _code, checks, _ = run(
        fakes, capsys, "--offline", "--env-file", str(env_file(tmp_path))
    )
    assert checks["host"]["status"] == P.WARN
    assert "scutil --set HostName" in checks["host"]["detail"]
