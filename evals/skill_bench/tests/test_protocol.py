"""Frozen protocol, F9 numbers, models.json provenance and the fingerprint."""

from __future__ import annotations

import json
import shlex
import shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.tools import fetch_models

FIXTURE = Path(__file__).parent / "fixtures" / "openrouter_models_subset.json"
LUNA = "openai/gpt-6-luna"
LUNA56 = "openai/gpt-5.6-luna"
#: When the records of config/models.json were fetched: luna and the embedding model
#: first, luna56 when it was added (``fetch_models --keep-frozen`` keeps each record).
FIRST_FETCH = "2026-09-29T03:37:30+00:00"
LUNA56_FETCH = "2026-10-03T09:32:06+00:00"
#: A synthetic chat model whose top provider serves one token less than its window80, for
#: the provider-window clamp: luna's and luna56's top providers serve their whole
#: context, so the clamp never binds for them.
CAPPED = "test/provider-capped"
CAPPED_MODELS = {
    "models": {
        CAPPED: {
            "kind": "chat",
            "context_length": 1_310_720,
            "max_completion_tokens": 943_717,
            "top_provider": {"context_length": 1_048_575},
        }
    }
}


# --- F9 numbers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "window80", "max_out80", "reserve", "chunk_cap"),
    [
        (LUNA, 840_000, 102_400, 102_400, 737_600),
        (LUNA56, 840_000, 102_400, 102_400, 737_600),
    ],
)
def test_f9_numbers_match_the_table(model, window80, max_out80, reserve, chunk_cap):
    limits = protocol.model_limits(model)
    assert (
        limits.window80,
        limits.max_out80,
        limits.planning_reserve,
        limits.distill_chunk_input_cap,
    ) == (
        window80,
        max_out80,
        reserve,
        chunk_cap,
    )
    stored = protocol.load_models()["models"][model]["f9"]
    assert stored == limits.f9()
    assert stored["gateway_input_budget"] == limits.max_input_for_output(
        protocol.MIN_OUTPUT_TOKENS
    )


def test_output_cap_and_input_budget_formulas():
    capped = protocol.model_limits(CAPPED, CAPPED_MODELS)
    assert capped.output_cap(10_000) == 754_973  # max-output bound
    # Top-provider window with the tokenizer margin: 1,048,575 - ceil(1.1 i) - 1.
    assert capped.output_cap(400_000) == 1_048_575 - 440_000 - 1
    assert capped.output_cap_bound(400_000) == "provider_window"
    assert capped.f9_output_cap(400_000) == 1_048_576 - 400_000  # F9 alone
    assert capped.output_cap(capped.gateway_input_budget) >= protocol.MIN_OUTPUT_TOKENS
    assert (
        capped.output_cap(capped.gateway_input_budget + 1) < protocol.MIN_OUTPUT_TOKENS
    )
    assert capped.input_budget(system_tokens=4_000) == 1_048_576 - 4_000 - 131_072
    assert capped.input_budget(system_tokens=0, reserve=0) == capped.window80
    # exact integer arithmetic (floats would give 1310720 * 0.8 = 1048576.0000000001)
    odd = protocol.ModelLimits("m", 1_310_721, 943_718)
    assert (odd.window80, odd.max_out80, odd.planning_reserve) == (
        1_048_576,
        754_974,
        131_072,
    )


@pytest.mark.parametrize(
    "estimate", [0, 1, 266_909, 266_910, 293_603, 500_000, 834_092, 917_504, 952_318]
)
def test_cap_stays_below_the_top_providers_window_with_the_margin(estimate):
    """Even if o200k undercounts the provider's tokenizer by 10%, input + max_tokens
    fits the top provider's window."""
    capped = protocol.model_limits(CAPPED, CAPPED_MODELS)
    assert (capped.window80, capped.provider_context_length) == (1_048_576, 1_048_575)
    assert capped.tokenizer_margin == protocol.TOKENIZER_MARGIN == Fraction(1, 10)
    cap = capped.output_cap(estimate)
    worst_input = -(-estimate * 11 // 10)  # ceil(1.1 x estimate): what it may count
    assert worst_input + cap < capped.provider_context_length
    assert cap <= capped.f9_output_cap(estimate)  # never above the F9 cap
    if estimate <= 266_909:  # up to here max_out80 still binds
        assert cap == 754_973 and capped.output_cap_bound(estimate) == "max_out80"
    else:
        assert cap == 1_048_575 - worst_input - 1
    assert capped.gateway_input_budget == 952_318
    # The largest input that keeps the planning reserve of output room.
    assert capped.max_input_for_output(capped.planning_reserve) == 834_092
    assert (
        capped.output_cap(834_092)
        >= capped.planning_reserve
        > capped.output_cap(834_093)
    )


def test_luna_is_not_affected_by_the_margin():
    """luna's top provider serves its whole context, far above window80."""
    luna = protocol.model_limits(LUNA)
    assert luna.input_ceiling == luna.window80 == 840_000
    for estimate in (0, 500_000, 737_600, 800_000, luna.gateway_input_budget):
        assert luna.output_cap(estimate) == luna.f9_output_cap(estimate)
    assert luna.output_cap(800_000) == 40_000
    assert luna.gateway_input_budget == 840_000 - protocol.MIN_OUTPUT_TOKENS
    no_margin = protocol.model_limits(
        CAPPED, CAPPED_MODELS, tokenizer_margin=Fraction(0)
    )
    assert no_margin.output_cap(400_000) == 1_048_575 - 400_000 - 1


def test_model_limits_rejects_non_chat_models():
    with pytest.raises(KeyError):
        protocol.model_limits(protocol.EMBEDDING_MODEL)
    with pytest.raises(KeyError):
        protocol.model_limits("openai/gpt-4o")


# --- models.json provenance ----------------------------------------------------------------


def test_models_json_records_source_and_fetch_time():
    config = protocol.load_models()
    datetime.fromisoformat(config["fetched_at"])
    assert config["sources"] == {
        "chat": "https://openrouter.ai/api/v1/models",
        "embeddings": "https://openrouter.ai/api/v1/embeddings/models",
    }
    assert set(config["models"]) == {LUNA, LUNA56, protocol.EMBEDDING_MODEL}
    # Each record says when it was fetched; the file's fetched_at is the latest fetch.
    assert config["fetched_at"] == LUNA56_FETCH
    assert {m: e["fetched_at"] for m, e in config["models"].items()} == {
        LUNA: FIRST_FETCH,
        LUNA56: LUNA56_FETCH,
        protocol.EMBEDDING_MODEL: FIRST_FETCH,
    }
    for model in (LUNA, LUNA56):
        entry = config["models"][model]
        assert (
            entry["kind"] == "chat" and entry["source_url"] == config["sources"]["chat"]
        )
        assert isinstance(entry["context_length"], int) and isinstance(
            entry["max_completion_tokens"], int
        )
        assert (
            "image" in entry["input_modalities"] and "text" in entry["input_modalities"]
        )
        assert {"prompt", "completion"} <= set(entry["pricing"])
        assert "seed" in entry["supported_parameters"]
    embedding = config["models"][protocol.EMBEDDING_MODEL]
    assert embedding["kind"] == "embedding" and embedding["f9"] is None
    assert embedding["source_url"] == config["sources"]["embeddings"]
    assert (
        config["models"][LUNA]["context_length"],
        config["models"][LUNA]["max_completion_tokens"],
    ) == (
        1_050_000,
        128_000,
    )


def test_the_models_under_test():
    """luna is ``openai/gpt-6-luna``, luna56 ``openai/gpt-5.6-luna``. Model ids only,
    never an alias OpenRouter re-points at each release (``~openai/gpt-luna-latest``)."""
    models = {"luna": LUNA, "luna56": LUNA56}
    assert dict(protocol.MODELS) == models
    assert LUNA == "openai/gpt-6-luna"
    assert protocol.EMBEDDING_MODEL == "google/gemini-embedding-001"
    assert protocol.protocol_constants()["models"] == models
    ids = (*protocol.CHAT_MODELS, protocol.EMBEDDING_MODEL)
    assert not any(model.startswith("~") for model in ids)
    config = protocol.load_models()
    luna = config["models"][LUNA]
    assert luna["alias"] == "luna" and luna["name"] == "OpenAI: GPT-6 Luna"
    assert luna["canonical_slug"] == "openai/gpt-6-luna-20260922"
    assert sorted(luna["input_modalities"]) == ["file", "image", "text"]
    assert (
        luna["context_length"],
        luna["top_provider"]["context_length"],
        luna["max_completion_tokens"],
    ) == (1_050_000, 1_050_000, 128_000)
    # OpenRouter's list prices as the record keeps them, USD per token: $0.10 / $0.50
    # per million in and out, $0.01 per million cached.
    pricing = luna["pricing"]
    assert (pricing["prompt"], pricing["completion"], pricing["input_cache_read"]) == (
        "0.0000001",
        "0.0000005",
        "0.00000001",
    )
    # The F9 numbers of its window and output cap.
    assert luna["f9"] == {
        "window80": 840_000,
        "max_out80": 102_400,
        "planning_reserve": 102_400,
        "distill_chunk_input_cap": 737_600,
        "input_ceiling": 840_000,
        "gateway_input_budget": 838_976,
    }


def test_luna56_is_a_model_under_test_of_its_own():
    """``luna56`` is ``openai/gpt-5.6-luna``, a model of its own. Its record takes the
    reasoning effort and tools the gateway and the methods send, lists ``medium``, and
    has luna's window and output cap, so its F9 numbers are luna's."""
    config = protocol.load_models()
    entry = config["models"][LUNA56]
    assert entry["alias"] == "luna56" and entry["kind"] == "chat"
    assert entry["name"] == "OpenAI: GPT-5.6 Luna"
    assert entry["canonical_slug"] == "openai/gpt-5.6-luna-20260709"
    assert {"reasoning_effort", "tools", "tool_choice", "seed"} <= set(
        entry["supported_parameters"]
    )
    assert protocol.REASONING_EFFORT in entry["reasoning"]["supported_efforts"]
    assert entry["reasoning"]["default_effort"] == "medium"
    assert sorted(entry["input_modalities"]) == ["file", "image", "text"]
    assert (
        entry["context_length"],
        entry["top_provider"]["context_length"],
        entry["max_completion_tokens"],
    ) == (1_050_000, 1_050_000, 128_000)
    # $0.20 / $1.20 per million tokens in and out (luna: $0.10 / $0.50).
    assert (entry["pricing"]["prompt"], entry["pricing"]["completion"]) == (
        "0.0000002",
        "0.0000012",
    )
    assert entry["f9"] == config["models"][LUNA]["f9"]
    limits = protocol.model_limits(LUNA56)
    assert limits.f9() == entry["f9"]
    assert limits.output_cap(limits.gateway_input_budget) >= protocol.MIN_OUTPUT_TOKENS
    # No other model's record changed when luna56 was added (fetch_models
    # --keep-frozen): luna's is its first record, raw hash included.
    assert config["models"][LUNA]["raw_sha256"] == (
        "d1668d002fc09c5c6c96efd0e688d6a0c6bd9ad8ea4c5a4da2ae342aff0d659b"
    )


def test_models_json_is_exactly_what_the_script_generates():
    """Every record is what fetch_models builds from the fixture's payload entry, at the
    time the record says it was fetched (the fixture holds each entry as fetched)."""
    fixture = json.loads(FIXTURE.read_text())
    committed = protocol.load_models()
    times = {e["fetched_at"] for e in committed["models"].values()}
    assert times == {FIRST_FETCH, LUNA56_FETCH}
    built = {
        at: fetch_models.build_models_config(
            fixture["chat"], fixture["embeddings"], fetched_at=at
        )
        for at in times
    }
    for model, entry in committed["models"].items():
        assert entry == built[entry["fetched_at"]]["models"][model], model
    generated = built[committed["fetched_at"]]
    assert {k: v for k, v in generated.items() if k != "models"} == {
        k: v for k, v in committed.items() if k != "models"
    }
    assert list(generated["models"]) == list(committed["models"])


def test_merge_keeps_the_frozen_records_and_adds_new_models():
    """``fetch_models --keep-frozen``: a model the frozen file lacks takes its live
    record; every record it has is kept as fetched (its own fetch time, else the file's),
    with its F9 numbers recomputed under this protocol's rules; the notes follow the
    records; a renamed key is refused."""
    fixture = json.loads(FIXTURE.read_text())
    frozen = fetch_models.build_models_config(
        fixture["chat"], fixture["embeddings"], fetched_at="2026-09-29T00:00:00+00:00"
    )
    # A file written before records carried their fetch time, without luna56.
    for entry in frozen["models"].values():
        del entry["fetched_at"]
    del frozen["models"][LUNA56]
    frozen["models"][LUNA]["f9"] = {"stale": 1}
    live_fixture = json.loads(FIXTURE.read_text())
    luna = next(e for e in live_fixture["chat"]["data"] if e["id"] == LUNA)
    luna["context_length"] = 1_100_000  # above its top provider's: a note when live
    live = fetch_models.build_models_config(
        live_fixture["chat"],
        fixture["embeddings"],
        fetched_at="2026-10-03T00:00:00+00:00",
    )
    merged, kept, fetched = fetch_models.merge_models_config(frozen, live)
    assert kept == [LUNA, protocol.EMBEDDING_MODEL] and fetched == [LUNA56]
    assert list(merged["models"]) == list(live["models"])
    assert merged["fetched_at"] == "2026-10-03T00:00:00+00:00"
    assert merged["models"][LUNA56] == live["models"][LUNA56]
    kept_luna = merged["models"][LUNA]
    assert kept_luna["fetched_at"] == frozen["fetched_at"]
    assert (
        kept_luna["context_length"] == 1_050_000
    )  # the frozen value, not the live one
    assert kept_luna["f9"] == protocol.model_limits(LUNA).f9()  # recomputed
    assert list(kept_luna)[:3] == ["kind", "source_url", "fetched_at"]
    # luna's notes, from its kept record (none), not from the live one.
    assert merged["notes"] == frozen["notes"] != live["notes"]
    # The kept luna is drift against the live metadata, as before: reported, not hidden.
    relevant, _ = fetch_models.compare_models_config(merged, live)
    assert f"models.{LUNA}.context_length" in relevant
    renamed = {**frozen, "models": {**frozen["models"]}}
    renamed["models"][LUNA] = {**frozen["models"][LUNA], "alias": "old"}
    with pytest.raises(ValueError, match="alias"):
        fetch_models.merge_models_config(renamed, live)


def test_a_records_fetch_time_is_no_drift():
    committed = protocol.load_models()
    live = json.loads(json.dumps(committed))
    live["fetched_at"] = "2027-01-01T00:00:00+00:00"
    for entry in live["models"].values():
        entry["fetched_at"] = "2027-01-01T00:00:00+00:00"
    assert fetch_models.compare_models_config(committed, live) == ([], [])


def test_fetch_models_keep_frozen_writes_only_new_models(tmp_path, capsys):
    fixture = json.loads(FIXTURE.read_text())
    chat, emb = tmp_path / "chat.json", tmp_path / "emb.json"
    chat.write_text(json.dumps(fixture["chat"]))
    emb.write_text(json.dumps(fixture["embeddings"]))
    out = tmp_path / "models.json"
    frozen = protocol.load_models()
    trimmed = {**frozen, "models": dict(frozen["models"])}
    del trimmed["models"][LUNA56]
    out.write_text(json.dumps(trimmed))
    args = ["--chat-json", str(chat), "--embeddings-json", str(emb), "--out", str(out)]
    assert fetch_models.main([*args, "--keep-frozen"]) == 0
    written = json.loads(out.read_text())
    printed = capsys.readouterr().out
    assert f"fetched now: {LUNA56}" in printed
    for model in (LUNA, protocol.EMBEDDING_MODEL):
        assert written["models"][model] == frozen["models"][model]
    assert (
        written["models"][LUNA56]["raw_sha256"]
        == (frozen["models"][LUNA56]["raw_sha256"])
    )
    with pytest.raises(SystemExit):
        fetch_models.main([*args, "--keep-frozen", "--check"])
    capsys.readouterr()


def test_build_models_config_flags_metadata_the_protocol_relies_on():
    fixture = json.loads(FIXTURE.read_text())
    # luna's record with two facts the protocol cannot rely on: a smaller top provider
    # window and no 'medium' reasoning effort.
    luna = next(e for e in fixture["chat"]["data"] if e["id"] == LUNA)
    luna["top_provider"]["context_length"] = 1_048_575
    luna["reasoning"]["supported_efforts"] = ["max", "high", "low"]
    config = fetch_models.build_models_config(
        fixture["chat"], fixture["embeddings"], fetched_at="t"
    )
    notes = "\n".join(config["notes"])
    assert (
        "top_provider.context_length 1048575" in notes
    )  # top provider window < context_length
    assert "does not list 'medium'" in notes  # reasoning efforts


def test_build_models_config_requires_every_model():
    fixture = json.loads(FIXTURE.read_text())
    chat = {"data": [m for m in fixture["chat"]["data"] if m["id"] != LUNA56]}
    with pytest.raises(KeyError, match="gpt-5.6-luna"):
        fetch_models.build_models_config(chat, fixture["embeddings"], fetched_at="t")
    with pytest.raises(KeyError, match="gemini-embedding-001"):
        fetch_models.build_models_config(fixture["chat"], {"data": []}, fetched_at="t")


def test_fetch_models_check_mode_detects_drift(tmp_path, capsys):
    fixture = json.loads(FIXTURE.read_text())
    chat, emb = tmp_path / "chat.json", tmp_path / "emb.json"
    chat.write_text(json.dumps(fixture["chat"]))
    emb.write_text(json.dumps(fixture["embeddings"]))
    args = [
        "--chat-json",
        str(chat),
        "--embeddings-json",
        str(emb),
        "--out",
        str(protocol.MODELS_JSON),
    ]
    assert fetch_models.main([*args, "--check"]) == 0
    fixture["chat"]["data"][0]["context_length"] += 1
    chat.write_text(json.dumps(fixture["chat"]))
    assert fetch_models.main([*args, "--check"]) == 1
    out = tmp_path / "models.json"
    assert (
        fetch_models.main(
            ["--chat-json", str(chat), "--embeddings-json", str(emb), "--out", str(out)]
        )
        == 0
    )
    assert (
        json.loads(out.read_text())["models"][fixture["chat"]["data"][0]["id"]][
            "context_length"
        ]
        > 0
    )
    capsys.readouterr()


# --- constants ------------------------------------------------------------------------------


def test_protocol_constants():
    assert protocol.SEED == 42
    assert dict(protocol.SPLIT_SIZES["alfworld"]) == {
        "train": 39,
        "val": 18,
        "test": 134,
    }
    # WebShop 40/20/140; a 15-step cap, which the diagnostic's 15
    # equals.
    assert dict(protocol.SPLIT_SIZES["webshop"]) == {
        "train": 40,
        "val": 20,
        "test": 140,
    }
    assert protocol.WEBSHOP_MAX_STEPS == 15 and protocol.WEBSHOP_DIAGNOSTIC_STEPS == 15

    assert protocol.WEBSHOP_GOAL_COUNT == 12087
    assert protocol.WEBSHOP_SERVER["seed"] == protocol.SEED
    assert protocol.REASONING_EFFORT == "medium" and protocol.CALL_DEADLINE_S == 600
    # Every framework client waits longer than the gateway's deadline.
    assert protocol.CLIENT_TIMEOUT_S > protocol.CALL_DEADLINE_S
    assert float(protocol.MIRIX_SERVER_ENV["MIRIX_LLM_TIMEOUT_SECONDS"]) == (
        protocol.CLIENT_TIMEOUT_S
    )
    from evals.skill_bench.agent import actor

    assert (
        actor.CLIENT_TIMEOUT_S == protocol.CLIENT_TIMEOUT_S > protocol.CALL_DEADLINE_S
    )
    assert protocol.MIRIX_SERVER_ENV["MIRIX_LLM_MAX_RETRIES"] == "0"
    assert protocol.ALFWORLD_MAX_STEPS == 50
    # 2 x 2 x 4 = 16 cells: ALFWorld and WebShop, luna and luna56, four methods.
    assert len(protocol.BENCHMARKS) * len(protocol.MODELS) * len(protocol.METHODS) == 16
    json.dumps(protocol.protocol_constants())  # JSON-stable


# --- fingerprint ------------------------------------------------------------------------------


def _json(value: object) -> object:
    """``value`` as a JSON record reads it back (tuples become lists)."""
    return json.loads(json.dumps(value))


def _mini_tree(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    shutil.copy(protocol.MODELS_JSON, root / "config" / "models.json")
    files = {
        "envs/templates/alfworld/actor_step.txt": "Your admissible actions are: {actions}\n",
        "envs/splits/webshop/test.json": '["q1", "q2"]\n',
        "methods/prompts/reasoningbank/extract_success.txt": "You are an expert AI agent analyst.\n",
        "agent/templates/webshop/rollout_system.md": "{skill_section}\n",
        "envs/alfworld.py": "X = 1\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_fingerprint_covers_prompts_templates_splits_and_models(tmp_path):
    root = _mini_tree(tmp_path / "sb")
    covered = [p.relative_to(root).as_posix() for p in protocol.fingerprint_files(root)]
    assert covered == [
        "agent/templates/webshop/rollout_system.md",
        "config/models.json",
        "envs/splits/webshop/test.json",
        "envs/templates/alfworld/actor_step.txt",
        "methods/prompts/reasoningbank/extract_success.txt",
    ]
    baseline = protocol.protocol_fingerprint(root)
    assert protocol.protocol_fingerprint(root) == baseline  # deterministic
    for rel in covered:
        path = root / rel
        original = path.read_bytes()
        path.write_bytes(
            original[:-1] + bytes([original[-1] ^ 0x01])
        )  # flip one bit of one byte
        assert protocol.protocol_fingerprint(root) != baseline, rel
        path.write_bytes(original)
        assert protocol.protocol_fingerprint(root) == baseline


def test_fingerprint_tracks_new_and_renamed_files_but_not_others(tmp_path):
    root = _mini_tree(tmp_path / "sb")
    baseline = protocol.protocol_fingerprint(root)
    new = root / "methods" / "prompts" / "awm" / "induce.txt"
    new.parent.mkdir(parents=True)
    new.write_text("induce\n")
    assert protocol.protocol_fingerprint(root) != baseline
    new.unlink()
    moved = root / "envs" / "splits" / "webshop" / "test.json"
    moved.rename(moved.with_name("test2.json"))
    assert protocol.protocol_fingerprint(root) != baseline
    moved.with_name("test2.json").rename(moved)
    assert protocol.protocol_fingerprint(root) == baseline
    for rel in (
        "envs/alfworld.py",
        "runs/cell/prompts/copy.txt",
        "data/splits/webshop/test.json",
        "tests/prompts/fixture.txt",
        "envs/templates/.DS_Store",
        "envs/templates/__pycache__/x.pyc",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
    assert protocol.protocol_fingerprint(root) == baseline


def test_fingerprint_changes_with_any_constant(tmp_path, monkeypatch):
    root = _mini_tree(tmp_path / "sb")
    baseline = protocol.protocol_fingerprint(root)
    monkeypatch.setattr(protocol, "ALFWORLD_MAX_STEPS", 51)
    assert protocol.protocol_fingerprint(root) != baseline
    monkeypatch.undo()
    monkeypatch.setattr(
        protocol,
        "SPLIT_SIZES",
        {**protocol.SPLIT_SIZES, "webshop": {"train": 1, "val": 1, "test": 1}},
    )
    assert protocol.protocol_fingerprint(root) != baseline


def test_verify_fingerprint_and_missing_models_json(tmp_path):
    root = _mini_tree(tmp_path / "sb")
    frozen = protocol.protocol_fingerprint(root)
    protocol.verify_fingerprint(frozen, root)
    (root / "envs" / "templates" / "alfworld" / "actor_step.txt").write_text(
        "drifted\n"
    )
    with pytest.raises(protocol.ProtocolDriftError):
        protocol.verify_fingerprint(frozen, root)
    (root / "config" / "models.json").unlink()
    with pytest.raises(FileNotFoundError):
        protocol.protocol_fingerprint(root)


def test_real_tree_fingerprint_manifest():
    manifest = protocol.fingerprint_manifest()
    assert "config/models.json" in manifest["files"]
    assert not any(
        p.startswith(("tests/", "runs/", "data/")) for p in manifest["files"]
    )
    assert manifest["fingerprint"] == protocol.protocol_fingerprint()


def test_protocol_asset_only_serves_fingerprinted_files(tmp_path):
    root = _mini_tree(tmp_path / "sb")
    rel = "envs/templates/alfworld/actor_step.txt"
    assert protocol.protocol_asset(rel, root) == root / rel
    assert protocol.read_protocol_text(rel, root).startswith("Your admissible actions")
    assert protocol.protocol_asset("config/models.json", root).is_file()
    for bad in (
        "envs/alfworld.py",
        "tests/prompts/x.txt",
        "runs/prompts/x.txt",
        "data/splits/x.json",
        "envs/templates/.hidden",
        "envs/templates/../alfworld.py",
        "../prompts/x.txt",
        "prompts",
        str(root / rel),
    ):
        with pytest.raises(ValueError):
            protocol.protocol_asset(bad, root)
    covered = {p.relative_to(root).as_posix() for p in protocol.fingerprint_files(root)}
    assert all(protocol.is_fingerprinted(c) for c in covered)


def test_fingerprint_covers_vendored_scoring_code_and_tool_texts(tmp_path, monkeypatch):
    """Scoring code, model-visible texts, limits and estimate constants are protocol."""
    root = tmp_path / "sb"
    shutil.copytree(
        protocol.SKILL_BENCH_DIR,
        root,
        ignore=shutil.ignore_patterns("__pycache__", "runs", "data", "tests"),
    )
    baseline = protocol.protocol_fingerprint(root)
    for rel in (
        "envs/vendor/alfworld_projection.py",
        "envs/vendor/webshop_projection.py",
        "envs/templates/webshop/format.json",
    ):
        assert protocol.is_fingerprinted(rel), rel
        assert protocol.FROZEN_ASSET_SHA256[rel]
        path = root / rel
        original = path.read_bytes()
        path.write_bytes(original.replace(b"4000", b"8000", 1) + b" ")
        assert protocol.protocol_fingerprint(root) != baseline, rel
        with pytest.raises(protocol.ProtocolDriftError):
            protocol.verify_frozen_assets(rel, root)
        path.write_bytes(original)
    assert protocol.protocol_fingerprint(root) == baseline
    for name, value in (
        ("TOKEN_ESTIMATE", {**protocol.TOKEN_ESTIMATE, "message_overhead_tokens": 4}),
        ("WEBSHOP_MAX_STEPS", protocol.WEBSHOP_MAX_STEPS + 1),
        ("CHAT_FORWARDED_FIELDS", (*protocol.CHAT_FORWARDED_FIELDS, "n")),
        ("MODEL_TIMEOUT_MAX_OVERHEAD_S", 30.0),
    ):
        monkeypatch.setattr(protocol, name, value)
        assert protocol.protocol_fingerprint(root) != baseline, name
        monkeypatch.undo()


def test_notice_names_every_vendored_file_and_its_source():
    """The third-party NOTICE names every vendored file and its source."""
    envs = protocol.SKILL_BENCH_DIR / "envs"
    notice = (envs / "vendor" / "NOTICE").read_text(encoding="utf-8")
    vendored = sorted(
        p.name for p in (envs / "vendor").glob("*.py") if p.stem != "__init__"
    )
    assert vendored == [
        "alfworld_projection.py",
        "webshop_projection.py",
    ]
    for name in vendored:
        assert f"envs/vendor/{name}" in notice, name
    # WebShop: verl-agent's projection and prompt, upstream's server.
    assert "20bd331bdbc9026a5668e11362178e10ab7400c8" in notice
    assert protocol.WEBSHOP_SERVER["upstream_commit"] in notice
    assert "envs/templates/webshop/" in notice and "Princeton" in notice


# --- the frozen value (H10.6) ---------------------------------------------------------------


def _committed_record() -> dict:
    """The repository's record, read explicitly: the tests' session fixture points every
    default reader at a copy (``SKILL_BENCH_FROZEN_RECORD``, tests/conftest.py)."""
    return protocol.load_frozen_protocol(protocol.FROZEN_PROTOCOL_JSON)


def _entries(record: dict) -> list[dict]:
    """Every value a record keeps, oldest first: its history, then its current value."""
    return [*record["history"], record]


def _entry(record: dict, fingerprint: str) -> dict:
    """The first entry of ``record`` (oldest first) that recorded ``fingerprint``."""
    return next(e for e in _entries(record) if e["fingerprint"] == fingerprint)


def _check_record_describes_itself(path: Path) -> None:
    record = protocol.load_frozen_protocol(path)
    assert (
        protocol.fingerprint_of(record["constants"], record["files"])
        == record["fingerprint"]
    )
    assert protocol.constants_sha256(record["constants"]) == record["constants_sha256"]
    assert record["protocol_id"] == protocol.PROTOCOL_ID and record["reason"]
    entries = _entries(record)
    for entry in entries:
        if entry.get("constants") is not None:
            assert protocol.recorded_constants(entry) is not None, entry["fingerprint"]
        assert isinstance(entry["git"], dict) and entry["git"].get("commit")
        assert entry.get("reason")
        auto = entry.get("auto_freeze")
        if auto is not None:
            assert {"cell_id", "attempt", "why"} <= set(auto), entry["fingerprint"]
    stamps = [entry["frozen_at"] for entry in entries]
    assert stamps == sorted(stamps)
    known = protocol.known_versions(path)
    for before, after in zip(entries, entries[1:]):
        if before.get("files") is None or after.get("files") is None:
            continue
        diff = protocol.version_diff(before["fingerprint"], after["fingerprint"], known)
        assert isinstance(protocol.unauditable_changes(diff), list)


def test_the_frozen_record_describes_itself():
    """The committed record is self-consistent: its per-key constants and file map
    reproduce its fingerprint and constants hash, every entry that keeps constants
    reproduces its own, the entries are in freeze order, each names its commit and why
    (an automatic freeze its cell, attempt and why), and every adjacent pair of versions
    with file maps can be compared. It does not require the tree to be the frozen value:
    a tree that moved on is frozen by the next launch, so this test holds after any later
    freeze, by hand or automatic."""
    _check_record_describes_itself(protocol.FROZEN_PROTOCOL_JSON)


def test_the_record_tests_hold_after_a_later_freeze(tmp_path, monkeypatch):
    """The record tests describe the record, not the current value. Freezing a copy of
    the committed record to a changed tree (a constant every cell runs under) keeps them
    true of that copy too, and the replaced value joins its history."""
    from evals.skill_bench.tools import freeze_protocol

    copy = tmp_path / "protocol_frozen.json"
    copy.write_bytes(protocol.FROZEN_PROTOCOL_JSON.read_bytes())
    monkeypatch.setattr(
        protocol, "BOOTSTRAP_RESAMPLES", protocol.BOOTSTRAP_RESAMPLES + 1
    )
    freeze_protocol.freeze(
        reason="a later freeze",
        path=copy,
        git=lambda: {"commit": "c0ffee0", "dirty": False},
    )
    committed = _committed_record()
    later = protocol.load_frozen_protocol(copy)
    assert later["history"][-1]["fingerprint"] == committed["fingerprint"]
    assert later["reason"] == "a later freeze"
    _check_record_describes_itself(copy)
    before = [entry["fingerprint"] for entry in _entries(committed)]
    after = [entry["fingerprint"] for entry in _entries(later)]
    assert after[: len(before)] == before


def test_the_frozen_record_is_not_fingerprinted():
    fingerprinted = protocol.fingerprint_manifest()["files"]
    rel = protocol.FROZEN_PROTOCOL_JSON.relative_to(protocol.SKILL_BENCH_DIR).as_posix()
    assert rel not in fingerprinted


def test_every_fingerprinted_file_is_committed():
    """A fingerprint over files git does not track could not be reproduced from a checkout."""
    import subprocess

    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "."],
            cwd=protocol.SKILL_BENCH_DIR,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    tracked_files = set(filter(None, tracked.split("\0")))
    missing = sorted(set(protocol.fingerprint_manifest()["files"]) - tracked_files)
    assert not missing, f"fingerprinted but not tracked by git: {missing}"


def test_freeze_tool_writes_checks_and_refuses_an_unexplained_refreeze(
    tmp_path, capsys
):
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    with pytest.raises(protocol.ProtocolDriftError, match="not been frozen"):
        protocol.frozen_fingerprint(out)
    assert (
        freeze_protocol.main(["--root", str(root), "--out", str(out), "--check"]) == 1
    )
    assert "not been frozen" in capsys.readouterr().err

    record = freeze_protocol.freeze(
        reason=None,
        root=root,
        path=out,
        now=lambda: "2026-09-28T00:00:00+00:00",
        git=lambda: {"commit": "abc", "dirty": False},
    )
    first = protocol.protocol_fingerprint(root)
    assert record["fingerprint"] == first == protocol.frozen_fingerprint(out)
    assert record["reason"] == "initial freeze" and record["history"] == []
    assert record["git"] == {"commit": "abc", "dirty": False}
    # The constants per key, next to their hash.
    assert record["constants"] == _json(protocol.protocol_constants())
    assert protocol.constants_sha256(record["constants"]) == record["constants_sha256"]
    assert set(record["files"]) == {
        p.relative_to(root).as_posix() for p in protocol.fingerprint_files(root)
    }
    assert (
        freeze_protocol.main(["--root", str(root), "--out", str(out), "--check"]) == 0
    )
    assert "equals the frozen value" in capsys.readouterr().out
    # Freezing an unchanged tree again changes nothing, reason or not.
    before = out.read_bytes()
    assert freeze_protocol.freeze(reason="again", root=root, path=out) == record
    assert out.read_bytes() == before

    # A prompt changes: the check names it, and a re-freeze needs a reason.
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("You are an expert analyst.\n")
    new = root / "envs" / "splits" / "webshop" / "val.json"
    new.write_text('["q3"]\n')
    drift = freeze_protocol.check(root, out)
    assert drift["drifted"] and not drift["constants_changed"]
    assert drift["changed"] == ["methods/prompts/reasoningbank/extract_success.txt"]
    assert drift["added"] == ["envs/splits/webshop/val.json"] and drift["removed"] == []
    message = protocol.describe_drift(drift)
    assert "extract_success.txt" in message and "--reason" in message
    with pytest.raises(protocol.ProtocolDriftError, match="extract_success"):
        protocol.verify_frozen_protocol(root, out)
    assert (
        freeze_protocol.main(["--root", str(root), "--out", str(out), "--check"]) == 1
    )
    assert "changed: methods/prompts" in capsys.readouterr().out
    assert freeze_protocol.main(["--root", str(root), "--out", str(out)]) == 2
    assert "needs --reason" in capsys.readouterr().err
    assert protocol.frozen_fingerprint(out) == first

    argv = ["--root", str(root), "--out", str(out), "--reason", "tighter RB prompt"]
    assert freeze_protocol.main(argv) == 0
    capsys.readouterr()
    refrozen = protocol.load_frozen_protocol(out)
    assert refrozen["fingerprint"] == protocol.protocol_fingerprint(root) != first
    assert refrozen["reason"] == "tighter RB prompt"
    assert [h["fingerprint"] for h in refrozen["history"]] == [first]
    assert refrozen["history"][0]["reason"] == "initial freeze"
    assert protocol.verify_frozen_protocol(root, out) == refrozen["fingerprint"]
    # The replaced value keeps its constants (per key and hash) and file map, so cells
    # run under it stay a known version whose changed files and constant keys can be
    # named.
    assert refrozen["history"][0] == {
        key: record[key]
        for key in (
            "fingerprint",
            "constants_sha256",
            "constants",
            "files",
            "frozen_at",
            "git",
            "reason",
        )
    }
    assert tuple(refrozen["history"][0]) == tuple(sorted(freeze_protocol.HISTORY_KEYS))


def test_known_versions_read_the_frozen_record_as_it_is():
    """Every value the frozen record names is a known version, the frozen one
    ``current``. Holds for the record as it is (older history entries may carry no file
    map) and after any re-freeze (the replaced value keeps its map)."""
    frozen = _committed_record()
    known = protocol.known_versions(protocol.FROZEN_PROTOCOL_JSON)
    history = [e["fingerprint"] for e in frozen["history"]]
    assert set(known) == {frozen["fingerprint"], *history}
    current = known[frozen["fingerprint"]]
    assert current["current"] is True and current["files"] == frozen["files"]
    assert current["constants_sha256"] == frozen["constants_sha256"]
    # The frozen record keeps its constants per key; read only when they reproduce its
    # fingerprint.
    assert current["constants"] == frozen["constants"]
    assert current["reason"] == frozen["reason"]
    for entry in frozen["history"]:
        if entry["fingerprint"] == frozen["fingerprint"]:
            continue  # a reverted change: the frozen value describes it
        version = known[entry["fingerprint"]]
        assert version["current"] is False
        assert (version["frozen_at"], version["reason"]) == (
            entry["frozen_at"],
            entry["reason"],
        )
        assert version["files"] == entry.get("files")
        assert version["constants"] == protocol.recorded_constants(entry)
        # Without both file maps the files that changed are not known.
        diff = protocol.version_diff(entry["fingerprint"], frozen["fingerprint"], known)
        assert (diff is None) == ("files" not in entry)
    assert protocol.version_diff(frozen["fingerprint"], "0" * 64, known) is None
    assert protocol.version_diff(None, frozen["fingerprint"], known) is None


def test_versions_and_their_differences_across_a_refreeze(tmp_path):
    """A re-freeze keeps the replaced version's file map: ``version_diff`` names the
    files that changed between two versions, and says when it cannot (an entry written
    before history kept file maps, an unknown fingerprint)."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    with pytest.raises(protocol.ProtocolDriftError, match="not been frozen"):
        protocol.known_versions(out)
    first = freeze_protocol.freeze(reason=None, root=root, path=out)
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("You are an expert analyst.\n")
    (root / "envs" / "splits" / "webshop" / "val.json").write_text('["q3"]\n')
    (root / "envs" / "splits" / "webshop" / "test.json").unlink()
    second = freeze_protocol.freeze(reason="tighter prompt", root=root, path=out)
    known = protocol.known_versions(out)
    a, b = first["fingerprint"], second["fingerprint"]
    assert list(known) == [a, b]
    assert (known[a]["current"], known[b]["current"]) == (False, True)
    assert known[a]["files"] == first["files"] and known[b]["files"] == second["files"]
    assert known[a]["reason"] == "initial freeze"
    assert known[b]["reason"] == "tighter prompt"
    diff = protocol.version_diff(a, b, known)
    assert diff == {
        "from": a,
        "to": b,
        "constants_changed": False,
        "constants_changed_keys": [],
        "inconsistent": [],
        "changed": ["methods/prompts/reasoningbank/extract_success.txt"],
        "added": ["envs/splits/webshop/val.json"],
        "removed": ["envs/splits/webshop/test.json"],
        "constants_scope": {},
        "files_scope": {
            "methods/prompts/reasoningbank/extract_success.txt": None,
            "envs/splits/webshop/val.json": "webshop",
            "envs/splits/webshop/test.json": "webshop",
        },
    }
    text = protocol.describe_version_diff(diff)
    assert "changed: methods/prompts/reasoningbank/extract_success.txt" in text
    assert "added: envs/splits/webshop/val.json" in text
    assert "removed: envs/splits/webshop/test.json" in text
    # The audit checks cells of the replaced value against this tree's constants,
    # templates and splits; a method's prompts it never reads.
    assert protocol.unauditable_changes(diff) == [
        "envs/splits/webshop/val.json",
        "envs/splits/webshop/test.json",
    ]
    assert protocol.unauditable_changes({**diff, "added": [], "removed": []}) == []
    warning = freeze_protocol.refreeze_warning(second)
    assert warning is not None and "envs/splits/webshop/val.json" in warning
    assert "protocol_version_unauditable" in warning and a[:12] in warning
    assert freeze_protocol.refreeze_warning(first) is None  # no history: no re-freeze
    assert protocol.version_diff(b, a, known)["added"] == [
        "envs/splits/webshop/test.json"
    ]
    assert protocol.version_diff(a, "0" * 64, known) is None
    # A record whose history entry predates file maps: the version is known, what
    # changed is not.
    record = json.loads(out.read_text())
    for key in ("files", "constants_sha256", "constants"):
        record["history"][0].pop(key)
    out.write_text(json.dumps(record))
    legacy = protocol.known_versions(out)
    assert legacy[a]["files"] is None and legacy[a]["current"] is False
    assert protocol.version_diff(a, b, legacy) is None
    assert "not recorded" in protocol.describe_version_diff(None)
    # A constants change alone.
    both = {
        "x": {"files": {"f": "1"}, "constants_sha256": "c1"},
        "y": {"files": {"f": "1"}, "constants_sha256": "c2"},
        "z": {"files": {"f": "1"}},
    }
    assert protocol.version_diff("x", "y", both)["constants_changed"] is True
    # Neither keeps its constants per key: which keys changed is not known.
    assert protocol.version_diff("x", "y", both)["constants_changed_keys"] is None
    assert "constants of protocol.py changed" in protocol.describe_version_diff(
        protocol.version_diff("x", "y", both)
    )
    assert protocol.version_diff("x", "z", both)["constants_changed"] is None
    assert "not recorded" in protocol.describe_version_diff(
        protocol.version_diff("x", "z", both)
    )
    assert protocol.unauditable_changes(protocol.version_diff("x", "y", both)) == [
        "constants of protocol.py"
    ]
    assert protocol.unauditable_changes(protocol.version_diff("x", "z", both)) == [
        "a constants hash that is not recorded"
    ]
    assert protocol.unauditable_changes(None) == ["a file map that is not recorded"]


def test_a_method_setup_template_change_leaves_every_cell_auditable():
    """``methods/templates/mirix/setup.json`` is read by MIRIX's method code alone and
    the audit rebuilds nothing from it, like the method prompts. A change there (MIRIX
    development on train/val, say ``retrieve_limit``) names no cell, every baseline's
    included. An environment template still is named."""
    files = {
        "methods/templates/mirix/setup.json": "1",
        "methods/prompts/awm/format.json": "p",
        "envs/templates/alfworld/format.json": "a",
    }
    versions = {
        "x": {"files": dict(files), "constants_sha256": "c"},
        "y": {
            "files": {**files, "methods/templates/mirix/setup.json": "2"},
            "constants_sha256": "c",
        },
        "z": {
            "files": {**files, "envs/templates/alfworld/format.json": "b"},
            "constants_sha256": "c",
        },
    }
    diff = protocol.version_diff("x", "y", versions)
    assert diff["changed"] == ["methods/templates/mirix/setup.json"]
    assert diff["files_scope"] == {"methods/templates/mirix/setup.json": None}
    assert protocol.unauditable_changes(diff) == []
    for benchmark in protocol.BENCHMARKS:
        assert protocol.unauditable_changes(diff, benchmark) == [], benchmark
    other = protocol.version_diff("x", "z", versions)
    assert protocol.unauditable_changes(other) == [
        "envs/templates/alfworld/format.json"
    ]
    assert protocol.unauditable_changes(other, "webshop") == []
    # The exemption covers exactly the files only a method's code reads: every
    # fingerprinted file under methods/ is a method prompt or setup template, and the
    # audit reads none of them.
    method_files = [
        p.relative_to(protocol.SKILL_BENCH_DIR).as_posix()
        for p in protocol.fingerprint_files()
        if p.relative_to(protocol.SKILL_BENCH_DIR).parts[0] == "methods"
    ]
    assert "methods/templates/mirix/setup.json" in method_files
    assert all(f.startswith(protocol.METHOD_FILE_PREFIXES) for f in method_files)
    audit_source = (protocol.SKILL_BENCH_DIR / "audit.py").read_text()
    assert "methods/templates" not in audit_source
    assert "setup_template" not in audit_source


def test_a_refreeze_of_constants_or_shared_files_warns(tmp_path, capsys):
    """A re-freeze that changes what the audit checks cells of the replaced value
    against from the new tree (constants, templates, splits) says so; one of a method's
    prompts alone does not."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    assert freeze_protocol.main(["--root", str(root), "--out", str(out)]) == 0
    capsys.readouterr()
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("A method prompt only.\n")
    argv = ["--root", str(root), "--out", str(out), "--reason", "prompt"]
    assert freeze_protocol.main(argv) == 0
    assert "warning" not in capsys.readouterr().err
    (root / "envs" / "splits" / "webshop" / "val.json").write_text('["q9"]\n')
    argv = ["--root", str(root), "--out", str(out), "--reason", "a split"]
    assert freeze_protocol.main(argv) == 0
    err = capsys.readouterr().err
    assert "warning: this re-freeze changes envs/splits/webshop/val.json" in err
    # Where to audit the replaced value's cells from: not the commit it was frozen from,
    # the one that recorded it.
    replaced = protocol.load_frozen_protocol(out)["history"][-1]
    assert (
        "audit them from "
        + protocol.version_checkout_hint(replaced["fingerprint"], replaced["git"])
        in err
    )
    # An unchanged tree: no re-freeze, no warning.
    assert freeze_protocol.main(argv) == 0
    assert "warning" not in capsys.readouterr().err


# --- the constants per key -------------------------------------------------------------


def _strip_constants(out: Path) -> dict:
    """``out`` as a record frozen before it kept its constants per key; returns it."""
    record = json.loads(out.read_text())
    record.pop("constants")
    out.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return record


def test_the_backfill_adds_the_constants_only_to_the_tree_they_were_frozen_from(
    tmp_path, capsys
):
    """A record frozen before it kept its constants per key gets them from the tree it
    was frozen from, and nothing else changes; a tree whose fingerprint or constants
    hash differs from the record's is refused, nothing written."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    freeze_protocol.freeze(reason=None, root=root, path=out)
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("A later prompt.\n")
    freeze_protocol.freeze(reason="a prompt", root=root, path=out)
    old = _strip_constants(out)
    assert protocol.known_versions(out)[old["fingerprint"]]["constants"] is None

    record, written = freeze_protocol.backfill_constants(root=root, path=out)
    assert written is True
    assert record["constants"] == _json(protocol.protocol_constants())
    # Nothing but the constants: fingerprint, file map, reason and history as they were.
    assert {k: v for k, v in record.items() if k != "constants"} == old
    assert protocol.frozen_fingerprint(out) == old["fingerprint"]
    assert protocol.verify_frozen_protocol(root, out) == old["fingerprint"]
    assert protocol.known_versions(out)[old["fingerprint"]]["constants"] == (
        _json(protocol.protocol_constants())
    )
    # Again: already recorded, nothing written.
    before = out.read_bytes()
    assert freeze_protocol.backfill_constants(root=root, path=out) == (record, False)
    assert out.read_bytes() == before
    argv = ["--root", str(root), "--out", str(out), "--backfill-constants"]
    assert freeze_protocol.main(argv) == 0
    assert "already records" in capsys.readouterr().out
    assert out.read_bytes() == before

    # The CLI backfills too, and names what it verified.
    _strip_constants(out)
    assert freeze_protocol.main(argv) == 0
    printed = capsys.readouterr().out
    assert "backfilled" in printed and old["fingerprint"] in printed
    assert old["constants_sha256"] in printed
    assert out.read_bytes() == before

    # A tree that is not the frozen value: refused, nothing written.
    _strip_constants(out)
    stripped = out.read_bytes()
    prompt.write_text("Not the frozen prompt.\n")
    with pytest.raises(ValueError, match="is not the frozen"):
        freeze_protocol.backfill_constants(root=root, path=out)
    assert freeze_protocol.main(argv) == 2
    assert "is not the frozen" in capsys.readouterr().err
    assert out.read_bytes() == stripped
    prompt.write_text("A later prompt.\n")
    # Constants that do not hash to the recorded constants_sha256 (the fingerprint
    # equal): refused.
    record = json.loads(stripped)
    out.write_text(json.dumps({**record, "constants_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the frozen constants"):
        freeze_protocol.backfill_constants(root=root, path=out)
    # A record that already keeps other constants: refused, never overwritten.
    out.write_text(json.dumps({**record, "constants": {"methods": ["none"]}}))
    with pytest.raises(ValueError, match="already records per-key constants"):
        freeze_protocol.backfill_constants(root=root, path=out)
    # No record at all, or the option with --check or --reason.
    assert (
        freeze_protocol.main(
            [
                "--root",
                str(root),
                "--out",
                str(tmp_path / "none.json"),
                "--backfill-constants",
            ]
        )
        == 2
    )
    assert "not been frozen" in capsys.readouterr().err
    for extra in (["--check"], ["--reason", "x"]):
        with pytest.raises(SystemExit) as exc:
            freeze_protocol.main([*argv, *extra])
        assert exc.value.code == 2
    assert "takes neither --check nor --reason" in capsys.readouterr().err


def test_every_freeze_keeps_the_canonical_slugs_of_each_models_json(tmp_path, capsys):
    """The record keeps, per sha256 of config/models.json, what each model is
    requested as, carried over by every re-freeze, so the audit judges the
    model-substitution check against a cell's own version after models.json changes.
    ``--backfill-model-slugs`` adds the frozen file's slugs to a record written before;
    a tree whose models.json is not the frozen one is refused."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    models = root / "config" / "models.json"
    luna = protocol.MODELS["luna"]
    first = freeze_protocol.freeze(reason=None, root=root, path=out)
    first_sha = first["files"][protocol.MODELS_JSON_REL]
    first_slugs = protocol.canonical_slugs(json.loads(models.read_text()))
    assert first_slugs[luna] and first["model_slugs"] == {first_sha: first_slugs}
    assert protocol.known_versions(out)[first["fingerprint"]]["model_slugs"] == {
        first_sha: first_slugs
    }

    data = json.loads(models.read_text())
    data["models"][luna]["canonical_slug"] = "openai/gpt-6-luna-turbo-20261001"
    models.write_text(json.dumps(data, indent=2) + "\n")
    second = freeze_protocol.freeze(reason="pin a slug", root=root, path=out)
    second_sha = second["files"][protocol.MODELS_JSON_REL]
    assert second_sha != first_sha
    assert second["model_slugs"][first_sha] == first_slugs
    assert second["model_slugs"][second_sha][luna] == "openai/gpt-6-luna-turbo-20261001"

    # A record frozen before it kept them: the backfill adds the frozen file's only.
    record = json.loads(out.read_text())
    record.pop("model_slugs")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    argv = ["--root", str(root), "--out", str(out), "--backfill-model-slugs"]
    assert freeze_protocol.main(argv) == 0
    assert "backfilled" in capsys.readouterr().out
    backfilled = json.loads(out.read_text())
    assert backfilled["model_slugs"] == {second_sha: second["model_slugs"][second_sha]}
    assert {k: v for k, v in backfilled.items() if k != "model_slugs"} == record
    assert freeze_protocol.main(argv) == 0
    assert "already records" in capsys.readouterr().out
    # The tree's models.json is not the frozen one: refused, nothing written.
    before = out.read_bytes()
    models.write_text(json.dumps({"models": {}}) + "\n")
    assert freeze_protocol.main(argv) == 2
    assert "not the frozen" in capsys.readouterr().err
    assert out.read_bytes() == before
    with pytest.raises(SystemExit):
        freeze_protocol.main([*argv, "--check"])


def test_a_refreeze_names_the_constant_keys_that_changed(tmp_path, monkeypatch, capsys):
    """The record keeps the constants per key, so a re-freeze of constants names the
    keys. A change of the study's composition (the method list) is
    auditable for the other cells; any other key is not, and the tool warns."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    first = freeze_protocol.freeze(reason=None, root=root, path=out)
    methods = tuple(m for m in protocol.METHODS if m != "awm")
    monkeypatch.setattr(protocol, "METHODS", methods)
    drift = protocol.protocol_drift(root=root, path=out)
    assert drift["constants_changed"] and drift["constants_changed_keys"] == ["methods"]
    assert "constants of protocol.py changed: methods" in protocol.describe_drift(drift)
    argv = ["--root", str(root), "--out", str(out), "--reason", "drop a method"]
    assert freeze_protocol.main(argv) == 0
    captured = capsys.readouterr()
    assert "since " + first["fingerprint"][:12] in captured.out
    assert "constants of protocol.py changed: methods" in captured.out
    assert "warning" not in captured.err  # composition only: nothing unauditable
    second = protocol.load_frozen_protocol(out)
    assert second["constants"]["methods"] == list(methods)
    assert second["history"][-1]["constants"] == first["constants"]
    known = protocol.known_versions(out)
    diff = protocol.version_diff(first["fingerprint"], second["fingerprint"], known)
    assert diff is not None
    assert (diff["constants_changed"], diff["constants_changed_keys"]) == (
        True,
        ["methods"],
    )
    assert protocol.describe_version_diff(diff) == (
        "constants of protocol.py changed: methods"
    )
    assert protocol.unauditable_changes(diff) == []

    # A constant every cell runs under: named, unauditable, and the tool warns.
    monkeypatch.setattr(protocol, "SEED", protocol.SEED + 1)
    argv = ["--root", str(root), "--out", str(out), "--reason", "another seed"]
    assert freeze_protocol.main(argv) == 0
    captured = capsys.readouterr()
    assert "constants of protocol.py changed: seed" in captured.out
    assert "warning: this re-freeze changes protocol.py constant seed" in captured.err
    third = protocol.load_frozen_protocol(out)
    known = protocol.known_versions(out)
    diff = protocol.version_diff(first["fingerprint"], third["fingerprint"], known)
    assert diff["constants_changed_keys"] == ["methods", "seed"]
    assert protocol.unauditable_changes(diff) == ["protocol.py constant seed"]


def test_a_version_without_per_key_constants_borrows_those_of_its_hash(tmp_path):
    """A history entry frozen before the record kept its constants per key reads them
    from a version with the same constants hash; one no such version shares keeps "not
    recorded", which fails closed. Constants that do not hash to their entry's hash are never read."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    a = freeze_protocol.freeze(reason=None, root=root, path=out)["fingerprint"]
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("A later prompt.\n")
    b = freeze_protocol.freeze(reason="a prompt", root=root, path=out)
    record = json.loads(out.read_text())
    record["history"][0].pop("constants")  # a: frozen before per-key constants
    out.write_text(json.dumps(record))
    known = protocol.known_versions(out)
    assert known[a]["constants"] is None
    assert known[a]["constants_sha256"] == b["constants_sha256"]
    # A later version whose method list changed: a's keys come from b (same hash).
    later = _consistent(
        {
            **dict(known[b["fingerprint"]]),
            "constants": {**b["constants"], "methods": ["none"]},
        }
    )
    c = later["fingerprint"]
    versions = {**known, c: later}
    diff = protocol.version_diff(a, c, versions)
    assert diff["constants_changed_keys"] == ["methods"]
    assert protocol.unauditable_changes(diff) == []
    # Without a version that shares a's hash, which keys changed is not known.
    alone = {a: known[a], c: later}
    diff = protocol.version_diff(a, c, alone)
    assert diff["constants_changed"] is True
    assert diff["constants_changed_keys"] is None
    assert protocol.unauditable_changes(diff) == ["constants of protocol.py"]
    # Tampered per-key constants (the hash kept) are never read, and the version they
    # belong to is not trusted at all: fail closed.
    tampered = {**later, "constants": {**later["constants"], "seed": 7}}
    assert protocol.recorded_constants(tampered) is None
    diff = protocol.version_diff(a, c, {**known, c: tampered})
    assert diff["constants_changed_keys"] is None
    assert diff["inconsistent"] == [c]
    assert protocol.unauditable_changes(diff) == [
        f"a record that does not reproduce its fingerprint ({c[:12]})"
    ]


def _consistent(entry: dict) -> dict:
    """``entry`` with the fingerprint and constants hash its per-key constants and file
    map give (what ``freeze_protocol`` records)."""
    return {
        **entry,
        "constants_sha256": protocol.constants_sha256(entry["constants"]),
        "fingerprint": protocol.fingerprint_of(entry["constants"], entry["files"]),
    }


# --- the record vouches for itself ------------------------------------------------------


def test_the_real_records_constants_reproduce_their_fingerprints():
    """Every value of the frozen record that keeps its constants per key reproduces its
    fingerprint from them and its file map, so they are read."""
    frozen = _committed_record()
    known = protocol.known_versions(protocol.FROZEN_PROTOCOL_JSON)
    for entry in [*frozen["history"], frozen]:
        if entry.get("constants") is not None:
            reproduced = protocol.fingerprint_of(entry["constants"], entry["files"])
            assert reproduced == entry["fingerprint"]
            assert known[entry["fingerprint"]]["constants"] == entry["constants"]


def test_per_key_constants_are_read_only_when_they_reproduce_the_fingerprint(
    tmp_path, monkeypatch
):
    """Per-key constants read because they hash to their own entry's
    ``constants_sha256`` would let an entry edited in both fields decide which keys a
    version change names: a false ``protocol.py constant seed``, or a real difference
    hidden. They are read only when, with the entry's file map, they reproduce its
    fingerprint (the value the cells carry); an entry whose constants do not, or whose
    constants hash another version's constants contradict, is not trusted at all (fail
    closed)."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    first = freeze_protocol.freeze(reason=None, root=root, path=out)
    monkeypatch.setattr(protocol, "METHODS", ("none", "mirix"))
    second = freeze_protocol.freeze(reason="fewer methods", root=root, path=out)
    a, b = first["fingerprint"], second["fingerprint"]
    for entry in (first, second):
        reproduced = protocol.fingerprint_of(entry["constants"], entry["files"])
        assert reproduced == entry["fingerprint"]
        assert protocol.recorded_constants(entry) == entry["constants"]
    diff = protocol.version_diff(a, b, protocol.known_versions(out))
    assert diff["constants_changed_keys"] == ["methods"] and diff["inconsistent"] == []
    assert protocol.unauditable_changes(diff) == []

    def edited(index: int | None, **constants: object) -> dict:
        """The record with the current value (None) or history entry ``index`` edited:
        its constants changed and their hash recomputed, its fingerprint kept."""
        record = json.loads(out.read_text())
        entry = record if index is None else record["history"][index]
        entry["constants"] = {**entry["constants"], **constants}
        entry["constants_sha256"] = protocol.constants_sha256(entry["constants"])
        return record

    # The frozen value's seed set to 7, its hash recomputed: it would name a change of
    # seed no cell ran under.
    record = edited(None, seed=7)
    assert protocol.recorded_constants(record) is None
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record))
    known = protocol.known_versions(tampered)
    assert known[b]["constants"] is None and known[b]["inconsistent"] is True
    diff = protocol.version_diff(a, b, known)
    assert diff["inconsistent"] == [b]
    assert (diff["constants_changed"], diff["constants_changed_keys"]) == (None, None)
    assert protocol.unauditable_changes(diff) == [
        f"a record that does not reproduce its fingerprint ({b[:12]})"
    ]
    assert "does not reproduce its fingerprint" in protocol.describe_version_diff(diff)

    # A real difference hidden: the history entry's seed differs from the frozen
    # value's; edited to the frozen seed (and hash), it would compare as "no constant
    # changed" but for the method list.
    monkeypatch.setattr(protocol, "SEED", protocol.SEED + 1)
    third = freeze_protocol.freeze(reason="another seed", root=root, path=out)
    c = third["fingerprint"]
    diff = protocol.version_diff(b, c, protocol.known_versions(out))
    assert protocol.unauditable_changes(diff) == ["protocol.py constant seed"]
    record = edited(1, seed=protocol.SEED)
    tampered.write_text(json.dumps(record))
    diff = protocol.version_diff(b, c, protocol.known_versions(tampered))
    assert diff["inconsistent"] == [b]
    assert protocol.unauditable_changes(diff) == [
        f"a record that does not reproduce its fingerprint ({b[:12]})"
    ]

    # An entry without per-key constants whose hash was set to another version's: those
    # constants (equal hash, so equal constants) do not reproduce its fingerprint.
    record = json.loads(out.read_text())
    entry = record["history"][1]
    entry.pop("constants")
    entry["constants_sha256"] = third["constants_sha256"]
    tampered.write_text(json.dumps(record))
    known = protocol.known_versions(tampered)
    assert known[b]["inconsistent"] is False  # nothing of its own contradicts it
    diff = protocol.version_diff(b, c, known)
    assert diff["inconsistent"] == [b] and diff["constants_changed"] is None
    assert protocol.unauditable_changes(diff) == [
        f"a record that does not reproduce its fingerprint ({b[:12]})"
    ]
    # Without per-key constants anywhere for its hash, it cannot be checked: as before.
    entry["constants_sha256"] = "0" * 64
    tampered.write_text(json.dumps(record))
    diff = protocol.version_diff(b, c, protocol.known_versions(tampered))
    assert diff["inconsistent"] == [] and diff["constants_changed_keys"] is None


def test_an_edited_frozen_record_is_drift_although_its_fingerprint_is_kept(
    tmp_path, capsys
):
    """A record whose constants hash, per-key constants or file map were edited (the
    fingerprint kept) must not pass ``--check``: those values decide what a version
    change names. The fingerprint covers the constants and the files, so the record must
    describe the tree in all of them; a re-freeze (a no-op for an equal fingerprint)
    cannot repair it: restore it."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    record = freeze_protocol.freeze(reason=None, root=root, path=out)
    argv = ["--root", str(root), "--out", str(out), "--check"]
    assert freeze_protocol.main(argv) == 0
    capsys.readouterr()
    seven = {**record["constants"], "seed": 7}
    first_file = sorted(record["files"])[0]
    edits = {
        "constants and their hash": {
            "constants": seven,
            "constants_sha256": protocol.constants_sha256(seven),
        },
        "the per-key constants alone": {"constants": seven},
        "the constants hash alone": {"constants_sha256": "0" * 64},
        "the file map": {"files": {**record["files"], first_file: "0" * 64}},
    }
    for what, edit in edits.items():
        out.write_text(json.dumps({**record, **edit}))
        drift = protocol.protocol_drift(root=root, path=out)
        assert drift["drifted"] and drift["current"] == drift["frozen"], what
        assert drift["record_problems"], what
        message = protocol.describe_drift(drift)
        assert "equals the frozen value, but the frozen record" in message, what
        assert "restore" in message, what
        with pytest.raises(protocol.ProtocolDriftError, match="frozen record"):
            protocol.verify_frozen_protocol(root, out)
        assert freeze_protocol.main(argv) == 1, what
        capsys.readouterr()
    assert "seed" in "\n".join(
        protocol.protocol_drift({**record, "constants": seven}, root=root, path=out)[
            "record_problems"
        ]
    )
    assert first_file in "\n".join(
        protocol.protocol_drift(
            {**record, "files": {**record["files"], first_file: "0" * 64}},
            root=root,
            path=out,
        )["record_problems"]
    )
    out.write_text(json.dumps(record))
    assert freeze_protocol.main(argv) == 0
    assert protocol.protocol_drift(root=root, path=out)["record_problems"] == []
    # A history entry edited by hand (its per-key constants no longer reproduce its
    # fingerprint) fails --check too, not only the audit of its cells.
    changed = root / first_file
    changed.write_bytes(changed.read_bytes() + b"\n")
    record = freeze_protocol.freeze(reason="a change", root=root, path=out)
    assert freeze_protocol.main(argv) == 0
    capsys.readouterr()
    earlier = record["history"][-1]
    assert earlier["constants"] == record["constants"]
    history = [*record["history"][:-1], {**earlier, "constants": seven}]
    out.write_text(json.dumps({**record, "history": history}))
    problems = protocol.protocol_drift(root=root, path=out)["record_problems"]
    assert problems == [
        f"its history entries {earlier['fingerprint'][:12]} keep constants that do not "
        "reproduce their fingerprints"
    ]
    assert freeze_protocol.main(argv) == 1
    assert "history entries" in capsys.readouterr().out


def test_the_hint_names_the_commit_that_recorded_a_version():
    """The hint to audit a cell of an earlier version names the command that finds the
    commit that recorded it, not the record's ``git.commit``: that is the commit the
    version was frozen *from*, and the freeze writes the record afterwards, so that
    commit's record names the value before."""
    fingerprint = "1" * 64
    git = {"commit": "b303fa45ad9a0123", "dirty": True}
    hint = protocol.version_checkout_hint(fingerprint, git)
    command = protocol.recorded_commit_command(fingerprint)
    rel = protocol.FROZEN_PROTOCOL_JSON.relative_to(protocol.REPO_ROOT).as_posix()
    assert command == (
        f'git log --reverse --format=%h -G \'^  "fingerprint": "{fingerprint}"\' '
        f"-- {rel}"
    )
    assert shlex.split(command)[:2] == ["git", "log"]
    assert command in hint and "run from the repository root" in hint
    assert "frozen from commit b303fa45ad9a with uncommitted changes" in hint
    assert "freeze_protocol --check" in hint
    assert "no commit of its own" in hint
    clean = protocol.version_checkout_hint(fingerprint, {**git, "dirty": False})
    assert "frozen from commit b303fa45ad9a," in clean
    assert "uncommitted" not in clean
    unknown = protocol.version_checkout_hint("f" * 64, None)
    assert "not recorded" in unknown and "f" * 64 in unknown


def test_only_composition_constants_are_auditable():
    """Pure: which changed constants an audit from the new tree can see past."""
    assert protocol.COMPOSITION_CONSTANTS == frozenset({"methods", "benchmarks"})
    assert set(protocol.COMPOSITION_CONSTANTS) <= set(protocol.protocol_constants())
    base = {"constants_changed": True, "changed": [], "added": [], "removed": []}
    assert (
        protocol.unauditable_changes({**base, "constants_changed_keys": ["methods"]})
        == []
    )
    assert protocol.unauditable_changes(
        {**base, "constants_changed_keys": ["methods", "roles"]}
    ) == ["protocol.py constant roles"]
    # Changed, but no key named (a diff that contradicts itself): fail closed.
    for keys in (None, []):
        assert protocol.unauditable_changes(
            {**base, "constants_changed_keys": keys}
        ) == ["constants of protocol.py"]
    assert (
        protocol.unauditable_changes(
            {**base, "constants_changed": False, "constants_changed_keys": ["seed"]}
        )
        == []
    )
    # Method prompts removed with the method: auditable; a template: not.
    removed = {
        **base,
        "constants_changed_keys": ["methods"],
        "removed": ["methods/prompts/x/curator.txt", "envs/templates/x.md"],
    }
    assert protocol.unauditable_changes(removed) == ["envs/templates/x.md"]


def test_adding_a_model_leaves_every_other_cell_auditable(
    tmp_path, monkeypatch, capsys
):
    """Adding ``luna56`` to ``protocol.MODELS`` and its record to
    ``config/models.json`` is a change of the study's composition. ``constant_scope`` of
    ``models`` is empty when every key both versions have keeps its model, and the
    audit reads models.json only through the canonical slugs every freeze records per
    file, so a re-freeze warns about nothing. Re-pointing a key is a change
    every cell of that key ran under: named, and the tool warns."""
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "frozen.json"
    models_json = root / "config" / "models.json"
    data = json.loads(models_json.read_text())
    del data["models"][LUNA56]
    models_json.write_text(json.dumps(data, indent=2) + "\n")
    without = {k: v for k, v in protocol.MODELS.items() if k != "luna56"}
    monkeypatch.setattr(protocol, "MODELS", without)
    first = freeze_protocol.freeze(reason=None, root=root, path=out)

    monkeypatch.setattr(protocol, "MODELS", {**without, "luna56": LUNA56})
    shutil.copy(protocol.MODELS_JSON, models_json)
    argv = ["--root", str(root), "--out", str(out), "--reason", "luna56"]
    assert freeze_protocol.main(argv) == 0
    captured = capsys.readouterr()
    assert "constants of protocol.py changed: models" in captured.out
    assert "warning" not in captured.err
    second = protocol.load_frozen_protocol(out)
    known = protocol.known_versions(out)
    diff = protocol.version_diff(first["fingerprint"], second["fingerprint"], known)
    assert diff["constants_changed_keys"] == ["models"]
    assert diff["constants_scope"] == {"models": []}
    assert diff["changed"] == [protocol.MODELS_JSON_REL]
    assert diff["model_slugs_recorded"] is True
    assert protocol.unauditable_changes(diff) == []
    for benchmark in protocol.BENCHMARKS:
        assert protocol.unauditable_changes(diff, benchmark) == []
    assert not protocol.touches_benchmark(diff, "models", None)
    assert not protocol.touches_benchmark(diff, "models", "alfworld")

    # Re-pointing a key changes what its cells ran: every benchmark, and a warning.
    monkeypatch.setattr(protocol, "MODELS", {**without, "luna56": LUNA, "luna": LUNA56})
    argv = ["--root", str(root), "--out", str(out), "--reason", "swap"]
    assert freeze_protocol.main(argv) == 0
    assert "warning: this re-freeze changes protocol.py constant models" in (
        capsys.readouterr().err
    )
    third = protocol.load_frozen_protocol(out)
    known = protocol.known_versions(out)
    diff = protocol.version_diff(second["fingerprint"], third["fingerprint"], known)
    assert diff["constants_scope"] == {"models": None}
    assert protocol.unauditable_changes(diff, "alfworld") == [
        "protocol.py constant models"
    ]


def test_scope_of_the_models_constant():
    scope = protocol.constant_scope
    before = {"models": {"luna": LUNA, "luna56": LUNA56}}
    assert (
        scope("models", before, {"models": {"luna": LUNA, "luna56": LUNA56, "x": "y"}})
        == []
    )
    assert scope("models", before, {"models": {"luna": LUNA}}) == []  # one dropped
    assert (
        scope("models", before, {"models": {"luna": LUNA56, "luna56": LUNA56}}) is None
    )
    assert scope("models", {"models": None}, before) is None
    # An empty scope touches no cell, read with or without a benchmark.
    diff = {"constants_scope": {"models": [], "seed": None, "alfworld_x": ["alfworld"]}}
    assert not protocol.touches_benchmark(diff, "models", None)
    assert protocol.touches_benchmark(diff, "seed", None)
    assert protocol.touches_benchmark(diff, "alfworld_x", None)
    assert not protocol.touches_benchmark(diff, "alfworld_x", "webshop")
    assert protocol.touches_benchmark({}, "models", None)  # no scope: fail closed


def test_per_benchmark_scopes_of_constants_and_files():
    """Pure: which benchmarks a changed constant or file touches."""
    before = {
        "benchmarks": ["alfworld", "webarena"],
        "split_sizes": {"alfworld": {"test": 1}, "webarena": {"test": 2}},
        "frozen_asset_sha256": {
            "envs/splits/alfworld/a.json": "1",
            "envs/vendor/NOTICE": "1",
            "config/models.json": "1",
        },
        "alfworld_max_steps": 50,
        "seed": 42,
    }
    after = {
        "benchmarks": ["alfworld", "webshop", "webarena"],
        "split_sizes": {
            "alfworld": {"test": 1},
            "webarena": {"test": 3},
            "webshop": {"test": 4},
        },
        "frozen_asset_sha256": {
            "envs/splits/alfworld/a.json": "1",
            "envs/vendor/NOTICE": "2",
            "config/models.json": "1",
            "envs/templates/webshop_server/server.py": "1",
        },
        "alfworld_max_steps": 30,
        "seed": 7,
    }
    scope = protocol.constant_scope
    assert scope("benchmarks", before, after) == ["webshop"]
    assert (
        scope("benchmarks", before, {**after, "benchmarks": ["webarena", "alfworld"]})
        == []
    )
    assert scope("split_sizes", before, after) == ["webarena", "webshop"]
    assert scope("frozen_asset_sha256", before, after) == ["webshop"]  # NOTICE: docs
    shared = {**after["frozen_asset_sha256"], "config/models.json": "2"}
    assert (
        scope("frozen_asset_sha256", before, {**after, "frozen_asset_sha256": shared})
        is None
    )
    assert scope("alfworld_max_steps", before, after) == ["alfworld"]
    assert scope("seed", before, after) is None  # every benchmark: fail closed
    # An operating limit (how many cells a scheduler runs at once) touches no cell,
    # whatever its values.
    assert "schedule_concurrency_range" in protocol.OPERATING_CONSTANTS
    assert (
        scope(
            "schedule_concurrency_range",
            {"schedule_concurrency_range": [1, 6]},
            {"schedule_concurrency_range": [1, 24]},
        )
        == []
    )
    assert scope("benchmarks", {"benchmarks": None}, after) is None
    assert protocol.asset_benchmark("envs/vendor/webshop_projection.py") == "webshop"
    assert protocol.asset_benchmark("envs/templates/webshop_server/Dockerfile") == (
        "webshop"
    )
    assert protocol.asset_benchmark("envs/vendor/__init__.py") is None
    assert protocol.asset_benchmark("methods/prompts/awm/format.json") is None
    assert protocol.asset_benchmark("envs/splits/otherbench/x.json") is None
    for path in ("envs/vendor/NOTICE", "envs/vendor/VENDOR_LICENSE-APACHE.txt"):
        assert protocol.is_documentation_file(path)
    assert not protocol.is_documentation_file("envs/vendor/vendor_reward.py")
    # By name, never by substring.
    for path in (
        "envs/vendor/VENDOR_NOTICE.txt",
        "envs/vendor/LICENSE-APACHE-2.0.txt",
        "envs/splits/webshop/LICENSE-SkillOpt-MIT.txt",
        "methods/prompts/awm/originals/agent-workflow-memory-8c0ff8c/LICENSE",
    ):
        assert protocol.is_documentation_file(path), path
    for path in (
        "methods/prompts/judge/license_terms.md",
        "envs/templates/webshop/notice_page.json",
        "envs/vendor/licensed_reward.py",
        "envs/vendor/LICENSE.py",
    ):
        assert not protocol.is_documentation_file(path), path
    # Every licence or notice file the fingerprint holds today is one.
    for path in protocol.fingerprint_files():
        name = path.name.upper()
        if "LICENSE" in name or "NOTICE" in name:
            assert protocol.is_documentation_file(path), path
    diff = {
        "constants_changed": True,
        "constants_changed_keys": ["alfworld_max_steps", "split_sizes", "seed"],
        "constants_scope": {
            "alfworld_max_steps": ["alfworld"],
            "split_sizes": ["webshop"],
            "seed": None,
        },
        "changed": ["envs/vendor/NOTICE", "envs/splits/webshop/test.json"],
        "added": [],
        "removed": [],
        "files_scope": {"envs/splits/webshop/test.json": "webshop"},
    }
    assert protocol.unauditable_changes(diff, "alfworld") == [
        "protocol.py constant alfworld_max_steps",
        "protocol.py constant seed",
    ]
    assert protocol.unauditable_changes(diff, "webshop") == [
        "protocol.py constant split_sizes",
        "protocol.py constant seed",
        "envs/splits/webshop/test.json",
    ]
    # A diff without scopes (written before scopes were recorded): every changed
    # constant is named (fail closed); a file still belongs to the benchmark its path
    # names.
    bare = {k: v for k, v in diff.items() if not k.endswith("_scope")}
    assert protocol.unauditable_changes(bare, "alfworld") == [
        "protocol.py constant alfworld_max_steps",
        "protocol.py constant split_sizes",
        "protocol.py constant seed",
    ]
    assert protocol.unauditable_changes(bare, "webshop")[-1] == (
        "envs/splits/webshop/test.json"
    )


def test_attempt_protocol_reads_the_attempt_then_the_history_then_the_cell():
    """What an attempt ran under. Its ``attempt.json`` records it; an older attempt is
    found in ``protocol_history`` once its cell adopted another protocol, and in
    ``cell.json``'s ``cell`` until then (an earlier run.py refused a cell whose manifest
    named another fingerprint)."""
    old, new = "a" * 64, "b" * 64
    legacy_manifest = {
        "cell": {"protocol_fingerprint": old, "protocol_constants_sha256": "ca"}
    }
    legacy_state = {"attempt": 1, "state": "complete"}
    assert protocol.attempt_protocol(legacy_manifest, legacy_state) == {
        "fingerprint": old,
        "constants_sha256": "ca",
        "source": "cell",
    }
    adopted = {
        "cell": {"protocol_fingerprint": new, "protocol_constants_sha256": "cb"},
        "protocol_history": [
            {"fingerprint": old, "constants_sha256": "ca", "attempts": [1]}
        ],
    }
    assert protocol.attempt_protocol(adopted, legacy_state)["fingerprint"] == old
    assert protocol.attempt_protocol(adopted, legacy_state)["source"] == "history"
    recorded = {
        "attempt": 2,
        "protocol": {"fingerprint": new, "constants_sha256": "cb"},
    }
    assert protocol.attempt_protocol(adopted, recorded) == {
        "fingerprint": new,
        "constants_sha256": "cb",
        "source": "attempt",
    }
    # The attempt's own record wins over a history entry that lists it.
    assert protocol.attempt_protocol(adopted, {**recorded, "attempt": 1}) == (
        protocol.attempt_protocol(adopted, recorded)
    )
    # An attempt the history does not list ran under the cell's current protocol; the
    # number can be passed beside the state, and wins over the state's.
    assert protocol.attempt_protocol(adopted, {"attempt": 3})["fingerprint"] == new
    assert protocol.attempt_protocol(adopted, {}, 1)["fingerprint"] == old
    assert protocol.attempt_protocol(adopted, {"attempt": 3}, 1)["fingerprint"] == old
    assert protocol.attempt_protocol({}, None) == {
        "fingerprint": None,
        "constants_sha256": None,
        "source": "cell",
    }


def test_a_frozen_record_without_a_fingerprint_is_refused(tmp_path):
    bad = tmp_path / "frozen.json"
    bad.write_text('{"files": {}}')
    with pytest.raises(protocol.ProtocolDriftError, match="does not hold"):
        protocol.load_frozen_protocol(bad)


def test_the_code_fingerprint_covers_model_visible_python_and_the_mirix_package(
    tmp_path,
):
    """The harness code that shapes model-visible text after the first prompt, and
    the MIRIX package (its prompts, distiller, curator, renderer), are hashed per launch;
    the protocol fingerprint does not change with them."""
    root = tmp_path / "skill_bench"
    for d in protocol.CODE_FINGERPRINT_DIRS:
        (root / d).mkdir(parents=True)
        (root / d / "a.py").write_text(f"# {d}\n")
    (root / "protocol.py").write_text("X = 1\n")
    (root / "run.py").write_text("# not model-visible\n")
    mirix = tmp_path / "mirix"
    (mirix / "prompts" / "system").mkdir(parents=True)
    (mirix / "prompts" / "system" / "base.txt").write_text("You are MIRIX.\n")
    (mirix / "client.py").write_text("def render(): ...\n")

    def fp() -> dict:
        return protocol.code_fingerprint(mirix=True, root=root, mirix_dir=mirix)

    base = fp()
    assert base["harness_files"] == len(protocol.CODE_FINGERPRINT_DIRS) + 1
    assert base["mirix_files"] == 2 and base["mirix"]
    # Not covered: harness orchestration, tests, caches.
    (root / "run.py").write_text("# changed\n")
    (root / "methods" / "tests").mkdir()
    (root / "methods" / "tests" / "t.py").write_text("x\n")
    (root / "methods" / "__pycache__").mkdir()
    (root / "methods" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
    assert fp() == base
    # Covered: a method's input builder, a new env module, the MIRIX prompt text.
    (root / "methods" / "a.py").write_text("# the learner keeps other turns\n")
    changed = fp()
    assert changed["harness"] != base["harness"] and changed["mirix"] == base["mirix"]
    (mirix / "prompts" / "system" / "base.txt").write_text("You are MIRIX v2.\n")
    assert fp()["mirix"] != changed["mirix"]
    assert "mirix" not in protocol.code_fingerprint(root=root)
    real = protocol.code_fingerprint(mirix=True)
    assert real["harness_files"] > 10 and real["mirix_files"] > 10
    before = protocol.protocol_fingerprint()
    assert protocol.protocol_fingerprint() == before  # independent of code_fingerprint


# --- provider policy -----------------------------------------------------------------

#: A provider policy for these tests only: no model of the study restricts its providers
#: (``protocol.PROVIDER_POLICY`` is empty), and the mechanism stays.
TEST_POLICY = {LUNA56: {"ignore": ("open-inference", "wafer")}}
TEST_SLUG_NAMES = {"open-inference": "OpenInference", "wafer": "Wafer"}


def _test_policy(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "PROVIDER_POLICY", TEST_POLICY)
    monkeypatch.setattr(protocol, "PROVIDER_SLUG_NAMES", TEST_SLUG_NAMES)


def test_the_provider_policy_ignores_its_providers_for_its_model_only(monkeypatch):
    assert protocol.PROVIDER_POLICY == {} and protocol.PROVIDER_SLUG_NAMES == {}
    for model in (*protocol.CHAT_MODELS, protocol.EMBEDDING_MODEL):
        assert protocol.provider_routing(model) is None
    # Every ignored slug has the name OpenRouter reports as the call's provider.
    ignored = {s for e in protocol.PROVIDER_POLICY.values() for s in e["ignore"]}
    assert ignored == set(protocol.PROVIDER_SLUG_NAMES)
    _test_policy(monkeypatch)
    assert protocol.provider_routing(LUNA56) == {"ignore": ["open-inference", "wafer"]}
    assert protocol.provider_routing(LUNA) is None
    assert protocol.provider_routing(protocol.EMBEDDING_MODEL) is None
    constants = protocol.protocol_constants()
    assert constants["provider_policy"] == {
        LUNA56: {"ignore": ["open-inference", "wafer"]}
    }
    assert constants["provider_slug_names"] == {
        "open-inference": "OpenInference",
        "wafer": "Wafer",
    }


def test_the_provider_policy_is_fingerprinted(tmp_path, monkeypatch):
    root = _mini_tree(tmp_path / "sb")
    baseline = protocol.protocol_fingerprint(root)
    monkeypatch.setattr(
        protocol, "PROVIDER_POLICY", {LUNA56: {"ignore": ("open-inference",)}}
    )
    assert protocol.protocol_fingerprint(root) != baseline
    monkeypatch.undo()
    monkeypatch.setattr(protocol, "PROVIDER_SLUG_NAMES", {"wafer": "Wafer"})
    assert protocol.protocol_fingerprint(root) != baseline
    monkeypatch.undo()
    assert protocol.protocol_fingerprint(root) == baseline


@pytest.mark.parametrize(
    ("provider", "ignored"),
    [
        ("OpenInference", "open-inference"),
        ("open-inference", "open-inference"),
        ("Open Inference", "open-inference"),
        ("Wafer", "wafer"),
        ("WAFER", "wafer"),
        ("DeepInfra", None),
        ("Inference Net", None),  # "inference" alone is another provider
        ("", None),
        (None, None),
    ],
)
def test_an_ignored_provider_is_known_by_its_slug_or_name(
    provider, ignored, monkeypatch
):
    _test_policy(monkeypatch)
    assert protocol.ignored_provider(LUNA56, provider) == ignored
    assert protocol.ignored_provider(LUNA, provider) is None


def test_freeze_protocol_refuses_a_malformed_frozen_record(tmp_path, capsys):
    """A frozen record that is not JSON, or holds no fingerprint, exits 2 with the
    reason, not a traceback."""
    from evals.skill_bench.tools import freeze_protocol

    out = tmp_path / "protocol_frozen.json"
    argv = ["--root", str(tmp_path), "--out", str(out), "--reason", "again"]
    out.write_text("{}", encoding="utf-8")
    assert freeze_protocol.main(argv) == 2
    assert "does not hold a frozen fingerprint" in capsys.readouterr().err
    out.write_text("{not json", encoding="utf-8")
    assert freeze_protocol.main(argv) == 2
    assert capsys.readouterr().err.strip()
    assert out.read_text(encoding="utf-8") == "{not json"


# --- the matrix the scheduler plans ----------------------------------------------------


def test_the_scheduler_plans_the_protocol_matrix():
    """``schedule.py`` without ``--study`` plans every benchmark x model x method of the
    protocol, and the ``phase16`` study selects that whole matrix."""
    from evals.skill_bench import schedule

    shape = (len(protocol.BENCHMARKS), len(protocol.MODELS), len(protocol.METHODS))
    cells = len(schedule.matrix())
    assert shape[0] * shape[1] * shape[2] == cells == 16
    assert len(schedule.study_matrix("phase16")) == 16


# --- versions recorded per attempt; automatic freeze ---------------------------------------

GIT = {"commit": "c0ffee0123456789", "dirty": False}


def _git() -> dict:
    return dict(GIT)


def _frozen_mini(tmp_path: Path) -> tuple[Path, Path]:
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    out = tmp_path / "record" / "protocol_frozen.json"
    freeze_protocol.freeze(reason=None, root=root, path=out, git=_git)
    return root, out


def test_the_frozen_record_path_follows_the_environment(tmp_path, monkeypatch):
    """every reader and writer not given a path reads the record the environment
    names, so the tests (and a cell a test runs, which freezes automatically) never write
    the repository's record."""
    from evals.skill_bench.tools import freeze_protocol

    monkeypatch.delenv(protocol.FROZEN_RECORD_ENV, raising=False)
    assert protocol.frozen_record_path() == protocol.FROZEN_PROTOCOL_JSON
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, "")
    assert protocol.frozen_record_path() == protocol.FROZEN_PROTOCOL_JSON
    copy = tmp_path / "copy.json"
    record = json.loads(protocol.FROZEN_PROTOCOL_JSON.read_text(encoding="utf-8"))
    # An earlier version as freeze_protocol keeps one (the record may have no history).
    earlier = {k: record[k] for k in freeze_protocol.HISTORY_KEYS if k in record}
    extra = {**earlier, "fingerprint": "e" * 64}
    copy.write_text(json.dumps({**record, "history": [*record["history"], extra]}))
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, str(copy))
    assert protocol.frozen_record_path() == copy
    assert protocol.load_frozen_protocol()["history"][-1]["fingerprint"] == "e" * 64
    assert "e" * 64 in protocol.known_versions()
    assert protocol.frozen_fingerprint() == record["fingerprint"]
    # A relative value is made absolute (a cell's subprocess may run elsewhere).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, "copy.json")
    assert protocol.frozen_record_path() == copy.absolute()
    assert protocol.frozen_record_path().is_absolute()


def test_protocol_drift_against_a_given_manifest(tmp_path, monkeypatch):
    """the drift of a manifest the caller already computed (the one a launch runs
    under) needs no second hashing of the tree."""
    root, out = _frozen_mini(tmp_path)
    record = protocol.load_frozen_protocol(out)
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("changed\n")
    manifest = protocol.fingerprint_manifest(root)

    def no_hashing(*_args, **_kwargs):
        raise AssertionError("the tree was hashed again")

    monkeypatch.setattr(protocol, "fingerprint_manifest", no_hashing)
    drift = protocol.protocol_drift(record, root, current=manifest)
    assert drift["drifted"] is True
    assert drift["changed"] == ["methods/prompts/reasoningbank/extract_success.txt"]
    assert drift["current"] == manifest["fingerprint"] != record["fingerprint"]
    # The texts no longer tell a person to freeze by hand before a run.
    assert "automatically" in protocol.REFREEZE_HINT
    assert "--reason" in protocol.REFREEZE_HINT


def test_the_code_fingerprint_lists_its_files_without_changing_its_digest(tmp_path):
    """with ``files`` the per-file hashes the digests are made of, so a later version
    segment names the code files that changed; the digests are unchanged."""
    root = tmp_path / "skill_bench"
    for d in protocol.CODE_FINGERPRINT_DIRS:
        (root / d).mkdir(parents=True)
        (root / d / "a.py").write_text(f"# {d}\n")
    (root / "protocol.py").write_text("X = 1\n")
    mirix = tmp_path / "mirix"
    (mirix / "prompts").mkdir(parents=True)
    (mirix / "prompts" / "base.txt").write_text("You are MIRIX.\n")
    (mirix / "client.py").write_text("def render(): ...\n")
    plain = protocol.code_fingerprint(root=root, mirix=True, mirix_dir=mirix)
    full = protocol.code_fingerprint(root=root, mirix=True, mirix_dir=mirix, files=True)
    assert {k: v for k, v in full.items() if not k.endswith("_sha256")} == plain
    assert set(full["harness_sha256"]) == {
        *(f"{d}/a.py" for d in protocol.CODE_FINGERPRINT_DIRS),
        "protocol.py",
    }
    assert set(full["mirix_sha256"]) == {"prompts/base.txt", "client.py"}
    for key in ("harness", "mirix"):
        entries = sorted(full[f"{key}_sha256"].items())
        digest = protocol._sha256(protocol._canonical(entries))
        assert digest == full[key]
    assert "harness_sha256" not in protocol.code_fingerprint(root=root)
    assert "mirix_sha256" not in protocol.code_fingerprint(root=root, files=True)
    real = protocol.code_fingerprint(mirix=True, files=True)
    assert len(real["harness_sha256"]) == real["harness_files"]
    assert len(real["mirix_sha256"]) == real["mirix_files"]


def test_the_mirix_code_digest_covers_the_server_launcher(tmp_path):
    """A MIRIX cell's server is started by ``scripts/start_server.py`` (it
    picks uvicorn or gunicorn and their options), so an edit to it is part of the MIRIX
    code a version records and names, keyed by its path relative to the package."""
    root = tmp_path / "skill_bench"
    (root / "agent").mkdir(parents=True)
    (root / "agent" / "a.py").write_text("# a\n")
    mirix = tmp_path / "mirix"
    mirix.mkdir()
    (mirix / "client.py").write_text("def render(): ...\n")
    launcher = tmp_path / "scripts" / "start_server.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("uvicorn.run(app, workers=1)\n")

    def fp() -> dict:
        return protocol.code_fingerprint(
            root=root, mirix=True, mirix_dir=mirix, files=True
        )

    base = fp()
    assert set(base["mirix_sha256"]) == {"client.py", "../scripts/start_server.py"}
    assert base["mirix_files"] == 2
    entries = sorted(base["mirix_sha256"].items())
    assert protocol._sha256(protocol._canonical(entries)) == base["mirix"]
    launcher.write_text("gunicorn(app, workers=4)\n")
    changed = fp()
    assert changed["mirix"] != base["mirix"] and changed["harness"] == base["harness"]
    # The real checkout's launcher is the one infra/mirix_server.py runs.
    from evals.skill_bench.infra import mirix_server

    assert mirix_server.START_SERVER == protocol.REPO_ROOT / protocol.MIRIX_LAUNCHER
    real = protocol.code_fingerprint(mirix=True, files=True)
    assert protocol.MIRIX_LAUNCHER_KEY in real["mirix_sha256"]
    # A missing package records no MIRIX digest, launcher or not.
    gone = protocol.code_fingerprint(root=root, mirix=True, mirix_dir=tmp_path / "x")
    assert gone["mirix"] is None and gone["mirix_files"] == 0


def _work(attempt, launch, fingerprint, harness="h1", commit="c1", phase="learning"):
    line = {
        "event": "work",
        "launch": launch,
        "at": f"2026-09-30T00:00:0{launch[-1]}+00:00",
        "attempt": attempt,
        "phase": phase,
        "host": {"hostname": "my-laptop.local"},
        "git": {"commit": commit, "dirty": False},
        "code": {"harness": harness, "harness_files": 3},
    }
    if fingerprint is not None:
        line["protocol"] = {"fingerprint": fingerprint, "constants_sha256": "cs"}
    return line


def test_attempt_versions_reads_segments_or_groups_legacy_work_lines():
    """the recorded segments as a copy, or legacy segments synthesized from the
    attempt's work lines (an attempt written before segments were recorded), or one
    segment from the attempt's own start."""
    a, b = "a" * 64, "b" * 64
    manifest = {"cell": {"protocol_fingerprint": a, "protocol_constants_sha256": "ca"}}
    segments = [{"index": 0, "legacy": False, "launches": ["L1"], "changed": None}]
    state = {"attempt": 2, "version_segments": segments}
    read = protocol.attempt_versions(manifest, state, 2, ())
    assert read == segments
    read[0]["launches"].append("mutated")
    assert segments[0]["launches"] == ["L1"]  # a copy
    assert protocol.attempt_versions(manifest, {"version_segments": []}, 1) == []

    lines = [
        _work(1, "L1", a),
        {"event": "start", "launch": "L2"},
        _work(2, "X9", b),  # another attempt's line
        _work(1, "L3", None),  # no protocol: the attempt's (the manifest's here)
        _work(1, "L4", b, harness="h2", commit="c2", phase="test"),
    ]
    legacy = protocol.attempt_versions(manifest, {"attempt": 1}, 1, lines, versions={})
    assert [s["index"] for s in legacy] == [0, 1]
    assert [s["launches"] for s in legacy] == [["L1", "L3"], ["L4"]]
    assert all(s["legacy"] and s["why"] == "legacy" for s in legacy)
    assert legacy[0]["changed"] is None and legacy[0]["protocol"]["fingerprint"] == a
    assert legacy[0]["first_launch"] == "L1" and legacy[0]["phase"] == "learning"
    assert legacy[1]["phase"] == "test" and legacy[1]["started_at"].endswith("4+00:00")
    assert legacy[1]["changed"] == {
        "protocol": True,
        "protocol_diff": None,  # neither version is known here
        "code": ["harness"],
        "code_files": None,
        "commit": True,
        "tree": False,
    }
    for key in ("tree", "version_file", "auto_freeze", "known_version", "note"):
        assert legacy[0][key] is None
    # A history entry of the manifest names the protocol of a line without one.
    adopted = {
        "cell": {"protocol_fingerprint": b},
        "protocol_history": [{"fingerprint": a, "attempts": [1]}],
    }
    only = protocol.attempt_versions(adopted, {"attempt": 1}, 1, [lines[3]])
    assert only[0]["protocol"]["fingerprint"] == a
    # No work line (an attempt from before them): its own start.
    started = {
        "attempt": 1,
        "started_by": "L0",
        "started_at": "2026-09-29T00:00:00+00:00",
        "code": {"harness": "h0"},
        "git": {"commit": "c0"},
        "host": {"hostname": "laptop-2.local"},
        "protocol": {"fingerprint": b, "constants_sha256": "cb"},
    }
    (one,) = protocol.attempt_versions(manifest, started, 1, ())
    assert one["launches"] == ["L0"] and one["legacy"] is True
    assert one["protocol"] == {"fingerprint": b, "constants_sha256": "cb"}
    assert (one["code"], one["git"], one["host"]) == (
        {"harness": "h0"},
        {"commit": "c0"},
        {"hostname": "laptop-2.local"},
    )


def test_a_legacy_protocol_change_names_its_files_from_the_known_versions(tmp_path):
    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    first = protocol.frozen_fingerprint(out)
    (root / "envs" / "templates" / "alfworld" / "actor_step.txt").write_text("new\n")
    second = freeze_protocol.freeze(reason="t", root=root, path=out, git=_git)
    lines = [_work(1, "L1", first), _work(1, "L2", second["fingerprint"])]
    known = protocol.known_versions(out)
    legacy = protocol.attempt_versions({}, {"attempt": 1}, 1, lines, versions=known)
    diff = legacy[1]["changed"]["protocol_diff"]
    assert diff["changed"] == ["envs/templates/alfworld/actor_step.txt"]
    assert diff["from"] == first and diff["to"] == second["fingerprint"]


def test_a_recorded_version_is_read_only_when_it_reproduces_its_fingerprint(tmp_path):
    """a document that describes one version (a frozen record, a history entry, a
    segment's version.json) is read only when its constants and files reproduce its
    fingerprint; otherwise None (fails closed)."""
    _root, out = _frozen_mini(tmp_path)
    record = protocol.load_frozen_protocol(out)
    version = protocol.recorded_version(record)
    assert version is not None and version["fingerprint"] == record["fingerprint"]
    assert version["current"] is False and version["files"] == record["files"]
    assert version["constants"] == record["constants"]
    # A version.json: frozen_at and reason from its frozen block.
    doc = {
        "fingerprint": record["fingerprint"],
        "constants_sha256": record["constants_sha256"],
        "constants": record["constants"],
        "files": record["files"],
        "frozen": {"frozen_at": "2026-09-30T01:00:00+00:00", "reason": "auto"},
    }
    from_doc = protocol.recorded_version(doc)
    assert from_doc["frozen_at"] == "2026-09-30T01:00:00+00:00"
    assert from_doc["reason"] == "auto"
    edited = json.loads(json.dumps(record))
    edited["constants"]["seed"] = 43
    assert protocol.recorded_version(edited) is None
    assert protocol.recorded_version({**record, "files": None}) is None
    assert (
        protocol.recorded_version({k: v for k, v in record.items() if k != "files"})
        is None
    )
    assert protocol.recorded_version({**record, "fingerprint": None}) is None
    assert protocol.recorded_version("not a mapping") is None  # type: ignore[arg-type]
    # constants_of is the public resolver version_diff uses.
    known = protocol.known_versions(out)
    assert protocol.constants_of(version, known) == (record["constants"], False)
    assert protocol.constants_of(
        {"fingerprint": "x", "constants": {"a": 1}}, known
    ) == (
        None,
        True,
    )


def test_freeze_waits_for_the_record_lock(tmp_path):
    """every writer holds the record's lock, so two writers never interleave; a
    writer that cannot get it in time raises and writes nothing."""
    import threading
    import time

    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q9"]\n')
    before = out.read_bytes()
    with freeze_protocol.record_lock(out):
        with pytest.raises(freeze_protocol.FreezeLockTimeout, match="lock"):
            freeze_protocol.freeze(
                reason="blocked", root=root, path=out, git=_git, lock_wait_s=0.1
            )
        with pytest.raises(freeze_protocol.FreezeLockTimeout):
            freeze_protocol.backfill_constants(root=root, path=out, lock_wait_s=0.1)
    assert out.read_bytes() == before
    held = threading.Event()
    released: list[float] = []

    def hold() -> None:
        with freeze_protocol.record_lock(out):
            held.set()
            time.sleep(0.3)
            released.append(time.monotonic())

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(5)
    record = freeze_protocol.freeze(
        reason="waited", root=root, path=out, git=_git, lock_wait_s=5
    )
    done = time.monotonic()
    thread.join()
    assert released and done >= released[0]
    assert record["reason"] == "waited" and len(record["history"]) == 1
    # No lock file lands next to the record.
    assert sorted(p.name for p in out.parent.iterdir()) == ["protocol_frozen.json"]


def _context(**extra) -> dict:
    return {
        "cell_id": "webshop-luna-awm",
        "attempt": 2,
        "why": "first attempt",
        "launch": "l4unch",
        "host": {"hostname": "my-laptop.local", "machine_id": "m1"},
        "cell_dir": "/runs/webshop-luna-awm",
        **extra,
    }


def test_auto_freeze_freezes_a_changed_tree_once_and_names_why(tmp_path):
    """a launch under a tree that is not the frozen value freezes it, once, with a
    reason naming the cell, the attempt, the commit and the changed files."""
    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    old = protocol.load_frozen_protocol(out)
    prompt = root / "methods" / "prompts" / "reasoningbank" / "extract_success.txt"
    prompt.write_text("You are an expert analyst.\n")
    manifest = protocol.fingerprint_manifest(root)
    result = freeze_protocol.auto_freeze(
        manifest=manifest,
        context=_context(),
        root=root,
        path=out,
        git=lambda: {**GIT, "dirty": True},
    )
    assert result["status"] == "frozen", result
    assert result["fingerprint"] == manifest["fingerprint"]
    assert result["previous_fingerprint"] == old["fingerprint"]
    assert result["changed"] == ["methods/prompts/reasoningbank/extract_success.txt"]
    assert result["constants_changed_keys"] == [] and result["revert_to_known"] is False
    assert result["error"] is None and result["waited_s"] >= 0
    reason = result["reason"]
    assert "auto-freeze" in reason
    for needle in (
        "webshop-luna-awm",
        "attempt 02",
        "first attempt",
        "c0ffee012345",
        "with uncommitted changes",
        "extract_success.txt",
        "constants [none]",
    ):
        assert needle in reason, needle
    record = protocol.load_frozen_protocol(out)
    assert (
        record["fingerprint"] == manifest["fingerprint"] and record["reason"] == reason
    )
    assert record["auto_freeze"]["previous_fingerprint"] == old["fingerprint"]
    assert record["auto_freeze"]["cell_id"] == "webshop-luna-awm"
    assert record["auto_freeze"]["attempt"] == 2
    assert record["history"][-1]["fingerprint"] == old["fingerprint"]
    assert "auto_freeze" not in record["history"][-1]  # a manual freeze's entry
    known = protocol.known_versions(out)
    assert known[manifest["fingerprint"]]["auto_freeze"]["why"] == "first attempt"
    # Once: the second launch finds it frozen and writes nothing.
    data = out.read_bytes()
    again = freeze_protocol.auto_freeze(
        manifest=manifest, context=_context(attempt=3), root=root, path=out, git=_git
    )
    assert again["status"] == "already_frozen" and again["warning"] is None
    assert again["frozen_at"] == record["frozen_at"]
    assert out.read_bytes() == data
    # The record the next freeze writes keeps the auto-frozen entry's block in history.
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q7"]\n')
    third = freeze_protocol.freeze(reason="by hand", root=root, path=out, git=_git)
    assert third["history"][-1]["auto_freeze"]["cell_id"] == "webshop-luna-awm"
    assert "auto_freeze" not in third


def test_auto_freeze_names_constant_keys_and_many_files(tmp_path, monkeypatch):
    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    for n in range(12):
        path = root / "envs" / "splits" / "webshop" / f"extra{n:02d}.json"
        path.write_text(f'["{n}"]\n')
    monkeypatch.setattr(protocol, "SEED", protocol.SEED + 1)
    manifest = protocol.fingerprint_manifest(root)
    result = freeze_protocol.auto_freeze(
        manifest=manifest, context=_context(), root=root, path=out, git=_git
    )
    assert result["status"] == "frozen" and result["constants_changed_keys"] == ["seed"]
    assert "constants [seed]" in result["reason"] and "+2 more" in result["reason"]
    assert len(result["added"]) == 12
    # A constant every cell runs under: the re-freeze warning comes with it.
    assert result["warning"] and "not re-verified" in result["warning"]


def test_auto_freeze_back_to_an_earlier_version(tmp_path):
    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    a = protocol.fingerprint_manifest(root)
    split = root / "envs" / "splits" / "webshop" / "test.json"
    original = split.read_text()
    split.write_text('["q1"]\n')
    b = protocol.fingerprint_manifest(root)
    assert (
        freeze_protocol.auto_freeze(
            manifest=b, context=_context(), root=root, path=out, git=_git
        )["status"]
        == "frozen"
    )
    split.write_text(original)
    back = freeze_protocol.auto_freeze(
        manifest=protocol.fingerprint_manifest(root),
        context=_context(why="test resume of attempt 2"),
        root=root,
        path=out,
        git=_git,
    )
    assert back["status"] == "frozen" and back["revert_to_known"] is True
    assert "back to the earlier version frozen at" in back["reason"]
    record = protocol.load_frozen_protocol(out)
    assert record["fingerprint"] == a["fingerprint"]
    assert [e["fingerprint"] for e in record["history"]][-2:] == [
        a["fingerprint"],
        b["fingerprint"],
    ]
    assert record["auto_freeze"]["revert_to_known"] is True


def test_two_auto_freezes_at_once_write_one_version(tmp_path):
    import threading

    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q5"]\n')
    manifest = protocol.fingerprint_manifest(root)
    history = len(protocol.load_frozen_protocol(out)["history"])
    barrier = threading.Barrier(2)
    results: list[dict] = []

    def freeze_once(cell: str) -> None:
        barrier.wait()
        results.append(
            freeze_protocol.auto_freeze(
                manifest=manifest,
                context=_context(cell_id=cell),
                root=root,
                path=out,
                git=_git,
            )
        )

    threads = [
        threading.Thread(target=freeze_once, args=(c,))
        for c in ("webshop-luna-awm", "webshop-luna-reasoningbank")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(r["status"] for r in results) == ["already_frozen", "frozen"]
    record = protocol.load_frozen_protocol(out)
    assert len(record["history"]) == history + 1
    assert record["fingerprint"] == manifest["fingerprint"]


def test_a_held_record_lock_is_recorded_not_waited_out(tmp_path):
    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q6"]\n')
    before = out.read_bytes()
    with freeze_protocol.record_lock(out):
        result = freeze_protocol.auto_freeze(
            manifest=protocol.fingerprint_manifest(root),
            context=_context(),
            root=root,
            path=out,
            git=_git,
            lock_wait_s=0.2,
        )
    assert result["status"] == "error" and "lock" in result["error"]
    assert result["waited_s"] >= 0.2
    assert out.read_bytes() == before


def test_auto_freeze_never_replaces_a_missing_or_malformed_record(tmp_path):
    from evals.skill_bench.tools import freeze_protocol

    root = _mini_tree(tmp_path / "sb")
    manifest = protocol.fingerprint_manifest(root)
    missing = tmp_path / "none" / "protocol_frozen.json"
    result = freeze_protocol.auto_freeze(
        manifest=manifest, context=_context(), root=root, path=missing, git=_git
    )
    assert result["status"] == "error" and "never recreated" in result["error"]
    assert not missing.exists()
    bad = tmp_path / "bad" / "protocol_frozen.json"
    bad.parent.mkdir()
    bad.write_text("{not json")
    result = freeze_protocol.auto_freeze(
        manifest=manifest, context=_context(), root=root, path=bad, git=_git
    )
    assert result["status"] == "error" and bad.read_text() == "{not json"
    # An incomplete manifest, or a git that raises: recorded, never raised.
    _root, out = _frozen_mini(tmp_path / "ok")
    incomplete = freeze_protocol.auto_freeze(
        manifest={"fingerprint": "f" * 64}, context=_context(), path=out
    )
    assert (
        incomplete["status"] == "error" and "incomplete manifest" in incomplete["error"]
    )

    def broken() -> dict:
        raise OSError("git is gone")

    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q8"]\n')
    failed = freeze_protocol.auto_freeze(
        manifest=protocol.fingerprint_manifest(root),
        context=_context(),
        root=root,
        path=out,
        git=broken,
    )
    assert failed["status"] == "error" and "git is gone" in failed["error"]


def test_the_record_lock_works_where_a_directory_cannot_be_locked(
    tmp_path, monkeypatch, capsys
):
    """NFS emulates flock as a whole-file fcntl lock, which needs a file
    opened for writing, so the lock on the record's directory fails there (EBADF), not
    with BlockingIOError. The lock then falls back to a lock file beside the record,
    which serializes writers the same way. Where nothing can be locked, a writer gets
    FreezeLockUnavailable naming the filesystem: auto_freeze records it, the tool prints
    it and exits 2 (no OSError traceback)."""
    import errno
    import os
    import stat

    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    real = freeze_protocol.fcntl.flock

    def nfs(fd: int, op: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real(fd, op)

    monkeypatch.setattr(freeze_protocol.fcntl, "flock", nfs)
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q5"]\n')
    before = out.read_bytes()
    lock = freeze_protocol.lock_file_path(out)
    assert lock == out.with_name(".protocol_frozen.json.lock")
    with freeze_protocol.record_lock(out):
        assert lock.is_file()
        with pytest.raises(freeze_protocol.FreezeLockTimeout, match="lock"):
            freeze_protocol.freeze(
                reason="blocked", root=root, path=out, git=_git, lock_wait_s=0.1
            )
    assert out.read_bytes() == before
    record = freeze_protocol.freeze(reason="on nfs", root=root, path=out, git=_git)
    assert record["reason"] == "on nfs"
    assert protocol.load_frozen_protocol(out)["fingerprint"] == record["fingerprint"]

    def none(fd: int, op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(freeze_protocol.fcntl, "flock", none)
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q4"]\n')
    written = out.read_bytes()
    with pytest.raises(freeze_protocol.FreezeLockUnavailable, match="No locks"):
        freeze_protocol.freeze(reason="x", root=root, path=out, git=_git)
    assert issubclass(
        freeze_protocol.FreezeLockUnavailable, freeze_protocol.FreezeLockError
    )
    assert issubclass(
        freeze_protocol.FreezeLockTimeout, freeze_protocol.FreezeLockError
    )
    result = freeze_protocol.auto_freeze(
        manifest=protocol.fingerprint_manifest(root),
        context=_context(),
        root=root,
        path=out,
        git=_git,
    )
    assert result["status"] == "error" and "FreezeLockUnavailable" in result["error"]
    for argv in (
        ["--reason", "x"],
        ["--backfill-constants"],
        ["--backfill-model-slugs"],
    ):
        code = freeze_protocol.main([*argv, "--root", str(root), "--out", str(out)])
        assert code == 2 and "No locks" in capsys.readouterr().err, argv
    assert out.read_bytes() == written


def test_the_frozen_record_is_written_durably(tmp_path, monkeypatch):
    """The new record reaches the disk before it replaces the old one, and
    the replace itself is made durable (a crash right after an automatic freeze must not
    leave an empty record, which no launch would ever recreate)."""
    import os

    from evals.skill_bench.tools import freeze_protocol

    events: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append(("fsync", _fd_kind(fd)))
        real_fsync(fd)

    def replace(src, dst) -> None:
        events.append(("replace", Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    out = tmp_path / "record" / "protocol_frozen.json"
    freeze_protocol._write(out, {"fingerprint": "f" * 64})
    assert events == [
        ("fsync", "file"),
        ("replace", "protocol_frozen.json"),
        ("fsync", "dir"),
    ]
    assert json.loads(out.read_text())["fingerprint"] == "f" * 64
    assert sorted(p.name for p in out.parent.iterdir()) == ["protocol_frozen.json"]


def _fd_kind(fd: int) -> str:
    import os
    import stat

    return "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"


def test_an_auto_freeze_into_a_git_tracked_record_says_how_to_pull(tmp_path):
    """An automatic freeze rewrites config/protocol_frozen.json, which git
    tracks, so on that host the record differs from its commit (git_info leaves it out of
    ``dirty``) and a later ``git pull`` that brings another record stops. The result
    says so (``record_tracked``, and a ``note``: commit the record or set it aside before
    pulling; the cells keep their versions either way), which run.py logs and records
    in the segment. A record git does not track, or where git cannot tell, gets none."""
    import subprocess

    from evals.skill_bench.tools import freeze_protocol

    root, out = _frozen_mini(tmp_path)
    repo = out.parent
    for args in (["init", "-q"], ["add", out.name], ["commit", "-q", "-m", "record"]):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    assert freeze_protocol.record_tracked_by_git(out) is True
    untracked = repo / "elsewhere" / "protocol_frozen.json"
    freeze_protocol.freeze(reason=None, root=root, path=untracked, git=_git)
    assert freeze_protocol.record_tracked_by_git(untracked) is False
    plain = tmp_path / "plain" / "protocol_frozen.json"
    freeze_protocol.freeze(reason=None, root=root, path=plain, git=_git)
    assert freeze_protocol.record_tracked_by_git(plain) is None
    (root / "envs" / "splits" / "webshop" / "test.json").write_text('["q5"]\n')
    manifest = protocol.fingerprint_manifest(root)
    results = {
        name: freeze_protocol.auto_freeze(
            manifest=manifest, context=_context(), root=root, path=path, git=_git
        )
        for name, path in (("tracked", out), ("untracked", untracked), ("plain", plain))
    }
    assert {name: r["status"] for name, r in results.items()} == dict.fromkeys(
        results, "frozen"
    )
    assert {name: r["record_tracked"] for name, r in results.items()} == {
        "tracked": True,
        "untracked": False,
        "plain": None,
    }
    note = results["tracked"]["note"]
    for needle in (
        "tracked by git",
        "git pull",
        "git stash",
        "version.json",
        "c0ffee0",
    ):
        assert needle in note, (needle, note)
    assert results["untracked"]["note"] is None and results["plain"]["note"] is None
    again = freeze_protocol.auto_freeze(
        manifest=manifest, context=_context(), root=root, path=out, git=_git
    )
    assert again["status"] == "already_frozen" and again["note"] is None


# --- gateway settings of a protocol ----------------------------------------------------


def test_gateway_settings_of_the_protocol_are_the_gateways_and_the_audits():
    """run.py holds a live gateway's settings to this tree's through
    ``protocol.gateway_settings_of``: the same settings, in the same form, as the gateway's
    ``PROTOCOL_SETTINGS``, and the same constant per setting and model settings as the
    audit judges a gateway start by (``SETTINGS_CONSTANTS``, ``MODEL_SETTINGS``)."""
    from evals.skill_bench import audit
    from evals.skill_bench.infra import gateway

    assert dict(protocol.GATEWAY_SETTINGS_CONSTANTS) == dict(audit.SETTINGS_CONSTANTS)
    assert protocol.GATEWAY_MODEL_SETTINGS == tuple(audit.MODEL_SETTINGS)
    constants = protocol.protocol_constants()
    assert protocol.gateway_settings_of(constants) == dict(gateway.PROTOCOL_SETTINGS)
    served = dict(gateway.PROTOCOL_SETTINGS)
    assert protocol.gateway_settings_differ(served, constants) == []
    other = {**served, "seed": 7, "max_retries": 99}
    other.pop("deadline_s")
    assert protocol.gateway_settings_differ(other, constants) == [
        "deadline_s",
        "max_retries",
        "seed",
    ]
    assert protocol.gateway_settings_differ(
        other, constants, protocol.GATEWAY_MODEL_SETTINGS
    ) == ["seed"]
    # Canonical JSON: key order inside the provider policy does not matter.
    policy = {k: v for k, v in reversed(list(served["provider_policy"].items()))}
    assert (
        protocol.gateway_settings_differ(
            {**served, "provider_policy": policy}, constants
        )
        == []
    )
    # Not a mapping: every setting the constants give differs; a constant they lack
    # is not judged.
    assert protocol.gateway_settings_differ(None, constants) == list(
        protocol.GATEWAY_SETTINGS_CONSTANTS
    )
    assert protocol.gateway_settings_differ({}, {"seed": 42}) == ["seed"]
