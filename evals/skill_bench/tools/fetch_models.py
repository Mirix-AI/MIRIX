"""Generate ``config/models.json`` from OpenRouter's free, key-less metadata endpoints.

Usage (from the repository root)::

    python -m evals.skill_bench.tools.fetch_models            # fetch and write
    python -m evals.skill_bench.tools.fetch_models --check    # fetch, compare, exit 1 on drift
    python -m evals.skill_bench.tools.fetch_models --chat-json a.json --embeddings-json b.json
    python -m evals.skill_bench.tools.fetch_models --keep-frozen  # add new models only

No API key is sent: ``/api/v1/models`` and ``/api/v1/embeddings/models`` are public.
The F9 numbers (the token limits) are derived with
:class:`evals.skill_bench.protocol.ModelLimits` and stored next to the raw metadata, so
every consumer reads the same values.

Every record carries the time its metadata was fetched (``fetched_at``); the top-level
``fetched_at`` is the latest fetch. ``--keep-frozen`` (:func:`merge_models_config`) writes
the live record of every model the frozen file lacks (a model added to
``protocol.MODELS``) and keeps every record the file already has as it was fetched, so
the cells that ran under those records keep the limits they ran with (their F9 numbers are recomputed from the kept fields, which only changes
them when the protocol's F9 rules changed). A live change of a kept model stays drift
that ``--check`` and preflight report (recorded, not blocking).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evals.skill_bench.protocol import (
    CHAT_MODELS,
    EMBEDDING_MODEL,
    MIN_OUTPUT_TOKENS,
    MODELS,
    MODELS_JSON,
    OUTPUT_FACTOR,
    PLANNING_RESERVE_WINDOW_FACTOR,
    REASONING_EFFORT,
    TOKEN_ENCODING,
    TOKENIZER_MARGIN,
    WINDOW_FACTOR,
    ModelLimits,
)

CHAT_MODELS_URL = "https://openrouter.ai/api/v1/models"
EMBEDDING_MODELS_URL = "https://openrouter.ai/api/v1/embeddings/models"
SCHEMA_VERSION = 1

# Raw fields copied verbatim from the OpenRouter entry.
_COPIED_FIELDS = (
    "name",
    "canonical_slug",
    "context_length",
    "top_provider",
    "pricing",
    "supported_parameters",
    "default_parameters",
    "reasoning",
)


def _index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(entry["id"]): entry for entry in payload.get("data", [])}


def _entry_sha256(entry: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        entry, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _model_record(
    entry: Mapping[str, Any], *, kind: str, source_url: str, fetched_at: str
) -> dict[str, Any]:
    arch = entry.get("architecture") or {}
    top = entry.get("top_provider") or {}
    record: dict[str, Any] = {
        "kind": kind,
        "source_url": source_url,
        "fetched_at": fetched_at,
    }
    record.update({k: entry.get(k) for k in _COPIED_FIELDS})
    record["max_completion_tokens"] = top.get("max_completion_tokens")
    record["input_modalities"] = list(arch.get("input_modalities") or [])
    record["output_modalities"] = list(arch.get("output_modalities") or [])
    record["raw_sha256"] = _entry_sha256(entry)
    return record


def _notes(model: str, record: Mapping[str, Any]) -> list[str]:
    """Facts about the metadata that the protocol relies on and a human should see."""
    notes = []
    top = record.get("top_provider") or {}
    top_ctx = top.get("context_length")
    if isinstance(top_ctx, int) and top_ctx < int(record["context_length"]):
        notes.append(
            f"{model}: top_provider.context_length {top_ctx} < context_length {record['context_length']}; "
            "F9 uses context_length; the gateway additionally keeps "
            "ceil(input * (1 + tokenizer_margin)) + max_tokens below the top provider "
            "window (provider_output_cap)."
        )
    efforts = (record.get("reasoning") or {}).get("supported_efforts")
    if isinstance(efforts, list) and REASONING_EFFORT not in efforts:
        notes.append(
            f"{model}: reasoning.supported_efforts {efforts} does not list {REASONING_EFFORT!r}; "
            "verify that the provider accepts it before a run."
        )
    if "seed" not in (record.get("supported_parameters") or []):
        notes.append(
            f"{model}: 'seed' not in supported_parameters; the gateway will not send it."
        )
    return notes


def build_models_config(
    chat_payload: Mapping[str, Any],
    embeddings_payload: Mapping[str, Any],
    *,
    fetched_at: str,
) -> dict[str, Any]:
    """Pure transformation from the two OpenRouter payloads to ``config/models.json``."""
    chat = _index(chat_payload)
    embeddings = _index(embeddings_payload)
    aliases = {model: alias for alias, model in MODELS.items()}
    models: dict[str, Any] = {}
    notes: list[str] = []
    for model in CHAT_MODELS:
        if model not in chat:
            raise KeyError(f"{model} not found at {CHAT_MODELS_URL}")
        record = _model_record(
            chat[model],
            kind="chat",
            source_url=CHAT_MODELS_URL,
            fetched_at=fetched_at,
        )
        record["alias"] = aliases[model]
        record["f9"] = _f9(model, record)
        models[model] = record
        notes.extend(_notes(model, record))
    if EMBEDDING_MODEL not in embeddings:
        raise KeyError(f"{EMBEDDING_MODEL} not found at {EMBEDDING_MODELS_URL}")
    record = _model_record(
        embeddings[EMBEDDING_MODEL],
        kind="embedding",
        source_url=EMBEDDING_MODELS_URL,
        fetched_at=fetched_at,
    )
    record["alias"] = "embedding"
    record["f9"] = None
    models[EMBEDDING_MODEL] = record
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": fetched_at,
        "sources": {"chat": CHAT_MODELS_URL, "embeddings": EMBEDDING_MODELS_URL},
        "f9_rules": f9_rules(),
        "models": models,
        "notes": notes,
    }


def _f9(model: str, record: Mapping[str, Any]) -> dict[str, int]:
    """The F9 numbers of a chat record (``protocol.ModelLimits``); ``ValueError`` when
    its limits are not integers."""
    if not isinstance(record.get("context_length"), int) or not isinstance(
        record.get("max_completion_tokens"), int
    ):
        raise ValueError(
            f"{model}: context_length and top_provider.max_completion_tokens must be integers"
        )
    top_ctx = (record.get("top_provider") or {}).get("context_length")
    return ModelLimits(
        model,
        record["context_length"],
        record["max_completion_tokens"],
        top_ctx if isinstance(top_ctx, int) else None,
    ).f9()


def merge_models_config(
    frozen: Mapping[str, Any], live: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """``(config, kept, fetched)``: ``live`` (a :func:`build_models_config`) with the
    record of every model ``frozen`` (the committed ``models.json``) already has kept as
    it is (pure).

    A kept record keeps its fields and its fetch time (a record written before records
    carried one gets the frozen file's ``fetched_at``); its F9 numbers are recomputed
    from its own limits under this protocol's rules, so they change only when the rules
    did. The models ``live`` adds (a new key of ``protocol.MODELS``) take their live
    record. A model the protocol no longer lists is dropped. A kept record whose alias
    is not the protocol's (a key renamed) is refused (``ValueError``): that is a change of
    what cells mean, for a full refresh. The notes are derived from the records as
    :func:`build_models_config` derives them."""
    before = frozen.get("models") if isinstance(frozen.get("models"), Mapping) else {}
    models: dict[str, Any] = {}
    kept: list[str] = []
    fetched: list[str] = []
    for model, record in live["models"].items():
        old = before.get(model)
        if not isinstance(old, Mapping):
            models[model] = record
            fetched.append(model)
            continue
        if old.get("alias") != record.get("alias"):
            raise ValueError(
                f"{model}: the frozen record's alias {old.get('alias')!r} is not the "
                f"protocol's {record.get('alias')!r}; refresh every record instead"
            )
        # The record's own key order (kind, source_url, fetched_at, ...), as built.
        merged = {key: old.get(key) for key in ("kind", "source_url")}
        merged["fetched_at"] = old.get("fetched_at", frozen.get("fetched_at"))
        merged.update((k, v) for k, v in old.items() if k not in merged)
        if merged.get("kind") == "chat":
            merged["f9"] = _f9(model, merged)
        models[model] = merged
        kept.append(model)
    notes = [
        note
        for model, record in models.items()
        if record.get("kind") == "chat"
        for note in _notes(model, record)
    ]
    return {**live, "models": models, "notes": notes}, kept, fetched


def f9_rules() -> dict[str, str]:
    """The formulas behind each model's ``f9`` numbers (``protocol.ModelLimits``)."""
    return {
        "token_encoding": TOKEN_ENCODING,
        "window80": f"floor(context_length * {WINDOW_FACTOR})",
        "max_out80": f"floor(max_completion_tokens * {OUTPUT_FACTOR})",
        "planning_reserve": f"min(max_out80, floor(context_length * {PLANNING_RESERVE_WINDOW_FACTOR}))",
        "distill_chunk_input_cap": "window80 - planning_reserve",
        "input_ceiling": "min(window80, top_provider.context_length)",
        "f9_output_cap": "min(max_out80, window80 - input_tokens)",
        "provider_output_cap": (
            "top_provider.context_length - ceil(input_tokens * "
            f"(1 + {TOKENIZER_MARGIN})) - 1"
        ),
        "output_cap": "min(f9_output_cap, provider_output_cap)",
        "gateway_input_budget": f"largest input_tokens with output_cap >= {MIN_OUTPUT_TOKENS}",
    }


def fetch_payloads(timeout_s: float = 60.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """GET both public metadata endpoints (no credentials)."""
    with httpx.Client(
        timeout=timeout_s, headers={"Accept": "application/json"}
    ) as client:
        chat = client.get(CHAT_MODELS_URL)
        chat.raise_for_status()
        emb = client.get(EMBEDDING_MODELS_URL)
        emb.raise_for_status()
        return chat.json(), emb.json()


def _without_timestamp(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k != "fetched_at"}


#: Top-level keys of ``models.json`` the protocol reads: the schema, where the metadata
#: comes from and the F9 formulas (derived from protocol constants).
PROTOCOL_TOP_LEVEL = ("schema_version", "sources", "f9_rules")


def protocol_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """The part of one model's record the protocol reads.

    The F9 limits (``context_length``, ``max_completion_tokens``, the top provider's
    window and output cap, and the ``f9`` numbers derived from them), the modalities,
    whether the gateway sends ``seed`` and the reasoning efforts the model lists
    (``reasoning.supported_efforts``). A live change to any of
    these changes what the gateway sends or how it caps a call: the frozen file no longer
    describes the model and the protocol must be refreshed and re-frozen. Everything else
    OpenRouter publishes (``raw_sha256`` of the raw entry, pricing, the name, the other
    supported parameters, ``is_moderated`` ...) the protocol does not read.
    """
    top = record.get("top_provider") or {}
    reasoning = record.get("reasoning") or {}
    return {
        "kind": record.get("kind"),
        "alias": record.get("alias"),
        "context_length": record.get("context_length"),
        "max_completion_tokens": record.get("max_completion_tokens"),
        "top_provider.context_length": top.get("context_length"),
        "top_provider.max_completion_tokens": top.get("max_completion_tokens"),
        "input_modalities": sorted(record.get("input_modalities") or []),
        "output_modalities": sorted(record.get("output_modalities") or []),
        "supported_parameters.seed": "seed"
        in (record.get("supported_parameters") or []),
        "reasoning.supported_efforts": reasoning.get("supported_efforts"),
        "f9": record.get("f9"),
    }


#: Record paths :func:`protocol_fields` reads in full (``supported_parameters`` is read
#: only for ``seed``: a change to its other entries is other drift).
_READ_PATHS = (
    "kind",
    "alias",
    "context_length",
    "max_completion_tokens",
    "top_provider.context_length",
    "top_provider.max_completion_tokens",
    "input_modalities",
    "output_modalities",
    "reasoning.supported_efforts",
    "f9",
)


def _changed_keys(before: Any, after: Any, prefix: str) -> list[str]:
    """``prefix.key`` for each key of two records that differs (one level into dicts)."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return [] if before == after else [prefix]
    names = []
    for key in sorted(set(before) | set(after), key=str):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            names += [
                f"{prefix}.{key}.{k}"
                for k in sorted(set(old) | set(new), key=str)
                if old.get(k) != new.get(k)
            ]
        else:
            names.append(f"{prefix}.{key}")
    return names


def compare_models_config(
    frozen: Mapping[str, Any], live: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """``(protocol drift, other drift)`` between the frozen ``models.json`` and one built
    from the live metadata (``fetched_at``, the file's and each record's, ignored).

    Protocol drift (:func:`protocol_fields`, :data:`PROTOCOL_TOP_LEVEL`, a model added or
    missing) means the frozen file must be refreshed and the protocol re-frozen. Other
    drift (``raw_sha256``, pricing, the derived ``notes`` ...) is reported and changes
    nothing: OpenRouter edits fields the protocol never reads, and re-freezing for them
    would split the matrix's cells across two fingerprints for no reason.
    """
    relevant: list[str] = []
    other: list[str] = []
    for key in sorted(set(frozen) | set(live)):
        if key in ("fetched_at", "models") or frozen.get(key) == live.get(key):
            continue
        (relevant if key in PROTOCOL_TOP_LEVEL else other).append(key)
    before = frozen.get("models") if isinstance(frozen.get("models"), Mapping) else {}
    after = live.get("models") if isinstance(live.get("models"), Mapping) else {}
    for model in sorted(set(before) | set(after)):
        if model not in before or model not in after:
            relevant.append(f"models.{model}")
            continue
        old, new = before[model], after[model]
        if old == new:
            continue
        prefix = f"models.{model}"
        old_view, new_view = protocol_fields(old), protocol_fields(new)
        relevant += [
            f"{prefix}.{name}" for name in old_view if old_view[name] != new_view[name]
        ]
        read = [f"{prefix}.{path}" for path in _READ_PATHS]
        other += [
            name
            for name in _changed_keys(old, new, prefix)
            if name != f"{prefix}.fetched_at"  # when, not what
            and not any(name == r or name.startswith(r + ".") for r in read)
        ]
    return relevant, other


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=MODELS_JSON)
    parser.add_argument(
        "--check", action="store_true", help="compare with --out instead of writing"
    )
    parser.add_argument(
        "--keep-frozen",
        action="store_true",
        help="keep every record --out already has; fetch only the models it lacks",
    )
    parser.add_argument(
        "--chat-json", type=Path, help="use a saved /api/v1/models payload"
    )
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        help="use a saved /api/v1/embeddings/models payload",
    )
    args = parser.parse_args(argv)

    if bool(args.chat_json) != bool(args.embeddings_json):
        parser.error("--chat-json and --embeddings-json go together")
    if args.check and args.keep_frozen:
        parser.error("--check compares; --keep-frozen writes")
    if args.chat_json:
        chat = json.loads(args.chat_json.read_text(encoding="utf-8"))
        emb = json.loads(args.embeddings_json.read_text(encoding="utf-8"))
    else:
        chat, emb = fetch_payloads()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = build_models_config(chat, emb, fetched_at=fetched_at)

    if args.check:
        frozen = json.loads(args.out.read_text(encoding="utf-8"))
        relevant, other = compare_models_config(frozen, config)
        if relevant:
            print(
                f"DRIFT: live OpenRouter metadata differs from {args.out} in what the "
                f"protocol reads: {', '.join(relevant)} (refresh and re-freeze)",
                file=sys.stderr,
            )
            return 1
        if other:
            print(
                f"OK: {args.out} matches the live metadata in every field the protocol "
                f"reads (frozen at {frozen.get('fetched_at')}); OpenRouter changed only "
                f"{', '.join(other)}, which needs no re-freeze"
            )
            return 0
        print(
            f"OK: {args.out} matches live metadata (frozen at {frozen.get('fetched_at')})"
        )
        return 0

    if args.keep_frozen:
        frozen = json.loads(args.out.read_text(encoding="utf-8"))
        config, kept, fetched = merge_models_config(frozen, config)
        print(f"kept as fetched before: {', '.join(kept) or 'none'}")
        print(f"fetched now: {', '.join(fetched) or 'none'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    for note in config["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
