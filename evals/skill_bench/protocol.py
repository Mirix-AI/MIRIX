"""Frozen protocol P1 and its fingerprint (H9, H10.6).

Everything here is a protocol decision, not a tuning knob. Changing a value,
a prompt/template file, a split manifest or ``config/models.json`` changes
:func:`protocol_fingerprint`. Versions are recorded and reported per cell, not enforced
across cells (a baseline fixed after its run re-runs only its own cells): ``audit.py``
passes a cell under the frozen fingerprint, warns about one under an earlier frozen
value (:func:`known_versions`, naming what changed: :func:`version_diff`) and rejects
one under an unknown fingerprint. An attempt records every version it ran
under (``attempt.json`` ``version_segments``, :func:`attempt_versions`; the version it
started under: :func:`attempt_protocol`); a launch under a tree that is not the frozen
value freezes it first (``tools/freeze_protocol.auto_freeze``).
The frozen record (``config/protocol_frozen.json``, or the copy
:data:`FROZEN_RECORD_ENV` names: :func:`frozen_record_path`) keeps the constants per key
(``constants``, next to their hash), so :func:`version_diff` names the constant keys
that changed, and a change limited to the keys that describe the study's composition
(:data:`COMPOSITION_CONSTANTS`) leaves the other cells auditable as they ran
(:func:`unauditable_changes`).

Fingerprinted files (paths relative to ``evals/skill_bench/``):

* ``config/models.json``;
* every file below a directory named ``prompts``, ``templates``, ``splits`` or
  ``vendor`` anywhere in the tree, except under ``data/``, ``runs/``, ``tests/``,
  ``__pycache__`` and dot-files. Environments and methods therefore keep their
  frozen prompt text, model-visible tool texts and limits, split manifests and
  vendored scoring code in such directories (for example
  ``envs/templates/alfworld/...``, ``envs/templates/webshop/format.json``,
  ``envs/splits/webshop/test.json`` or ``envs/vendor/webshop_projection.py``).

Code that only moves data around (the environments' and rollout's Python) is not
fingerprinted; run records carry the git commit for that.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

SKILL_BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILL_BENCH_DIR.parents[1]
MODELS_JSON = SKILL_BENCH_DIR / "config" / "models.json"
#: The frozen protocol (H10.6): the fingerprint every cell must carry and the per-file hashes
#: it was computed from. Written by ``tools/freeze_protocol.py``; not fingerprinted itself.
#: The repository's record; every reader and writer defaults to :func:`frozen_record_path`.
FROZEN_PROTOCOL_JSON = SKILL_BENCH_DIR / "config" / "protocol_frozen.json"
#: An environment variable naming another frozen record for every reader and writer that
#: is not given a path (:func:`frozen_record_path`): the tests point it at a copy, so no
#: test (and no automatic freeze a test's cell runs) ever writes the repository's record.
FROZEN_RECORD_ENV = "SKILL_BENCH_FROZEN_RECORD"
RUNS_DIR = SKILL_BENCH_DIR / "runs"
DATA_DIR = SKILL_BENCH_DIR / "data"

PROTOCOL_ID = "skill-bench-P1"
SKILLOPT_COMMIT = "79124b3"

# --- H1: data, seed, matrix -------------------------------------------------
SEED = 42
#: The benchmarks of the study. The list is a composition constant
#: (:data:`COMPOSITION_CONSTANTS`): adding or dropping a benchmark leaves the cells of the
#: others auditable as they ran.
BENCHMARKS: tuple[str, ...] = ("alfworld", "webshop")
SPLIT_SIZES: Mapping[str, Mapping[str, int]] = {
    "alfworld": {"train": 39, "val": 18, "test": 134},
    # WebShop: about 200 items, not the full set.
    "webshop": {"train": 40, "val": 20, "test": 140},
}
#: The models under test, by the short key cell ids use (H1.8). Always the model's own
#: id: never an alias such as ``~openai/gpt-luna-latest``, which OpenRouter re-points at
#: each new release.
#: ``luna`` is a prefix of ``luna56``: every reader matches a key whole (cell ids split
#: at ``-``, ``study.py`` tries the longest key first), never by prefix. Adding or
#: removing a key while the others keep their ids changes only the study's composition:
#: no cell of another model runs differently (:func:`constant_scope`).
MODELS: Mapping[str, str] = {
    "luna": "openai/gpt-6-luna",
    "luna56": "openai/gpt-5.6-luna",
}
CHAT_MODELS: tuple[str, ...] = tuple(MODELS.values())
EMBEDDING_MODEL = "google/gemini-embedding-001"
#: The memory methods the study compares. A method removed from the study leaves this
#: list, and its code and prompts the tree; ``study.REMOVED_METHODS`` records why, and
#: its cells on disk are set aside by the study tools. The list is a composition
#: constant (:data:`COMPOSITION_CONSTANTS`): changing it leaves the other cells auditable.
METHODS: tuple[str, ...] = (
    "none",
    "mirix",
    "reasoningbank",
    "awm",
)
ROLES: tuple[str, ...] = ("actor", "learner", "judge", "mirix", "embedding")

# --- H1.9 / H3: call parameters enforced by the gateway -----------------------
REASONING_EFFORT = "medium"
CALL_DEADLINE_S = 600.0  # total per call, retries included
GATEWAY_MAX_RETRIES = 5  # infrastructure failures only
GATEWAY_BACKOFF_BASE_S = 1.0
GATEWAY_BACKOFF_CAP_S = 60.0
DROPPED_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
)  # provider defaults
#: The only caller fields the gateway forwards on chat calls (H3). Every other field
#: (n, top_k, penalties, logit_bias, logprobs, presets, ...) is stripped and logged, so a
#: method cannot change sampling or routing behind the protocol's back.
CHAT_FORWARDED_FIELDS: tuple[str, ...] = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "stop",
    "parallel_tool_calls",
    "user",
    "functions",
    "function_call",
)
#: OpenRouter provider routing the gateway sets per chat model (``provider`` in the
#: request body; a caller's own ``provider`` is still dropped), as ``{model: {"ignore":
#: [slug, ...]}}``. Slugs are the endpoint tags' provider prefix (``GET
#: /api/v1/models/<model>/endpoints``). A model without an entry is not restricted; this
#: release restricts none.
PROVIDER_POLICY: Mapping[str, Mapping[str, tuple[str, ...]]] = {}
#: The ignored slugs' ``provider`` names as OpenRouter reports them (the ledger's spelling;
#: the audit matches either, :func:`provider_key`).
PROVIDER_SLUG_NAMES: Mapping[str, str] = {}
#: Every framework-side client (actor, MIRIX) waits longer than the gateway's deadline, so
#: it always receives the gateway's classified 504 instead of racing it with its own timeout.
CLIENT_TIMEOUT_MARGIN_S = 30.0
CLIENT_TIMEOUT_S = CALL_DEADLINE_S + CLIENT_TIMEOUT_MARGIN_S
#: H8 blame for a 504: a timeout is the model's only when its single upstream attempt had
#: (nearly) the whole deadline. Time lost before the attempt (queueing for a gateway slot,
#: the balance check, request preparation) above min(5 s, 25% of the deadline) makes the
#: timeout an infrastructure failure instead.
MODEL_TIMEOUT_MAX_OVERHEAD_S = 5.0
MODEL_TIMEOUT_MAX_OVERHEAD_FRACTION = Fraction(1, 4)

# --- F9: budget formulas -------------------------------------------------------
TOKEN_ENCODING = "o200k_base"
WINDOW_FACTOR = Fraction(4, 5)  # input + output <= floor(window * 0.8)
OUTPUT_FACTOR = Fraction(4, 5)  # output <= floor(max_output * 0.8)
PLANNING_RESERVE_WINDOW_FACTOR = Fraction(
    1, 10
)  # reserve = min(max_out80, floor(window * 0.1))
MIN_OUTPUT_TOKENS = (
    1024  # an output cap below this is a budget error, never a silent truncation
)
LARGE_PAYLOAD_CHARS = 50_000  # token counting above this size runs in a worker thread
#: Safety margin for the provider-window clamp. Every model is counted with o200k,
#: which may undercount another tokenizer. Where a provider's hard window binds,
#: the gateway assumes the real input is up to (1 + margin) x the estimate, so that
#: ``input + max_tokens`` stays below ``top_provider.context_length``:
#: ``max_tokens <= context_length - ceil(input * (1 + margin)) - 1``. The F9 numbers are
#: unchanged; only calls whose output room the provider window limits get a smaller cap.
TOKENIZER_MARGIN = Fraction(1, 10)
#: Input-estimate constants (infra/tokens.py): they set the rewritten max_tokens of every call.
TOKEN_ESTIMATE: Mapping[str, int] = {
    "fallback_chars_per_token": 3,
    "message_overhead_tokens": 3,  # per message (OpenAI chat format)
    "reply_priming_tokens": 3,
    "tool_call_overhead_tokens": 3,
}

# --- H4: episodes --------------------------------------------------------------
ALFWORLD_MAX_STEPS = 50
#: The game engine packages the ALFWorld results depend on (parsing, admissible commands,
#: observations). pyproject.toml pins the same versions; the env refuses others.
ALFWORLD_PACKAGES: Mapping[str, str] = {"alfworld": "0.4.2", "textworld": "1.7.0"}

# --- WebShop ---------------------------------------------------------------------------
#: An episode ends at Buy Now or after this many steps; the prompt does not state it. 15
#: follows verl-agent's WebShop runs (every ``examples/*/run_webshop*.sh`` of 20bd331 sets
#: ``env.max_steps=15``; ``envs/templates/webshop/format.json`` quotes it).
WEBSHOP_MAX_STEPS = 15
#: Success within this many steps is reported as a diagnostic (the 15-step lineage of
#: ReAct, Reflexion, ExpeL and verl-agent). With the cap at 15 it equals the success
#: rate; it stays in the records and the report, so that the lineage's number is named.
WEBSHOP_DIAGNOSTIC_STEPS = 15
#: Task membership is fixed by the released JSON manifests and their SHA-256 pins.
#: The human-instruction goals of the full product set (12,087 instructions;
#: upstream ``baseline_models/data/human_goals.json`` holds them in goal order).
WEBSHOP_GOAL_COUNT = 12087
#: The WebShop engine every cell's server runs. The image is
#: ``envs/templates/webshop_server/Dockerfile`` FROM ``base_image``; ``server.py`` there
#: seeds ``random`` with :data:`SEED` immediately before ``SimServer()`` is built and
#: refuses to serve when the frozen manifest or the price table differ.
WEBSHOP_SERVER: Mapping[str, Any] = {
    "upstream_repo": "https://github.com/princeton-nlp/WebShop",
    "upstream_commit": "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd",
    # python:3.9-bullseye (Debian bullseye ships openjdk-11; bookworm does not), pinned
    # by its multi-arch index digest.
    "base_image": "python:3.9-bullseye@sha256:3dcaf39bbe2d6d8031a4a7ce68a20f2d494419aca18ec064f9e82e146e97a0ed",
    "java": "openjdk-11-jdk-headless",
    "observation_mode": "text",
    "human_goals": True,
    "num_products": None,
    "seed": SEED,
    "spacy_model": "en_core_web_sm==3.3.0",
    # thefuzz without python-Levenshtein falls back to difflib's SequenceMatcher.
    "fuzzy_backend": "difflib",
    "pyserini": "0.17.0",
    "index_threads": 1,
    "memory": "24g",
    "cpus": 2,
    "container_port": 8000,
    # Sessions idle this long are dropped by the server (an episode a dead harness left).
    "session_idle_s": 3600,
}
#: The product data (a Hugging Face mirror: the upstream Google Drive files are not
#: reachable),
#: pinned by revision and each file's size and sha256. ``items_human_ins.json`` is
#: byte-identical to upstream's ``baseline_models/data/items_human_ins.json``;
#: ``human_goals`` is upstream's ``baseline_models/data/human_goals.json`` (the goal order).
WEBSHOP_DATA: Mapping[str, Any] = {
    "hf_repo": "YWZBrandon/webshop-data",
    "hf_revision": "ce990fff5aee388db2706f07820c578ab68e0453",
    "files": {
        "items_shuffle.json": {
            "size": 5479720229,
            "sha256": "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
        },
        "items_ins_v2.json": {
            "size": 186295270,
            "sha256": "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
        },
        "items_human_ins.json": {
            "size": 5137548,
            "sha256": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
        },
    },
    "human_goals": {
        "path": "baseline_models/data/human_goals.json",
        "sha256": "b68746ed66cd31fdc5f70eb3f5831a46b38163563b57a66ddcab8fd60ee0cbdc",
    },
}
#: WebShop frozen files: the split manifests (written by
#: ``tools/prepare_webshop.py``), the actor-prompt pieces (verl-agent 20bd331), the server
#: image's Docker context and the vendored projection. ``prepare_webshop pin`` rewrites
#: this block from the files (after ``freeze``, or any change to them).
# BEGIN WEBSHOP_ASSET_SHA256
WEBSHOP_ASSET_SHA256: Mapping[str, str] = {
    "envs/splits/webshop/source.json": "5d1a3f0987386f2e094a3ca3aa4d52488c19501ee539c81d0290a16ae091883f",
    "envs/splits/webshop/split_manifest.json": "8a95047feb15e5127ddd348a489bb7461c0fe43d02e52dae5367b69e5db9da8b",
    "envs/splits/webshop/test.json": "fdc01f7d661efd0283901b81be3feef1504f824a7ac76db415840934c5cf797c",
    "envs/splits/webshop/train.json": "5d25c75fd9c89408d7783155ff2c9e63a292ec7bcce39cac997ec620cca2452c",
    "envs/splits/webshop/val.json": "51fd28e7fe63d1834bd4912d49bb3dc7b4bce032ba5bc0ce54934ac77556325e",
    "envs/templates/webshop/format.json": "c22b1bf781389e533128d04bc8b658dfe4e815573e846d8dfeff52bd7a4344a0",
    "envs/templates/webshop/rollout_no_history.md": "2f43c909547409ee09280e57d37514ef3bd63ce53e06d1c41c554d6abaf7ba26",
    "envs/templates/webshop/rollout_turn.md": "e393035378c2544ba16d83b32fd606c2bfdeb4090874d54439494016b29b844f",
    "envs/templates/webshop/rollout_with_history.md": "e433317451d83c183cef41a13a6765d2c77331464c060d0c3e1dcb492f6c3428",
    "envs/templates/webshop_server/Dockerfile": "11cc8f59b65319cc553e5617b014032ab5a6df28eed8b1717c1b5b5d28d4f035",
    "envs/templates/webshop_server/data.json": "b2d2a90cf0add8a5c5da92ad1d9eb1e57de6a3353689aab2033489f29b24b1ca",
    "envs/templates/webshop_server/data_index.py": "b71f3384bff83478d72196fe6846c2c7d93d035878da39dfa8b48f2c0db55302",
    "envs/templates/webshop_server/requirements.in": "1ae9cbd88d15b315ae36994a9789703ae19dde589e6c32e590658a6e7ddc5fef",
    "envs/templates/webshop_server/requirements.txt": "65995322e9eb9a5b1bb0b384c254d07bf9626d7fd4c4a81dbafda87fbbc5e148",
    "envs/templates/webshop_server/server.py": "2be95578e43dc3951b6716f6b7cee56bd8b61d769aedd113f6d79cf87b68229d",
    "envs/vendor/webshop_projection.py": "2a2eded1a4db414954f9e0dc8b353410aadd840ffc47b8cea955fa1c2a41f81e",
}
# END WEBSHOP_ASSET_SHA256

# --- H5: opaque task ids ----------------------------------------------------------
OPAQUE_UID_PREFIX = "t"
OPAQUE_UID_HEX_CHARS = 20

# --- H8 / H9: isolation, scheduling, statistics --------------------------------------
#: MIRIX's session idle timeout in a cell: ten years, i.e. never within a phase however
#: long a resume waits (MIRIX's default is 172,800 s; see MIRIX_SERVER_ENV).
MIRIX_SESSION_IDLE_TIMEOUT_S = 10 * 365 * 86_400
MIRIX_SERVER_ENV: Mapping[str, str] = {
    "MIRIX_REDIS_ENABLED": "false",
    # H7.2: the gateway alone retries, inside its one deadline.
    "MIRIX_LLM_MAX_RETRIES": "0",
    # Longer than the gateway deadline (CLIENT_TIMEOUT_S), so MIRIX always sees the
    # gateway's classified 504 rather than racing it with its own read timeout.
    "MIRIX_LLM_TIMEOUT_SECONDS": f"{CLIENT_TIMEOUT_S:g}",
    # The same for MIRIX's embedding calls (default 60 s): the gateway may hold one for
    # up to its deadline (Retry-After, back-off, a slot, a balance pause). A shorter
    # timeout abandons the held call, which runs on and is billed, and MIRIX's own
    # retry sends it again unlinked to it.
    "MIRIX_EMBEDDING_TIMEOUT_SECONDS": f"{CLIENT_TIMEOUT_S:g}",
    # A task session never ends by idling: the harness closes every task session
    # itself and finalize flushes what is open, so an idle end could only come from a
    # learning resume restoring a played checkpoint more than the default 48 h after it
    # was taken. The resumed server's startup sweep would then close the session as idle
    # (and could fire a sweep batch), and after_task's close would answer
    # already_closed: a HARNESS stop only relearning gets past.
    "MIRIX_SESSION_IDLE_TIMEOUT_SECONDS": f"{MIRIX_SESSION_IDLE_TIMEOUT_S}",
}
#: The MIRIX settings (``mirix.settings.settings``, env prefix ``MIRIX_``) that must read the
#: variables above; ``infra/mirix_server.py`` probes them before a cell starts.
MIRIX_SETTINGS_CONTRACT: Mapping[str, str] = {
    "llm_max_retries": "MIRIX_LLM_MAX_RETRIES",
    "llm_timeout_seconds": "MIRIX_LLM_TIMEOUT_SECONDS",
    "embedding_timeout_seconds": "MIRIX_EMBEDDING_TIMEOUT_SECONDS",
}
#: The same for the module constants of ``mirix.constants`` that read a variable above.
MIRIX_CONSTANTS_CONTRACT: Mapping[str, str] = {
    "SESSION_IDLE_TIMEOUT_SECONDS": "MIRIX_SESSION_IDLE_TIMEOUT_SECONDS",
}
#: Cells one scheduler runs at once. An operating limit, not a condition any cell ran
#: under (the host registry's memory gate, the host-wide WebShop cap and the memory
#: emergency keep a host safe), so a change of it touches no cell
#: (:data:`OPERATING_CONSTANTS`).
SCHEDULE_CONCURRENCY_RANGE: tuple[int, int] = (1, 24)
LEARNING_CURVE_WINDOW = 10
BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95

# --- H4.1 / H4.3: frozen third-party assets ------------------------------------------
# sha256 of every file copied or derived from SkillOpt 79124b3 / ALFWorld.
# The byte-exact copies carry the upstream file's own hash. Environments call
# verify_frozen_assets() before using any of them.
FROZEN_ASSET_SHA256: Mapping[str, str] = {
    # sha256 of the 191 manifest games and the shared logic files (alfworld-download).
    "envs/splits/alfworld/gamefiles.sha256.json": "e70b1c463c985ead072e50ddc7ffa58546cefc1a99089153dc624ac5f2a9b4f1",
    "envs/splits/alfworld/split_manifest.json": "9a6e0e267e5ac24a53a70c312ee62c20b85afc30c8635c138b164ec91bca3978",
    "envs/splits/alfworld/test.json": "34fc9439b0555e9bea76e2a7d1947953250e5a0cd9363d75ea64c2605ac0c33e",
    "envs/splits/alfworld/train.json": "76dc1f738fafc9c3c0dae606eb9434b1e1a34b814874e4d791242f9ec90e6378",
    "envs/splits/alfworld/val.json": "1533c56bd98577314a2ed12daffb9be4c1e1de2b052b61d0093b865b753da3f9",
    "envs/templates/alfworld/config_tw.yaml": "f32e8244351d896ec9073aa68474d8adee3ef41ee27a920fef601fba04643725",
    "envs/templates/alfworld/format.json": "a5661df432088348e29888704ac254a3f7e6272e19568e561bbf346268c7e5fc",
    "envs/templates/alfworld/rollout_no_history.md": "fcf6fe7a561729ccc4581697f5c8ad5986c150e56e32d761bad944b2fc9da614",
    "envs/templates/alfworld/rollout_with_history.md": "7b17c85e102f3e49400df99bd3f333337827df9441abd873926a566ad9858027",
    **WEBSHOP_ASSET_SHA256,  # WebShop, defined above
    **{
        # Vendored scoring and parsing code (whole files, headers included). Each
        # environment checks its own modules when it is constructed.
        "envs/vendor/alfworld_projection.py": "8fe960e5316a4078a7101d2a3d5d3b2fcd867346f8214a27fec025a76cfb5f31",
    },
}

# --- fingerprint scope -------------------------------------------------------------
#: ``vendor`` holds third-party scoring and parsing code (the ALFWorld and WebShop action
#: projections): a change there changes scores, so it is protocol too.
FINGERPRINT_DIR_NAMES: frozenset[str] = frozenset(
    {"prompts", "templates", "splits", "vendor"}
)
FINGERPRINT_EXCLUDED_TOP: frozenset[str] = frozenset({"data", "runs", "tests"})


class ProtocolDriftError(RuntimeError):
    """The protocol fingerprint differs from the frozen value."""


# ---------------------------------------------------------------------------
# F9 budget formulas (exact integer arithmetic; 0.8 == 4/5 without float error)
# ---------------------------------------------------------------------------


def _floor_mul(value: int, factor: Fraction) -> int:
    return value * factor.numerator // factor.denominator


def _ceil_mul(value: int, factor: Fraction) -> int:
    return -(-value * factor.numerator // factor.denominator)


def _floor_div(value: int, factor: Fraction) -> int:
    """floor(value / factor), exactly."""
    return value * factor.denominator // factor.numerator


@dataclass(frozen=True)
class ModelLimits:
    """Context window and maximum output of one chat model, with the F9 numbers derived from them.

    ``provider_context_length`` is OpenRouter's ``top_provider.context_length``. When it
    is below window80, the F9 cap alone would let ``input + max_tokens`` exceed the top
    provider's window. And the input is an o200k estimate, which may undercount the
    provider's own tokenizer. The gateway therefore also keeps
    ``ceil(input * (1 + tokenizer_margin)) + max_tokens`` below that window
    (:meth:`provider_output_cap`). The F9 numbers are unchanged; only calls whose output
    room the provider window limits get a smaller cap.
    """

    model: str
    context_length: int
    max_completion_tokens: int
    provider_context_length: int | None = None
    tokenizer_margin: Fraction = TOKENIZER_MARGIN

    @property
    def window80(self) -> int:
        """floor(window * 0.8): the ceiling for input + output of any call."""
        return _floor_mul(self.context_length, WINDOW_FACTOR)

    @property
    def max_out80(self) -> int:
        """floor(max_output * 0.8)."""
        return _floor_mul(self.max_completion_tokens, OUTPUT_FACTOR)

    @property
    def planning_reserve(self) -> int:
        """Output reserve callers plan with: min(max_out80, floor(window * 0.1))."""
        return min(
            self.max_out80,
            _floor_mul(self.context_length, PLANNING_RESERVE_WINDOW_FACTOR),
        )

    @property
    def distill_chunk_input_cap(self) -> int:
        """Distiller chunk input ceiling before the system prompt: window80 - planning_reserve."""
        return self.window80 - self.planning_reserve

    @property
    def input_ceiling(self) -> int:
        """The hard ceiling for input + output: window80, or the top provider's window if smaller."""
        if self.provider_context_length is None:
            return self.window80
        return min(self.window80, self.provider_context_length)

    @property
    def gateway_input_budget(self) -> int:
        """Largest input the gateway forwards: it must leave MIN_OUTPUT_TOKENS of output room."""
        return self.max_input_for_output(MIN_OUTPUT_TOKENS)

    def f9_output_cap(self, input_tokens: int) -> int:
        """The F9 formula alone: min(max_out80, window80 - input_tokens)."""
        return min(self.max_out80, self.window80 - input_tokens)

    def provider_output_cap(self, input_tokens: int) -> int | None:
        """The top provider's window with the tokenizer margin, or None if unknown:
        ``provider_context_length - ceil(input * (1 + tokenizer_margin)) - 1``."""
        if self.provider_context_length is None:
            return None
        counted = _ceil_mul(input_tokens, 1 + self.tokenizer_margin)
        return self.provider_context_length - counted - 1

    def output_cap(self, input_tokens: int) -> int:
        """Output cap for one call: F9, also kept below the top provider's window (with margin)."""
        cap = self.f9_output_cap(input_tokens)
        provider = self.provider_output_cap(input_tokens)
        return cap if provider is None else min(cap, provider)

    def output_cap_bound(self, input_tokens: int) -> str:
        """Which limit sets :meth:`output_cap`: ``max_out80``, ``window80`` or ``provider_window``."""
        cap = self.output_cap(input_tokens)
        if cap == self.max_out80:
            return "max_out80"
        if cap == self.window80 - input_tokens:
            return "window80"
        return "provider_window"

    def max_input_for_output(self, output_tokens: int) -> int:
        """Largest input whose :meth:`output_cap` still allows ``output_tokens`` (-1 if none)."""
        if output_tokens > self.max_out80:
            return -1
        best = self.window80 - output_tokens
        if self.provider_context_length is not None:
            room = self.provider_context_length - 1 - output_tokens
            best = min(best, _floor_div(room, 1 + self.tokenizer_margin))
        return max(best, -1)

    def input_budget(self, system_tokens: int, reserve: int | None = None) -> int:
        """Input budget: window80 - system prompt - output reserve (planning reserve by default)."""
        return (
            self.window80
            - system_tokens
            - (self.planning_reserve if reserve is None else reserve)
        )

    def f9(self) -> dict[str, int]:
        return {
            "window80": self.window80,
            "max_out80": self.max_out80,
            "planning_reserve": self.planning_reserve,
            "distill_chunk_input_cap": self.distill_chunk_input_cap,
            "input_ceiling": self.input_ceiling,
            "gateway_input_budget": self.gateway_input_budget,
        }


def provider_routing(
    model: str, policy: Mapping[str, Mapping[str, Sequence[str]]] | None = None
) -> dict[str, list[str]] | None:
    """The ``provider`` object the gateway sends with ``model``'s chat calls
    (:data:`PROVIDER_POLICY` unless ``policy`` is given), JSON-ready with its lists in
    order; None when the model's provider choice is not restricted."""
    entry = (PROVIDER_POLICY if policy is None else policy).get(model)
    if not entry:
        return None
    return {str(k): [str(v) for v in values] for k, values in sorted(entry.items())}


def provider_policy_settings(
    policy: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Every model's routing (:func:`provider_routing`) as one JSON-ready mapping: what the
    gateway's settings, its start records and the audit compare."""
    chosen = PROVIDER_POLICY if policy is None else policy
    return {
        model: routing
        for model in sorted(chosen)
        if (routing := provider_routing(model, chosen)) is not None
    }


def provider_key(name: Any) -> str:
    """A provider slug or name compared the audit's way: lower case, only letters and
    digits (``OpenInference`` and ``open-inference`` both give ``openinference``)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def ignored_provider(model: str, provider: Any) -> str | None:
    """The ignored slug of ``model``'s :data:`PROVIDER_POLICY` that ``provider`` (a
    ledger ``provider``, a name or a slug) is, else None. A provider is named by its slug
    or its :data:`PROVIDER_SLUG_NAMES` name, compared with :func:`provider_key`."""
    return ignored_by(PROVIDER_POLICY.get(model), provider)


def ignored_by(routing: Any, provider: Any) -> str | None:
    """The slug of ``routing``'s ``ignore`` list (a :func:`provider_routing` object, or
    None) that ``provider`` is, else None (see :func:`ignored_provider`): what the
    gateway checks against the routing it sent with the call."""
    if (
        provider is None
        or not str(provider).strip()
        or not isinstance(routing, Mapping)
    ):
        return None
    key = provider_key(provider)
    for slug in routing.get("ignore") or ():
        names = {provider_key(slug), provider_key(PROVIDER_SLUG_NAMES.get(slug, slug))}
        if key in names:
            return str(slug)
    return None


def load_models(path: Path = MODELS_JSON) -> dict[str, Any]:
    """Read the frozen model metadata written by ``tools/fetch_models.py``."""
    return json.loads(path.read_text(encoding="utf-8"))


#: ``config/models.json`` as a version's ``files`` map names it.
MODELS_JSON_REL = "config/models.json"


def canonical_slugs(models: Mapping[str, Any]) -> dict[str, str]:
    """Model id -> ``canonical_slug`` of a loaded ``config/models.json`` (pure): the
    dated snapshot OpenRouter may serve each model as (H1.8)."""
    entries = models.get("models") if isinstance(models, Mapping) else None
    return {
        str(model): str(entry["canonical_slug"])
        for model, entry in sorted((entries or {}).items())
        if isinstance(entry, Mapping)
        and isinstance(entry.get("canonical_slug"), str)
        and entry["canonical_slug"]
    }


def model_slugs_record(root: Path = SKILL_BENCH_DIR) -> dict[str, dict[str, str]]:
    """``{sha256 of config/models.json: {model id: canonical_slug}}`` of the tree at
    ``root``, hashed and parsed from one read of the file (blocking). Keyed by the hash
    a version's ``files`` map gives the file, so the audit resolves what each model was
    requested as under a version whose ``models.json`` is no longer this tree's (the
    frozen record's and each ``version.json``'s ``model_slugs``)."""
    data = (root / MODELS_JSON_REL).read_bytes()
    return {_sha256(data): canonical_slugs(json.loads(data))}


def model_slugs_of(
    manifest: Mapping[str, Any], root: Path = SKILL_BENCH_DIR
) -> dict[str, dict[str, str]]:
    """:func:`model_slugs_record` of the tree at ``root`` when its models.json is the one
    ``manifest`` (a :func:`fingerprint_manifest`) hashed, else ``{}``: the file changed
    since, and another file's slugs are never recorded under this hash. Blocking; never
    raises for an unreadable file."""
    try:
        found = model_slugs_record(root)
    except (OSError, ValueError):
        return {}
    wanted = (manifest.get("files") or {}).get(MODELS_JSON_REL)
    return {sha: slugs for sha, slugs in found.items() if sha == wanted}


def recorded_model_slugs(document: Any) -> dict[str, dict[str, str]]:
    """The well-formed ``model_slugs`` a frozen record or a ``version.json`` keeps
    (models.json sha256 -> model id -> canonical_slug), ``{}`` for none (pure)."""
    found = document.get("model_slugs") if isinstance(document, Mapping) else None
    out: dict[str, dict[str, str]] = {}
    if not isinstance(found, Mapping):
        return out
    for sha, slugs in found.items():
        if isinstance(sha, str) and isinstance(slugs, Mapping):
            out[sha] = {
                str(model): str(slug)
                for model, slug in slugs.items()
                if isinstance(slug, str) and slug
            }
    return out


def model_limits(
    model: str,
    models: Mapping[str, Any] | None = None,
    *,
    tokenizer_margin: Fraction = TOKENIZER_MARGIN,
) -> ModelLimits:
    """F9 limits of a whitelisted chat model from ``config/models.json``."""
    entries = (models if models is not None else load_models())["models"]
    if model not in entries or entries[model].get("kind") != "chat":
        raise KeyError(f"{model!r} is not a chat model in config/models.json")
    entry = entries[model]
    top = entry.get("top_provider") or {}
    top_ctx = top.get("context_length")
    return ModelLimits(
        model,
        int(entry["context_length"]),
        int(entry["max_completion_tokens"]),
        int(top_ctx) if isinstance(top_ctx, int) else None,
        Fraction(tokenizer_margin),
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def protocol_constants() -> dict[str, Any]:
    """Every frozen constant, in a JSON-stable form."""
    return {
        "protocol_id": PROTOCOL_ID,
        "skillopt_commit": SKILLOPT_COMMIT,
        "seed": SEED,
        "benchmarks": list(BENCHMARKS),
        "split_sizes": {b: dict(s) for b, s in SPLIT_SIZES.items()},
        "models": dict(MODELS),
        "embedding_model": EMBEDDING_MODEL,
        "methods": list(METHODS),
        "roles": list(ROLES),
        "reasoning_effort": REASONING_EFFORT,
        "call_deadline_s": CALL_DEADLINE_S,
        "gateway_max_retries": GATEWAY_MAX_RETRIES,
        "gateway_backoff_base_s": GATEWAY_BACKOFF_BASE_S,
        "gateway_backoff_cap_s": GATEWAY_BACKOFF_CAP_S,
        "dropped_sampling_params": list(DROPPED_SAMPLING_PARAMS),
        "chat_forwarded_fields": list(CHAT_FORWARDED_FIELDS),
        "provider_policy": provider_policy_settings(),
        "provider_slug_names": dict(sorted(PROVIDER_SLUG_NAMES.items())),
        "client_timeout_s": CLIENT_TIMEOUT_S,
        "model_timeout_max_overhead_s": MODEL_TIMEOUT_MAX_OVERHEAD_S,
        "model_timeout_max_overhead_fraction": str(MODEL_TIMEOUT_MAX_OVERHEAD_FRACTION),
        "token_encoding": TOKEN_ENCODING,
        "token_estimate": dict(TOKEN_ESTIMATE),
        "window_factor": str(WINDOW_FACTOR),
        "output_factor": str(OUTPUT_FACTOR),
        "planning_reserve_window_factor": str(PLANNING_RESERVE_WINDOW_FACTOR),
        "min_output_tokens": MIN_OUTPUT_TOKENS,
        "tokenizer_margin": str(TOKENIZER_MARGIN),
        "large_payload_chars": LARGE_PAYLOAD_CHARS,
        "alfworld_max_steps": ALFWORLD_MAX_STEPS,
        "alfworld_packages": dict(ALFWORLD_PACKAGES),
        "webshop_max_steps": WEBSHOP_MAX_STEPS,
        "webshop_diagnostic_steps": WEBSHOP_DIAGNOSTIC_STEPS,
        "webshop_goal_count": WEBSHOP_GOAL_COUNT,
        "webshop_server": dict(WEBSHOP_SERVER),
        "webshop_data": copy.deepcopy(dict(WEBSHOP_DATA)),
        "opaque_uid": {"prefix": OPAQUE_UID_PREFIX, "hex_chars": OPAQUE_UID_HEX_CHARS},
        "mirix_server_env": dict(MIRIX_SERVER_ENV),
        "mirix_settings_contract": dict(MIRIX_SETTINGS_CONTRACT),
        "mirix_constants_contract": dict(MIRIX_CONSTANTS_CONTRACT),
        "schedule_concurrency_range": list(SCHEDULE_CONCURRENCY_RANGE),
        "learning_curve_window": LEARNING_CURVE_WINDOW,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_level": CI_LEVEL,
        "frozen_asset_sha256": dict(sorted(FROZEN_ASSET_SHA256.items())),
    }


#: Each gateway setting (``infra/gateway.GatewayConfig.protocol_settings``, what its
#: ``/health`` and start record report) -> the :func:`protocol_constants` key it is frozen
#: as (the audit's ``SETTINGS_CONSTANTS``).
GATEWAY_SETTINGS_CONSTANTS: Mapping[str, str] = {
    "deadline_s": "call_deadline_s",
    "max_retries": "gateway_max_retries",
    "backoff_base_s": "gateway_backoff_base_s",
    "backoff_cap_s": "gateway_backoff_cap_s",
    "reasoning_effort": "reasoning_effort",
    "seed": "seed",
    "tokenizer_margin": "tokenizer_margin",
    "provider_policy": "provider_policy",
}
#: The gateway settings that shape what the model is asked (H10.6-gateway-settings, the
#: audit's ``MODEL_SETTINGS``): calls a gateway sent under others than the attempt's
#: version records fail the audit (``gateway_start_settings``); the rest are recorded.
GATEWAY_MODEL_SETTINGS: tuple[str, ...] = (
    "reasoning_effort",
    "seed",
    "provider_policy",
)


def gateway_settings_of(constants: Mapping[str, Any]) -> dict[str, Any]:
    """The gateway settings ``constants`` (a :func:`protocol_constants`, or a version's
    per-key constants) give, in the gateway's own form (pure); a setting whose constant
    ``constants`` lacks is left out."""
    return {
        setting: constants[key]
        for setting, key in GATEWAY_SETTINGS_CONSTANTS.items()
        if key in constants
    }


def gateway_settings_differ(
    served: Any, constants: Mapping[str, Any], keys: Sequence[str] | None = None
) -> list[str]:
    """The gateway settings (of ``keys``, default all, in that order) whose value in
    ``served`` (a gateway's ``/health`` ``settings``) is not the one ``constants`` give,
    compared as canonical JSON (pure). A setting missing from ``served`` differs; one the
    constants do not give is not judged."""
    served = served if isinstance(served, Mapping) else {}
    expected = gateway_settings_of(constants)
    return [
        key
        for key in (GATEWAY_SETTINGS_CONSTANTS if keys is None else keys)
        if key in expected
        and (key not in served or _canonical(served[key]) != _canonical(expected[key]))
    ]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_fingerprinted(rel_path: str | Path) -> bool:
    """Whether a path relative to ``evals/skill_bench`` is covered by the fingerprint."""
    parts = Path(rel_path).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        return False
    if parts == ("config", "models.json"):
        return True
    if parts[0] in FINGERPRINT_EXCLUDED_TOP:
        return False
    if any(p.startswith(".") or p == "__pycache__" for p in parts):
        return False
    return any(p in FINGERPRINT_DIR_NAMES for p in parts[:-1])


def fingerprint_files(root: Path = SKILL_BENCH_DIR) -> list[Path]:
    """Files covered by the fingerprint, sorted by their POSIX path relative to ``root``."""
    models = root / "config" / "models.json"
    if not models.is_file():
        raise FileNotFoundError(f"protocol requires {models}")
    found = {models}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root).parts
        # Prune before descending: runs/ and data/ can hold many large files.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d != "__pycache__"
            and not (not rel_dir and d in FINGERPRINT_EXCLUDED_TOP)
        )
        found.update(here / f for f in filenames if is_fingerprinted(Path(*rel_dir, f)))
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def protocol_asset(rel_path: str | Path, root: Path = SKILL_BENCH_DIR) -> Path:
    """Path of a frozen prompt/template/split file; refuses files the fingerprint would miss.

    Environments and methods load their prompt text through this (or
    :func:`read_protocol_text`) so every prompt that reaches a model is fingerprinted.
    One recorded exception: the framing words of the agent context
    (``envs/agent_context.py``: the intro, section titles and placeholder
    descriptions around the fingerprinted templates) are harness code; every attempt
    records the text's digest (``attempt.json`` ``agent_context``) and
    ``tools/aggregate_replicates.py`` compares it across replicates and methods.
    """
    if Path(rel_path).is_absolute() or not is_fingerprinted(rel_path):
        raise ValueError(
            f"{rel_path} is not covered by the protocol fingerprint; keep prompt, template "
            "and split files below a prompts/, templates/ or splits/ directory"
        )
    return root / rel_path


def read_protocol_text(rel_path: str | Path, root: Path = SKILL_BENCH_DIR) -> str:
    return protocol_asset(rel_path, root).read_text(encoding="utf-8")


def verify_frozen_assets(prefix: str = "", root: Path = SKILL_BENCH_DIR) -> list[str]:
    """Check every frozen asset whose path starts with ``prefix`` against its recorded sha256.

    Returns the checked paths; raises :class:`ProtocolDriftError` naming each file that is
    missing or changed, or when ``prefix`` matches no recorded asset.
    """
    selected = sorted(p for p in FROZEN_ASSET_SHA256 if p.startswith(prefix))
    if not selected:
        raise ProtocolDriftError(f"no frozen assets recorded under {prefix!r}")
    problems = []
    for rel in selected:
        path = protocol_asset(rel, root)
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif (actual := _sha256(path.read_bytes())) != FROZEN_ASSET_SHA256[rel]:
            problems.append(
                f"{rel}: sha256 {actual} != frozen {FROZEN_ASSET_SHA256[rel]}"
            )
    if problems:
        raise ProtocolDriftError("frozen assets changed: " + "; ".join(problems))
    return selected


def constants_sha256(constants: Mapping[str, Any]) -> str:
    """The ``constants_sha256`` of a set of protocol constants (key -> JSON value): sha256
    of their canonical JSON, the hash every frozen record and cell carries."""
    return _sha256(_canonical(dict(constants)))


def fingerprint_of(constants: Mapping[str, Any], files: Mapping[str, str]) -> str:
    """The protocol fingerprint of some constants (key -> JSON value) and a file map
    (path -> sha256): sha256 of their canonical JSON. A frozen record's per-key
    ``constants`` and ``files`` are read only when they reproduce its ``fingerprint``
    this way (:func:`recorded_constants`): the fingerprint is what every cell carries."""
    document = {"constants": dict(constants), "files": sorted(files.items())}
    return _sha256(_canonical(document))


def fingerprint_manifest(root: Path = SKILL_BENCH_DIR) -> dict[str, Any]:
    """Constants (per key and their hash), per-file hashes and the resulting fingerprint
    (for reports and audits). The fingerprint covers the constants as a whole; the frozen
    record keeps them per key too (``constants``) so a later version can name the keys
    that changed (:func:`version_diff`)."""
    constants = protocol_constants()
    files = {
        p.relative_to(root).as_posix(): _sha256(p.read_bytes())
        for p in fingerprint_files(root)
    }
    return {
        "fingerprint": fingerprint_of(constants, files),
        "constants_sha256": constants_sha256(constants),
        "constants": constants,
        "files": files,
    }


def protocol_fingerprint(root: Path = SKILL_BENCH_DIR) -> str:
    """sha256 over the protocol constants and every fingerprinted file (paths and bytes)."""
    return fingerprint_manifest(root)["fingerprint"]


#: Python below ``evals/skill_bench`` that shapes what a model sees or how a call is made,
#: beyond the fingerprinted text: the environments (history rendering, tool output),
#: the actor and the rollouts, the methods (J's and the learners' input builders), the
#: gateway's rewrite and the token budget. Not part of the protocol fingerprint (every code
#: fix would re-freeze it); recorded per invocation and compared by the audit.
CODE_FINGERPRINT_DIRS: tuple[str, ...] = ("agent", "envs", "infra", "methods")
CODE_FINGERPRINT_FILES: tuple[str, ...] = ("protocol.py",)
#: The MIRIX package every MIRIX cell's server runs (its prompts, distiller, curator and
#: the renderer whose output is MIRIX's injection).
MIRIX_PACKAGE_DIR = REPO_ROOT / "mirix"
#: The launcher that starts that server (``infra/mirix_server.py``: it picks uvicorn or
#: gunicorn and their options), relative to the repository's top. The MIRIX code digest
#: covers it under :data:`MIRIX_LAUNCHER_KEY`, its path relative to the
#: package (no file inside the package has a key that starts with ``..``).
MIRIX_LAUNCHER = "scripts/start_server.py"
MIRIX_LAUNCHER_KEY = "../" + MIRIX_LAUNCHER


def _tree_entries(root: Path, paths: Sequence[Path]) -> list[tuple[str, str]]:
    """``(relative POSIX path, sha256)`` of each file, sorted (blocking)."""
    return sorted(
        (p.relative_to(root).as_posix(), _sha256(p.read_bytes())) for p in paths
    )


def _tree_digest(root: Path, paths: Sequence[Path]) -> str:
    return _entries_digest(_tree_entries(root, paths))


def _entries_digest(entries: Sequence[tuple[str, str]]) -> str:
    """The digest :func:`code_fingerprint` records over sorted ``(path, sha256)`` pairs."""
    return _sha256(_canonical([list(entry) for entry in entries]))


def _code_files(directory: Path, suffixes: tuple[str, ...] | None) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".") and d not in ("__pycache__", "tests")
        )
        for name in filenames:
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            if suffixes is None or name.endswith(suffixes):
                found.append(Path(dirpath) / name)
    return found


def code_fingerprint(
    *,
    mirix: bool = False,
    root: Path = SKILL_BENCH_DIR,
    mirix_dir: Path | None = None,
    files: bool = False,
) -> dict[str, Any]:
    """sha256 of the model-visible harness code and, with ``mirix``, of the MIRIX package
    and the launcher that starts its server (:data:`MIRIX_LAUNCHER`, keyed
    :data:`MIRIX_LAUNCHER_KEY`) (blocking). ``run.py`` records it for every invocation
    (``attempt.json`` and each ``launches.jsonl`` work line); the audit and the report
    name the version segments of an attempt whose code changed (``attempt.json``
    ``version_segments``).

    With ``files``, also the per-file hashes the digests are made of: ``harness_sha256``
    and, with ``mirix``, ``mirix_sha256`` (``{relative path: sha256}``), so a later version
    can name the files that changed (``run.py`` keeps them in each version segment's
    ``version.json``). The digests and file counts are the same either way."""
    code_files = [
        p for d in CODE_FINGERPRINT_DIRS for p in _code_files(root / d, (".py",))
    ] + [root / f for f in CODE_FINGERPRINT_FILES if (root / f).is_file()]
    harness = _tree_entries(root, code_files)
    result: dict[str, Any] = {
        "harness": _entries_digest(harness),
        "harness_files": len(code_files),
    }
    if files:
        result["harness_sha256"] = dict(harness)
    if mirix:
        package = mirix_dir or MIRIX_PACKAGE_DIR
        mirix_files = _code_files(package, None) if package.is_dir() else []
        entries = _tree_entries(package, mirix_files)
        launcher = package.parent / MIRIX_LAUNCHER
        if mirix_files and launcher.is_file():
            # The launcher that starts the package's server.
            mirix_files.append(launcher)
            entries = sorted(
                [*entries, (MIRIX_LAUNCHER_KEY, _sha256(launcher.read_bytes()))]
            )
        result["mirix"] = _entries_digest(entries) if mirix_files else None
        result["mirix_files"] = len(mirix_files)
        if files:
            result["mirix_sha256"] = dict(entries)
    return result


def verify_fingerprint(expected: str, root: Path = SKILL_BENCH_DIR) -> None:
    """Raise :class:`ProtocolDriftError` if the current fingerprint is not ``expected``."""
    actual = protocol_fingerprint(root)
    if actual != expected:
        raise ProtocolDriftError(f"protocol fingerprint {actual} != frozen {expected}")


# ---------------------------------------------------------------------------
# The frozen value (H10.6)
# ---------------------------------------------------------------------------

REFREEZE_HINT = (
    "run.py freezes the tree automatically when a new attempt, a learning resume or a "
    "test resume starts; to freeze by hand: python -m "
    "evals.skill_bench.tools.freeze_protocol --reason <why> (the old value stays in the "
    "record history)"
)


def frozen_record_path() -> Path:
    """The frozen record every reader and writer uses when not given a path: the file
    :data:`FROZEN_RECORD_ENV` names (made absolute) when it is set and not empty, else the
    repository's :data:`FROZEN_PROTOCOL_JSON`."""
    override = os.environ.get(FROZEN_RECORD_ENV, "").strip()
    return Path(override).absolute() if override else FROZEN_PROTOCOL_JSON


def load_frozen_protocol(path: Path | None = None) -> dict[str, Any]:
    """The frozen protocol record written by ``tools/freeze_protocol.py`` (default
    :func:`frozen_record_path`)."""
    path = frozen_record_path() if path is None else path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProtocolDriftError(
            f"{path} is missing: the protocol has not been frozen "
            "(python -m evals.skill_bench.tools.freeze_protocol --reason '<why>')"
        ) from None
    if not isinstance(record, dict) or not isinstance(record.get("fingerprint"), str):
        raise ProtocolDriftError(f"{path} does not hold a frozen fingerprint")
    return record


def frozen_fingerprint(path: Path | None = None) -> str:
    """The frozen protocol fingerprint every cell must carry (H10.6)."""
    return str(load_frozen_protocol(path)["fingerprint"])


def protocol_drift(
    frozen: Mapping[str, Any] | None = None,
    root: Path = SKILL_BENCH_DIR,
    *,
    path: Path | None = None,
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """How the protocol at ``root`` differs from the frozen record (empty lists: no drift).

    ``changed`` / ``added`` / ``removed`` name fingerprinted files; ``constants_changed``
    says whether a constant of this module differs. ``current`` is the tree's
    :func:`fingerprint_manifest` when the caller already computed it (``run.py``'s
    automatic freeze compares the manifest its launch runs under), else it is computed.

    ``record_problems``: with the fingerprint equal to the tree's, how the record's own
    constants hash, per-key constants or file map are not the tree's. The fingerprint covers the constants and
    the files, so such a record was edited by hand, and those values decide which keys
    and files a version change names (:func:`version_diff`); it counts as drift.
    """
    record = dict(frozen) if frozen is not None else load_frozen_protocol(path)
    current = dict(current) if current is not None else fingerprint_manifest(root)
    before = dict(record.get("files") or {})
    after = current["files"]
    changed = record.get("constants_sha256") != current["constants_sha256"]
    drift = {
        "frozen": record["fingerprint"],
        "current": current["fingerprint"],
        "drifted": record["fingerprint"] != current["fingerprint"],
        "constants_changed": changed,
        "constants_changed_keys": _changed_keys(
            recorded_constants(record), current.get("constants")
        )
        if changed
        else [],
        "changed": sorted(p for p in after if p in before and before[p] != after[p]),
        "added": sorted(p for p in after if p not in before),
        "removed": sorted(p for p in before if p not in after),
        "record_problems": [],
    }
    if not drift["drifted"]:
        drift["record_problems"] = _record_problems(record, current, drift)
        drift["drifted"] = bool(drift["record_problems"])
    return drift


def _record_problems(
    record: Mapping[str, Any], current: Mapping[str, Any], drift: Mapping[str, Any]
) -> list[str]:
    """How a frozen record whose fingerprint is the tree's does not describe the tree
    (:func:`protocol_drift`); empty when it does."""
    problems = []
    if drift["constants_changed"]:
        problems.append(
            f"its constants_sha256 {record.get('constants_sha256')} is not the tree's "
            f"{current['constants_sha256']}"
        )
    constants = record.get("constants")
    if constants is not None:
        keys = (
            _changed_keys(constants, current["constants"])
            if isinstance(constants, Mapping)
            else None
        )
        if keys is None or keys:
            problems.append(
                "its per-key constants are not the tree's"
                + (f" ({', '.join(keys)})" if keys else "")
            )
    for key in ("changed", "added", "removed"):
        if drift[key]:
            problems.append(
                f"its file map is not the tree's ({key}: {', '.join(drift[key])})"
            )
    # History entries edited by hand: --check names them, rather than leaving the
    # audit to fail their cells closed with nothing said at the freeze tool.
    versions = _versions_of(record)
    edited = sorted(
        fingerprint[:12]
        for fingerprint, version in versions.items()
        if not version["current"] and _constants_of(version, versions)[1]
    )
    if edited:
        problems.append(
            "its history entries "
            + ", ".join(edited)
            + " keep constants that do not reproduce their fingerprints"
        )
    return problems


def _constants_line(changed_keys: Sequence[str] | None) -> str:
    """ "constants of protocol.py changed", naming the keys when they are known."""
    if changed_keys:
        return "constants of protocol.py changed: " + ", ".join(changed_keys)
    return "constants of protocol.py changed"


def describe_drift(drift: Mapping[str, Any]) -> str:
    """One line per difference, for error messages and the freeze tool."""
    if not drift.get("drifted"):
        return f"protocol fingerprint {drift.get('current')} equals the frozen value"
    if drift.get("record_problems"):
        return "\n".join(
            [
                f"protocol fingerprint {drift.get('current')} equals the frozen value, "
                "but the frozen record does not describe this tree, although its "
                "fingerprint covers the constants and the files (edited by hand?):",
                *(f"  {problem}" for problem in drift["record_problems"]),
                "  restore config/protocol_frozen.json from git (a re-freeze leaves a "
                "record whose fingerprint is the tree's as it is)",
            ]
        )
    lines = [
        f"protocol fingerprint {drift.get('current')} != frozen {drift.get('frozen')}"
    ]
    if drift.get("constants_changed"):
        lines.append("  " + _constants_line(drift.get("constants_changed_keys")))
    for key in ("changed", "added", "removed"):
        lines.extend(f"  {key}: {rel}" for rel in drift.get(key) or ())
    lines.append("  " + REFREEZE_HINT)
    return "\n".join(lines)


def verify_frozen_protocol(
    root: Path = SKILL_BENCH_DIR, path: Path | None = None
) -> str:
    """Return the frozen fingerprint, or raise :class:`ProtocolDriftError` naming each drift."""
    drift = protocol_drift(root=root, path=path)
    if drift["drifted"]:
        raise ProtocolDriftError(describe_drift(drift))
    return str(drift["frozen"])


# ---------------------------------------------------------------------------
# Recorded versions (recorded and reported, not enforced across cells)
# ---------------------------------------------------------------------------


def _reproduces(constants: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Whether per-key ``constants`` are ``entry``'s: they hash to its
    ``constants_sha256`` and, with its file map, reproduce its ``fingerprint``
    (:func:`fingerprint_of`). Without a file map nothing can vouch for them."""
    files = entry.get("files")
    return (
        isinstance(files, Mapping)
        and constants_sha256(constants) == entry.get("constants_sha256")
        and fingerprint_of(constants, files) == entry.get("fingerprint")
    )


def recorded_constants(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """The protocol constants a frozen record (or one of its ``history`` entries, or a
    :func:`known_versions` value) keeps per key, or None when it keeps none or they do not
    reproduce its fingerprint with its file map (:func:`_reproduces`: checked against
    its own ``constants_sha256`` alone, an entry edited in both fields would decide which
    keys a change names). Never trusted otherwise: a key list read from them could make a
    change look auditable that is not (:func:`unauditable_changes`). Records written by
    ``tools/freeze_protocol.py`` keep them; older entries may keep only the hash."""
    constants = entry.get("constants")
    if not isinstance(constants, Mapping) or not _reproduces(constants, entry):
        return None
    return dict(constants)


def _changed_keys(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> list[str] | None:
    """The constant keys whose values differ (one side lacking a key counts), sorted;
    None when either side's per-key constants are unknown."""
    if before is None or after is None:
        return None
    return sorted(
        key
        for key in set(before) | set(after)
        if key not in before
        or key not in after
        or _canonical(before[key]) != _canonical(after[key])
    )


def _version(entry: Mapping[str, Any], *, current: bool) -> dict[str, Any]:
    files = entry.get("files")
    git = entry.get("git")
    auto = entry.get("auto_freeze")
    constants = recorded_constants(entry)
    return {
        "fingerprint": str(entry["fingerprint"]),
        "frozen_at": entry.get("frozen_at"),
        "reason": entry.get("reason"),
        "git": dict(git) if isinstance(git, Mapping) else None,
        "constants_sha256": entry.get("constants_sha256"),
        # Per key, when the entry keeps them and they reproduce its fingerprint.
        "constants": constants,
        # It keeps per-key constants that do not: edited by hand, so nothing of its
        # constants is read (version_diff: ``inconsistent``).
        "inconsistent": entry.get("constants") is not None and constants is None,
        # A history entry written before freeze_protocol kept file maps has none.
        "files": dict(files) if isinstance(files, Mapping) else None,
        "current": current,
        # Why an automatic freeze (run.py) registered it: the
        # cell, attempt and launch, and what changed; None for a freeze by hand.
        "auto_freeze": dict(auto) if isinstance(auto, Mapping) else None,
        # models.json sha256 -> model id -> canonical_slug the entry keeps
        # (:func:`model_slugs_record`): the frozen record's, of every models.json frozen
        # since it kept them; a version.json's, of its own; {} for none.
        "model_slugs": recorded_model_slugs(entry),
    }


def known_versions(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every protocol version the frozen record knows, by fingerprint: the frozen value
    (``current: True``) and each earlier value in its ``history`` (``current: False``),
    each with when and why it was frozen (``frozen_at``, ``reason``), the git state it was
    frozen from (``git``), and its ``constants_sha256``, ``constants`` (per key,
    :func:`recorded_constants`) and ``files`` (path -> sha256) when the record kept them:
    history entries written before ``tools/freeze_protocol.py`` kept them have ``None``
    there. ``inconsistent``: the entry keeps per-key constants that do not reproduce its
    fingerprint (edited by hand; :func:`version_diff` then reads nothing of its
    constants).

    A cell under a known earlier version is reported as one (audit warning
    ``protocol_version``, report "(earlier)"); only a fingerprint no freeze produced fails
    the audit. A fingerprint frozen twice (a change reverted) is
    one version described by its latest freeze, and the frozen value is always the
    ``current`` one. Raises :class:`ProtocolDriftError` when the record is missing or
    holds no fingerprint (blocking)."""
    return _versions_of(load_frozen_protocol(path))


def _versions_of(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """:func:`known_versions` of a loaded record."""
    versions: dict[str, dict[str, Any]] = {}
    for entry in record.get("history") or ():
        if isinstance(entry, Mapping) and isinstance(entry.get("fingerprint"), str):
            versions[entry["fingerprint"]] = _version(entry, current=False)
    versions[str(record["fingerprint"])] = _version(record, current=True)
    return versions


def constants_of(
    version: Mapping[str, Any], known: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    """``(constants, inconsistent)`` of a version (public: the audit resolves a version's
    per-key constants the way :func:`version_diff` does; see :func:`_constants_of`)."""
    return _constants_of(version, known)


def _constants_of(
    version: Mapping[str, Any], known: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    """``(constants, inconsistent)``: ``version``'s constants per key and whether its
    record contradicts itself.

    Its own (:func:`recorded_constants`), else those of any known version with the same
    ``constants_sha256`` (equal hashes, equal constants: a version frozen before the
    record kept them per key), else None. Either way they must reproduce
    ``version``'s own fingerprint with its file map (:func:`_reproduces`); when they do
    not, the record was edited (per-key constants, or a constants hash, that are not the
    ones the version's fingerprint was made from) and ``inconsistent`` is True: nothing
    of its constants is read. A version without per-key constants whose hash no version
    with them shares cannot be checked: ``(None, False)``."""
    own = recorded_constants(version)
    if own is not None:
        return own, False
    if version.get("inconsistent") or version.get("constants") is not None:
        return None, True
    sha = version.get("constants_sha256")
    if not sha or not isinstance(version.get("files"), Mapping):
        return None, False
    for other in known.values():
        if other.get("constants_sha256") != sha:
            continue
        found = recorded_constants(other)
        if found is not None:
            return (found, False) if _reproduces(found, version) else (None, True)
    return None, False


def version_diff(
    a: str | None,
    b: str | None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """How protocol version ``b`` differs from version ``a``: the fingerprinted files
    ``changed``, ``added`` and ``removed`` on the way from ``a`` to ``b``,
    ``constants_changed`` (None when a constants hash is not recorded, or not to be
    trusted) and ``constants_changed_keys``, the constant keys whose values differ
    (``[]`` when no constant changed; None when they changed but a side's per-key
    constants are not recorded, :func:`_constants_of`). ``inconsistent`` names the
    versions (of ``a`` and ``b``) whose record contradicts its own fingerprint (per-key
    constants and their hash edited together): their constants are not read,
    which :func:`unauditable_changes` fails closed on. None when either version, or its
    file map, is unknown (``versions``: :func:`known_versions` of the frozen record by
    default). When ``config/models.json`` differs, ``model_slugs_recorded`` says whether
    the canonical slugs of both versions' files are recorded (:func:`file_touches_benchmark`).
    The audit's ``protocol_version`` warning and the report name these files and keys."""
    known = known_versions() if versions is None else versions
    before = known.get(a) if a else None
    after = known.get(b) if b else None
    if before is None or after is None:
        return None
    old, new = before.get("files"), after.get("files")
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return None
    old_constants, old_inconsistent = _constants_of(before, known)
    new_constants, new_inconsistent = _constants_of(after, known)
    inconsistent = list(
        dict.fromkeys(
            str(fingerprint)
            for fingerprint, flagged in ((a, old_inconsistent), (b, new_inconsistent))
            if flagged
        )
    )
    constants = (before.get("constants_sha256"), after.get("constants_sha256"))
    changed = (
        None if inconsistent or None in constants else constants[0] != constants[1]
    )
    keys: list[str] | None
    if changed is None:
        keys = None
    elif not changed:
        keys = []
    else:
        keys = _changed_keys(old_constants, new_constants)
    known_benchmarks = benchmarks_of(old_constants, new_constants)
    files = {
        "changed": sorted(p for p in new if p in old and old[p] != new[p]),
        "added": sorted(p for p in new if p not in old),
        "removed": sorted(p for p in old if p not in new),
    }
    models_json: dict[str, Any] = {}
    if old.get(MODELS_JSON_REL) != new.get(MODELS_JSON_REL):
        # Whether the slugs of both files are recorded (by any known version: the
        # frozen record carries every models.json frozen so far, the auditing tree its
        # own; :func:`file_touches_benchmark`).
        slugs = {
            sha for version in known.values() for sha in recorded_model_slugs(version)
        }
        models_json["model_slugs_recorded"] = (
            old.get(MODELS_JSON_REL) in slugs and new.get(MODELS_JSON_REL) in slugs
        )
    return {
        "from": a,
        "to": b,
        "constants_changed": changed,
        "constants_changed_keys": keys,
        # The benchmarks each changed key touches (None: every benchmark).
        "constants_scope": (
            {
                key: constant_scope(key, old_constants or {}, new_constants or {})
                for key in keys
            }
            if keys
            else {}
        ),
        "inconsistent": inconsistent,
        **files,
        # The benchmark each changed, added or removed file belongs to.
        "files_scope": {
            path: asset_benchmark(path, known_benchmarks)
            for kind in ("changed", "added", "removed")
            for path in files[kind]
        },
        # Only when config/models.json differs.
        **models_json,
    }


def _inconsistent_text(fingerprints: Sequence[str]) -> str:
    return (
        "a record that does not reproduce its fingerprint ("
        + ", ".join(str(fp)[:12] for fp in fingerprints)
        + ")"
    )


def describe_version_diff(diff: Mapping[str, Any] | None) -> str:
    """:func:`version_diff` as one line, for audit findings and the report."""
    if diff is None:
        return (
            "which files differ is not recorded (a version frozen before the record kept "
            "its file map)"
        )
    parts = []
    if diff.get("inconsistent"):
        parts.append(
            _inconsistent_text(diff["inconsistent"])
            + " with its file map (edited by hand?): its constants are not read"
        )
    if diff.get("constants_changed"):
        parts.append(_constants_line(diff.get("constants_changed_keys")))
    for key in ("changed", "added", "removed"):
        if diff.get(key):
            parts.append(f"{key}: {', '.join(diff[key])}")
    if parts:
        return "; ".join(parts)
    if diff.get("constants_changed") is None:
        return "no fingerprinted file differs (a constants hash is not recorded)"
    return "no fingerprinted file or constant differs"


#: The fingerprinted files that only a method's own learners and judge read. The audit
#: never rebuilds anything from them, so a version that differs from the auditing tree's
#: only there is audited as it ran (see :func:`unauditable_changes`).
METHOD_PROMPTS_PREFIX = "methods/prompts/"
#: Every fingerprinted directory only a method's own code reads: its prompts and its
#: setup templates (``methods/templates/mirix/setup.json``, read by ``methods/mirix.py``
#: alone). The audit rebuilds nothing from either, so a change there leaves every cell
#: auditable as it ran, the other methods' included.
METHOD_FILE_PREFIXES: tuple[str, ...] = (METHOD_PROMPTS_PREFIX, "methods/templates/")
#: Protocol constants that describe the study's composition, which cells it has, rather
#: than how any one cell ran: ``methods``, the memory methods the study compares. A cell
#: runs its own method's code and prompts; nothing it sends, receives or is scored on
#: reads which other methods the list names, and none of the audit's per-call checks
#: (the gateway settings hash, the reasoning effort, the provider policy) or its first-
#: prompt rebuild reads the list. So a version that differs from the auditing tree's only
#: in this list (and in :data:`METHOD_PROMPTS_PREFIX` files) is audited as it ran: when a
#: method is added or removed, the other methods' cells are not "earlier versions the
#: audit cannot see past". A cell of a method no longer listed is set aside by the study
#: tools (``study.py``), not audited. ``benchmarks`` likewise: a cell runs its own
#: benchmark, and adding or dropping a benchmark changes no other benchmark's cells.
#: Every other constant changes how cells run and stays unauditable, except where only
#: another benchmark's entries changed (:data:`PER_BENCHMARK_CONSTANTS`,
#: :func:`constant_scope`) and where a change touches no cell at all (``models`` gaining
#: or losing a key while the other keys keep their models).
COMPOSITION_CONSTANTS: frozenset[str] = frozenset({"methods", "benchmarks"})
#: Constants with an entry per benchmark: a change of another benchmark's entry does not
#: change how a cell of this one ran (:func:`constant_scope`). Every key named
#: ``<benchmark>_...`` (``alfworld_max_steps``, ``webshop_server``) belongs to that
#: benchmark too.
PER_BENCHMARK_CONSTANTS: frozenset[str] = frozenset(
    {"split_sizes", "extra_split_sizes", "frozen_asset_sha256", "benchmarks"}
)
#: The fingerprinted directories whose first level names a benchmark (``envs/splits/
#: alfworld/``, ``envs/templates/webshop_server/``, ``envs/vendor/webshop_projection.py``).
_BENCHMARK_ASSET_DIRS: tuple[str, ...] = ("splits", "templates", "vendor")


#: A licence or notice file's name (upper-cased): ``LICENSE``, ``NOTICE``, an optional
#: ``<WORD>_`` prefix (``VENDOR_NOTICE.txt``), an optional ``-<what>`` suffix
#: (``LICENSE-APACHE-2.0.txt``, ``LICENSE-MIT.txt``) and an optional ``.txt``.
#: Never a substring test: a fingerprinted ``license_terms.md`` a prompt reads is not one.
_DOCUMENTATION_NAME = re.compile(
    r"(?:[A-Z0-9]+_)?(?:LICENSE|NOTICE)(?:-[A-Z0-9.-]+)?(?:\.TXT)?"
)


def is_documentation_file(path: str | Path) -> bool:
    """Whether a fingerprinted file only documents provenance or a licence (a vendored
    directory's ``NOTICE``, a ``LICENSE``, a ``LICENSE-*.txt``, a ``<WORD>_NOTICE.txt``,
    by name: :data:`_DOCUMENTATION_NAME`): nothing a cell sends, receives or is scored on
    reads it, so a change there is audited as it ran (pure)."""
    name = PurePosixPath(str(path)).name.upper()
    return _DOCUMENTATION_NAME.fullmatch(name) is not None


def asset_benchmark(
    path: str | Path, benchmarks: Sequence[str] = BENCHMARKS
) -> str | None:
    """The benchmark a fingerprinted file belongs to, or None for a file every benchmark
    shares (``config/models.json``, ``methods/prompts/...``, ``envs/vendor/__init__.py``):
    ``envs/{splits,templates,vendor}/<benchmark>[_<anything>]...`` (pure)."""
    parts = PurePosixPath(str(path)).parts
    if len(parts) < 3 or parts[0] != "envs" or parts[1] not in _BENCHMARK_ASSET_DIRS:
        return None
    head = PurePosixPath(parts[2]).stem if len(parts) == 3 else parts[2]
    for benchmark in sorted(benchmarks, key=len, reverse=True):
        if head == benchmark or head.startswith(benchmark + "_"):
            return benchmark
    return None


def benchmarks_of(*constants: Mapping[str, Any] | None) -> list[str]:
    """Every benchmark this module or one of ``constants`` (per-key protocol constants,
    None for a version that does not keep them) names, sorted (pure)."""
    found = set(BENCHMARKS)
    for entry in constants:
        listed = entry.get("benchmarks") if isinstance(entry, Mapping) else None
        if isinstance(listed, (list, tuple)):
            found |= {str(b) for b in listed}
    return sorted(found)


#: Recorded constants that only limit how a host runs cells (how many at once), never what
#: a cell ran under: a change of one touches no cell (:func:`constant_scope`).
OPERATING_CONSTANTS: frozenset[str] = frozenset({"schedule_concurrency_range"})


def constant_scope(
    key: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str] | None:
    """The benchmarks a change of protocol constant ``key`` touches, between the per-key
    constants ``before`` and ``after`` (pure): sorted names, or None when the
    change touches every benchmark.

    ``benchmarks``: the benchmarks added or removed (an order change touches none);
    ``split_sizes`` / ``extra_split_sizes``: the benchmarks whose entry differs;
    ``frozen_asset_sha256``: the benchmarks of the files whose hash differs
    (:func:`asset_benchmark`; a shared file touches every benchmark, a documentation file
    none); ``models``: none when only keys were added or removed and every key both sides
    have names the same model (a model added to the study, or one dropped: a cell runs
    its own model, and no other cell sent anything else), else
    every benchmark (a key re-pointed at another model); a key named ``<benchmark>_...``:
    that benchmark; an operating limit (:data:`OPERATING_CONSTANTS`): none. Any other key
    touches every benchmark (fail closed). An empty list touches no cell
    (:func:`touches_benchmark`)."""
    if key in OPERATING_CONSTANTS:
        return []
    known = benchmarks_of(before, after)
    old, new = before.get(key), after.get(key)
    if key == "models":
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return None
        same = all(
            _canonical(old[k]) == _canonical(new[k]) for k in set(old) & set(new)
        )
        return [] if same else None
    if key == "benchmarks":
        if not isinstance(old, (list, tuple)) or not isinstance(new, (list, tuple)):
            return None
        return sorted({str(b) for b in old} ^ {str(b) for b in new})
    if key in ("split_sizes", "extra_split_sizes"):
        old, new = old or {}, new or {}
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return None
        return sorted(
            str(b)
            for b in set(old) | set(new)
            if _canonical(old.get(b)) != _canonical(new.get(b))
        )
    if key == "frozen_asset_sha256":
        old, new = old or {}, new or {}
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return None
        scopes = {
            asset_benchmark(path, known)
            for path in set(old) | set(new)
            if old.get(path) != new.get(path) and not is_documentation_file(path)
        }
        return None if None in scopes else sorted(str(b) for b in scopes)
    head, sep, _ = key.partition("_")
    return [head] if sep and head in known else None


def touches_benchmark(diff: Mapping[str, Any], key: str, benchmark: str | None) -> bool:
    """Whether the change of constant ``key`` in ``diff`` (a :func:`version_diff`) touches
    the cells of ``benchmark`` (pure), or with ``benchmark`` None the cells of any
    benchmark. True when the diff does not say (a diff without ``constants_scope``: fail
    closed) and for a change of every benchmark (scope None); False for a change that
    touches no benchmark (an empty scope: a model added to ``models``, only documentation
    files in ``frozen_asset_sha256``)."""
    scope = diff.get("constants_scope")
    if not isinstance(scope, Mapping) or key not in scope:
        return True
    touched = scope[key]
    if touched is None:
        return True
    return bool(touched) if benchmark is None else benchmark in touched


def file_touches_benchmark(
    diff: Mapping[str, Any], path: str, benchmark: str | None
) -> bool:
    """Whether a changed, added or removed fingerprinted file of ``diff`` can change how a
    cell of ``benchmark`` ran, as the audit checks it (pure): documentation files never
    do; ``config/models.json`` does not when the canonical slugs of both versions' files
    are recorded (``model_slugs_recorded``): the audit reads nothing else of it, and
    judges the model a call was served by against the slugs of the cell's own version
    (``H10.3:model_served``), so a model added to the file leaves every other
    cell auditable; another benchmark's files (``files_scope``, else
    :func:`asset_benchmark`) do not when ``benchmark`` is given."""
    if is_documentation_file(path):
        return False
    if path == MODELS_JSON_REL and diff.get("model_slugs_recorded") is True:
        return False
    if benchmark is None:
        return True
    scope = diff.get("files_scope")
    owner = (
        scope[path]
        if isinstance(scope, Mapping) and path in scope
        else asset_benchmark(path)
    )
    return owner is None or owner == benchmark


def unauditable_changes(
    diff: Mapping[str, Any] | None, benchmark: str | None = None
) -> list[str]:
    """What of a :func:`version_diff` (an earlier version -> the frozen one) an audit run
    on the frozen tree cannot see past for a cell under the earlier version: empty when
    only method files (:data:`METHOD_FILE_PREFIXES`: method prompts and method setup
    templates) and :data:`COMPOSITION_CONSTANTS` differ.

    The audit checks every call against this tree's constants (the gateway settings
    hash, the reasoning effort, the provider policy), the gateway start records against
    this tree's settings, and rebuilds the first prompts from this tree's environment
    templates and splits (H10.2, H10.5). A "known earlier version" therefore passes with
    only the ``protocol_version`` warning when every constant outside the composition
    keys and every fingerprinted file outside the method prompts equal the frozen tree's
    (for example when only a method's prompt changed, or when a method was removed:
    ``methods`` changed and that method's prompts went); otherwise those checks would
    report the version change itself as violations. Named here: ``protocol.py constant
    <key>`` for each other changed key, ``constants of protocol.py`` when the constants changed but which keys is not
    recorded (a version frozen before the record kept them per key, whose hash no
    version with per-key constants shares), ``a record that does not reproduce its
    fingerprint (<fingerprint>)`` (``inconsistent``: its constants were edited, so none
    of them is read), ``a constants hash that is not recorded``, each other changed,
    added or removed file, or ``a file map that is not recorded`` (a version frozen
    before the record kept one). Such a cell is audited from a checkout of its own
    protocol (:func:`version_checkout_hint`) (pure).

    With ``benchmark`` (a cell's), a change of another benchmark's entries or files is not
    named (adding or dropping a benchmark leaves the other benchmarks' cells auditable as
    they ran; :func:`touches_benchmark`,
    :func:`file_touches_benchmark`). Documentation files (:func:`is_documentation_file`),
    a change that touches no benchmark (a model added to ``models``) and a
    ``config/models.json`` whose slugs both versions record are never named."""
    if diff is None:
        return ["a file map that is not recorded"]
    out = []
    if diff.get("inconsistent"):
        out.append(_inconsistent_text(diff["inconsistent"]))
    elif diff.get("constants_changed") is None:
        out.append("a constants hash that is not recorded")
    elif diff.get("constants_changed"):
        keys = diff.get("constants_changed_keys")
        if not keys:
            # Changed, but not known where (or a diff that claims no key): fail closed.
            out.append("constants of protocol.py")
        else:
            out += [
                f"protocol.py constant {key}"
                for key in keys
                if key not in COMPOSITION_CONSTANTS
                and touches_benchmark(diff, key, benchmark)
            ]
    out += [
        path
        for key in ("changed", "added", "removed")
        for path in diff.get(key) or ()
        if not str(path).startswith(METHOD_FILE_PREFIXES)
        and file_touches_benchmark(diff, str(path), benchmark)
    ]
    return out


def version_checkout_hint(fingerprint: str, git: Mapping[str, Any] | None) -> str:
    """Where to audit a cell of protocol version ``fingerprint`` from, for a warning
    (pure): a checkout whose tree is that version.

    Its record's ``git.commit`` is the commit the version was frozen *from*, not one that
    holds it: the freeze writes the record afterwards (often ``dirty``: with uncommitted
    changes), so that commit's record still names the value before. The commit that
    recorded the version is the oldest one whose record has it as the top-level
    ``fingerprint`` (``git log -G`` on that line; ``-S`` would also match the value inside
    ``history``); a version replaced before it was committed has no commit of its own and
    the command prints nothing.
    ``freeze_protocol --check`` in a checkout says whether its tree is the version."""
    git = git if isinstance(git, Mapping) else {}
    commit = git.get("commit")
    frozen_from = (
        f"the record says it was frozen from commit {str(commit)[:12]}"
        + (" with uncommitted changes" if git.get("dirty") else "")
        + ", before the freeze wrote the record"
        if commit
        else "the commit it was frozen from is not recorded"
    )
    return (
        f"a checkout whose tree is protocol {fingerprint}, where `python -m "
        "evals.skill_bench.tools.freeze_protocol --check` prints it as the frozen "
        "value: the commit that recorded it is the first line of "
        f"`{recorded_commit_command(fingerprint)}`, run from the repository root "
        f"({frozen_from}); a version replaced before it was committed has no commit of "
        "its own (the command prints nothing)"
    )


def recorded_commit_command(fingerprint: str) -> str:
    """The git command whose first output line is the commit that recorded protocol
    ``fingerprint`` as the frozen value (:func:`version_checkout_hint`): the record's
    top-level ``"fingerprint"`` line, two spaces in (``history`` entries are deeper)."""
    record = FROZEN_PROTOCOL_JSON.relative_to(REPO_ROOT).as_posix()
    return (
        f'git log --reverse --format=%h -G \'^  "fingerprint": "{fingerprint}"\' '
        f"-- {record}"
    )


def attempt_protocol(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    number: int | None = None,
) -> dict[str, Any]:
    """The protocol attempt ``number`` started under, as the cell's records say:
    ``fingerprint``, ``constants_sha256`` and ``source``. A learning or test resume may go
    on under another version: every version the attempt ran under
    is in :func:`attempt_versions`. ``source`` is

    * ``attempt``: its ``attempt.json`` ``protocol``, which ``run.py`` records for every
      attempt since versions are recorded per attempt;
    * ``history``: an attempt from before that which ``cell.json`` ``protocol_history``
      lists (the cell adopted another protocol for a later attempt);
    * ``cell``: else ``cell.json`` ``cell``'s ``protocol_fingerprint`` and
      ``protocol_constants_sha256``. Until a cell adopts another protocol it has run one:
      an earlier ``run.py`` refused a cell whose manifest named another fingerprint,
      so every attempt of a cell written by it ran under the manifest's value.

    Pure: ``manifest`` is ``cell.json`` as read, ``state`` the attempt's ``attempt.json``
    (``number`` defaults to its ``attempt``)."""
    state = state or {}
    if number is None and isinstance(state.get("attempt"), int):
        number = int(state["attempt"])
    recorded = state.get("protocol")
    if isinstance(recorded, Mapping) and recorded.get("fingerprint"):
        return {
            "fingerprint": str(recorded["fingerprint"]),
            "constants_sha256": recorded.get("constants_sha256"),
            "source": "attempt",
        }
    if number is not None:
        for entry in manifest.get("protocol_history") or ():
            if isinstance(entry, Mapping) and number in (entry.get("attempts") or ()):
                return {
                    "fingerprint": entry.get("fingerprint"),
                    "constants_sha256": entry.get("constants_sha256"),
                    "source": "history",
                }
    cell = manifest.get("cell")
    cell = cell if isinstance(cell, Mapping) else {}
    return {
        "fingerprint": cell.get("protocol_fingerprint"),
        "constants_sha256": cell.get("protocol_constants_sha256"),
        "source": "cell",
    }


def recorded_version(document: Mapping[str, Any]) -> dict[str, Any] | None:
    """A :func:`known_versions`-style version read from a document that describes one
    protocol version by itself: a frozen record, one of its ``history`` entries, or a
    version segment's ``attempt-NN/versions/SS/version.json`` (``run.py``). So a
    fingerprint frozen only on another host (its automatic freeze went into that host's
    record) is resolved from the cell's own records.

    Only when its per-key ``constants`` and its ``files`` reproduce its ``fingerprint``
    (:func:`recorded_constants`), else None: a document that cannot vouch for itself never
    describes a version (fails closed). ``frozen_at`` and ``reason`` fall back to a
    version.json's ``frozen`` block (pure)."""
    if not isinstance(document, Mapping) or not isinstance(
        document.get("fingerprint"), str
    ):
        return None
    if recorded_constants(document) is None:
        return None
    version = _version(document, current=False)
    frozen = document.get("frozen")
    if isinstance(frozen, Mapping):
        for key in ("frozen_at", "reason"):
            if version.get(key) is None:
                version[key] = frozen.get(key)
    return version


#: ``launches.jsonl`` ``event`` of a work line (``cell_records.LAUNCH_WORK``).
_WORK_EVENT = "work"


def _segment_protocol(
    line: Mapping[str, Any], started: Mapping[str, Any]
) -> dict[str, Any]:
    recorded = line.get("protocol")
    if isinstance(recorded, Mapping) and recorded.get("fingerprint"):
        return {
            "fingerprint": str(recorded["fingerprint"]),
            "constants_sha256": recorded.get("constants_sha256"),
        }
    return {
        "fingerprint": started.get("fingerprint"),
        "constants_sha256": started.get("constants_sha256"),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _legacy_segment(
    index: int,
    *,
    phase: Any,
    launch: Any,
    at: Any,
    protocol_value: Mapping[str, Any],
    code: Any,
    git: Any,
    host: Any,
) -> dict[str, Any]:
    return {
        "index": index,
        "legacy": True,
        "why": "legacy",
        "phase": phase,
        "first_launch": launch,
        "started_at": at,
        "launches": [launch] if launch else [],
        "learning_from": None,
        "test_from": None,
        "protocol": dict(protocol_value),
        "code": dict(code) if isinstance(code, Mapping) else code,
        "git": dict(git) if isinstance(git, Mapping) else git,
        "host": dict(host) if isinstance(host, Mapping) else host,
        "tree": None,
        "version_file": None,
        "auto_freeze": None,
        "known_version": None,
        "note": None,
        "changed": None,
    }


def attempt_versions(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    number: int | None = None,
    work_lines: Sequence[Mapping[str, Any]] = (),
    *,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every version attempt ``number`` ran under, in order: its ``attempt.json``
    ``version_segments`` (``run.py``), each ``{index, legacy, why,
    phase, first_launch, started_at, launches, learning_from | test_from, protocol, code,
    git, host, tree, version_file, auto_freeze, known_version, note, changed}``, index 0
    the version it started under (``changed`` None), each later one opened by a learning
    or test resume under another protocol, code, commit or uncommitted tree, ``changed``
    naming what differs from the one before.

    An attempt written before segments were recorded gets legacy segments
    (``legacy``: True, ``why`` ``legacy``) synthesized from its ``launches.jsonl`` work
    lines (``work_lines``, the cell's lines in file order; only this attempt's are read),
    grouped consecutively by protocol fingerprint (a line without one: the attempt's
    :func:`attempt_protocol`), ``code.harness``, ``code.mirix`` and ``git.commit``; their
    ``changed`` names ``protocol`` (and its ``protocol_diff``, :func:`version_diff` over
    ``versions``, default the frozen record's :func:`known_versions`, None when unknown),
    ``code`` (the digests that differ), ``commit``; ``code_files`` is None and ``tree``
    False (neither was recorded). Without work lines: one segment from the attempt's own
    start (``started_by``, ``started_at``, ``code``, ``git``, ``host``).

    ``manifest`` is ``cell.json`` as read, ``state`` the attempt's ``attempt.json``
    (``number`` defaults to its ``attempt``). Returns a copy (the caller may change it).
    Pure, except that a legacy protocol change without ``versions`` reads the frozen
    record (blocking)."""
    state = state or {}
    if number is None and isinstance(state.get("attempt"), int):
        number = int(state["attempt"])
    recorded = state.get("version_segments")
    if isinstance(recorded, list):
        return copy.deepcopy(recorded)
    started = attempt_protocol(manifest, state, number)
    lines = [
        line
        for line in work_lines
        if isinstance(line, Mapping)
        and line.get("event") == _WORK_EVENT
        and line.get("attempt") == number
    ]
    if not lines:
        return [
            _legacy_segment(
                0,
                phase="learning",
                launch=state.get("started_by"),
                at=state.get("started_at"),
                protocol_value=_segment_protocol({}, started),
                code=state.get("code"),
                git=state.get("git"),
                host=state.get("host"),
            )
        ]
    known: dict[str, Mapping[str, Any]] | None = (
        dict(versions) if versions is not None else None
    )

    def diff(a: Any, b: Any) -> dict[str, Any] | None:
        nonlocal known
        if known is None:
            try:
                known = dict(known_versions())
            except ProtocolDriftError:
                known = {}
        return version_diff(a, b, known) if a and b else None

    segments: list[dict[str, Any]] = []
    last_key: tuple[Any, ...] | None = None
    for line in lines:
        proto = _segment_protocol(line, started)
        code = _mapping(line.get("code"))
        git = _mapping(line.get("git"))
        key = (
            proto["fingerprint"],
            code.get("harness"),
            code.get("mirix"),
            git.get("commit"),
        )
        launch = line.get("launch")
        if segments and key == last_key:
            if launch and launch not in segments[-1]["launches"]:
                segments[-1]["launches"].append(launch)
            continue
        segment = _legacy_segment(
            len(segments),
            phase=line.get("phase"),
            launch=launch,
            at=line.get("at"),
            protocol_value=proto,
            code=line.get("code"),
            git=line.get("git"),
            host=line.get("host"),
        )
        if segments and last_key is not None:
            protocol_changed = key[0] != last_key[0]
            segment["changed"] = {
                "protocol": protocol_changed,
                "protocol_diff": diff(last_key[0], key[0])
                if protocol_changed
                else None,
                "code": [
                    name
                    for name, before, after in (
                        ("harness", last_key[1], key[1]),
                        ("mirix", last_key[2], key[2]),
                    )
                    if before != after
                ],
                "code_files": None,
                "commit": key[3] != last_key[3],
                "tree": False,
            }
        segments.append(segment)
        last_key = key
    return segments
