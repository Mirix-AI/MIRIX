"""Audit the invariants of a skill_bench run (H10, H9).

Usage (from the repository root)::

    python -m evals.skill_bench.audit --runs-dir evals/skill_bench/runs
    python -m evals.skill_bench.audit --split-mode smoke --cell alfworld-luna-awm-smoke-l5-t5

It reads a runs directory the way ``report.py`` does (``cell_records.py`` has the layout),
plus the scorer-side items (through the environments, for the label strings), the gateway
ledgers and the gateway start records, and writes ``audit.json`` and ``audit.md``
(default ``<runs-dir>/_audit/``). Exit code 0: no violation, every selected cell complete
and every check run; 1: a violation (a validity check, in a cell or across cells); 2: no
violation, but some cells are incomplete (not audited), some check could not run on this
machine (``not checked``: the cell's items or environment are missing), or there is
nothing to audit. A failing audit made under another
protocol state (frozen or auditor fingerprint) than a passing record already in the output
directory never replaces it: it goes to ``protocol-<fingerprint>/`` beside it
(``--replace-passing`` overrides).

**Removed methods** (``study.REMOVED_METHODS``).
A selected cell of a method the study removed is set aside before anything of it is
judged: it is listed under ``removed`` in audit.json (each method's decision and reason;
each cell with its status) and in audit.md, gets no finding, is never an incomplete cell
and counts toward no exit code, and takes part in no check across cells (duplicate cell
directories, code, protocol, platform, host, sandbox image, no-memory first prompts).
``report.py`` never counts such a cell as not audited. A removed method's cell whose
directory ``tools/remove_method_data.py`` deleted is listed from its removal manifest
(``_removed/`` and each ``_hosts/<host>/_removed/``, one per removal id;
``removed.deleted``); a manifest that cannot be read is a warning
(``removed_manifest_unreadable``). Set aside the same way: a cell of a method
``protocol.METHODS`` does not list and no removal record names
(``not_in_matrix``, one warning per method, ``method_not_in_matrix``), and every directory
but one of a cell id several directories hold (``duplicates``; the latest-started complete
one is audited, ``duplicate_cell_directories`` recorded).

**Severities** (record rather than block). Only the
validity checks (:data:`VALIDITY_CHECKS`, listed in audit.json ``validity_checks``: labels
reaching a method, the actor template, calls outside the gateway or its role policy, memory
writes in the test phase, the model parameters, the scorer's records) are **violations**:
they fail the cell they concern, and a check across cells fails exactly the cells it names
(``cells``; one that names no audited cell fails the audit, ``unattributed_violations``).
Every other check is **recorded** as a warning (:data:`RECORDED_CHECKS`, each with the
rule that downgrades it: the finding keeps ``downgraded: "record"``, counted per cell in
``info.recorded_counts``). A validity check
that reads this tree (its templates, splits, model metadata or constants, :data:`TREE_INPUTS`)
whose inputs differ from the cell's own protocol version, or are not recorded for it, is
**not re-verified** under that version: its findings are warnings (``downgraded:
"not_reverified"``, ``info.not_reverified``, and one ``protocol_version_unauditable``
warning per cell naming what differs and the commit to audit it from). A check this machine
cannot run (the cell's items or environment are missing, :data:`NOT_CHECKED_CHECKS`) is
**not checked**: the cell is neither passed nor failed (exit 2). A check nobody classified
fails closed (a violation).

The five H10 invariants and the H9 results check, each with the checks that establish it
(which of them are violations is :data:`VALIDITY_CHECKS`; see Severities):

* **H10.1 no labels reach a method.** What a method received is rebuilt from each private
  trajectory: the ``TaskView`` (opaque uid, ``instruction``, ``first_messages`` and
  ``agent_context``) of every item, and in the learning phase every message
  (``on_message``, ``TrajectoryView``).

  - the uid is ``"t"`` + 20 lowercase hex characters, is not the raw id, and equals
    HMAC(cell salt, raw id) recomputed from the cell's salt (H5, :func:`uid_problem`).
    Opacity is what the HMAC gives, never judged by substrings: a raw id of a few digits
    sits inside a random 20-hex uid now and then. Whether a
    raw id reached a method is the label-string search below: every raw id is a label
    of its item, found only as a whole token;
  - every message has exactly its visible shape (``visible_message`` of it is itself),
    and no key anywhere is a label key (``trajectory.FORBIDDEN_LABEL_KEYS``);
  - the ``instruction`` is the task statement: the game's "Your task is to"
    sentence (ALFWorld) or the instruction the first page states (WebShop), and the
    learning messages start with the ``first_messages`` (both recorded);
  - no scorer-side label string (``env.label_strings``: gold answers, raw ids, task types,
    source documents, page references, game files) occurs in harness-authored text
    (instruction, environment messages), compared case- and whitespace-insensitively at
    word boundaries. The item's own instruction (WebShop) and what the item shows by
    design (``env.visible_by_design``) are blanked first. Nothing is blanked on
    ALFWorld, whose instruction must equal the game's task sentence exactly. Model output
    (``assistant``) and tool output (document excerpts) are not text the harness wrote;
  - no other item's scorer-side identifier (raw id, game file, source URL ...; gold
    answers excluded, since different questions share documents, and category labels
    shared by many items, ``env.taxonomy_labels``: ALFWorld task types, WebShop product
    categories) occurs in that harness-written text either, nor in what the methods
    produced (every injection, the frozen memory): a method can only hold
    such a string if it received it;
  - a test item's own specific gold answer is not in its test-phase injection or the
    frozen memory unless a learning input held the same text (``answer_in_method_output``;
    numbers also in ``digit_form``, so ``256865`` holds ``256,865``); telling text answers
    (``text_answer_tier``: two words outside :data:`COMMON_ANSWER_WORDS`, such as
    ``scissors menthol fresh``, searched without the common words around them,
    ``text_answer_needle``) are searched too: a violation in the item's own injection; in
    the memory a warning, and a violation once the memory holds those of more than
    ``memory_text_answers_allowed`` test items; a one-word text answer is a warning in
    the injection. In the memory a specific answer is a violation only when
    ``strong_in_memory`` (5+ digits; a short number with a word that makes it a code or
    an address; or a path/key/address mark); a name marked only by its punctuation
    (``r.j. reynolds``) and a short number in a telling phrase are counted as text
    answers are, other short ones are warnings;
  - the agent context (the actor's prompt template, ``envs/agent_context.py``) is
    harness-written text: the label searches above cover it. It is one text for every
    item of the attempt (``agent_context_varies``: a violation, a warning when the
    attempt ran several harness code versions; records written before the task view
    carried it hold none, which is one value too; a trajectory whose reset failed is
    left out: no task view was handed out, and an older rollout recorded an empty text
    for it), and a text that is not the environment's as this tree renders it is a
    warning (``agent_context_not_this_tree``: a template or code change
    since); ``info.agent_context`` summarises it;
  - a discarded trajectory (a learning resume's rolled-back try) that cannot be read is
    a violation, unless the resume that discarded it restored its checkpoint exactly
    (a warning: the try left nothing in the memory);
  - accepted, not checked: an ALFWorld trajectory's length reveals ``won`` (H1.7).

* **H10.2 one actor template.** For every trajectory the first actor request is exactly
  what was hashed (``first_prompt_sha256``), sends only ``model`` and ``messages`` with
  the cell's model (recorded: the gateway's field whitelist and the model checks of H10.3
  guard what was sent), and with the injection taken out of its memory
  slot (the frozen template rendered with and without it) hashes to the no-memory
  ``first_prompt_no_memory_sha256``; every actor call of the episode carries the same
  memory, in its slot. A WebShop request is one conversation:
  ``[user, (assistant, user)*]`` with the memory in its first message only; the first
  call is that message alone, and every later call's messages are exactly the previous
  call's followed by one assistant and one user message: the reply the previous call
  got, as its recorded response holds it (``(no reply)`` for none, or an empty one), and
  ``rollout_turn.md`` opened for the next step (``conversation_not_append_only``). Every
  actor call's recorded request, rewritten as
  the gateway rewrites it with the ledger line's ``wire`` values, hashes to the ledger's
  ``request_sha256`` (the records are what was sent). Across the cells whose versions agree
  on what those prompts are made of (the benchmark's templates and splits and the seed:
  ``info.prompt_key``), each item's no-memory first prompt is byte-identical
  for every method and model: the cell whose prompt differs from its key's majority fails
  (every cell of the key when none leads); a difference only across versions is a warning
  (``first_prompt_differs_across_versions``) the report flags.

* **H10.3 every call goes through the gateway, with a role.**

  - every actor call of every trajectory is in the attempt's ledger, as role ``actor``
    with the item's uid, phase and the cell's model; every actor ledger line belongs to a
    trajectory, except those of items H8 re-ran in the test phase;
  - every call has a protocol role, and only the roles its method uses in that phase
    (:data:`ROLE_POLICY`); chat roles use the cell's model, ``embedding`` the protocol's;
    OpenRouter served every ok call as the requested model's own family (its id's or its
    ``canonical_slug``'s, :func:`requested_families`; H1.8); no request the gateway refused
    for breaking the protocol (credential, role, model, run, phase); the run's phase
    (``run_phase``, or the ledger's phase timeline) decides, and a caller tag that
    disagrees with it is a violation;
  - ``repeat_of`` by role (recorded): ``mirix``, and ``embedding`` in MIRIX cells,
    are MIRIX's application-level retries and are counted; ``actor``, and
    ``embedding`` in the other cells, only on items H8 re-ran in the test phase;
    ``learner`` and ``judge`` never (``info.repeat_of_flagged`` counts the lines flagged
    otherwise). The same request sent for a different item is not a retry;
  - an infrastructure failure on the ``mirix`` or ``embedding`` role of the learning
    phase that no retry recovered is recorded (``learning_infra_unrecovered``), naming
    the calls run.py accepted after rolling the MIRIX batch's checkpoint back
    (``attempt.json`` ``accepted_learning_calls``: "accepted after N rollbacks", also one
    ``mirix_learning_infra`` warning per accepted batch);
  - a MIRIX cell's test phase, per item (recorded): every scored test item
    was scored on a retrieval whose log row (``mirix_retrievals.jsonl`` ``evidence``,
    re-judged here with the method's own ``retrieval_problem``) lost neither its topic
    extraction nor its query vector to anything but the model, or its row is flagged
    ``retrieval_degraded`` (the method injected the degraded retrieval after its retry
    limit: a warning per item, ``info.mirix_test_retrievals.degraded_scored``); and
    every ``mirix`` / ``embedding`` infrastructure failure no retry recovered falls inside
    a retrieval the method refused to score for it (H8 re-ran that item) or a flagged one;
  - MIRIX, per learning run (its ``started_at``..``finished_at`` window): the usage it
    reports does not exceed the ledger's calls in its window, every run reports it and has
    a window (violations); calls outside every run beyond the learning retrievals, a
    failed run, deferred session or stopped curator batch, and a learning session left
    unsettled at the freeze are recorded;
  - every attempt of the cell, not only the audited one: its run id is
    ``<cell>-aNN-<instance>`` (recorded); an invalidated attempt that had scored test
    items is recorded (``tested_attempt_replaced``), one a tested cell's ``--new-attempt
    --note`` superseded is listed (``info.superseded_attempts``) with a warning
    (``attempt_superseded``); an operator re-roll and a replaced violation are warnings; a
    test item has one scored row, and only a row that failed on infrastructure or the
    harness may precede it (violations; its listing in ``test_reruns`` is recorded);
  - the failures the harness recorded and went on from (``failures.jsonl`` pairs outside
    :data:`LEGACY_RECOVERIES`: a provider-unserved item scored 0, a MIRIX batch accepted
    after its rollbacks) are warnings, ``failure_counted_and_continued``
    (``info.counted_failures``); what a method tolerated inside itself is
    ``info.tolerated`` (``cell_records.tolerated_failures``);
  - every run of the cell id in the gateway runs directory is one of the cell directory's
    attempts (H1.3, recorded): another instance's run that ran test items is the cell
    run again (``cell_rerun_after_test``; ``cell_rerun_acknowledged`` when an operator
    note in ``cell.json`` recorded before this directory's test phase acknowledges it; a
    note recorded later, or one that names no earlier run here, is a warning that says
    so; ``smoke_cell_rerun`` for a smoke cell, which is outside the H1.3 matrix); one
    that ran none is a warning; a tested run of the directory's own instance that no
    attempt records is recorded (``unrecorded_attempt``);
  - the gateways that served the attempt (matched by each line's ``gateway_start`` id)
    forwarded to the expected upstream;
  - a learning phase resumed from its checkpoints (:meth:`CellAuditor._check_resumes`):
    the calls between a ``checkpoint`` and a ``resume`` marker are exactly those
    ``discarded.jsonl`` lists
    (``discarded_segment_mismatch``, ``discarded_segment_unrecorded``); their actor calls
    need no live trajectory, a later call that repeats one of them is allowed for every
    role (``info.repeat_of_discarded``), and they are left out of the
    learning-infrastructure check and of MIRIX's reconciliation (each segment's own
    retrievals and runs explain its ``mirix`` calls); every resume restored its
    checkpoint exactly: files, MIRIX's database digest, and the rebuilt memory's snapshot
    against the checkpoint marker's (``learning_resume_mismatch``); a try killed before
    its trajectory was written is a warning (``discarded_inputs_unrecorded``).
    ``info.learning_resumes`` lists the segments. Discarded trajectories are scanned for
    labels (H10.1) and prompts (H10.2) like live ones, but are no source of what the
    memory holds.

* **H10.4 no memory writes in the test phase.** The frozen snapshot matches the sha256
  recorded when it was frozen; every snapshot compared with it (on resume, after the test
  phase) is equal and its stored copy matches its record; every invocation that scored
  test items ended with an after-test snapshot (an interrupted one is covered by the next
  resume check: a warning); no ``learner`` or ``judge`` call in the run's test phase,
  after the freeze, or for an item that is not a learning item.

* **H10.6 the protocol versions and the model parameters.** Versions are recorded and
  reported per cell, not enforced across cells (a baseline fixed after its run re-runs
  only its own cells). The audited attempt's protocol
  (``protocol.attempt_protocol``: the version it started under) and every version it ran
  under since (``protocol.attempt_versions``: its ``version_segments``; for an attempt
  written before them, its work lines grouped by protocol, code and commit) are the
  frozen value (``config/protocol_frozen.json``); a known earlier one of its history
  (``protocol.known_versions``: a warning, ``protocol_version``, naming that version,
  when and why it was frozen, the freeze that replaced it and what differs,
  ``protocol.version_diff``); a version only the cell's own records describe (a
  segment's ``version.json`` or ``attempt.json`` ``protocol`` whose constants and file
  map reproduce its fingerprint, ``protocol.recorded_version``: ``protocol_version``,
  "registered only in its own attempt record"); or one nothing describes (recorded,
  ``fingerprint``). The auditing tree is one more version (``auditor_fingerprint``;
  ``tree_current`` says whether it is the frozen value): the checks that read it are
  judged per cell against the cell's own versions (see Severities), never as one
  run-level finding. ``info.protocol`` holds the attempt's (``fingerprint``,
  ``current``, ``known``, ``source``), ``info.version_segments`` each later segment
  (from which item, when, under which protocol and code, what changed); an attempt under
  more than one protocol or model-visible code is one warning, ``mixed_version_attempt``,
  which the report marks. A ``work`` line under a protocol, or model-visible code
  (``code.harness``, ``code.mirix`` for MIRIX), that neither ``attempt.json`` nor a
  segment records is recorded (``protocol_changed_within_attempt``,
  ``model_visible_code_changed``); records written before they carried one are not
  checked, and an invocation that ran nothing does not count. Every call is held
  to the start record of the gateway that served it (its ``gateway_start``): the
  reasoning effort and seed it went out with (``reasoning_effort``, ``seed``) and its
  provider routing are validity checks, a line served without the protocol settings or
  with another settings hash than its start record's is recorded. Each gateway's model
  settings must be those the attempt's version records (``gateway_start_settings``), and
  a call no start record covers is a violation (``gateway_start_missing``); a gateway of
  another known version (``gateway_protocol_version``), of one no freeze produced
  (recorded, ``gateway_fingerprint``) and any other setting that differs from the
  attempt's version or this tree (``gateway_settings_differ``: deadline, retries,
  backoff, tokenizer margin, encoding) are warnings. The cells of one benchmark that ran
  other code or another protocol are a warning (``code_differs_across_cells``,
  ``protocol_differs_across_cells``): the report flags each comparison across versions.
  Warnings: the attempt's invocations ran on more than one commit, or a dirty tree (each
  segment's own tree record when it has one).
  The provider policy: every chat
  call of a model with a routing (``protocol.PROVIDER_POLICY``) went out with it in its
  ``wire`` (``provider_policy_missing``), no other call with one
  (``provider_policy_unexpected``), each against the policy its
  gateway started with, and no ok call or upstream attempt came from an ignored provider
  (recorded, ``ignored_provider_served``; an attempt the gateway refused for it, never
  returned, is only counted); the policy, where it comes from and the counts are in
  ``info.provider_policy``. Hosts (recorded): an attempt
  ran on one host (its ``host``, its invocations' ``work`` lines, its resumes;
  ``host_changed_within_attempt``; a host is its machine id, not its hostname, which a
  Mac changes with its network); the cells of one benchmark ran on one
  platform (``platform_differs_across_cells``) and one sandbox image (WebShop's server
  image, ``sandbox_image_differs_across_cells``); different machines are a warning.

* **H9 the reported results are the scorer's own.** The cell's ``results.jsonl``, which
  ``report.py`` publishes, is row for row the attempt's last scored row of each test item
  in test order; each row's score is its trajectory's result and the environment's
  ``env.score`` gives it again (``rescore``); a row flagged
  ``provider_unserved`` (the provider refused the actor's request: HTTP 400/413 or a
  404 "no endpoints", a test item only on its second try) is scored 0 as
  unanswered, never by the environment, and listed (``provider_unserved_scored``, a
  warning; ``info.provider_unserved``); the ledger must back the flag: the
  item's last actor call refused upstream with a class its phase scores unanswered,
  after an unscored no-endpoint try for a test item, else ``score_record``. The test items are the protocol's (the whole test
  split, or the smoke head of the seed-42 val order); the learning stream holds only
  train-split items, each once (``learning_items``), in the seed-42 order (recorded,
  ``learning_items_order``); the live ``learning.jsonl`` plays only the stream's items
  (``learning_rows``), no position twice and, once the memory froze, positions 0..n-1 of
  the stream (recorded: ``learning_position_repeated``, ``learning_rows_order``). The
  digest of the published records (``records_sha256``) lets the report tell whether this
  audit saw them.

Records are read where they are, not where they were written: a ledger relative to the
runs directory first (``cell_records.resolve_ledger_path``), then ``--gateway-runs-dir``
and ``_hosts/*/`` of a merged view (``tools/sync_runs.py``), and a recorded absolute path
only when it names the run's own directory; blobs likewise; the gateway start records
are the union of every ``_gateway/starts.jsonl`` that can hold them. A runs directory
copied from another host is audited as it was on that host. Over a merged view, what its
``_merged.json`` says is shown (``merged_view``): when the view was built
and each host synced, and, as H9 warnings that change no verdict, why it is not final
(``merged_view_not_final``: never built, a ``--mirror-only`` sync or one stopped by
conflicts or an rsync failure since it was built, a host's last rsync that did not exit
0; ``merged_view_conflict``: each conflict it lists; ``merged_view_unreadable``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from evals.skill_bench import cell_records, protocol, study
from evals.skill_bench.agent.trajectory import (
    BLOB_KEY,
    FORBIDDEN_LABEL_KEYS,
    canonical_json,
    internalize_blobs,
    sha256_hex,
    visible_message,
)
from evals.skill_bench.cell_records import (
    ATTEMPT_COMPLETE,
    HOST_IDENTITY_FIELDS,
    HOSTS_DIR,
    LAUNCH_START,
    LAUNCH_WORK,
    LEDGER_EVENT_CHECKPOINT,
    LEDGER_EVENT_RESUME,
    NON_CELL_DIRS,
    NOTE_RERUN_ACKNOWLEDGED,
    STATUS_COMPLETE,
    CellPaths,
    GatewayRun,
    LedgerLine,
    call_placements,
    canonical_sha256,
    cell_gateway_runs,
    checkpoint_ordinal,
    host_key,
    host_text,
    parse_checkpoint_id,
    read_discarded,
    read_json,
    read_jsonl,
    read_jsonl_counted,
    read_ledger_lines,
    records_digest,
    resolve_blob_dir,
    resolve_ledger_path,
    same_host,
    segment_calls,
    unknown_cost_attempts,
)
from evals.skill_bench.envs.agent_context import env_agent_context
from evals.skill_bench.envs.base import Env, Task, opaque_task_uid
from evals.skill_bench.infra.failures import (
    PROVIDER_UNSERVED_TRIES,
    REFUSAL_CLASSES,
    REFUSAL_NO_ENDPOINT,
    ledger_line_failure,
    provider_unserved_row,
    refused_no_endpoint_try,
    unrecovered_infra_calls,
    upstream_refusal,
)
from evals.skill_bench.infra.gateway import (
    DEFAULT_UPSTREAM,
    IGNORED_PROVIDER_OUTCOME,
    PROTOCOL_SETTINGS,
    rewrite_chat_body,
)
from evals.skill_bench.tools.sync_runs import merged_view_status

SCHEMA_VERSION = 2
AUDIT_JSON = "audit.json"
AUDIT_MD = "audit.md"
VIOLATION = "violation"
WARNING = "warning"
#: A third severity: a check the auditing machine could not run because it lacks the
#: cell's items or environment (:data:`NOT_CHECKED_CHECKS`). The cell is neither passed
#: nor failed, and the audit exits 2 (the audit fails closed).
NOT_CHECKED = "not_checked"
#: ``Finding.downgraded``: a bookkeeping check that is recorded, never blocked on
#: (:data:`RECORDED_CHECKS`) ...
RECORD = "record"
#: ... or a validity check that reads this tree (:data:`TREE_INPUTS`) whose inputs differ
#: between the cell's own protocol version and the auditing tree: its findings are not
#: re-verified under the cell's version, so they are warnings the report lists.
NOT_REVERIFIED = "not_reverified"
#: A trajectory's ``failure_stage`` when its episode never started (``agent/rollout.py``
#: ``STAGE_RESET``: ``env.episode()`` or ``reset`` raised): no task view was handed out.
RESET_STAGE = "reset"
INVARIANTS: Mapping[str, str] = {
    "H10.1": "no labels reach a method",
    "H10.2": "one actor template; first prompts differ only in the memory slot",
    "H10.3": "every call goes through the gateway, with a role",
    "H10.4": "no memory writes in the test phase",
    "H10.6": "the protocol is a frozen one, one per attempt",
    "H9": "the reported results are the scorer's own records",
}

# ---------------------------------------------------------------------------
# What blocks: record the conditions and the course of a run rather than stop the
# whole run on a bookkeeping rule
# ---------------------------------------------------------------------------

#: The only checks that can be violations: those that protect the validity of the numbers
#: (methods never see labels or test items; the test phase never learns; every call goes
#: through the gateway with the recorded parameters; each item is scored once; the
#: published numbers are the raw records; the audit's own inputs fail closed).
#: ``invariant:check``. Every other check is recorded as a warning
#: (:data:`RECORDED_CHECKS`), and a check nobody classified stays a violation (fail
#: closed; ``tests/test_audit.py`` asserts that every emitted check is classified and that
#: each of these still fails the audit).
VALIDITY_CHECKS: frozenset[str] = frozenset(
    {
        # H9: the published numbers are the scorer's records.
        "H9:test_items",
        "H9:learning_items",
        "H9:learning_rows",
        "H9:score_record",
        "H9:rescore",
        "H9:cell_results",
        # H10.1: no label reaches a method.
        "H10.1:uid_format",
        "H10.1:uid_salt",
        "H10.1:message_shape",
        "H10.1:label_key",
        "H10.1:unknown_item",
        "H10.1:label_string",
        "H10.1:agent_context_varies",
        "H10.1:cross_item_label",
        "H10.1:method_output_label",
        "H10.1:structural_leak",
        "H10.1:answer_in_method_output",
        "H10.1:salt_missing",
        "H10.1:trajectory_missing",
        # H10.2: one actor template; the records are what was sent.
        "H10.2:memory_changed",
        "H10.2:request_unreadable",
        "H10.2:first_prompt_hash",
        "H10.2:memory_slot",
        "H10.2:outside_memory_slot",
        "H10.2:first_prompt_varies",
        "H10.2:request_not_as_sent",
        "H10.2:first_prompt_differs_across_cells",
        "H10.2:conversation_not_append_only",
        # H10.3: every call through the gateway, with its role; each item scored once.
        "H10.3:trajectory_missing",
        "H10.3:scored_twice",
        "H10.3:rerun_without_failure",
        "H10.3:upstream",
        "H10.3:ledger_missing",
        "H10.3:actor_call_unlinked",
        "H10.3:actor_call_not_in_ledger",
        "H10.3:actor_call_mismatch",
        "H10.3:role",
        "H10.3:phase",
        "H10.3:role_not_allowed",
        "H10.3:model",
        "H10.3:embedding_model",
        "H10.3:unaccounted_actor_call",
        "H10.3:model_served",
        "H10.3:mirix_run_unwindowed",
        "H10.3:mirix_usage_exceeds_ledger",
        "H10.3:mirix_usage_missing",
        # H10.4: the test phase never learns.
        "H10.4:frozen_missing",
        "H10.4:frozen_record",
        "H10.4:after_test_missing",
        "H10.4:snapshot_changed",
        "H10.4:snapshot_record",
        "H10.4:unverified_launch",
        "H10.4:learning_call_in_test",
        "H10.4:learning_call_after_freeze",
        "H10.4:learning_call_for_test_item",
        # H10.6: the model parameters the attempt records.
        "H10.6:reasoning_effort",
        "H10.6:seed",
        "H10.6:provider_policy_missing",
        "H10.6:provider_policy_unexpected",
        "H10.6:gateway_start_settings",
        "H10.6:gateway_start_missing",
    }
)

_ITEM_4 = "record-rather-than-block"
_RERUN_ONCE = (
    "H1.3-cell-runs-once|test-no-reroll|H10.3-tested-attempt-replaced|H10.3-other-runs"
)
_RESUME = "audit-resume"
_MIRIX_TEST = "mirix-degraded-retrieval|H10.3-mirix-test-items"
_MIRIX_LEARNING = (
    "mirix-learning-failure|learn-lost-call-rollback|H10.3-learning-infra|"
    "H10.3-mirix-pool-embedding"
)
_UNKNOWN_VERSION = (
    "V-audit-unknown-fingerprint|V-gateway-version|unknown-fingerprint-never-compared|"
    "V-report-unknown-never-compared"
)
_CODE = (
    "resume-guards|V-no-learning-resume-across-code|V-test-resume-commit|"
    "test-resume-commit|V-code-within-attempt"
)
_HOSTS = (
    "several-hosts|hosts-split|same-platform-per-benchmark|H10.6-host-within-attempt|"
    "H10.6-platform-across-cells"
)
#: The bookkeeping checks recorded as warnings, never violations, each with a short name
#: of the rule that downgrades it. The finding
#: keeps ``downgraded: "record"`` so a reader can tell it from a warning that always was
#: one, and ``info.recorded_counts`` counts every one of them per cell, uncapped.
RECORDED_CHECKS: Mapping[str, str] = MappingProxyType(
    {
        "H9:learning_items_order": "learning-items-order",
        "H9:learning_rows_order": _RESUME,
        "H9:learning_position_repeated": _RESUME,
        "H10.1:instruction": _ITEM_4,
        "H10.1:first_messages": _ITEM_4,
        "H10.2:request_fields": _ITEM_4,
        "H10.2:request_model": _ITEM_4,
        "H10.3:run_id": _ITEM_4,
        "H10.3:tested_attempt_replaced": _RERUN_ONCE,
        "H10.3:rerun_unlisted": _ITEM_4,
        "H10.3:unrecorded_attempt": _RERUN_ONCE,
        "H10.3:rerun_acknowledged_after_test": _RERUN_ONCE,
        "H10.3:cell_rerun_after_test": _RERUN_ONCE,
        "H10.3:discarded_segment_mismatch": _RESUME,
        "H10.3:discarded_segment_unrecorded": _RESUME,
        "H10.3:learning_resume_mismatch": _RESUME,
        "H10.3:unknown_method": "H10.3-role-policy",
        "H10.3:phase_mismatch": "H10.3-phase",
        "H10.3:protocol_rejection": "H10.3-protocol-rejection",
        "H10.3:repeat_of": (
            "baselines-never-retry|H10.3-repeat-of|H10.3-rb-embedding-repeat|gw-repeat-flag"
        ),
        "H10.3:learning_infra_unrecovered": _MIRIX_LEARNING,
        "H10.3:mirix_learning_retrieval_degraded": _MIRIX_TEST,
        "H10.3:mirix_retrieval_unrecorded": _MIRIX_TEST,
        "H10.3:mirix_retrieval_unchecked": _MIRIX_TEST,
        "H10.3:mirix_degraded_retrieval_scored": _MIRIX_TEST,
        "H10.3:mirix_test_infra_unrecovered": _MIRIX_TEST,
        "H10.3:mirix_learning_infra": "H10.3-mirix-reconcile",
        "H10.3:mirix_session_unsettled": "H10.3-mirix-reconcile",
        "H10.3:mirix_unexplained_calls": "H10.3-mirix-reconcile",
        "H10.3:duplicate_cell_directories": "duplicate-cells-error|H10.3-duplicate-dirs",
        "H10.6:not_frozen": _UNKNOWN_VERSION,
        "H10.6:fingerprint": _UNKNOWN_VERSION,
        "H10.6:gateway_fingerprint": _UNKNOWN_VERSION,
        "H10.6:nonconformant_call": "H10.6-gateway-settings",
        "H10.6:gateway_settings": "H10.6-gateway-settings",
        "H10.6:host_changed_within_attempt": _HOSTS,
        "H10.6:platform_differs_across_cells": _HOSTS,
        "H10.6:sandbox_image_differs_across_cells": _HOSTS,
        "H10.6:ignored_provider_served": "gw-provider-exclusion|H10.6-provider-policy",
        "H10.6:code_unrecorded": _CODE,
        "H10.6:model_visible_code_changed": _CODE,
        "H10.6:protocol_changed_within_attempt": (
            "protocol-per-attempt|resume-no-protocol-cross|V-no-resume-across-protocol|"
            "V-protocol-within-attempt"
        ),
        # Not emitted (there is no run-level tree check); classified so that an
        # audit.json of schema 1 reads the same way.
        "H10.6:auditor_tree": "V-audit-unauditable|V-constants-one-hash|V-auditor-tree",
    }
)
#: Checks the auditing machine could not run: its items or environment are missing.
NOT_CHECKED_CHECKS: frozenset[str] = frozenset(
    {"H10.1:items_unavailable", "H10.1:env_unavailable", "H9:items_unavailable"}
)

#: The validity checks that read this tree, not only the cell's records, each with what
#: of it they read: ``(protocol_constants keys, fingerprinted-file prefixes)``, a prefix's
#: ``{benchmark}`` being the cell's. When the cell's protocol version and the auditing
#: tree differ in those inputs (``protocol.version_diff``), or the version's inputs are
#: not recorded, the check's findings are ``not re-verified`` warnings
#: (checks under an earlier version). Every other check
#: reads the cell's own records only (the gateway settings and provider policy are
#: judged against the attempt's own gateway start records and version).
TREE_INPUTS: Mapping[str, tuple[frozenset[str], tuple[str, ...]]] = MappingProxyType(
    {
        "H9:test_items": (
            frozenset({"seed", "split_sizes", "benchmarks"}),
            ("envs/splits/{benchmark}/",),
        ),
        "H9:learning_items": (
            frozenset({"seed", "split_sizes", "benchmarks"}),
            ("envs/splits/{benchmark}/",),
        ),
        "H10.1:uid_format": (frozenset({"opaque_uid"}), ()),
        "H10.1:unknown_item": (
            frozenset({"split_sizes"}),
            ("envs/splits/{benchmark}/",),
        ),
        "H10.1:structural_leak": (frozenset(), ("envs/templates/{benchmark}/",)),
        "H10.2:memory_slot": (frozenset(), ("envs/templates/{benchmark}/",)),
        "H10.2:outside_memory_slot": (frozenset(), ("envs/templates/{benchmark}/",)),
        # WebShop only: its templates say the prompt is one conversation.
        "H10.2:conversation_not_append_only": (
            frozenset(),
            ("envs/templates/{benchmark}/",),
        ),
        "H10.3:role": (frozenset({"roles"}), ()),
        "H10.3:embedding_model": (frozenset({"embedding_model"}), ()),
    }
)
#: The :data:`TREE_INPUTS` checks that compare with one protocol constant: when every
#: version the attempt ran under records its constants per key (``protocol.constants_of``)
#: they are judged against those versions' own values, so a change of the constant in
#: this tree does not downgrade them; else against this tree's, not
#: re-verified where the inputs differ.
VERSION_CONSTANT_CHECKS: Mapping[str, str] = MappingProxyType(
    {"H10.3:role": "roles", "H10.3:embedding_model": "embedding_model"}
)
#: What an attempt whose versions and gateway starts record no protocol settings
#: cannot vouch for: the checks of a call no gateway start record covers (and of a start
#: that differs from this tree) are not re-verified for those findings only.
SETTINGS_NOT_RECORDED = "the attempt's protocol settings are not recorded"
UNRECORDED_SETTINGS_CHECKS: tuple[str, ...] = (
    "H10.6:reasoning_effort",
    "H10.6:seed",
    "H10.6:provider_policy_missing",
    "H10.6:provider_policy_unexpected",
)
#: H10.3:model_served reads no file hash: it judges the served model families
#: against what the cell's model was requested as under each of the attempt's own
#: versions (``CellAuditor._requested_families``, the canonical slug of that version's
#: ``config/models.json``, found by the file's hash); only a family some recorded
#: models.json gives the model, under a version whose slugs are not recorded, is not
#: re-verified.
MODEL_SERVED_CHECK = "H10.3:model_served"
#: Validity checks that compare the attempt with itself, judged between the attempt's own
#: versions (``version_segments``): the no-memory first prompt of an item may differ
#: across two versions whose templates, splits or seed differ.
WITHIN_ATTEMPT_INPUTS: Mapping[str, tuple[frozenset[str], tuple[str, ...]]] = (
    MappingProxyType(
        {
            "H10.2:first_prompt_varies": (
                frozenset({"seed"}),
                ("envs/templates/{benchmark}/", "envs/splits/{benchmark}/"),
            ),
            # The agent context is rendered from the benchmark's templates.
            "H10.1:agent_context_varies": (
                frozenset(),
                ("envs/templates/{benchmark}/",),
            ),
        }
    )
)
#: What a benchmark's no-memory first prompts are made of: cells whose versions agree on
#: these share a prompt key (``info.prompt_key``), and only they must share every item's
#: no-memory first prompt (H10.2-first-prompt-across-cells).
PROMPT_INPUTS: tuple[frozenset[str], tuple[str, ...]] = (
    frozenset({"seed"}),
    ("envs/templates/{benchmark}/", "envs/splits/{benchmark}/"),
)
#: Gateway settings (start records) -> the protocol constant of the same JSON shape.
SETTINGS_CONSTANTS: Mapping[str, str] = MappingProxyType(
    {
        "deadline_s": "call_deadline_s",
        "max_retries": "gateway_max_retries",
        "backoff_base_s": "gateway_backoff_base_s",
        "backoff_cap_s": "gateway_backoff_cap_s",
        "reasoning_effort": "reasoning_effort",
        "seed": "seed",
        "tokenizer_margin": "tokenizer_margin",
        "provider_policy": "provider_policy",
    }
)
#: The gateway settings that shape what the model is asked (H10.6-gateway-settings): a
#: gateway that ran other values than the attempt's version records is a violation; the
#: others (deadline, retries, backoff, tokenizer margin) are recorded only.
MODEL_SETTINGS: tuple[str, ...] = ("reasoning_effort", "seed", "provider_policy")
#: ``failures.jsonl`` ``(kind, recovery)`` pairs of a stop or a re-run, which records
#: written under the earlier failure policy hold. Every other pair is a failure recorded
#: instead of stopped (``failure_counted_and_continued``).
LEGACY_RECOVERIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("model", "score_and_continue"),
        ("model", "count_and_continue"),
        ("infra", "rerun_item"),
        ("harness", "rerun_item"),
        ("infra", cell_records.RECOVERY_RESUME_FROM_CHECKPOINT),
        ("infra", "restart_cell"),
        ("harness", "restart_cell"),
    }
)

#: The gateway roles each method's calls may use, per ledger phase (H6, H7, H1.6). The
#: framework's ``actor`` plays every item; the test phase only retrieves (ReasoningBank's
#: query embedding, MIRIX's retrieval), and nothing learns in it.
ROLE_POLICY: Mapping[str, Mapping[str, frozenset[str]]] = {
    "none": {
        "setup": frozenset(),
        "learning": frozenset(),
        "test": frozenset({"actor"}),
    },
    "mirix": {
        "setup": frozenset({"mirix", "embedding"}),
        "learning": frozenset({"actor", "mirix", "embedding"}),
        "test": frozenset({"actor", "mirix", "embedding"}),
    },
    "reasoningbank": {
        "setup": frozenset(),
        "learning": frozenset({"actor", "judge", "learner", "embedding"}),
        "test": frozenset({"actor", "embedding"}),
    },
    "awm": {
        "setup": frozenset(),
        "learning": frozenset({"actor", "judge", "learner"}),
        "test": frozenset({"actor"}),
    },
}
CHAT_ROLES = frozenset({"actor", "learner", "judge", "mirix"})
#: Learning-side roles: a call on them in the test phase would be a write (H10.4).
LEARNING_ROLES = frozenset({"learner", "judge"})
#: Gateway refusals that mean a caller broke the protocol (credential, role, model, run).
PROTOCOL_REJECTIONS = frozenset(
    {
        "unauthorized",
        "invalid_role",
        "role_endpoint_mismatch",
        "unsupported_endpoint",
        "model_mismatch_for_run",
        "model_not_allowed",
        "run_abandoned",
        "invalid_run_id",
        "invalid_item",
        "invalid_phase",
        "phase_mismatch",
    }
)
#: The only fields the actor sends (H5); the gateway owns every other parameter (H3).
ACTOR_REQUEST_FIELDS = frozenset({"model", "messages"})
UID_RE = re.compile(
    rf"^{re.escape(protocol.OPAQUE_UID_PREFIX)}[0-9a-f]{{{protocol.OPAQUE_UID_HEX_CHARS}}}$"
)
#: At most this many findings of one kind per cell are listed (the count is kept).
MAX_FINDINGS_PER_CHECK = 25
#: ``snapshot_checks`` kind of the memory a learning resume rebuilt (run.py): compared
#: with the checkpoint it restored, never with the frozen memory.
LEARNING_RESUME_CHECK = "learning_resume"

EnvFactory = Callable[[str], Env]


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    invariant: str
    check: str
    detail: str
    severity: str = VIOLATION
    cell_id: str | None = None
    item: str | None = None
    #: Why a check that was a violation is a warning here: :data:`RECORD` (a bookkeeping
    #: check, :data:`RECORDED_CHECKS`) or :data:`NOT_REVERIFIED` (a check reading inputs
    #: of this tree that differ from the cell's version, :data:`TREE_INPUTS`); None for
    #: a finding of its own severity.
    downgraded: str | None = None
    #: A finding across cells: the cells it concerns (a run-level violation fails only
    #: these, each of which holds a copy with its ``cell_id``).
    cells: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cells"] = list(self.cells)
        return data


class Findings:
    """Findings of one cell, with how many things each invariant verified.

    :meth:`add` is the one gate that decides severity: a check in
    :data:`RECORDED_CHECKS` is recorded as a warning, a check whose inputs of this tree
    changed since the cell's version (``not_reverified``) is a warning too, and every
    other check stays what its call site says, so an unclassified one fails closed."""

    def __init__(self, cell_id: str | None) -> None:
        self.cell_id = cell_id
        self.items: list[Finding] = []
        self.checked: Counter[str] = Counter()
        self.suppressed: Counter[tuple[str, str]] = Counter()
        self._per_check: Counter[tuple[str, str, str]] = Counter()
        #: ``invariant:check`` -> the inputs of this tree that differ from the cell's
        #: version (:meth:`CellAuditor._not_reverified`): its violations are warnings.
        self.not_reverified: dict[str, list[str]] = {}
        #: ``invariant:check`` -> inputs not recorded for some of its findings only: a
        #: finding is downgraded when its call site says so (``add(unverified=True)``,
        #: only the calls no gateway start record covers).
        self.partly_not_reverified: dict[str, list[str]] = {}
        #: ``(invariant:check, RECORD | NOT_REVERIFIED)`` -> findings downgraded, counted
        #: before the per-check cap (``info.recorded_counts``, ``info.not_reverified``).
        self.downgraded: Counter[tuple[str, str]] = Counter()
        #: Run-level violations that name their cells, uncapped (``_attribute_across``).
        self.across: list[Finding] = []

    def add(
        self,
        invariant: str,
        check: str,
        detail: str,
        *,
        item: str | None = None,
        severity: str = VIOLATION,
        cells: Iterable[str] = (),
        unverified: bool = False,
    ) -> None:
        downgraded: str | None = None
        if severity == VIOLATION:
            key = f"{invariant}:{check}"
            if key in RECORDED_CHECKS:
                downgraded = RECORD
            elif key in self.not_reverified or (
                unverified and key in self.partly_not_reverified
            ):
                downgraded = NOT_REVERIFIED
            if downgraded is not None:
                severity = WARNING
                self.downgraded[(key, downgraded)] += 1
        finding = Finding(
            invariant,
            check,
            detail,
            severity,
            self.cell_id,
            item,
            downgraded,
            tuple(cells),
        )
        if severity == VIOLATION and finding.cells:
            self.across.append(finding)
        per_check = (invariant, check, severity)
        self._per_check[per_check] += 1
        if self._per_check[per_check] > MAX_FINDINGS_PER_CHECK:
            self.suppressed[(invariant, check)] += 1
            return
        self.items.append(finding)

    def warn(self, invariant: str, check: str, detail: str, **kw: Any) -> None:
        self.add(invariant, check, detail, severity=WARNING, **kw)

    def not_checked(self, invariant: str, check: str, detail: str, **kw: Any) -> None:
        """A check the auditing machine could not run (:data:`NOT_CHECKED_CHECKS`)."""
        self.add(invariant, check, detail, severity=NOT_CHECKED, **kw)

    def ok(self, invariant: str, n: int = 1) -> None:
        self.checked[invariant] += n

    def count(self, severity: str = VIOLATION) -> int:
        return sum(n for (_, _, s), n in self._per_check.items() if s == severity)

    def violations_by_invariant(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for (invariant, _, severity), n in self._per_check.items():
            if severity == VIOLATION:
                counts[invariant] += n
        return dict(counts)


@dataclass
class CellAudit:
    cell_id: str
    root: str
    benchmark: str | None
    model: str | None
    method: str | None
    split_mode: str | None
    status: str | None
    attempt: int | None
    audited: bool
    reason: str | None = None
    findings: list[Finding] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)
    violations: int = 0
    warnings: int = 0
    violations_by_invariant: dict[str, int] = field(default_factory=dict)
    suppressed: dict[str, int] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    #: Checks the auditing machine could not run (:data:`NOT_CHECKED`): not passed.
    not_checked: int = 0

    @property
    def passed(self) -> bool:
        return self.audited and self.violations == 0 and self.not_checked == 0

    def invariants(self) -> dict[str, str]:
        failed = {inv for inv, n in self.violations_by_invariant.items() if n}
        unchecked = {f.invariant for f in self.findings if f.severity == NOT_CHECKED}
        return {
            inv: "not audited"
            if not self.audited
            else "FAIL"
            if inv in failed
            else "not checked"
            if inv in unchecked
            else "pass"
            if self.checked.get(inv)
            else "n/a"
            for inv in INVARIANTS
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        data["invariants"] = self.invariants()
        data["findings"] = [f.to_dict() for f in self.findings]
        data["info"] = {k: v for k, v in self.info.items() if k != "first_prompts"}
        return data


@dataclass
class RemovedCell:
    """A cell of a method the study removed (``study.REMOVED_METHODS``): listed, never
    audited. ``root`` is spelled like :attr:`CellAudit.root`."""

    cell_id: str
    root: str
    benchmark: str | None
    model: str | None
    method: str
    split_mode: str | None
    status: str | None
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateCell:
    """A cell directory holding a cell id another directory of the runs directory also
    holds (two hosts, a rerun as a new directory, a copy renamed in place), set aside: the
    audit reads only the chosen one (:func:`choose_directory`), the report sums every
    directory's money. ``role``: ``superseded`` (it sorts below the chosen one) or
    ``newer_incomplete`` (a newer run that is not complete, while an older complete one is
    reported)."""

    cell_id: str
    root: str
    role: str
    chosen_root: str
    started_at: str | None
    status: str | None
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    runs_dir: str
    frozen_fingerprint: str | None
    auditor_fingerprint: str
    expected_upstream: str
    cells: list[CellAudit]
    findings: list[Finding]
    generated_at: str
    #: What a merged view's _merged.json says; None for a host's own
    #: runs directory.
    merged_view: dict[str, Any] | None = None
    #: The selected cells of removed methods (study.REMOVED_METHODS): set aside, never
    #: audited, never incomplete.
    removed: list[RemovedCell] = field(default_factory=list)
    #: The removal records the audit applied (``study.REMOVED_METHODS`` by default).
    removed_methods: Mapping[str, study.Removal] = field(default_factory=dict)
    #: Run-level violations no audited cell answers for (a check across cells that names
    #: none): they fail the whole audit.
    unattributed_violations: int = 0
    #: Cell directories set aside because another directory holds the same cell id.
    duplicates: list[DuplicateCell] = field(default_factory=list)
    #: Selected cells of a method that ``protocol.METHODS`` does not list and no removal
    #: record names: set aside, not audited, not reported (H10.3-role-policy).
    not_in_matrix: list[RemovedCell] = field(default_factory=list)
    #: Removed methods' cells whose directories were deleted, as their removal manifests
    #: list them (:func:`removed_manifests`).
    deleted: list[dict[str, Any]] = field(default_factory=list)
    #: Whether the auditing tree's protocol is the frozen value (None: nothing frozen).
    tree_current: bool | None = None

    @property
    def violations(self) -> int:
        """Every violation: the cells' own (with the copies of the run-level violations
        that name them) and the run-level violations that name no audited cell."""
        return sum(c.violations for c in self.cells) + self.unattributed_violations

    @property
    def incomplete(self) -> list[str]:
        return [c.cell_id for c in self.cells if not c.audited]

    @property
    def not_checked(self) -> list[str]:
        """Audited cells some check of which the auditing machine could not run."""
        return [c.cell_id for c in self.cells if c.audited and c.not_checked]

    @property
    def passed(self) -> bool:
        return (
            self.violations == 0
            and any(c.audited for c in self.cells)
            and not self.not_checked
        )

    @property
    def exit_code(self) -> int:
        if self.violations:
            return 1
        if (
            self.incomplete
            or self.not_checked
            or not any(c.audited for c in self.cells)
        ):
            return 2
        return 0

    def to_dict(self) -> dict[str, Any]:
        removed_ids: dict[str, set[str]] = defaultdict(set)
        for r in self.removed:
            removed_ids[r.method].add(r.cell_id)
        for d in self.deleted:
            removed_ids[str(d.get("method"))].add(str(d.get("cell_id")))
        not_in_matrix: dict[str, list[str]] = defaultdict(list)
        for r in self.not_in_matrix:
            if r.cell_id not in not_in_matrix[r.method]:
                not_in_matrix[r.method].append(r.cell_id)
        return {
            "schema": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "runs_dir": self.runs_dir,
            "frozen_fingerprint": self.frozen_fingerprint,
            "auditor_fingerprint": self.auditor_fingerprint,
            "expected_upstream": self.expected_upstream,
            "passed": self.passed,
            "violations": self.violations,
            "unattributed_violations": self.unattributed_violations,
            "incomplete": self.incomplete,
            "not_checked": self.not_checked,
            # The only checks that can fail a cell.
            "validity_checks": sorted(VALIDITY_CHECKS),
            "invariants": dict(INVARIANTS),
            "findings": [f.to_dict() for f in self.findings],
            "cells": [c.to_dict() for c in self.cells],
            "merged_view": self.merged_view,
            # Set aside, not audited: report.py never reads it,
            # it tells a reader why those cells have no entry above.
            "removed": {
                "methods": {
                    method: {
                        "decision": removal.decision,
                        "reason": removal.reason,
                        "cells": sorted(removed_ids.get(method, ())),
                    }
                    for method, removal in sorted(self.removed_methods.items())
                },
                "cells": [r.to_dict() for r in self.removed],
                "deleted": [dict(d) for d in self.deleted],
            },
            "duplicates": [d.to_dict() for d in self.duplicates],
            "not_in_matrix": {
                "cells": [r.to_dict() for r in self.not_in_matrix],
                "methods": {m: sorted(ids) for m, ids in sorted(not_in_matrix.items())},
            },
        }

    def to_markdown(self) -> str:
        lines = [
            "# skill_bench audit (H10)",
            "",
            f"- Generated: {self.generated_at}",
            f"- Runs: `{self.runs_dir}`",
            *(
                [f"- Merged view (`_merged.json`): {self.merged_view.get('summary')}"]
                if self.merged_view
                else []
            ),
            f"- Frozen protocol fingerprint: `{self.frozen_fingerprint}`",
            f"- Auditor tree fingerprint: `{self.auditor_fingerprint}`",
            "- Auditor tree = frozen value: "
            + (
                "yes"
                if self.tree_current
                else "no (the checks that read this tree list what differs, per "
                "cell, under 'Not re-verified')"
                if self.tree_current is False
                else "unknown"
            ),
            f"- Expected upstream: `{self.expected_upstream}`",
            "- Violations: only the validity checks "
            f"({len(VALIDITY_CHECKS)}, `validity_checks` in audit.json) fail a cell; "
            "every other check is recorded as a warning",
            f"- Result: **{'PASS' if self.passed else 'FAIL'}** "
            f"({self.violations} violation(s); "
            f"{len(self.incomplete)} cell(s) not audited)"
            + (
                f"; {len(self.not_checked)} cell(s) not checked"
                if self.not_checked
                else ""
            )
            + (
                f"; {len(self.removed)} cell(s) of removed methods set aside (see "
                "'Removed methods')"
                if self.removed
                else ""
            )
            + (
                f"; {len(self.duplicates)} duplicate cell director(ies) set aside (see "
                "'Duplicate cell directories')"
                if self.duplicates
                else ""
            )
            + (
                f"; {len(self.not_in_matrix)} cell(s) of methods not in the matrix set "
                "aside (see 'Methods not in the matrix')"
                if self.not_in_matrix
                else ""
            ),
            "",
            "| cell | status | "
            + " | ".join(INVARIANTS)
            + " | violations | warnings |",
            "|---|---|" + "---|" * len(INVARIANTS) + "---|---|",
        ]
        for cell in self.cells:
            marks = cell.invariants()
            lines.append(
                f"| `{cell.cell_id}` | {cell.status} | "
                + " | ".join(marks[i] for i in INVARIANTS)
                + f" | {cell.violations} | {cell.warnings} |"
            )
        lines += [
            "",
            "## Versions",
            "",
            "Recorded per cell, not enforced across cells: the "
            "protocol the audited attempt ran under (current: the frozen value; earlier: "
            "a known earlier one; **unknown**: no freeze produced it), its model-visible "
            "code and the commit it started from; an attempt that ran under more than one "
            "version lists each later one.",
            "",
            "| cell | protocol | harness | mirix | commit |",
            "|---|---|---|---|---|",
            *(_versions_row(cell) for cell in self.cells),
        ]
        segments = [
            (cell, seg)
            for cell in self.cells
            for seg in cell.info.get("version_segments") or ()
        ]
        if segments:
            lines += ["", "| cell | from segment | at | phase | protocol | harness |"]
            lines.append("|---|---|---|---|---|---|")
            for cell, seg in segments:
                code = seg.get("code") if isinstance(seg.get("code"), Mapping) else {}
                known = "" if seg.get("known") else " (not registered)"
                lines.append(
                    f"| `{cell.cell_id}` | {seg.get('segment')} | {seg.get('at')} | "
                    f"{seg.get('phase')} | `{str(seg.get('fingerprint'))[:12]}`{known} | "
                    f"`{str(code.get('harness'))[:12]}` |"
                )
        lines += ["", "## Invariants", ""]
        lines += [f"- **{k}**: {v}" for k, v in INVARIANTS.items()]
        unverified = [
            (cell, entry)
            for cell in self.cells
            for entry in cell.info.get("not_reverified") or ()
        ]
        if unverified:
            lines += [
                "",
                "## Not re-verified under the cell's own version",
                "",
                "Checks that read this tree (its templates, splits, model metadata or "
                "constants), whose inputs differ from the cell's own protocol version "
                "or are not recorded for it: their findings are warnings. Audit such a "
                "cell from a checkout of its "
                "version to re-verify them.",
                "",
                "| cell | check | findings | inputs |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| `{cell.cell_id}` | `{entry['check']}` | {entry['findings']} | "
                + "; ".join(str(i) for i in entry["inputs"])
                + " |"
                for cell, entry in unverified
            ]
        if self.findings:
            lines += ["", "## Across cells", ""]
            lines += [
                f"- {f.severity} {f.invariant} `{f.check}`{_downgraded_text(f)}: "
                f"{f.detail}" + (f" (cells: {', '.join(f.cells)})" if f.cells else "")
                for f in self.findings
            ]
        for cell in self.cells:
            if not cell.findings and cell.audited:
                continue
            lines += ["", f"## `{cell.cell_id}`", ""]
            if not cell.audited:
                lines.append(f"Not audited: {cell.reason}")
            for f in cell.findings:
                where = f" (item `{f.item}`)" if f.item else ""
                across = f" (across cells: {', '.join(f.cells)})" if f.cells else ""
                severity = "not checked" if f.severity == NOT_CHECKED else f.severity
                lines.append(
                    f"- {severity} {f.invariant} `{f.check}`{where}"
                    f"{_downgraded_text(f)}{across}: {f.detail}"
                )
            for key, n in cell.suppressed.items():
                lines.append(f"- ... {n} more `{key}`")
        if self.duplicates:
            lines += [
                "",
                "## Duplicate cell directories",
                "",
                "Directories holding a cell id another directory also holds: the audit "
                "reads the latest-started complete one (else the latest-started one) and "
                "sets the others aside; the report sums every "
                "directory's money.",
                "",
                "| cell | directory | role | audited directory | started | status |",
                "|---|---|---|---|---|---|",
            ]
            lines += [
                f"| `{d.cell_id}` | `{Path(d.root).name}` | {d.role} | "
                f"`{Path(d.chosen_root).name}` | {d.started_at or '-'} | "
                f"{d.status or '-'} |"
                for d in self.duplicates
            ]
        if self.not_in_matrix:
            lines += [
                "",
                "## Methods not in the matrix",
                "",
                "Cells of a method `protocol.METHODS` does not list and no removal "
                "record names: set aside, not audited and not reported "
                "(H10.3-role-policy).",
                "",
                "| cell | directory | method | status |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| `{r.cell_id}` | `{Path(r.root).name}` | `{r.method}` | "
                f"{r.status or '-'} |"
                for r in self.not_in_matrix
            ]
        if self.removed or self.deleted:
            lines += [
                "",
                "## Removed methods",
                "",
                "Cells of methods the study removed (`study.REMOVED_METHODS`): set "
                "aside, never audited, in no comparison across cells and never an "
                "incomplete cell; listed from their removal manifest once their "
                "directories are deleted.",
                "",
            ]
            methods = {r.method for r in self.removed} | {
                str(d.get("method")) for d in self.deleted
            }
            for method, removal in sorted(self.removed_methods.items()):
                if method in methods:
                    lines.append(f"- `{method}`, {removal.decision}: {removal.reason}")
            lines += ["", "| cell | directory | status |"]
            lines.append("|---|---|---|")
            lines += [
                f"| `{r.cell_id}` | `{Path(r.root).name}` | {r.status or '-'} |"
                for r in self.removed
            ]
            lines += [
                f"| `{d.get('cell_id')}` | deleted {d.get('deleted_at') or '(deleting)'} "
                f"on {d.get('host') or '?'}, manifest `{d.get('manifest')}` | "
                f"{d.get('status') or '-'} |"
                for d in self.deleted
            ]
        return "\n".join(lines) + "\n"


def _downgraded_text(finding: Finding) -> str:
    """How a finding's line says why it is a warning (``AuditReport.to_markdown``)."""
    if finding.downgraded == RECORD:
        return " (recorded)"
    if finding.downgraded == NOT_REVERIFIED:
        return " (not re-verified under its version)"
    return ""


def _versions_row(cell: CellAudit) -> str:
    """A row of the audit's versions table (``AuditReport.to_markdown``)."""
    recorded = cell.info.get("protocol") or {}
    code = cell.info.get("code") or {}
    fingerprint = recorded.get("fingerprint")
    if not fingerprint:
        version = "-"
    else:
        state = (
            "current"
            if recorded.get("current")
            else "earlier"
            if recorded.get("known")
            else "in its attempt record"
            if cell.info.get("protocol_record") == "attempt record"
            else "**unknown**"
        )
        version = f"`{str(fingerprint)[:12]}` ({state})"
        if cell.info.get("version_segments"):
            version += f", +{len(cell.info['version_segments'])} later version(s)"

    def short(value: Any, n: int) -> str:
        return f"`{str(value)[:n]}`" if value else "-"

    return (
        f"| `{cell.cell_id}` | {version} | {short(code.get('harness'), 12)} | "
        f"{short(code.get('mirix'), 12)} | {short(cell.info.get('commit'), 10)} |"
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_ALNUM = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_IDENTIFIER_MARKS = frozenset("/:_.=?&")


def normalise(text: str) -> str:
    """Case- and whitespace-insensitive form (the H10.1 string comparison)."""
    return " ".join(text.lower().split())


def identifier_like(needle: str) -> bool:
    """A normalised label string specific enough that no other item's text holds it by
    chance: at least 6 characters, with a path/id delimiter or both letters and digits."""
    if len(needle) < 6:
        return False
    if any(ch in _IDENTIFIER_MARKS for ch in needle):
        return True
    return any(ch.isalpha() for ch in needle) and any(ch.isdigit() for ch in needle)


def uid_problem(uid: str, raw_id: str, salt: bytes | None) -> tuple[str, str] | None:
    """Why ``uid`` is not the opaque task id of ``raw_id`` (H5), as ``(check, detail)``,
    or None.

    * ``uid_format``: not ``"t"`` + 20 lowercase hex characters (:data:`UID_RE`), or the
      raw id itself;
    * ``uid_salt``: not HMAC(cell salt, raw id), recomputed when the auditor has the salt.

    A uid that passes is opaque by construction: HMAC-SHA256 under the cell's secret salt
    says nothing about the raw id. It is never judged by substrings: a random 20-hex uid
    holds a given 3-digit string with probability about (21 - 3) / 16**3 = 0.44%, so a
    ``raw_id in uid`` test would report short numeric raw ids as leaks by chance. Whether
    a raw id
    itself reached a method is the label-string search (``_check_labels``: every raw id
    is a label of its item, matched only as a whole token).
    """
    if not UID_RE.fullmatch(uid):
        return (
            "uid_format",
            f"task uid {uid!r} is not {protocol.OPAQUE_UID_PREFIX!r} + "
            f"{protocol.OPAQUE_UID_HEX_CHARS} lowercase hex characters",
        )
    if uid == raw_id:
        return "uid_format", f"task uid {uid!r} is the raw id"
    if salt is not None and uid != opaque_task_uid(salt, raw_id):
        return "uid_salt", f"task uid {uid} is not HMAC(cell salt, raw id)"
    return None


class Corpus:
    """Strings joined into one text, so thousands of needles are each found with one
    ``str.find`` pass; matches count only at word boundaries."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._offsets: list[int] = []
        self._owners: list[tuple[str, str]] = []
        self._size = 0
        self._text: str | None = None

    def add(self, text: str, owner: str, where: str) -> None:
        if not text:
            return
        self._offsets.append(self._size)
        self._owners.append((owner, where))
        self._parts.append(text)
        self._size += len(text) + 1
        self._text = None

    def __len__(self) -> int:
        return len(self._parts)

    def find(self, needle: str) -> Iterator[tuple[str, str]]:
        """``(owner, where)`` of every text holding ``needle`` at word boundaries."""
        if self._text is None:
            self._text = "\x00".join(self._parts)
        text = self._text
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                return
            j = i + len(needle)
            left = needle[0] not in _ALNUM or i == 0 or text[i - 1] not in _ALNUM
            right = needle[-1] not in _ALNUM or j >= len(text) or text[j] not in _ALNUM
            if left and right:
                yield self._owners[bisect_right(self._offsets, i) - 1]
            start = i + 1


def _holds(text: str, needle: str) -> bool:
    """``needle`` occurs in ``text`` at word boundaries (the H10.1 comparison)."""
    corpus = Corpus()
    corpus.add(text, "-", "-")
    return next(corpus.find(needle), None) is not None


def _same_score(score: Any, recorded: Mapping[str, Any]) -> bool:
    """A :class:`~evals.skill_bench.envs.base.Score` equals a recorded score dict."""
    tolerance = 1e-9
    if abs(float(score.primary) - float(recorded.get("primary") or 0.0)) > tolerance:
        return False
    metrics = dict(score.metrics)
    theirs = dict(recorded.get("metrics") or {})
    if set(metrics) != set(theirs):
        return False
    if any(abs(float(metrics[k]) - float(theirs[k])) > tolerance for k in metrics):
        return False
    return bool(score.answered) == bool(recorded.get("answered", True))


def _blank(text: str, statements: Iterable[str]) -> str:
    for statement in sorted({s for s in statements if s}, key=len, reverse=True):
        text = text.replace(statement, " ")
    return text


def harness_strings(
    value: Any, path: str = "$", skip_roles: Sequence[str] = ("assistant", "tool")
) -> Iterator[tuple[str, str]]:
    """``(path, text)`` of every string the harness authored in a method-visible object:
    model output (``assistant``), tool output and ``data:`` URLs are skipped."""
    if isinstance(value, Mapping):
        role = value.get("role")
        if isinstance(role, str) and role in skip_roles:
            return
        for key, child in value.items():
            yield from harness_strings(child, f"{path}.{key}", skip_roles)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from harness_strings(child, f"{path}[{i}]", skip_roles)
    elif isinstance(value, str) and not value.startswith("data:"):
        yield path, value


def all_strings(value: Any) -> Iterator[str]:
    """Every string in a method-visible object, model and tool output included (``data:``
    URLs skipped): everything a method was given."""
    if isinstance(value, Mapping):
        for child in value.values():
            yield from all_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from all_strings(child)
    elif isinstance(value, str) and not value.startswith("data:"):
        yield value


def specific_answer(needle: str) -> bool:
    """A normalised gold answer specific enough that a method holding it must have got it
    from somewhere: identifier-like, or at least 4 characters with a digit."""
    return identifier_like(needle) or (
        len(needle) >= 4 and any(ch.isdigit() for ch in needle)
    )


def strong_answer(needle: str) -> bool:
    """A specific answer too long to sit in a shared memory by coincidence: identifier-like,
    or at least 6 characters with a digit (``256,368``, not ``1944``)."""
    return identifier_like(needle) or (
        len(needle) >= 6 and any(ch.isdigit() for ch in needle)
    )


#: Words any memory, template or document may hold, so a gold answer made only of them says
#: nothing about where a method got it: function words, number and time words, and
#: the vocabulary of documents and of the method templates. Lower case, as ``normalise``.
COMMON_ANSWER_WORDS = frozenset(
    """
    a an the and or nor but if so of in on at to for from by with without as into onto
    over under above below before after between per via about against during through up
    down out off is are was were be been being am do does did has have had can could will
    would shall should may might must it its this that these those there here which who
    whom whose what when where why how not no yes none nil all any some each every both
    either neither more most less least than then also only just very same other such
    he she they them their his her him we our us you your me my mine one ones own
    zero two three four five six seven eight nine ten eleven twelve twenty thirty forty
    fifty sixty seventy eighty ninety hundred thousand million billion first second third
    fourth fifth last half quarter dozen several many few once twice percent cent
    day days week weeks month months year years hour hours minute minutes min mins
    seconds time times date dates today daily weekly monthly annual yearly morning
    afternoon evening night january february march april may june july august september
    october november december jan feb mar apr jun jul aug sep sept oct nov dec monday
    tuesday wednesday thursday friday saturday sunday mon tue wed thu fri sat sun
    answer answers question questions document documents page pages table tables figure
    figures chart charts graph graphs form forms letter letters memo memorandum report
    reports note notes section sections title titles name names number numbers total
    totals subtotal amount amounts value values item items list lists row rows column
    columns heading header footer text image box field line lines signature signed stamp
    logo address phone fax email subject reference attachment enclosure copy draft final
    approved approval confidential company companies corporation corp inc ltd limited
    group groups department dept office offices university college institute school
    committee council board association society center centre division program
    programme project projects meeting meetings agenda budget cost costs price prices
    sales income revenue profit profits expense expenses tax rate rates percentage
    average mean median maximum minimum high low increase decrease change level levels
    unit units size weight height width length area volume study studies research data
    result results analysis test tests sample samples control summary conclusion
    introduction background method methods objective objectives purpose goal goals plan
    plans policy review update information details description product products brand
    brands service services market marketing customer customers client author president
    director manager chairman secretary member members staff dear sincerely regards
    mrs miss prof sir madam messrs
    new old current previous next general special national international american united
    states state city county street avenue road north south east west central age
    memory task tasks step steps strategy strategies pattern patterns example examples
    problem problems solution solutions approach instruction instructions
    workflow workflows skill skills content assistant agent environment
    """.split()
)
_ANSWER_WORD = re.compile(r"[^\W\d_]+")
#: ``text_answer_tier`` values.
TEXT_STRONG = "strong"
TEXT_WEAK = "weak"


def text_answer_tier(needle: str) -> str | None:
    """How telling a normalised gold answer without a digit or an identifier mark is when
    it turns up in what a method produced (``specific_answer`` covers the rest).

    * ``strong``: at least 8 characters and two distinct content words (words of 3+
      letters outside :data:`COMMON_ANSWER_WORDS`): ``scissors menthol fresh``,
      ``perkin elmer corp``. No template writes such a phrase by chance;
    * ``weak``: two or more words, or at least 8 characters, with one content word:
      ``nashville``, ``university of maine``. A single word may sit in a template;
    * None: shorter, or made of common words only (``yes``, ``four weeks``, ``final
      approval``): not scanned.
    """
    if not needle or any(ch.isdigit() for ch in needle) or specific_answer(needle):
        return None
    words = _ANSWER_WORD.findall(needle)
    if len(words) < 2 and len(needle) < 8:
        return None
    content = {w for w in words if _content_word(w)}
    if not content:
        return None
    return TEXT_STRONG if len(content) >= 2 and len(needle) >= 8 else TEXT_WEAK


def _content_word(word: str) -> bool:
    """A word of 3+ letters outside :data:`COMMON_ANSWER_WORDS` (lower case)."""
    return len(word) >= 3 and word not in COMMON_ANSWER_WORDS


def word_text(text: str) -> str:
    """``text`` as its lower-case letter words joined by single spaces: the form text
    answers are compared in, so punctuation, case and line breaks cannot hide one
    (``Scissors-Menthol  Fresh`` holds ``scissors menthol fresh``)."""
    return " ".join(_ANSWER_WORD.findall(text.lower()))


#: How many test items' strong text answers a frozen memory may hold, unexplained, before
#: that is a violation: at least this many ...
MEMORY_TEXT_ANSWERS_MIN = 3
#: ... or this share of the cell's test items, whichever is more.
MEMORY_TEXT_ANSWERS_SHARE = 0.02


def memory_text_answers_allowed(test_items: int) -> int:
    """Test items whose strong text gold answer a frozen memory may hold, although no
    learning input held it, and still only draw warnings: a memory learned from pages of
    one source may name what a few test pages name, not what many do. 7 of 350 test
    items; 3 for a small cell."""
    return max(MEMORY_TEXT_ANSWERS_MIN, int(test_items * MEMORY_TEXT_ANSWERS_SHARE))


def text_answer_needle(needle: str) -> tuple[str, str] | None:
    """The word sequence a normalised text gold answer is searched as, with its tier
    (``text_answer_tier``); None when it is not searched.

    The words before the first and after the last content word (an article, a title, a
    common noun such as ``company``) are dropped, so they can neither hide the answer
    nor explain it away: ``the coca-cola company`` is searched as ``coca cola``, which
    ``Coca-Cola Company`` holds. Only a core of two words or more that keeps the whole
    answer's tier replaces it; otherwise the whole answer is searched (``university of
    maine``, not ``maine``; ``washington university``, not ``washington``: one word says
    less than the phrase)."""
    tier = text_answer_tier(needle)
    if tier is None:
        return None
    words = word_text(needle).split()
    content = [i for i, word in enumerate(words) if _content_word(word)]
    core = words[content[0] : content[-1] + 1]
    if len(core) >= 2 and text_answer_tier(" ".join(core)) == tier:
        return " ".join(core), tier
    return " ".join(words), tier


_CURRENCY = re.compile(r"[$\u00a2-\u00a5\u20a0-\u20bf]")
#: A thousands separator: a comma, space or apostrophe between a digit and a group of
#: exactly three digits (``256,865``, ``256 865``, ``1'000``).
_GROUP_SEPARATOR = re.compile(r"(?<=\d)[,'\u2019\s](?=\d{3}(?!\d))")


def digit_form(text: str) -> str:
    """A normalised text without currency signs and thousands separators: the second form
    numeric gold answers are compared in, so ``256865``, ``256 865`` and ``$256,865``
    all hold ``256,865``, and ``500.00`` holds ``$500.00``.

    Decimal points are kept, and only a separator followed by exactly three digits goes:
    ``12,34`` and ``2019 2020`` stay apart. ``3 100`` does become ``3100``. Such a
    false join rarely makes a number of 6+ digits (the length that is a violation in the
    memory), and when the learning inputs the memory came from hold the same text, the
    join explains the answer there as well."""
    text = _CURRENCY.sub(" ", text)
    text = _GROUP_SEPARATOR.sub("", text)
    return " ".join(text.split())


#: Digits a gold answer needs to be a violation in the frozen memory (``strong_in_memory``)
#: ...
MEMORY_ANSWER_DIGITS = 5
#: ... or, without a digit, one of these marks: a path, key, query or address
#: (``robotics/analytical chemist``, ``jdoe@example.com``). ``.``, ``&``, ``,``, ``-`` and
#: spaces mark names (``r.j. reynolds``, ``johnson & johnson``), which are not.
_MEMORY_IDENTIFIER_MARKS = frozenset("/:_=?@")
#: Words that make a short number a generic label or amount in the memory (``step 2``,
#: ``schedule 11``, ``1/16 tsp.``), besides :data:`COMMON_ANSWER_WORDS` (which holds the
#: month names: ``december 7``), and the common units and count nouns (``100 dollars``,
#: ``25 people``, ``1,250 tons``). A short number with any other word is a
#: code or an address (``ma 1253 ag``, ``1455 broad street``): no memory holds it by
#: coincidence.
MEMORY_LABEL_WORDS = frozenset(
    """
    step figure fig figs page pp schedule attachment table tab exhibit appendix annex
    item chapter chap ch part article art no nos num number line question room box grade
    level phase round stage version ver vol volume issue series class type sheet slide
    clause paragraph para point option case plate rev revision edition ed wk yr mo qtr fy
    tsp tbsp cup cups oz lb lbs g kg mg mcg ml cl dl cc cm mm km inch inches ft feet yd yds
    gal qt pt fl sq cu pct kcal cal calories hr hrs sec secs rpm mph psi deg
    glass glasses serving servings piece pieces bottle bottles can cans pack packs
    pm p m noon midnight est edt cst cdt mst mdt pst pdt gmt utc st nd rd th
    dollar dollars usd cents eur euro euros yen pound pounds
    people person persons employee employees worker workers member members staff
    student students patient patients household households family families
    mile miles meter meters metre metres kilometer kilometers kilometre kilometres
    gram grams kilogram kilograms ton tons tonne tonnes ounce ounces liter liters
    litre litres gallon gallons acre acres hectare hectares
    degree degrees point points percentage pts unit units bn mn
    """.split()
)
#: A zero fraction (``500.00``): the same amount as without it.
_ZERO_FRACTION = re.compile(r"(?<=\d)\.0+(?!\d)")


def answer_digits(needle: str) -> int:
    """The digits of a normalised gold answer as numbers are compared (``digit_form``),
    without a zero fraction: ``$1,000`` and ``$500.00`` have 4 and 3, ``256,865`` 6."""
    return sum(ch.isdigit() for ch in _ZERO_FRACTION.sub("", digit_form(needle)))


def strong_in_memory(needle: str) -> bool:
    """A specific gold answer (``specific_answer``) that the frozen memory cannot hold by
    coincidence, so it is a violation there:

    * with a digit: at least :data:`MEMORY_ANSWER_DIGITS` digits (``answer_digits``):
      ``256,865``, ``(555) 253 3012``, ``march 27, 1979``; or, with fewer, a code or an
      address of 6+ characters: a word no label, unit or common word explains
      (:data:`MEMORY_LABEL_WORDS`, :data:`COMMON_ANSWER_WORDS`) next to the number
      (``ma 1253 ag``, ``5954 b``, ``1455 broad street``). ``$1,000``, ``$ 1795``,
      ``$500.00``, ``1/16 tsp.``, ``5 to 6 glasses``, ``step 2``, ``page 14``,
      ``december 7`` are not: one memory serves every test item, and a number of four
      digits or fewer, alone or with a label, a unit or a month, is what any memory may
      write. Nor is a short
      number in a telling phrase (``american cheese production by states, 1952``):
      that is counted with the text answers (``counted_in_memory``);
    * without one: a path, key, query or address mark (:data:`_MEMORY_IDENTIFIER_MARKS`).

    Strength is judged on the digits, not the formatted string, because the memory is
    searched in ``digit_form`` too: a bare ``1000`` in the memory holds ``$1,000``. What is
    specific but not strong is a warning, or counted with the text answers
    (``counted_in_memory``)."""
    if any(ch.isdigit() for ch in needle):
        if answer_digits(needle) >= MEMORY_ANSWER_DIGITS:
            return True
        if len(needle) < 6 or text_answer_tier(word_text(needle)) == TEXT_STRONG:
            return False
        return any(
            word not in COMMON_ANSWER_WORDS and word not in MEMORY_LABEL_WORDS
            for word in _ANSWER_WORD.findall(needle)
        )
    return len(needle) >= 6 and any(ch in _MEMORY_IDENTIFIER_MARKS for ch in needle)


def counted_in_memory(needle: str) -> bool:
    """A specific gold answer below ``strong_in_memory`` that the frozen memory's count
    rule (``memory_text_answers_allowed``) judges with the strong text answers: a name
    specific only through its punctuation (``r.j. reynolds``, ``pathmark stores inc.``:
    documents of one source share names), or a short number in a telling
    phrase (``american cheese production by states, 1952``, whose words alone are a
    strong text answer). Other short ones (``$1,000``, ``step 2``) are warnings."""
    if not specific_answer(needle) or strong_in_memory(needle):
        return False
    if not any(ch.isdigit() for ch in needle):
        return True
    return text_answer_tier(word_text(needle)) == TEXT_STRONG


class Texts:
    """Texts searched three ways: normalised (``normalise``: specific answers, labels), in
    ``digit_form`` (numeric answers, whatever their thousands separators and currency
    signs) and as word sequences (``word_text``: text answers)."""

    def __init__(self) -> None:
        self.plain = Corpus()
        self.digits = Corpus()
        self.words = Corpus()

    def add(self, text: str, owner: str, where: str) -> None:
        normalised = normalise(text)
        self.plain.add(normalised, owner, where)
        self.digits.add(digit_form(normalised), owner, where)
        self.words.add(word_text(text), owner, where)

    def holds(self, needle: str, *, textual: bool) -> bool:
        """``needle`` occurs: a text answer's word sequence (``textual``), else a
        normalised string, as written or, when it has a digit, in ``digit_form``."""
        if textual:
            return _found(self.words, needle)
        if _found(self.plain, needle):
            return True
        if not any(ch.isdigit() for ch in needle):
            return False
        digits = digit_form(needle)
        return bool(digits) and _found(self.digits, digits)


def _found(corpus: Corpus, needle: str) -> bool:
    return next(corpus.find(needle), None) is not None


def memory_texts(snapshot: Any) -> list[str]:
    """The normalised texts of a frozen memory to search for answers and labels: its JSON,
    and each string (and key) in it on its own. JSON escapes a line break as ``\\n``, so a
    value right after one would miss its word boundary in the JSON alone."""
    texts = [normalise(json.dumps(snapshot, ensure_ascii=False))]

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(key, str):
                    texts.append(normalise(key))
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, str) and not value.startswith("data:"):
            texts.append(normalise(value))

    walk(snapshot)
    return [t for t in texts if t]


def forbidden_keys(value: Any, path: str = "$") -> list[str]:
    """Paths of every label key (``FORBIDDEN_LABEL_KEYS``) at any depth."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_LABEL_KEYS:
                found.append(f"{path}.{key}")
            found.extend(forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{path}[{i}]"))
    return found


def blob_placeholders(value: Any) -> Any:
    """``value`` with every stored blob reference (``{"$blob": sha256}``, the trajectory
    record's image data URLs) replaced by a ``data:`` URL naming it: the shape and label
    checks treat it as the image it stands for, without reading the blob."""
    if isinstance(value, Mapping):
        if set(value) == {BLOB_KEY}:
            return f"data:application/x-skill-bench-blob;sha256,{value[BLOB_KEY]}"
        return {k: blob_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [blob_placeholders(v) for v in value]
    return value


def request_fingerprint(request: Mapping[str, Any]) -> str:
    """The rollout's first-prompt hash (``agent/rollout.py`` ``_Request.fingerprint``)."""
    return sha256_hex(
        canonical_json(
            {
                "messages": request.get("messages"),
                "tools": request.get("tools"),
                "tool_choice": request.get("tool_choice"),
            }
        )
    )


def sent_request_sha256(request: Mapping[str, Any], wire: Mapping[str, Any]) -> str:
    """sha256 of the body the gateway forwards for ``request`` (blobs inlined) under the
    ledger line's ``wire`` values (the provider routing too):
    ``rewrite_chat_body``, then the gateway's compact JSON (the ledger's
    ``request_sha256``)."""
    effort = (wire.get("reasoning") or {}).get("effort")
    provider = wire.get("provider")
    body = rewrite_chat_body(
        dict(request),
        model=str(wire.get("model")),
        max_tokens=int(wire["max_tokens"]),
        seed=wire.get("seed"),
        effort=str(effort),
        provider=provider if isinstance(provider, Mapping) else None,
    )
    content = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def model_family(served: str) -> str:
    """``model_served`` without the provider prefix and a trailing dated snapshot suffix."""
    name = served.strip().lower().rsplit("/", 1)[-1]
    return re.sub(r"[-@:](?:\d{4}-?\d{2}-?\d{2}|\d{4}|\d{8})(?:[-:.]\w+)?$", "", name)


@lru_cache(maxsize=None)
def requested_families(model: str) -> frozenset[str]:
    """The model families OpenRouter may serve ``model`` as (H1.8): its own id's and its
    ``canonical_slug``'s in ``config/models.json`` (``openai/gpt-6-luna-20260922`` for
    ``openai/gpt-6-luna``). Another variant is another model: ``gpt-6-luna-pro`` (the same
    weights in another reasoning mode) or ``gpt-5.6-luna``."""
    families = {model_family(model)}
    entry = protocol.load_models()["models"].get(model) or {}
    slug = entry.get("canonical_slug")
    if isinstance(slug, str) and slug:
        families.add(model_family(slug))
    return frozenset(families)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _checkpoint_ordinal(cid: Any) -> int | None:
    """The ordinal of a checkpoint id (``cell_records.checkpoint_ordinal``), None for
    none or an unreadable one."""
    if cid is None:
        return None
    try:
        return checkpoint_ordinal(*parse_checkpoint_id(str(cid)))
    except ValueError:
        return None


def _ts(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _superseded_marker(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The superseded marker of an earlier attempt (``--new-attempt --note`` on a
    tested cell): ``attempt.json`` ``superseded``, else ``{note}`` of an attempt
    invalidated as ``superseded``; None for any other attempt."""
    marker = state.get("superseded")
    if isinstance(marker, Mapping):
        return marker
    if state.get("invalidation") == "superseded":
        return {"note": state.get("invalid_reason"), "at": state.get("invalidated_at")}
    return None


def _number(value: Any) -> float | None:
    """A score's number, None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _canon(value: Any) -> str:
    """A JSON value compared by content (settings, routings)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Versions (recorded per attempt and per segment)
# ---------------------------------------------------------------------------


def _tree_version(frozen: str | None) -> dict[str, Any]:
    """The auditing tree as a :func:`protocol.known_versions` entry (blocking: it hashes
    every fingerprinted file once): what the checks of :data:`TREE_INPUTS` read, and what
    a cell's version is diffed against (``current``: it is the frozen value)."""
    manifest = protocol.fingerprint_manifest()
    fingerprint = str(manifest["fingerprint"])
    return {
        "fingerprint": fingerprint,
        "frozen_at": None,
        "reason": "the auditing tree",
        "git": None,
        "constants_sha256": manifest.get("constants_sha256"),
        "constants": dict(manifest.get("constants") or {}),
        "inconsistent": False,
        "files": dict(manifest.get("files") or {}),
        "current": fingerprint == frozen,
        # What each model is requested as under this tree's models.json, by its hash.
        "model_slugs": protocol.model_slugs_of(manifest),
    }


def diff_versions(
    versions: Mapping[str, Mapping[str, Any]],
    carried: Mapping[str, Mapping[str, Any]],
    tree: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """The versions a cell's protocol is diffed with (pure): the recorded ones, the
    cell's own records (``carried``) and the auditing tree's entry (:func:`_tree_version`),
    which describes its fingerprint exactly (its constants and files give it).

    The tree's entry keeps the ``model_slugs`` of an entry of the same fingerprint it
    replaces. Once the tree is the frozen value, the frozen record's entry for it holds
    the slugs of every ``config/models.json`` frozen so far, an earlier version's among
    them, and :func:`protocol.version_diff` needs both files' slugs to see that a model
    added to the file touches no cell of an earlier version (``model_slugs_recorded``).
    The tree's own slugs, of the file it hashes, win for its hash."""
    merged: dict[str, Mapping[str, Any]] = {**versions, **carried}
    fingerprint = str(tree["fingerprint"])
    entry = dict(tree)
    replaced = merged.get(fingerprint)
    if replaced is not None:
        entry["model_slugs"] = {
            **protocol.recorded_model_slugs(replaced),
            **protocol.recorded_model_slugs(tree),
        }
    merged[fingerprint] = entry
    return merged


def _inconsistent_text(fingerprints: Sequence[Any]) -> str:
    """The input a version whose record contradicts its fingerprint stands for."""
    return (
        "a record that does not reproduce its fingerprint ("
        + ", ".join(str(fp)[:12] for fp in fingerprints)
        + ")"
    )


def _changed_inputs(
    diff: Mapping[str, Any] | None,
    benchmark: str,
    constants: frozenset[str],
    prefixes: Sequence[str],
) -> list[str]:
    """What of ``constants`` (protocol constant keys) and of the fingerprinted files under
    ``prefixes`` (``{benchmark}`` formatted) a :func:`protocol.version_diff` changed, as
    the report lists it (pure); ``[]`` when nothing did. A constant whose change touches
    only other benchmarks (``protocol.touches_benchmark``: another benchmark's
    ``split_sizes`` entry, a benchmark added to ``benchmarks``) is not named. A
    diff that cannot say (no file map, a constants hash or per-key constants not
    recorded, a record that contradicts its fingerprint) names that instead: fail
    closed."""
    if diff is None:
        return ["a file map that is not recorded"]
    out: list[str] = []
    if diff.get("inconsistent"):
        out.append(_inconsistent_text(diff["inconsistent"]))
    elif constants:
        changed = diff.get("constants_changed")
        keys = diff.get("constants_changed_keys")
        if changed is None:
            out.append("a constants hash that is not recorded")
        elif changed and not keys:
            out.append("constants of protocol.py (which keys is not recorded)")
        elif changed:
            # Another benchmark's entry of split_sizes (or a benchmark added
            # to or dropped from the list) does not change what this cell's checks read.
            out += [
                f"protocol.py constant {k}"
                for k in keys
                if k in constants and protocol.touches_benchmark(diff, k, benchmark)
            ]
    wanted = tuple(p.format(benchmark=benchmark) for p in prefixes)
    if wanted:
        out += [
            str(path)
            for kind in ("changed", "added", "removed")
            for path in diff.get(kind) or ()
            if str(path).startswith(wanted)
        ]
    return out


def _segment_index(segment: Mapping[str, Any]) -> int | None:
    index = segment.get("index")
    return index if _is_int(index) else None


def _segment_fingerprint(segment: Mapping[str, Any]) -> str | None:
    recorded = segment.get("protocol")
    if isinstance(recorded, Mapping) and recorded.get("fingerprint"):
        return str(recorded["fingerprint"])
    return None


def _from_text(segment: Mapping[str, Any]) -> str:
    """Where a version segment took over (``learning_from`` / ``test_from``)."""
    learning = segment.get("learning_from")
    if isinstance(learning, Mapping):
        return (
            f"learning position {learning.get('position')} ({learning.get('stage')}, "
            f"checkpoint {learning.get('checkpoint')})"
        )
    test = segment.get("test_from")
    if isinstance(test, Mapping):
        return (
            f"the test phase ({test.get('scored')} scored, {test.get('pending')} "
            "pending)"
        )
    return f"the {segment.get('phase') or '?'} phase"


#: A results or learning row of an item no provider could serve (run.py), shared with
#: the report (``infra/failures.provider_unserved_row``).
_provider_unserved = provider_unserved_row


def _refused_tries_before(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Raw id -> the consecutive tries left unscored by a no-endpoint refusal of the
    actor right before the item's scored ``provider_unserved`` row (results.jsonl order;
    ``infra/failures.refused_no_endpoint_try``, pure)."""
    trailing: dict[str, int] = {}
    before: dict[str, int] = {}
    for row in rows:
        key = row.get("raw_id") or row.get("item")
        if key is None:
            continue
        if row.get("score") is not None and provider_unserved_row(row):
            before.setdefault(str(key), trailing.get(str(key), 0))
        refused = refused_no_endpoint_try(row)
        trailing[str(key)] = trailing.get(str(key), 0) + 1 if refused else 0
    return before


def _changed_text(changed: Any) -> str:
    """A segment's ``changed`` record as one clause: protocol files and constant keys,
    model-visible code (and its files), commit, tree."""
    if not isinstance(changed, Mapping):
        return ""
    parts: list[str] = []
    diff = changed.get("protocol_diff")
    if isinstance(diff, Mapping):
        files = [
            str(p)
            for kind in ("changed", "added", "removed")
            for p in diff.get(kind) or ()
        ]
        if files:
            parts.append("protocol files " + ", ".join(files))
        keys = diff.get("constants_changed_keys")
        if keys:
            parts.append("constants " + ", ".join(str(k) for k in keys))
        elif diff.get("constants_changed"):
            parts.append("constants (which keys is not recorded)")
    elif changed.get("protocol"):
        parts.append("protocol (what is not recorded)")
    code = changed.get("code")
    if code:
        files = changed.get("code_files")
        named = []
        for key in code if isinstance(code, list) else ():
            entry = files.get(key) if isinstance(files, Mapping) else None
            listed = (
                [
                    str(p)
                    for kind in ("changed", "added", "removed")
                    for p in entry.get(kind) or ()
                ]
                if isinstance(entry, Mapping)
                else []
            )
            named.append(f"{key} ({', '.join(listed)})" if listed else str(key))
        parts.append("code " + ", ".join(named))
    if changed.get("commit"):
        parts.append("commit")
    if changed.get("tree"):
        parts.append("tree")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The memory slot, per benchmark
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _alfworld_templates() -> Any:
    from evals.skill_bench.envs.alfworld import AlfWorldTemplates

    return AlfWorldTemplates.load()


@lru_cache(maxsize=None)
def _webshop_templates() -> Any:
    from evals.skill_bench.envs.webshop import WebShopTemplates

    return WebShopTemplates.load()


def strip_memory(
    benchmark: str, messages: Sequence[Mapping[str, Any]], memory: str
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The first actor request's messages with ``memory`` taken out of its slot, using
    the frozen templates; ``(None, why)`` when the request is not the template with that
    memory in the slot."""
    msgs = [copy.deepcopy(dict(m)) for m in messages]
    if benchmark == "webshop":
        # One conversation without a system message:
        # verl-agent's prompt as the first user message, then one reply and one user
        # message per later step. A first call is that first message alone.
        t = _webshop_templates()
        if len(msgs) % 2 != 1 or any(
            m.get("role") != ("assistant" if i % 2 else "user")
            or not isinstance(m.get("content"), str)
            for i, m in enumerate(msgs)
        ):
            return None, "the WebShop request is not [user, (assistant, user)*] text"
        content = str(msgs[0]["content"])
        if memory.strip():
            prefix = t.memory_slot.format(memory=memory) + t.memory_joiner
            if not content.startswith(prefix):
                return None, "the memory is not at the start of the user prompt"
            if any(str(m["content"]).startswith(prefix) for m in msgs[2::2]):
                return None, "a later user message carries the memory slot"
            msgs[0]["content"] = content[len(prefix) :]
        return msgs, None
    if len(msgs) < 2 or msgs[0].get("role") != "system":
        return None, "the first actor request does not start with the system prompt"
    if benchmark == "alfworld":
        t = _alfworld_templates()
        if msgs[0].get("content") != t.system_prompt:
            return None, "the system prompt is not the frozen ALFWorld one"
        content = msgs[1].get("content")
        if (
            len(msgs) != 2
            or msgs[1].get("role") != "user"
            or not isinstance(content, str)
        ):
            return None, "the ALFWorld request is not [system, user text]"
        if memory.strip():
            prefix = t.memory_slot.format(memory=memory) + t.memory_joiner
            if not content.startswith(prefix):
                return None, "the memory is not at the start of the user prompt"
            msgs[1]["content"] = content[len(prefix) :]
        return msgs, None
    return None, f"no memory-slot rule for benchmark {benchmark!r}"


# ---------------------------------------------------------------------------
# Scorer-side items
# ---------------------------------------------------------------------------


class Envs:
    """Environments and their items, loaded once per benchmark and split."""

    def __init__(self, factory: EnvFactory | None = None) -> None:
        self.factory = factory
        self._envs: dict[str, Env] = {}
        self._tasks: dict[tuple[str, str], dict[str, Task]] = {}

    def env(self, benchmark: str) -> Env:
        if benchmark not in self._envs:
            factory = self.factory
            if factory is None:
                from evals.skill_bench.run import default_env

                factory = default_env
            self._envs[benchmark] = factory(benchmark)
        return self._envs[benchmark]

    def tasks(self, benchmark: str, split: str) -> dict[str, Task]:
        key = (benchmark, split)
        if key not in self._tasks:
            loaded = self.env(benchmark).load_tasks(split)  # type: ignore[arg-type]
            self._tasks[key] = {t.raw_id: t for t in loaded}
        return self._tasks[key]


def _answers(task: Task) -> set[str]:
    raw = task.raw
    if task.benchmark == "webshop":
        # The gold product: a test item's ASIN in a method's output that no
        # learning page showed (search results show many) is one it must have read.
        return {str(raw["asin"])} if raw.get("asin") else set()
    return set()


def runs_ledger(
    runs_dir: Path,
    state: Mapping[str, Any],
    gateway_runs_dirs: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """The gateway ledger of an attempt (blocking), found relative to the runs directory
    first (``cell_records.resolve_ledger_path``: a runs directory copied from another host
    reads its own ledgers, never a stale file at a recorded absolute path)."""
    return read_jsonl(
        resolve_ledger_path(
            state, runs_dir=runs_dir, gateway_runs_dirs=gateway_runs_dirs
        )
    )


def _statements(task: Task) -> list[str]:
    """The instruction of an item (WebShop), which methods receive by design
    and which may itself contain a gold answer: it is blanked before the label string
    search. ALFWorld's labels (ids, task types, game files) never occur in a task
    sentence, so nothing is blanked there."""
    if task.benchmark == "webshop":
        # The instruction, which the first message states by design.
        return [
            text
            for key in ("instruction_text", "instruction")
            if isinstance(task.raw.get(key), str)
            for text in (str(task.raw[key]), str(task.raw[key]).strip())
        ]
    return []


def _call_messages(call: Mapping[str, Any], blob_dir: Path) -> list[Any] | None:
    """An actor call's recorded messages, or None when its request cannot be restored
    (reported by the checks that read it)."""
    try:
        request = internalize_blobs(call.get("request") or {}, blob_dir)
    except (OSError, ValueError):
        return None
    return list(request.get("messages") or [])


def _webshop_reply_turn(call: Mapping[str, Any], blob_dir: Path) -> str | None:
    """The assistant message WebShop's conversation keeps for actor call ``call``,
    rebuilt from its recorded response: the reply's text as the
    actor reads it (``actor._content_text``) with U+FFFD in place of each lone surrogate,
    or ``no_reply_turn`` for a call without a reply (the rollout plays the no-op) and for
    an empty or whitespace-only one. None when the response cannot be restored."""
    from evals.skill_bench.agent.actor import _content_text
    from evals.skill_bench.envs.webshop import without_surrogates

    t = _webshop_templates()
    try:
        response = internalize_blobs(call.get("response"), blob_dir)
    except (OSError, ValueError):
        return None
    choices = response.get("choices") if isinstance(response, Mapping) else None
    message = (
        choices[0].get("message")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
        else None
    )
    if call.get("http_status") != 200 or not isinstance(message, Mapping):
        return str(t.no_reply_turn)
    text = _content_text(message.get("content"))
    return without_surrogates(text) if text.strip() else str(t.no_reply_turn)


def _webshop_turn_problem(turn: Any, first: Any, step: int) -> str | None:
    """Why ``turn`` is not WebShop's user message of step ``step`` >= 2:
    ``rollout_turn.md`` up to its page, with that step and the task
    the conversation's first message ``first`` states, and its text after the last
    per-step value; None when it is."""
    t = _webshop_templates()
    head = str(t.turn).partition("{current_observation}")[0]
    before, _, after = head.partition("{task_description}")
    after = after.format(current_step=step)
    tail = str(t.turn).rpartition("}")[2]
    stated = str(t.no_history).partition("{current_observation}")[0]
    if not isinstance(turn, str) or not turn.startswith(before):
        return f"its user message is not rollout_turn.md for step {step}"
    end = turn.find(after, len(before))
    task = turn[len(before) : end]
    if end < 0 or not turn.endswith(tail):
        return f"its user message is not rollout_turn.md for step {step}"
    if stated.format(task_description=task) not in str(first):
        return "its user message states another task than the first message"
    return None


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content if isinstance(p, Mapping)
        )
    return ""


def _some(values: Sequence[str], limit: int = 3) -> str:
    """``a, b, c`` and how many more (finding texts)."""
    shown = ", ".join(values[:limit])
    return shown + (f" and {len(values) - limit} more" if len(values) > limit else "")


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


@dataclass
class _Trajectory:
    phase: str
    raw_id: str
    path: Path
    data: dict[str, Any]
    row: dict[str, Any]
    #: The discarded segment it was moved to by a learning resume (None: live).
    segment: int | None = None

    @property
    def label(self) -> str:
        """Where findings say it was: its phase, and a discarded one's segment."""
        if self.segment is None:
            return self.phase
        return f"{self.phase} (discarded segment {self.segment})"


@dataclass(frozen=True)
class RerunNote:
    """One operator note (``cell.json`` ``operator_notes``) as it covers one earlier run."""

    file: str
    #: When run.py recorded the note (ISO 8601), None in a record it did not write.
    at: Any
    #: The run's test-phase actor calls when the note was recorded.
    test_actor_calls: int

    def order(self) -> tuple[bool, datetime]:
        """Earliest note first; a note without a readable time last."""
        at = _ts(self.at)
        return (at is None, at or datetime.max.replace(tzinfo=timezone.utc))


class CellAuditor:
    """Audits one cell directory (see the module docstring)."""

    def __init__(
        self,
        root: Path,
        *,
        runs_dir: Path,
        envs: Envs,
        frozen: str | None,
        expected_upstream: str,
        blob_dir: Path | None = None,
        include_incomplete: bool = False,
        gateway_runs_dirs: Sequence[Path] = (),
        versions: Mapping[str, Mapping[str, Any]] | None = None,
        tree: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.paths = CellPaths(root)
        self.runs_dir = runs_dir
        #: Other gateway runs directories to find ledgers in (``--gateway-runs-dir``).
        self.gateway_runs_dirs = tuple(Path(d) for d in gateway_runs_dirs)
        self.envs = envs
        self.frozen = frozen
        #: The protocol versions a cell may carry (``protocol.known_versions``; the frozen
        #: value among them): another one is an earlier version, a warning.
        self.versions: Mapping[str, Mapping[str, Any]] = (
            versions
            if versions is not None
            else ({frozen: {"fingerprint": frozen, "current": True}} if frozen else {})
        )
        #: The audited attempt's protocol fingerprint (``_check_protocol``).
        self.attempt_fingerprint: str | None = None
        #: The auditing tree as a version (:func:`_tree_version`), built lazily when the
        #: caller passes none: what the checks of :data:`TREE_INPUTS` read.
        self._tree: Mapping[str, Any] | None = tree
        #: The attempt's version segments (``protocol.attempt_versions``; index 0 is its
        #: start version) and every protocol fingerprint it ran under, start first.
        self.segments: list[dict[str, Any]] = []
        self.attempt_versions: list[str] = []
        #: Versions registered only in the cell's own records (a segment's
        #: ``version.json``, or ``attempt.json`` ``protocol`` when it describes itself):
        #: an auto-freeze on another host (``protocol.recorded_version``).
        self._carried: dict[str, dict[str, Any]] = {}
        #: models.json sha256 -> model id -> canonical_slug the cell's own records keep
        #: (each segment's ``version.json``, ``_carried_versions``).
        self._cell_model_slugs: dict[str, dict[str, str]] = {}
        #: The gateway start records of the attempt (``_gateway_starts``), by start id.
        self.starts: list[dict[str, Any]] = []
        self._starts_by_id: dict[str, dict[str, Any]] = {}
        #: The attempt's model settings (``_attempt_settings``) and where they come from.
        self._allowed_settings: list[dict[str, Any]] = []
        #: The auditing machine lacks the cell's own items or its environment: the checks
        #: that need them are not checked, never violations.
        self._items_missing = False
        #: Discarded trajectories that could not be read: ``(segment, path, error)``.
        self._unreadable_discarded: list[tuple[int, str, str]] = []
        #: Discarded segments the resume check found (and those it found not restored).
        self._resumed_segments: set[Any] = set()
        self._restore_failed: set[Any] = set()
        self.expected_upstream = expected_upstream
        self.blob_dir_override = blob_dir
        self.include_incomplete = include_incomplete
        self.manifest: dict[str, Any] = {}
        self.cell: dict[str, Any] = {}
        self.f = Findings(root.name)
        self.info: dict[str, Any] = {}
        # Set by run() once an attempt is chosen.
        self.complete = False
        self.number = 0
        self.state: dict[str, Any] = {}
        self.attempt_paths = self.paths.attempt(0)
        self.run_id = ""
        self.method = self.benchmark = self.model = ""
        self.ledger_path = runs_dir / "gateway.jsonl"
        self.ledger: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.by_call_id: dict[str, dict[str, Any]] = {}
        self.rerun_uids: set[str] = set()
        self.frozen_record: Any = None
        #: ``(ts, phase)`` of every phase the run was put in (the ledger's ``phase`` events).
        self.phase_events: list[tuple[datetime, str]] = []
        self.learning_uids: set[str] = set()
        #: The raw id of each learning uid (recomputed from the cell's salt).
        self.learning_raw: dict[str, str] = {}
        #: The harness code versions the attempt's invocations recorded (``_check_code``).
        self.harness_code_versions: set[str] = set()
        self.trajectories: list[_Trajectory] = []
        #: Trajectories a learning resume moved to ``discarded/seg-SS/`` (H10.1: what the
        #: method received before the rollback).
        self.discarded_trajectories: list[_Trajectory] = []
        #: The calls learning resumes discarded, as the ledger's markers delimit them
        #: (``_check_resumes``): call id -> segment.
        self.discarded: dict[str, int] = {}
        self.resume_rows: list[dict[str, Any]] = []
        #: How many calls the provider policy covered, and how many an ignored provider
        #: served anyway (``info.provider_policy``).
        self._policy_counts: Counter[str] = Counter(
            {"policy_calls": 0, "served_by_ignored": 0, "refused_ignored": 0}
        )
        self.tasks: dict[str, Task] = {}
        self.all_tasks: dict[str, Task] = {}

    # -- entry --------------------------------------------------------------------------

    def run(self) -> CellAudit:
        self.manifest = read_json(self.paths.manifest) or {}
        self.cell = dict(self.manifest.get("cell") or {})
        status = read_json(self.paths.status) or {}
        cell_id = str(self.cell.get("cell_id") or self.root.name)
        self.f.cell_id = cell_id
        audit = CellAudit(
            cell_id=cell_id,
            root=str(self.root),
            benchmark=self.cell.get("benchmark"),
            model=self.cell.get("model"),
            method=self.cell.get("method"),
            split_mode=self.cell.get("split_mode"),
            status=status.get("status") if isinstance(status, dict) else None,
            attempt=None,
            audited=False,
        )
        numbers = self.paths.attempt_numbers()
        states = {n: read_json(self.paths.attempt(n).state) or {} for n in numbers}
        complete = [n for n in numbers if states[n].get("state") == ATTEMPT_COMPLETE]
        is_complete = bool(complete) and audit.status == STATUS_COMPLETE
        if is_complete:
            number: int | None = complete[-1]
        elif self.include_incomplete:
            live = [n for n in numbers if states[n].get("state") != "invalid"]
            number = live[-1] if live else None
        else:
            number = None
        audit.attempt = number
        if number is None:
            audit.reason = (
                f"status {audit.status!r}: only complete cells are audited "
                "(--include-incomplete audits the current attempt)"
                if numbers
                else "no attempt"
            )
            return audit
        self.complete = is_complete
        self.number = number
        self.state = states[number]
        self.attempt_paths = self.paths.attempt(number)
        self.run_id = str(self.state.get("run_id"))
        self.method = str(self.cell.get("method"))
        self.benchmark = str(self.cell.get("benchmark"))
        self.model = str(self.cell.get("model"))
        self.ledger_path = self._ledger_path()
        self.ledger, torn = read_jsonl_counted(self.ledger_path)
        if torn:
            self.f.warn(
                "H10.3",
                "torn_lines",
                f"{self.ledger_path.name} has {torn} torn line(s) (a crash mid-write): "
                "the calls they recorded are not counted",
            )
        self.phase_events = sorted(
            (stamp, str(r.get("phase")))
            for r in self.ledger
            if r.get("event") == "phase" and (stamp := _ts(r.get("ts"))) is not None
        )
        self.calls = [r for r in self.ledger if r.get("event") == "call"]
        self.by_call_id = {str(r.get("call_id")): r for r in self.calls}
        self.rerun_uids = {
            str(r.get("task_uid")) for r in self.state.get("test_reruns") or ()
        }
        self.frozen_record = read_json(self.attempt_paths.frozen_snapshot)
        salt = self._salt_quiet()
        if salt is not None:
            self.learning_raw = {
                opaque_task_uid(salt, str(raw_id)): str(raw_id)
                for raw_id in self.cell.get("learning_items") or ()
            }
            self.learning_uids = set(self.learning_raw)

        self._check_protocol()
        self._check_hosts()
        self._check_attempts(numbers, states)
        self._check_other_runs(states)
        self._load_items()
        self._load_trajectories()
        self._check_resumes()
        self._report_unreadable_discarded()
        self._check_labels()
        self._check_prompts()
        self._check_request_hashes()
        self._check_gateway()
        self._check_counted_failures()
        self._check_frozen_memory()
        self._check_results()
        self._report_not_reverified()

        audit.audited = True
        audit.findings = list(self.f.items)
        audit.checked = dict(self.f.checked)
        audit.violations = self.f.count(VIOLATION)
        audit.warnings = self.f.count(WARNING)
        audit.not_checked = self.f.count(NOT_CHECKED)
        audit.violations_by_invariant = self.f.violations_by_invariant()
        audit.suppressed = {f"{i}:{c}": n for (i, c), n in self.f.suppressed.items()}
        # Every finding the gate recorded instead of failing on, per check, uncapped.
        self.info["recorded_counts"] = {
            key: n
            for (key, why), n in sorted(self.f.downgraded.items())
            if why == RECORD
        }
        audit.info = self.info
        return audit

    # -- loading ------------------------------------------------------------------------

    def _ledger_path(self) -> Path:
        """Relative to the runs directory first (a copy from another host reads its own
        ledger); a recorded absolute path only when nothing relative exists and its
        directory is the run's (``cell_records.resolve_ledger_path``)."""
        return resolve_ledger_path(
            self.state,
            runs_dir=self.runs_dir,
            gateway_runs_dirs=self.gateway_runs_dirs,
        )

    @property
    def blob_dir(self) -> Path:
        if self.blob_dir_override is not None:
            return self.blob_dir_override
        return resolve_blob_dir(self.manifest, self.runs_dir)

    def _load_items(self) -> None:
        """The cell's own items (its learning and test splits), and every item of the
        benchmark, whose identifiers no method may hold either. Own items the auditing
        machine cannot load leave the checks that need them not checked (exit 2), never
        passed and never violations (the audit fails closed)."""
        own = {self.cell.get("learning_split"), self.cell.get("test_split")} - {None}
        # The benchmark's extra splits too (WebShop's pilot, protocol.EXTRA_SPLIT_SIZES).
        extra = tuple(protocol.EXTRA_SPLIT_SIZES.get(str(self.benchmark), {}))
        for split in ("train", "val", "test", *extra):
            try:
                tasks = self.envs.tasks(self.benchmark, split)
            except Exception as exc:  # noqa: BLE001 - reported as a finding
                if split in own:
                    self._items_missing = True
                    self.f.not_checked(
                        "H10.1",
                        "items_unavailable",
                        f"this machine cannot load the scorer-side {split} items, so "
                        f"the label checks were not run: {type(exc).__name__}: {exc}",
                    )
                else:
                    self.f.warn(
                        "H10.1",
                        "items_unavailable",
                        f"the {split} items are not available, so their identifiers "
                        f"are not searched for: {type(exc).__name__}: {exc}",
                    )
                continue
            self.all_tasks.update(tasks)
            if split in own:
                self.tasks.update(tasks)

    def _load_trajectories(self) -> None:
        rows = [
            *(("learning", r) for r in read_jsonl(self.attempt_paths.learning)),
            *(("test", r) for r in read_jsonl(self.attempt_paths.results)),
        ]
        for phase, row in rows:
            rel = row.get("trajectory")
            raw_id = str(row.get("raw_id"))
            if not rel:
                self.f.add(
                    "H10.3",
                    "trajectory_missing",
                    f"a {phase} result row names no trajectory",
                    item=raw_id,
                )
                continue
            path = self.root / str(rel)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.f.add(
                    "H10.3",
                    "trajectory_missing",
                    f"{rel} cannot be read ({type(exc).__name__}); nothing vouches "
                    "for this item",
                    item=raw_id,
                )
                continue
            self.trajectories.append(_Trajectory(phase, raw_id, path, data, row))
        self.info["trajectories"] = len(self.trajectories)
        self._load_discarded_trajectories()

    def _load_discarded_trajectories(self) -> None:
        """Every trajectory a learning resume moved to ``discarded/seg-SS/trajectories/``
        (failed tries included): what the method received before the rollback, scanned
        like a live one (H10.1). A segment still ``.partial`` (a resume that
        died mid-move) is read too."""
        root = self.attempt_paths.root / "discarded"
        if not root.is_dir():
            return
        for segment_dir in sorted(root.glob("seg-*")):
            match = re.fullmatch(r"seg-(\d+)(?:\.partial)?", segment_dir.name)
            if match is None or not segment_dir.is_dir():
                continue
            segment = int(match.group(1))
            for path in sorted((segment_dir / "trajectories").rglob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    # Judged once the resumes are checked (_report_unreadable_discarded).
                    self._unreadable_discarded.append(
                        (
                            segment,
                            path.relative_to(self.root).as_posix(),
                            type(exc).__name__,
                        )
                    )
                    continue
                if not isinstance(data, dict):
                    continue
                self.discarded_trajectories.append(
                    _Trajectory(
                        str(data.get("phase") or "learning"),
                        str(data.get("raw_id")),
                        path,
                        data,
                        {},
                        segment,
                    )
                )
        self.info["discarded_trajectories"] = len(self.discarded_trajectories)

    # -- the run phase -----------------------------------------------------------------

    def _phase_at(self, stamp: datetime) -> str | None:
        """The phase run.py had put the run in at ``stamp`` (the ledger's phase events)."""
        phase = None
        for at, value in self.phase_events:
            if at > stamp:
                break
            phase = value
        return phase

    def _phase(self, row: Mapping[str, Any]) -> str | None:
        """The phase a call belongs to: the run's phase the gateway recorded with it
        (``run_phase``), else the run's phase in force at its time, else its own tag. A
        caller's ``x-skill-bench-phase`` header never decides."""
        value = row.get("run_phase")
        if value:
            return str(value)
        stamp = _ts(row.get("ts"))
        if stamp is not None:
            phase = self._phase_at(stamp)
            if phase is not None:
                return phase
        tagged = row.get("phase")
        return str(tagged) if tagged is not None else None

    # -- attempts (H8, H1.3) ------------------------------------------------------------

    def _check_attempts(
        self, numbers: Sequence[int], states: Mapping[int, Mapping[str, Any]]
    ) -> None:
        """Every attempt of the cell, not only the audited one: a tested attempt is never
        replaced, each test item is scored once, and a re-run replaces only an unscored
        row (H8: only failed test items are re-run; H1.3: every cell runs once)."""
        f = self.f
        instance = self.manifest.get("instance")
        cell_id = str(self.cell.get("cell_id") or self.root.name)
        superseded_attempts: list[dict[str, Any]] = []
        self.info["superseded_attempts"] = superseded_attempts
        for number in numbers:
            state = states[number]
            f.ok("H10.3")
            run_id = str(state.get("run_id"))
            # An attempt started before instance ids existed ran as <cell>-aNN; every
            # later one records the cell directory's instance id and runs as
            # <cell>-aNN-<id>.
            recorded = state.get("instance")
            if recorded and instance and recorded != instance:
                f.add(
                    "H10.3",
                    "run_id",
                    f"attempt {number} records instance {recorded!r}, the cell "
                    f"directory is instance {instance!r}: it belongs to another cell",
                )
            expected = f"{cell_id}-a{number:02d}" + (f"-{recorded}" if recorded else "")
            if run_id != expected:
                f.add(
                    "H10.3",
                    "run_id",
                    f"attempt {number} ran as gateway run {run_id!r}, expected "
                    f"{expected!r}: its ledger may hold another cell's calls",
                )
            marker = _superseded_marker(state)
            if number == self.number or (
                state.get("state") != "invalid" and marker is None
            ):
                continue
            paths = self.paths.attempt(number)
            scored = {
                str(r.get("raw_id"))
                for r in read_jsonl(paths.results)
                if r.get("score") is not None
            }
            ledger = runs_ledger(self.runs_dir, state, self.gateway_runs_dirs)
            tested = [
                r
                for r in ledger
                if r.get("event") == "call"
                and r.get("role") == "actor"
                and (r.get("run_phase") or r.get("phase")) == "test"
            ]
            superseded_attempts.append(
                {
                    "attempt": number,
                    "run_id": state.get("run_id"),
                    "state": state.get("state"),
                    "previous_state": state.get("previous_state"),
                    "invalidation": state.get("invalidation"),
                    "superseded": marker is not None,
                    "scored": len(scored),
                    "test_actor_calls": len(tested),
                    "note": (marker or {}).get("note"),
                    "at": (marker or {}).get("at"),
                    "by_attempt": (marker or {}).get("by_attempt"),
                }
            )
            if marker is not None:
                # A tested cell run again with --new-attempt --note keeps the old
                # attempt, marked superseded (H1.3 as recorded).
                by = marker.get("by_attempt")
                f.warn(
                    "H10.3",
                    "attempt_superseded",
                    f"attempt {number} ({len(scored)} scored item(s), {len(tested)} "
                    "test-phase actor call(s)) was superseded by attempt "
                    f"{by if by is not None else '(not yet recorded)'} at "
                    f"{marker.get('at')}: {str(marker.get('note'))[:200]}. Its records "
                    "stay; the report reads the latest complete attempt and lists every "
                    "attempt's scores, never mixing them (H1.3 as recorded)",
                )
            elif scored or tested:
                f.add(
                    "H10.3",
                    "tested_attempt_replaced",
                    f"attempt {number} was invalidated ({state.get('invalidation') or 'H8'}: "
                    f"{str(state.get('invalid_reason'))[:200]}) after its test phase had "
                    f"run ({len(scored)} scored item(s), {len(tested)} test-phase actor "
                    "call(s)): the cell ran more than once (H1.3, H8)",
                )
            elif state.get("invalidation") == "operator_reroll":
                f.warn(
                    "H10.3",
                    "operator_reroll",
                    f"attempt {number} was learned again at the operator's request "
                    f"(--new-attempt) before any test item was scored: "
                    f"{str(state.get('invalid_reason'))[:200]}",
                )
            if state.get("violation") or state.get("previous_state") == "violation":
                f.warn(
                    "H10.4",
                    "violation_history",
                    f"attempt {number} was in violation before it was replaced: "
                    f"{(state.get('violation') or {}).get('detail')}",
                )
        rows = read_jsonl(self.attempt_paths.results)
        by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_item[str(row.get("raw_id"))].append(row)
        reruns = {str(r.get("raw_id")) for r in self.state.get("test_reruns") or ()}
        for raw_id, item_rows in by_item.items():
            f.ok("H10.3")
            scored = [i for i, r in enumerate(item_rows) if r.get("score") is not None]
            if len(scored) > 1:
                f.add(
                    "H10.3",
                    "scored_twice",
                    f"test item {raw_id} has {len(scored)} scored rows: an item is "
                    "scored once; only an unscored item is re-run (H8)",
                    item=raw_id,
                )
            before = item_rows[: scored[0]] if scored else item_rows
            bad = [
                r
                for r in before
                if (r.get("failure") or {}).get("kind") not in ("infra", "harness")
            ]
            if bad:
                f.add(
                    "H10.3",
                    "rerun_without_failure",
                    f"test item {raw_id} has a row before its scored one that records "
                    "no infrastructure or harness failure",
                    item=raw_id,
                )
            if before and scored and raw_id not in reruns:
                f.add(
                    "H10.3",
                    "rerun_unlisted",
                    f"test item {raw_id} was re-run but attempt.json test_reruns does "
                    "not list it",
                    item=raw_id,
                )

    # -- the cell ran once, across cell directories (H1.3) ------------------------------

    def _host_mirrors(self) -> list[Path]:
        """``<runs_dir>/_hosts/*``: each host's mirrored runs directory in a merged view
        of several hosts (``tools/sync_runs.py``)."""
        hosts = self.runs_dir / HOSTS_DIR
        try:
            return sorted(p for p in hosts.iterdir() if p.is_dir())
        except OSError:
            return []

    def _gateway_runs_dirs(self, states: Mapping[int, Mapping[str, Any]]) -> list[Path]:
        """The gateway runs directories that served this cell: where its ledgers are
        (through a merged view's links too), the runs directory the audit reads, each
        host mirror of a merged view, ``--gateway-runs-dir``, and a recorded ledger's
        directory when it is the run's own. One entry per real directory."""
        candidates = [
            self.ledger_path.parent.parent,
            self.ledger_path.resolve().parent.parent,
            self.runs_dir,
            *self.gateway_runs_dirs,
            *self._host_mirrors(),
        ]
        for state in states.values():
            recorded = state.get("ledger")
            if recorded and Path(str(recorded)).parent.name == str(state.get("run_id")):
                candidates.append(Path(str(recorded)).parent.parent)
        seen: set[Path] = set()
        dirs: list[Path] = []
        for path in candidates:
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key not in seen and path.is_dir():
                seen.add(key)
                dirs.append(path)
        return dirs

    def _acknowledged_runs(self) -> dict[str, list[RerunNote]]:
        """Earlier gateway runs the operator notes in ``cell.json`` acknowledge (run.py
        ``--acknowledge-rerun``), each with every note that covers it (the note's time,
        file and the run's test-phase actor calls then). A note whose copy in the cell is
        missing or changed acknowledges nothing (a warning)."""
        acknowledged: dict[str, list[RerunNote]] = defaultdict(list)
        for note in self.manifest.get("operator_notes") or ():
            if not isinstance(note, Mapping) or note.get("kind") != (
                NOTE_RERUN_ACKNOWLEDGED
            ):
                continue
            rel = str(note.get("file") or "")
            path = (self.root / rel).resolve()
            try:
                inside = path.is_relative_to(self.root.resolve())
                data = path.read_bytes() if inside and rel else None
            except OSError:
                data = None
            if data is None or hashlib.sha256(data).hexdigest() != note.get("sha256"):
                self.f.warn(
                    "H10.3",
                    "operator_note_invalid",
                    f"the operator note {rel or '?'} recorded in cell.json is missing, "
                    "outside the cell directory or changed since it was recorded: it "
                    "acknowledges nothing",
                )
                continue
            # run.py records a late or unmatched note too (H1.3 recorded, not refused):
            # a warning says what it cannot vouch for.
            if note.get("after_test") is True:
                self.f.warn(
                    "H10.3",
                    "operator_note_after_test",
                    f"the operator note {rel} was recorded after this cell directory "
                    "had run test items "
                    + (
                        f"({', '.join(str(r) for r in note.get('tested_runs') or ())})"
                        if note.get("tested_runs")
                        else ""
                    )
                    + ": it records the re-run, it cannot show that the re-run was "
                    "sanctioned before any test score was seen",
                )
            if note.get("superseded") == []:
                self.f.warn(
                    "H10.3",
                    "operator_note_without_runs",
                    f"the operator note {rel} names no earlier run of this cell in the "
                    "gateway runs directory run.py saw (a run on another host?): it "
                    "acknowledges nothing here",
                )
            for run in note.get("superseded") or ():
                if isinstance(run, Mapping) and run.get("run_id"):
                    try:
                        calls = int(run.get("test_actor_calls") or 0)
                    except (TypeError, ValueError):
                        calls = 0  # a record run.py did not write covers nothing
                    acknowledged[str(run["run_id"])].append(
                        RerunNote(rel, note.get("at"), calls)
                    )
        return acknowledged

    def _check_other_runs(self, states: Mapping[int, Mapping[str, Any]]) -> None:
        """H1.3 across cell directories. ``--new-attempt`` cannot
        replace a tested attempt, but a cell directory can be moved away and the cell
        started over in a new one: that one gets a new instance id and a clean record,
        while the old runs stay in the gateway runs directory. Every run of this cell id
        there is compared with this directory's attempts:

        * a run of another instance (or an unowned run from before instance ids) that ran
          test items: the cell was run again after its test scores could be seen
          (recorded, ``cell_rerun_after_test``; ``cell_rerun_acknowledged`` when an
          operator note in ``cell.json`` acknowledges it, still listed). The note must be
          older than this directory's first test-phase actor call: a note recorded once
          both runs' scores could be seen cannot tell a sanctioned re-run from keeping the
          better run (recorded, ``rerun_acknowledged_after_test``);
        * a smoke cell's earlier run that ran test items (a smoke cell tests val items):
          a warning (``smoke_cell_rerun``). H1.3 runs each protocol cell once; a smoke
          cell is outside that matrix and may be run again whenever it is needed;
        * a run of another instance that ran no test item: the cell was started over
          before its test phase, like an operator re-roll (a warning);
        * a run of this directory's own instance that no ``attempt.json`` records and that
          ran test items: an attempt's records were removed (recorded,
          ``unrecorded_attempt``). One that ran nothing is an attempt that never
          started.
        """
        f = self.f
        cell_id = str(self.cell.get("cell_id") or self.root.name)
        instance = self.manifest.get("instance")
        smoke = self.cell.get("split_mode") == "smoke"
        own = {str(s["run_id"]) for s in states.values() if s.get("run_id")}
        runs: dict[str, GatewayRun] = {}
        for directory in self._gateway_runs_dirs(states):
            for run in cell_gateway_runs(directory, cell_id):
                runs.setdefault(run.run_id, run)
        acknowledged = self._acknowledged_runs()
        others = [r for r_id, r in sorted(runs.items()) if r_id not in own]
        self.info["other_gateway_runs"] = [r.to_dict() for r in others]
        # When this directory's own test scores could first be seen.
        tested_at = sorted(
            (stamp, str(r.first_test_actor_at))
            for r in runs.values()
            if r.run_id in own and (stamp := _ts(r.first_test_actor_at)) is not None
        )
        own_first = tested_at[0] if tested_at else None
        f.ok("H10.3")
        for run in others:
            f.ok("H10.3")
            seen = (
                f"gateway run {run.run_id} ({run.test_actor_calls} test-phase actor "
                f"call(s), {run.calls} call(s) in all)"
            )
            if instance and run.instance == instance:
                if run.tested:
                    f.add(
                        "H10.3",
                        "unrecorded_attempt",
                        f"{seen} belongs to this cell directory (instance {instance}) "
                        "and ran test items, but no attempt.json records it: an "
                        "attempt's records were removed (H1.3, H8)",
                    )
                continue
            if not run.tested:
                reached = " and reached its test phase" if run.test_phase else ""
                f.warn(
                    "H10.3",
                    "cell_restarted",
                    f"{seen} is an earlier run of this cell from another cell directory "
                    f"(instance {run.instance}){reached}; it ran no test item, so the "
                    "cell was started over before any test score could be seen",
                )
                continue
            earlier = (
                f"{seen} is an earlier run of this cell from another cell directory "
                f"(instance {run.instance}) that ran test items"
            )
            if smoke:
                f.warn(
                    "H10.3",
                    "smoke_cell_rerun",
                    f"{earlier}. A smoke cell (it tests val items) is outside the protocol "
                    "matrix H1.3 runs once, so running it again is allowed; listed for "
                    "the record",
                )
                continue
            notes = acknowledged.get(run.run_id, [])
            covering = [n for n in notes if run.test_actor_calls <= n.test_actor_calls]
            if covering:
                note = min(covering, key=RerunNote.order)
                at = _ts(note.at)
                if own_first is not None and (at is None or at >= own_first[0]):
                    f.add(
                        "H10.3",
                        "rerun_acknowledged_after_test",
                        f"{earlier}; the operator note {note.file} that acknowledges "
                        f"it was recorded at {note.at or 'an unknown time'}, not before "
                        f"this cell directory's first test-phase actor call "
                        f"({own_first[1]}): by then the test scores of both runs could "
                        "be seen, so the note cannot show that the re-run was "
                        "sanctioned rather than the better run kept (H1.3). A re-run is "
                        "acknowledged before the new directory's test phase",
                    )
                    continue
                f.warn(
                    "H10.3",
                    "cell_rerun_acknowledged",
                    f"{earlier}; an operator note in cell.json ({note.file}, "
                    f"{note.at}) acknowledges that the cell was run again",
                )
                continue
            covered = max((n.test_actor_calls for n in notes), default=None)
            later = (
                f"; the operator note covers only {covered} of its test-phase actor "
                "calls (it ran more test items after the note)"
                if covered is not None
                else ""
            )
            f.add(
                "H10.3",
                "cell_rerun_after_test",
                f"{earlier}: the cell was run again from scratch after its test scores "
                "could be seen (H1.3: every cell runs once)"
                + (
                    later
                    or ". Record an operator note with run.py --acknowledge-rerun, "
                    "before the new directory's test phase, if that was sanctioned"
                ),
            )

    # -- H10.6 ----------------------------------------------------------------------------

    def _protocol_known(self, fingerprint: Any) -> bool:
        """A fingerprint some freeze produced: the frozen value or an earlier one."""
        return bool(fingerprint) and (
            fingerprint == self.frozen or fingerprint in self.versions
        )

    def _version_known(self, fingerprint: Any) -> bool:
        """Registered in the frozen record or in the cell's own records (a version
        auto-frozen on another host, ``_carried``)."""
        return self._protocol_known(fingerprint) or fingerprint in self._carried

    @property
    def tree(self) -> Mapping[str, Any]:
        """The auditing tree as a version (:func:`_tree_version`)."""
        if self._tree is None:
            self._tree = _tree_version(self.frozen)
        return self._tree

    def _replaced_by(self, fingerprint: str) -> Mapping[str, Any] | None:
        """The known version frozen next after ``fingerprint`` (by ``frozen_at``): the
        freeze that replaced it, whose reason says why (the version's own reason says why
        it was frozen). None when either time is not recorded."""
        own = _ts((self.versions.get(fingerprint) or {}).get("frozen_at"))
        if own is None:
            return None
        later = [
            (at, version)
            for version in self.versions.values()
            if (at := _ts(version.get("frozen_at"))) is not None and at > own
        ]
        return min(later, key=lambda pair: pair[0])[1] if later else None

    def _carried_versions(self) -> dict[str, dict[str, Any]]:
        """Versions of the attempt that ``config/protocol_frozen.json`` of this checkout
        does not register but the cell's own records describe: each segment's
        ``version.json`` (segment 0 describes the attempt's start) and ``attempt.json``
        ``protocol`` when it keeps its constants and file map. Only a record whose per-key
        constants and file map reproduce its fingerprint counts
        (``protocol.recorded_version``); a ``version_file`` outside the cell directory is
        never read."""
        sources: list[tuple[str, Any]] = [
            ("attempt.json protocol", self.state.get("protocol"))
        ]
        root = self.root.resolve()
        for seg in self.segments:
            rel = seg.get("version_file")
            if not isinstance(rel, str) or not rel:
                continue
            path = self.root / rel
            try:
                if not path.resolve().is_relative_to(root):
                    continue
                sources.append((rel, read_json(path)))
            except (OSError, ValueError):
                continue
        carried: dict[str, dict[str, Any]] = {}
        wanted = set(self.attempt_versions)
        for source, document in sources:
            if not isinstance(document, Mapping):
                continue
            for sha, slugs in protocol.recorded_model_slugs(document).items():
                self._cell_model_slugs.setdefault(sha, slugs)
            version = protocol.recorded_version(document)
            if version is None:
                continue
            fingerprint = str(version.get("fingerprint"))
            if fingerprint not in wanted or self._protocol_known(fingerprint):
                continue
            frozen = document.get("frozen")
            frozen = frozen if isinstance(frozen, Mapping) else {}
            carried.setdefault(
                fingerprint,
                {
                    **version,
                    "frozen_at": version.get("frozen_at") or frozen.get("frozen_at"),
                    "reason": version.get("reason") or frozen.get("reason"),
                    "source": source,
                },
            )
        return carried

    def _check_protocol(self) -> None:  # noqa: C901 - one pass over the versions
        """H10.6 (see the module docstring). The audited attempt's protocol
        (``protocol.attempt_protocol``, the version it started under) and every version
        it ran under since (``protocol.attempt_versions``: its ``version_segments``) are
        the frozen value, known earlier ones, or versions only the cell's own records
        describe (warnings: versions are recorded and reported).
        Before any other check, which checks that read this tree are not re-verified for
        them (:data:`TREE_INPUTS`). Then every call against its own gateway start
        record, the code and protocol of the attempt's invocations, and its gateways."""
        f = self.f
        recorded = protocol.attempt_protocol(self.manifest, self.state, self.number)
        fingerprint = recorded["fingerprint"]
        self.attempt_fingerprint = fingerprint
        launches = read_jsonl(self.paths.launches)
        self.segments = [
            dict(seg)
            for seg in protocol.attempt_versions(
                self.manifest, self.state, self.number, launches
            )
            if isinstance(seg, Mapping)
        ]
        self.attempt_versions = list(
            dict.fromkeys(
                str(fp)
                for fp in (fingerprint, *map(_segment_fingerprint, self.segments))
                if fp
            )
        )
        self._carried = self._carried_versions()
        # The tree's own entry describes its fingerprint exactly (its constants and files
        # give it); the recorded versions and the cell's own records describe the rest.
        self.diff_versions = diff_versions(self.versions, self._carried, self.tree)
        current = self.frozen is not None and fingerprint == self.frozen
        self.info["protocol"] = {
            "fingerprint": fingerprint,
            "current": current,
            "known": self._protocol_known(fingerprint),
            "source": recorded["source"],
        }
        self.info["protocol_record"] = (
            "frozen record"
            if self._protocol_known(fingerprint)
            else "attempt record"
            if fingerprint in self._carried
            else None
        )
        self.info["commit"] = (self.state.get("git") or {}).get("commit")
        # Before any finding a changed input could downgrade.
        self.f.not_reverified = self._not_reverified()
        self.info["prompt_key"] = self._prompt_key()
        # The one protocol the attempt ran under, if one: cells of one
        # fingerprint share every template, split and the seed, so they share every
        # item's no-memory first prompt even when the version's file map is not
        # recorded and no prompt key can be computed.
        self.info["prompt_fingerprint"] = (
            self.attempt_versions[0] if len(self.attempt_versions) == 1 else None
        )
        if self.frozen is None:
            f.add(
                "H10.6", "not_frozen", "no frozen protocol fingerprint to compare with"
            )
        elif current:
            pass
        elif self._protocol_known(fingerprint):
            version = self.versions.get(str(fingerprint)) or {}
            diff = protocol.version_diff(fingerprint, self.frozen, self.versions)
            replaced = self._replaced_by(str(fingerprint))
            f.warn(
                "H10.6",
                "protocol_version",
                f"attempt {self.number} ran under protocol {str(fingerprint)[:12]}, an "
                f"earlier version than the frozen {self.frozen[:12]} (it was frozen "
                f"{version.get('frozen_at')}: {version.get('reason')}"
                + (
                    f"; replaced by the freeze of {replaced.get('frozen_at')}: "
                    f"{replaced.get('reason')}"
                    if replaced
                    else ""
                )
                + f"); since then: {protocol.describe_version_diff(diff)}. Versions "
                "are reported per cell, not enforced across cells",
            )
        elif fingerprint in self._carried:
            version = self._carried[str(fingerprint)]
            diff = protocol.version_diff(fingerprint, self.frozen, self.diff_versions)
            frozen_text = (
                f", frozen {version.get('frozen_at')}: {version.get('reason')}"
                if version.get("frozen_at") or version.get("reason")
                else ""
            )
            f.warn(
                "H10.6",
                "protocol_version",
                f"attempt {self.number} ran under protocol {str(fingerprint)[:12]}, "
                f"registered only in its own attempt record ({version.get('source')}"
                f"{frozen_text}), not in config/protocol_frozen.json of this checkout "
                f"(an auto-freeze on another host?); since then: "
                f"{protocol.describe_version_diff(diff)}. Versions are reported per "
                "cell, not enforced across cells",
            )
        else:
            f.add(
                "H10.6",
                "fingerprint",
                f"the cell ran under protocol {fingerprint}, which no freeze of "
                f"config/protocol_frozen.json produced (the frozen one is {self.frozen})"
                " and no record of the cell describes (a version not registered: the "
                "checks that read this tree are listed as not re-verified)",
            )
        f.ok("H10.6")
        # Every call against the gateway that served it (H10.6-gateway-settings): its
        # start record, else the attempt's own version.
        self.starts = self._gateway_starts()
        self._starts_by_id = {
            str(s["start_id"]): s for s in self.starts if s.get("start_id")
        }
        allowed, source = self._attempt_settings()
        self._allowed_settings = allowed
        if not allowed and any(self._start_for(row) is None for row in self.calls):
            # Nothing of the attempt says which settings its calls were made under: the
            # calls without a start record are compared with this tree's settings, and
            # only their findings are not re-verified (a call its start record covers
            # is judged against that record as usual).
            for key in UNRECORDED_SETTINGS_CHECKS:
                inputs = self.f.partly_not_reverified.setdefault(key, [])
                if SETTINGS_NOT_RECORDED not in inputs:
                    inputs.append(SETTINGS_NOT_RECORDED)
        for row in self.calls:
            start = self._start_for(row)
            expected = self._expected_settings(start)
            # Judged against this tree's settings, which nothing of the attempt vouches
            # for: its findings are not re-verified.
            unverified = start is None and not allowed
            if row.get("protocol_conformant") is not True:
                f.add(
                    "H10.6",
                    "nonconformant_call",
                    f"call {row.get('call_id')} ({row.get('role')}) was served by a "
                    "gateway without the protocol settings",
                )
            elif (
                start is not None
                and start.get("settings_sha256")
                and row.get("gateway_settings_sha256") != start.get("settings_sha256")
            ):
                f.add(
                    "H10.6",
                    "gateway_settings",
                    f"call {row.get('call_id')} was served with gateway settings "
                    f"{row.get('gateway_settings_sha256')}: the line's settings hash is "
                    f"not its gateway start record's ({start.get('settings_sha256')})",
                )
            if row.get("status") == "ok" and row.get("role") in CHAT_ROLES:
                wire = row.get("wire") or {}
                effort = expected.get("reasoning_effort")
                if wire.get("reasoning") != {"effort": effort}:
                    f.add(
                        "H10.6",
                        "reasoning_effort",
                        f"call {row.get('call_id')} went out with reasoning "
                        f"{wire.get('reasoning')!r}; its gateway start record says "
                        f"effort {effort!r} (H1.9)",
                        unverified=unverified,
                    )
                if "seed" in wire and wire.get("seed") != expected.get("seed"):
                    f.add(
                        "H10.6",
                        "seed",
                        f"call {row.get('call_id')} went out with seed {wire['seed']}; "
                        f"its gateway start record says {expected.get('seed')!r}",
                        unverified=unverified,
                    )
            self._check_provider_policy(
                row, expected.get("provider_policy"), unverified=unverified
            )
            f.ok("H10.6")
        self.info["provider_policy"] = {
            "policy": allowed[0].get("provider_policy")
            if allowed
            else protocol.provider_policy_settings(),
            "source": source,
            **self._policy_counts,
        }
        options = self.manifest.get("options") or {}
        if options.get("allow_nonconformant_gateway"):
            f.warn(
                "H10.6",
                "nonconformant_allowed",
                "the cell's last invocation accepted a non-conformant gateway",
            )
        started = _ts(self.state.get("started_at"))
        commits = {(self.state.get("git") or {}).get("commit")}
        code: dict[str, set[str | None]] = defaultdict(set)
        for key in ("harness", "mirix"):
            code[key].add((self.state.get("code") or {}).get(key))
        for launch in launches:
            if not self._worked_on_attempt(launch, started):
                continue
            commits.add((launch.get("git") or {}).get("commit"))
            for key in ("harness", "mirix"):
                code[key].add((launch.get("code") or {}).get(key))
        commits.discard(None)
        self._check_code(code)
        self._check_work_lines(launches)
        self._check_protocol_within_attempt(launches)
        self._check_version_segments()
        if len(commits) > 1:
            f.warn(
                "H10.6",
                "code_changed",
                f"the attempt's invocations ran on {len(commits)} commits: "
                + ", ".join(sorted(str(c) for c in commits)),
            )
        self._check_dirty_tree()
        self._check_gateway_starts()

    def _not_reverified(self) -> dict[str, list[str]]:
        """``invariant:check`` -> what of this tree it reads that differs from the
        attempt's versions (:data:`TREE_INPUTS`: each version against the auditing tree;
        :data:`WITHIN_ATTEMPT_INPUTS`: the attempt's versions against each other). A
        version whose file map or constants are not recorded differs in everything it
        cannot vouch for (fail closed: its checks are listed)."""
        tree_fp = str(self.tree["fingerprint"])
        found: dict[str, list[str]] = {}

        def merge(key: str, inputs: Sequence[str]) -> None:
            if inputs:
                have = found.setdefault(key, [])
                have += [i for i in inputs if i not in have]

        for fingerprint in self.attempt_versions or [None]:
            diff = (
                protocol.version_diff(fingerprint, tree_fp, self.diff_versions)
                if fingerprint
                else None
            )
            for key, (constants, prefixes) in TREE_INPUTS.items():
                if (
                    key == "H10.2:conversation_not_append_only"
                    and self.benchmark != "webshop"
                ):
                    continue
                if (
                    key in VERSION_CONSTANT_CHECKS
                    and self._attempt_constant(VERSION_CONSTANT_CHECKS[key]) is not None
                ):
                    continue  # judged against the attempt's own constants
                merge(key, _changed_inputs(diff, self.benchmark, constants, prefixes))
        for a, b in zip(self.attempt_versions, self.attempt_versions[1:], strict=False):
            diff = protocol.version_diff(a, b, self.diff_versions)
            for key, (constants, prefixes) in WITHIN_ATTEMPT_INPUTS.items():
                merge(key, _changed_inputs(diff, self.benchmark, constants, prefixes))
        return found

    def _attempt_constant(self, key: str) -> list[Any] | None:
        """The values protocol constant ``key`` had under the attempt's versions, in
        order, or None when a version's per-key constants are not recorded, contradict
        its fingerprint or lack the key (:data:`VERSION_CONSTANT_CHECKS`)."""
        values: list[Any] = []
        for fingerprint in self.attempt_versions:
            version = self.diff_versions.get(fingerprint)
            if version is None:
                return None
            constants, inconsistent = protocol.constants_of(version, self.diff_versions)
            if constants is None or inconsistent or key not in constants:
                return None
            if constants[key] not in values:
                values.append(constants[key])
        return values or None

    def _prompt_key(self) -> str | None:
        """What the attempt's no-memory first prompts are made of (:data:`PROMPT_INPUTS`:
        the benchmark's templates and splits and the seed), as one hash;
        None when a version's file map or constants are not recorded, or the attempt's
        versions differ there. Cells of one key must share every item's no-memory
        first prompt."""
        constant_keys, prefixes = PROMPT_INPUTS
        wanted = tuple(p.format(benchmark=self.benchmark) for p in prefixes)
        keys: set[str] = set()
        for fingerprint in self.attempt_versions:
            version = self.diff_versions.get(fingerprint)
            if version is None:
                return None
            files = version.get("files")
            constants, _inconsistent = protocol.constants_of(
                version, self.diff_versions
            )
            if not isinstance(files, Mapping) or constants is None:
                return None
            keys.add(
                canonical_sha256(
                    {
                        "files": {
                            path: digest
                            for path, digest in sorted(files.items())
                            if str(path).startswith(wanted)
                        },
                        "constants": {
                            k: constants.get(k) for k in sorted(constant_keys)
                        },
                    }
                )
            )
        return keys.pop() if len(keys) == 1 else None

    def _report_not_reverified(self) -> None:
        """``info.not_reverified`` and one summary warning
        (``protocol_version_unauditable``): the checks whose inputs differ between this
        tree and the attempt's versions, with how many findings each had (warnings, not
        re-verified under the cell's own version) and where to audit the cell from."""
        listed: dict[str, list[str]] = {
            key: list(inputs) for key, inputs in self.f.not_reverified.items()
        }
        for key, inputs in self.f.partly_not_reverified.items():
            have = listed.setdefault(key, [])
            have += [i for i in inputs if i not in have]
        entries = [
            {
                "check": key,
                "inputs": inputs,
                "findings": self.f.downgraded[(key, NOT_REVERIFIED)],
            }
            for key, inputs in sorted(listed.items())
        ]
        self.info["not_reverified"] = entries
        if not entries:
            return
        fingerprint = str(self.attempt_fingerprint)
        tree_fp = str(self.tree["fingerprint"])
        diff = (
            protocol.version_diff(self.attempt_fingerprint, tree_fp, self.diff_versions)
            if self.attempt_fingerprint
            else None
        )
        unauditable = protocol.unauditable_changes(diff, self.benchmark)
        against = (
            "the frozen one"
            if tree_fp == self.frozen
            else f"the auditing tree's protocol {tree_fp[:12]}"
        )
        first = (
            f"attempt {self.number}'s protocol {fingerprint[:12]} differs from "
            f"{against} in {', '.join(unauditable)}: "
            if unauditable
            else ""
        )
        version = self.versions.get(fingerprint) or self._carried.get(fingerprint) or {}
        self.f.warn(
            "H10.6",
            "protocol_version_unauditable",
            first
            + "these checks read inputs of this tree that differ from, or are not "
            "recorded for, the attempt's versions, and were not re-verified under them "
            "(their findings are warnings): "
            + "; ".join(
                f"{e['check']} ({e['findings']} finding(s); inputs: "
                f"{', '.join(e['inputs'])})"
                for e in entries
            )
            + ". Audit it from "
            + protocol.version_checkout_hint(fingerprint, version.get("git")),
        )

    def _start_for(self, row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The gateway start record of the gateway that wrote ``row``: the one its
        ``gateway_start`` names, else (a line written before gateways stamped their
        lines) the latest start at or before its time; None when neither is known."""
        stamp = row.get("gateway_start")
        if stamp:
            return self._starts_by_id.get(str(stamp))
        at = _ts(row.get("ts"))
        if at is None:
            return None
        timed = [
            (t, s)
            for s in self.starts
            if (t := _ts(s.get("ts"))) is not None and t <= at
        ]
        return max(timed, key=lambda pair: pair[0])[1] if timed else None

    def _attempt_settings(self) -> tuple[list[dict[str, Any]], str]:
        """The gateway settings the attempt's own versions give (``SETTINGS_CONSTANTS``
        of each version's per-key constants, ``protocol.constants_of``), else those of the
        gateway start records made under one of its versions; with where they come from:
        ``attempt version``, ``gateway start record`` or ``unknown`` (none)."""
        allowed: list[dict[str, Any]] = []

        def add(settings: dict[str, Any]) -> None:
            if not any(_canon(settings) == _canon(a) for a in allowed):
                allowed.append(settings)

        for fingerprint in self.attempt_versions:
            version = self.diff_versions.get(fingerprint)
            if version is None:
                continue
            constants, _inconsistent = protocol.constants_of(
                version, self.diff_versions
            )
            if constants is None or any(
                key not in constants for key in SETTINGS_CONSTANTS.values()
            ):
                continue
            add({s: constants[c] for s, c in SETTINGS_CONSTANTS.items()})
        if allowed:
            return allowed, "attempt version"
        for start in self.starts:
            settings = start.get("protocol_settings")
            if start.get(
                "protocol_fingerprint"
            ) in self.attempt_versions and isinstance(settings, Mapping):
                add(dict(settings))
        if allowed:
            return allowed, "gateway start record"
        return [], "unknown"

    def _expected_settings(self, start: Mapping[str, Any] | None) -> dict[str, Any]:
        """The settings a call must have gone out with: its gateway start record's, over
        the attempt's own (else this tree's) for anything the record lacks."""
        expected = (
            dict(self._allowed_settings[0])
            if self._allowed_settings
            else dict(PROTOCOL_SETTINGS)
        )
        settings = start.get("settings") if isinstance(start, Mapping) else None
        if isinstance(settings, Mapping):
            expected.update(settings)
        return expected

    def _check_hosts(self) -> None:
        """H10.6, hosts: the attempt ran on one host. Its
        ``attempt.json`` ``host``, every ``work`` line of an invocation that ran it and
        every learning resume name the same host (its machine id, system and machine;
        the hostname only for records without a machine id: ``cell_records.same_host``),
        else ``host_changed_within_attempt``: a resume on
        another host would restore a database and HOME that live on the first. The host
        and the sandbox image (WebShop's server image) are kept for the checks across
        cells."""
        f = self.f
        f.ok("H10.6")
        host = self.state.get("host")
        preflight = self.manifest.get("preflight") or {}
        env = preflight.get("env") if isinstance(preflight, Mapping) else None
        image = env.get("sandbox_image_id") if isinstance(env, Mapping) else None
        if image:
            self.info["sandbox_image_id"] = str(image)
        if not isinstance(host, Mapping):
            f.warn(
                "H10.6",
                "host_unrecorded",
                f"attempt {self.number} records no host (an attempt started before "
                "hosts were recorded): which host ran it cannot be checked",
            )
            return
        self.info["host"] = {
            k: host.get(k) for k in (*HOST_IDENTITY_FIELDS, "release", "python")
        }
        others: list[tuple[str, Mapping[str, Any]]] = []
        for line in read_jsonl(self.paths.launches):
            if line.get("event") == LAUNCH_WORK and line.get("attempt") == self.number:
                if isinstance(line.get("host"), Mapping):
                    others.append((f"launch {line.get('launch')}", line["host"]))
        for resume in self.state.get("resumes") or ():
            if isinstance(resume, Mapping) and isinstance(resume.get("host"), Mapping):
                others.append(
                    (f"the resume of segment {resume.get('segment')}", resume["host"])
                )
        for where, other in others:
            f.ok("H10.6")
            if not same_host(host, other):
                f.add(
                    "H10.6",
                    "host_changed_within_attempt",
                    f"{where} ran attempt {self.number} on "
                    f"{host_text(other)}, the attempt started on {host_text(host)}",
                )

    def _check_provider_policy(
        self,
        row: Mapping[str, Any],
        expected_policy: Any = None,
        *,
        unverified: bool = False,
    ) -> None:
        """H10.6, the provider policy, judged against the call's own gateway start
        record (``expected_policy``: its settings' ``provider_policy``; a later policy
        never fails an older call): a chat call of a model with a routing went out with
        exactly that routing in its ``wire`` (``provider_policy_missing``), and a call of any other model or
        role with none (``provider_policy_unexpected``); both are validity checks. No ok
        call and no upstream attempt of it was served by a provider the routing it went
        out with ignores (``ignored_provider_served``, by slug or name,
        ``protocol.ignored_by``; recorded), except an attempt the gateway refused for
        that (``outcome: ignored_provider``: its answer was never returned),
        which ``info.provider_policy.refused_ignored`` counts. A line refused before the
        rewrite has no ``wire`` and sent nothing."""
        f = self.f
        counts = self._policy_counts
        call_id, role = row.get("call_id"), row.get("role")
        model = str(row.get("model") or "")
        policy = expected_policy if isinstance(expected_policy, Mapping) else None
        wire = row.get("wire")
        routing_sent: Any = None
        if isinstance(wire, Mapping) and "model" in wire:
            # The routing belongs to the model the body named (the wire is one body).
            sent = str(wire.get("model"))
            expected = (
                protocol.provider_routing(sent, policy) if role in CHAT_ROLES else None
            )
            actual = wire.get("provider")
            routing_sent = actual if isinstance(actual, Mapping) else None
            if expected is not None:
                counts["policy_calls"] += 1
                if actual != expected:
                    f.add(
                        "H10.6",
                        "provider_policy_missing",
                        f"call {call_id} ({role}, {sent}) went out with provider "
                        f"routing {actual!r}, not {expected!r}, the routing of its "
                        "gateway start record",
                        unverified=unverified,
                    )
            elif actual is not None:
                f.add(
                    "H10.6",
                    "provider_policy_unexpected",
                    f"call {call_id} ({role}, {sent}) went out with provider routing "
                    f"{actual!r}; its gateway start record gives it none",
                    unverified=unverified,
                )
        routing = (
            routing_sent
            if routing_sent is not None
            else protocol.provider_routing(model, policy)
        )
        if routing is None:
            return
        served: list[str] = []
        if row.get("status") == "ok" and protocol.ignored_by(
            routing, row.get("provider")
        ):
            served.append(f"the answer ({row.get('provider')})")
        for attempt in row.get("attempts") or ():
            if not isinstance(attempt, Mapping) or not protocol.ignored_by(
                routing, attempt.get("provider")
            ):
                continue
            if attempt.get("outcome") == IGNORED_PROVIDER_OUTCOME:
                counts["refused_ignored"] += 1
                continue
            served.append(f"attempt {attempt.get('n')} ({attempt.get('provider')})")
        if served:
            counts["served_by_ignored"] += 1
            f.add(
                "H10.6",
                "ignored_provider_served",
                f"call {call_id} ({role}, {model}): {', '.join(served)} came from a "
                "provider the routing it went out with ignores",
            )

    def _worked_on_attempt(
        self, launch: Mapping[str, Any], started: datetime | None
    ) -> bool:
        """A ``launches.jsonl`` line of an invocation that ran the audited attempt's work.

        ``work`` lines name their attempt: only those of this attempt count. A ``start``
        line is every invocation, including those that ran nothing (the cell already
        complete, a refusal, a pause, the gateway down), so it never counts. A line
        written before launch lines carried an ``event`` has none: every invocation since
        the attempt started counts."""
        event = launch.get("event")
        if event == LAUNCH_WORK:
            return launch.get("attempt") == self.number
        if event is not None:
            return False
        at = _ts(launch.get("at"))
        return started is None or at is None or at >= started

    def _check_work_lines(self, launches: Sequence[Mapping[str, Any]]) -> None:
        """The code check reads only ``work`` lines, so every invocation that wrote the
        attempt's learning or test rows must have one: an invocation that has a ``start``
        line and rows of the attempt, but no ``work`` line for it, ran the attempt on code
        nobody recorded."""
        f = self.f
        started = {
            str(line.get("launch"))
            for line in launches
            if line.get("event") == LAUNCH_START
        }
        worked = {
            str(line.get("launch"))
            for line in launches
            if line.get("event") == LAUNCH_WORK and line.get("attempt") == self.number
        }
        rows = [
            *read_jsonl(self.attempt_paths.learning),
            *read_jsonl(self.attempt_paths.results),
        ]
        producers = {str(r["launch"]) for r in rows if r.get("launch")}
        f.ok("H10.6")
        for launch in sorted((producers & started) - worked):
            f.add(
                "H10.6",
                "code_unrecorded",
                f"launch {launch} wrote attempt {self.number}'s learning or test rows "
                "but launches.jsonl has no work line of it for this attempt: the code "
                "it ran was not recorded",
            )

    def _check_protocol_within_attempt(
        self, launches: Sequence[Mapping[str, Any]]
    ) -> None:
        """The protocol versions of an attempt are recorded (``run.py`` resumes under
        another version and appends a ``version_segments`` entry): the ``protocol`` of
        every ``work`` line of the attempt is its ``attempt.json`` ``protocol`` or a version a (written, not synthesized) segment
        records, else ``protocol_changed_within_attempt`` (recorded, a warning). Records
        written before run.py recorded the protocol carry none and are not checked: an
        attempt without ``attempt.json`` ``protocol`` is compared with the protocol its
        cell's records attribute to it (``protocol.attempt_protocol``), and only when a
        ``work`` line of it carries one; a ``work`` line without one is skipped."""
        f = self.f
        recorded = self.state.get("protocol")
        lines = [
            (str(line.get("launch")), line["protocol"])
            for line in launches
            if line.get("event") == LAUNCH_WORK
            and line.get("attempt") == self.number
            and isinstance(line.get("protocol"), Mapping)
        ]
        if not isinstance(recorded, Mapping) and not lines:
            return  # written before attempts and work lines recorded their protocol
        expected = {self.attempt_fingerprint} | {
            _segment_fingerprint(seg) for seg in self.segments if not seg.get("legacy")
        }
        expected.discard(None)
        f.ok("H10.6")
        for launch, line_protocol in lines:
            if line_protocol.get("fingerprint") not in expected:
                f.add(
                    "H10.6",
                    "protocol_changed_within_attempt",
                    f"launch {launch} ran attempt {self.number} under protocol "
                    f"{str(line_protocol.get('fingerprint'))[:12]}, the attempt runs "
                    f"under {str(self.attempt_fingerprint)[:12]}, and no "
                    "version_segments entry records it",
                )

    def _check_code(self, code: Mapping[str, set[str | None]]) -> None:
        """The model-visible code of an attempt is recorded. Every invocation that
        ran the attempt's work (learning, or a resumed test phase) recorded its harness
        code (``protocol.code_fingerprint``: the environments, rollouts, methods, gateway
        and token budget) and, for MIRIX, its MIRIX package; an invocation that ran
        nothing does not count. A value neither ``attempt.json`` nor a (written)
        version segment records is ``model_visible_code_changed`` (recorded). The start's
        values are kept for the check across the
        cells of a benchmark."""
        f = self.f
        needed = ["harness"] + (["mirix"] if self.method == "mirix" else [])
        self.info["code"] = {}
        state_code = self.state.get("code") or {}
        for key in needed:
            f.ok("H10.6")
            values = code.get(key, set())
            known = {v for v in values if v}
            if None in values or not known:
                f.add(
                    "H10.6",
                    "code_unrecorded",
                    f"an invocation of the attempt recorded no {key} code fingerprint: "
                    "the code that shaped its model calls cannot be compared",
                )
            allowed = {state_code.get(key)} | {
                (seg.get("code") or {}).get(key)
                for seg in self.segments
                if not seg.get("legacy") and isinstance(seg.get("code"), Mapping)
            }
            unexplained = sorted(str(v) for v in known - allowed)
            if key == "harness":
                self.harness_code_versions = {str(v) for v in known}
            if len(known) > 1 and unexplained:
                f.add(
                    "H10.6",
                    "model_visible_code_changed",
                    f"the attempt ran {len(known)} versions of the {key} code ("
                    + ", ".join(sorted(str(v)[:12] for v in known))
                    + "); "
                    + ", ".join(v[:12] for v in unexplained)
                    + " no version_segments entry records",
                )
            if state_code.get(key):
                self.info["code"][key] = state_code[key]
            elif len(known) == 1:
                self.info["code"][key] = next(iter(known))

    def _check_version_segments(self) -> None:
        """An attempt that ran under more than one version (``version_segments``; legacy
        attempts: its work lines grouped by protocol, code and commit,
        ``protocol.attempt_versions``). ``info.version_segments`` lists each later
        segment (index 1 on: from which item, at what time, under which protocol and
        code, what changed). A mixed-version attempt (another protocol or model-visible
        code; a commit or tree change alone is listed, not mixed) is one warning,
        ``mixed_version_attempt``: the report marks the cell."""
        segments = self.segments
        self.info["version_segments"] = [
            self._segment_record(seg) for seg in segments if _segment_index(seg) != 0
        ]
        fingerprints = {_segment_fingerprint(seg) for seg in segments}
        codes = {
            (
                (seg.get("code") or {}).get("harness"),
                (seg.get("code") or {}).get("mirix"),
            )
            for seg in segments
            if isinstance(seg.get("code"), Mapping)
        }
        if len(segments) < 2 or (len(fingerprints) < 2 and len(codes) < 2):
            return
        parts = []
        for seg in segments:
            fingerprint = str(_segment_fingerprint(seg))
            harness = str((seg.get("code") or {}).get("harness"))
            where = "its start" if _segment_index(seg) == 0 else _from_text(seg)
            known = "" if self._version_known(fingerprint) else " (not registered)"
            changed = _changed_text(seg.get("changed"))
            parts.append(
                f"segment {_segment_index(seg)} from {where} ({seg.get('started_at')}): "
                f"protocol {fingerprint[:12]}{known}, harness {harness[:12]}"
                + (f", changed {changed}" if changed else "")
            )
        self.f.warn(
            "H10.6",
            "mixed_version_attempt",
            f"attempt {self.number} ran under {len(segments)} version segments ("
            f"{len(fingerprints)} protocol(s), {len(codes)} code version(s)): "
            + "; ".join(parts)
            + ". The report marks the cell mixed-version",
        )

    def _segment_record(self, seg: Mapping[str, Any]) -> dict[str, Any]:
        """``info.version_segments``: one later segment of the attempt."""
        fingerprint = _segment_fingerprint(seg)
        known = seg.get("known_version")
        return {
            "segment": _segment_index(seg),
            "at": seg.get("started_at"),
            "phase": seg.get("phase"),
            "from": seg.get("learning_from") or seg.get("test_from"),
            "fingerprint": fingerprint,
            "known": known
            if isinstance(known, bool)
            else self._version_known(fingerprint),
            "code": seg.get("code"),
            "changed": seg.get("changed"),
        }

    def _check_dirty_tree(self) -> None:
        """A warning when the attempt ran from a dirty checkout: the segments' ``tree``
        records when they have one (each version's own tree, its diff kept beside it),
        else ``attempt.json`` ``git.dirty``."""
        trees = [
            (seg, seg["tree"])
            for seg in self.segments
            if isinstance(seg.get("tree"), Mapping)
            and isinstance(seg["tree"].get("dirty"), bool)
        ]
        if trees:
            dirty = [(seg, tree) for seg, tree in trees if tree["dirty"]]
            if dirty:
                self.f.warn(
                    "H10.6",
                    "dirty_tree",
                    f"{len(dirty)} version segment(s) of the attempt ran from a dirty "
                    "checkout (their diffs: "
                    + ", ".join(
                        f"segment {_segment_index(seg)}: {tree.get('diff') or '-'}"
                        for seg, tree in dirty
                    )
                    + ")",
                )
        elif (self.state.get("git") or {}).get("dirty"):
            self.f.warn(
                "H10.6", "dirty_tree", "the attempt started from a dirty checkout"
            )

    def _starts_files(self) -> list[Path]:
        """Every ``_gateway/starts.jsonl`` that may hold the start records of the gateways
        that served this attempt (see :meth:`_gateway_starts`), one per real file."""
        dirs = [
            self.ledger_path.parent.parent,
            self.ledger_path.resolve().parent.parent,
            self.runs_dir,
            *self._host_mirrors(),
            *self.gateway_runs_dirs,
        ]
        seen: set[Path] = set()
        files: list[Path] = []
        for directory in dirs:
            path = directory / "_gateway" / "starts.jsonl"
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key not in seen and path.is_file():
                seen.add(key)
                files.append(path)
        return files

    def _starts_text(self, files: Sequence[Path]) -> str:
        if not files:
            return f"{self.ledger_path.parent.parent / '_gateway' / 'starts.jsonl'} (missing)"
        return ", ".join(str(p) for p in files)

    def _gateway_starts(self) -> list[dict[str, Any]]:
        """The start records of the gateways that served this attempt.

        Every ledger line names the gateway process that wrote it (``gateway_start``), so
        a line is matched to exactly its gateway's start record, whatever other gateways
        shared the runs directory. Lines written before gateways stamped their
        lines fall back to time: the last start before the attempt's first such line and
        every start while it ran. A stamp without a start record is reported.

        The records are the union of every ``_gateway/starts.jsonl`` that can hold them:
        the ledger's gateway runs directory (and, through a merged view's link, the host
        mirror it points into), the runs directory, each ``_hosts/*`` mirror and
        ``--gateway-runs-dir`` (one record per ``start_id``)."""
        files = self._starts_files()
        records: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for starts in files:
            for r in read_jsonl(starts):
                if r.get("event") != "gateway_start":
                    continue
                key = r.get("start_id") or (r.get("ts"), r.get("pid"))
                if key in seen:
                    continue
                seen.add(key)
                records.append(r)
        by_id = {str(r["start_id"]): r for r in records if r.get("start_id")}
        found: dict[str, dict[str, Any]] = {}
        unstamped: list[datetime] = []
        missing: set[str] = set()
        for row in self.ledger:
            stamp = row.get("gateway_start")
            if stamp:
                if str(stamp) in by_id:
                    found[str(stamp)] = by_id[str(stamp)]
                else:
                    missing.add(str(stamp))
            elif (t := _ts(row.get("ts"))) is not None and row.get("event") in (
                "call",
                "call_started",
            ):
                unstamped.append(t)
        for stamp in sorted(missing):
            self.f.add(
                "H10.6",
                "gateway_start_missing",
                f"ledger lines were written by gateway {stamp}, which has no start "
                f"record in {self._starts_text(files)}",
            )
        starts = list(found.values())
        timed = [(t, r) for r in records if (t := _ts(r.get("ts"))) is not None]
        if unstamped and timed:
            first, last = min(unstamped), max(unstamped)
            before = [r for t, r in timed if t <= first]
            during = [r for t, r in timed if first < t <= last]
            for record in ([before[-1]] if before else []) + during:
                if record not in starts:
                    starts.append(record)
        return starts

    def _check_gateway_starts(self) -> None:
        """The gateways that served the attempt (``self.starts``): each ran one of the
        attempt's protocol versions (another known one: a warning; one no freeze
        produced: recorded), with the model settings the attempt's version records
        (``gateway_start_settings``, validity: calls went out with other model
        parameters than the attempt records), and forwarded to the expected upstream.
        Any other difference from the attempt's version or from this tree (deadline,
        retries, backoff, tokenizer margin, encoding, conformance) is one warning per
        gateway, ``gateway_settings_differ`` (H10.6-gateway-settings)."""
        f = self.f
        starts = self.starts
        self.info["gateway_starts"] = len(starts)
        if self.calls and not starts:
            f.add(
                "H10.6",
                "gateway_start_missing",
                "no gateway start record covers the attempt's calls "
                f"({self._starts_text(self._starts_files())})",
            )
        expected = self.attempt_fingerprint or self.cell.get("protocol_fingerprint")
        versions = set(self.attempt_versions) | ({expected} if expected else set())
        allowed = self._allowed_settings
        for start in starts:
            f.ok("H10.6")
            f.ok("H10.3")
            gateway = f"a gateway (pid {start.get('pid')}, {start.get('ts')})"
            served = start.get("protocol_fingerprint")
            if served and served in versions:
                pass
            elif self._version_known(served):
                # Another frozen version (a gateway started before a re-freeze):
                # reported, not enforced; the per-call checks
                # above hold every call to its own start record's settings and routing.
                f.warn(
                    "H10.6",
                    "gateway_protocol_version",
                    f"{gateway} ran protocol {str(served)[:12]}, the attempt ran under "
                    f"{str(expected)[:12]}",
                )
            else:
                f.add(
                    "H10.6",
                    "gateway_fingerprint",
                    f"{gateway} ran protocol {served}, which no freeze produced",
                )
            settings = start.get("settings")
            settings = settings if isinstance(settings, Mapping) else {}
            # Nothing of the attempt records its settings: the start is held
            # to this tree's, and a difference is listed as not re-verified.
            against = allowed or [dict(PROTOCOL_SETTINGS)]
            if not any(
                all(_canon(settings.get(k)) == _canon(a.get(k)) for k in MODEL_SETTINGS)
                for a in against
            ):
                fields = [
                    k
                    for k in MODEL_SETTINGS
                    if _canon(settings.get(k)) != _canon(against[0].get(k))
                ]
                if not allowed:
                    inputs = f.partly_not_reverified.setdefault(
                        "H10.6:gateway_start_settings", []
                    )
                    if SETTINGS_NOT_RECORDED not in inputs:
                        inputs.append(SETTINGS_NOT_RECORDED)
                f.add(
                    "H10.6",
                    "gateway_start_settings",
                    f"{gateway} ran "
                    + ", ".join(f"{k}={settings.get(k)!r}" for k in fields)
                    + (
                        f"; the attempt's protocol {str(expected)[:12]} says "
                        if allowed
                        else "; the attempt records no protocol settings, and this "
                        "tree says "
                    )
                    + ", ".join(f"{k}={against[0].get(k)!r}" for k in fields)
                    + ": calls went out with other model parameters than the attempt "
                    "records",
                    unverified=not allowed,
                )
            reference = (
                allowed[0]
                if allowed
                else start.get("protocol_settings")
                if isinstance(start.get("protocol_settings"), Mapping)
                else {}
            )
            differences = []
            for key in sorted(set(settings) | set(PROTOCOL_SETTINGS) | set(reference)):
                value, attempt_value = settings.get(key), reference.get(key)
                tree_value = PROTOCOL_SETTINGS.get(key)
                if (
                    key not in MODEL_SETTINGS
                    and _canon(value) != _canon(attempt_value)
                    or _canon(value) != _canon(tree_value)
                ):
                    differences.append(
                        f"{key}: start {value!r} / attempt {attempt_value!r} / tree "
                        f"{tree_value!r}"
                    )
            if start.get("encoding") != protocol.TOKEN_ENCODING:
                differences.append(
                    f"encoding: start {start.get('encoding')!r} / tree "
                    f"{protocol.TOKEN_ENCODING!r}"
                )
            if start.get("protocol_conformant") is not True:
                differences.append(
                    f"protocol_conformant: {start.get('protocol_conformant')!r}"
                )
            if differences:
                f.warn(
                    "H10.6",
                    "gateway_settings_differ",
                    f"{gateway} started with settings that differ: "
                    + "; ".join(differences)
                    + ". deadline_s, max_retries, backoff_base_s, backoff_cap_s and "
                    "tokenizer_margin are recorded only; the model settings "
                    "(reasoning_effort, seed, provider_policy) are held to the attempt's "
                    "version (H10.6-gateway-settings)",
                )
            if start.get("upstream") != self.expected_upstream:
                f.add(
                    "H10.3",
                    "upstream",
                    f"a gateway (pid {start.get('pid')}) forwarded to "
                    f"{start.get('upstream')}, not {self.expected_upstream}",
                )

    # -- resumed learning --------------------------------------------------------------

    def _check_resumes(self) -> None:  # noqa: C901 - one pass over the segments
        """A learning phase resumed from its checkpoints (H10.3; run.py ``_resume_learning``).

        * **Segments.** Each ``discarded.jsonl`` row (segments 1, 2, ... in order, of this
          attempt and run) names the resume marker it answers and the checkpoint it
          restored; both markers are in the ledger, the checkpoint's before the resume's.
          Its calls, recomputed from the ledger, are every call whose ``call_started``
          lies strictly between the two markers, less those an earlier segment discarded
          (``cell_records.segment_calls``); the row must list exactly those
          (``discarded_segment_mismatch``; a final line written later, ``final_line:
          false``, is fine). A resume marker no row answers is a resume that died before
          it recorded its segment: allowed only when no call started between it and the
          next resume marker (``discarded_segment_unrecorded``).
        * **Restores.** Every row shows its method files restored equal and, for MIRIX,
          the database digest equal; the snapshot it expected is the checkpoint marker's,
          and the snapshot the method rebuilt equals it, as the matching
          ``snapshot_checks`` entry (kind ``learning_resume``) says and its stored copy
          shows (``learning_resume_mismatch``). A segment followed by another may lack the
          rebuilt snapshot (the invocation died before the method's setup; the next resume
          restored again), the last one may not. A mismatch that a later resume from the
          same or an earlier checkpoint superseded is a warning.
        * **Records.** A trajectory moved to ``discarded/`` holds only discarded actor
          calls, a live one none; a discarded actor call no trajectory recorded (a try
          killed mid-rollout) is a warning (``discarded_inputs_unrecorded``).

        What it finds is used everywhere else: discarded calls need no live trajectory,
        a later repeat of one is the redo (every role), and they are left out of the
        learning-infrastructure check and MIRIX's reconciliation (checked per segment).
        """
        f = self.f
        rows = [r for r in read_discarded(self.attempt_paths) if isinstance(r, dict)]
        self.resume_rows = rows
        lines = read_ledger_lines(self.ledger_path)
        by_offset = {line.offset: line for line in lines}
        resume_markers = [
            line for line in lines if line.row.get("event") == LEDGER_EVENT_RESUME
        ]
        checks = [
            c
            for c in self.state.get("snapshot_checks") or ()
            if isinstance(c, Mapping) and c.get("kind") == LEARNING_RESUME_CHECK
        ]
        if not rows and not resume_markers and not checks:
            return
        earlier: set[str] = set()
        answered: set[int] = set()
        segments: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            f.ok("H10.3")
            segment = row.get("segment")
            where = f"discarded segment {segment}"
            if segment != index + 1:
                f.add(
                    "H10.3",
                    "discarded_segment_mismatch",
                    f"{where} is row {index + 1} of discarded.jsonl: segments are "
                    "numbered 1, 2, ... in order",
                )
            if row.get("run_id") != self.run_id or row.get("attempt") != self.number:
                f.add(
                    "H10.3",
                    "discarded_segment_mismatch",
                    f"{where} belongs to run {row.get('run_id')} attempt "
                    f"{row.get('attempt')}, not to this attempt ({self.run_id})",
                )
            offset = row.get("resume_marker_offset")
            marker = by_offset.get(offset) if isinstance(offset, int) else None
            if (
                marker is None
                or marker.row.get("event") != LEDGER_EVENT_RESUME
                or marker.row.get("segment") != segment
            ):
                f.add(
                    "H10.3",
                    "discarded_segment_mismatch",
                    f"{where} names no resume marker of its segment at ledger offset "
                    f"{offset}: its calls cannot be read from the ledger",
                )
                continue
            answered.add(marker.offset)
            restored = (
                row.get("checkpoint")
                if isinstance(row.get("checkpoint"), Mapping)
                else {}
            )
            cid, c_offset = restored.get("id"), restored.get("offset")
            if (marker.row.get("checkpoint"), marker.row.get("checkpoint_offset")) != (
                cid,
                c_offset,
            ):
                f.add(
                    "H10.3",
                    "discarded_segment_mismatch",
                    f"{where} restored {cid} (offset {c_offset}), its resume marker "
                    f"names {marker.row.get('checkpoint')} (offset "
                    f"{marker.row.get('checkpoint_offset')})",
                )
            cmark: LedgerLine | None = None
            if cid is not None:
                cmark = by_offset.get(c_offset) if isinstance(c_offset, int) else None
                if (
                    cmark is None
                    or cmark.row.get("event") != LEDGER_EVENT_CHECKPOINT
                    or cmark.row.get("id") != cid
                    or cmark.offset >= marker.offset
                ):
                    f.add(
                        "H10.3",
                        "discarded_segment_mismatch",
                        f"{where} restored checkpoint {cid}, whose marker is not in the "
                        f"ledger at offset {c_offset} before the resume marker",
                    )
                    cmark = None
                    continue
            calls = segment_calls(
                lines,
                checkpoint_offset=cmark.offset if cmark is not None else None,
                resume_offset=marker.offset,
                exclude=earlier,
            )
            computed = {str(c["call_id"]) for c in calls}
            recorded = {
                str(c.get("call_id"))
                for c in row.get("calls") or ()
                if isinstance(c, Mapping) and c.get("call_id")
            }
            if computed != recorded:
                f.add(
                    "H10.3",
                    "discarded_segment_mismatch",
                    f"{where} lists {len(recorded)} call(s); the ledger's markers "
                    f"delimit {len(computed)} ({len(recorded - computed)} listed that "
                    f"are not between them: {_some(sorted(recorded - computed))}; "
                    f"{len(computed - recorded)} between them not listed: "
                    f"{_some(sorted(computed - recorded))})",
                )
            for call_id in computed:
                self.discarded[call_id] = (
                    segment
                    if isinstance(segment, int) and not isinstance(segment, bool)
                    else index + 1
                )
            earlier |= computed
            segments.append(
                {
                    "row": row,
                    "segment": segment,
                    "checkpoint": cid,
                    "ordinal": _checkpoint_ordinal(cid),
                    "calls": calls,
                    "checkpoint_sha": (
                        cmark.row.get("snapshot_sha256") if cmark else None
                    ),
                }
            )
        # Resume markers no row answers.
        for position, marker in enumerate(resume_markers):
            if marker.offset in answered:
                continue
            f.ok("H10.3")
            end = (
                resume_markers[position + 1].offset
                if position + 1 < len(resume_markers)
                else None
            )
            started = [
                line
                for line in call_placements(lines)
                if line.offset > marker.offset and (end is None or line.offset < end)
            ]
            if started:
                f.add(
                    "H10.3",
                    "discarded_segment_unrecorded",
                    f"the resume marker of segment {marker.row.get('segment')} (ledger "
                    f"offset {marker.offset}) has no discarded.jsonl row, yet "
                    f"{len(started)} call(s) started after it"
                    + ("" if end is None else " before the next resume marker")
                    + ": the calls it discarded are not recorded",
                )
        self._resumed_segments = {s["segment"] for s in segments}
        self._check_restores(segments, checks)
        self._check_discarded_records()
        self.info["learning_resumes"] = [self._segment_info(s) for s in segments]

    def _report_unreadable_discarded(self) -> None:
        """A discarded trajectory that cannot be read (H10.1-discarded-scan): what the
        method received in that try was not scanned for labels. A validity violation,
        unless the resume that discarded it restored its checkpoint exactly (the
        resume check found its segment and no restore problem): then the rolled-back
        try left nothing in the memory, and it is a warning."""
        for segment, rel, error in self._unreadable_discarded:
            if (
                segment in self._resumed_segments
                and segment not in self._restore_failed
            ):
                self.f.warn(
                    "H10.1",
                    "trajectory_missing",
                    f"{rel} cannot be read ({error}): what the method received in "
                    f"discarded segment {segment} was not scanned; that resume restored "
                    "its checkpoint exactly, so the rolled-back try left nothing in the "
                    "memory (H10.1-discarded-scan)",
                )
            else:
                self.f.add(
                    "H10.1",
                    "trajectory_missing",
                    f"{rel} cannot be read ({error}): what the method received in "
                    f"discarded segment {segment} cannot be checked, and no resume shows "
                    "that its checkpoint was restored exactly",
                )

    def _check_restores(
        self, segments: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]
    ) -> None:
        """The restores of :meth:`_check_resumes` (``learning_resume_mismatch``)."""
        f = self.f
        for index, seg in enumerate(segments):
            f.ok("H10.3")
            row, segment = seg["row"], seg["segment"]
            where = f"the resume of discarded segment {segment}"
            later = segments[index + 1 :]
            # A later resume from this checkpoint or an earlier one restored again.
            superseded = any(
                (
                    s["ordinal"] is not None
                    and seg["ordinal"] is not None
                    and s["ordinal"] <= seg["ordinal"]
                )
                or s["checkpoint"] is None
                for s in later
            )
            restore = (
                row.get("restore") if isinstance(row.get("restore"), Mapping) else {}
            )
            problems: list[str] = []
            if restore.get("files_equal") is not True:
                problems.append("its method files were not restored equal")
            mirix_db = self.method == "mirix" and seg["checkpoint"] is not None
            if restore.get("db_digest_equal") is False or (
                mirix_db and restore.get("db_digest_equal") is not True
            ):
                problems.append(
                    "the restored database's digest is not the checkpoint's"
                )
            expected = restore.get("snapshot_sha256_expected")
            actual = restore.get("snapshot_sha256_actual")
            if seg["checkpoint"] is not None:
                if expected != seg["checkpoint_sha"]:
                    problems.append(
                        f"it expected snapshot {str(expected)[:12]}, the checkpoint "
                        f"marker records {str(seg['checkpoint_sha'])[:12]}"
                    )
                if actual is None:
                    if not later:
                        problems.append(
                            "the rebuilt memory's snapshot was never compared with the "
                            "checkpoint's"
                        )
                elif actual != expected:
                    problems.append(
                        f"the rebuilt memory hashes to {str(actual)[:12]}, the "
                        f"checkpoint's to {str(expected)[:12]}"
                    )
                matching = [c for c in checks if c.get("segment") == segment]
                if actual is not None and not matching:
                    problems.append("no learning_resume snapshot check records it")
                for check in matching:
                    if not (
                        check.get("equal") is True
                        and check.get("sha256")
                        == actual
                        == check.get("expected_sha256")
                        and check.get("expected_sha256") == expected
                    ):
                        problems.append(
                            f"its snapshot check ({check.get('path')}) found "
                            f"{str(check.get('sha256'))[:12]}, expected "
                            f"{str(check.get('expected_sha256'))[:12]}"
                        )
                    stored = read_json(self.root / str(check.get("path")))
                    if not isinstance(stored, dict) or canonical_sha256(
                        stored.get("snapshot")
                    ) != check.get("sha256"):
                        problems.append(
                            f"the stored snapshot {check.get('path')} does not match "
                            "its record"
                        )
            if not problems:
                continue
            self._restore_failed.add(segment)
            detail = f"{where} ({seg['checkpoint']}): " + "; ".join(problems)
            if superseded:
                f.warn(
                    "H10.3",
                    "learning_resume_mismatch",
                    detail + " (a later resume restored the same or an earlier "
                    "checkpoint again)",
                )
            else:
                f.add("H10.3", "learning_resume_mismatch", detail)
        known = {s["segment"] for s in segments}
        for check in checks:
            if check.get("segment") not in known:
                f.add(
                    "H10.3",
                    "learning_resume_mismatch",
                    f"a learning_resume snapshot check ({check.get('path')}) names "
                    f"segment {check.get('segment')}, which discarded.jsonl does not "
                    "record",
                )

    def _check_discarded_records(self) -> None:
        """Trajectories and calls agree on what was discarded (see :meth:`_check_resumes`)."""
        f = self.f
        recorded: set[str] = set()
        for t in self.discarded_trajectories:
            for call in t.data.get("calls") or ():
                call_id = (
                    call.get("gateway_call_id") if isinstance(call, Mapping) else None
                )
                if not call_id:
                    continue
                f.ok("H10.3")
                recorded.add(str(call_id))
                if str(call_id) not in self.discarded:
                    f.add(
                        "H10.3",
                        "discarded_segment_mismatch",
                        f"actor call {call_id} of a trajectory moved to discarded segment "
                        f"{t.segment} is not a discarded call: the rollback moved a "
                        "try the ledger shows as live",
                        item=t.raw_id,
                    )
        for t in self.trajectories:
            for call in t.data.get("calls") or ():
                call_id = (
                    call.get("gateway_call_id") if isinstance(call, Mapping) else None
                )
                if call_id and str(call_id) in self.discarded:
                    f.add(
                        "H10.3",
                        "discarded_segment_mismatch",
                        f"actor call {call_id} of a live {t.phase} trajectory lies in "
                        f"discarded segment {self.discarded[str(call_id)]}",
                        item=t.raw_id,
                    )
        unrecorded: dict[int, int] = defaultdict(int)
        for call_id, segment in self.discarded.items():
            line = self.by_call_id.get(call_id)
            if (
                line is not None
                and line.get("role") == "actor"
                and call_id not in recorded
            ):
                unrecorded[segment] += 1
        for segment, count in sorted(unrecorded.items()):
            f.warn(
                "H10.3",
                "discarded_inputs_unrecorded",
                f"discarded segment {segment} holds {count} actor call(s) of a try killed "
                "before its trajectory was written: what the method received in them "
                "was rolled back and cannot be scanned (H10.1)",
            )

    def _segment_info(self, seg: Mapping[str, Any]) -> dict[str, Any]:
        """``info.learning_resumes``: one discarded segment, as the ledger shows it."""
        row = seg["row"]
        calls = seg["calls"]
        roles = Counter(str(c.get("role")) for c in calls)
        cost = 0.0
        unknown = 0
        for c in calls:
            line = self.by_call_id.get(str(c["call_id"]))
            if line is not None:
                cost += float(line.get("cost_total") or 0.0)
                unknown += int(unknown_cost_attempts(line) > 0)
        mirix = row.get("mirix") if isinstance(row.get("mirix"), Mapping) else {}
        in_flight = (
            row.get("in_flight") if isinstance(row.get("in_flight"), Mapping) else {}
        )
        return {
            "segment": seg["segment"],
            "checkpoint": seg["checkpoint"],
            "reason": row.get("reason"),
            "in_flight": {
                k: in_flight.get(k)
                for k in ("position", "stage", "trajectory_recorded")
            },
            "calls": len(calls),
            "calls_by_role": dict(sorted(roles.items())),
            "cost_total": round(cost, 10),
            "unknown_cost_calls": unknown,
            "retrievals": len(mirix.get("retrievals") or ()),
            "runs": len(mirix.get("learning_runs") or ()),
        }

    # -- H10.1 ----------------------------------------------------------------------------

    def _received(self, t: _Trajectory) -> dict[str, Any]:
        """What the method was given for this item (``TaskView``; learning: the messages)."""
        data = t.data
        received: dict[str, Any] = {
            "task_uid": data.get("task_uid"),
            "instruction": data.get("instruction"),
            "first_messages": data.get("first_messages") or [],
        }
        if "agent_context" in data:  # older records have none
            received["agent_context"] = data.get("agent_context")
        if t.phase == "learning":
            received["messages"] = data.get("messages") or []
        return blob_placeholders(received)

    def _check_labels(self) -> None:  # noqa: C901 - one pass over every trajectory
        f = self.f
        salt = self._salt()
        env = self._env()
        corpus = Corpus()
        own: dict[str, set[str]] = defaultdict(set)
        cross: dict[str, set[str]] = defaultdict(set)
        #: Everything the method was given in the learning phase (model and tool output
        #: included): the only legitimate source of anything its memory holds.
        learned = Texts()
        taxonomy: set[str] = set()
        #: Each agent context the items' task views carried (None: an older record
        #: without one) -> the items.
        contexts: dict[str | None, list[str]] = defaultdict(list)
        for task in self.all_tasks.values():
            taxonomy |= self._taxonomy(env, task)
        for raw_id, task in self.all_tasks.items():
            labels = self._labels(env, task)
            answers = {normalise(a) for a in _answers(task)}
            for needle in labels:
                if raw_id in self.tasks:
                    own[needle].add(raw_id)
                # Category names shared by many items (ALFWorld task types, WebShop
                # product categories) are not an item's identifier: a memory may use the
                # same words.
                if (
                    needle not in answers
                    and needle not in taxonomy
                    and identifier_like(needle)
                ):
                    cross[needle].add(raw_id)
        # Discarded tries too: what the method received before a learning resume rolled
        # it back. They are not a source of what the memory holds.
        for t in [*self.trajectories, *self.discarded_trajectories]:
            data = t.data
            task = self.tasks.get(t.raw_id)
            uid = str(data.get("task_uid"))
            received = self._received(t)
            context = received.get("agent_context")
            if data.get("failure_stage") != RESET_STAGE:
                # A reset that failed handed out no task view (before_task never ran):
                # its record says nothing of what the channel carried.
                contexts[context if isinstance(context, str) else None].append(t.raw_id)
            f.ok("H10.1")
            # H5: the opaque uid (by its HMAC, never by substrings).
            uid_issue = uid_problem(uid, t.raw_id, salt)
            if uid_issue is not None:
                f.add("H10.1", uid_issue[0], uid_issue[1], item=t.raw_id)
            # Visible message shapes and label keys.
            messages = [*received["first_messages"], *received.get("messages", [])]
            for index, message in enumerate(messages):
                try:
                    shaped = visible_message(message)
                except Exception as exc:  # noqa: BLE001 - the finding says why
                    f.add(
                        "H10.1",
                        "message_shape",
                        f"message {index} is not a visible message: {exc}",
                        item=t.raw_id,
                    )
                    continue
                if shaped != message:
                    f.add(
                        "H10.1",
                        "message_shape",
                        f"message {index} carries more than its visible shape "
                        f"(keys {sorted(message)})",
                        item=t.raw_id,
                    )
            if t.phase == "learning":
                first = received["first_messages"]
                if received["messages"][: len(first)] != first:
                    f.add(
                        "H10.1",
                        "first_messages",
                        "the learning messages do not start with the first messages",
                        item=t.raw_id,
                    )
            for path in forbidden_keys(received):
                f.add("H10.1", "label_key", f"label key at {path}", item=t.raw_id)
            if task is None:
                # Without the items on this machine every item would look unknown: the
                # cell is not checked (items_unavailable), not in violation.
                if not self._items_missing:
                    f.add(
                        "H10.1",
                        "unknown_item",
                        "the trajectory's item is not in the cell's splits",
                        item=t.raw_id,
                    )
                continue
            instruction = str(received["instruction"] or "")
            problem = self._instruction_problem(task, instruction, received)
            if problem:
                f.add("H10.1", "instruction", problem, item=t.raw_id)
            # A benchmark whose labels cannot be searched for on its pages (WebShop: the
            # target's ASIN and title appear legitimately) checks the messages' structure
            # instead: no done-page text, the score line last.
            leaks = getattr(env, "leak_problems", None)
            if callable(leaks):
                shown = received.get("messages") or received["first_messages"]
                for index, why in leaks(shown):
                    f.add(
                        "H10.1",
                        "structural_leak",
                        f"message {index}: {why}",
                        item=t.raw_id,
                    )
            # The instruction (WebShop), and what the item shows by design, are blanked
            # before the search.
            statements = _statements(task) + self._visible(env, task)
            for path, text in harness_strings(received):
                corpus.add(
                    normalise(_blank(text, statements)),
                    t.raw_id,
                    f"{t.label}{path[1:]}",
                )
            if t.phase == "learning" and t.segment is None:
                for text in all_strings(received):
                    learned.add(text, t.raw_id, "learning")
        self._check_agent_context(contexts, env)
        self.info["label_strings_scanned"] = len(corpus)
        self.info["label_needles"] = len(own)
        self.info["taxonomy_labels"] = len(taxonomy)
        for needle in sorted(set(own) | set(cross)):
            for owner, where in corpus.find(needle):
                if owner in own.get(needle, ()):
                    f.add(
                        "H10.1",
                        "label_string",
                        f"{where} holds the item's own label {needle!r}",
                        item=owner,
                    )
                elif needle in cross:
                    f.add(
                        "H10.1",
                        "cross_item_label",
                        f"{where} holds {needle!r}, a label of item(s) "
                        + ", ".join(sorted(cross[needle])[:3]),
                        item=owner,
                    )
        self._check_method_outputs(cross, learned)

    def _check_agent_context(
        self, contexts: Mapping[str | None, Sequence[str]], env: Env | None
    ) -> None:
        """Every item's task view carried one and the
        same agent context, the environment's static prompt template, so the channel can
        carry nothing of any one item (its text is searched for labels with the rest of
        what the method received, :meth:`_check_labels`).

        ``agent_context_varies`` (validity): the attempt's items (discarded tries
        included; a trajectory whose reset failed received no task view and is not in
        ``contexts``) received more than one text, or some received one and others none; a
        warning when the attempt ran several harness code versions, which render it,
        and not re-verified when its versions differ in the benchmark's templates
        (:data:`WITHIN_ATTEMPT_INPUTS`). ``agent_context_not_this_tree`` (a warning): a
        text that is not the environment's as this tree renders it. Older records carry
        no agent context and are not checked (``info.agent_context`` None)."""
        recorded = {text: items for text, items in contexts.items() if text is not None}
        if not recorded:
            self.info["agent_context"] = None
            return
        f = self.f
        f.ok("H10.1")
        expected: str | None = None
        if env is not None:
            try:
                expected = env_agent_context(env)
            except Exception:  # noqa: BLE001 - not comparable; the rest still holds
                expected = None
        self.info["agent_context"] = {
            "texts": len(contexts),
            "items": sum(len(items) for items in contexts.values()),
            "chars": sorted(len(text) for text in recorded),
            "sha256": sorted(
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                for text in recorded
            ),
            "this_tree": None
            if expected is None
            else all(text == expected for text in recorded),
        }
        if len(contexts) > 1:
            counts = ", ".join(
                f"{len(items)} item(s) {'none' if text is None else f'{len(text)} chars'}"
                for text, items in sorted(
                    contexts.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))
                )
            )
            detail = (
                f"the items' task views carried {len(contexts)} different agent "
                f"contexts ({counts}): the channel must be one text for every item"
            )
            if len(self.harness_code_versions) > 1:
                f.warn(
                    "H10.1",
                    "agent_context_varies",
                    detail + "; the attempt ran "
                    f"{len(self.harness_code_versions)} harness code versions, which "
                    "render it",
                )
            else:
                f.add("H10.1", "agent_context_varies", detail)
        if expected is not None:
            for text, items in sorted(recorded.items(), key=lambda kv: kv[1][0]):
                if text != expected:
                    f.warn(
                        "H10.1",
                        "agent_context_not_this_tree",
                        f"{len(items)} item(s) received an agent context of "
                        f"{len(text)} chars that is not the {self.benchmark} "
                        f"environment's as this tree renders it ({len(expected)} "
                        "chars): a template or harness code change since",
                        item=items[0],
                    )

    def _check_method_outputs(
        self, cross: Mapping[str, set[str]], learned: Texts
    ) -> None:
        """Injections and the frozen memory hold no scorer-side identifier of any item, and
        no test item's gold answer the method did not see while learning."""
        outputs = Corpus()
        rows = read_jsonl(self.attempt_paths.injections)
        for row in rows:
            outputs.add(
                normalise(str(row.get("text") or "")),
                str(row.get("raw_id")),
                f"injection ({row.get('phase')})",
            )
        self._check_answers_in_outputs(rows, learned)
        if isinstance(self.frozen_record, dict):
            for text in memory_texts(self.frozen_record.get("snapshot")):
                outputs.add(text, "-", "frozen memory")
        self.f.ok("H10.1")
        for needle in sorted(cross):
            for owner, where in sorted(set(outputs.find(needle))):
                items = ", ".join(sorted(cross[needle])[:3])
                self.f.add(
                    "H10.1",
                    "method_output_label",
                    f"the method's {where} holds {needle!r}, a scorer-side label of "
                    + items
                    + ": the method must have received it",
                    item=None if owner == "-" else owner,
                )

    def _check_answers_in_outputs(
        self, injections: Sequence[Mapping[str, Any]], learned: Texts
    ) -> None:
        """H10.1 by the gold answer's text. A test item's gold answer that no
        learning input held (learning trajectories, model and tool output included) and
        that is not in the item's own question:

        * in the item's test-phase injection but not in the frozen memory: the method
          produced it for that item alone, so it must have read the scorer's records. A
          violation for a specific answer (``specific_answer``: identifier-like, or 4+
          characters with a digit) and for a telling text answer (``text_answer_tier``
          ``strong``: 8+ characters, two content words, such as ``scissors menthol
          fresh``); a warning for a text answer with one content word (``nashville``),
          which a template may hold;
        * in the frozen memory: a violation when a specific answer is one no memory holds
          by coincidence (``strong_in_memory``: 5+ digits, such as ``256,865``, or a path,
          key or address mark); a warning when it is short (a year, ``$1,000``, ``step
          2``: one memory serves every test item). A strong text answer is a warning while
          the memory holds those of only a few test items (documents of one source share
          names: a memory learned from the learning pages may name the organisation a
          test page names) and a violation once it holds those of more than
          :func:`memory_text_answers_allowed` test items: no coincidence explains that
          many. Names specific only through their punctuation (``r.j. reynolds``) and
          short numbers in a telling phrase are counted with them
          (``counted_in_memory``). Weak text answers are not searched in the memory.

        Text answers are compared as word sequences (``word_text``) without the common
        words around them (``text_answer_needle``: ``the coca-cola company`` as ``coca
        cola``), numeric ones also in ``digit_form`` (``256865`` holds ``256,865``), so
        case, punctuation, line breaks, separators and an article can neither hide one
        nor explain one away.
        """
        f = self.f
        test_ids = set(self.cell.get("test_items") or ())
        answers: dict[str, list[str]] = {}
        #: Text answers as searched (``text_answer_needle``), with their tier.
        texts: dict[str, dict[str, str]] = {}
        questions: dict[str, Texts] = {}
        for raw_id in test_ids:
            task = self.tasks.get(raw_id)
            if task is None:
                continue
            normalised = {n for a in _answers(task) if (n := normalise(a))}
            specific = sorted(n for n in normalised if specific_answer(n))
            needles: dict[str, str] = {}
            for n in sorted(normalised):
                found = text_answer_needle(n)
                if found is not None and needles.get(found[0]) != TEXT_STRONG:
                    needles[found[0]] = found[1]
            if specific:
                answers[raw_id] = specific
            if needles:
                texts[raw_id] = needles
            if specific or needles:
                question = Texts()
                question.add(" ".join(_statements(task)), raw_id, "question")
                questions[raw_id] = question
        # Each scanned item by its most telling answer: a specific one can be a violation
        # in the injection and, when strong_in_memory (5+ digits, a code or an address,
        # or a path/key/address mark), in the memory; a strong text one in the injection
        # and, by count, in the memory; a weak one is a warning at most.
        kinds = Counter(
            "specific"
            if raw_id in answers
            else "strong_text"
            if TEXT_STRONG in texts.get(raw_id, {}).values()
            else "weak_text"
            for raw_id in questions
        )
        self.info["gold_answers_scanned"] = {
            "test_items": len(test_ids),
            "scanned": len(questions),
            "specific": kinds["specific"],
            "strong_text": kinds["strong_text"],
            "weak_text": kinds["weak_text"],
        }

        def unexplained(raw_id: str, needle: str, textual: bool) -> bool:
            if learned.holds(needle, textual=textual):
                return False
            question = questions.get(raw_id)
            return not (question and question.holds(needle, textual=textual))

        memory = Texts()
        if isinstance(self.frozen_record, dict):
            for text in memory_texts(self.frozen_record.get("snapshot")):
                memory.add(text, "-", "frozen memory")

        def candidates(raw_id: str) -> list[tuple[str, str | None]]:
            """``(needle, tier)``: tier None for a specific answer, else its text tier."""
            found: list[tuple[str, str | None]] = [
                (n, None) for n in answers.get(raw_id, ())
            ]
            found += texts.get(raw_id, {}).items()
            return found

        f.ok("H10.1")
        reported: set[tuple[str, str]] = set()
        for row in injections:
            raw_id = str(row.get("raw_id"))
            if row.get("phase") != "test" or raw_id not in questions:
                continue
            injection = Texts()
            injection.add(str(row.get("text") or ""), raw_id, "injection")
            for needle, tier in candidates(raw_id):
                textual = tier is not None
                if (raw_id, needle) in reported or not injection.holds(
                    needle, textual=textual
                ):
                    continue
                if memory.holds(needle, textual=textual):
                    continue  # the memory's text: judged below
                if not unexplained(raw_id, needle, textual):
                    continue
                reported.add((raw_id, needle))
                detail = (
                    f"the test-phase injection of {raw_id} holds its gold answer "
                    f"{needle!r}, which neither a learning input nor the frozen "
                    "memory held: the method must have read the scorer's records"
                )
                if tier != TEXT_WEAK:
                    f.add("H10.1", "answer_in_method_output", detail, item=raw_id)
                else:
                    f.warn(
                        "H10.1",
                        "answer_in_method_output",
                        detail + " (a single content word: possibly template text)",
                        item=raw_id,
                    )
        # ``(raw_id, needle, weight)``: weight ``strong`` (a violation), ``counted``
        # (the count rule) or ``short`` (a warning).
        in_memory: list[tuple[str, str, str]] = []
        for raw_id in sorted(questions):
            for needle, tier in candidates(raw_id):
                textual = tier is not None
                if tier == TEXT_WEAK or not memory.holds(needle, textual=textual):
                    continue
                if not unexplained(raw_id, needle, textual):
                    continue
                if tier == TEXT_STRONG or counted_in_memory(needle):
                    weight = "counted"
                elif strong_in_memory(needle):
                    weight = "strong"
                else:
                    weight = "short"
                in_memory.append((raw_id, needle, weight))
        counted = sorted({r for r, _n, weight in in_memory if weight == "counted"})
        allowed = memory_text_answers_allowed(len(test_ids))
        self.info["gold_answers_in_memory"] = {
            "specific": len({r for r, _n, weight in in_memory if weight != "counted"}),
            "strong_text": len(counted),
            "strong_text_allowed": allowed,
        }
        for raw_id, needle, weight in in_memory:
            detail = (
                f"the frozen memory holds the gold answer {needle!r} of test item "
                f"{raw_id}, which no learning input held"
            )
            if weight == "strong":
                f.add("H10.1", "answer_in_method_output", detail, item=raw_id)
            elif weight == "counted" and len(counted) > allowed:
                f.add(
                    "H10.1",
                    "answer_in_method_output",
                    f"{detail}; it holds the telling text answers of "
                    f"{len(counted)} test items, more than the {allowed} a memory "
                    "learned from pages of the same source may share by coincidence",
                    item=raw_id,
                )
            else:
                f.warn(
                    "H10.1",
                    "answer_in_method_output",
                    detail
                    + (
                        " (short: possibly a coincidence)"
                        if weight == "short"
                        else " (a phrase the learning pages may have shown)"
                        if any(ch.isdigit() for ch in needle)
                        else " (a name the learning pages may have shown)"
                    ),
                    item=raw_id,
                )

    def _salt_quiet(self) -> bytes | None:
        try:
            return self.paths.salt.read_bytes()
        except OSError:
            return None

    def _salt(self) -> bytes | None:
        try:
            return self.paths.salt.read_bytes()
        except OSError:
            self.f.add(
                "H10.1",
                "salt_missing",
                f"{self.paths.salt} is unreadable: the uids cannot be checked",
            )
            return None

    def _env(self) -> Env | None:
        try:
            return self.envs.env(self.benchmark)
        except Exception as exc:  # noqa: BLE001 - reported by _load_items already
            self._items_missing = True
            self.f.not_checked(
                "H10.1",
                "env_unavailable",
                f"this machine cannot build the {self.benchmark} environment, so the "
                f"checks that need it were not run: {exc}",
            )
            return None

    @staticmethod
    def _taxonomy(env: Env | None, task: Task) -> set[str]:
        labels = getattr(env, "taxonomy_labels", None)
        return (
            {n for s in labels(task) if (n := normalise(str(s)))}
            if callable(labels)
            else set()
        )

    @staticmethod
    def _visible(env: Env | None, task: Task) -> list[str]:
        visible = getattr(env, "visible_by_design", None)
        return [str(s) for s in visible(task) if s] if callable(visible) else []

    @staticmethod
    def _labels(env: Env | None, task: Task) -> set[str]:
        strings = getattr(env, "label_strings", None)
        labels = [str(s) for s in strings(task)] if callable(strings) else []
        # H5: the raw id is a label of its item whatever ``label_strings``
        # lists: no method may receive it as a whole token.
        labels.append(task.raw_id)
        visible = getattr(env, "visible_by_design", None)
        allowed = {normalise(s) for s in visible(task)} if callable(visible) else set()
        return {n for s in labels if (n := normalise(s)) and n not in allowed}

    def _instruction_problem(
        self, task: Task, instruction: str, received: Mapping[str, Any]
    ) -> str | None:
        if self.benchmark == "alfworld":
            first = received.get("first_messages") or []
            text = _message_text(first[0]) if first else ""
            marker = _alfworld_templates().task_marker
            _, found, rest = text.partition(marker)
            sentence = rest.split("\n", 1)[0].strip() if found else None
            if not instruction or instruction != sentence:
                return (
                    "the instruction is not the game's task sentence of the first "
                    "observation"
                )
        elif self.benchmark == "webshop":
            # The first message is the raw search page, which states it.
            first = received.get("first_messages") or []
            parts = (_message_text(first[0]) if first else "").split(
                _webshop_templates().observation_separator
            )
            marker = _webshop_templates().instruction_marker
            on_page = parts[2] if len(parts) > 2 and parts[1] == marker else None
            if not instruction or instruction != on_page:
                return "the instruction is not the one the first page states"
            frozen = task.raw.get("instruction_text")
            # The page text is stripped (upstream's text mode); the frozen text may end
            # in a space when it has no price clause.
            if isinstance(frozen, str) and instruction != frozen.strip():
                return "the instruction is not the frozen manifest's"
        return None

    # -- H10.2 ----------------------------------------------------------------------------

    def _check_prompts(self) -> None:
        f = self.f
        first_prompts: dict[str, set[str]] = defaultdict(set)
        for t in [*self.trajectories, *self.discarded_trajectories]:
            data = t.data
            calls = data.get("calls") or []
            injection = data.get("injection") or {}
            memory = str(injection.get("text") or "")
            memory_sha = hashlib.sha256(memory.encode("utf-8")).hexdigest()
            if not calls:
                continue
            f.ok("H10.2")
            no_memory = data.get("first_prompt_no_memory_sha256")
            if no_memory:
                first_prompts[t.raw_id].add(str(no_memory))
            conversation = self.benchmark == "webshop"
            previous: list[Any] | None = None
            for index, call in enumerate(calls):
                if call.get("memory_sha256") != memory_sha:
                    f.add(
                        "H10.2",
                        "memory_changed",
                        f"actor call {index} carried another memory than the "
                        "episode's injection",
                        item=t.raw_id,
                    )
                later = (
                    self._check_later_call(t, index, call, memory) if index else None
                )
                if conversation:
                    messages = later if index else _call_messages(call, self.blob_dir)
                    self._check_append_only(
                        t,
                        index,
                        previous,
                        messages,
                        calls[index - 1] if index else None,
                    )
                    previous = messages
            try:
                request = internalize_blobs(
                    calls[0].get("request") or {}, self.blob_dir
                )
            except (OSError, ValueError) as exc:
                f.add(
                    "H10.2",
                    "request_unreadable",
                    f"the first request cannot be restored: {exc}",
                    item=t.raw_id,
                )
                continue
            extra = sorted(set(request) - ACTOR_REQUEST_FIELDS)
            if extra:
                f.add(
                    "H10.2",
                    "request_fields",
                    f"the actor sent fields beyond model/messages: {extra}",
                    item=t.raw_id,
                )
            if request.get("model") != self.model:
                f.add(
                    "H10.2",
                    "request_model",
                    f"the actor asked for {request.get('model')!r}, the cell's model is "
                    f"{self.model}",
                    item=t.raw_id,
                )
            if request_fingerprint(request) != data.get("first_prompt_sha256"):
                f.add(
                    "H10.2",
                    "first_prompt_hash",
                    "the recorded first-prompt sha256 is not the first request's",
                    item=t.raw_id,
                )
            stripped, why = strip_memory(
                self.benchmark, request.get("messages") or [], memory
            )
            if stripped is None:
                f.add("H10.2", "memory_slot", str(why), item=t.raw_id)
            elif request_fingerprint({**request, "messages": stripped}) != no_memory:
                f.add(
                    "H10.2",
                    "outside_memory_slot",
                    "without the injection the first prompt is not the no-memory "
                    "prompt: something besides the memory slot differs",
                    item=t.raw_id,
                )
        self.info["first_prompts"] = {k: sorted(v) for k, v in first_prompts.items()}
        for raw_id, hashes in first_prompts.items():
            if len(hashes) > 1:
                f.add(
                    "H10.2",
                    "first_prompt_varies",
                    f"{len(hashes)} different no-memory first prompts for one item",
                    item=raw_id,
                )

    def _check_later_call(
        self, t: _Trajectory, index: int, call: Mapping[str, Any], memory: str
    ) -> list[Any] | None:
        """Actor call ``index`` > 0: its recorded request is the one the gateway hashed
        (checked below for every call) and carries the injection in its memory slot.
        Returns the request's messages, or None when it cannot be restored."""
        try:
            request = internalize_blobs(call.get("request") or {}, self.blob_dir)
        except (OSError, ValueError) as exc:
            self.f.add(
                "H10.2",
                "request_unreadable",
                f"actor call {index}'s request cannot be restored: {exc}",
                item=t.raw_id,
            )
            return None
        messages = list(request.get("messages") or [])
        stripped, why = strip_memory(self.benchmark, messages, memory)
        if stripped is None:
            self.f.add(
                "H10.2",
                "memory_slot",
                f"actor call {index}: {why}",
                item=t.raw_id,
            )
        return messages

    def _check_append_only(
        self,
        t: _Trajectory,
        index: int,
        previous: Sequence[Any] | None,
        messages: Sequence[Any] | None,
        replied: Mapping[str, Any] | None = None,
    ) -> None:
        """WebShop: actor call 0 is the conversation's first
        message alone, and call ``index``'s messages are call ``index - 1``'s followed by
        exactly one assistant and one user message: the conversation only grows, nothing
        sent before is changed, dropped or re-rendered. The assistant message is the reply
        call ``index - 1`` (``replied``) got, as its recorded response holds it
        (:func:`_webshop_reply_turn`), and the user message is ``rollout_turn.md`` for
        step ``index + 1`` of the task the first message states. A request that cannot be
        restored is reported where it is read."""
        if index == 0 and messages is not None and len(messages) != 1:
            self.f.add(
                "H10.2",
                "conversation_not_append_only",
                f"actor call 0 sent {len(messages)} messages, not the conversation's "
                "first message alone",
                item=t.raw_id,
            )
            return
        if previous is None or messages is None:
            return
        size = len(previous)
        if (
            len(messages) != size + 2
            or list(messages[:size]) != list(previous)
            or [
                m.get("role") if isinstance(m, Mapping) else None
                for m in messages[size:]
            ]
            != ["assistant", "user"]
        ):
            self.f.add(
                "H10.2",
                "conversation_not_append_only",
                f"actor call {index}'s messages are not call {index - 1}'s followed by "
                "one assistant and one user message (WebShop's conversation is "
                "append-only)",
                item=t.raw_id,
            )
            return
        said, turn = (
            m.get("content") if isinstance(m, Mapping) else None
            for m in messages[size:]
        )
        reply = _webshop_reply_turn(replied, self.blob_dir) if replied else None
        problems = []
        if reply is not None and said != reply:
            problems.append(
                f"its assistant message is not the reply call {index - 1} got (as "
                "its recorded response holds it, or (no reply) for none)"
            )
        first = messages[0].get("content") if isinstance(messages[0], Mapping) else None
        problem = _webshop_turn_problem(turn, first, index + 1)
        if problem:
            problems.append(problem)
        if problems:
            self.f.add(
                "H10.2",
                "conversation_not_append_only",
                f"actor call {index} appended an exchange the episode did not have: "
                f"{'; '.join(problems)}",
                item=t.raw_id,
            )

    def _check_request_hashes(self) -> None:
        """Every actor call's recorded request, rewritten as the gateway rewrites it
        (``rewrite_chat_body`` with the ledger line's ``wire`` values), hashes to the
        ledger's ``request_sha256``: the trajectory records the audit reads are the
        requests the gateway actually forwarded."""
        f = self.f
        for t in [*self.trajectories, *self.discarded_trajectories]:
            for index, call in enumerate(t.data.get("calls") or []):
                line = self.by_call_id.get(str(call.get("gateway_call_id")))
                wire = (line or {}).get("wire") or {}
                digest = (line or {}).get("request_sha256")
                if not line or not digest or "max_tokens" not in wire:
                    continue  # not answered / refused before the rewrite, or no ledger
                f.ok("H10.2")
                try:
                    request = internalize_blobs(
                        call.get("request") or {}, self.blob_dir
                    )
                except (OSError, ValueError):
                    continue  # reported by the prompt checks
                if sent_request_sha256(request, wire) != digest:
                    f.add(
                        "H10.2",
                        "request_not_as_sent",
                        f"actor call {index}'s recorded request is not the request the "
                        f"gateway forwarded (ledger call {line.get('call_id')})",
                        item=t.raw_id,
                    )

    # -- H10.3 ----------------------------------------------------------------------------

    def _check_gateway(self) -> None:  # noqa: C901 - one pass over the ledger
        f = self.f
        policy = ROLE_POLICY.get(self.method)
        if policy is None:
            f.add(
                "H10.3", "unknown_method", f"no role policy for method {self.method!r}"
            )
            policy = {}
        if not self.ledger_path.is_file():
            f.add("H10.3", "ledger_missing", f"{self.ledger_path} is missing")
        referenced: dict[str, _Trajectory] = {}
        for t in [*self.trajectories, *self.discarded_trajectories]:
            for call in t.data.get("calls") or []:
                call_id = call.get("gateway_call_id")
                f.ok("H10.3")
                if not call_id:
                    if call.get("http_status") is not None:
                        f.add(
                            "H10.3",
                            "actor_call_unlinked",
                            "an actor call the gateway answered carries no call id",
                            item=t.raw_id,
                        )
                    continue
                referenced[str(call_id)] = t
                line = self.by_call_id.get(str(call_id))
                if line is None:
                    f.add(
                        "H10.3",
                        "actor_call_not_in_ledger",
                        f"actor call {call_id} is not in {self.ledger_path.name}",
                        item=t.raw_id,
                    )
                    continue
                expected = {
                    "role": "actor",
                    "item": t.data.get("task_uid"),
                    "phase": t.phase,
                }
                seen = {
                    "role": line.get("role"),
                    "item": line.get("item"),
                    "phase": self._phase(line),
                }
                wrong = {k: seen[k] for k, v in expected.items() if seen[k] != v}
                if wrong:
                    f.add(
                        "H10.3",
                        "actor_call_mismatch",
                        f"ledger line {call_id} has {wrong}, expected {expected}",
                        item=t.raw_id,
                    )
        repeats: Counter[str] = Counter()
        #: The repeats that are recorded as problems, by role.
        flagged: Counter[str] = Counter()
        repeats_of_discarded: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        families: dict[str, set[str]] = defaultdict(set)
        served: dict[str, Counter[str]] = defaultdict(Counter)
        # The roles and embedding model of the attempt's own versions when recorded,
        # else this tree's (then not re-verified where they differ).
        recorded_roles = self._attempt_constant("roles")
        roles = (
            {str(r) for value in recorded_roles for r in value}
            if recorded_roles
            else set(protocol.ROLES)
        )
        embedding_models = [
            str(m) for m in self._attempt_constant("embedding_model") or ()
        ] or [protocol.EMBEDDING_MODEL]
        for row in self.calls:
            f.ok("H10.3")
            role, phase = row.get("role"), self._phase(row)
            call_id = str(row.get("call_id"))
            status = row.get("status")
            tagged = (
                row.get("phase_header") if "phase_header" in row else row.get("phase")
            )
            if tagged is not None and phase is not None and tagged != phase:
                # The caller tagged the call with another phase than the run was in.
                f.add(
                    "H10.3",
                    "phase_mismatch",
                    f"call {call_id} ({role}) is tagged {tagged!r} but the run was in "
                    f"its {phase!r} phase",
                )
            if role not in roles:
                f.add(
                    "H10.3", "role", f"call {call_id} has no protocol role ({role!r})"
                )
                continue
            if status == "rejected":
                kind = str((row.get("error") or {}).get("type"))
                rejected[kind] += 1
                if kind in PROTOCOL_REJECTIONS:
                    f.add(
                        "H10.3",
                        "protocol_rejection",
                        f"the gateway refused call {call_id} ({role}, {phase}): {kind}",
                    )
                continue
            allowed = policy.get(str(phase))
            if allowed is None:
                f.add(
                    "H10.3",
                    "phase",
                    f"call {call_id} ({role}) has phase {phase!r}",
                )
            elif role not in allowed:
                f.add(
                    "H10.3",
                    "role_not_allowed",
                    f"call {call_id}: {self.method} does not use role {role!r} in the "
                    f"{phase} phase",
                )
            models = [self.model] if role in CHAT_ROLES else embedding_models
            if row.get("model") not in models:
                # The embedding model is a protocol constant (its versions' own when
                # recorded, else this tree's); a chat role's is the cell's own.
                f.add(
                    "H10.3",
                    "model" if role in CHAT_ROLES else "embedding_model",
                    f"call {call_id} ({role}) used {row.get('model')!r}, not "
                    f"{' / '.join(map(str, models))}",
                )
            if status == "ok" and row.get("model_served"):
                key = str(row.get("model"))
                families[key].add(model_family(str(row["model_served"])))
                served[key][str(row["model_served"])] += 1
            if (
                role == "actor"
                and call_id not in referenced
                and call_id not in self.discarded  # a rolled-back try
            ):
                if not (phase == "test" and row.get("item") in self.rerun_uids):
                    f.add(
                        "H10.3",
                        "unaccounted_actor_call",
                        f"actor call {call_id} ({phase}, item {row.get('item')}) "
                        "belongs to no trajectory and to no H8 re-run",
                    )
            if row.get("repeat_of"):
                repeats[str(role)] += 1
                if self._redo_of_discarded(row):
                    repeats_of_discarded[str(role)] += 1
                problem = self._repeat_problem(row)
                if problem:
                    flagged[str(role)] += 1
                    f.add("H10.3", "repeat_of", problem)
        started = {
            str(r.get("call_id"))
            for r in self.ledger
            if r.get("event") == "call_started"
        }
        unfinished = started - set(self.by_call_id)
        if unfinished:
            f.warn(
                "H10.3",
                "call_without_final_line",
                f"{len(unfinished)} call(s) started without a final ledger line (a "
                "gateway that stopped mid-call; their cost is unknown)",
            )
        # A family nothing recorded ever requested the model as is a violation whatever
        # the version; only a family an unrecorded version's slug may have requested is
        # not re-verified, for that finding alone.
        unsure: list[tuple[str, list[str], frozenset[str], list[str]]] = []
        for model, names in sorted(families.items()):
            expected, missing, recorded_any = self._requested_families(model)
            other = sorted(names - expected)
            strict = sorted(set(other) - recorded_any) if missing else other
            if strict:
                # Every ok call of the cell must be the requested model: a cell served
                # entirely as another one is as wrong as a mix.
                f.add(
                    "H10.3",
                    "model_served",
                    f"OpenRouter served {model} as {', '.join(strict)}, not "
                    f"{' / '.join(sorted(expected))} ({dict(served[model])}; H1.8)",
                )
            elif len(names) > 1:
                f.add(
                    "H10.3",
                    "model_served",
                    f"OpenRouter served {model} as several model families "
                    f"{dict(served[model])} (H1.8)",
                )
            elif other:
                unsure.append((model, other, expected, missing))
        for model, other, expected, missing in unsure:
            inputs = f.partly_not_reverified.setdefault(MODEL_SERVED_CHECK, [])
            inputs += [m for m in missing if m not in inputs]
            f.add(
                "H10.3",
                "model_served",
                f"OpenRouter served {model} as {', '.join(other)}, not "
                f"{' / '.join(sorted(expected))} ({dict(served[model])}; H1.8): a "
                "canonical slug some recorded config/models.json gives the model, and "
                "the attempt ran under a version whose slugs are not recorded",
                unverified=True,
            )
        self.info["repeat_of"] = dict(repeats)
        self.info["repeat_of_flagged"] = dict(flagged)
        if repeats_of_discarded:
            self.info["repeat_of_discarded"] = dict(repeats_of_discarded)
        self.info["rejected"] = dict(rejected)
        self.info["model_served"] = {k: dict(v) for k, v in served.items()}
        self.info["ledger_calls"] = len(self.calls)
        self._check_learning_infra()
        if self.method == "mirix":
            self._check_mirix_test_infra()
            self._reconcile_mirix()
            self._report_accepted_learning_calls()

    def _unserved_unbacked(self, t: _Trajectory, refused_before: int) -> str | None:
        """Why a row flagged ``provider_unserved`` is not what the ledger shows, None
        when it is: the trajectory's last actor call is in the ledger,
        refused upstream with a class the phase scores unanswered (any of
        ``REFUSAL_CLASSES`` in the learning phase, a no-endpoint 404 in the test phase),
        and a test item's earlier try was left unscored by the same refusal
        (``PROVIDER_UNSERVED_TRIES``)."""
        calls = [c for c in t.data.get("calls") or [] if isinstance(c, Mapping)]
        if not calls:
            return "its trajectory records no actor call"
        call_id = calls[-1].get("gateway_call_id")
        line = self.by_call_id.get(str(call_id)) if call_id else None
        if line is None:
            return f"its last actor call ({call_id}) is not in {self.ledger_path.name}"
        failure = ledger_line_failure(line)
        refusal = upstream_refusal(failure) if failure is not None else None
        allowed = REFUSAL_CLASSES if t.phase == "learning" else (REFUSAL_NO_ENDPOINT,)
        if refusal not in allowed:
            return (
                f"the ledger records its last actor call {call_id} as "
                f"{line.get('status')} (HTTP {line.get('http_status')}), not an upstream "
                f"refusal the {t.phase} phase scores unanswered ("
                + ", ".join(allowed)
                + ")"
            )
        need = max(1, PROVIDER_UNSERVED_TRIES - 1)
        if t.phase == "test" and refused_before < need:
            return (
                f"results.jsonl holds {refused_before} earlier try left unscored by a "
                f"no-endpoint refusal, not {need} (a test item is scored unanswered "
                f"only on try {PROVIDER_UNSERVED_TRIES} of consecutive such refusals)"
            )
        return None

    def _requested_families(
        self, model: str
    ) -> tuple[frozenset[str], list[str], frozenset[str]]:
        """What OpenRouter may serve ``model`` as under the attempt's own versions
        (H1.8): its id's family and the family of the ``canonical_slug`` each
        version's ``config/models.json`` gave it, found by the file's hash in the
        version's file map (this tree's file, the frozen record's ``model_slugs``, the
        cell's own ``version.json``). Returns those families, the versions whose slugs are
        not recorded (as the not-re-verified list names them) and the families of every
        slug any recorded models.json gives the model."""
        tree = self.tree
        index: dict[str, Mapping[str, str]] = dict(
            protocol.recorded_model_slugs(tree)
            if "model_slugs" in tree
            else protocol.model_slugs_of(tree)
        )
        for version in self.diff_versions.values():
            for sha, slugs in protocol.recorded_model_slugs(version).items():
                index.setdefault(sha, slugs)
        for sha, slugs in self._cell_model_slugs.items():
            index.setdefault(sha, slugs)
        families = {model_family(model)}
        missing: list[str] = []
        for fingerprint in self.attempt_versions or [None]:
            version = self.diff_versions.get(fingerprint) if fingerprint else None
            files = (version or {}).get("files")
            sha = (
                files.get(protocol.MODELS_JSON_REL)
                if isinstance(files, Mapping)
                else None
            )
            slugs = index.get(str(sha)) if sha else None
            if slugs is None:
                missing.append(
                    f"{protocol.MODELS_JSON_REL} canonical_slug of {model} under "
                    f"protocol {str(fingerprint)[:12]} (not recorded)"
                )
            elif slugs.get(model):
                families.add(model_family(slugs[model]))
        recorded_any = frozenset(
            model_family(slugs[model]) for slugs in index.values() if slugs.get(model)
        )
        return frozenset(families), missing, recorded_any

    def _redo_of_discarded(self, row: Mapping[str, Any]) -> bool:
        """Whether a ``repeat_of`` line repeats a call a learning resume discarded, from a
        later segment or the live stream: the redo of a rolled-back try,
        never a retry within one try."""
        original = self.discarded.get(str(row.get("repeat_of")))
        if original is None:
            return False
        return self.discarded.get(str(row.get("call_id"))) != original

    def _repeat_problem(self, row: Mapping[str, Any]) -> str | None:
        """None when a ``repeat_of`` line is allowed, else why it is not."""
        role, phase, item = row.get("role"), self._phase(row), row.get("item")
        original = self.by_call_id.get(str(row.get("repeat_of")))
        if original is not None and original.get("item") != item:
            return None  # another item sent the same request: not a retry
        if self._redo_of_discarded(row):
            return None  # the redo of a try a learning resume rolled back (every role)
        if role == "mirix" or (role == "embedding" and self.method == "mirix"):
            return None  # MIRIX's application-level retries
        if (
            role in ("actor", "embedding")
            and phase == "test"
            and item in self.rerun_uids
        ):
            return None  # an H8 test-phase re-run of this item
        return (
            f"call {row.get('call_id')} ({role}, {phase}, item {item}) repeats failed call "
            f"{row.get('repeat_of')}: {role} calls are never retried"
        )

    def _check_learning_infra(self) -> None:
        """H8 in the learning phase: no infrastructure failure on the ``mirix`` or
        ``embedding`` role that no retry recovered (the caller swallowed it, and the
        memory is not what the attempt would have learned without the outage). A call a
        learning resume discarded is left out: the resume rolled its effect back, and the
        redo made it again."""
        lost = unrecovered_infra_calls(
            self._live_ledger(),
            roles=("mirix", "embedding"),
            phases=("setup", "learning"),
        )
        self.f.ok("H10.3")
        accepted = self._accepted_calls()
        for line in lost:
            error = line.get("error") if isinstance(line.get("error"), Mapping) else {}
            record = accepted.get(str(line.get("call_id")))
            self.f.add(
                "H10.3",
                "learning_infra_unrecovered",
                f"call {line.get('call_id')} ({line.get('role')}, learning) failed on "
                f"infrastructure (HTTP {line.get('http_status')}, {error.get('type')}) "
                "and no retry recovered it"
                + (
                    f": accepted after {record.get('rollbacks')} rollbacks of checkpoint "
                    f"{record.get('checkpoint')} (attempt.json accepted_learning_calls)"
                    if record is not None
                    else ": no accepted_learning_calls record of attempt.json covers it"
                ),
            )

    def _accepted_calls(self) -> dict[str, Mapping[str, Any]]:
        """``attempt.json`` ``accepted_learning_calls`` by call id (run.py: the lost
        calls of a MIRIX batch accepted after its checkpoint was rolled back
        ``failures.MIRIX_BATCH_ROLLBACKS`` times)."""
        return {
            str(record["call_id"]): record
            for record in self.state.get("accepted_learning_calls") or ()
            if isinstance(record, Mapping) and record.get("call_id")
        }

    def _report_accepted_learning_calls(self) -> None:
        """MIRIX learning calls lost and accepted after the rollbacks: one
        ``mirix_learning_infra`` warning per accepted batch naming its calls, next to the
        run problems ``_reconcile_mirix`` records (H10.3-mirix-reconcile)."""
        batches: dict[tuple[str, str], list[str]] = defaultdict(list)
        for call_id, record in self._accepted_calls().items():
            key = (str(record.get("checkpoint")), str(record.get("rollbacks")))
            batches[key].append(call_id)
        for (checkpoint, rollbacks), calls in sorted(batches.items()):
            self.f.warn(
                "H10.3",
                "mirix_learning_infra",
                f"{len(calls)} lost MIRIX learning call(s) accepted after {rollbacks} "
                f"rollbacks of checkpoint {checkpoint} (attempt.json "
                f"accepted_learning_calls): {_some(sorted(calls), 10)}",
            )

    def _check_counted_failures(self) -> None:
        """Failures the harness recorded and went on from, from
        ``failures.jsonl`` of the audited attempt: every ``(kind, recovery)`` pair outside
        :data:`LEGACY_RECOVERIES` (a provider-unserved item scored 0 and the cell went
        on, a MIRIX batch accepted after its rollbacks, ...) is one warning per
        ``(phase, role, code, kind, recovery)``, ``failure_counted_and_continued``
        (``info.counted_failures``). Refusals a method answered inside itself (learner
        and judge refusals, AWM inductions) have no failures row: they are read from the
        method records (``cell_records.tolerated_failures``, ``info.tolerated``)."""
        groups: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
        for row in read_jsonl(self.attempt_paths.failures):
            if row.get("attempt") not in (None, self.number):
                continue
            kind, recovery = str(row.get("kind")), str(row.get("recovery"))
            if (kind, recovery) in LEGACY_RECOVERIES:
                continue
            key = (
                str(row.get("phase")),
                str(row.get("role") or "-"),
                str(row.get("code")),
                kind,
                recovery,
            )
            groups[key].append(str(row.get("item") or "-"))
        counted = []
        for (phase, role, code, kind, recovery), items in sorted(groups.items()):
            counted.append(
                {
                    "phase": phase,
                    "role": role,
                    "code": code,
                    "kind": kind,
                    "recovery": recovery,
                    "count": len(items),
                    "items": sorted(set(items)),
                }
            )
            self.f.warn(
                "H10.3",
                "failure_counted_and_continued",
                f"{len(items)} {phase}-phase {role} failure(s) ({kind} {code}) were "
                f"recorded and the cell went on ({recovery.replace('_', ' ')}), not "
                f"stopped: items {_some(sorted(set(items)))}",
            )
        self.info["counted_failures"] = counted
        try:
            self.info["tolerated"] = cell_records.tolerated_failures(self.attempt_paths)
        except Exception as exc:  # noqa: BLE001 - a record summary, never a verdict
            self.info["tolerated"] = {"error": f"{type(exc).__name__}: {exc}"}

    def _live_ledger(self) -> list[dict[str, Any]]:
        """The ledger without the calls a learning resume discarded."""
        if not self.discarded:
            return self.ledger
        return [r for r in self.ledger if str(r.get("call_id")) not in self.discarded]

    def _check_mirix_test_infra(self) -> None:
        """H8 in a MIRIX test phase, per item.

        MIRIX's retrieval falls back silently when it loses its topic extraction
        (``mirix`` role: it searches with the raw text) or its query vector
        (``embedding``: BM25 alone, no gate). The method reads each retrieval's log row
        and refuses to inject a result degraded by anything but the model, so the item is
        not scored and H8 re-runs it; it keeps the row's ``evidence`` in
        ``mirix_retrievals.jsonl``. Here:

        * every scored test item was scored on a recorded retrieval (the last one of its
          task uid before its scored row) whose evidence the method's own
          ``retrieval_problem`` finds clean (``mirix_degraded_retrieval_scored``,
          ``mirix_retrieval_unrecorded``, ``mirix_retrieval_unchecked``);
        * every test-phase ``mirix`` / ``embedding`` call that failed on infrastructure
          and that no retry recovered falls inside the window (``started_at``..``at``) of
          a retrieval the method refused (``mirix_test_infra_unrecovered``): anything
          else is a failure MIRIX swallowed without its log showing it;
        * a learning-phase retrieval the method refused means the attempt should have
          restarted (``mirix_learning_retrieval_degraded``).

        Other methods are not checked here: ReasoningBank's query embedding raises, and
        H8 re-runs the item (its failed line may stay unrecovered when the re-run comes
        later than the gateway's repeat window)."""
        from evals.skill_bench.methods.mirix import (
            RETRIEVAL_RECORD_FILE,
            retrieval_problem,
        )

        f = self.f
        f.ok("H10.3")
        records = [
            r
            for r in read_jsonl(self.attempt_paths.method_dir / RETRIEVAL_RECORD_FILE)
            if r.get("event") == "retrieval"
        ]
        refused_windows: list[tuple[datetime, datetime]] = []
        by_uid: dict[str, list[tuple[datetime, Mapping[str, Any]]]] = defaultdict(list)
        refused_learning = 0
        for record in records:
            evidence = record.get("evidence")
            problem = (
                retrieval_problem(evidence) if isinstance(evidence, Mapping) else None
            )
            refused = bool(record.get("failure")) or problem is not None
            if not record.get("frozen"):
                if refused:
                    refused_learning += 1
                    f.add(
                        "H10.3",
                        "mirix_learning_retrieval_degraded",
                        f"learning retrieval {record.get('retrieval_id')} of "
                        f"{record.get('task_uid')} lost its "
                        f"{(problem.role if problem else 'infrastructure')}: recorded",
                    )
                continue
            start, end = _ts(record.get("started_at")), _ts(record.get("at"))
            if refused and start is not None and end is not None:
                refused_windows.append((start, end))
            if end is not None:
                by_uid[str(record.get("task_uid"))].append((end, record))
        for rows in by_uid.values():
            rows.sort(key=lambda pair: pair[0])
        scored = 0
        degraded_scored = 0
        excused_windows: list[tuple[datetime, datetime]] = []
        for row in read_jsonl(self.attempt_paths.results):
            if row.get("score") is None or row.get("phase", "test") != "test":
                continue
            scored += 1
            uid, at = str(row.get("task_uid")), _ts(row.get("at"))
            earlier = [
                record
                for stamp, record in by_uid.get(uid, ())
                if at is None or stamp <= at
            ]
            if not earlier:
                f.add(
                    "H10.3",
                    "mirix_retrieval_unrecorded",
                    "the item was scored, but the MIRIX method recorded no retrieval "
                    "for it: whether its retrieval lost its extraction or query vector "
                    "is unknown",
                    item=str(row.get("raw_id")),
                )
                continue
            record = earlier[-1]
            evidence = record.get("evidence")
            if row.get("retrieval_degraded") is True:
                # Recorded: the method injected the third
                # consecutive degraded retrieval of the item and flagged the row.
                degraded_scored += 1
                start, end = _ts(record.get("started_at")), _ts(record.get("at"))
                if start is not None and end is not None:
                    excused_windows.append((start, end))
                f.warn(
                    "H10.3",
                    "mirix_degraded_retrieval_scored",
                    f"scored on retrieval {record.get('retrieval_id')} after the "
                    "method's retry limit, flagged retrieval_degraded",
                    item=str(row.get("raw_id")),
                )
                continue
            if not isinstance(evidence, Mapping):
                f.add(
                    "H10.3",
                    "mirix_retrieval_unchecked",
                    f"retrieval {record.get('retrieval_id')} has no log-row evidence: "
                    "whether it lost its extraction or query vector is unknown",
                    item=str(row.get("raw_id")),
                )
                continue
            problem = retrieval_problem(evidence)
            if problem is not None or record.get("failure"):
                why = problem.detail if problem else str(record.get("failure"))
                f.add(
                    "H10.3",
                    "mirix_degraded_retrieval_scored",
                    f"scored on retrieval {record.get('retrieval_id')}, which lost "
                    f"its {problem.role if problem else 'infrastructure'} to something "
                    f"other than the model ({why[:300]}): H8 re-runs such an item",
                    item=str(row.get("raw_id")),
                )
        lost = unrecovered_infra_calls(
            self.ledger, roles=("mirix", "embedding"), phases=("test",)
        )
        unexplained = 0
        for line in lost:
            stamp = _ts(line.get("ts"))
            if stamp is not None and any(
                a <= stamp <= b for a, b in [*refused_windows, *excused_windows]
            ):
                # A retrieval the method refused (its item was re-run, H8), or one it
                # injected flagged after its retry limit.
                continue
            unexplained += 1
            error = line.get("error") if isinstance(line.get("error"), Mapping) else {}
            item = f", item {line['item']}" if line.get("item") else ""
            f.add(
                "H10.3",
                "mirix_test_infra_unrecovered",
                f"call {line.get('call_id')} ({line.get('role')}, test, at "
                f"{line.get('ts')}{item}) failed on infrastructure (HTTP "
                f"{line.get('http_status')}, {error.get('type')}) and no retry recovered "
                "it, outside every retrieval the MIRIX method refused to score: MIRIX "
                "swallowed it without its retrieval log showing it",
            )
        self.info["mirix_test_retrievals"] = {
            "retrievals": sum(len(v) for v in by_uid.values()),
            "refused": len(refused_windows),
            "refused_learning": refused_learning,
            "scored_items_checked": scored,
            "lost_calls": len(lost),
            "lost_calls_unexplained": unexplained,
            "degraded_scored": degraded_scored,
        }

    def _reconcile_mirix(self) -> None:
        """MIRIX's own learning record against the gateway ledger, both ways.

        MIRIX runs one learning run at a time per scope and ``wait_until_idle`` keeps the
        learning items from overlapping a run, so every ``mirix``-role call of the
        learning phase falls either inside a run's ``started_at``..``finished_at`` (that
        run's distillation and curation) or outside every run (the retrieval of a
        learning item: at most one topic-extraction call each):

        * each run's reported ``llm_usage`` (calls and tokens) must not exceed the ledger
          calls in its window (more: calls that bypassed the gateway);
        * the calls outside every run must not outnumber the learning retrievals
          (``mirix_retrievals.jsonl``): anything else is unexplained traffic;
        * a failed run, a session deferred on infrastructure, a curator batch stopped by
          it, or a learning item's session left unsettled at the freeze is a violation;
        * an exception a curator tool raised while it ran that was not the model's own
          mistake (MIRIX's own error or an infrastructure one, which MIRIX answered to the
          model as a failed tool result) is a warning
          (``mirix_curator_tool_exception``).
        """
        from evals.skill_bench.methods.mirix import (
            RETRIEVAL_RECORD_FILE,
            learning_problems,
            learning_warnings,
            unsettled_sessions,
        )

        f = self.f
        snapshot = (self.frozen_record or {}).get("snapshot") or {}
        runs = [
            r for r in snapshot.get("learning_runs") or () if isinstance(r, Mapping)
        ]
        f.ok("H10.3")
        for problem in learning_problems(runs):
            f.add("H10.3", "mirix_learning_infra", problem)
        for warning in learning_warnings(runs):
            # An exception inside a curator tool, answered to the model as a failed
            # tool result; the attempt stays valid, the finding says what it met.
            f.warn("H10.3", "mirix_curator_tool_exception", warning)
        sessions = sorted(
            str(t.data.get("task_uid"))
            for t in self.trajectories
            if t.phase == "learning"
        )
        for sid, why in sorted(unsettled_sessions(runs, sessions).items()):
            f.add(
                "H10.3",
                "mirix_session_unsettled",
                f"learning session {sid} was not learned before the freeze ({why}): a "
                "sweep in the test phase could still learn from it (H10.4)",
            )
        calls = [
            row
            for row in self.calls
            if row.get("role") == "mirix"
            and row.get("status") != "rejected"
            and self._phase(row) in ("setup", "learning")
            and str(row.get("call_id")) not in self.discarded
        ]
        self._reconcile_discarded_mirix()
        windows = []
        for run in runs:
            start, end = _ts(run.get("started_at")), _ts(run.get("finished_at"))
            if start is None or end is None:
                if run.get("status") in ("succeeded", "failed") and run.get(
                    "llm_usage"
                ):
                    f.add(
                        "H10.3",
                        "mirix_run_unwindowed",
                        f"learning run {run.get('id')} reports usage but no "
                        "started_at/finished_at: its calls cannot be matched to the ledger",
                    )
                continue
            windows.append((start, end, run))
        in_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        outside: list[Mapping[str, Any]] = []
        for row in calls:
            stamp = _ts(row.get("ts"))
            owner = next(
                (r for a, b, r in windows if stamp is not None and a <= stamp <= b),
                None,
            )
            if owner is None:
                outside.append(row)
            else:
                in_run[str(owner.get("id"))].append(row)
        reported: Counter[str] = Counter()
        seen: Counter[str] = Counter()
        with_usage = 0
        for _, _, run in windows:
            usage = run.get("llm_usage") or {}
            with_usage += int(bool(usage))
            rows = in_run.get(str(run.get("id")), [])
            ok = [r for r in rows if r.get("status") == "ok"]
            ledger = {
                "calls": len(rows),
                "prompt_tokens": sum(
                    int(r.get("input_tokens_actual") or 0) for r in ok
                ),
                "completion_tokens": sum(int(r.get("output_tokens") or 0) for r in ok),
            }
            for key in ("calls", "prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                count = (
                    int(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else 0
                )
                reported[key] += count
                seen[key] += ledger[key]
                if count > ledger[key]:
                    f.add(
                        "H10.3",
                        "mirix_usage_exceeds_ledger",
                        f"learning run {run.get('id')} reports {count} {key}, the "
                        f"gateway ledger saw {ledger[key]} on the mirix role while it "
                        "ran: some model calls bypassed the gateway",
                    )
            if usage and len(ok) > int(usage.get("calls") or 0):
                f.warn(
                    "H10.3",
                    "mirix_usage_underreported",
                    f"learning run {run.get('id')} reports {usage.get('calls')} calls, "
                    f"the ledger saw {len(ok)} successful ones while it ran",
                )
        learned = [r for r in runs if r.get("status") == "succeeded"]
        if learned and not with_usage:
            f.add(
                "H10.3",
                "mirix_usage_missing",
                f"{len(learned)} learning run(s) succeeded without llm_usage: the "
                "ledger cannot be reconciled with MIRIX",
            )
        retrievals = sum(
            1
            for r in read_jsonl(self.attempt_paths.method_dir / RETRIEVAL_RECORD_FILE)
            if r.get("event") == "retrieval" and not r.get("frozen")
        )
        # A retried topic extraction (a timed-out call sent once more as is) is flagged
        # repeat_of by the gateway; it is the same extraction, not another one.
        first_tries = [r for r in outside if not r.get("repeat_of")]
        if len(first_tries) > retrievals:
            f.add(
                "H10.3",
                "mirix_unexplained_calls",
                f"{len(first_tries)} mirix-role call(s) of the learning phase (retries "
                f"not counted) fall in no learning run, and MIRIX made only {retrievals} "
                "learning retrieval(s) (one topic extraction each at most): the rest is "
                "unexplained",
            )
        self.info["mirix_usage"] = {
            "ledger_in_runs": dict(seen),
            "learning_runs": dict(reported),
            "runs": len(runs),
            "runs_with_usage": with_usage,
            "calls_outside_runs": len(outside),
            "learning_retrievals": retrievals,
            "retry_events": sum(len(r.get("retry_events") or []) for r in runs),
        }

    def _reconcile_discarded_mirix(self) -> None:
        """MIRIX's discarded segments checked on their own: a segment's
        ``mirix``-role calls outside its discarded learning runs' windows (from the
        post-crash database, ``discarded.jsonl`` ``mirix.learning_runs``; a run killed
        mid-flight has no end) must not outnumber (first tries) its discarded retrievals
        (``mirix.retrievals``), at most one topic extraction each
        (``mirix_unexplained_calls``)."""
        by_segment: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for call_id, segment in self.discarded.items():
            row = self.by_call_id.get(call_id)
            if (
                row is not None
                and row.get("role") == "mirix"
                and row.get("status") != "rejected"
                and self._phase(row) in ("setup", "learning")
            ):
                by_segment[segment].append(row)
        rows = {r.get("segment"): r for r in self.resume_rows}
        for segment, calls in sorted(by_segment.items()):
            self.f.ok("H10.3")
            mirix = (rows.get(segment) or {}).get("mirix")
            mirix = mirix if isinstance(mirix, Mapping) else {}
            windows = []
            for run in mirix.get("learning_runs") or ():
                if not isinstance(run, Mapping):
                    continue
                start = _ts(run.get("started_at"))
                if start is not None:
                    windows.append((start, _ts(run.get("finished_at"))))
            outside = [
                r
                for r in calls
                if not r.get("repeat_of")
                and not any(
                    (stamp := _ts(r.get("ts"))) is not None
                    and a <= stamp
                    and (b is None or stamp <= b)
                    for a, b in windows
                )
            ]
            retrievals = len(mirix.get("retrievals") or ())
            if len(outside) > retrievals:
                self.f.add(
                    "H10.3",
                    "mirix_unexplained_calls",
                    f"discarded segment {segment}: {len(outside)} mirix-role call(s) "
                    f"(retries not counted) fall in none of its {len(windows)} "
                    f"discarded learning run(s), and it discarded only {retrievals} "
                    "retrieval(s) (one topic extraction each at most)",
                )

    # -- H10.4 ----------------------------------------------------------------------------

    def _check_frozen_memory(self) -> None:  # noqa: C901 - one check per record
        f = self.f
        record = self.frozen_record
        recorded = (self.state.get("frozen") or {}).get("sha256")
        f.ok("H10.4")
        if not isinstance(record, dict) or "snapshot" not in record:
            f.add("H10.4", "frozen_missing", "the attempt has no frozen snapshot")
            return
        actual = canonical_sha256(record.get("snapshot"))
        if not (actual == record.get("sha256") == recorded):
            f.add(
                "H10.4",
                "frozen_record",
                f"the frozen snapshot hashes to {actual}; its record says "
                f"{record.get('sha256')} and attempt.json {recorded}",
            )
        # A learning resume's check compares the rebuilt memory with its checkpoint, not
        # with the frozen one (``_check_restores``).
        checks = [
            c
            for c in self.state.get("snapshot_checks") or []
            if c.get("kind") != LEARNING_RESUME_CHECK
        ]
        after = [c for c in checks if c.get("kind") == "after_test"]
        if not after:
            if self.complete:
                f.add(
                    "H10.4",
                    "after_test_missing",
                    "no snapshot was compared with the frozen one after the test phase",
                )
            else:
                f.warn("H10.4", "after_test_missing", "the test phase has not ended")
        for check in checks:
            f.ok("H10.4")
            where = f"{check.get('kind')} check of launch {check.get('launch')}"
            if not (
                check.get("equal") is True
                and check.get("sha256") == recorded
                and check.get("frozen_sha256") == recorded
            ):
                f.add(
                    "H10.4",
                    "snapshot_changed",
                    f"the {where} found memory {check.get('sha256')}, frozen {recorded}",
                )
            stored = read_json(self.root / str(check.get("path")))
            if not isinstance(stored, dict) or canonical_sha256(
                stored.get("snapshot")
            ) != check.get("sha256"):
                f.add(
                    "H10.4",
                    "snapshot_record",
                    f"the stored snapshot of the {where} ({check.get('path')}) does not "
                    "match its record",
                )
        by_launch: dict[str, datetime] = {}
        for row in read_jsonl(self.attempt_paths.results):
            stamp = _ts(row.get("at"))
            launch = str(row.get("launch"))
            if stamp is not None and (
                launch not in by_launch or stamp > by_launch[launch]
            ):
                by_launch[launch] = stamp
        for launch, last_row in by_launch.items():
            closing = [
                c
                for c in after
                if str(c.get("launch")) == launch
                and (_ts(c.get("at")) or last_row) >= last_row
            ]
            if closing:
                continue
            later_resume = [
                c
                for c in checks
                if c.get("kind") == "resume"
                and (_ts(c.get("at")) or last_row) > last_row
            ]
            if later_resume:
                f.warn(
                    "H10.4",
                    "interrupted_launch",
                    f"launch {launch} scored test items but stopped before its "
                    "after-test snapshot; the next resume check verified the stored "
                    "memory, not that launch's in-memory state",
                )
            else:
                f.add(
                    "H10.4",
                    "unverified_launch",
                    f"launch {launch} scored test items and no snapshot was compared "
                    "after them",
                )
        frozen_at = _ts((self.state.get("frozen") or {}).get("at"))
        for row in self.calls:
            if row.get("role") not in LEARNING_ROLES:
                continue
            f.ok("H10.4")
            call_id, role = row.get("call_id"), row.get("role")
            if self._phase(row) == "test":
                f.add(
                    "H10.4",
                    "learning_call_in_test",
                    f"call {call_id} ({role}) ran in the frozen test phase",
                )
            stamp = _ts(row.get("ts"))
            if frozen_at is not None and stamp is not None and stamp > frozen_at:
                f.add(
                    "H10.4",
                    "learning_call_after_freeze",
                    f"call {call_id} ({role}) arrived at {row.get('ts')}, after the "
                    f"memory was frozen at {self.state['frozen']['at']}",
                )
            item = row.get("item")
            if (
                self.learning_uids
                and item is not None
                and item not in self.learning_uids
            ):
                f.add(
                    "H10.4",
                    "learning_call_for_test_item",
                    f"call {call_id} ({role}) was made for item {item}, which is not a "
                    "learning item of the cell",
                )

    # -- H9: the results the report reads ----------------------------------------------

    def _check_results(self) -> None:  # noqa: C901 - one check per record
        """The numbers ``report.py`` publishes are the scorer's own:

        * the cell's items are the protocol's: the whole test split (protocol) or the head
          of the seed-42 val order (smoke), and the learning stream is the seed-42 train
          order (or its head);
        * every scored row carries the score of its trajectory's recorded result, and
          ``env.score`` of the trajectory's final answer gives that score again;
        * a complete cell's ``results.jsonl`` is, row for row, the last scored attempt row
          of each test item, in test order.
        """
        from evals.skill_bench.run import seeded_order

        f = self.f
        f.ok("H9")
        env = self._env_quiet()
        split_mode = str(self.cell.get("split_mode") or "protocol")
        learning = [str(x) for x in self.cell.get("learning_items") or ()]
        test = [str(x) for x in self.cell.get("test_items") or ()]
        try:
            if split_mode == "smoke":
                n_test = int(self.cell.get("n_test") or 0)
                n_train = int(self.cell.get("n_train") or 0)
                want_test = [
                    t.raw_id
                    for t in seeded_order(
                        list(self.envs.tasks(self.benchmark, "val").values())
                    )[:n_test]
                ]
                want_learning = [
                    t.raw_id
                    for t in seeded_order(
                        list(self.envs.tasks(self.benchmark, "train").values())
                    )[:n_train]
                ]
            elif split_mode == "pilot":
                # The no-memory method on the whole pilot split, no learning.
                want_test = [
                    t.raw_id
                    for t in seeded_order(
                        list(self.envs.tasks(self.benchmark, "pilot").values())
                    )
                ]
                want_learning = []
            else:
                want_test = [
                    t.raw_id
                    for t in seeded_order(
                        list(self.envs.tasks(self.benchmark, "test").values())
                    )
                ]
                want_learning = [
                    t.raw_id
                    for t in seeded_order(
                        list(self.envs.tasks(self.benchmark, "train").values())
                    )
                ]
        except Exception as exc:  # noqa: BLE001 - the split is reported unavailable
            self._items_missing = True
            f.not_checked(
                "H9",
                "items_unavailable",
                f"this machine cannot load the splits, so the cell's items were not "
                f"compared with the protocol's: {exc}",
            )
        else:
            if sorted(test) != sorted(want_test) or len(test) != len(want_test):
                f.add(
                    "H9",
                    "test_items",
                    f"the cell tested {len(test)} item(s), the {split_mode} protocol "
                    f"tests {len(want_test)} ({len(set(test) - set(want_test))} "
                    "not in it, "
                    f"{len(set(want_test) - set(test))} missing)",
                )
            # The whole train split (a smoke cell learns from its seeded head): an item
            # outside it is one a method may not learn from; any other difference from
            # the seed-42 order is recorded.
            train = set(self.envs.tasks(self.benchmark, "train"))
            outside = sorted(set(learning) - train)
            twice = len(set(learning)) != len(learning)
            if outside or twice:
                f.add(
                    "H9",
                    "learning_items",
                    f"the cell's learning stream holds {len(outside)} item(s) outside "
                    f"the protocol's train split ({_some(outside)})"
                    + (" and items listed twice" if twice else "")
                    + ": a method may not learn from them (H1.4)",
                )
            elif learning and learning != want_learning:
                f.add(
                    "H9",
                    "learning_items_order",
                    "the cell's learning stream is not the seed-42 train order (H1.4)",
                )
        self._check_learning_rows(learning)
        rows, torn = read_jsonl_counted(self.attempt_paths.results)
        if torn:
            f.warn(
                "H9",
                "torn_lines",
                f"results.jsonl of attempt {self.number} has {torn} torn line(s)",
            )
        unserved: dict[str, list[str]] = {"test": [], "learning": []}
        refused_before = _refused_tries_before(rows)
        for t in self.trajectories:
            score = t.row.get("score")
            if score is None:
                continue
            f.ok("H9")
            result = t.data.get("result") or {}
            if result.get("score") != score:
                f.add(
                    "H9",
                    "score_record",
                    f"the {t.phase} row's score is not its trajectory's recorded result",
                    item=t.raw_id,
                )
            if _provider_unserved(t.row):
                # No provider could serve the actor's request (gw-404-no-endpoint):
                # scored 0 as unanswered, never by the environment.
                unserved.setdefault(t.phase, []).append(t.raw_id)
                if not (
                    isinstance(score, Mapping)
                    and _number(score.get("primary")) == 0.0
                    and score.get("answered") is False
                ):
                    f.add(
                        "H9",
                        "score_record",
                        f"the {t.phase} row is flagged provider_unserved but not scored "
                        "0 as unanswered",
                        item=t.raw_id,
                    )
                why = self._unserved_unbacked(t, refused_before.get(t.raw_id, 0))
                if why is not None:
                    # The flag skips the environment's rescore and leaves the item out of
                    # the report's accuracy without unserved items: it must be the
                    # ledger's refusal, not the harness's word.
                    f.add(
                        "H9",
                        "score_record",
                        f"the {t.phase} row is flagged provider_unserved, but {why}",
                        item=t.raw_id,
                    )
                continue
            task = self.tasks.get(t.raw_id)
            if env is None or task is None:
                continue
            try:
                again = env.score(
                    task, t.data.get("final_answer"), t.data.get("final_private") or {}
                )
            except Exception as exc:  # noqa: BLE001 - reported as a finding
                f.add(
                    "H9",
                    "rescore",
                    f"the {t.phase} item cannot be scored again: {type(exc).__name__}: "
                    f"{exc}",
                    item=t.raw_id,
                )
                continue
            if not _same_score(again, score):
                f.add(
                    "H9",
                    "rescore",
                    f"scoring the {t.phase} trajectory's final answer again gives "
                    f"{again.primary} {dict(again.metrics)}, the row says "
                    f"{score.get('primary')} {score.get('metrics')}",
                    item=t.raw_id,
                )
        self.info["provider_unserved"] = {k: len(v) for k, v in unserved.items()}
        if any(unserved.values()):
            f.warn(
                "H9",
                "provider_unserved_scored",
                "; ".join(
                    f"{len(items)} {phase} item(s) scored 0 as unanswered because the "
                    f"provider refused their actor call (HTTP 400/413 or "
                    f"gw-404-no-endpoint; a test item on its second try): "
                    f"{_some(sorted(items))}"
                    for phase, items in sorted(unserved.items())
                    if items
                ),
            )
        if not self.complete:
            return
        # What report.py reads, so it can tell whether this audit saw today's records.
        self.info["records_sha256"] = records_digest(self.paths)
        latest = {str(r.get("raw_id")): r for r in rows if r.get("score") is not None}
        cell_rows, torn = read_jsonl_counted(self.paths.results)
        expected = [latest.get(raw_id) for raw_id in test]
        if torn or cell_rows != expected:
            where = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(cell_rows, expected, strict=False))
                    if a != b
                ),
                min(len(cell_rows), len(expected)),
            )
            f.add(
                "H9",
                "cell_results",
                f"{self.paths.results.name} is not the attempt's last scored row per test "
                f"item in test order (first difference at row {where}; {len(cell_rows)} "
                f"row(s), {len(expected)} expected): the report would publish other "
                "numbers than the scorer recorded",
            )

    def _check_learning_rows(self, learning: Sequence[str]) -> None:
        """The live ``learning.jsonl`` plays each learning item once, in stream order
        (each item is played once): no position twice
        (``learning_position_repeated``, what a learning resume that did not roll back
        would leave), and once the learning phase is over, positions 0..n-1 in order,
        each the stream's item (``learning_rows``)."""
        f = self.f
        rows = read_jsonl(self.attempt_paths.learning)
        positions = [r.get("position") for r in rows]
        f.ok("H9")
        twice = sorted(
            {p for p in positions if positions.count(p) > 1}, key=lambda p: str(p)
        )
        if twice:
            f.add(
                "H9",
                "learning_position_repeated",
                f"the learning stream played position(s) {_some([str(p) for p in twice])} "
                "more than once: each learning item is played once (fairness "
                "constraint 3; a learning resume rolls the redone try back)",
            )
        played = [str(r.get("raw_id")) for r in rows]
        if learning:
            stream = set(learning)
            outside = sorted({raw for raw in played if raw not in stream})
            if outside:
                f.add(
                    "H9",
                    "learning_rows",
                    f"learning.jsonl of attempt {self.number} played {len(outside)} "
                    f"item(s) outside the cell's learning stream ({_some(outside)}): the "
                    "method learned from items it must not see",
                )
        if not learning or not (self.complete or self.state.get("frozen")):
            return
        if positions != list(range(len(learning))) or played != list(learning):
            where = next(
                (
                    i
                    for i, (p, raw, want) in enumerate(
                        zip(positions, played, learning, strict=False)
                    )
                    if p != i or raw != want
                ),
                min(len(rows), len(learning)),
            )
            f.add(
                "H9",
                "learning_rows_order",
                f"learning.jsonl of attempt {self.number} is not the learning stream "
                f"played once in order ({len(rows)} row(s) for {len(learning)} item(s); "
                f"first difference at row {where})",
            )

    def _env_quiet(self) -> Env | None:
        try:
            return self.envs.env(self.benchmark)
        except Exception:  # noqa: BLE001 - reported by _load_items
            return None


# ---------------------------------------------------------------------------
# A runs directory
# ---------------------------------------------------------------------------


def discover_cells(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(
        child
        for child in runs_dir.iterdir()
        if child.is_dir()
        and child.name not in NON_CELL_DIRS
        and (child / "cell.json").is_file()
    )


def _removed_cell(
    root: Path,
    definition: Mapping[str, Any],
    cell_id: str,
    method: str,
    removal: study.Removal,
) -> RemovedCell:
    """A removed method's cell as the audit lists it (never raises on its records);
    ``method`` as :func:`study.cell_method` read it."""
    try:
        status = read_json(root / "status.json")
    except (OSError, ValueError):
        status = None

    def text(key: str) -> str | None:
        value = definition.get(key)
        return str(value) if value is not None else None

    return RemovedCell(
        cell_id=cell_id,
        root=str(root),
        benchmark=text("benchmark"),
        model=text("model"),
        method=method,
        split_mode=text("split_mode"),
        status=str(status.get("status"))
        if isinstance(status, Mapping) and status.get("status") is not None
        else None,
        decision=removal.decision,
        reason=removal.reason,
    )


def _frozen_or_none() -> str | None:
    try:
        return protocol.frozen_fingerprint()
    except protocol.ProtocolDriftError:
        return None


def _known_versions_or_empty() -> dict[str, dict[str, Any]]:
    try:
        return protocol.known_versions()
    except protocol.ProtocolDriftError:
        return {}


#: How the cells of a method that ``protocol.METHODS`` does not list, and that no
#: removal record names, are set aside.
NOT_IN_MATRIX = study.Removal(
    decision="not in protocol.METHODS",
    reason="a method the study neither compares nor removed: set aside, not audited and "
    "not reported (H10.3-role-policy)",
)
#: Where ``tools/remove_method_data.py`` keeps its removal manifests (its
#: ``REMOVED_DIR``), in a host's runs directory and in each ``_hosts/<host>/`` mirror.
REMOVED_DATA_DIR = "_removed"


@dataclass(frozen=True)
class DirectoryFacts:
    """What the choice among directories of one cell id reads, from ``cell.json``,
    ``status.json``, the attempts' ``attempt.json`` and ``results.jsonl`` only."""

    root: Path
    complete: bool
    started_at: str | None
    started: datetime | None
    created: datetime | None
    status: str | None

    def order(self) -> tuple[datetime, datetime, str]:
        """``(started, created, directory name)``, missing times lowest."""
        lowest = datetime.min.replace(tzinfo=timezone.utc)
        return (self.started or lowest, self.created or lowest, self.root.name)


def directory_facts(root: Path) -> DirectoryFacts:
    """:class:`DirectoryFacts` of a cell directory (blocking; never raises on its
    records). Complete: ``status.json`` ``complete``, some attempt ``complete`` and
    ``results.jsonl`` written. A complete directory is dated by its reported attempt,
    the highest-numbered complete one (a superseded attempt is ``invalid``, never
    reported), any other by its last attempt (``report.directory_started_at``)."""
    paths = CellPaths(root)

    def load(path: Path) -> Any:
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    manifest = load(paths.manifest)
    status = load(paths.status)
    status_text = (
        str(status.get("status"))
        if isinstance(status, Mapping) and status.get("status") is not None
        else None
    )
    numbers = sorted(paths.attempt_numbers())
    states: dict[int, Mapping[str, Any]] = {}
    for number in numbers:
        state = load(paths.attempt(number).state)
        states[number] = state if isinstance(state, Mapping) else {}
    complete_attempts = [
        n for n, s in states.items() if s.get("state") == ATTEMPT_COMPLETE
    ]
    complete = (
        status_text == STATUS_COMPLETE
        and bool(complete_attempts)
        and paths.results.is_file()
    )
    # Dated by the reported attempt of a complete directory, else by its last attempt
    # (report.directory_started_at: one rule for both).
    dated = max(complete_attempts) if complete else (numbers[-1] if numbers else None)
    started_at = states[dated].get("started_at") if dated is not None else None
    return DirectoryFacts(
        root=root,
        complete=complete,
        started_at=str(started_at) if started_at is not None else None,
        started=_ts(started_at) if started_at is not None else None,
        created=_ts(manifest.get("created_at"))
        if isinstance(manifest, Mapping) and manifest.get("created_at")
        else None,
        status=status_text,
    )


def choose_directory(
    roots: Sequence[Path],
) -> tuple[DirectoryFacts, list[tuple[DirectoryFacts, str]]]:
    """The directory the audit reads among directories holding one cell id, and the
    others with their role (``report.py`` and ``grade.py`` apply the same rule):
    chosen is the max by (complete, reported attempt's ``started_at``, ``cell.json``
    ``created_at``, directory name), missing times lowest. Another directory is
    ``superseded`` when its (started, created, name) sorts below the chosen one's,
    else ``newer_incomplete`` (a newer run that is not complete)."""
    facts = [directory_facts(root) for root in roots]
    chosen = max(facts, key=lambda f: (f.complete, *f.order()))
    others = [
        (f, "superseded" if f.order() < chosen.order() else "newer_incomplete")
        for f in facts
        if f is not chosen
    ]
    return chosen, others


def removed_manifests(runs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The cells of removed methods whose directories ``tools/remove_method_data.py``
    deleted, from its removal manifests in ``runs_dir/_removed/`` and in each
    ``_hosts/<host>/_removed/`` mirror of a merged view, one per ``removal_id``; a
    removal still ``deleting`` counts too (blocking; never raises). Each entry:
    ``method, cell_id, status, host, deleted_at, manifest`` (runs-relative),
    ``split_mode`` and ``cell_dir``. The second value lists the manifests that could not
    be read (``removed_manifest_unreadable``)."""
    directories = [runs_dir]
    try:
        directories += sorted(p for p in (runs_dir / HOSTS_DIR).iterdir() if p.is_dir())
    except OSError:
        pass
    if not any((d / REMOVED_DATA_DIR).is_dir() for d in directories):
        return [], []
    try:
        from evals.skill_bench.tools import remove_method_data
    except ImportError as exc:
        return [], [
            f"{REMOVED_DATA_DIR}/: tools/remove_method_data.py cannot be imported to "
            f"read its removal manifests ({type(exc).__name__}: {exc})"
        ]

    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        # load_manifests may read the host mirrors itself: one removal once, by its
        # removal_id (a manifest that cannot be read: its directory's name).
        for manifest in remove_method_data.load_manifests(directory):
            path = Path(manifest.path)
            data = manifest.data if isinstance(manifest.data, Mapping) else {}
            removal_id = str(data.get("removal_id") or path.parent.name)
            if removal_id in seen:
                continue
            seen.add(removal_id)
            try:
                rel = path.relative_to(runs_dir).as_posix()
            except ValueError:
                rel = str(path)
            problems += [
                problem.replace(str(path), rel)
                if str(path) in problem
                else f"{rel}: {problem}"
                for problem in manifest.problems or ()
            ]
            if not data:
                continue
            if data.get("state") not in ("deleting", "complete"):
                problems.append(
                    f"{rel}: state {data.get('state')!r} is neither deleting nor complete"
                )
                continue
            host = data.get("host")
            hostname = host.get("hostname") if isinstance(host, Mapping) else None
            for cell in data.get("cells") or ():
                if not isinstance(cell, Mapping) or not cell.get("cell_id"):
                    continue
                entries.append(
                    {
                        "method": str(cell.get("method") or data.get("method")),
                        "cell_id": str(cell["cell_id"]),
                        "status": cell.get("status"),
                        "split_mode": cell.get("split_mode"),
                        "cell_dir": cell.get("cell_dir"),
                        "host": hostname,
                        "deleted_at": data.get("completed_at"),
                        "state": data.get("state"),
                        "removal_id": removal_id,
                        "manifest": rel,
                    }
                )
    return entries, problems


def audit_runs(  # noqa: C901 - selection, set-asides, then the audits
    runs_dir: Path,
    *,
    env_factory: EnvFactory | None = None,
    frozen: str | None = None,
    expected_upstream: str = DEFAULT_UPSTREAM,
    cells: Sequence[str] | None = None,
    split_mode: str | None = "protocol",
    include_incomplete: bool = False,
    blob_dir: Path | None = None,
    gateway_runs_dirs: Sequence[Path] = (),
    versions: Mapping[str, Mapping[str, Any]] | None = None,
    removed_methods: Mapping[str, study.Removal] | None = None,
) -> AuditReport:
    """Audit every selected cell of ``runs_dir`` and the invariants across cells (blocking).

    ``runs_dir`` may be a merged view of several hosts (``tools/sync_runs.py``): its cells
    and gateway runs are links into ``_hosts/<name>/``, whose gateway start records are
    read too. ``gateway_runs_dirs`` names other directories that hold ledgers.

    ``versions`` are the protocol versions a cell may carry (``protocol.known_versions``):
    by default those of ``config/protocol_frozen.json`` when ``frozen`` is read from it
    too, and only ``frozen`` itself when the caller names it. The auditing tree is read
    once (:func:`_tree_version`): the checks that read it are judged per cell against
    the cell's own versions (:data:`TREE_INPUTS`).

    Set aside before anything of them is judged, never audited, never incomplete and in
    no check across cells:

    * a selected cell of a method the study removed (``removed_methods``, by default
      ``study.REMOVED_METHODS``), listed under ``removed``, and
      those whose directories were deleted, from their removal manifests
      (:func:`removed_manifests`, ``removed.deleted``);
    * a cell of a method ``protocol.METHODS`` does not list (``not_in_matrix``, one
      warning per method, H10.3-role-policy);
    * every directory but one of a cell id several directories hold (``duplicates``,
      :func:`choose_directory`: the latest-started complete one is audited).

    A run-level violation names the cells it concerns: each gets a copy and fails; one
    that names no audited cell fails the audit as a whole
    (``unattributed_violations``)."""
    removals = study.REMOVED_METHODS if removed_methods is None else removed_methods
    if frozen is None:
        frozen = _frozen_or_none()
        if versions is None:
            versions = _known_versions_or_empty()
    tree = _tree_version(frozen)
    auditor = str(tree["fingerprint"])
    findings = Findings(None)
    if frozen is None:
        findings.add(
            "H10.6",
            "not_frozen",
            "config/protocol_frozen.json is missing: nothing to compare",
        )
    # A merged view of several hosts: why it is not final, as warnings that change no
    # verdict (its conflicts are unresolved, not violations).
    merged_view = merged_view_status(runs_dir)
    for warning in (merged_view or {}).get("warnings") or ():
        findings.warn("H9", warning["check"], warning["detail"])
    envs = Envs(env_factory)
    set_aside: list[RemovedCell] = []
    outside: list[RemovedCell] = []
    groups: dict[str, list[Path]] = {}
    for root in discover_cells(runs_dir):
        manifest = read_json(root / "cell.json") or {}
        definition = manifest.get("cell") or {}
        cell_id = str(definition.get("cell_id") or root.name)
        if cells is not None and cell_id not in cells:
            continue
        if (
            split_mode is not None
            and definition.get("split_mode", "protocol") != split_mode
        ):
            continue
        # The definition's method, else the id's (a removed cell whose cell.json lost
        # its method is never audited as a study cell).
        method = study.cell_method(definition, cell_id)
        if method is not None and method in removals:
            # Before the directory count: a removed cell's two directories are no
            # duplicate of the study's (its method is part of its cell id).
            set_aside.append(
                _removed_cell(root, definition, cell_id, method, removals[method])
            )
            continue
        if method is None or method not in protocol.METHODS:
            outside.append(
                _removed_cell(
                    root, definition, cell_id, str(method or "?"), NOT_IN_MATRIX
                )
            )
            continue
        groups.setdefault(cell_id, []).append(root)
    results: list[CellAudit] = []
    duplicates: list[DuplicateCell] = []
    for cell_id, roots in groups.items():
        findings.ok("H10.3")
        chosen = roots[0]
        if len(roots) > 1:
            facts, others = choose_directory(roots)
            chosen = facts.root
            for other, role in others:
                duplicates.append(
                    DuplicateCell(
                        cell_id=cell_id,
                        root=str(other.root),
                        role=role,
                        chosen_root=str(chosen),
                        started_at=other.started_at,
                        status=other.status,
                        complete=other.complete,
                    )
                )
            # Two hosts, or a rerun in a new directory, hold one
            # cell id: the newer complete run is reported, the others are listed.
            findings.add(
                "H10.3",
                "duplicate_cell_directories",
                f"{len(roots)} cell directories of {runs_dir} hold cell {cell_id} ("
                + ", ".join(sorted(r.name for r in roots))
                + f"): the audit reads {chosen.name}, the latest-started "
                + ("complete one" if facts.complete else "one (none is complete)")
                + "; "
                + ", ".join(f"{o.root.name} ({role})" for o, role in others)
                + " are listed under duplicates and not audited ("
                "the newer complete run is reported, the older listed as superseded; "
                "their money stays counted by the report)",
                cells=(cell_id,),
            )
        results.append(
            CellAuditor(
                chosen,
                runs_dir=runs_dir,
                envs=envs,
                frozen=frozen,
                expected_upstream=expected_upstream,
                blob_dir=blob_dir,
                include_incomplete=include_incomplete,
                gateway_runs_dirs=gateway_runs_dirs,
                versions=versions,
                tree=tree,
            ).run()
        )
    by_method: dict[str, list[str]] = defaultdict(list)
    for cell in outside:
        by_method[cell.method].append(cell.cell_id)
    for method, ids in sorted(by_method.items()):
        ids = sorted(set(ids))
        findings.warn(
            "H10.3",
            "method_not_in_matrix",
            f"{len(ids)} cell(s) of method {method!r}, which protocol.METHODS does not "
            f"list and no removal record names ({_some(ids)}): set aside, not audited "
            "and not reported (H10.3-role-policy)",
            cells=tuple(ids),
        )
    deleted, problems = removed_manifests(runs_dir)
    live = {(r.cell_id, Path(r.root).name) for r in set_aside}
    deleted = [
        d
        for d in deleted
        if (cells is None or d["cell_id"] in cells)
        and (split_mode is None or (d.get("split_mode") or "protocol") == split_mode)
        # An interrupted deletion whose directory is still there is read live.
        and (d["cell_id"], Path(str(d.get("cell_dir") or "")).name) not in live
    ]
    for problem in problems:
        findings.warn(
            "H10.6",
            "removed_manifest_unreadable",
            f"{problem}: the cells it lists are not known",
        )
    _check_across_cells(results, findings)
    unattributed = _attribute_across_cells(results, findings)
    return AuditReport(
        runs_dir=str(runs_dir),
        frozen_fingerprint=frozen,
        auditor_fingerprint=auditor,
        expected_upstream=expected_upstream,
        cells=results,
        findings=list(findings.items),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        merged_view=merged_view,
        removed=set_aside,
        removed_methods=dict(removals),
        unattributed_violations=unattributed,
        duplicates=duplicates,
        not_in_matrix=outside,
        deleted=deleted,
        tree_current=bool(tree["current"]) if frozen is not None else None,
    )


def _attribute_across_cells(cells: Sequence[CellAudit], findings: Findings) -> int:
    """Each run-level violation that names cells (``Findings.across``, uncapped) fails
    exactly the audited cells it names: a copy with the cell's id joins each one's
    findings (at most :data:`MAX_FINDINGS_PER_CHECK` per check, the rest counted) and
    violations. Returns how many run-level violations name no audited cell: they fail
    the audit as a whole (``AuditReport.unattributed_violations``)."""
    audited = {cell.cell_id: cell for cell in cells if cell.audited}
    copies: Counter[tuple[str, str]] = Counter()
    attributed = 0
    for finding in findings.across:
        named = [audited[c] for c in dict.fromkeys(finding.cells) if c in audited]
        if not named:
            continue
        attributed += 1
        key = f"{finding.invariant}:{finding.check}"
        for cell in named:
            cell.violations += 1
            cell.violations_by_invariant[finding.invariant] = (
                cell.violations_by_invariant.get(finding.invariant, 0) + 1
            )
            copies[(cell.cell_id, key)] += 1
            if copies[(cell.cell_id, key)] > MAX_FINDINGS_PER_CHECK:
                cell.suppressed[key] = cell.suppressed.get(key, 0) + 1
            else:
                cell.findings.append(replace(finding, cell_id=cell.cell_id))
    return findings.count(VIOLATION) - attributed


def _check_across_cells(cells: Sequence[CellAudit], findings: Findings) -> None:
    """H10.2 across cells: every item's no-memory first prompt is one across the cells
    whose versions agree on what those prompts are made of (``info.prompt_key``,
    :data:`PROMPT_INPUTS`; or that ran one fingerprint, :func:`_prompt_groups`): a cell
    whose prompt differs from its key's majority (every cell of the key when none leads)
    fails (``first_prompt_differs_across_cells``); a
    difference only across keys is a warning, ``first_prompt_differs_across_versions``
    (H10.2-first-prompt-across-cells). H10.6 across cells (versions are recorded and
    reported, not enforced across cells): cells of one benchmark that ran another
    harness code, MIRIX package or protocol are a warning (``code_differs_across_cells``,
    ``protocol_differs_across_cells``, listing each version's cells); the report flags
    each comparison across versions. Every finding names the cells it concerns
    (``cells``).

    ``cells`` are the audited study's cells only."""
    for key, label in (("harness", "harness code"), ("mirix", "MIRIX package")):
        by_benchmark: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for cell in cells:
            value = (cell.info.get("code") or {}).get(key)
            if cell.audited and value:
                by_benchmark[str(cell.benchmark)][str(value)].append(cell.cell_id)
        for benchmark, variants in sorted(by_benchmark.items()):
            findings.ok("H10.6")
            if len(variants) > 1:
                findings.warn(
                    "H10.6",
                    "code_differs_across_cells",
                    f"{benchmark} cells ran {len(variants)} versions of the {label} ("
                    + "; ".join(
                        f"{v[:12]}: {', '.join(sorted(c))}"
                        for v, c in sorted(variants.items())
                    )
                    + "): the report flags their comparisons",
                    cells=tuple(sorted(c for v in variants.values() for c in v)),
                )
        overall = {v for variants in by_benchmark.values() for v in variants}
        if len(overall) > 1 and all(len(v) == 1 for v in by_benchmark.values()):
            findings.warn(
                "H10.6",
                "code_differs_across_benchmarks",
                f"the benchmarks ran {len(overall)} versions of the {label}",
            )
    _check_protocol_across_cells(cells, findings)
    _check_hosts_across_cells(cells, findings)
    _check_first_prompts_across_cells(cells, findings)


def _prompt_groups(cells: Sequence[CellAudit]) -> dict[str, str | None]:
    """Cell id -> the group of cells that must share every item's no-memory first
    prompt: its ``prompt_key``; else, when the attempt ran one protocol fingerprint
    (``prompt_fingerprint``), the key of another cell of that fingerprint, or
    ``fp:<fingerprint>`` (one fingerprint means the same templates, splits and
    seed); else None (a group of its own)."""
    by_fingerprint: dict[str, str] = {}
    for cell in cells:
        key, fingerprint = (
            cell.info.get("prompt_key"),
            cell.info.get("prompt_fingerprint"),
        )
        if key and fingerprint:
            by_fingerprint.setdefault(str(fingerprint), str(key))
    groups: dict[str, str | None] = {}
    for cell in cells:
        key = cell.info.get("prompt_key")
        fingerprint = cell.info.get("prompt_fingerprint")
        if key:
            groups[cell.cell_id] = str(key)
        elif fingerprint:
            groups[cell.cell_id] = by_fingerprint.get(
                str(fingerprint), f"fp:{fingerprint}"
            )
        else:
            groups[cell.cell_id] = None
    return groups


def _check_first_prompts_across_cells(
    cells: Sequence[CellAudit], findings: Findings
) -> None:
    """H10.2-first-prompt-across-cells (see :func:`_check_across_cells`)."""
    keys = _prompt_groups(cells)
    by_item: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for cell in cells:
        for raw_id, hashes in (cell.info.get("first_prompts") or {}).items():
            for digest in hashes:
                by_item[(str(cell.benchmark), raw_id)][digest].add(cell.cell_id)
    for (benchmark, raw_id), variants in sorted(by_item.items()):
        findings.ok("H10.2")
        if len(variants) < 2:
            continue
        text = (
            f"{benchmark} item {raw_id}: {len(variants)} different no-memory first "
            "prompts across cells ("
            + "; ".join(
                f"{d[:12]}: {', '.join(sorted(c))}" for d, c in sorted(variants.items())
            )
            + ")"
        )
        # Per prompt group (_prompt_groups; a cell of none is a group of its own).
        by_key: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for digest, ids in variants.items():
            for cell_id in ids:
                key = keys.get(cell_id)
                by_key[key if key else f"unknown:{cell_id}"][digest].add(cell_id)
        blamed: set[str] = set()
        for key, digests in by_key.items():
            if key.startswith("unknown:") or len(digests) < 2:
                continue
            sizes = {digest: len(ids) for digest, ids in digests.items()}
            top = max(sizes.values())
            leaders = [digest for digest, n in sizes.items() if n == top]
            for digest, ids in digests.items():
                if len(leaders) > 1 or digest not in leaders:
                    blamed |= ids
        if blamed:
            findings.add(
                "H10.2",
                "first_prompt_differs_across_cells",
                text,
                item=raw_id,
                cells=tuple(sorted(blamed)),
            )
        else:
            findings.warn(
                "H10.2",
                "first_prompt_differs_across_versions",
                text + " across protocol versions whose templates, splits or seed "
                "differ, or are not recorded (H10.2-first-prompt-across-cells): the "
                "report flags the comparison",
                item=raw_id,
                cells=tuple(sorted({c for ids in variants.values() for c in ids})),
            )


def _check_protocol_across_cells(
    cells: Sequence[CellAudit], findings: Findings
) -> None:
    """H10.6 across cells: the cells of one benchmark that ran
    more than one protocol version are a warning (``protocol_differs_across_cells``)
    listing each version's cells; each cell's own version is judged in the cell
    (``protocol_version``, ``fingerprint``)."""
    by_benchmark: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell in cells:
        fingerprint = (cell.info.get("protocol") or {}).get("fingerprint")
        if cell.audited and fingerprint:
            by_benchmark[str(cell.benchmark)][str(fingerprint)].append(cell.cell_id)
    for benchmark, variants in sorted(by_benchmark.items()):
        findings.ok("H10.6")
        if len(variants) > 1:
            findings.warn(
                "H10.6",
                "protocol_differs_across_cells",
                f"{benchmark} cells ran {len(variants)} protocol versions ("
                + "; ".join(
                    f"{v[:12]}: {_some(sorted(c))}" for v, c in sorted(variants.items())
                )
                + "): the report flags their comparisons",
                cells=tuple(sorted(c for v in variants.values() for c in v)),
            )


def _variant_text(key: str, names: Mapping[str, set[str]]) -> str:
    """A host key of :func:`_check_hosts_across_cells` as people read it: the hostnames
    seen with it (and the id), or the platform as it is."""
    if key.startswith("id:"):
        return f"{'/'.join(sorted(names.get(key) or ()))} (id {key[3:11]})"
    if key.startswith("hostname:"):
        return key[len("hostname:") :]
    return key


def _check_hosts_across_cells(cells: Sequence[CellAudit], findings: Findings) -> None:
    """H10.6 across cells, hosts: the cells of one benchmark,
    which the report compares, ran on one platform (``platform.system`` and
    ``platform.machine``: an arm64 Mac and an x86 Linux host are two, whatever their code
    digests say; ``platform_differs_across_cells``) and one sandbox image (WebShop's
    server image, ``cell.json`` preflight ``sandbox_image_id``: an x86 and an arm64
    build differ; ``sandbox_image_differs_across_cells``); both are recorded as
    warnings. Machines that differ (by machine id, else hostname:
    ``cell_records.host_key``) are a warning (``host_differs_across_cells``):
    the benchmark was split across machines of one platform."""
    for key, check, label, severity in (
        (
            lambda h: f"{h.get('system')} {h.get('machine')}",
            "platform_differs_across_cells",
            "platforms",
            VIOLATION,
        ),
        (host_key, "host_differs_across_cells", "hosts", WARNING),
    ):
        by_benchmark: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        names: dict[str, set[str]] = defaultdict(set)
        for cell in cells:
            host = cell.info.get("host")
            if cell.audited and isinstance(host, Mapping):
                by_benchmark[str(cell.benchmark)][key(host)].append(cell.cell_id)
                names[key(host)].add(str(host.get("hostname")))
        for benchmark, variants in sorted(by_benchmark.items()):
            findings.ok("H10.6")
            if len(variants) > 1:
                findings.add(
                    "H10.6",
                    check,
                    f"{benchmark} cells ran on {len(variants)} {label} ("
                    + "; ".join(
                        f"{_variant_text(v, names)}: {_some(sorted(c))}"
                        for v, c in sorted(variants.items())
                    )
                    + "): they are compared with each other",
                    severity=severity,
                    cells=tuple(sorted(c for v in variants.values() for c in v)),
                )
    images: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        image = cell.info.get("sandbox_image_id")
        if cell.audited and image:
            images[str(cell.benchmark)][str(image)].append(cell.cell_id)
    for benchmark, variants in sorted(images.items()):
        findings.ok("H10.6")
        if len(variants) > 1:
            findings.add(
                "H10.6",
                "sandbox_image_differs_across_cells",
                f"{benchmark} cells ran {len(variants)} sandbox images ("
                + "; ".join(
                    f"{v[:19]}: {_some(sorted(c))}" for v, c in sorted(variants.items())
                )
                + "): the environment's server image must be one per benchmark",
                cells=tuple(sorted(c for v in variants.values() for c in v)),
            )


def kept_passing_record(report: AuditReport, out_dir: Path) -> Path | None:
    """The passing ``audit.json`` in ``out_dir`` that ``report`` must not replace, if any.

    A record is kept when it passed, ``report`` does not, and the two were made under
    different protocol states (frozen or auditor fingerprint): an audit of a tree whose
    protocol has moved since the cells ran (an audit.json of schema 1 failed every cell
    with H10.6 ``auditor_tree``; the cells' own fingerprints after a re-freeze) says
    nothing about what the cells did at their commit, and ``report.py`` would mark every
    cell failed from it. A failing audit
    under the same protocol state replaces the record: that is news about the runs."""
    path = out_dir / AUDIT_JSON
    previous = read_json(path) if path.is_file() else None
    if not isinstance(previous, Mapping) or previous.get("passed") is not True:
        return None
    if report.passed:
        return None
    same_state = (
        previous.get("frozen_fingerprint"),
        previous.get("auditor_fingerprint"),
    ) == (report.frozen_fingerprint, report.auditor_fingerprint)
    return None if same_state else path


def write_audit(
    report: AuditReport, out_dir: Path, *, replace_passing: bool = False
) -> tuple[Path, Path]:
    """Write ``audit.json`` and ``audit.md`` into ``out_dir``; returns their paths.

    Unless ``replace_passing``, a report that must not replace the passing record there
    (:func:`kept_passing_record`) is written to ``out_dir/protocol-<auditor
    fingerprint's first 12 hex>/`` instead."""
    if not replace_passing and kept_passing_record(report, out_dir) is not None:
        out_dir = out_dir / f"protocol-{report.auditor_fingerprint[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / AUDIT_JSON, out_dir / AUDIT_MD
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    parser.add_argument("--out", type=Path, help="default <runs-dir>/_audit")
    parser.add_argument(
        "--split-mode",
        default="protocol",
        choices=("protocol", "smoke", "pilot", "all"),
    )
    parser.add_argument(
        "--cell",
        action="append",
        help="audit only this cell id (repeatable; a removed method's cell is listed "
        "as removed, not audited)",
    )
    parser.add_argument(
        "--expected-upstream",
        default=DEFAULT_UPSTREAM,
        help="the upstream the gateways must forward to (default %(default)s)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also audit the current attempt of cells that are not complete",
    )
    parser.add_argument(
        "--gateway-runs-dir",
        action="append",
        type=Path,
        default=[],
        help="another directory holding gateway runs (<dir>/<run_id>/gateway.jsonl), "
        "searched after the runs directory (repeatable)",
    )
    parser.add_argument(
        "--replace-passing",
        action="store_true",
        help="let a failing audit made under another protocol state replace the passing "
        "record in the output directory (by default it is written beside it)",
    )
    args = parser.parse_args(argv)
    # Absolute, as run.py and schedule.py record every path.
    args.runs_dir = args.runs_dir.absolute()
    report = audit_runs(
        args.runs_dir,
        cells=args.cell,
        split_mode=None if args.split_mode == "all" else args.split_mode,
        expected_upstream=args.expected_upstream,
        include_incomplete=args.include_incomplete,
        gateway_runs_dirs=[d.absolute() for d in args.gateway_runs_dir],
    )
    out_dir = args.out or args.runs_dir / "_audit"
    kept = None if args.replace_passing else kept_passing_record(report, out_dir)
    json_path, md_path = write_audit(
        report, out_dir, replace_passing=args.replace_passing
    )
    printed: dict[str, Any] = {
        "passed": report.passed,
        "violations": report.violations,
        "incomplete": report.incomplete,
        # Checks this machine could not run (its items or environment are missing).
        "not_checked": report.not_checked,
        # Set aside, not audited (study.REMOVED_METHODS): no part of the exit status.
        "removed": [r.cell_id for r in report.removed],
        "duplicates": [d.root for d in report.duplicates],
        "not_in_matrix": [r.cell_id for r in report.not_in_matrix],
        "json": str(json_path),
        "markdown": str(md_path),
    }
    if kept is not None:
        printed["kept_passing_record"] = str(kept)
    print(json.dumps(printed))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
