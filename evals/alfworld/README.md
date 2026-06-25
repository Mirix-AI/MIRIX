# ALFWorld Eval Harness

This package contains the MIRIX ALFWorld eval harness. It follows SkillOpt's
ALFWorld interface for data manifests, prompts, and `<think>/<action>` parsing,
while using MIRIX procedural memory for online retrieval and consolidation.

## Data Manifests

The manifest layout matches SkillOpt:

```text
data/alfworld_path_split/
  train/items.json
  val/items.json
  test/items.json
```

Each `items.json` is either a JSON list or an object with an `items` list.  Each
row must include:

```json
{"id": "trial-id", "gamefile": "relative/path/game.tw-pddl", "task_type": "pick_and_place"}
```

The lightweight SkillOpt path manifest is vendored under
`evals/alfworld/data/alfworld_path_split`. It contains 39 train, 18 val, and 134
test paths. It is not the raw ALFWorld payload.

Use `load_manifest(root)` to load available splits, or pass
`require_all=True` to require `train`, `val`, and `test`.  Game files are
resolved with `resolve_gamefile(gamefile)`, which uses `$ALFWORLD_DATA` for
relative paths.

`summarize_manifest(items)` returns `total`, `by_split`, `by_task_type`, and
`by_split_task_type` distributions.

Compatibility helpers `load_split(path)` and `summarize_splits(raw_splits)` are
also available for scripts that already pass concrete manifest files or raw
dict rows.

## Action Parsing

`parse_action(text)` follows SkillOpt-compatible priority:

1. Use the first non-empty `<action>...</action>` block.
2. Mark the format valid only when `<think>` is also present and the response
   has no Chinese characters.
3. Fall back to `look` when no action tag exists.

JSON action parsing is opt-in via `allow_json=True`; the default runner does not
use it. Any `<think>...</think>` blocks are preserved on `ParsedAction.thought`,
and the full original model text is kept on `ParsedAction.raw_text` for logging.

`parse_model_response(text)` is kept as a simple dict compatibility wrapper.
New harness code should prefer `parse_action(text)` because its missing-action
behavior is the eval contract.

## Running

Install optional dependencies and ALFWorld data:

```bash
pip install -e ".[eval,alfworld]"
alfworld-download
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

Start MIRIX separately and make sure OpenRouter/OpenAI credentials are in the
environment or `.env`:

```bash
export OPENROUTER_API_KEY=...
export MIRIX_URL=http://127.0.0.1:8531
```

Default online run:

```bash
python -m evals.alfworld \
  --arm mirix \
  --episodes 10 \
  --consolidate-every 5 \
  --split train
```

This runs 10 ALFWorld episodes from the SkillOpt train path manifest. Each
episode is written to MIRIX as one session. After episodes 5 and 10, the runner
writes a constant sentinel session to seal the latest real episode, then calls:

```http
POST /memory/auto_dream?user_id=<eval user>
{"mode": "procedural", "last_n_sessions": 5}
```

Outputs are written to `evals/alfworld/runs/<arm>-<timestamp>/`:

- `config.json`: run config and selected manifest items
- `episodes.jsonl`: one record per episode
- `consolidations.jsonl`: procedural auto-dream events
- `summary.json`: success rate and task-type breakdown
- `predictions/<item-id>/conversation.json`: per-step transcript

MIRIX session ids are generated with only letters, digits, `_`, and `-`, matching
the server-side message validator.

SkillOpt-aligned two-stage eval:

```bash
# Stage 1: generate MIRIX procedural skills from SkillOpt train only.
python -m evals.alfworld \
  --arm mirix \
  --split train \
  --episodes 0 \
  --consolidate-every 5 \
  --consolidate-final-remainder \
  --mirix-user-id alfworld-skillopt-aligned \
  --run-id alfworld-skillopt-train

# Stage 2: evaluate the frozen MIRIX skill state on the full 134-item test split.
python -m evals.alfworld \
  --arm mirix \
  --split test \
  --episodes 0 \
  --memory-mode frozen \
  --mirix-user-id alfworld-skillopt-aligned \
  --run-id alfworld-skillopt-test
```

In `frozen` memory mode the runner still retrieves existing procedural memories,
but it does not call `/memory/add_sync` and does not trigger procedural
consolidation. This keeps the 134 test episodes as held-out evaluation episodes
instead of online adaptation data.

The train split has 39 items. `--consolidate-final-remainder` keeps the normal
5-episode consolidation cadence and then performs one final consolidation for the
remaining 4 train episodes before the frozen test run.

Use `--dry-run` to validate item selection without importing ALFWorld or calling
models:

```bash
python -m evals.alfworld --dry-run --episodes 10 --split train
```
