# MetaClaw Benchmark CLI

Command-line interface for running and evaluating the MetaClaw Evolution Benchmark.

## Entry point

```
metaclaw-bench <command> [options]
```

Or via Python module (run from the `benchmark/` directory):

```
python -m src.cli <command> [options]
```

All relative paths are resolved against the **project root** (the directory
that contains `benchmark/`), not against `benchmark/` itself.

---

## Commands

### `check`

Validate a benchmark dataset before running inference.

```
metaclaw-bench check -p <path/to/all_tests.json>
```

**Options**

| Flag | Required | Description |
|------|----------|-------------|
| `-p`, `--path` | yes | Path to `all_tests.json` |

**Checks performed (8 total)**

| # | Checker | Description |
|---|---------|-------------|
| 1 | AllTests Structure | Top-level fields and test array structure; unique agent ID |
| 2 | Basic Integrity | All referenced files exist on disk |
| 3 | ID Consistency | Session IDs unique; internal IDs match filenames |
| 4 | File Format | JSONL and `questions.json` files are valid JSON |
| 5 | Directory Structure | `eval/` and `sessions/` directories exist |
| 6 | Workspace Integrity | `workspace_src` contains required identity files |
| 7 | Session Format | Session JSONL first/second line roles are correct |
| 8 | Questions Integrity | Round types, feedback strings, and eval field structure |

**Example**

```bash
# Run from the project root (MetaClaw/)
metaclaw-bench check -p benchmark/data/metaclaw-bench/all_tests.json
```

---

### `infer`

Run the openclaw agent for each test scenario and save per-question results.

```
metaclaw-bench infer -i <input> -o <output> [options]
```

**Options**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-i`, `--input` | yes | — | Path to `all_tests.json`, a directory of `all_tests.json` files, or a JSON list of paths |
| `-o`, `--output` | yes | — | Output directory |
| `-w`, `--workers` | no | `1` | Maximum concurrent tests |
| `-n`, `--retry` | no | `0` | Retries per failed question |
| `--scene-per-train` | no | disabled | Trigger `metaclaw train-step` every N scenes |
| `--memory` | no | off | Trigger `POST /v1/memory/ingest` after each scene |
| `--memory-proxy-port` | no | `30000` | MetaClaw proxy port for memory ingest |

> **Note:** Tests always run serially (workers=1) when `--scene-per-train` or
> `--memory` is active, because these features require strict ordering.

**questions.json format**

```json
{
  "id": "day01",
  "desc": "Time format preference",
  "rounds": [
    {
      "id": "r1",
      "type": "file_check",
      "question": "Save meeting notes to tasks/day01/meeting.json.",
      "feedback": {
        "correct": "Format is correct!",
        "incorrect": "Please use ISO 8601 for time fields."
      },
      "eval": {
        "command": "python scripts/check_meeting.py day01/meeting.json",
        "expect_exit": 0,
        "expect_stdout": "OK"
      }
    },
    {
      "id": "r2",
      "type": "multi_choice",
      "question": "Which time format did you use?",
      "feedback": { "correct": "Correct!", "incorrect": "Review ISO 8601." },
      "eval": {
        "options": { "A": "ISO 8601", "B": "Unix timestamp", "C": "Plain text" },
        "answer": ["A"]
      }
    }
  ]
}
```

Feedback injection: each round (except the first) receives the previous round's
feedback prepended as `[Previous Feedback] <text>\n\n<question>`.  A standalone
feedback message is sent after the last round.

**Example**

```bash
metaclaw-bench infer \
  -i benchmark/data/metaclaw-bench/all_tests.json \
  -o /tmp/infer_out \
  -n 1
```

---

### `scoring`

Score inference results against correct answers.

```
metaclaw-bench scoring -i <input> -r <result_dir>
```

**Options**

| Flag | Required | Description |
|------|----------|-------------|
| `-i`, `--input` | yes | Path to `all_tests.json` |
| `-r`, `--result` | yes | Directory to search recursively for `infer_result.json` files |

Scoring rules:
- `file_check` rounds: scored from the `inline_score.passed` field written during inference.
- `multi_choice` rounds: extracts `\bbox{X}` from the agent response and compares to `eval.answer`.

**Example**

```bash
metaclaw-bench scoring \
  -i benchmark/data/metaclaw-bench/all_tests.json \
  -r /tmp/infer_out
```

---

### `report`

Generate an accuracy and token-usage report from scoring results.

```
metaclaw-bench report -r <result_dir> [-c <compaction_results.json>] [-o <output_dir>]
```

**Options**

| Flag | Required | Description |
|------|----------|-------------|
| `-r`, `--result` | yes | Directory containing `scoring.json` files |
| `-c`, `--compaction` | no | Path to `compaction_results.json` for token aggregation |
| `-o`, `--output` | no | Output directory for `report.json` and `report.md`; prints to terminal if omitted |

**Example**

```bash
metaclaw-bench report -r /tmp/infer_out -o /tmp/report_out
```

---

### `run`

Full pipeline: infer → scoring → report.

```
metaclaw-bench run -i <input> -o <output> [options]
```

Accepts the same options as `infer`.  If a `compaction_results.json` exists
alongside `all_tests.json` it is automatically picked up for the report step.
When multiple test sets are processed, a combined `reports.md` is written to
the output root.

**Options**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-i`, `--input` | yes | — | Path to `all_tests.json` or a directory of them |
| `-o`, `--output` | yes | — | Output directory |
| `-w`, `--workers` | no | `1` | Maximum concurrent tests |
| `-n`, `--retry` | no | `0` | Retries per failed question |
| `--scene-per-train` | no | disabled | Trigger `metaclaw train-step` every N scenes |
| `--memory` | no | off | Trigger memory ingest after each scene |
| `--memory-proxy-port` | no | `30000` | MetaClaw proxy port for memory ingest |

**Example**

```bash
# Run from the project root (MetaClaw/)
metaclaw-bench run \
  -i benchmark/data/metaclaw-bench/all_tests.json \
  -o /tmp/run_out
```

---

### `clean`

Remove `work/` isolation directories created by `infer`.

```
metaclaw-bench clean -p <root_dir>
```

**Options**

| Flag | Required | Description |
|------|----------|-------------|
| `-p`, `--path` | yes | Root directory to search recursively for `work/` directories |

**Example**

```bash
metaclaw-bench clean -p benchmark/data/metaclaw-bench
```
