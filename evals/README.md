# Evaluation of MIRIX on public benchmarks

This directory hosts several independent benchmarks:

| Benchmark | Where | How to run |
|---|---|---|
| **MetaClaw 30-day** (procedural skill learning) | `evals/metaclaw/` | See `evals/metaclaw/README.md` → "Reproducing a run" |
| **ALFWorld** (SkillOpt-aligned, online + frozen) | `evals/alfworld/` | See `evals/alfworld/README.md` → "Running" |
| **MAB / LongMemEval / RULER** | `evals/mab/` | `evals/mab/run_mab_longmem_eval.sh` and the per-suite `*_eval.py` scripts |
| **LoCoMo** (conversation memory) | this directory | Steps below |

The MetaClaw and ALFWorld harnesses exercise the procedural-memory skill
system; on the `eval/skill-test` branch the MIRIX core is byte-identical to
`feat/procedural-memory-main-pr`, so their results reflect exactly the code
proposed for `main`.

The remainder of this file is the LoCoMo evaluation guide.

0. Step 0 — dataset (not in git):
`main_eval.py` expects the LoCoMo-10 dataset at `evals/data/locomo10.json`,
which the broad `data/` gitignore rule keeps out of the repo. Download
`locomo10.json` from the official LoCoMo release
(https://github.com/snap-research/locomo) and place it there.

1. Step 1 — install (from the repo root):
Install uv with `brew install uv`, then run:
```
uv venv
source .venv/bin/activate # on windows, use `.\.venv\Scripts\Activate.ps1`
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

2. Step 2 — backend (from the repo root):
```
uv run python scripts/start_server.py
```

3. Step 3 — run (in another terminal tab, from the `evals/` directory —
`main_eval.py`, `./configs/0201c.yaml`, and `data/locomo10.json` are all
resolved relative to it):
```
cd evals
uv run python main_eval.py --limit 1 --run-llm --mirix_config_path ./configs/0201c.yaml --output_path results/0201c
```

4. Step 4 — metrics (still from `evals/`):
```
uv run organize_results.py results/0201c
```

Then there would be `metrics.json` in `results/0201c` where you can see all the metrics.
