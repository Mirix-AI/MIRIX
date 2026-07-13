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

1. Step 1:
Install uv with `brew install uv`, then run:
```
uv venv
source .venv/bin/activate # on windows, use `.\.venv\Scripts\Activate.ps1`
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

2. Step 2:
Start the backend:
In the `MIRIX` folder, run:
```
uv run python scripts/start_server.py
```

3. Step 3:
In another terminal tab, run:
```
uv run python main_eval.py --limit 1 --run-llm --mirix_config_path ./configs/0201c.yaml --output_path results/0201c
```

4. Step 4:
Evaluation. Run:
```
uv run organize_results.py results/0201c
```

Then there would be `metrics.json` in `results/0201c` where you can see all the metrics.
