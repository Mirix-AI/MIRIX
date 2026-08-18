#!/bin/bash
# Run v4 graph-memory eval on LoCoMo sample 0 (conv-26) end-to-end.
#
# Prereqs:
#   1. `.env` has OPENAI_API_KEY and MIRIX_ENABLE_GRAPH_MEMORY=true
#   2. Infra is up:    docker compose --profile graph up -d
#   3. Server is up:   http://localhost:8531/health returns 200
#
# Outputs:
#   evals/results/locomo/v4_<timestamp>/conv-26.json     — per-question records
#   evals/results/locomo/v4_<timestamp>/metrics.json     — LLM-judge accuracy
#
# Final stdout prints overall accuracy + per-category breakdown.

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- pre-flight ------------------------------------------------------------
if ! lsof -ti:8531 >/dev/null 2>&1; then
  echo "ERROR: nothing listening on :8531" >&2
  echo "       start the stack first: docker compose --profile graph up -d" >&2
  exit 1
fi

python - <<'PY' || { echo "ERROR: server /health not 200" >&2; exit 1; }
import sys, urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8531/health", timeout=5)
    sys.exit(0 if r.status == 200 else 2)
except Exception as e:
    print(f"  health probe: {e}", file=sys.stderr)
    sys.exit(3)
PY

if [ ! -f locomo10.json ]; then
  echo "ERROR: locomo10.json not in repo root" >&2
  echo "       download from the LoCoMo benchmark and place at $(pwd)/locomo10.json" >&2
  exit 1
fi

# ---- run -------------------------------------------------------------------
cd evals
TS=$(date +%Y%m%d_%H%M%S)
OUT="v4_${TS}"

echo "[1/2] running main_eval.py → results/locomo/${OUT}/"
python main_eval.py \
  --data ../locomo10.json \
  --limit 1 \
  --run-llm \
  --mirix_config_path ./configs/0201c.yaml \
  --output_path "$OUT"

echo
echo "[2/2] running organize_results.py (LLM judge)"
python organize_results.py "$OUT"

# ---- summary ---------------------------------------------------------------
echo
echo "========================================================"
echo "  Summary"
echo "========================================================"
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path("results/locomo") / sys.argv[1]
m = json.loads((out_dir / "metrics.json").read_text())
mt = m["metrics"]
cn = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain"}

print(f"  Output:           {out_dir}")
print(f"  Acc:              {mt['accuracy']:.4f}  ({mt['total_correct']}/{mt['total_judged']})")
print(f"  Avg latency:      {mt['average_answer_latency_seconds']:.2f}s")
print(f"  Total LLM calls:  {mt['total_answer_requests']}")
print()
print("  Per category:")
for cid, d in sorted(m["accuracy_by_category"].items(), key=lambda kv: int(kv[0])):
    name = cn.get(int(cid), str(cid))
    print(f"    {name:15s}  n={d['total_judged']:3d}  acc={d['accuracy']:.4f}")
PY
