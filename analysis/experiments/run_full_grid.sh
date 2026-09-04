#!/bin/bash
# Run the full nested-CV grid: 8 configurations x 20 cells = 160 trainings.
#
#   2 embeddings (ESM-2, ESM-C 6B)
#     x boundary head {off, on}
#     x input adapter {off, on}
#
# Cells are resumable: a cell whose cell_result.json already exists is skipped,
# so re-running after an interruption picks up where it stopped.
#
# Usage (from repo root):  bash analysis/experiments/run_full_grid.sh [concurrency]
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs

QUEUE=analysis/experiments/nested_cv_queue.jsonl
CONC="${1:-1}"
N_FOLDS=5

run_cell() {
  local name="$1" base="$2" emb="$3" split="$4" o="$5" k="$6"
  if [ -f "runs/${name}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "[skip] ${name} outer${o}_inner${k}"
    return
  fi
  PYTHONPATH=. python3 analysis/experiments/train_nested_cell.py \
    --base "$base" --emb "$emb" --split "$split" \
    --out "$name" --outer "$o" --inner "$k" --n_folds "$N_FOLDS" \
    > "logs/${name}_o${o}_i${k}.log" 2>&1
  if [ ! -f "runs/${name}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "FAIL ${name} outer${o}_inner${k}" | tee -a logs/failures.txt
  fi
}

while IFS= read -r line; do
  [ -z "$line" ] && continue
  name=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['name'])"  "$line")
  base=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['base'])"  "$line")
  emb=$(python3  -c "import json,sys; print(json.loads(sys.argv[1])['emb'])"   "$line")
  split=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['split'])" "$line")
  echo "=== $name ==="
  running=0
  for o in $(seq 0 $((N_FOLDS-1))); do
    for k in $(seq 0 $((N_FOLDS-1))); do
      [ "$o" = "$k" ] && continue
      run_cell "$name" "$base" "$emb" "$split" "$o" "$k" &
      running=$((running+1))
      if [ "$running" -ge "$CONC" ]; then wait; running=0; fi
    done
  done
  wait
done < "$QUEUE"

echo "grid complete; see logs/failures.txt for any failed cells"
