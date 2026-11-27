#!/usr/bin/env bash
# Judge every 10-epoch model on its own test set, one after another.
# usage: bash pipeline/judge_all.sh            (run from anywhere)
cd "$(dirname "$0")/.."

export HF_HOME=/work/mimir/hf_cache
export PYTHONUNBUFFERED=1
export JUDGE=1
N=300                                   # test samples per model

# TRAIN_COUNT  checkpoint folder (10-epoch runs, mlp lr 1e-4 / llm lr 1e-5)
MODELS=(
  #"1k  1k_20261125-142303"
  #"10k 10k_20261125-160807"
  "28k 28k_20261125-202607"
  "76k 76k_20261127-001436"
)

# bash pipeline/judge_all.sh

SUMMARY=pipeline/training/logs/judge_all_$(date +%Y%m%d-%H%M%S).txt
for m in "${MODELS[@]}"; do
  read -r TRAIN_COUNT DIR <<< "$m"
  export TRAIN_COUNT
  echo "=== $TRAIN_COUNT  ($DIR)  $(date +%H:%M) ==="
  uv run pipeline/training/eval_saved.py pipeline/training/model_saves/$DIR $N \
    | tee /dev/stderr | grep "judge acc" | sed "s/^/$TRAIN_COUNT  /" >> "$SUMMARY"
done 2>&1

echo; echo "── summary ──"; cat "$SUMMARY"
