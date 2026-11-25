#!/usr/bin/env bash
cd "$(dirname "$0")/.."

export HF_HOME=/work/mimir/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p pipeline/training/logs

# TRAIN_COUNT  EPOCHS  MLP_LR  LLM_LR   (most important first)
RUNS=(
  #"76k 3 1e-4 1e-5"
  #"28k 3 1e-4 1e-5"
  "1k  5 1e-4 1e-5"
)



for r in "${RUNS[@]}"; do
  read -r TRAIN_COUNT EPOCHS MLP_LR LLM_LR <<< "$r"
  export TRAIN_COUNT EPOCHS MLP_LR LLM_LR
  LOG=pipeline/training/logs/run_${TRAIN_COUNT}_mlp${MLP_LR}_llm${LLM_LR}_$(date +%Y%m%d-%H%M%S).txt
  {
    echo "start: $(date)  |  $TRAIN_COUNT  epochs=$EPOCHS  mlp_lr=$MLP_LR  llm_lr=$LLM_LR"
    echo "commit: $(git rev-parse --short HEAD)"
    uv run pipeline/training/train.py || echo "RUN FAILED (exit $?)"
    echo "end: $(date)"
  } 2>&1 | tee "$LOG"
done