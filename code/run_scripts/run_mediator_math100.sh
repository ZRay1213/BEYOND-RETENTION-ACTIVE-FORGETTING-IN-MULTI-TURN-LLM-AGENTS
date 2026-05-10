#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
SCHED=${1:-NONE}
DATASET=${DATASET:-data/sharded_stage3_math100.json}
LOGDIR=${LOGDIR:-logs_stage3_a1_math100}
N=${N:-3}
WORKERS=${WORKERS:-4}
echo "[run_mediator] sched=$SCHED dataset=$DATASET logdir=$LOGDIR N=$N workers=$WORKERS"
/root/miniconda3/bin/python run_simulations_mediator.py \
  --dataset_file "$DATASET" \
  --schedule "$SCHED" \
  --models qwen2.5-14b \
  --mediator_model qwen2.5-14b \
  --system_model qwen2.5-7b --user_model qwen2.5-7b --tracker_model qwen2.5-7b \
  --tasks math \
  --N $N --N_workers $WORKERS --log_folder "$LOGDIR"
