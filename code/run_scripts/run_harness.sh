#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
DATASET=${DATASET:-data/sharded_stage3_math100.json}
SUBSET=${SUBSET:-}
N=${N:-3}
WORKERS=${WORKERS:-4}
LOGDIR=${LOGDIR:-logs_harness_g0}
CONV=${CONV:-harness_g0}
ARGS="--dataset_file $DATASET --N $N --workers $WORKERS --log_folder $LOGDIR --conv_type $CONV"
if [ -n "$SUBSET" ]; then ARGS="$ARGS --task_subset $SUBSET"; fi
echo "[harness] $ARGS"
/root/miniconda3/bin/python harness.py $ARGS
