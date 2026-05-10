#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
HARNESS_FILE=$1
CONV=$2
LOGDIR=$3
N=${N:-2}
WORKERS=${WORKERS:-3}
echo "[eval] $HARNESS_FILE → $CONV, $LOGDIR"
/root/miniconda3/bin/python "$HARNESS_FILE" \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/train_subset_10.json \
  --N $N --workers $WORKERS \
  --log_folder $LOGDIR --conv_type $CONV
