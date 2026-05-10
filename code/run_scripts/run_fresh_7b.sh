#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-7b
TYPE=$1  # last or every
HARNESS=harness_fresh_${TYPE}.py
LOG=logs_fresh_${TYPE}_7b
CONV=fresh_${TYPE}_7b
/root/miniconda3/bin/python $HARNESS \
  --dataset_file data/sharded_stage1_K25.json \
  --task_subset /root/autodl-tmp/DCC/multi_heldout.json \
  --N 2 --workers 2 \
  --log_folder $LOG --conv_type $CONV
