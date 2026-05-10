#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
for i in 0 1 2 3; do
  LOGDIR=logs_random_${i}_heldout
  CONV=random_${i}_heldout
  rm -rf $LOGDIR
  echo "=== random_$i start: $(date) ==="
  /root/miniconda3/bin/python harness_random_$i.py \
    --dataset_file data/sharded_stage3_math100.json \
    --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
    --N 3 --workers 1 \
    --log_folder $LOGDIR --conv_type $CONV
  echo "=== random_$i done: $(date) ==="
done
