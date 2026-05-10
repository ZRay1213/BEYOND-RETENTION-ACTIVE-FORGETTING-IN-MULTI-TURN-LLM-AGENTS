#!/usr/bin/env bash
set -e
WHICH=$1
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
LOGDIR=logs_gepa_${WHICH}_heldout
CONV=gepa_${WHICH}_heldout
HARNESS=harness_gepa_g1_${WHICH}.py
rm -rf $LOGDIR
/root/miniconda3/bin/python $HARNESS \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
  --N 3 --workers 1 \
  --log_folder $LOGDIR --conv_type $CONV
