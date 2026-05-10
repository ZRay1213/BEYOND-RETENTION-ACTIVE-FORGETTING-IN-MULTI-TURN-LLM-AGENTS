#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
/root/miniconda3/bin/python harness_g4_v2c0.py \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
  --N 3 --workers 2 \
  --log_folder logs_g4v2c0_heldout --conv_type g4v2c0_heldout
