#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-14b
/root/miniconda3/bin/python harness_fresh_last.py \
  --dataset_file data/sharded_f1_additive.json \
  --N 2 --workers 2 \
  --log_folder logs_f1_fresh_last --conv_type f1_fresh_last
