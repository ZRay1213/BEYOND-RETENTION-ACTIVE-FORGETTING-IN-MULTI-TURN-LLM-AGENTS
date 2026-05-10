#!/usr/bin/env bash
set -e
DROP=$1
TAG=${2:-dl${DROP}}
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-14b
/root/miniconda3/bin/python harness_drop_latest.py \
  --dataset_file data/sharded_stage1_K25.json \
  --task_subset /root/autodl-tmp/DCC/multi_heldout.json \
  --N 2 --workers 2 --drop_frac $DROP \
  --log_folder logs_drop_latest_${TAG} --conv_type drop_latest_${TAG}
