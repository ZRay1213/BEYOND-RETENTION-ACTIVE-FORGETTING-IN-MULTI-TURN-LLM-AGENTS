#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-14b
export GENERATOR_MODEL=qwen2.5-7b
/root/miniconda3/bin/python harness_counterfactual.py \
  --source_logs_dir logs_multi_baseline --source_conv_type sharded \
  --N_per_task 2 --max_samples 30 --workers 4 \
  --log_folder logs_counterfactual_multi
