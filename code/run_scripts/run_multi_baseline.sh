#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
/root/miniconda3/bin/python run_simulations.py \
  --dataset_file data/multi_heldout48.json \
  --N_full_runs 0 --N_concat_runs 2 --N_sharded_runs 2 \
  --tasks code actions math data2text summary \
  --models qwen2.5-14b \
  --system_model qwen2.5-7b --user_model qwen2.5-7b \
  --N_workers 1 --log_folder logs_multi_baseline
