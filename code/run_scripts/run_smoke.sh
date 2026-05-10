#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
/root/miniconda3/bin/python run_simulations.py \
    --dataset_file data/sharded_smoketest.json \
    --models qwen2.5-14b \
    --system_model qwen2.5-7b \
    --user_model qwen2.5-7b \
    --tasks math code \
    --N_full_runs 1 \
    --N_concat_runs 0 \
    --N_sharded_runs 1 \
    --N_workers 4 \
    --log_folder logs_smoke \
    --verbose
