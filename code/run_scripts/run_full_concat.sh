#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
LOGDIR=${LOGDIR:-logs_preflight_full_concat}
N_FULL=${N_FULL:-2}
N_CONCAT=${N_CONCAT:-3}
WORKERS=${WORKERS:-8}
echo "[preflight] FULL=$N_FULL CONCAT=$N_CONCAT workers=$WORKERS logdir=$LOGDIR"
/root/miniconda3/bin/python run_simulations.py \
  --dataset_file data/sharded_stage3_math100.json \
  --models qwen2.5-14b \
  --system_model qwen2.5-7b --user_model qwen2.5-7b \
  --tasks math \
  --N_full_runs $N_FULL --N_concat_runs $N_CONCAT --N_sharded_runs 0 \
  --N_workers $WORKERS \
  --log_folder $LOGDIR
