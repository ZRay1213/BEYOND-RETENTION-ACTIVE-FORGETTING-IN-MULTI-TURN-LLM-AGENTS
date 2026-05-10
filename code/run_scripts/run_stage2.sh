#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
N=${N:-3}
WORKERS=${WORKERS:-8}
MODE=${MODE:-state_only}
TASKS=${TASKS:-math}
/root/miniconda3/bin/python run_simulations_state.py \
    --dataset_file data/sharded_stage1_K20.json \
    --models qwen2.5-14b \
    --system_model qwen2.5-7b --user_model qwen2.5-7b --tracker_model qwen2.5-7b \
    --tasks $TASKS \
    --mode $MODE \
    --N $N --N_workers $WORKERS \
    --log_folder logs_stage2
