#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
N=${N:-3}
WORKERS=${WORKERS:-3}
LOGDIR=${LOGDIR:-logs_llmclf_v2_heldout}
echo "[llmclf-v2] N=$N workers=$WORKERS logdir=$LOGDIR controller=14B+few-shot"
/root/miniconda3/bin/python run_llmclf_v2.py --N $N --N_workers $WORKERS --log_folder $LOGDIR
