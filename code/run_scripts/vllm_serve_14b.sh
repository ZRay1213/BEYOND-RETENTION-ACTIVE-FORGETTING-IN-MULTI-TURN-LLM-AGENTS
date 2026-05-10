#!/usr/bin/env bash
set -euo pipefail
MODEL=/root/autodl-tmp/models/Qwen2.5-14B-Instruct
PORT=${PORT:-8001}
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port $PORT \
    --served-model-name qwen2.5-14b \
    --gpu-memory-utilization 0.40 \
    --max-model-len 24576 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code
