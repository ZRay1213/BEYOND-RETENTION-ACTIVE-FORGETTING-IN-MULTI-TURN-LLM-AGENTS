#!/usr/bin/env bash
set -euo pipefail
MODEL=/root/autodl-tmp/DCC/models/Qwen2.5-7B-Instruct
PORT=${PORT:-8002}
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port $PORT \
    --served-model-name qwen2.5-7b \
    --gpu-memory-utilization 0.28 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code
