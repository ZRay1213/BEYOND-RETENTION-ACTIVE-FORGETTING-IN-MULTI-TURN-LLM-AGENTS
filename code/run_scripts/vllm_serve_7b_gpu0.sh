#!/usr/bin/env bash
set -euo pipefail
MODEL=/root/autodl-tmp/DCC/models/Qwen2.5-7B-Instruct
PORT=${PORT:-8003}
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port $PORT \
    --served-model-name qwen2.5-7b \
    --gpu-memory-utilization 0.30 \
    --max-model-len 12288 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code
