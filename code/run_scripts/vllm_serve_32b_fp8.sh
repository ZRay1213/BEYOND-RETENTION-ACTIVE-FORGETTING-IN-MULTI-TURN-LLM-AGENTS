#!/usr/bin/env bash
set -euo pipefail
MODEL=/root/autodl-tmp/LADC/models/Qwen2.5-32B-Instruct
PORT=${PORT:-8001}
mkdir -p /root/autodl-tmp/DCC/outputs
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port $PORT \
    --served-model-name qwen2.5-32b \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --quantization fp8 \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
