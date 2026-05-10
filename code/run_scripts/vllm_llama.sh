#!/usr/bin/env bash
set -e
MODEL=/root/autodl-tmp/llama_model
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port 8004 \
    --served-model-name llama-3.1-8b \
    --gpu-memory-utilization 0.22 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code
