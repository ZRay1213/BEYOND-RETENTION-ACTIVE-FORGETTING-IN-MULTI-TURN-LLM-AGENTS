#!/usr/bin/env bash
MODEL=/root/autodl-tmp/llama_model
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port 8004 \
    --served-model-name llama-3.1-8b \
    --gpu-memory-utilization 0.60 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code
