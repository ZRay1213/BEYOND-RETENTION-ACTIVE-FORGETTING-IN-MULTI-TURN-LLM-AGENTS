#!/usr/bin/env bash
MODEL=/root/autodl-tmp/models/_modelscope_cache/LLM-Research/Mistral-7B-Instruct-v0.2
TMPL=/root/autodl-tmp/DCC/mistral_chat_template.jinja
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port 8004 \
    --served-model-name mistral \
    --gpu-memory-utilization 0.17 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code \
    --chat-template "$TMPL"
