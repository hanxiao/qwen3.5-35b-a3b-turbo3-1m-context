#!/bin/bash
set -e

export LD_LIBRARY_PATH=/opt/llama-turbo/build/bin:$LD_LIBRARY_PATH

# Start llama-server
/opt/llama-turbo/build/bin/llama-server \
  -m /opt/models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -c 1048576 \
  -ngl 99 \
  -fa on \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  -np 1 \
  -ub 128 \
  -b 512 \
  --rope-scaling yarn \
  --rope-freq-scale 0.25 \
  --override-kv "qwen35moe.context_length=int:1048576" \
  --port 8080 \
  --host 0.0.0.0 \
  --chat-template-kwargs '{"enable_thinking":false}' &

# Wait for server
sleep 30

# Start proxy
python3 /opt/proxy.py &

# Start nginx
nginx -g "daemon off;"
