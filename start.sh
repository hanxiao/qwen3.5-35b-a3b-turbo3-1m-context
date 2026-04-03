#!/bin/bash
set -e

export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH

# Start llama-server
/opt/llama.cpp/build/bin/llama-server \
  -m /opt/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --ctx-size 1048576 \
  --parallel 1 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --ubatch-size 256 \
  --checkpoint-every-n-tokens 256 \
  --slot-save-path /tmp/slots \
  --reasoning off \
  --host 0.0.0.0 --port 8080 &

echo "Waiting for llama-server..."
until curl -sf http://localhost:8080/health > /dev/null 2>&1; do
  sleep 5
done
echo "llama-server ready"

# Start proxy (blocks, handles requests)
python3 /opt/proxy.py
