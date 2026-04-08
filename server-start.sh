#!/bin/bash
GGML_TURBO_DECODE_NATIVE=1 /tmp/turbo3-cuda/build/bin/llama-server \
  -m /home/hanxiao/models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -c 1048576 -ctk turbo3 -ctv turbo3 \
  -fa on -ngl 999 --port 8080 \
  --slot-save-path /home/hanxiao/slots -np 1 -ub 128 --no-warmup \
  --reasoning off
