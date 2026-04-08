# Qwen3.5-35B-A3B 1M Context on L4 24GB

Run Qwen3.5-35B-A3B with **1 million token context** on a single NVIDIA L4 (24GB VRAM) using TurboQuant extreme KV cache quantization.

## Key Results

| Metric | Value |
|--------|-------|
| **VRAM Budget** | |
| Model weights (Q3_K_M) | 15,190 MiB |
| KV cache turbo3 (1M tokens) | 4,000 MiB |
| Compute buffer (ubatch=128) | 779 MiB |
| Recurrent state buffer | 62.8 MiB |
| Checkpoints (32 x 62.8 MiB) | 2,010 MiB |
| **Peak VRAM** | **22,052 MiB / 23,034 MiB (95.7%)** |
| **Performance** | |
| Cold prefill (905K tokens) | 221 tok/s (4,112s total) |
| Slot restore time | 2.6s (905K tokens) |
| Decode @ 905K context | 8.8 tok/s |
| TTFT with append_to_slot | 0.53s (skips tokenization and cache update) |
| TTFT with full prompt resend | 6.3s (includes 3s tokenization + 3s cache update) |
| **Compression & Quality** | |
| KV compression vs fp16 | 5.12x (turbo3 = 3.25 bits/val) |
| Model quality (Q3_K_M) | 3.51 bpw |
| Thinking output | Disabled (`--reasoning off`) |

## How It Works

1. **Model quantization**: Q3_K_M (3-bit weights, 3.51 bpw) reduces model from 20GB to 15.2GB
2. **TurboQuant KV cache**: turbo3 compresses KV cache from 23GB (fp16) to 4GB for 1M tokens (5.12x compression)
3. **Native turbo3 Flash Attention**: Madreag/spiritbuun fork computes attention directly on turbo3 KV without decompressing to fp16, eliminating a ~1.7GB temporary buffer that would cause OOM. Enabled via `GGML_TURBO_DECODE_NATIVE=1`
4. **YaRN RoPE scaling**: Extends 262K training context to 1M via position interpolation
5. **Low ubatch**: ubatch=128 reduces compute buffer from 3GB to 779MB
6. **Slot save/restore**: One-time 68-minute prefill saved to disk (3.5GB slot file). Subsequent restores take 2.3s
7. **`append_to_slot` mode**: Query tokens are appended directly to the cached KV state without resending the 905K base tokens. Eliminates network transfer and tokenization overhead. Requires a patched llama-server (see `patches/append-to-slot.patch`)
8. **Fallback: text prompt + prefix matching**: Alternatively, the full document can be sent as text. llama-server tokenizes internally (~3s) and uses `cache_prompt` to match against the 905K tokens already in VRAM, only evaluating the new query tokens
9. **Hybrid/recurrent model patches**: Qwen3.5 uses a hybrid attention+recurrent architecture requiring patches for correct KV cache truncation and slot restore ([ggml-org/llama.cpp#20225](https://github.com/ggml-org/llama.cpp/pull/20225))
10. **Context checkpoints**: 32 checkpoints created every 8,192 tokens during prefill (62.8 MiB each), enabling efficient cache reuse

## Quick Start

### Manual

```bash
# 1. Build TurboQuant fork (Madreag/spiritbuun native FA)
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git
cd llama-cpp-turboquant-cuda
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# 2. Download model
pip install huggingface_hub
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF', 'Qwen3.5-35B-A3B-Q3_K_M.gguf', local_dir='./models')"

# 3. Start server
GGML_TURBO_DECODE_NATIVE=1 ./build/bin/llama-server \
  -m models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  -c 1048576 \
  -ctk turbo3 \
  -ctv turbo3 \
  -fa on \
  -ngl 999 \
  --port 8080 \
  --host 0.0.0.0 \
  --slot-save-path /home/hanxiao/slots \
  -np 1 \
  -ub 128 \
  --no-warmup \
  --reasoning off

# 4. Apply append_to_slot patch
cd llama-cpp-turboquant-cuda
git apply /path/to/patches/append-to-slot.patch
cmake --build build -j$(nproc)

# 5. Generate 1M token KV cache slot (~68 mins)
# Downloads the corpus and pre-computes the KV cache
python3 prefill.py

# 6. Start proxy (port 8082)
# Loads the saved KV cache and handles chat queries via append_to_slot
python3 proxy.py
```

**Key parameters:**
- `GGML_TURBO_DECODE_NATIVE=1`: Use native turbo3 Flash Attention (no KV decompression overhead)
- `-ctk turbo3 -ctv turbo3`: TurboQuant KV cache (5.12x compression)
- `--slot-save-path /home/hanxiao/slots`: Enable KV cache slot save/restore (3.5 GB file, 2.3s restore for 905K tokens)
- `--no-warmup`: Skip warmup prefill (use slot restore instead)
- `--reasoning off`: Disable thinking output to save tokens

## Optimal Sampling Parameters

Qwen3.5-35B-A3B is prone to repetition loops with default sampling settings. Use these [officially recommended parameters](https://github.com/QwenLM/Qwen3.5/issues/88):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | 0.7 | Non-thinking mode (thinking mode uses 0.6) |
| `top_p` | 0.8 | |
| `top_k` | 20 | |
| `min_p` | 0.0 | |
| `presence_penalty` | 1.5 | Critical for preventing repetition loops |
| `repetition_penalty` | 1.0 | No additional repeat penalty needed |

Sources:
- [Unsloth Qwen3 guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune) (thinking: temp=0.6, top_p=0.95, top_k=20)
- [Qwen3-VL guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) (presence_penalty=1.5)
- [QwenLM/Qwen3.5#88](https://github.com/QwenLM/Qwen3.5/issues/88) (community-confirmed)

## Proxy + UI

The proxy serves both the web UI and the chat API on port 8082. On startup it restores the saved KV cache slot. For each query, it sends ONLY the query tokens with `append_to_slot: true`, which appends them directly to the cached 905K KV state. After each request, the clean slot is restored for the next query.

```bash
python3 proxy.py  # serves UI + API on port 8082
```

## Needle in a Haystack Test

See `niah/` directory for the NIAH benchmark script. Tests retrieval accuracy at different depths across 1M context.

## Architecture

```
Browser --> proxy.py (8082) --> llama-server (8080)
                |                      |
         serves UI +            905K tokens KV cache
         sends only query       locked in VRAM via
         tokens (append mode)   slot restore + append_to_slot
```

## Hardware Requirements

- **GPU**: NVIDIA L4 24GB (or any GPU with 23+ GB VRAM)
- **CUDA**: 12.8+
- **RAM**: 16GB+ system RAM
- **Disk**: 20GB+ for model

## GCP Deployment

```bash
gcloud compute instances create qwen-1m-l4 \
  --project=YOUR_PROJECT \
  --zone=us-west1-a \
  --machine-type=g2-standard-8 \
  --accelerator=count=1,type=nvidia-l4 \
  --boot-disk-size=100GB \
  --image-family=common-cu128-ubuntu-2204 \
  --image-project=deeplearning-platform-release
```

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, 2025)
- [spiritbuun CUDA fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
