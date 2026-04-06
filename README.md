# Qwen3.5-35B-A3B 1M Context on L4 24GB

Run Qwen3.5-35B-A3B with **1 million token context** on a single NVIDIA L4 (24GB VRAM) using TurboQuant extreme KV cache quantization.

## Key Results

| Component | VRAM |
|-----------|------|
| Model weights (Q3_K_M) | 15,190 MiB |
| KV cache turbo3 (1M tokens) | 4,480 MiB |
| Compute buffer | 777 MiB |
| **Total** | **~20.5 GB / 23 GB** |

- **Prefill speed**: ~513 tok/s
- **Decode speed**: ~9 tok/s (end-to-end through proxy at 905K context)
- **KV compression**: 4.9x vs fp16 (turbo3 = 3.25 bits/val)
- **Quality**: PPL +1.1% vs q8_0 baseline

## How It Works

1. **Model quantization**: Q3_K_M (3-bit weights) reduces model from ~20GB to ~16GB
2. **TurboQuant KV cache**: turbo3 compresses KV cache from ~23GB (fp16) to ~4.5GB for 1M tokens
3. **YaRN RoPE scaling**: Extends 262K training context to 1M via position interpolation
4. **Low ubatch**: ubatch=128 reduces compute buffer from 3GB to 777MB

## Quick Start

### Docker

```bash
docker build -t qwen-1m .
docker run --gpus all -p 8080:8080 qwen-1m
```

### Manual

```bash
# 1. Build TurboQuant fork
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git
cd llama-cpp-turboquant-cuda
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# 2. Download model
pip install huggingface_hub
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF', 'Qwen3.5-35B-A3B-Q3_K_M.gguf', local_dir='./models')"

# 3. Start server
./build/bin/llama-server \
  -m models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
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
  --reasoning off

> **Note**: `--reasoning off` disables thinking output to save tokens.
```

## Proxy + UI

The proxy holds document context server-side and prepends it to each query, avoiding the need to send 1M+ tokens from the browser.

```bash
python3 proxy.py  # port 8082
```

UI is served via nginx on port 8081.

## Needle in a Haystack Test

See `niah/` directory for the NIAH benchmark script. Tests retrieval accuracy at different depths across 1M context.

## Architecture

```
Browser --> nginx (8081) --> proxy.py (8082) --> llama-server (8080)
                                  |
                          holds 1M tokens
                          of pre-tokenized
                          document context
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
