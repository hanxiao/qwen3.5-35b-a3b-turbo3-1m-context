# Qwen3.5-35B-A3B: 1M Context on a Single L4 24GB

Run the full **1 million token context** of Qwen3.5-35B-A3B on a single NVIDIA L4 GPU (24GB VRAM). Load an entire novel, codebase, or paper collection into context and query it interactively.

## Results

Tested with the complete Chinese novel *天龙八部* (Demi-Gods and Semi-Devils) by Jin Yong - 1.27M characters, 905K tokens.

| Metric | Value |
|--------|-------|
| Cold prefill | 59.6 min (253 tok/s) |
| Warm query (prefix match) | 0.5-0.8s |
| Decode speed | 5.1 tok/s |
| Total query time | 15-45s |
| Slot save to disk | 5.0 GB, ~1s |
| Slot restore from disk | <1s |

## VRAM Budget

| Component | Size |
|-----------|------|
| Model weights (IQ2_M, 2.7 bpw) | 10,520 MiB |
| KV cache (q4_0, 1M tokens) | 5,760 MiB |
| Recurrent state (GDN) | 63 MiB |
| Compute buffer | 1,554 MiB |
| CPU-mapped model | 333 MiB |
| **GPU total** | **~17,900 / 23,034 MiB** |

## Architecture

```
Browser --> UI (8081) --> proxy.py (8082) --> llama-server (8080)
                               |
                        holds 905K tokens
                        in memory, prefix
                        matches on each query
```

1. **Proxy startup**: reads document, tokenizes via `/tokenize` endpoint, stores token IDs in memory
2. **Cache warm**: sends token IDs to `/completion` with `cache_prompt=true`, `n_predict=1` - one-time prefill
3. **Each query**: proxy tokenizes only the new question (~20 tokens), prepends base token IDs, sends to `/completion` with `cache_prompt=true`
4. **llama.cpp prefix-matches** 905K tokens instantly (<1s), only processes new query tokens
5. **Response** in 15-45s depending on answer length (decode at ~5 tok/s with 900K context)

## Requirements

- NVIDIA L4 24GB (or any GPU with 23+ GB VRAM)
- CUDA 12.x
- cmake, build-essential, git, python3

## Setup

### 1. Build llama.cpp with Patch

Qwen3.5 is a hybrid architecture (GDN + attention). Two patches are required:

**Patch A: Remove slot context cap** - llama-server caps slot n_ctx to `n_ctx_train` (262K), blocking 1M context.

```bash
# In tools/server/server-context.cpp, comment out line ~752:
# n_ctx_slot = n_ctx_train;
```

**Patch B: Fix hybrid recurrent KV cache reuse** ([PR #21099](https://github.com/ggml-org/llama.cpp/pull/21099)) - without this, every query re-processes the full 905K tokens instead of prefix-matching.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Apply patches (see patches/ directory)
git apply patches/remove-slot-cap.patch
git apply patches/hybrid-cache-reuse.patch

# Build
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(nproc)
```

### 2. Download Model

```bash
pip3 install huggingface_hub
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
                'Qwen3.5-35B-A3B-UD-IQ2_M.gguf',
                local_dir='./models')
"
```

**Why IQ2_M?** Q3_K_M (16GB) + 1M KV cache (5.7GB) + compute (1.5GB) = 23.2GB, exceeding L4's 23GB usable VRAM. IQ2_M (10.5GB) fits comfortably with 5GB headroom.

### 3. Start Server

```bash
mkdir -p /tmp/slots

./build/bin/llama-server \
  -m ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
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
  --host 0.0.0.0 --port 8080
```

Key flags:
- `--cache-type-k q4_0 --cache-type-v q4_0`: 4-bit KV cache quantization (3.75x compression vs fp16)
- `--checkpoint-every-n-tokens 256`: enables KV cache checkpointing for hybrid model cache reuse
- `--slot-save-path /tmp/slots`: enables slot save/restore to skip cold prefill
- `--reasoning off`: disables `<think>` output
- No `--rope-scaling` needed - Qwen3.5 uses native 10M RoPE freq_base

### 4. Start Proxy

```bash
# Upload your document
scp your_document.txt server:/tmp/system_prompt.txt

# Start proxy
python3 proxy.py  # port 8082
```

First startup warms the KV cache (cold prefill ~60 min for 905K tokens). Subsequent restarts with saved slot: <1s.

### 5. Slot Save/Restore

After the first prefill, save the slot to skip cold prefill on restart:

```bash
# Save
curl -X POST "http://localhost:8080/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_document"}'

# Restore (after server restart)
curl -X POST "http://localhost:8080/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_document"}'
```

## Query API

```bash
curl -X POST http://localhost:8082 \
  -H "Content-Type: application/json" \
  -d '{"question": "Who helped Qiao Feng at Juexian Manor?"}'
```

Response:
```json
{
  "answer": "Qiao Feng fought alone against hundreds...",
  "elapsed_s": 18.0,
  "prefix_match_ms": 535,
  "decode_tok_s": 5.1,
  "tokens_cached": 905364
}
```

## Why These Specific Choices

| Decision | Why |
|----------|-----|
| IQ2_M over Q3_K_M | Q3_K_M OOMs at 1M context on L4. IQ2_M (2.7 bpw) trades quality for fitting |
| q4_0 KV over turbo3 | turbo3 requires special fork build; q4_0 works on mainline llama.cpp |
| `/completion` over `/v1/chat/completions` | `/completion` accepts raw token IDs, enabling prefix matching |
| Proxy architecture | Browser can't send 905K token IDs (7MB payload hangs mobile); proxy holds tokens server-side |
| `--checkpoint-every-n-tokens 256` | Required for GDN hybrid model to reuse KV cache across queries |
| Standard over Spot | Spot instances get preempted during 60-min prefill |

## Known Issues

1. **IQ2_M quality**: 2.7 bpw quantization degrades output quality vs Q3_K_M or higher. Model sometimes hallucinates details. Acceptable for search/retrieval, not ideal for creative writing.
2. **Decode speed**: 5 tok/s at 900K context is slow. Attention computation scales with context length. No workaround on L4.
3. **`<think>` tags**: Even with `--reasoning off`, model occasionally outputs `<think></think>` tags. Proxy strips them.
4. **Hybrid cache reuse**: Requires [PR #21099](https://github.com/ggml-org/llama.cpp/pull/21099) patch. Without it, every query re-prefills the full 905K tokens (~60 min).

## GCP Deployment

```bash
gcloud compute instances create qwen-1m-l4 \
  --project=jinaai-dev \
  --zone=us-west1-a \
  --machine-type=g2-standard-8 \
  --accelerator=count=1,type=nvidia-l4 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image-family=common-cu128-ubuntu-2204 \
  --image-project=deeplearning-platform-release
```

Cost: ~$0.86/hr (~$620/month). Use standard, not spot.

## Comparison with Other Setups

Three configurations of Qwen3.5-35B-A3B on L4 24GB, each making different trade-offs within the same 23GB VRAM budget.

### What's the Same

All three share:
- Same GPU: NVIDIA L4 24GB
- Same model family: Qwen3.5-35B-A3B (hybrid GDN + MoE, 256 experts, 8 active, 3B active params)
- Same KV cache quantization: q4_0 (4-bit keys and values)
- Same llama.cpp backend
- `--flash-attn on`, `--n-gpu-layers 999`, `--parallel 1`

### What's Different

| | General Chat ([L4-deploy](https://github.com/hanxiao/Qwen3.5-35B-A3B-L4-deploy)) | 250K Papers QA (production) | 1M Novel QA (this repo) |
|---|---|---|---|
| **Purpose** | General chat + vision API | Jina research papers search | Full novel in-context QA |
| **Model quant** | Q4_K_M (20 GB) | Q4_K_M (20 GB) | IQ2_M (10.5 GB) |
| **Vision (mmproj)** | Yes (858 MB) | No | No |
| **Context size** | 98K | 262K (native max) | 1M (4x training length) |
| **KV cache size** | 540 MB | 1,440 MB | 5,760 MB |
| **VRAM headroom** | ~0.5 GB free | ~1.5 GB free | ~5 GB free |
| **Decode speed** | 67 tok/s | 18-22 tok/s | 5.1 tok/s |
| **Prefill** | On-demand | ~9 min (245K tokens) | ~60 min (905K tokens) |
| **Proxy** | None (direct API) | Yes (holds token IDs) | Yes (holds token IDs) |
| **Runtime** | Docker (official image) | Docker (official image) | Local build + patch |
| **Patches needed** | None | None | Slot cap removal |
| **Instance type** | Spot | Spot | Standard |
| **Cost** | ~$0.26/hr | ~$0.26/hr | ~$0.86/hr |

### Why Each Differs

Every difference is driven by a single constraint: **23 GB VRAM budget**.

| Trade-off | Reason |
|-----------|--------|
| IQ2_M instead of Q4_K_M | Q4_K_M (20 GB) + 1M KV cache (5.7 GB) = 25.7 GB. OOM. IQ2_M (10.5 GB) + 5.7 GB = 16.2 GB. Fits. |
| No mmproj | 858 MB saved. General chat needs vision; papers/novel QA doesn't. |
| Local build instead of Docker | Official image caps slot context to `n_ctx_train` (262K). 1M requires patching `server-context.cpp`. |
| Standard instead of Spot | 60-min cold prefill can't survive preemption. 250K prefill (9 min) is short enough to risk spot. |
| Proxy architecture | At 905K tokens, the prompt is 7 MB of token IDs. Browser can't send this per query. Proxy holds it in memory. |

### Quality Impact

IQ2_M (2.7 bits per weight) is a significant quality reduction vs Q4_K_M (4.5 bpw):
- Factual recall from context: generally correct (聚贤庄 test passed)
- Occasional hallucination of details not in source text
- Acceptable for search/retrieval over known documents; not ideal for creative or reasoning tasks
- If quality matters more than context length, use Q3_K_M on A100 40GB

### Is Anything Overkill?

No. Each configuration sits at a different point on the same Pareto frontier:

```
Quality ▲
        │  ● General (Q4_K_M, 98K, +vision)
        │
        │       ● Papers (Q4_K_M, 262K)
        │
        │              ● Novel (IQ2_M, 1M)
        └──────────────────────────────────► Context Length
```

Moving right on context length forces moving down on quality (smaller quant) and dropping features (no vision). There's no wasted VRAM in any configuration.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) - hybrid GDN + MoE architecture
- [Unsloth GGUF quantizations](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [Hybrid recurrent cache reuse fix](https://github.com/ggml-org/llama.cpp/pull/21099)
