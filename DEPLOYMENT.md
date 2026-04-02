# Deployment Guide

## GCP L4 Deployment (Recommended)

### 1. Create Instance

```bash
gcloud compute instances create qwen-1m-l4 \
  --project=YOUR_PROJECT \
  --zone=us-west1-a \
  --machine-type=g2-standard-8 \
  --accelerator=count=1,type=nvidia-l4 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image=common-cu128-ubuntu-2204-nvidia-570-v20260305 \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=true" \
  --scopes=default
```

**Important**: Use standard (non-preemptible) instance. Spot instances get preempted frequently during long prefill operations.

### 2. Build and Deploy

SSH into the instance:

```bash
gcloud compute ssh qwen-1m-l4 --project=YOUR_PROJECT --zone=us-west1-a
```

Install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential git nginx python3-pip
```

Build TurboQuant:

```bash
cd /tmp
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git llama-turbo
cd llama-turbo
git checkout feature/turboquant-kv-cache
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j$(nproc)
```

Download model:

```bash
pip3 install huggingface_hub
mkdir -p ~/models
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF', 'Qwen3.5-35B-A3B-Q3_K_M.gguf', local_dir='/home/$(whoami)/models')"
```

Start server:

```bash
cd /tmp/llama-turbo
export LD_LIBRARY_PATH=/tmp/llama-turbo/build/bin:$LD_LIBRARY_PATH

nohup ./build/bin/llama-server \
  -m ~/models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
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
  --chat-template-kwargs '{"enable_thinking":false}' \
  > /tmp/server.log 2>&1 &
```

Wait for server to start (~30 seconds), then check:

```bash
tail -f /tmp/server.log
# Look for "server is listening on http://0.0.0.0:8080"
```

### 3. Deploy Proxy + UI

Upload your document context:

```bash
# On your local machine:
gcloud compute scp your_documents.txt qwen-1m-l4:/tmp/system_prompt.txt --project=YOUR_PROJECT --zone=us-west1-a
```

Start proxy:

```bash
python3 proxy.py &
```

Configure nginx for UI:

```bash
sudo cp nginx.conf /etc/nginx/sites-available/default
sudo cp -r ui/* /opt/ui/
sudo nginx -s reload
```

### 4. Test

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
```

## Docker Deployment

```bash
docker build -t qwen-1m .
docker run --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 qwen-1m
```

## Performance Tuning

### Prefill Speed

Default: ~513 tok/s on L4

To increase:
- Increase `-ub` (ubatch size) if VRAM allows
- Increase `-b` (batch size) for better GPU utilization
- Warning: Higher ubatch increases compute buffer VRAM usage

### Context Size

Maximum tested: 1,048,576 tokens (1M)

To increase beyond 1M:
- Requires more VRAM (A100 40GB+ recommended)
- Adjust `--rope-freq-scale` (lower = more extension)
- May degrade quality beyond 4x training context

### KV Cache Quantization

Current: turbo3 (3.25 bits, 4.9x compression)

Alternatives:
- turbo4: 4.25 bits, 3.8x compression, slightly better quality
- q4_0: 4 bits, standard, better quality but 3x larger
- turbo2: 2.5 bits (experimental), not recommended

## Cost

**GCP L4 Pricing** (us-west1):
- Standard: ~$0.86/hr (~$620/month if running 24/7)
- Spot: ~$0.26/hr (~$187/month) - NOT recommended due to frequent preemption

**Recommendations**:
- Use standard instance for production
- Use spot only for short experiments

## Troubleshooting

### OOM Error

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating XXXXX MiB on device 0: cudaMalloc failed: out of memory
```

Solutions:
1. Reduce `-ub` (ubatch) from 128 to 64
2. Reduce `-c` (context) from 1048576 to 700000
3. Use smaller model quantization (Q2_K instead of Q3_K_M)

### Slot Context Capped to 262K

```
srv    load_model: the slot context (1048576) exceeds the training context of the model (262144) - capping
```

Solution: Add `--override-kv "qwen35moe.context_length=int:1048576"`

### Server Crashes During Prefill

This is usually due to spot instance preemption. Use standard instance.

### Slow Prefill

Expected: ~513 tok/s on L4, ~33 minutes for 1M tokens

This is normal for 1M context. For faster prefill:
- Use A100 (~1200 tok/s)
- Or reduce context size
- Or cache KV state with slot save/restore

## Monitoring

Check GPU usage:

```bash
nvidia-smi
```

Check prefill progress:

```bash
tail -f /tmp/server.log | grep "progress"
```

Expected output during prefill:

```
slot update_slots: progress = 0.234567
```
