FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake build-essential git python3 python3-pip nginx curl \
    && rm -rf /var/lib/apt/lists/*

# Build TurboQuant fork
WORKDIR /opt
RUN git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git llama-turbo && \
    cd llama-turbo && \
    git checkout feature/turboquant-kv-cache && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;80;86" && \
    cmake --build build -j$(nproc)

# Download model
RUN pip3 install huggingface_hub && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF', 'Qwen3.5-35B-A3B-Q3_K_M.gguf', local_dir='/opt/models')"

# Copy proxy and UI
COPY proxy.py /opt/proxy.py
COPY ui/ /opt/ui/
COPY nginx.conf /etc/nginx/sites-available/default
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

EXPOSE 8080 8081 8082

CMD ["/opt/start.sh"]
