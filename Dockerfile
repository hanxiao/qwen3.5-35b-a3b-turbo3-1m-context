FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake build-essential git python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with patches
WORKDIR /opt
COPY patches/ /opt/patches/
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git apply /opt/patches/remove-slot-cap.patch && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;80;86" -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --target llama-server -j$(nproc)

# Download model
RUN pip3 install huggingface_hub && \
    mkdir -p /opt/models && \
    python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF', \
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='/opt/models')"

COPY proxy.py /opt/proxy.py
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh && mkdir -p /tmp/slots

EXPOSE 8080 8082

CMD ["/opt/start.sh"]
