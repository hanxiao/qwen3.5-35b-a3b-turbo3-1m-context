#!/usr/bin/env python3
"""
Proxy for 1M context KV cache demo.
Qwen3.5-35B-A3B Q3_K_M + turbo3/turbo3 on L4 24GB.
Single corpus: tianlong_full.txt (905K tokens, already prefilled).
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
import urllib.request
import threading
import os
import time
import socket
import signal
import sys
import re

LLAMA_API = "http://localhost:8080"
READY = False

ACTIVE_TOKEN_IDS = []
BASE_TOKENS = 0

# GPU queue: only one request at a time
GPU_LOCK = threading.Semaphore(1)
GPU_QUEUE_SIZE = 0
GPU_QUEUE_LOCK = threading.Lock()
MAX_QUEUE = 10

CORPUS_INFO = {
    "name": "天龙八部 (全本)",
    "description": "金庸《天龙八部》完整全文 (~905K tokens)",
}


class ThinkingFilter:
    """Strip <think>...</think> blocks from streaming text."""
    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
    
    def filter(self, text):
        """Process incoming chunk, return filtered output."""
        if not text:
            return ""
        
        self.buffer += text
        output = ""
        
        # Process buffer until no more complete tags
        while True:
            if not self.in_thinking:
                # Look for opening tag
                start_match = re.search(r'<think>', self.buffer, re.IGNORECASE)
                if start_match:
                    # Output everything before tag
                    output += self.buffer[:start_match.start()]
                    self.buffer = self.buffer[start_match.end():]
                    self.in_thinking = True
                else:
                    # No opening tag - keep last 7 chars
                    if len(self.buffer) > 7:
                        output += self.buffer[:-7]
                        self.buffer = self.buffer[-7:]
                    return output
            else:
                # Inside thinking block, look for closing tag
                end_match = re.search(r'</think>', self.buffer, re.IGNORECASE)
                if end_match:
                    # Discard everything up to and including closing tag
                    self.buffer = self.buffer[end_match.end():]
                    self.in_thinking = False
                else:
                    # No closing tag yet - keep last 8 chars
                    if len(self.buffer) > 8:
                        self.buffer = self.buffer[-8:]
                    return output
    
    def flush(self):
        """Flush remaining buffer at end of stream."""
        if not self.in_thinking:
            output = self.buffer
            self.buffer = ""
            return output
        else:
            self.buffer = ""
            return ""

def api_call(endpoint, data, timeout=900):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{LLAMA_API}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def tokenize(text):
    result = api_call("/tokenize", {"content": text, "with_pieces": False})
    return result["tokens"]

def preload():
    global READY, ACTIVE_TOKEN_IDS, BASE_TOKENS
    
    print("Loading tianlong_full.txt...", flush=True)
    with open("/tmp/tianlong_full.txt", "r", errors="replace") as f:
        text = f.read()
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    
    system_instruction = (
        "\n\n---\n\n"
        "你是一个关于金庸小说《天龙八部》的问答系统。上面是小说完整全文。"
        "回答用户关于人物、情节、武功、对话的提问。"
        "规则:\n"
        "- 直接回答，不要输出 thinking 过程、不要输出 <think> 标签。\n"
        "- 引用原文时注明章节。\n"
        "- 优先直接引用原文。\n"
        "- 回答简洁准确，不要废话。\n"
    )
    base_text = "<|im_start|>system\n" + text + system_instruction + "<|im_end|>\n"
    
    print("Tokenizing...", flush=True)
    t0 = time.time()
    ACTIVE_TOKEN_IDS = tokenize(base_text)
    BASE_TOKENS = len(ACTIVE_TOKEN_IDS)
    print(f"Tokenized: {BASE_TOKENS} tokens in {time.time()-t0:.1f}s", flush=True)
    
    # Check if KV cache is already warm (slot has tokens)
    try:
        slots = json.loads(urllib.request.urlopen(f"{LLAMA_API}/slots", timeout=10).read())
        n_past = slots[0].get("n_past", 0) if slots else 0
        if n_past >= BASE_TOKENS - 100:
            print(f"KV cache already warm: {n_past} tokens in slot", flush=True)
            READY = True
            return
    except Exception as e:
        print(f"Slots check failed: {e}", flush=True)
    
    # Try slot restore
    try:
        result = api_call("/slots/0?action=restore", {"filename": "tianlong_1m_Q3KM_turbo3"}, timeout=30)
        if result.get("n_restored", 0) > 0:
            print(f"Restored slot cache: {result['n_restored']} entries", flush=True)
            # Warm the cache with a minimal query to establish prefix
            print("Warming prefix match...", flush=True)
            api_call("/completion", {
                "prompt": ACTIVE_TOKEN_IDS,
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": True
            }, timeout=1200)
            print("Cache warm!", flush=True)
            READY = True
            return
    except Exception as e:
        print(f"Slot restore failed: {e}", flush=True)
    
    # Full prefill
    print(f"Full prefill ({BASE_TOKENS} tokens)... this will take ~18 min", flush=True)
    result = api_call("/completion", {
        "prompt": ACTIVE_TOKEN_IDS,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True
    }, timeout=3600)
    
    speed = result.get('timings', {}).get('prompt_per_second', 0)
    print(f"Prefill done: {speed:.1f} tok/s", flush=True)
    
    # Save slot
    try:
        api_call("/slots/0?action=save", {"filename": "tianlong_1m_Q3KM_turbo3"}, timeout=60)
        print("Slot saved", flush=True)
    except Exception as e:
        print(f"Slot save failed: {e}", flush=True)
    
    READY = True
    print("Ready!", flush=True)

def try_acquire_gpu(timeout=60):
    global GPU_QUEUE_SIZE
    with GPU_QUEUE_LOCK:
        if GPU_QUEUE_SIZE >= MAX_QUEUE:
            return False
        GPU_QUEUE_SIZE += 1
    acquired = GPU_LOCK.acquire(timeout=timeout)
    if not acquired:
        with GPU_QUEUE_LOCK:
            GPU_QUEUE_SIZE -= 1
    return acquired

def release_gpu():
    global GPU_QUEUE_SIZE
    GPU_LOCK.release()
    with GPU_QUEUE_LOCK:
        GPU_QUEUE_SIZE -= 1

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "status": "ok",
                "ready": READY,
                "base_tokens": BASE_TOKENS,
                "corpus": CORPUS_INFO,
                "queue": GPU_QUEUE_SIZE,
            }
            self.wfile.write(json.dumps(resp).encode())
            return
        
        if self.path == "/" or self.path.startswith("/?" ) or self.path == "/index.html":
            try:
                with open("/home/hanxiao/qwen3.5-35b-a3b-turbo3-1m-context/index.html", "r") as f:
                    html = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode())
            except:
                self.send_response(500)
                self.end_headers()
            return
        
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        if not READY:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Still loading... prefilling 905K tokens (~18 min)"}).encode())
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        query = body.get("query", "")
        max_tokens = min(body.get("max_tokens", 2048), 4096)
        temperature = body.get("temperature", 0.3)

        if not try_acquire_gpu(timeout=120):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "GPU busy, try again."}).encode())
            return

        try:
            base_tokens = list(ACTIVE_TOKEN_IDS)
            query_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            query_tokens = tokenize(query_text)
            full_tokens = base_tokens + query_tokens

            payload = json.dumps({
                "prompt": full_tokens,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": True,
                "cache_prompt": True,
                "stop": ["<|im_end|>", "<|im_start|>"]
            }).encode()

            req = urllib.request.Request(
                f"{LLAMA_API}/completion",
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    buffer = b''
                    while True:
                        chunk = resp.read(1024)
                        if not chunk:
                            break
                        buffer += chunk
                        while b'\n\n' in buffer:
                            event, buffer = buffer.split(b'\n\n', 1)
                            for line in event.split(b'\n'):
                                if line.startswith(b'data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        chat_ev = {"choices": [{"delta": {"content": data.get('content', '')}, "index": 0}]}
                                        if data.get('stop'):
                                            chat_ev['usage'] = {
                                                'prompt_tokens': data.get('tokens_evaluated', 0),
                                                'completion_tokens': data.get('tokens_predicted', 0),
                                                'tokens_cached': data.get('tokens_cached', 0)
                                            }
                                        self.wfile.write(("data: " + json.dumps(chat_ev) + "\n\n").encode())
                                        self.wfile.flush()
                                    except:
                                        pass
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except Exception as e:
                try:
                    self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                except:
                    pass
        finally:
            release_gpu()

    def log_message(self, format, *args):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except Exception as e:
            print(f"Request error: {e}", flush=True)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()

if __name__ == "__main__":
    sys.stdout = open('/home/hanxiao/qwen3.5-35b-a3b-turbo3-1m-context/proxy.log', 'a', buffering=1)
    sys.stderr = sys.stdout
    
    threading.Thread(target=preload, daemon=True).start()
    
    server = ThreadedHTTPServer(("0.0.0.0", 8082), Handler)
    signal.signal(signal.SIGTERM, lambda *_: (server.shutdown(), sys.exit(0)))
    print(f"Proxy on port 8082 (pid {os.getpid()})", flush=True)
    server.serve_forever()
