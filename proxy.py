#!/usr/bin/env python3
"""
Proxy for 1M context KV cache demo.
Qwen3.5-35B-A3B Q3_K_M + turbo3/turbo3 on L4 24GB.
Single corpus: tianlong_full.txt (905K tokens, already prefilled and saved as slot).

Architecture:
- On startup: restore saved slot (905K tokens KV cache, ~2.5s)
- On each request: send ONLY query tokens + n_keep=BASE_TOKENS
  llama-server keeps the base 905K KV cache locked in VRAM, only processes query tokens
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

LLAMA_API = "http://localhost:8080"
READY = False
BASE_TOKENS = 0
SLOT_FILE = "tianlong_1m_Q3KM_turbo3"

# GPU queue: only one request at a time
GPU_LOCK = threading.Semaphore(1)
GPU_QUEUE_SIZE = 0
GPU_QUEUE_LOCK = threading.Lock()
MAX_QUEUE = 10

CORPUS_INFO = {
    "name": "天龙八部 (全本)",
    "description": "金庸《天龙八部》完整全文 (~905K tokens)",
}


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
    global READY, BASE_TOKENS

    # Just restore the saved slot - no warm-up, no prefill
    try:
        result = api_call("/slots/0?action=restore", {"filename": SLOT_FILE}, timeout=30)
        n_restored = result.get("n_restored", 0)
        if n_restored > 0:
            BASE_TOKENS = n_restored
            print(f"Restored slot: {BASE_TOKENS} tokens in KV cache", flush=True)
            READY = True
            return
        else:
            print("ERROR: Slot restore returned 0 tokens!", flush=True)
    except Exception as e:
        print(f"ERROR: Slot restore failed: {e}", flush=True)
        print("Run full prefill + save slot first, then restart proxy.", flush=True)


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
            vram = {}
            try:
                import subprocess
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], timeout=5).decode().strip()
                used, total = out.split(', ')
                vram = {"used_gb": round(int(used)/1024, 1), "total_gb": round(int(total)/1024, 1)}
            except:
                pass
            resp = {
                "status": "ok",
                "ready": READY,
                "base_tokens": BASE_TOKENS,
                "corpus": CORPUS_INFO,
                "queue": GPU_QUEUE_SIZE,
                "vram": vram,
            }
            self.wfile.write(json.dumps(resp).encode())
            return

        if self.path == "/" or self.path.startswith("/?") or self.path == "/index.html":
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
            self.wfile.write(json.dumps({"error": "Slot not loaded. Check server logs."}).encode())
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        query = body.get("query", "")
        max_tokens = min(body.get("max_tokens", 2048), 4096)
        temperature = body.get("temperature", 0.7)

        if not try_acquire_gpu(timeout=120):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "GPU busy, try again."}).encode())
            return

        try:
            # Only send query tokens - base 905K tokens stay locked in KV cache via n_keep
            query_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            query_tokens = tokenize(query_text)

            payload = json.dumps({
                "prompt": query_tokens,
                "n_predict": max_tokens,
                "n_keep": BASE_TOKENS,
                "temperature": temperature,
                "stream": True,
                "cache_prompt": True,
                "stop": ["<|im_end|>", "<|im_start|>"],
                "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 1.5, "repeat_penalty": 1.0
            }).encode()

            req = urllib.request.Request(
                f"{LLAMA_API}/completion",
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            # Wait for slot to be free
            for _wait in range(30):
                try:
                    slots = json.loads(urllib.request.urlopen(f"{LLAMA_API}/slots", timeout=5).read())
                    if not slots[0].get("is_processing", False):
                        break
                except:
                    pass
                time.sleep(1)
                print(f"Waiting for slot to free... ({_wait+1}s)", flush=True)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            resp = None
            try:
                print(f"Query: {query[:50]}... ({len(query_tokens)} tokens, payload {len(payload)} bytes)", flush=True)
                resp = urllib.request.urlopen(req, timeout=300)
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
                                except (BrokenPipeError, ConnectionResetError):
                                    raise
                                except:
                                    pass
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                print("Done", flush=True)
            except (BrokenPipeError, ConnectionResetError):
                if resp:
                    resp.close()
                print("Client disconnected, upstream closed", flush=True)
            except Exception as e:
                try:
                    self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                except:
                    pass
            finally:
                if resp:
                    try: resp.close()
                    except: pass
        finally:
            release_gpu()

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format%args))
        sys.stderr.flush()

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
