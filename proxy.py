#!/usr/bin/env python3
"""
Proxy for 1M context QA.
Holds pre-tokenized document context in memory.
Each query: tokenize question, prepend base tokens, send to /completion with cache_prompt.
llama.cpp prefix-matches the base tokens instantly, only processes new query tokens.
"""
import json
import re
import time
import urllib.request
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_URL = "http://localhost:8080"
CONTEXT_FILE = "/tmp/system_prompt.txt"

BASE_TOKENS = []


def api(endpoint, data, timeout=14400):
    req = urllib.request.Request(
        f"{LLAMA_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def init():
    global BASE_TOKENS

    print("Loading context...")
    with open(CONTEXT_FILE) as f:
        text = f.read()
    print(f"Context: {len(text)} chars")

    print("Tokenizing...")
    t0 = time.time()
    result = api("/tokenize", {"content": text})
    BASE_TOKENS = result["tokens"]
    print(f"Tokens: {len(BASE_TOKENS)} in {time.time() - t0:.1f}s")

    print("Warming KV cache (cold prefill)...")
    t0 = time.time()
    result = api("/completion", {
        "prompt": BASE_TOKENS,
        "n_predict": 1,
        "cache_prompt": True,
        "temperature": 0,
    })
    elapsed = time.time() - t0
    cached = result.get("tokens_cached", 0)
    print(f"Cache warm: {cached} tokens cached in {elapsed:.1f}s")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        question = body.get("question", "")

        if not question:
            self._respond(400, {"error": "missing 'question' field"})
            return

        if not BASE_TOKENS:
            self._respond(503, {"error": "context not loaded"})
            return

        # Tokenize only the question
        q_result = api("/tokenize", {
            "content": f"\n\n请根据以上内容回答：{question}",
        }, timeout=30)
        q_tokens = q_result["tokens"]

        # Prepend base tokens, send with cache_prompt for prefix matching
        all_tokens = BASE_TOKENS + q_tokens
        t0 = time.time()
        result = api("/completion", {
            "prompt": all_tokens,
            "n_predict": 500,
            "cache_prompt": True,
            "temperature": 0,
        }, timeout=600)
        elapsed = time.time() - t0

        # Strip <think> tags
        content = result.get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        timings = result.get("timings", {})
        self._respond(200, {
            "answer": content,
            "elapsed_s": round(elapsed, 1),
            "prefix_match_ms": round(timings.get("prompt_ms", 0)),
            "decode_tok_s": round(timings.get("predicted_per_second", 0), 1),
            "tokens_cached": result.get("tokens_cached", 0),
        })

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _respond(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


class ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    init()
    print("Proxy listening on :8082")
    server = ThreadedServer(("0.0.0.0", 8082), Handler)
    server.serve_forever()
