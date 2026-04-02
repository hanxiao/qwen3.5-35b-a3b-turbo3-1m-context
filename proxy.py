#!/usr/bin/env python3
"""
Proxy server for 1M context QA.
Holds pre-tokenized document context and prepends to each query.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, urllib.request, threading

LLAMA_API = "http://localhost:8080"
CONTEXT_FILE = "/tmp/system_prompt.txt"
TOKEN_IDS = []
READY = False
LOCK = threading.Lock()

def api_call(endpoint, data, timeout=900):
    req = urllib.request.Request(
        f"{LLAMA_API}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def init_context():
    global TOKEN_IDS, READY
    print("Loading context...")
    with open(CONTEXT_FILE) as f:
        context = f.read()
    print(f"Context size: {len(context)} chars")
    
    print("Tokenizing...")
    result = api_call("/tokenize", {"content": context, "with_pieces": False})
    TOKEN_IDS = result["tokens"]
    print(f"Context tokens: {len(TOKEN_IDS)}")
    READY = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        
        if not READY:
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Context not ready")
            return
        
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        
        # Prepend context to first message
        messages = body.get("messages", [])
        if messages and messages[0]["role"] == "user":
            # Insert system message with context
            messages.insert(0, {"role": "system", "content": f"__TOKEN_IDS__:{','.join(map(str, TOKEN_IDS))}"})
        
        body["messages"] = messages
        
        # Forward to llama-server
        result = api_call("/v1/chat/completions", body)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

if __name__ == "__main__":
    threading.Thread(target=init_context, daemon=True).start()
    server = HTTPServer(("0.0.0.0", 8082), Handler)
    print("Proxy listening on :8082")
    server.serve_forever()
