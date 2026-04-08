#!/usr/bin/env python3
"""
Prefill and save KV cache slot for Qwen3.5-35B-A3B 1M Context Demo.
This script:
1. Downloads the corpus (Tianlong Babu)
2. Formats it with the system prompt
3. Sends to llama-server for full prefill (~68 minutes on L4)
4. Saves the KV cache slot to disk
"""
import json
import urllib.request
import time
import os

LLAMA = "http://localhost:8080"
SLOT_FILE = "tianlong_1m_Q3KM_turbo3"
CORPUS_URL = "https://gist.githubusercontent.com/zhenghaoz/7ad0948d5835468844abd2db8f67faff/raw/%E5%A4%A9%E9%BE%99%E5%85%AB%E9%83%A8.txt"
CORPUS_FILE = "tianlong_full.txt"

def wait_for_server():
    print(f"Waiting for llama-server at {LLAMA}...", flush=True)
    for _ in range(60):
        try:
            req = urllib.request.Request(f"{LLAMA}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                if data.get("status") in ["ok", "error"]: # Even error means HTTP is up
                    print("Server is up!", flush=True)
                    return True
        except:
            pass
        time.sleep(5)
    print("Timeout waiting for server.")
    return False

def main():
    if not wait_for_server():
        return

    print("1. Downloading corpus...", flush=True)
    if not os.path.exists(CORPUS_FILE):
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_FILE)
        print(f"Downloaded {CORPUS_FILE} ({os.path.getsize(CORPUS_FILE)/1024/1024:.1f} MB)")
    else:
        print(f"Using existing {CORPUS_FILE}")

    with open(CORPUS_FILE, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

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

    print("2. Tokenizing...", flush=True)
    t0 = time.time()
    req = urllib.request.Request(
        f"{LLAMA}/tokenize", 
        data=json.dumps({"content": base_text}).encode(), 
        headers={"Content-Type": "application/json"}
    )
    ids = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
    print(f"Tokenized: {len(ids)} tokens in {time.time()-t0:.1f}s")

    print(f"3. Starting full prefill (this will take ~68 minutes on L4)...", flush=True)
    t0 = time.time()
    req = urllib.request.Request(
        f"{LLAMA}/completion", 
        data=json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0.0, "cache_prompt": True}).encode(), 
        headers={"Content-Type": "application/json"}
    )
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        elapsed = time.time() - t0
        speed = r.get("timings", {}).get("prompt_per_second", 0)
        tc = r.get("tokens_cached", 0)
        print(f"Prefill done: {elapsed:.0f}s, {speed:.1f} tok/s, cached={tc}")
    except Exception as e:
        print(f"Prefill failed: {e}")
        return

    print("4. Saving slot to disk...", flush=True)
    req = urllib.request.Request(
        f"{LLAMA}/slots/0?action=save", 
        data=json.dumps({"filename": SLOT_FILE}).encode(), 
        headers={"Content-Type": "application/json"}
    )
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print(f"Slot saved successfully: {r}")
    print("\nPrefill complete! You can now start proxy.py")

if __name__ == "__main__":
    main()