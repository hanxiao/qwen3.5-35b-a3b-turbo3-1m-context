#!/usr/bin/env python3
"""
Needle-in-a-Haystack test for 1M context.
Inserts a secret code at various depths and tests retrieval.
"""
import json, urllib.request, time, sys

LLAMA_API = "http://localhost:8080"
NEEDLE = "The magic verification code for Project Nightingale is: TURBO3-JINA-2026-KVCACHE"
EXPECTED = "TURBO3-JINA-2026-KVCACHE"
QUERY = "What is the magic verification code for Project Nightingale? Report it exactly."

def tokenize(text):
    req = urllib.request.Request(
        f"{LLAMA_API}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["tokens"]

def query_model(system_text, user_text, max_tokens=100):
    req = urllib.request.Request(
        f"{LLAMA_API}/v1/chat/completions",
        data=json.dumps({
            "model": "qwen",
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start
    return result, elapsed

def build_haystack(base_text, target_tokens, needle_depth=0.5):
    """Build haystack with needle at specified depth (0.0=start, 1.0=end)"""
    base_tokens = len(tokenize(base_text[:10000])) * (len(base_text) / 10000)  # estimate
    copies = int(target_tokens / base_tokens) + 1
    
    needle_pos = int(copies * needle_depth)
    parts = []
    for i in range(copies):
        if i == needle_pos:
            parts.append(f"\n\n=== SECRET INFORMATION ===\n{NEEDLE}\n=== END SECRET ===\n\n")
        parts.append(base_text)
    
    return "".join(parts)

def main():
    base_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/system_prompt.txt"
    target_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    
    with open(base_file) as f:
        base = f.read()
    
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for depth in depths:
        print(f"\n{'='*60}")
        print(f"Testing depth={depth:.0%} ({target_tokens:,} tokens)")
        print(f"{'='*60}")
        
        haystack = build_haystack(base, target_tokens, depth)
        actual_tokens = len(tokenize(haystack[:50000])) * (len(haystack) / 50000)
        print(f"Haystack: ~{int(actual_tokens):,} tokens")
        
        result, elapsed = query_model(haystack, QUERY)
        content = result["choices"][0]["message"]["content"]
        found = EXPECTED in content
        prompt_tokens = result["usage"]["prompt_tokens"]
        prefill_s = result["timings"]["prompt_ms"] / 1000
        
        print(f"Response: {content[:200]}")
        print(f"Found: {'YES' if found else 'NO'}")
        print(f"Prompt tokens: {prompt_tokens:,}")
        print(f"Prefill: {prefill_s:.1f}s ({prompt_tokens/prefill_s:.0f} tok/s)")
        
        results.append({
            "depth": depth,
            "found": found,
            "prompt_tokens": prompt_tokens,
            "prefill_s": prefill_s,
            "response": content[:200]
        })
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["found"] else "FAIL"
        print(f"  Depth {r['depth']:.0%}: {status} | {r['prompt_tokens']:,} tokens | {r['prefill_s']:.0f}s")

if __name__ == "__main__":
    main()
