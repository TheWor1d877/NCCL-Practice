"""decode 计时探针：向 vLLM 发一条流式请求，测 TTFT 与 TPOT。

用法:
    python bench_decode.py                 # 默认打 127.0.0.1:8000
    python bench_decode.py <url>           # 指定 completions 接口地址

输出含义:
    TTFT  请求发出到第一个 token 返回的时间
    TPOT  第一个 token 之后每个 token 的平均间隔（decode 延迟指标）
    decode_tok/s  1/TPOT
"""
import json
import sys
import time

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "写一个200字左右的小故事"}],
    "max_tokens": 256,
    "temperature": 0,  # 固定采样，两遍对比才可比
    "stream": True,
}

t0 = time.time()
ttft, n = None, 0
resp = requests.post(URL, json=payload, stream=True)
resp.raise_for_status()
for line in resp.iter_lines():
    if not line or not line.startswith(b"data: "):
        continue
    body = line[6:]
    if body == b"[DONE]":
        break
    delta = json.loads(body)["choices"][0]["delta"]
    if delta.get("content"):
        if ttft is None:
            ttft = time.time() - t0
        n += 1

tpot = (time.time() - t0 - ttft) / max(n - 1, 1)
print(f"tokens={n} TTFT={ttft * 1000:.1f}ms TPOT={tpot * 1000:.2f}ms decode_tok/s={1 / tpot:.1f}")
