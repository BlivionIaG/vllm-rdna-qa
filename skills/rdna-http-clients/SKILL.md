---
name: rdna-http-clients
description: use this when wiring a client (OmO, OMP, LiteLLM, OpenCode, Codex, custom) to the gfx1030 dest fork's HTTP server. Not for kernels, fatbins, or VLLM_* dest knobs.
---

# HTTP clients

The dest fork exposes an OpenAI-compatible HTTP API — that's it.
Anything that talks to vLLM upstream at `/v1/...` talks to the
dest fork the same way. This skill keeps engine work out of client
configs.

## 1. Public surface

Write endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat-style requests (most clients) |
| `POST /v1/completions` | raw prompt completion |
| `POST /v1/responses` | newer Responses API (some clients) |

Read endpoints:

- `GET /v1/models` — list-models / health probe. Call before sending
  prompts — confirms the server is up even if inference is broken.
- `GET /health` — readiness (some setups)
- `GET /metrics` — Prometheus-style

That's the entire public surface. There is no `/v1/dest/...`, no
`/v1/rdna/...`, no kernel-status endpoint.

## 2. What clients do NOT send

The fork honours standard OpenAI fields and a small set of vLLM
extras (`reasoning_parser`, `guided_decoding_backend`). It ignores
or 400s anything else fork-specific. Clients must not send:

- `VLLM_*` env vars — clients don't set server env
- `ROCM_*` / `HSA_*` / `HIP_*` / `flash_attention_*` — clients
  don't pick the kernel
- Fork-specific `extra_body` like `{"rdna": true, "fast_path": "w4a16"}`

If the fork's defaults are wrong for your workload, fix it in the
**server** launcher, not the client. Client stays OpenAI-compat.

## 3. Boundary

Clients sit above HTTP. The dest fork is one provider among many.
Routing, retries, model selection, and provider fallback live in
the client layer (or a gateway between client and server).

```
Client (OmO/OMP/LiteLLM/OpenCode/...)
  │   HTTP /v1/chat/completions
  ▼
Dest fork server (gfx1030)
```

The client does not know which GPU is behind the server. The
server does not know which client is in front. If you have
multiple backends (dest fork + llama.cpp + upstream vLLM), the
client picks per request by cost / latency / capability — not by
"which one is the RDNA one".

## 4. Wiring recipes

### `curl` / `httpx` / `requests`

```bash
curl -s http://<server>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-id>",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

`model` must match what the server has loaded. Use
`GET /v1/models` to discover it.

### Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://<server>:<port>/v1",
    api_key="ignored",
)
resp = client.chat.completions.create(
    model="<model-id>",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=256,
    temperature=0,
)
```

### LiteLLM

```yaml
model_list:
  - model_name: qwen-rdna
    litellm_params:
      model: openai/<model-id>
      api_base: http://<server>:<port>/v1
      api_key: ignored   # server doesn't auth by default
```

### OpenCode / Codex / Claude Code

Set the OpenAI-compat base:

```
OPENAI_API_BASE=http://<server>:<port>/v1
OPENAI_API_KEY=ignored
```

Or in OpenCode's `provider` config:

```json
{
  "provider": {
    "vllm-rdna": {
      "apiBase": "http://<server>:<port>/v1",
      "apiKey": "ignored"
    }
  }
}
```

### OmO / OMP

`provider.openai.apiBase = http://<server>:<port>/v1`. No
fork-specific config sections; the client doesn't need them.

## 5. Health probe

Before sending prompts:

```bash
# 1. Models — fast, 200 if worker loaded
curl -fsS http://<server>:<port>/v1/models | jq '.data[].id'

# 2. Tiny completion — confirms inference works
curl -fsS http://<server>:<port>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<id>","prompt":"The","max_tokens":4,"temperature":0}'
```

If (1) returns the model id and (2) returns non-empty
`choices[0].text`, the server is healthy. If (1) works but (2)
returns garbage / NaN / constant token / hangs, see
[rdna-graph-qa §7](../rdna-graph-qa/SKILL.md).

## 6. Probe hygiene

Many agent harnesses print `(<N>/<M>)` from a probe run, and
`tools/probe_greedy_correctness.py` prints `PASS=N/M`. **Read the
first number as passes**, not as failures — the inverted
interpretation is a recurring source of confusion.

## 7. Streaming

```python
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

SSE over HTTP, one JSON per `data:` line, ends with `data: [DONE]`.
Same semantics as upstream vLLM.

## 8. Concurrency

The client controls concurrency. The server caps via
`--max-num-seqs` and rate limits.

gfx1030 cudagraph can corrupt recurrent state at ≥8 concurrent
requests with shared long prefixes — keep `--max-num-seqs` capped
on the server. The client doesn't need to know why; it respects
the server's capacity.

For throughput probes, use `vllm bench serve` — server-side
benchmark, not client. The client just sends requests.

## 9. Failure modes

| Symptom | What to do |
|---|---|
| 502 / 503 | Server not ready. Retry with backoff. Check `GET /v1/models` |
| 200 with constant token ("duct", etc.) | Server up, kernel broken. [rdna-graph-qa §7](../rdna-graph-qa/SKILL.md) |
| 200 with empty `choices` | Prompt too long for context. Reduce input or raise `--max-model-len` on the **server** |
| 200 with NaN logprobs | Attention / sampler broken. [rdna-graph-qa §7](../rdna-graph-qa/SKILL.md) |
| 60-second timeout | Server hung. Check dmesg for `Runlist is getting oversubscribed` |
| Connection refused | Server down. Re-launch |

Log the symptom, route to the right server skill.
