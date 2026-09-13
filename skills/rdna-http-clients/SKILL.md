---
name: rdna-http-clients
description: use this when wiring OmO/OMP/LiteLLM/OpenCode to the fork.
---

# HTTP clients

Depth:
[llm-tooling-wiki](https://github.com/BlivionIaG/llm-tooling-wiki).

1. **OpenAI-compat `/v1` only.** `/v1/chat/completions` or
   `/v1/responses`, then stop.
2. No `VLLM_*` / hippihx / HIP knobs in OmO / OMP / LiteLLM /
   OpenCode.
3. Gateway stays **above HTTP**. BYO providers. Client routing ≠
   engine.
