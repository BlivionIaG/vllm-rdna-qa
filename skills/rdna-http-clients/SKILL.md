---
name: rdna-http-clients
description: use this when wiring OmO/OMP/LiteLLM/OpenCode (or any HTTP client) to vllm-rdna serve. Not for kernels, fatbins, or VLLM_* dest knobs.
---

# HTTP clients

Depth:
[llm-tooling-wiki](https://github.com/BlivionIaG/llm-tooling-wiki).
Families stay server-side:
[`rdna-arch-family`](../rdna-arch-family/SKILL.md).

1. **OpenAI-compat `/v1` only.** `/v1/chat/completions` or
   `/v1/responses`, then stop. Base URL + model id + key.
2. No `VLLM_*` / hippihx / HIP knobs in OmO / OMP / LiteLLM /
   OpenCode.
3. Gateway stays **above HTTP**. BYO providers. Client routing ≠
   engine.
