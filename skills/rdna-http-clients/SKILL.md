---
name: rdna-http-clients
description: use this when wiring OmO/OMP/LiteLLM/OpenCode to the fork.
---

# HTTP clients

1. Stay **OpenAI-compat `/v1` only**. Hit
   `/v1/chat/completions` or `/v1/responses` and **stop**.
2. Client may use base URL, model name, API key. Client must **not**
   checkout `rdna_extras`, import hippihx, or set HIP / `VLLM_*` tile
   knobs.
3. Never pull **rdna_extras**, **hippihx**, or HIP binds into OmO /
   OMP / LiteLLM / OpenCode. Dest knobs stay on the server.
4. Family names are server-side:
   [`rdna-arch-family`](../rdna-arch-family/SKILL.md).
