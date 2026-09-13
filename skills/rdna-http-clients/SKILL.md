---
name: rdna-http-clients
description: use this when wiring OmO/OMP/LiteLLM/OpenCode to the fork.
---

# HTTP clients

OpenAI-compat **`/v1` only**. Clients are not dest maintainers.

## Surface

OmO / OMP / LiteLLM / OpenCode hit `/v1/chat/completions` or
`/v1/responses` and **stop**.

| May | Must not |
|---|---|
| Base URL, model name, API key | `rdna_extras` checkout |
| Chat Completions / Responses | hippihx `plan` / `bind` / `run` |
| HTTP soak through the server | HIP knobs, hsaco, `VLLM_*` tile gates |

Never pull **rdna_extras**, **hippihx**, or HIP binds into the
harness. Dest knobs stay on the server. Family registry:
[`rdna-arch-family`](../rdna-arch-family/SKILL.md).
