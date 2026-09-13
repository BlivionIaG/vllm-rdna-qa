# HTTP clients

HTTP consumers stay **OpenAI-compat only**. They are not dest
maintainers and must not grow a HIP harness.

## Surface

OmO / OMP / LiteLLM / OpenCode (and any other gateway or GUI) hit:

- `/v1/chat/completions`, or
- `/v1/responses`

…and **stop**.

No extras import. No hippihx. No fatbin. No `torch.ops`. No env that
selects a HIP bind.

| May | Must not |
|---|---|
| Base URL, model name, API key | `rdna_extras` checkout |
| Chat Completions / Responses | hippihx `plan` / `bind` / `run` |
| Soak *through* the server | HIP binds, hsaco paths, `VLLM_*` tile gates |

Quality / soak that needs dest knobs lives in
[06-soak-matrix.md](06-soak-matrix.md) on the **server** side, not in
the client.

## Leave

- Pulling dest extras or hippihx into OmO / OMP / LiteLLM / OpenCode
  “to make gfx1030 work.”
- Client-side Triton / HIP fallback.
- Teaching a gateway the family registry
  ([09-arch-families.md](09-arch-families.md)).

`TODO(engine):` dest OpenAI-compat base URL + model id used by OmO /
OMP / LiteLLM / OpenCode. One row each. No HIP columns.
`TODO(engine):` confirm Responses vs Chat Completions per client; do
not invent a third route.
