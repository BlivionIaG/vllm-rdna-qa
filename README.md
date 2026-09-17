# vllm-rdna-qa

Agent **skills** for maintainers of the gfx1030 vLLM dest fork. Clone
this repo and point your agents at `skills/*/SKILL.md`.

Not a book. Not a second kernel wiki. Depth lives in
[`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki)
and `arch-research/…/BRIEF.md`. Dest is
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Op zoo:
[`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx).

## Load

```bash
git clone https://github.com/BlivionIaG/vllm-rdna-qa.git
```

Point Cursor / other agents at each `skills/<id>/SKILL.md` (project
skills dir, or your agent's skill loader). Read a skill when its
description matches the task.

## Skills

| Skill | Fires |
|---|---|
| [`rdna-dest-review`](skills/rdna-dest-review/SKILL.md) | use this when reviewing or landing a dest commit on opengfx1030/vllm-rdna rdna_extras. |
| [`rdna-graph-qa`](skills/rdna-graph-qa/SKILL.md) | use this when debugging or reviewing cudagraph / HIP graph capture on gfx1030 dest. |
| [`rdna-silicon-gate`](skills/rdna-silicon-gate/SKILL.md) | use this when reviewing HIP / ISA / dot-product instructions on a dest kernel or fatbin for gfx1030. |
| [`rdna-hip-runtime`](skills/rdna-hip-runtime/SKILL.md) | use this when checking ROCm version pin, capture runtime, all-reduce mode, or cache wipe after a tip move on the gfx1030 dest fork. |
| [`rdna-arch-family`](skills/rdna-arch-family/SKILL.md) | use this when a new model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, M-RoPE, DeepSeek v4 / v4.1) on the gfx1030 dest fork. |
| [`rdna-http-clients`](skills/rdna-http-clients/SKILL.md) | use this when wiring a client (OmO, OMP, LiteLLM, OpenCode, Codex, custom) to the gfx1030 dest fork's HTTP server. |

Add a skill: [CONTRIBUTING.md](CONTRIBUTING.md).

## How these skills fit together

```
                  ┌─────────────────────────────┐
   new kernel ──▶ │ rdna-silicon-gate          │ ── tile / ISA / LDS review
                  └────────────┬────────────────┘
                               ▼
                  ┌─────────────────────────────┐
   dest commit ──▶│ rdna-dest-review            │ ── gate / family / soak matrix
                  └────────────┬────────────────┘
                               ▼
                  ┌─────────────────────────────┐
   graph fault ──▶│ rdna-graph-qa               │ ── capture-time invariants
                  └────────────┬────────────────┘
                               ▼
                  ┌─────────────────────────────┐
   runtime bug ──▶│ rdna-hip-runtime            │ ── ROCm pin, AR, cache wipe
                  └─────────────────────────────┘

   new family ──▶ rdna-arch-family (parallel to all of the above)

   client wiring ──▶ rdna-http-clients (orthogonal — never touches engine)
```

The first four cover **engine** work in roughly top-down order
(silicon → dest → graph → runtime). `rdna-arch-family` is the
side-rail that ties them together when a new model lands.
`rdna-http-clients` is the boundary that keeps engine work out
of client configs.
