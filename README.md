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
| [`rdna-dest-review`](skills/rdna-dest-review/SKILL.md) | use this when reviewing or landing a dest commit on opengfx1030/vllm-rdna rdna_extras. Not for HTTP clients, fatbin ISA, or wiki drafts. |
| [`rdna-graph-qa`](skills/rdna-graph-qa/SKILL.md) | use this when debugging or reviewing cudagraph / HIP graph capture on gfx1030 dest. Not for ROCm pin, fatbin ISA, or HTTP clients. |
| [`rdna-silicon-gate`](skills/rdna-silicon-gate/SKILL.md) | use this when reviewing HIP / ISA / dot-product instructions on a dest kernel or fatbin for gfx1030. Not for landing classification, ROCm pin, or HTTP clients. |
| [`rdna-hip-runtime`](skills/rdna-hip-runtime/SKILL.md) | use this when checking ROCm version pin, capture runtime, all-reduce mode, or cache wipe after a tip move on the gfx1030 dest fork. Not for tile ISA, family registry, or HTTP clients. |
| [`rdna-arch-family`](skills/rdna-arch-family/SKILL.md) | use this when a new model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, M-RoPE, DeepSeek v4 / v4.1) on the gfx1030 dest fork. Not for dest landing, HIP/ISA, or HTTP clients. |
| [`rdna-http-clients`](skills/rdna-http-clients/SKILL.md) | use this when wiring a client (OmO, OMP, LiteLLM, OpenCode, Codex, custom) to the gfx1030 dest fork's HTTP server. Not for kernels, fatbins, or VLLM_* dest knobs. |

Add a skill: [CONTRIBUTING.md](CONTRIBUTING.md).

## How these skills fit together

```
   new kernel     ──▶ rdna-silicon-gate     (ISA / LDS / DOT)
   dest commit    ──▶ rdna-dest-review      (Dest / harvest / Leave / soak)
   graph fault    ──▶ rdna-graph-qa         (capture invariants)
   runtime / pin  ──▶ rdna-hip-runtime      (ROCm, AR, cache wipe)
   new family     ──▶ rdna-arch-family      (parallel; never PARK)
   client wiring  ──▶ rdna-http-clients     (orthogonal; OpenAI /v1 only)
```

These are **parallel fires**, not a pipeline. Depth stays in
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki).
`rdna-http-clients` keeps engine knobs out of client configs.
