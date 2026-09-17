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
skills dir, or your agent’s skill loader). Read a skill when its
description matches the task.

## Skills

| Skill | Fires |
|---|---|
| [`rdna-dest-review`](skills/rdna-dest-review/SKILL.md) | use this when reviewing/landing a dest commit on opengfx1030/vllm-rdna rdna_extras. Not for HTTP clients, fatbin ISA, or wiki drafts. |
| [`rdna-graph-qa`](skills/rdna-graph-qa/SKILL.md) | use this when debugging or reviewing cudagraph/HIP graph capture on gfx1030 dest. Not for ROCm pin, fatbin ISA, or HTTP clients. |
| [`rdna-silicon-gate`](skills/rdna-silicon-gate/SKILL.md) | use this when reviewing HIP/ISA on a dest commit or fatbin. Not for landing classification, ROCm pin, or HTTP clients. |
| [`rdna-hip-runtime`](skills/rdna-hip-runtime/SKILL.md) | use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move. Not for tile ISA, family registry, or HTTP clients. |
| [`rdna-arch-family`](skills/rdna-arch-family/SKILL.md) | use this when a model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, deepseek_v4 vs v41). Not for dest landing, HIP/ISA, or HTTP clients. |
| [`rdna-http-clients`](skills/rdna-http-clients/SKILL.md) | use this when wiring OmO/OMP/LiteLLM/OpenCode (or any HTTP client) to vllm-rdna serve. Not for kernels, fatbins, or VLLM_* dest knobs. |

Add a skill: [CONTRIBUTING.md](CONTRIBUTING.md).
