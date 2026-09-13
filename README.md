# vllm-rdna-qa

Maintainer landing / QA playbook for the gfx1030 vLLM dest fork.

This repo is **not** a second kernel wiki. Silicon notes, occupancy
dumps, and format contracts stay in
[`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki).
HIP tile contracts and fatbins live in the op zoo. Dest extras is serve
wiring only.

## Purpose

Gate what may land on dest. Chapters below are short: headings, hard
rules, and `TODO` markers for engine / silicon specialists. Do not treat
a stub as dest-signed. Do not paste tok/s into dest.

## Links

| Tree | Role |
|---|---|
| [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (`rdna_extras`) | **Dest.** Serve wiring, capture, model hooks. |
| [`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx) | **Op zoo.** Tile contracts, HIP ISA, per-arch fatbins. |
| [`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki) | Kernel / silicon wiki. Cite it; do not copy it here. |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | Upstream. Do not open dest PRs there. |

Dest hardware contract: **gfx1030** (Radeon PRO V620), **wave32**,
**ROCm 7.14**, 4× V620 PCIe. Inner ops: `fdot2` / `sdot4`. Leave WMMA /
MFMA / TMA.

## Playbook

1. [Overview — Dest vs harvest vs Leave](docs/00-overview.md)
2. [ISA dest — gfx1030 DOT, wave32, fatbins](docs/01-isa-dest.md)
3. [Tile contracts — LDS, banks, launch bounds, K_STEP](docs/02-tile-contracts.md)
4. [Graph QA — zeros, no D2H, arenas, `Tensor!`](docs/03-graph-qa.md)
5. [Residuals — three tracks, never collapse](docs/04-residuals.md)
6. [Family binds — GDN ≠ KDA ≠ M-RoPE ≠ 2D-RoPE](docs/05-family-binds.md)
7. [Soak matrix — TP × eager / graph / APC](docs/06-soak-matrix.md)
8. [HIP runtime — ROCm pin, capture, cache wipe](docs/07-hip-runtime.md)
9. [HTTP clients — OpenAI-compat only](docs/08-http-clients.md)
10. [Arch families — registry, never PARK across](docs/09-arch-families.md)

How to expand a stub: [CONTRIBUTING.md](CONTRIBUTING.md).
