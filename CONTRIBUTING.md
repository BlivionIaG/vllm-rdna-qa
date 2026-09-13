# Contributing

This repo is the **maintainer QA playbook**. Short chapters. Specialists
expand stubs. Dest is
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`, not this tree.

## Where work belongs

| Change | Land in |
|---|---|
| QA gate, residual track, soak cell | **this repo** (`docs/`) |
| Tile contract, HIP ISA, fatbin | [`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx) |
| Serve wiring, capture, model hooks | dest `rdna_extras` |
| Silicon / occupancy / format notes | [`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki) |
| Upstream vLLM | **do not open PRs** |

Do not grow a second kernel wiki here. Cite the wiki. Link hippihx tile
READMEs. Keep chapters short.

## No tok/s as dest

Throughput numbers are **not** a dest gate. A soak may record them as
unvalidated lab notes. Dest-signed means: ISA match, tile contract,
graph QA, residual still split, family bind, soak cell green for
**correctness**.

Do not paste tok/s into dest extras, hippihx tile locks, or these
chapters as if they were a contract.

## Expand a stub

1. Keep the heading. Fill the `TODO` under it. Do not invent a parallel
   page for the same gate.
2. Engine specialist owns residuals, graph QA, family binds, soak,
   HTTP clients, arch-family registry.
3. Silicon specialist owns ISA dest, tile contracts, and HIP runtime.
4. Mark dest-signed vs harvest vs Leave. Never collapse the three
   residuals ([docs/04-residuals.md](docs/04-residuals.md)).
5. Never land a **TP≤2 gate** or **Triton as dest**. Triton may exist as
   harvest / fallback; dest consume is one HIP / hippihx V1 op.

## Chapter style

- Headings first. Hard rules in tables or short lists.
- `TODO(engine):` / `TODO(silicon):` for work that is not dest-signed.
- Link dest commit / hippihx tile / wiki page. Do not dump `.cu`.
- One idea per subsection. If a section needs a kernel walkthrough, it
  belongs in the wiki.

## Attribution

Dest extras HIP is **BlivionIaG**. Foreign HIP keeps the source Author
and Committer. This playbook does not re-author kernels.
