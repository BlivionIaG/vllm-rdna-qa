# Overview — Dest vs harvest vs Leave

Audience: dest maintainers. Not a kernel walkthrough.

Dest: [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Op zoo:
[`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx). Wiki:
[`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki).

## Dest

**Dest** is what `rdna_extras` may ship as the gfx1030 serve path.

- One consume path per op: extras `torch.ops` wrapping a dest HIP body
  **or** one hippihx V1 id. Not both Triton and HIP for the same op.
- gfx1030, wave32, `fdot2` / `sdot4`. See [01-isa-dest.md](01-isa-dest.md).
- Tile locks (LDS, banks, `__launch_bounds__`, `K_STEP`) must hold.
  See [02-tile-contracts.md](02-tile-contracts.md).
- Capture-safe. See [03-graph-qa.md](03-graph-qa.md).
- Residuals stay three tracks. See [04-residuals.md](04-residuals.md).

Dest is **not** “present on the branch.” A file on tip can still be
harvest, Later, or Leave.

## Harvest

**Harvest** is observed extras / foreign / wiki material that is
pertinent but **not dest-signed**.

Use harvest to:

- Record LDS / launch numbers on a hippihx tile README.
- Track env knobs, Later overlays, and soak cells.
- Cite wiki silicon notes.

Do not treat harvest as dest. Do not copy `csrc/rocm/*.cu` into hippihx
until extras can consume one V1 op. Dual copies are how serve bugs
accrete.

## Leave

**Leave** is an explicit reject. Examples already locked elsewhere:

- WMMA / MFMA / TMA on gfx1030 dest ([01-isa-dest.md](01-isa-dest.md)).
- ConfigH `K_STEP=64` as dest prefill ([02-tile-contracts.md](02-tile-contracts.md)).
- TP≤2 gate or Triton as dest ([04-residuals.md](04-residuals.md)).
- Second W4 family (a17t unique AWQ GEMM), produce/packers, AITER/CDNA
  paths, `HSA_OVERRIDE`.

Leave stays Leave until a specialist reopens it with a soak that beats
the current dest path. Do not silently resurrect.

## Take / Leave / Import

Three verbs. Do not invent a fourth (“maybe dest”, “unvalidated dest”).

| Verb | Meaning | Lands in |
|---|---|---|
| **Take** | Accept into dest extras as serve wiring or dest HIP consume. | `rdna_extras` |
| **Leave** | Reject. Stay harvest / wiki / foreign / Later. | nowhere dest |
| **Import** | Consume a zoo fatbin or packer artifact through a defined bind. Not a file dump. | hippihx fatbin + extras bind |

**Take** is serve-side. **Import** is zoo-side then extras rewire. A
kernel body migrate is Import only when all hippihx migrate gates hold
(tile locks filled, one V1 entry, extras calls it, extras copy deleted).

```text
harvest ──Take──► dest extras (wiring / dest HIP)
harvest ──Import─► hippihx tile + fatbin ──bind──► dest extras
harvest ──Leave──► wiki / Later / foreign
```

## Non-goals

- Not a second wiki. Not tok/s as dest. Not upstream vLLM PRs.
- Not a TP≤2 product gate. Dest must keep a TP=4 soak cell.
- Not collapsing residuals or family binds.

## Specialist TODOs

- `TODO(engine):` pin the dest tip SHA this playbook was last checked
  against. Update when extras moves.
- `TODO(engine):` one-line Take/Leave/Import verdict table for live
  extras HIP families (FA, W4A16, GDN, EXL3, causal_conv, leapdragon AR).
- `TODO(silicon):` confirm dest ROCm pin (7.14) still matches hippihx
  `HIPPIHX_ROCM_PIN`.
