# Family binds

Op families are **not** interchangeable. Dest bind is one family → one
hsaco / fatbin object. Do not retarget tiles across families.

Zoo directories are **classes**, not SKUs: `gdn_scan`, `kda_scan`,
`fa_fdot2`, … — never `qwen38/` or `glm53/`.

## Four families (do not fuse)

| Family | Scan / RoPE class | Tile class | Dest note |
|---|---|---|---|
| **Qwen GDN** | Gated Delta Net | `attn/gdn_scan` | Register-resident decode; prefill chain default-off |
| **GLM KDA** | Kimi / GLM Delta Attention | `attn/kda_scan` | Later. Drop `glm5_` filename. **≠ GDN** |
| **M-RoPE** | Multimodal / multi-section RoPE | serve + FA bind | Not a GDN/KDA scan |
| **DeepSeek 2D-RoPE** | YaRN / 2D inv-RoPE + indexer | `attn/qsa_indexer`, `attn/dsa_nope` | Sparse MLA class |

**Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE.**

Expanded registry (Qwen3.x / `qwen4_exp` / Glm5Next / `deepseek_v4` /
`deepseek_v41`): [09-arch-families.md](09-arch-families.md). Never PARK
across.

GDN 16/48 layouts do **not** retarget onto KDA 64×128. Causal conv
(`sequence/causal_conv`, `state_len≈4`) is **not** under `gdn_scan`.

## Separate hsaco

One `--offload-arch` per fatbin ([01-isa-dest.md](01-isa-dest.md)) **and**
one object per family bind.

- Do not ship a “universal scan” hsaco that `#ifdef`s GDN vs KDA vs
  MLA.
- gfx110x WMMA overlay is a **different object**, not a `#ifdef` on
  shared DOT.
- Product names (`qwen3`, `glm5`, `deepseek_v4`) stay in extras model
  hooks. Zoo / dest HIP symbols stay class names.

`TODO(engine):` dest bind table: model arch → family → `torch.ops` /
V1 id → fatbin path. One row per family, no aliases.
`TODO(silicon):` hsaco / archive names per family × dest arch
(`gfx1030`). Confirm no multi-family `.co`.
`TODO(engine):` M-RoPE vs DeepSeek 2D-RoPE serve hooks — which extras
files, which must not share inv-RoPE kernels.

## Leave

- a17t second W4 family as a “Qwen” bind.
- Naming tiles after a product to dodge this chapter.
- Loading GDN HIP for a KDA model (or the reverse) as dest.
