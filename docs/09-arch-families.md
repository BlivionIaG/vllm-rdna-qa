# Arch families

Family **registry**. Bind hsaco / tiles stay in
[05-family-binds.md](05-family-binds.md). This page names the arches
and what must not alias.

Do **not** paste papers. Point at `arch-research/…/BRIEF.md` only.

## Registry — never PARK across

| Family | Class | Not |
|---|---|---|
| **`qwen4_exp`** | Qwen4 experimental serve (harvest / Leave as dest HIP) | Qwen3.x GDN+QSA |
| **Qwen3.x GDN+QSA** | Gated Delta Net + Qwen sparse attn | Glm5Next KDA+DSA |
| **Glm5Next KDA+DSA** | Linear/KDA state + DSA KV | Qwen GDN pages |
| **`deepseek_v4`** | Lightning / sparse MLA + indexer | `deepseek_v41` |
| **`deepseek_v41`** | **CED / CSA2** | `deepseek_v4` |

**Never PARK across.** A parked bind, cached graph, hsaco, or
scheduler slot for one row must not be reused for another. No
“close enough” alias (`glm_moe_dsa` ≠ Glm5Next; `deepseek_v4` ≠
`deepseek_v41`).

`TODO(engine):` dest registry file + exact arch strings. One row, one
id, no aliases.

## Hybrid memory — four heaps

On hybrid arches, these are **not** one cache:

| Heap | Owns | Not |
|---|---|---|
| **State rings** | GDN / KDA SSM + conv slots | KV pages |
| **KV pages** | DSA / FA paged KV | state rings |
| **Engram** | memory / retrieve table | indexer scores |
| **Indexer** | top-k / QSA / lightning scores | Engram rows |

Prefix-cache and chunked-prefill boundaries must split state vs KV
(same class as residual 3). Do not PARK an Engram buffer on an
indexer slot.

`TODO(engine):` dest heap owners (extras modules) per family. Four
columns, no merge.

## Attn binds

**Full / Reindex / Reuse ≠ lightning indexer ≠ QSA ≠ MLA.**

| Bind | Family use |
|---|---|
| Full / Reindex / Reuse | dest FA / page policy — not an indexer |
| Lightning indexer | `deepseek_v4` (not QSA) |
| QSA | Qwen3.x sparse — not MLA |
| MLA | DeepSeek sparse MLA / DSA-NoPE |

Do not retarget QSA hsaco onto lightning, or MLA onto GDN.

`TODO(silicon):` dest `torch.ops` / V1 id per bind. Separate objects.

## Leftover BF16 / MTP-DSpark

Leftover **BF16** and **MTP-DSpark** stay **off the expert path**.
Expert GEMM is dest W4 / DOT (`moe/routed`, `moe/shared`). Leftover
is `moe/leftover_bf16`. DSpark / MTP ride-along is serve, not a
second expert tile.

`TODO(engine):` dest gates that keep leftover + MTP off experts.

## BRIEF.md references (do not paste)

| Family | Brief |
|---|---|
| `qwen4_exp` | `arch-research/qwen4_exp/BRIEF.md` |
| Qwen3.x GDN+QSA | `arch-research/qwen3x-gdn-qsa/BRIEF.md` |
| Glm5Next KDA+DSA | `arch-research/glm5next-kda-dsa/BRIEF.md` |
| `deepseek_v4` | `arch-research/deepseek_v4/BRIEF.md` |
| `deepseek_v41` CED/CSA2 | `arch-research/deepseek_v41/BRIEF.md` |

`TODO(engine):` pin the arch-research tree URL. Paths only; no paper
dumps in this playbook.
