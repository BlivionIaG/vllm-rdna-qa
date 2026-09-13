# Soak matrix

Correctness matrix for dest extras on the 4× V620 box. Cells are
**green / red / skip**, not tok/s. Tok/s may be logged as unvalidated
lab notes only.

Reference model: **Qwen3.8-27B AWQ INT4**, **16k context × 8** concurrent
(16k×8). That cell is the dest yardstick. Other models fill family
binds ([05-family-binds.md](05-family-binds.md)); they do not replace
this reference.

## Axes

**TP=2 vs TP=4** × **eager vs graph vs APC**.

|  | eager | graph | APC |
|---|---|---|---|
| **TP=2** | | | |
| **TP=4** | | | |

- **eager** — no cudagraph / piecewise capture.
- **graph** — captured decode (and captured W4 where dest claims it).
- **APC** — prefix / piecewise / automatic cache **on**. Residual 3
  lives here ([04-residuals.md](04-residuals.md)).

TP=2 is a **cell**, not a dest ceiling. TP=4 is required. A TP≤2-only
pass is not dest-signed.

## Reference cell

| Knob | Dest reference |
|---|---|
| Model | Qwen3.8-27B AWQ INT4 |
| Shape | 16k×8 |
| Precision | W4A16 consume (`fdot2`, ConfigA prefill) |
| Box | 4× V620, gfx1030, ROCm 7.14 |
| Pass | numerics + no capture abort + residuals still split |

`TODO(engine):` exact HF id / checkpoint, max seq, batch, KV dtype,
`VLLM_*` dest env set. Pin extras SHA.
`TODO(engine):` fill the six cells (TP=2/4 × eager/graph/APC) with
pass/fail + date. Red cells must name a residual (1, 2, or 3) — never
“graph+TP”.
`TODO(engine):` GDN / KDA / DeepSeek family soaks are **additional**
rows, not substitutes for the Qwen3.8-27B reference.

## How to call a cell

1. Record dest SHA, hippihx SHA (if bound), ROCm, wave size.
2. Eager first. Graph without eager-green is not a graph bug until
   eager is signed.
3. APC last. KV-only warmup is not an APC pass.
4. If TP=4 graph fails and TP=2 graph passes, that is residual **2**,
   not residual 1.
5. Do not close a cell with Triton dest or a TP≤2 gate.

## Out of matrix (still Leave as dest)

- tok/s as the pass condition.
- Multi-arch fatbin or `HSA_OVERRIDE` soaks.
- WMMA / AITER / CDNA boxes.
- Produce/pack (AWQ produce, EXL3 `-cb 3inst` Viterbi).
