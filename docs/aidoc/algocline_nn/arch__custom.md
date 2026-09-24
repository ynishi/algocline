# algocline-nn::arch::custom

Spec-driven customization points for the GPT-2 stack (Phase 1).

Generalizes the seam the dense-MoE block opened: instead of one
`Option` field per experiment, [`Gpt2Custom`] collects the
architecture axes a block can deviate from the GPT-2 reference on,
and `Block::new` builds from the spec. `Gpt2Config::custom = None`
keeps the stock GPT-2 behaviour bit-for-bit (same ops, same VarMap
names).

Phase 1 axes (VarMap-light: names stay identical or only gain
entries):

- **activation** — GELU (reference) / ReLU / SiLU, plus the gated
  variants SwiGLU / GeGLU from Shazeer 2020 (arXiv:2002.05202).
  Gated variants add a `mlp.c_gate` projection (the activated
  branch; `mlp.c_fc` stays the linear branch, Llama's `up_proj`).
- **norm** — LayerNorm (reference) / RMSNorm (Zhang & Sennrich
  2019, arXiv:1910.07467). RMSNorm keeps the `ln_*.weight` names
  and simply has no bias, and goes through the backward-safe
  `rms_norm_slow` shim like TinyLlama.
- **residual topology** — sequential (reference) / parallel
  attention + MLP (GPT-J / PaLM, arXiv:2204.02311):
  `y = x + attn(ln_1(x)) + ff(ln_2(x))`.
- **mlp_ratio** — the MLP expansion factor (reference 4).

Phase 2 axes (VarMap entries move or the attention wiring changes):

- **position** — learned `wpe` (reference) / RoPE (Su 2021,
  arXiv:2104.09864) / ALiBi (Press 2022, arXiv:2108.12409) / NoPE
  (Kazemnejad 2023, arXiv:2305.19466). Every non-learned variant
  drops the `wpe` Var; RoPE reuses TinyLlama's backward-safe
  `apply_rope` shim, ALiBi adds a constant per-head score bias.
- **GQA** — `kv_heads` shrinks the fused `c_attn` projection to
  `[dim + 2·kv·head_dim, dim]` and shares each KV head across
  `heads / kv_heads` query heads via `repeat_kv` (Ainslie 2023,
  arXiv:2305.13245).
- **sliding-window attention** — `window` bands the causal mask so
  position `i` attends to `(i - w, i]` (Mistral 2023).
- **untied head** — an independent `lm_head.weight` Var instead of
  reusing `wte`.
- **slot conditioning** — `cond_slots` adds a `cond_wte` table
  (`[slots, dim]`) whose selected row is added to the residual
  stream at every position, so the condition a row is generated
  under is an argument of the forward pass rather than a token the
  caller has to keep at a fixed offset.
- **allowed-id input** — `allowed_input` adds an `allowed_wte`
  table (`[vocab, dim]`) whose mean over the ids permitted at a
  position is added to the residual stream there, so a model over a
  constrained id space is told what is available instead of having
  to infer it from the sequence.
- **Post-LN** — norm after the sublayer + residual add (Xiong et
  al. 2020, arXiv:2002.04745) instead of the Pre-LN reference. Its
  known training instability is a probe subject, not a defect.
  Post-LN combined with the parallel residual topology has no
  canonical wiring and is rejected at build time.

All axes are experiment equipment, so a config that sets `custom`
keeps the HuggingFace-hub loaders shut: [`super::gpt2::Gpt2Model::from_pretrained`]
and the merged exporter refuse it (same guard family as MoE). Bundles
written by this crate's own trainer (`VarMap::save`) *are* loadable —
they carry exactly the Vars the spec declares — which is why every
type in this module is `Serialize` / `Deserialize`: a Card records
its spec so the load path can rebuild the identical config.

The serde representation is the same lowercase vocabulary the Lua
bridge accepts (`"swiglu"` / `"rmsnorm"` / `"preln"` / `"nope"` /
...), so a Card's `custom` table reads the way the caller wrote it.

## Types

- `Activation` — MLP activation. `Gelu` is the GPT-2 reference; the gated variants
- `Gpt2Custom` — Architecture customization spec. `Default` reproduces the GPT-2
- `NormKind` — Block / final normalization kind.
- `NormPlacement` — Where the block norms sit relative to the sublayers (Xiong et al.
- `PosKind` — How the model injects position information.
- `ResidualKind` — How the block combines its two halves with the residual stream.

## Constants

- `ALLOWED_TABLE_PREFIX` — `VarBuilder` prefix of the allowed-id table
- `ALLOWED_TABLE_TENSOR` — Tensor the allowed-id table is stored under: the embedding weight
- `COND_TABLE_PREFIX` — `VarBuilder` prefix of the conditioning table
- `COND_TABLE_TENSOR` — Tensor the conditioning table is stored under: the embedding

