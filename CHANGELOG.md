# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `algocline-nn` gains a `sampling` module carrying the Layer 1 of the
  Sampler 3-layer plan: a `Sampler` trait (`sample(&mut self, logits:
  &Tensor) -> Result<u32>`) plus three Rust default implementations —
  `GreedySampler` (argmax), `TemperatureSampler` (temperature-scaled
  multinomial), and `TopKTopPSampler` (top-k + nucleus + temperature).
  Every stochastic sampler carries its own `StdRng` seeded at
  construction, so a fixed seed + fixed logits stream reproduces the
  same tokens (save/load reproducibility). `temperature <= 0` degrades
  to greedy on both stochastic samplers to avoid dividing by zero.
  `top_k = Some(0)` and `top_p` outside `[0.0, 1.0]` are refused at
  `sample` time — a caller programming error, not something to hide
  behind a silent fallback. Layer 2 (constraint DSL — grammar / JSON
  schema / regex) and Layer 3 (schedule / mid-generation sampler swap
  from Lua) attach later as additional `impl Sampler` types (including
  a Lua-callback bridge on the engine side) without changing the trait.
- Sampler Layer 2 scaffold (`sampling::constraint`): a `Constraint`
  trait (`mask(prefix)` + `is_terminal(prefix)`) and a
  `ConstrainedSampler<S, C>` wrapper that masks logits before
  delegating to any Layer 1 sampler. Masks are sparse
  (`TokenMask::AllowAll` / `Deny(ids)` / `Allow(ids)`) so the common
  no-masking step passes the caller's tensor through untouched instead
  of materialising a vocab-sized mask per token. A mask that leaves no
  valid token (`Allow` of an empty set, `Deny` covering the full
  vocab) and out-of-range token ids are refused loudly rather than
  falling back to argmax. First constraint shipped:
  `StopTokensConstraint`, which never masks (the stop token itself is
  sampled, matching llama.cpp semantics) and flips `is_terminal` once
  the generated prefix ends on a stop token.
- `RegexConstraint`: constrained decoding against a regex pattern.
  The pattern is compiled once into a `regex-automata` dense DFA with
  full-match semantics (wrapped as `^(?:pattern)$` — `Anchored::Yes`
  alone only anchors the start, so an unanchored tail would keep every
  token alive after the first match) and `MatchKind::All` (leftmost-
  first would commit `a|ab` to `a` and deny the `b`). Each step walks
  the generated prefix to the current DFA state and keeps exactly the
  tokens whose surface bytes do not drive the DFA into a dead state;
  the mask picks whichever sparse representation is smaller.
  `is_terminal` fires once the prefix is a full match. Tokens that
  decode to an empty string are always denied (they cannot advance
  the DFA, so permitting them would loop forever). The vocabulary is
  supplied as plain `Vec<String>` — decoupled from any tokenizer type
  — and `HfTokenizer` gained `vocab_strings()` to produce it (one
  surface string per token id; empty entries are the tokenizers-crate
  normal case for special / gap ids, real decode failures propagate).
- `JsonSchemaConstraint`: constrained decoding against a JSON Schema
  subset, compiled schema → regex → the same `regex-automata` DFA
  machinery `RegexConstraint` uses (the two-stage design Outlines
  ships). Supported: `object` / `properties` / `required`, `string`,
  `integer`, `number`, `boolean`, `null`, `enum` (members
  type-checked when `type` is declared alongside), and `array` /
  `items` including nesting. Every unsupported keyword (`$ref`,
  `anyOf`, `pattern`, `additionalProperties`, annotations like
  `description`, …) is rejected loudly by allowlist — silently
  ignoring a keyword would emit output the schema author believes is
  impossible. Known limitations, by design of the v1: all properties
  must be `required` (optional-property comma placement is deferred),
  output is compact JSON (no inter-token whitespace), and object keys
  are emitted in sorted order — sorted explicitly, because
  serde_json's `preserve_order` is an additive feature any transitive
  crate could flip, silently changing which documents the compiled
  DFA accepts. Depth (32) and compiled-regex size (1 MiB) guards turn
  would-be stack overflows on pathological schemas into errors.
- Token-by-token generation from Lua: `handle:generate_session(
  prompt_tokens)` on Llama adapter handles returns a `GenSession`
  that owns its own fresh KV cache — the model weights stay shared
  behind the existing `Arc`, but cache state is per-session, so two
  sessions over one handle cannot corrupt each other's attention
  history (the handle itself deliberately does not expose a raw
  `forward`; the session is the only way in). The session drives the
  loop from Lua: `session:next_logits()` forwards the pending tokens
  and returns an opaque `[vocab]` f32 `LogitsHandle`,
  `session:append(id)` records the sampled token,
  `session:tokens()` / `session:position()` report progress. Calling
  `next_logits` twice without an `append`, an empty prompt, and
  out-of-range token ids all error loudly. On the crate side,
  `LlamaAdapter` gained `new_cache()` and `forward_with_cache()`; the
  built-in-cache `forward` is unchanged and now documented as the
  single-loop legacy path.
- `alc.nn.tokenize(preset, text)` and `alc.nn.detokenize(preset,
  ids)`: the HuggingFace tokenizer wrapper (first-use download,
  cached under the app dir) is now reachable from Lua, closing the
  text → token ids → generation loop → text circle that
  `generate_session` needs.
- `alc.nn.chat_prompt(preset, messages)` and, on the crate side,
  `HfTokenizer::apply_chat_template(&[Message], add_generation_prompt)`:
  a conversation (`{ { role = "user", content = "..." }, ... }`) is
  rendered into the exact prompt string an instruction-tuned model was
  tuned on. The wrapping is model-specific and shipped by the model
  author as a Jinja2 program in `tokenizer_config.json`, so the
  template is rendered rather than reimplemented — `minijinja` is the
  new (minimal-feature) dependency behind it. The first-use tokenizer
  fetch now picks `tokenizer_config.json` up alongside `tokenizer.json`
  (cached as `<preset>-config.json`); a repo that ships none answers
  404 and stays fully usable for encode / decode, with only
  `apply_chat_template` refusing by name. `add_generation_prompt` is
  fixed to `true` from Lua (the caller wants a prompt to continue);
  the Rust API takes it as an argument. Roles are checked against
  `system` / `user` / `assistant` / `tool`, and a missing `role` /
  `content` names the offending index — a template branches on the
  role string, so an unrecognised one silently drops the turn from the
  prompt. Nothing degrades to a hand-built fallback: a prompt in the
  wrong shape yields plausible-looking garbage instead of a visible
  failure.
- Layer 3 of the sampler plan: samplers and constraints are now Lua
  values. `alc.nn.sampler.greedy()` / `.temperature(t, seed)` /
  `.top_k_top_p(k, p, t, seed)` build the Rust Layer 1 samplers;
  `alc.nn.sampler.lua(fn)` wraps a Lua function as a sampler (return
  values are validated — non-integers and out-of-vocab ids error);
  `alc.nn.sampler.constrained(inner, constraint)` composes with
  `alc.nn.constraint.stop_tokens(ids)` / `.regex(pattern, vocab)` /
  `.json_schema(schema, vocab)` (vocab is a tokenizer preset name or
  a plain table of strings). Composition is move semantics: the
  inner sampler and the constraint are consumed, and using a moved
  handle errors loudly — sharing one seeded sampler across two
  compositions would interleave its RNG stream. `LogitsHandle`
  gained `:top(n)` and `:argmax()` so a custom Lua sampler can read
  the distribution without marshalling the full vocab row per token.
  There is deliberately no schedule primitive: the generation loop
  already lives in Lua, so mid-generation sampler swaps are plain
  Lua control flow (`local s = pos < 20 and greedy or temp`).
- `algocline-nn` inference adapters now have a typed contract:
  `arch::adapter::InferenceAdapter` (`meta()` + `forward(tokens,
  index_pos)`), `AdapterMeta` (family / variant / shape parameters /
  device / dtype / logits shape), and `LogitsShape`
  (`LastToken` | `FullSeq`). The contract was previously prose in the
  module doc, which meant a second adapter had nothing to implement
  against. Construction stays outside the trait — the `VarBuilder`
  origin differs per arch (single file / sharded mmap / GGUF) and a
  `Self: Sized` constructor would rule out `dyn InferenceAdapter` — so
  a new adapter is one `InferenceAdapter` impl plus one `ARCH_OPS`
  entry on the bridge side.
- `LogitsShape` makes the adapter-vs-trainable output-shape difference
  a value instead of per-arch prose: trainable arches return
  `[batch, seq, vocab]`, the Llama adapter slices the final position
  and returns `[batch, vocab]`. `alc.nn` handles compute
  `forward_shape` from this rather than branching on the architecture.
- The typed `alc.nn` handles (`preset.gpt2` / `preset.tinyllama` /
  `preset.llama`) now expose the same accessor set as the arch-neutral
  `alc.nn.preset(arch, ...)` handle. Concretely, GPT-2 handles gained
  `kv_heads()` (mirrors `heads()`, since GPT-2 is multi-head attention)
  and Llama adapter handles gained `pretrained()` (always `true`, the
  adapter path being inference-only). Previously the available methods
  depended on which entry point built the handle. Additive: no existing
  accessor changed its name, arity, or value.
- Mixed-precision BF16 training (design §7.1): the trainer entrypoints
  (`alc.nn.trainer.full_ft` / `.lora` / `run_lora_ft` / `run_distill` /
  `alc.nn.wrap_lora`) now accept a BF16 base handle. The training loop
  picks its optimizer by parameter dtype — F32 keeps the stock
  candle-nn AdamW bit-identical, BF16 routes through the new
  `MixedAdamW` (FP32 master weights + FP32 moments, gradients upcast
  per step, updated master cast back into the live parameters), and
  the loss (log_softmax + NLL) is always scored in F32. LoRA adapters
  inherit the base dtype, so a BF16 LoRA / distill run goes through
  the same mixed path. BF16 checkpoints save as BF16 safetensors.
  Behavior changes: the trainer-entry guard that refused bf16
  (`training requires an f32 base`) is gone, and an **f16** base —
  previously unguarded at trainer entry — is now refused up front with
  a directional message (f16 needs loss scaling, which does not ship;
  bf16 shares f32's exponent range and trains without a scaler).
  Note candle 0.11 has no CPU BF16 matmul, so BF16 runs are CUDA-side
  in practice (the preset device/dtype matrix already enforces bf16 →
  CUDA); the offline regression fence covers the optimizer math
  (F32-parity vs stock AdamW, master-precision retention) and the
  full `run_full_ft` BF16 loop on a matmul-free toy model.
  Gradient checkpointing (the other half of design §7.1) is NOT
  included: candle 0.11 has no recompute-in-backward hook, so it is
  deferred to its own issue rather than shipped as a fragile autograd
  workaround.
- `alc.nn.data.parquet` now reads for real (the scaffold that surfaced
  `NotImplemented` on iteration is gone). The reader goes through the
  `parquet` crate's row API with `default-features = false` — no
  arrow-* tree — with codec features for snappy / zstd / gzip / lz4
  (a file using another codec fails loudly with the codec name). The
  column named `text_field` (default `"text"`) is tokenized exactly
  like the JSONL adapter (`tokenizer` opt, default `"gpt2"`), rows
  stream row-group by row-group, `shuffle = true` materialises rows
  first (same deterministic placeholder as JSONL), and `len_hint`
  reports the footer row count even when streaming. A wrong
  `text_field` fails at construction naming the available top-level
  fields; a present-but-non-string column errors with the column kind.
  Breaking for library consumers of `algocline-nn` only:
  `ParquetDataset::new` now takes the tokenizer and returns `Result`,
  and `DatasetError::NotImplemented` was replaced by
  `DatasetError::Parquet` (the Lua wire surface is unchanged —
  `alc.nn.data.parquet(path, opts)` keeps its signature and now also
  accepts the `tokenizer` opt the JSONL entry already had).
- `alc.nn.preset.gpt2("custom", { ... })` — Lua expose of the
  `Gpt2Custom` architecture spec (nn arch Phase 3). All Phase 1+2 axes
  are settable from a flat opts table (`act` / `norm` / `residual` /
  `mlp_ratio` / `placement` / `pos` / `kv_heads` / `window` /
  `untied_head`), a nested `moe = { n_experts, top_k?, alpha? }` table
  co-places the dense-MoE MLP, and `layers` / `heads` / `dim` / `ctx` /
  `vocab` override the tiny base shape (e.g. to match a real
  tokenizer's vocab). Custom models are random-init only: `pretrained =
  false` is required and the default `true` is rejected with a
  directional message. Every Rust-side validation error (PostLN ×
  Parallel, GQA divisibility, RoPE odd head_dim, MoE dense-knob
  combination) propagates to Lua as an actionable string; a present
  opts key of the wrong Lua type is a hard error naming the key and
  expected type, and custom-only keys passed to a stock variant
  (`"medium"` etc.) are rejected instead of silently not taking
  effect. The resulting handle trains through the existing
  `alc.nn.trainer.*` bindings unchanged. Regression fence:
  `nn_bridge_smoke.rs` gains the normal / representative-error /
  type-mismatch / MoE-composition quartet (the default `alc` binary is
  built without the `nn` feature, so the fence lives at the engine
  bridge layer that `alc_run` executes when the feature is on).
- `alc.nn.data.from_card` now recognizes the Tier 1 Card convention
  `[metadata] loss_mask = "response"`. For a declaring Card the sample's
  `prompt` field is tokenized separately to locate the token boundary,
  and the entry returns a `TeacherCardDataset` whose per-token loss mask
  is `0.0` over the prompt region and `1.0` over the response region, so
  `alc.nn.trainer.run_distill` scores only the response. Cards without
  the declaration are unchanged — same token ids, same mask-free
  `TokenizedDataset`, and `run_distill` still accepts them (an unmasked
  dataset degrades to a plain full fine-tune). An unrecognized
  `metadata.loss_mask` value is refused loudly instead of being ignored.
  The function signature and argument set are unchanged: the declaration
  travels with the Card, and the teacher log itself lives in the Card
  samples sidecar (`alc.card.write_samples`), so no raw corpus is
  committed to the repository. The mask stays internal to the training
  loop — the Lua batch table still exposes `input_ids` / `is_last` only.
- `crates/algocline-engine/tests/nn_distill_teacher_card_e2e.rs`: opt-in
  end-to-end test of the supply loop (`alc.card.write_samples` →
  `alc.nn.data.from_card` → `alc.nn.trainer.run_distill`) with the real
  gpt2 tokenizer, gated behind `NN_SMOKE_DISTILL_CARD=1`. Without the
  variable it prints a skip message and returns, so the default test run
  stays network-free.
- `crates/algocline-nn/tests/gpt2_grad_coverage.rs`: gradient-coverage
  gate for the GPT-2 trainable path. One masked-CE backward pass on a
  tiny from-scratch model must yield a non-zero gradient for every Var
  in the VarMap (28 on the tiny shape). The loss-threshold tests cannot
  localize which parameters learned: with the autograd graph severed
  inside the blocks (the candle-nn 0.11 LayerNorm fast-path cliff), the
  tied `wte` head still receives gradient through the logits matmul and
  can memorize a repeated corpus on its own. This gate fails with the
  offending parameter names instead, and doubles as the pin against a
  future candle bump routing around `apply_slow_layer_norm`.
- Dense mixture-of-experts feed-forward for the GPT-2 stack
  (`arch/moe.rs`, wired through the new `Gpt2Config::moe:
  Option<MoeConfig>` field). Every block's MLP can be swapped for a
  router + `n_experts` GPT-2-shaped experts with standard top-k routing
  (softmax the router logits, keep the top-k probabilities per token,
  renormalize, mix expert outputs; Switch Transformer for top-1,
  Mixtral for the top-2 default) plus the Switch §2.2 load-balancing
  aux term, exposed unscaled through the additive
  `Gpt2Model::forward_with_aux` / `forward_with_router_probs` methods —
  `forward` and every existing caller are unchanged, and a dense model
  returns `None` / no probs from the new methods. Compute is dense by
  design (every expert runs on every token): candle has no
  expert-dispatch kernel and a scatter/gather custom op would sit in
  the same no-backward `CustomOp` trap the LayerNorm / RMSNorm / RoPE
  slow-path shims exist for, while the dense mixture composes only ops
  with proper backwards. Expert parameters live under the new
  `h.<i>.moe.*` names (disjoint from `h.<i>.mlp.*`), initialized
  exactly like the stock MLP including the `1/sqrt(2·n_layer)`
  residual-write scaling per expert. MoE models are random-init only:
  `from_pretrained` / `from_safetensors_file` refuse a `MoeConfig`
  loudly, `export_merged` errors instead of emitting an incomplete
  bundle, and `wrap_lora` rejects the `up` / `down` targets on a MoE
  model (attention targets stay available) rather than silently
  training fewer parameters than requested.
- `tests/moe_grad_coverage.rs`: MoE version of the gradient-coverage
  gate, run in dense-mixture mode (`top_k = n_experts`) so structural
  reachability is separated from the legitimate sparsity of top-k
  routing — an unselected expert receiving no gradient in a step is
  routing, not a severed graph. One backward over `masked-CE + α·aux`
  must reach all 38 Vars of the tiny 2-layer / 2-expert shape, router
  included. Its first run caught the `softmax_last_dim` cliff below.
- `tests/tinyllama_grad_coverage.rs`: same full-inventory gate for
  TinyLlama (21 Vars on the tiny shape). Architecture-specific reason
  to exist: TinyLlama's standalone `q_proj` / `k_proj` have the
  attention softmax as their only gradient path, so this gate sees a
  severed scores path directly instead of through the fused-projection
  blind spot described below.
- Spec-driven architecture customization for the GPT-2 stack
  (`arch/custom.rs`, wired through the new `Gpt2Config::custom:
  Option<Gpt2Custom>` field) — the generalization of the seam the MoE
  block opened. Phase 1 axes, all built from ops with proper backwards:
  MLP activation (GELU reference / ReLU / SiLU, plus the gated SwiGLU /
  GeGLU from Shazeer 2020, which add a plain `mlp.c_gate` projection
  for the activated branch), normalization kind (LayerNorm reference /
  RMSNorm per Zhang & Sennrich 2019 — keeps the `ln_*.weight` names,
  registers no bias, and runs through the same backward-safe
  `rms_norm_slow` shim as TinyLlama), residual topology (sequential
  reference / parallel attention + MLP per GPT-J / PaLM), and the MLP
  expansion ratio. `custom: None` (all shipped presets) is bit-for-bit
  the reference architecture, and `Gpt2Custom::default()` registers
  exactly the same VarMap names as `None` (pinned by a test). Custom
  models are random-init only — the pretrained loaders and
  `export_merged` refuse them. LoRA keeps working on custom models
  (`up` / `down` wrap `c_fc` / `c_proj` as usual; `c_gate` is not a
  LoRA target).
- Phase 2 customization axes for the same `Gpt2Custom` spec — the
  VarMap-moving / attention-rewiring group: **position**
  (`pos`: learned `wpe` reference / RoPE per Su 2021 via TinyLlama's
  backward-safe `apply_rope` shim and the canonical θ=10000 cache /
  ALiBi per Press 2022 as a constant per-head additive score bias with
  the geometric slopes `2^(-8(h+1)/H)` / NoPE per Kazemnejad 2023 —
  every non-learned kind drops the `wpe` Var), **grouped-query
  attention** (`kv_heads`: the fused `c_attn` shrinks to
  `[dim + 2·kv·head_dim, dim]` under its unchanged name and KV heads
  are shared across query heads via the TinyLlama `repeat_kv` helper;
  `heads % kv_heads == 0` enforced at build), **sliding-window
  attention** (`window`: the cached causal mask is banded to the last
  `w` positions, Mistral convention), **untied LM head**
  (`untied_head`: independent `lm_head.weight` Var instead of reusing
  `wte`; design §4 sketched this axis as `tied_head` and the field is
  inverted so `Gpt2Custom::default()` stays the reference on every
  axis), and **norm placement** (`placement`: Pre-LN reference /
  Post-LN per Xiong et al. 2020 — norm the residual sum after each
  sublayer; its training instability is a probe subject, and Post-LN ×
  parallel residual is rejected at validation because the combination
  has no canonical wiring). RoPE requires an even head_dim (checked at
  build). The Phase-1 blanket rejection of `custom` + `moe` is relaxed
  to exactly the conflicting knobs: the combination now builds unless
  `act` / `mlp_ratio` are non-default (those address the dense MLP the
  MoE seam replaces), so norm / pos / GQA / window / untied-head /
  placement experiments compose with the MoE feed-forward.
- `tests/custom_grad_coverage.rs`: "kitchen sink" gradient-coverage
  gates — sink A turns every legally-combinable axis away from the
  reference at once (SwiGLU + RMSNorm + parallel residual + ratio 2 +
  RoPE + MQA + sliding window + untied head, 27 Vars), sink B covers
  the axes A cannot carry (Post-LN + ALiBi, 27 Vars), and a third gate
  certifies the custom × MoE composition (RMSNorm + RoPE + MQA over
  the expert mixture, 32 Vars including the router path via the aux
  term). Each is one masked-CE backward that must reach every Var in a
  pinned inventory. The per-Var walk itself moved to
  `tests/common/mod.rs` (`assert_full_grad_coverage`), shared by the
  GPT-2 / MoE / TinyLlama gates, so a future axis gets coverage by
  building a config and pinning a count instead of re-deriving the
  loop.
- `examples/arch_probe.rs`: generic architecture A/B probe — takes a
  list of `(label, Gpt2Config)` arms and reports checkpoint CE
  min/median/max over n independent draws plus each arm's trainable
  parameter count (param mismatches between arms stay visible). First
  measurement (4 layers / dim 256 / vocab 50257, 8 cycled rows, 120
  steps, lr 3e-4, n=5 per arm), recorded as-is: SwiGLU at ratio 3
  (16.31M params, +1.6% over the 16.04M baseline) ends at median CE
  2.252 (min–max 1.861–2.348) against the GELU baseline's 2.304
  (2.179–3.228) — the direction matches Shazeer 2020 and the SwiGLU
  arm's worst draw beats the baseline's by a wide margin, but the
  ranges overlap, so at this scale the result is "mildly favorable,
  not separated". RMSNorm (GELU, ratio 4, 16.04M params) ends at
  median 2.277 (2.196–2.491) — parity with LayerNorm within noise,
  consistent with the Zhang & Sennrich claim that RMSNorm trades
  nothing for its lower cost. Phase 2 arms (fresh n=5 run, same
  recipe; that run's own baseline: median 2.369, 2.188–2.620),
  recorded as-is: **RoPE** (16.03M params — no wpe) ends at median
  2.695 (2.579–2.931), *behind* learned wpe with barely-overlapping
  ranges — Su 2021's advantage does not reproduce on this
  fixed-length memorization task, where an absolute learned embedding
  is evidently the easier fit. **NoPE** ends at median 2.674
  (2.567–2.747), indistinguishable from RoPE here — consistent with
  Kazemnejad 2023 in the weak sense that the causal mask alone
  supports learning, and the gap to learned wpe shows position info
  still pays at this scale. **Post-LN** reproduces Xiong et al. 2020's
  instability claim in a readable form: it *leads* the Pre-LN baseline
  mid-run (median CE 3.457 vs 3.806 at step 80) and then destabilizes
  late — every draw finishes behind the baseline (median 2.908) and
  the worst draw diverges from 3.44 at step 80 up to 5.26 at step 119,
  exactly the no-warmup failure mode the paper predicts.
- `examples/moe_router_probe.rs`: router-behaviour probe in the
  `init_loss_probe` mold (n=5 independent draws per arm, checkpoint
  min/median/max) A/B-ing the load-balancing coefficient — `α = 0.01`
  (Switch §2.2) vs `α = 0` — on a 2-layer / 4-expert / top-2 / dim-256
  stack memorizing 8 cycled rows for 120 steps on CPU. Observed per
  checkpoint: CE, expert utilization (top-1 share per expert), routing
  entropy, and the raw aux value. Measured result, recorded as-is: the
  "no aux ⇒ router collapse" direction does **not** reproduce at this
  scale. The `α = 0` arm stays spread — median max-expert share 0.34
  (uniform = 0.25, collapse = 1.0) and median routing entropy 1.151
  (uniform = ln 4 ≈ 1.386, collapse = 0) at step 119 — and the
  `α = 0.01` arm is indistinguishable from it (median 0.32 / 1.099,
  ranges overlapping at every checkpoint; CE identical). A 120-step
  memorization run on a 2-layer stack evidently is not the regime where
  collapse develops; the aux term ships as standard equipment with the
  claim's small-scale test honestly negative, not as a measured
  improvement.

### Fixed

- The attention-scores softmax no longer severs the autograd graph.
  candle-nn 0.11's `ops::softmax_last_dim` is a `CustomOp1` registered
  via `apply_op1_no_bwd`: its output carries `BackpropOp::none()`, so
  backward silently treats it as a constant — the same cliff family as
  the LayerNorm / RMSNorm fast paths this crate already shims around.
  Both attention paths (`arch/gpt2.rs`, `arch/tinyllama.rs`) used it,
  which zeroed the Q/K gradient everywhere: on TinyLlama the standalone
  `q_proj` / `k_proj` (and any LoRA legs wrapped onto them) never
  learned at all, and on GPT-2 the Q and K rows of the fused `c_attn`
  never learned while the V rows kept the whole-Var gradient non-zero —
  which is exactly why the existing per-Var coverage gate could not see
  it. The hole surfaced the moment a parameter had the softmax as its
  *only* gradient path: the new MoE router failed
  `tests/moe_grad_coverage.rs` with `h.<i>.moe.router.weight` missing
  from the GradStore. All three sites (GPT-2 attention, TinyLlama
  attention, MoE router) now go through `arch::softmax_last_dim_slow`,
  the backward-safe `ops::softmax` basic-op composition; the GPT-2 gate
  additionally pins Q-row / K-row slice magnitudes of the fused
  `c_attn` gradient, and the new TinyLlama gate pins the standalone
  projections, so a future candle bump or forward refactor that routes
  back through a no-backward kernel fails loudly.
  `DatasetError::FullyMaskedRow`, carrying row index / `ctx_len` / mask
  length) any row whose loss mask keeps no scored position after the row
  is truncated to `ctx_len` and shifted against the targets (mask
  position 0 gates no target token). Previously such a row — a response
  fully cut off by `ctx_len`, a prompt long enough to fill the context
  on its own, or an all-zero mask — trained as a silent no-op: the
  fully-masked batch produces a loss of exactly 0.0 with no gradient,
  the step counter still advances, and `min_train_loss` records 0.0 as
  if the model had learned perfectly (a value that also trivially passes
  the `< ln(vocab)` E2E threshold). Rows where truncation trims only
  part of the response region remain accepted.
- `Gpt2Model::new` now draws `wte` / `wpe` from `N(0, 0.02)` — the GPT-2
  reference initialization — instead of inheriting candle-nn's
  `embedding()` default of `N(0, 1)`. The scale matters more here than
  in an untied model because the forward pass reuses `wte` as the LM
  head, so the logit scale is `sqrt(dim) * stdev(wte)`: at `stdev = 1.0`
  a from-scratch `gpt2-medium` produced logits with `std ~= 32` and a
  masked cross-entropy of ~140 at step 0, against the `ln(50257) ~=
  10.82` a uniform softmax gives. Training from that point saturated the
  softmax — on a repeated-token corpus the masked loss dropped to
  exactly `0.0` after one step and stayed there while the logits kept
  drifting on optimizer momentum. With the reference scale the same
  model starts at ~11 and descends normally. Only the random-init path
  changes; `from_pretrained` / `from_safetensors_file` load stored
  weights as before, and warm-started runs are unaffected.
  `run_lora_ft_reduces_loss_on_overfit_corpus` was recalibrated against
  the corrected baseline: its model is widened to `dim = 64` because
  LoRA cannot touch the tied head, and the threshold moved from
  `0.7 * baseline` to `0.75 * baseline`.
- `Gpt2Model`'s four per-block linear projections now follow the same
  GPT-2 reference initialization: weights from `N(0, 0.02)` and a zero
  bias, replacing candle-nn's Kaiming weight + uniform bias default,
  with the two projections that write back into the residual stream
  (`attn.c_proj`, `mlp.c_proj`) scaled to `0.02 / sqrt(2 * n_layer)` so
  the variance the stream accumulates does not grow with depth. Unlike
  the `wte` / `wpe` fix above this is a conformance change, and it is
  not a measured improvement — the A/B available here points the other
  way. On a 12-layer / dim-384 / vocab-50257 stack cycling 8 sequences
  for 120 steps at lr 3e-4, measured over `n = 5` independent draws per
  arm, both initializations start at `ln(vocab)` (median 10.92 vs
  10.97, the final LayerNorm normalizes the residual before the tied
  head) but the previous Kaiming draw converges faster: median 5.38 /
  2.92 / 1.32 (min–max 5.24–5.49 / 2.56–3.29 / 0.97–1.66) at steps
  40 / 80 / 119 against median 6.41 / 4.87 / 4.18 (min–max 6.18–7.13 /
  3.76–5.75 / 2.98–5.50) for the reference draw — the two arms' ranges
  do not overlap at any of the three checkpoints. That is the expected
  trade — the `1/sqrt(2 * n_layer)` scaling starts each block closer to
  the identity — and a 120-step memorization task is precisely the
  regime where the larger draw wins. The regime the scaling is for,
  long-horizon training of a deep stack, is not something this
  repository measures, so the justification for shipping it is
  conformance with the reference the module documents, not an observed
  gain. Reproduce with `examples/init_loss_probe.rs`. Random-init only,
  as above.

### Changed

- **Behavior change**: `LlamaAdapterConfig::use_kv_cache` now defaults to
  `true` in every constructor (`from_variant` for each variant, and
  `tiny()`). It previously defaulted to `false` while the engine bridge's
  `opts.use_kv_cache` defaulted to `true`, so a Rust caller building a
  config directly got different behaviour from a Lua caller reaching the
  same adapter through `alc.nn.preset.llama`. Lua callers see no change
  (the bridge default already won). A Rust caller relying on the old
  `false` default should set the field explicitly.
- `alc.nn` handle accessors are dispatched through a single arch-neutral
  projection (`HandleMeta`) instead of each accessor carrying its own
  match over the architecture arms, and `LlamaHandle` is built from the
  adapter's `InferenceAdapter::meta()` rather than from a clone of the
  upstream `candle_transformers` config. No caller-visible change; this
  removes the transcription step that a new architecture previously had
  to repeat across the adapter, its handle, and the neutral union.
- `algocline-nn` fetches HuggingFace artifacts through a small internal
  `ureq` client instead of `hf-hub`. Every call site already copied the
  downloaded file into its own cache directory and ignored the hub
  client's cache layout, so the dependency only provided a single-file
  HTTP GET. The new path also writes to `<dest>.partial` and renames on
  completion, so an interrupted download no longer leaves a truncated
  file that the `dest.exists()` first-use guard would accept as
  complete. `HF_ENDPOINT` and `HF_TOKEN` are honoured as before;
  `hf-hub`'s reuse of a pre-existing `~/.cache/huggingface` download is
  not (artifacts are re-fetched into the algocline cache on first use).

### Deprecated

### Removed

- **BREAKING** (matchers only): the `HubApi` variant of
  `algocline_nn::tokenizer::TokenizerError`,
  `algocline_nn::arch::gpt2::PretrainedError`, and
  `algocline_nn::arch::tinyllama::PretrainedError`. It reported hub
  *client construction* failure, which no longer has a failure mode
  after the `hf-hub` removal above; keeping an unconstructible variant
  would misdescribe the error surface. Transport failures continue to
  surface as `Download(String)`. Migration: fold any `HubApi` match arm
  into the `Download` arm. The `Download` / `HubApi` display strings
  also lost their `hf-hub ` prefix (`hf-hub download: …` →
  `hub download: …`).

### Fixed

- Downloading a tokenizer or pretrained weights from the HuggingFace hub
  failed outright on any machine without a warm cache, with
  `hub download: Bad URL: failed to parse URL: RelativeUrlWithoutBase`.
  The hub answers `/resolve/<rev>/<file>` with `307` and a *relative*
  `Location` (`/api/resolve-cache/models/...`); `hf-hub` 0.3–0.5 follow
  that redirect by handing the raw header value to the HTTP client
  without resolving it against the request URI, so the fetch aborted
  before reading a byte. Affected `HfTokenizer::load_cached` and both
  `Gpt2Model::from_pretrained` / `TinyLlamaModel::from_pretrained` — in
  practice every first-use path of `alc.nn` presets and
  `alc.nn.data.from_card`. Now handled by letting `ureq` follow its own
  redirects (see the Changed entry above).
- Return a directional Lua error when a bf16 base handle is passed to
  any of the four nn trainer entrypoints (`alc.nn.wrap_lora` /
  `alc.nn.trainer.run_lora_ft` / `run_full_ft` / `run_distill`),
  instructing users to rebuild the preset with `dtype="f32"`. Previously
  such calls surfaced an opaque candle
  `unexpected dtype, expected: F32, got: BF16` error deep inside a
  backward pass. bf16 remains supported for the inference path
  (`alc.nn.preset.*` on CUDA and downstream inference bindings are
  unchanged).
- Documentation drift around distillation datasets: the 0.46.0 entry
  advertised a `TeacherCardDataset` teacher adapter that no Lua caller
  could reach, and the `alc.nn.trainer.run_distill` example in
  `docs/lua-stdlib.md` built its dataset with `alc.nn.data.jsonl`, a
  path that provably cannot carry a loss mask. Both now describe the
  actual masked path.

### Security

## [0.47.2] - 2026-07-24

### Added

- `tests/common/mod.rs`: `TempAlcHome` RAII harness with per-child
  `Command::env("ALC_HOME", <tempdir>)` + `ALC_PACKAGES_PATH` injection.
  Guard exposes `client` / `home_path()` / `installed_json()` /
  `packages_dir()` / `cancel()`. Field-declaration-ordered drop
  (`client` → `_tmp` → `home`, per Rust Reference §5.8.1) guarantees
  the alc child terminates before the tempdir is removed. No
  `serial_test` crate / global `Mutex` is required because
  `std::process::Command::env` sets the variable only in the child
  process — parallel `cargo test` threads never race on parent-process
  env state.

### Changed

- `tests/e2e.rs`: seven e2e tests that used to leak into the developer's
  real `~/.algocline/` are now routed through `TempAlcHome::connect()`,
  eliminating the recurring pollution first documented in issue
  `213cad4a` (`installed.json` via pkg-family tests) and issue
  `efd45582` (`config.toml` via `test_alc_info` snapshot).
  Migrated tests: `test_alc_info` /
  `test_pkg_doctor_reports_installed_missing` /
  `test_alc_fork_roundtrip` / `test_pkg_install_returns_types_path` /
  `test_pkg_install_returns_alc_shapes_types_path` /
  `test_pkg_remove_scope_global_cleans_manifest_not_files` /
  `test_pkg_repair_reinstalls_deleted_dir`. Each `dirs::home_dir()`
  based cleanup / assertion path in these tests is replaced with the
  harness accessors (`harness.installed_json()` /
  `harness.packages_dir()`), and the `<tempdir>` prefix is stably
  redacted in the `alc_info` snapshot via the new
  `redact_paths_with_alc_home` helper. No production code changed.

### Fixed

- e2e test isolation: `cargo test --workspace` (and by extension
  `just ci` / `@alc-pre-push mode=full`) no longer adds
  `e2e_alc_shapes_types_test` / `e2e_doctor_pkg` / `e2e_fork_a` /
  `e2e_fork_b` / `e2e_repair_pkg` / `e2e_types_test` entries to the
  developer's real `~/.algocline/installed.json`. Previously each test
  run would re-introduce these six entries even after manual
  `alc_pkg_remove --scope=global` cleanup, requiring per-release
  cleanup ceremony (documented in the 2026-07-23 v0.47.1 release
  journal chapter).

## [0.47.1] - 2026-07-23

### Added

### Changed

### Deprecated

### Removed

- `BUNDLED_SOURCES` (`src/init.rs`) and `AUTO_INSTALL_SOURCES`
  (`crates/algocline-app/src/service/resolve.rs`) no longer include
  `algocline-swarm-frame`. The upstream repo moved to a `packages/`
  nesting layout (v0.9.0+) that is incompatible with the Collection-
  layout contract in `discover_packages`, so `alc update` / `alc init`
  were emitting a non-fatal `No packages found ... Expected
  subdirectories with init.lua` warning for it on every run. The
  swarm-frame V3 stack is deprecated in favor of the flow.ir + mse
  successor, so retaining the entry no longer justified the warning
  cost. Users still relying on swarm-frame V3 can install it manually
  via `alc_pkg_install <url>` once `discover_packages` grows nested-
  layout support (tracked in issue a31ad3ca).

### Fixed

### Security

## [0.47.0] - 2026-07-22

### Added

- `algocline-nn`: TinyLlama-1.1B trainable architecture (Layer 1 of GH
  #10, landed in two sub-phases).
  - Layer 1a — backward-safe primitives in `arch::tinyllama`:
    `apply_slow_rms_norm` (routes through `candle_nn::ops::rms_norm_slow`
    to keep the autograd chain intact — candle-nn 0.11 `RmsNorm::forward`
    is a `CustomOp3` with no backward, gated by
    `tests/rms_norm_autograd_gate.rs`), `apply_rope` (routes through
    `candle_nn::rotary_emb::rope_slow` for the same reason,
    `tests/rope_autograd_gate.rs`), `build_rope_cache` (canonical
    Llama-family `theta_i = base^{-2i/head_dim}` cos/sin cache), and
    `repeat_kv` (grouped-query-attention KV expansion).
  - Layer 1b — model:
    `arch::tinyllama::{TinyLlamaConfig, TinyLlamaModel}` (re-exported
    from `arch::mod`). Presets `tinyllama-1.1b` (22 layers, 32 heads,
    4 KV heads, dim 2048, hidden 5632, ctx 2048, vocab 32000,
    rope_theta 10000) and `tinyllama-tiny` (CPU smoke shape).
    `TinyLlamaModel::forward` returns `[batch, seq, vocab]` with GQA +
    RoPE + SwiGLU MLP and a cached causal mask.
    `TinyLlamaModel::from_pretrained` downloads via `hf-hub` and mmaps
    a `TinyLlama/TinyLlama-1.1B-*` safetensors bundle under the HF
    weight layout (`model.embed_tokens.weight`, `model.layers.<i>.*`,
    `model.norm.weight`, `lm_head.weight`). Verified by
    `tests/tinyllama_forward_shape.rs`.
  - Layer 2 — LoRA wrap surface:
    `TinyLlamaModel::wrap_lora(&LoraConfig) -> CandleResult<VarMap>` and
    `TinyLlamaModel::default_lora_targets()` (the 7 canonical HF Llama
    target names: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
    `up_proj`, `down_proj`). Wrap returns a fresh `VarMap` carrying
    only `lora_a` / `lora_b` params under `h.<i>.self_attn.<name>` /
    `h.<i>.mlp.<name>` — the base `VarMap` handed to `Self::new` is
    byte-identical before / after by construction so a training loop
    can hand only the returned `VarMap` to AdamW. Refactored
    `LinearVariant` + `wrap_variant_in_place` out of `arch::gpt2` into
    `arch::lora` (`pub(crate)`) as the shared substrate now consumed
    by both architectures. Verified by 5 inline unit tests
    (`arch::tinyllama::tests::wrap_lora_*` +
    `default_lora_targets_matches_canonical_seven`) and 2 integration
    tests (`tests/tinyllama_lora_merge_equivalence.rs`). Layer 3
    (training-loop generalisation) landed in the same `[Unreleased]`
    window — see below.
  - Layer 3 — LoRA fine-tune training loop generalisation for the
    trainable model family. The three trainer entry points
    (`train::run_ft_core`, `train::run_full_ft`, `train::run_lora_ft`)
    are now generic over `M: candle_nn::Module + DeviceView
    [+ LoraWrappable]` rather than monomorphic in `Gpt2Model`, so the
    same loop drives GPT-2 and TinyLlama through identical code
    paths. Two new local traits carry the shape:
    * `pub trait DeviceView { fn device(&self) -> &Device; }` in
      `train::mod` — pairs with `candle_nn::Module` so the loop can
      place input tensors on the model's device before the forward
      pass (`Module::forward` alone doesn't expose device info).
    * `pub trait LoraWrappable { fn wrap_lora(&mut self, &LoraConfig)
      -> CandleResult<VarMap>; }` in `arch::lora` — trait extraction
      of the previously-inherent `wrap_lora` method so `run_lora_ft`
      can dispatch uniformly.
    Both `Gpt2Model` and `TinyLlamaModel` receive `impl Module`,
    `impl DeviceView`, and `impl LoraWrappable` — all thin delegates
    to the existing inherent methods, so every existing call site
    (test suites, Lua bridge, external callers) resolves through the
    inherent path unchanged; only the generic loop dispatches through
    the traits. Verified by 4 new TinyLlama integration tests
    (`tests/tinyllama_lora_ft.rs`: freeze invariant, loss reduction
    on overfit corpus, LoRA weight movement, delta safetensors shape
    = 28 vars = 2 layers × 7 targets × 2) and 3 new inline unit tests
    for trait dispatch (`train::fullft::tests::gpt2_and_tinyllama_impl_*`).
    All 6 pre-existing GPT-2 LoRA integration tests re-monomorphise
    under the new generic bounds with no source change. Engine-side
    Lua bridge (`bridge/nn_card.rs`) updated to deref through
    `MutexGuard<Gpt2Model>` explicitly (`&*model` / `&mut *model`)
    since the guard doesn't itself impl the trait bounds. New public
    API surface is additive; the trainer entry-point signature change
    is only breaking for a downstream that spelled the concrete
    `Gpt2Model` type in a `fn` pointer or `type` alias (none known).
- `algocline-engine`: arch-neutral bridge for the `alc.nn.preset.*`
  / `alc.nn.card.load*` surface (Layer 4b of GH #10). Adding a new
  trainable arch used to require a full preset + load fn per arch
  on the Lua bridge (`alc.nn.preset.<arch>` + `alc.nn.card.load_<arch>`),
  driving an N×M proliferation across arch × training_path. Layer
  4b introduces a single dispatch registry so:
  - `alc.nn.preset(arch, variant, opts?) -> NnHandle` — arch-neutral
    trainable/inference preset entry, dispatched via the new
    `ARCH_OPS: &[(family_prefix, ArchOps)]` static table. Callers
    who want an arch-pinned entry keep using the typed aliases
    (`alc.nn.preset.gpt2` / `.tinyllama` / `.llama`); they remain
    callable and return the same typed Handle they did pre-4b.
  - `alc.nn.preset.tinyllama(variant, opts?)` — new typed alias,
    fills the missing entry that made TinyLlama unreachable from
    Lua pre-4b (Layer 4a shipped `TinyLlamaModel::wrap_lora` on
    the Rust side, but the Lua bridge had no way to build a
    trainable TinyLlama handle).
  - `alc.nn.card.load_handle(card_id) -> NnHandle` — arch-neutral
    self-contained card loader (`training_path == "full_ft" /
    "merged" / "distillation"`). Reads the card's `architecture`,
    dispatches through `ARCH_OPS.build_from_safetensors`, mmaps
    the bundle at `<nn_dir>/<card_id>.safetensors`. Refuses LoRA
    cards with a directional error pointing at `load_wrap`.
  - `alc.nn.card.load_wrap(card_id, base) -> NnHandle` —
    arch-neutral LoRA card loader. Accepts either an `NnHandle`
    (from `alc.nn.preset(arch, ...)`) or a typed handle (from
    `alc.nn.preset.<arch>`) as `base`; enforces arch-match
    between the card and the base. Refuses self-contained cards
    with a directional error pointing at `load_handle`.
  - `alc.nn.card.load_gpt2` continues to work as a deprecated
    typed shortcut, delegating to the shared
    `wrap_gpt2_lora_from_meta` core. Migration: call
    `alc.nn.card.load_wrap(card_id, base)` instead.
  - `alc.nn.card.load` continues to return raw vars (backward
    compat alias for `alc.nn.card.load_vars`); the arch-neutral
    handle-returning entry is registered as
    `alc.nn.card.load_handle` during the deprecation window and
    will take over the `load` slot in a future minor release.
  - New `pub(super) enum NnHandle { Gpt2, TinyLlama, Llama }` with
    a single UserData impl that fans method calls (`:arch()`,
    `:variant()`, `:layers()`, `:kv_heads()`, `:forward_shape()`,
    etc.) out to the wrapped typed handle. `:kv_heads()` returns
    `heads` on GPT-2 for a uniform Lua surface. `NnHandle`
    variants + typed handles now `derive(Clone)` so backward-compat
    load_wrap callers can pass a typed base and have it lifted
    into `NnHandle`.
  - New arch-neutral card-load discipline (Layer 4b §Q3-A): two
    load surfaces (`load` = self-contained, `load_wrap` = LoRA)
    reflect the base-handle contract in the signature rather than
    hiding it behind an `Option<base>` argument. Directional
    errors on each surface point the caller at the correct
    sibling entry.
  - Follow-ups called out in the Layer 4b design notes
    §8: `alc.nn.trainer.*` bindings migrating from typed
    `Gpt2Handle` to `NnHandle` (mechanical once §Q1 lands, still
    breaks-only-if-you-borrow-Gpt2Handle-directly); `LlamaAdapter`
    unification into `NnHandle` beyond the current symmetry
    wrap; `NnLoadable` trait extraction from `ARCH_OPS` once a
    third trainable arch (qwen2 / phi / gemma) makes the shape
    obvious. Adding one of those arches today is three grep-able
    edits: `ARCH_OPS` tuple + typed `build_<arch>_handle` +
    neutral adapter — no new Lua-visible fn per arch.
- `algocline-engine`: `alc.nn.card.merge_lora` Lua binding for the
  Layer 4a merged inference checkpoint export (Layer 5a of GH #10).
  The Rust-side `export_merged` + `MergedProvenance` surface is now
  reachable from Lua through the same arch-neutral card bridge as
  Layer 4b's `load_handle` / `load_wrap`.
  - `alc.nn.card.merge_lora(wrapped_handle, opts) -> merged_card_id`
    consumes a LoRA-wrapped `NnHandle` (from
    `alc.nn.card.load_wrap`) plus `opts = { name, lora_card }`,
    writes the merged safetensors bundle under
    `<nn_dir>/<merged_card_id>.safetensors`, records a Card with
    `training_path == "merged"` + `lineage.parent == opts.lora_card`,
    and returns the freshly-minted `merged_card_id` string. The
    merged Card is self-contained: a subsequent
    `alc.nn.card.load_handle(merged_card_id)` returns an unwrapped
    inference handle with no LoRA delta needed.
  - `MergedProvenance.arch` and `bundle_ref` are derived inside the
    bridge (from the handle's `arch_family_variant()` + the
    pre-generated `card_id`) rather than caller-supplied, so the
    resulting Card is structurally correct by construction. The
    caller only ever supplies `name` + `lora_card`.
  - Base (non-LoRA-wrapped) handles are refused up front with a
    directional error pointing at `alc.nn.card.load_wrap`. The
    `NnHandle::is_lora_wrapped()` guard reads a new
    `pub(super) has_lora: bool` field on `Gpt2Handle` /
    `TinyLlamaHandle`, set to `true` only in the two
    `wrap_*_lora_from_meta` return paths (all other handle
    constructors default it to `false`).
  - Backward compat: accepts either an `NnHandle` (arch-neutral
    from `load_wrap`) or a typed `Gpt2Handle` / `TinyLlamaHandle`
    (once a future Layer 5b adds a Lua-side wrap that returns
    typed) as the first argument, same as `load_wrap`.
  - Verified by 6 integration tests
    (`bridge::nn_card::merge_lora_bridge_tests`) covering happy
    paths for both GPT-2 and TinyLlama arms, refuse-base-handle
    directional error, refuse-missing-opts, refuse-empty-string
    opts, and end-to-end `load_wrap → merge_lora → load_handle`
    round-trip against the Card store on `gpt2-tiny` /
    `tinyllama-tiny` micro shapes (CPU F32).
  - GPU verification status: TinyLlama full/LoRA/merge paths are
    exercised only on CPU F32 in-tree so far. A40 smoke examples
    (`nn_tinyllama_gpu_smoke.rs`, `nn_tinyllama_lora_gpu_smoke.rs`,
    and `nn_tinyllama_merge_lora_gpu_smoke.rs`) landed as the next
    entry in this section; actual A40 execution remains the
    follow-up tracked in the spike notes (spike-status §7).
- `algocline-engine`: `alc.nn.wrap_lora` +
  `alc.nn.trainer.run_lora_ft` Lua bindings for the Layer 2 LoRA
  wrap and the Layer 3 LoRA fine-tuning loop (Layer 5b of GH #10,
  landed in two sub-phases: S1 = wrap surface, S2 = trainer surface
  + trailing docs). Both entries dispatch on the arch-neutral
  `NnHandle` that Layer 4b established (GPT-2 / TinyLlama arms
  today; `Llama` refused with an arch-directional error). Config
  schema is wider than Layer 5a and drives most of the added test
  surface.
  - `alc.nn.wrap_lora(base_handle, opts) -> NnHandle` (S1) — wrap a
    base model in-memory with a fresh LoRA layout and return a
    LoRA-wrapped handle (wrap state tracked engine-side and enforced
    by the double-wrap / trainer refusals; not a Lua method). Sits at the
    top level of the `alc.nn` sub-table (not under `card`) because
    `wrap_lora` writes nothing to disk. Consumes `opts = { rank,
    alpha, target_modules?, dropout? }`; `target_modules` defaults
    to the arch's canonical set from `LoraConfig::default_targets`
    (GPT-2, 6 targets) or `TinyLlamaModel::default_lora_targets`
    (TinyLlama, 7 targets). Base-freeze invariant: the base
    `VarMap` is byte-identical before / after wrap (each block's
    linears are moved into `LoraLinear`, but the base tensors
    themselves are not touched). Refuses already-wrapped handles
    with a directional error (double-wrap protection). Verified by
    7 integration tests
    (`bridge::nn_wrap::wrap_lora_bridge_tests`) covering happy
    paths for both GPT-2 and TinyLlama arms (A1/A2) plus 5
    config-schema refusals (B1-B5: zero rank / missing alpha /
    empty target array / arch-mismatched target / out-of-range
    dropout).
  - `alc.nn.trainer.run_lora_ft(base_handle, dataset, opts) ->
    lora_card_id` (S2) — one-call LoRA fine-tuning surface: wrap
    the base, run the training loop, save the Δ safetensors, and
    write a `training_path="lora"` Card in a single call. Extends
    the pre-existing `alc.nn.trainer` sub-table alongside
    `full_ft` / `lora` / `distill` (the sibling `lora` entry
    returns a Checkpoint table for callers who want to drive Card
    assembly by hand; `run_lora_ft` is the "batteries-included"
    path that mints the Card in one call). Consumes `opts = {
    rank, alpha, target_modules?, dropout?, lr, batch, steps,
    warmup?, schedule?, name? }`; `grad_accum_steps` and
    `ckpt_every` are NOT exposed and remain pinned to their crate
    defaults (design's explicit non-exposure). All config errors
    surface pre-flight as loud `LuaError::external` with prefix
    `alc.nn.trainer.run_lora_ft:` — no silent fallback, no
    `warn!` swallow. The Δ safetensors is written to
    `<nn_dir>/nn/lora-<lora_card_id>.safetensors` (the Rust
    surface convention — not caller-configurable); the Card
    records `candle.bundle_ref = "nn/<lora_card_id>"` and
    `candle.lora.{rank, alpha, target_modules, dropout,
    delta_path, base_bundle_ref}` where `base_bundle_ref` is
    derived from the base handle as `"nn/<family>-<variant>"`. The
    returned `lora_card_id` is loadable via
    `alc.nn.card.load_wrap` (Layer 4b) for inference or feedable
    to `alc.nn.card.merge_lora` (Layer 5a) for merged export.
    Verified by 9 integration tests
    (`bridge::nn_trainer::run_lora_ft_bridge_tests`) covering
    happy paths for both GPT-2 and TinyLlama arms (A3/A4) plus 2
    config-schema refusals (B6/B7: zero steps / unknown
    schedule), 3 state invariants (C1: base VarMap byte-identical
    before / after; C2: Δ safetensors contains exactly `n_layers ×
    n_targets × 2` LoRA A/B tensors and no base leakage; C3:
    cross-surface — the freshly-written LoRA Card is consumable by
    `alc.nn.card.load_wrap` and yields a handle with
    `is_lora_wrapped() == true`), and 2 Card + Δ round-trips
    through `load_wrap` (D1/D2: GPT-2 / TinyLlama arms).
    Concurrency: a fresh `TrainingLease` is constructed per-call
    and NOT shared with the sibling `full_ft` / `lora` / `distill`
    entries (documented limitation; sharing across Lua calls is
    out of scope).
  - Reused shared helpers (widened to `pub(super)` with
    justification comments so cross-file reuse is documented
    at the widening site rather than the consumer): `sanitize_name`,
    `compact_epoch_us`, `build_create_payload_from_meta`,
    `extract_full_ft_opts`, `load_wrap_impl`, and a new
    `DatasetHandle::inner_lock` accessor for cross-module dataset
    downcast without exposing the underlying `Mutex` field.
  - Verification status: all 16 new bridge tests
    (`wrap_lora_bridge_tests` + `run_lora_ft_bridge_tests`) pass
    on `gpt2-tiny` / `tinyllama-tiny` micro shapes with no HF hub
    download and no > 1s train step per test (CPU). Real-scale GPU
    verification landed in the same `[Unreleased]` window: the Rust
    route via `nn_tinyllama_lora_gpu_smoke.rs` (see the GPU smoke
    example suite entry below) and the Lua-bridge route via
    `nn_bridge_gpu_smoke.rs` (see the Layer 5c S4 entry), both run
    end-to-end on an A40.
- `algocline-engine`: `alc.nn.trainer.run_full_ft` Lua binding for
  the one-call full-fine-tune surface (Layer 5c S1 of GH #10).
  Sibling of the Layer 5b S2 `alc.nn.trainer.run_lora_ft` — same
  arch-neutral `NnHandle` dispatch (GPT-2 / TinyLlama arms;
  `Llama` refused as inference-only), same one-call Card mint
  discipline (returns a `card_id` string rather than a raw
  Checkpoint table), same per-call `TrainingLease` isolation.
  Diverges from `run_lora_ft` on three axes: no LoRA config
  (only `lr` / `batch` / `steps` / `warmup` / `schedule`);
  refuses `pretrained = true` handles because full-fine-tune
  needs the base handle's original `VarMap` for AdamW; refuses
  LoRA-wrapped handles with a directional error pointing at
  `alc.nn.wrap_lora` (drop the wrap first — a wrapped handle
  is a LoRA-training target, not a full-fine-tune target). Also
  registered alongside the pre-existing `alc.nn.trainer.full_ft`
  entry (which returns a raw Checkpoint table); pick
  `run_full_ft` for Card-first Lua flows and `full_ft` for
  pipelines that need to inspect the Checkpoint before
  assembling the Card. The trained safetensors bundle lives at
  `<nn_dir>/<card_id>.safetensors` (the `alc.nn.load(card_id)`
  resolve path) and the Card records
  `training_path = "full_ft"` +
  `candle.bundle_ref = "nn/<card_id>"` (no LoRA branch,
  matching the sibling `full_ft` + `alc.nn.card.save` flow).
  All errors surface pre-flight as loud `LuaError::external`
  with prefix `alc.nn.trainer.run_full_ft:` — no silent
  fallback, no `warn!` swallow, per the L5b one-prefix-per-
  surface contract. Verified by 4 integration tests
  (`bridge::nn_trainer::run_ft_bridge_tests::run_full_ft_*`):
  happy paths for both GPT-2 and TinyLlama arms + zero-steps
  refusal + LoRA-wrapped-handle refusal. `run_lora_ft` tests
  co-located in the same `run_ft_bridge_tests` module continue
  to pass unchanged (module rename from
  `run_lora_ft_bridge_tests` → `run_ft_bridge_tests` reflects
  the shared coverage; test names carry the surface prefix).
  Follow-up: the `alc.nn.trainer.run_distill` bind landed in this
  same `[Unreleased]` window — see the Layer 5c S2 entry below.
- `algocline-engine`: `alc.nn.trainer.run_distill` Lua binding for
  the one-call distillation surface (Layer 5c S2 of GH #10).
  Sibling of the Layer 5c S1 `alc.nn.trainer.run_full_ft` — same
  arch-neutral `NnHandle` dispatch (GPT-2 / TinyLlama arms; `Llama`
  refused as inference-only), same one-call Card mint discipline,
  same per-call `TrainingLease` isolation, same pretrained /
  LoRA-wrapped handle refusals (distillation IS a full fine-tune
  under a distillation loss; the teacher signal lives in the
  dataset, not in a second model instance). Adds the
  `opts.loss_kind` field (default `"ce"`, the only
  `DistillLossKind` variant shipped; unknown values refused rather
  than silently falling back to CE). The written Card records
  `training_path = "distillation"` + `candle.bundle_ref =
  "nn/<card_id>"` (no LoRA branch) and carries `loss_kind` under
  `hyperparams` so the loss selection is auditable from the Card.
  Diverges from the pre-existing `alc.nn.trainer.distill` entry
  (raw Checkpoint return, typed `Gpt2Handle` only); both entries
  stay registered. All errors surface as loud `LuaError::external`
  with prefix `alc.nn.trainer.run_distill:` per the
  one-prefix-per-surface contract. Verified by 4 integration tests
  (`bridge::nn_trainer::run_ft_bridge_tests::run_distill_*`):
  happy paths for both arms + unknown-loss_kind refusal +
  pretrained-handle refusal.
- `algocline-nn`: `train::run_distill` generalised from the
  `Gpt2Model`-hardcoded signature to `run_distill<M>` with the same
  bound as `run_full_ft` (`M: Module + DeviceView`) — distillation
  places no extra requirement on the student model, and the
  function already forwarded to the generic `run_full_ft`
  internally. Existing concrete-type callers (the
  `alc.nn.trainer.distill` bridge, `tests/distill_synthetic.rs`)
  resolve through the generic unchanged.
- `algocline-engine`: pretrained-handle refusal on the
  `run_full_ft` / `run_distill` bridges is now covered by
  integration tests through the full dispatch path
  (`run_full_ft_refuses_pretrained_handle` /
  `run_distill_refuses_pretrained_handle`), closing the L5c S1
  coverage gap where the guard was documented but untestable. New
  test-only constructor `Gpt2Handle::for_test_pretrained_like`
  (`#[cfg(test)]`, same cross-module pattern as
  `DatasetHandle::for_test`) strips the `VarMap` off a
  from-scratch handle to mimic the varmap-less pretrained state
  without an HF hub download.
- `algocline-engine`: trainer-bind refusal coverage extended to
  both arms (Layer 5c S3 of GH #10, test-only).
  `run_distill_refuses_lora_wrapped_handle` exercises the
  `run_distill` LoRA-wrapped refusal through the full dispatch
  path (building a real wrapped handle via `run_lora_ft` →
  `load_wrap` first), and the pretrained-handle refusals gain
  TinyLlama-arm mirrors
  (`run_full_ft_refuses_pretrained_handle_tinyllama` /
  `run_distill_refuses_pretrained_handle_tinyllama`) backed by
  the new test-only constructor
  `TinyLlamaHandle::for_test_pretrained_like` (`#[cfg(test)]`,
  mirror of the `Gpt2Handle` sibling).
- `algocline-engine`: `nn-cuda` cargo feature — the CUDA variant
  of `nn`. Enables candle's CUDA backend across `candle-core`
  AND `candle-nn` in lockstep (candle-nn 0.11's `LayerNorm` is a
  `CustomOp3` whose `cuda_fwd` is `#[cfg(feature = "cuda")]`-
  gated, so a core-only CUDA build still dispatches LayerNorm on
  CPU and fails at forward time) plus `algocline-nn/nn-cuda` for
  the model layer (Layer 5c S4 of GH #10).
- `algocline-engine`: `examples/nn_bridge_gpu_smoke.rs` —
  Lua-bridge GPU smoke driver (Layer 5c S4 of GH #10). Exercises
  the full `alc.nn` bridge chain (`preset.tinyllama` →
  `data.synthetic` → `wrap_lora` → `trainer.run_lora_ft` →
  `card.load_wrap` round-trip → `trainer.run_full_ft` →
  `trainer.run_distill`) through a single Lua VM bootstrapped
  with `install_for_pkg_test`, on CUDA (`--features nn-cuda`) or
  CPU (`--features nn`); gated behind
  `required-features = ["nn"]` so default builds never compile
  it. `NN_SMOKE_DTYPE` defaults to `"f32"`: the CUDA preset
  default dtype (bf16) is rejected deep inside the f32 trainer
  paths ("unexpected dtype, expected: F32, got: BF16"), so the
  smoke pins f32 the same way the Rust GPU smoke examples do.
  Verified end-to-end on an A40 (TinyLlama-1.1B random-init,
  50 steps each: LoRA FT 9.8s / full FT 28.5s / distill 28.5s).
- `algocline-nn`: TinyLlama-1.1B GPU smoke example suite covering
  the base full-FT, LoRA fine-tune, and LoRA → merge round-trip
  paths on CUDA. Siblings of the existing
  `nn_medium_gpu_smoke.rs` / `nn_medium_lora_gpu_smoke.rs` GPT-2
  examples, using the same env-var / stderr-tracing / synthetic-
  corpus plumbing so behaviour is directly comparable across arches.
  - `examples/nn_tinyllama_gpu_smoke.rs` — full FT on
    `TinyLlamaConfig::from_variant` (default `tinyllama-1.1b`;
    also accepts `1.1b` / `tinyllama-tiny` / `tiny`), calls
    `run_full_ft` for `NN_SMOKE_STEPS` optimizer steps, dumps a
    `.safetensors` bundle to `NN_SMOKE_CKPT`, and prints final /
    min loss + LR. Zero invariant checks beyond exit-code = "the
    loop didn't panic". Env vars: `NN_SMOKE_STEPS` /
    `NN_SMOKE_BATCH` / `NN_SMOKE_CTX` / `NN_SMOKE_LR` /
    `NN_SMOKE_CKPT` / `NN_SMOKE_VARIANT`.
  - `examples/nn_tinyllama_lora_gpu_smoke.rs` — LoRA fine-tune via
    `run_lora_ft`, then the three LoRA invariants: (1) base
    `VarMap` byte-identical before / after `wrap_lora` +
    `run_lora_ft` (via pre / post `.safetensors` dumps and a
    streaming byte-compare so TinyLlama-1.1B F32 base ≈ 4.4 GB
    does not need to be resident twice), (2) Δ bundle size under
    the `NN_SMOKE_DELTA_MAX_BYTES` ceiling (default 64 MB, sized
    for rank 16 × 7-target = Q/K/V/O attn + `gate_proj` /
    `up_proj` / `down_proj` MLP triple at hidden_dim 5632 —
    versus the GPT-2 medium 32 MB ceiling that assumed only 6
    targets and hidden_dim ≈ 4·dim), and (3) per-step loss
    trajectory observable via a `tracing_subscriber` on stderr
    (`RUST_LOG=algocline_nn=info`). `NN_SMOKE_LORA_TARGETS=attn`
    switches to the 4-target Q/K/V/O attention-only variant.
  - `examples/nn_tinyllama_merge_lora_gpu_smoke.rs` — extends the
    LoRA fine-tune with the Layer 4a merged-export round-trip:
    `export_merged(&wrapped_model, &MergedProvenance { lora_card,
    arch, bundle_ref }, merged_path)` after training, then
    `TinyLlamaModel::from_safetensors_file` to reload the merged
    bundle, then a `max_abs_diff_f32(wrapped.forward,
    reloaded.forward)` parity assertion under
    `NN_SMOKE_MERGED_TOLERANCE` (default `1e-4`, aligned with the
    CPU-side
    `merged_bundle_tinyllama_forward_matches_wrapped_forward`
    parity test). The 2026-07-20 A40 verify measured
    `max_abs_diff = 2.003e-5` — a 5x margin under this ceiling —
    so an earlier `1e-3` guess for CUDA fused mul-add drift has
    been tightened to `1e-4`. Loosen via env only when a wider
    dtype / higher-rank / larger-corpus configuration surfaces
    genuine drift above the tighter bound. Also asserts the merged bundle size stays under
    `NN_SMOKE_MERGED_MAX_BYTES` (default 4.5 GB, sized for
    TinyLlama-1.1B F32 base weight footprint + safetensors header
    + headroom). `NN_SMOKE_SKIP_MERGE=1` drops back to the pure
    LoRA smoke behaviour so a caller can isolate a LoRA
    regression from a merge regression without changing example.
    All three LoRA invariants (base frozen, Δ size, loss
    trajectory) remain in force alongside the two new
    merge-specific invariants (merged forward parity, merged
    bundle size bounded) — five invariants total.
  - Each example is a plain `--example` binary compilable without
    the `nn-cuda` feature so the dev-host CPU compile check (`cargo
    check --example …`) exercises it as a regression gate; the
    actual A40 execution follows the GPU smoke runbook Path A (git push
    → pod git clone → `cargo build --release --features nn-cuda`
    → `cargo run --release --features nn-cuda --example
    nn_tinyllama_merge_lora_gpu_smoke`). GPU execution results
    themselves are not part of this release entry — they land as
    a `spike-status.md` §7 verification pass.
- `algocline-nn`: merged inference checkpoint export (Layer 4a of
  GH #10). A LoRA-wrapped model can now be composed with its base
  into a single safetensors bundle that downstream inference stacks
  load as if it were a plain pretrained model — no more carrying
  `<base>.safetensors` plus a `<delta>.safetensors` and re-running
  `wrap_lora` at load time.
  - New `pub trait algocline_nn::arch::lora::MergeableLora` with a
    single `export_merged(&self) -> CandleResult<HashMap<String,
    Tensor>>` method. Both `Gpt2Model` and `TinyLlamaModel` impl
    the trait as thin walkers: every `LinearVariant::Lora`
    projection collapses via `LoraLinear::merged_weight()`; every
    `Plain` projection passes through unchanged. Emitted keys
    match the HF-native safetensors layout the same arch's
    `from_pretrained` reads (GPT-2: `wte.weight` / `wpe.weight` /
    `h.<i>.{ln_1,ln_2,attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}.{weight,bias}`
    / `ln_f.{weight,bias}`; TinyLlama:
    `model.embed_tokens.weight` /
    `model.layers.<i>.{input_layernorm,post_attention_layernorm,self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}.weight`
    / `model.norm.weight` / `lm_head.weight`). TinyLlama's GQA
    K/V projections retain their `[kv_heads * head_dim, dim]`
    shape — never broadcast up to Q's shape (guarded by
    `merged_bundle_tinyllama_preserves_gqa_kv_shape`).
  - New `algocline_nn::merged` module with the `MergedProvenance
    { lora_card, arch, bundle_ref }` struct + `MergeError` enum
    (`Provenance` / `Merge` / `Io` / `Serialize`) + free fn
    `export_merged<M: MergeableLora>(model, provenance, out_path)
    -> Result<(usize, NnCardMeta), MergeError>`. The entry point
    walks the model on its live device, moves each tensor to CPU,
    writes safetensors via `candle_core::safetensors::save`,
    stats the file for bytes-written, and returns both the byte
    count and the projected `NnCardMeta`.
  - New `Gpt2Model::from_safetensors_file` and
    `TinyLlamaModel::from_safetensors_file` constructors — plain
    on-disk safetensors load without touching the HF hub. Both
    read the same HF-native layout `from_pretrained` reads
    (GPT-2 root-scoped; TinyLlama routes through
    `new_from_split(cfg, root.pp("model"), root)` to honour the
    `model.*` + top-level `lm_head` split). These are what the
    parity oracle uses to reload a merged bundle and what a wide
    4b load-side integration will dispatch to when it recognises
    `training_path == "merged"`.
  - Card provenance follows the new **Model-side struct +
    `to_card_*` projection** pattern (workspace design
    `layer-4-merged-ckpt-design.md` §Q0): `MergedProvenance` is
    the SoT for what a merged card carries; `to_card_meta(name)`
    projects into existing `NnCardMeta` slots only
    (`lineage.parent` ← `lora_card`, `architecture` ← `arch`,
    `candle.bundle_ref` ← `bundle_ref`, `training_path` ←
    `"merged"`). No `NnCandleBranch.merged` sub-branch, no new
    lineage field, no denormalisation of LoRA hyperparams. The
    base model reference is transitively reachable via the LoRA
    card's `NnLoraBranch.base_bundle_ref` (single SoT for LoRA
    hyperparams stays on the LoRA card). Future training paths
    are expected to adopt the same pattern so per-branch Card
    schema negotiations become local to the Model side rather
    than re-opening the Card schema each time.
  - `algocline_nn::card::SUPPORTED_TRAINING_PATHS` const +
    `validate_training_path` fn. `"merged"` joins `"full_ft"` /
    `"lora"` / `"distillation"` as an accepted `training_path`
    value. `NnCardMeta.training_path` remains a free `String` at
    deserialisation time (mirrors the `validate_architecture`
    opt-in pattern) so foundation-era cards with non-listed
    values still round-trip.
  - Wire shape: additive across the board. Existing
    `alc.nn.card.load_gpt2` / `alc.nn.card.save` bridges continue
    to work; the load-side path that dispatches
    `training_path == "merged"` to `from_safetensors_file`
    without re-wrapping the model is a Layer 4b follow-up (not
    in this release).
- `algocline-nn`: arch-neutral prerequisites for the inference-fleet
  expansion tracked in GH #9 (Layer 1 of 3).
  - `algocline_nn::card::validate_architecture` + the
    `SUPPORTED_ARCHITECTURE_FAMILIES` allowlist (`gpt2` / `llama` /
    `tinyllama` / `qwen2` / `phi` / `gemma`) accepted at Card save
    time. `alc.nn.card.save` now rejects typos and unknown architecture
    identifiers early with a clear Lua-side error instead of persisting
    an unloadable Card; existing `gpt2-medium` / `gpt2-large` cards
    remain round-trippable unchanged.
  - `NnStore::load_gguf_path` — read path for pretrained GGUF (Q4 / Q8)
    bundles alongside the existing safetensors read path. The trait
    method defaults to `Err("GGUF read path is not implemented ...")`
    so a backend that does not manage quantized bundles never has to
    acknowledge the extension; `FsStore` overrides it to resolve
    `<root>/<name>.gguf` under the same root and name-safety rules as
    the safetensors path. Additive on the MCP wire; breaking only for
    downstream code that implements `NnStore` and wants to expose GGUF.
- `algocline-nn`: inference adapter over `candle-transformers` for
  Llama-family models (Layer 2 of GH #9).
  - New optional-feature-free dependency on `candle-transformers = "0.11"`
    (lockstep with `candle-core` / `candle-nn`; the `nn-cuda` feature
    now also enables `candle-transformers/cuda`).
  - New `algocline_nn::arch::adapter::{LlamaAdapter, LlamaAdapterConfig}`
    module wrapping `candle_transformers::models::llama::Llama` behind
    an inference handle. `LlamaAdapter::load` / `from_safetensors_files`
    load parameters through a `VarBuilder`; `LlamaAdapter::forward`
    threads through an owned KV `Cache` so token-by-token generation
    reuses cache state without exposing candle-transformers types to
    the caller.
  - `alc.nn.preset.llama(variant, opts)` — Lua binding returning a
    `LlamaHandle` UserData with the same metadata surface as
    `Gpt2Handle` (`variant` / `layers` / `heads` / `kv_heads` / `dim` /
    `ctx` / `vocab` / `device` / `dtype` / `forward_shape`).
    Accepts `"tiny"` / `"7b-v1"` / `"7b-v2"` (plus their `"llama-*"`
    aliases) with `device` / `dtype` / `weights` / `use_kv_cache` /
    `flash_attn` options.
  - The handle is inference-only by construction (no `VarMap`), so
    `alc.nn.trainer.*` continues to refuse handles that were not built
    from a from-scratch VarMap — no training-path regression.
  - `docs/lua-stdlib.md` gains an `alc.nn.preset.llama` section
    documenting variants, opts, methods, and the "wrap the handle in
    a Lua closure + `alc.nn.register`" recipe for exposing the model
    through `alc.llm(prompt, {role="nn", model=name})`.
- `algocline-nn`: `nn-metal` cargo feature + device / dtype matrix
  expansion (Layer 3 of GH #9).
  - New `nn-metal = ["candle-core/metal", "candle-nn/metal",
    "candle-transformers/metal"]` feature enables candle's Metal
    backend in lockstep across the three candle crates (same
    P3-pitfall discipline as `nn-cuda`). Compiles on macOS without an
    external toolchain; on non-macOS hosts the feature still builds
    but the runtime `Device::new_metal(...)` call errors when no
    Metal device is present.
  - `alc.nn.preset.{gpt2,llama}` device string now accepts `"metal"`
    and `"metal:N"` alongside `"cpu"` / `"cuda"` / `"cuda:N"`. The
    device / dtype default when the caller omits `opts.dtype` picks
    `"bf16"` on CUDA, `"f16"` on Metal, and `"f32"` elsewhere.
  - The bf16 pre-flight guard now spans both CPU and Metal: candle-nn
    0.11 only ships bf16 kernels for the CUDA backend, so a
    `dtype = "bf16"` call on Metal errors up front with a clear
    "use dtype='f16' or move to CUDA" message rather than failing
    deep inside a forward pass. The GPT-2 / Llama presets share the
    guard through the new `guard_device_dtype_matrix` helper so the
    two paths keep identical failure modes.
  - Bridge-layer `parse_device` / `parse_dtype` were refactored into
    preset-tagged `parse_device_for` / `parse_dtype_for` /
    `default_dtype_for_device` helpers so both presets consume the
    same allowlist. No caller-visible behaviour change on the pre-
    existing `"cpu"` / `"cuda"` / `"cuda:N"` paths.
  - Smoke tests: `alc_nn_preset_llama_{metal_device_string_is_recognised,
    bf16_on_metal_errors}` on `nn_bridge_smoke.rs` — 2 new tests, all
    green.
  - `docs/lua-stdlib.md` `alc.nn.preset.llama` section updated with
    the new device string list, dtype defaults, and the bf16 / Metal
    error path.

### Changed

- `algocline-nn`: `LoraLinear::wrap` now follows the canonical LoRA
  init from Hu et al. 2021 §4.1 — `lora_a` keeps candle-nn's default
  (Kaiming uniform) random init, but `lora_b` is now zero-init via
  `VarBuilder::get_with_hints(..., Init::Const(0.0))`. Under this init
  `ΔW = scaling * (B · A) = 0` at construction, so a freshly-wrapped
  model produces bit-identical forward output to the un-wrapped base;
  training moves `B` off zero first (step 1 gradient reaches only `B`,
  since `dL/dA` flows through `B^T`), then `A` at step 2+. Prep for
  the Layer 3 (GH #10) fine-tuning loop, which relies on this
  canonical "wrap is identity at `t=0`" property to reason about the
  additive update. Affects both `Gpt2Model::wrap_lora` and
  `TinyLlamaModel::wrap_lora` since they share the same low-level
  wrap helper. Behavior-only — no API signature change; downstream
  callers that manually inspected pre-training LoRA weights and
  assumed a non-zero `lora_b` will now see zeros.

### Deprecated

### Removed

### Fixed

### Security

## [0.46.1] - 2026-07-20

Hotfix for the v0.46.0 CI pipeline. The release commit landed with a
chain of drift between the developer laptop's environment and the
Ubuntu CI runner that only surfaced after tagging — stylua parser
features, environment-dependent snapshot assertions, missing bundled
scenarios, and pre-existing service-layer invariant violations
introduced by the alc-nn merge. No user-facing API change; every fix
is either CI plumbing or an internal invariant repair.

### Fixed

- `just lua-fmt-check` on CI now installs `stylua` with the `lua54`
  Cargo feature enabled so it can parse Lua 5.3+ syntax (bitwise
  `>>` / `&`, `goto` labels) used in prod Lua sources. Homebrew's
  stylua bottle enables the feature by default, which hid the drift
  locally.
- Vendored evalframe Lua files (`crates/algocline-engine/src/vendor/`)
  are now excluded from `stylua --check .` via `.styluaignore`, and
  the one drifted `examples/nn_lora_smoke.lua` was reformatted.
  Vendor files remain untouched per their `DO NOT EDIT` header.
- `test_alc_info` snapshot no longer captures the developer's
  `gh auth status`, `git config user.*`, `~/.ssh/` presence, or
  `~/.algocline/settings.toml` contents. `redact_gh_credentials`
  now collapses every environment-varying field
  (`gh_auth.available` / `.error` / `.logged_in`,
  `git_config.complete` / `.user_name` / `.user_email`,
  `ssh_keys.any_present`, and `settings.resolved` / `.sources`)
  into stable placeholders so the snapshot is portable across
  developer laptops and CI runners.
- `test_alc_card_*` (3 cases) and `test_alc_eval_llm_rubric_*`
  (4 cases) now succeed on CI. The runner had no
  `$HOME/.algocline/packages` (fixed by running `alc init` before
  the test suite) and no default `git user.email` / `user.name`
  (fixed by a `git config --global` step before the tests, needed
  by `alc_card_publish`'s internal `git commit`). Both are CI-only
  workflow additions; the developer laptop already had them
  populated.
- `test_alc_fork_roundtrip` was collateral damage from the missing
  `$HOME/.algocline/packages` above and is fixed by the same
  `alc init` bootstrap.
- `crates/algocline-engine/tests/lua/eval_integration_test.lua`
  resolves `math_basic.lua` via `~/.algocline/scenarios/`, which
  `alc init` does not populate (it installs `packages/` only).
  CI now clones the pinned `algocline-bundled-packages` tag a
  second time and copies `scenarios/*.lua` into place so the
  spec resolves the scenario the same way the laptop does.
- `just check-invariants` Inv-1 now stays green: the SSH key
  enumeration in `service::gh_credentials::check_ssh_keys` no
  longer calls `dirs::home_dir()` directly. HOME is resolved once
  in the whitelisted `service::config::AppConfig::resolve_home`
  and threaded through `diagnose(app_dir, home)` to the three call
  sites (`logging.rs`, and both branches of
  `card.rs::card_publish`).
- `just check-invariants` Inv-3 no longer trips on rustdoc
  intra-doc links. The three hits inside `crates/algocline-engine/
  src/bridge/mod.rs` and `executor.rs` were all documentation
  references (`/// [`algocline_core::AppDir::nn_dir`]`) explaining
  what the invariant protects; there is no `use algocline_core::
  AppDir` in engine code. The Inv-3 grep now strips `///` and `//!`
  lines before deciding failure — ordinary `//` code comments stay
  in the scan window so a smuggled `// use algocline_core::AppDir;`
  is still caught.

### Added

- `@alc-maint:alc-pre-push` sensor Agent
  (`plugins/alc-maint/agents/alc-pre-push.md`). Runs each sub-recipe
  of `just ci` individually via `mcp__lds__recipe_run`, aggregates
  per-step PASS / BLOCKED, and returns a single verdict with
  evidence and fix hints. Three modes: `full` (default, ~5–10 min),
  `quick` (fmt / clippy only, ~1 min), `custom` (caller-provided
  list). Sensor only — no autofix, no commit, no push. Sibling of
  `@alc-maint:alc-bundled-sync`. Motivated by the v0.46.0 CI
  ping-pong: the author ran `cargo test` locally but forgot
  `cargo fmt --check` and `just lua-fmt-check`, and each miss burnt
  a CI round-trip.
- `/alc-maint-wake` skill now guides callers to `@alc-pre-push`
  alongside `@alc-bundled-sync`, so the dispatch example is visible
  from the maintenance-session dashboard.

### Changed

- `justfile` recipe group markers unified to
  `[group('allow-agent')]`. The previous `[group: 'agent']` syntax
  was accepted silently by `just` but did not match task-mcp's
  `allow-agent` allowlist, so `ci` / `fmt-check` / `lua-fmt-check`
  / `clippy` / `test` / `check-invariants` / `check-agent-index`
  were not dispatchable via `mcp__lds__recipe_run` — a hidden gate
  that made `@alc-pre-push`'s first live run BLOCKED at preflight.
  All 17 previously-mistagged recipes are now exposed under a
  single `[allow-agent]` group; CI shell path is unaffected.

## [0.46.0] - 2026-07-19

### Added

- New opt-in `algocline-nn` workspace crate + `alc.nn.*` Lua bridge, gated
  behind the `nn` feature on `algocline-engine` (default off — the default
  build does not link candle). Provides a thin candle-based
  neural-network layer exposing GPT-2 training primitives through
  `alc.nn.trainer.*`:
  - `alc.nn.trainer.full_ft` — Full fine-tuning training loop with
    optimizer / loss / scheduler / checkpoint store abstractions
  - `alc.nn.trainer.lora` — LoRA wrap + `run_lora_ft` with delta-only
    checkpoint (`alc_shapes` `Gpt2Model::wrap_lora`), merge-equivalence
    guarantee (wrapped forward matches merged linear within tolerance)
  - `alc.nn.trainer.distill` — Hard-label distillation loop.
    `TeacherCardDataset` shipped alongside it as a Rust-side teacher
    adapter; no Lua-reachable constructor produced a mask-carrying
    dataset at this release (see the `[Unreleased]` entry for the
    `alc.nn.data.from_card` path that makes it reachable)
  - `alc.nn.data.synthetic` — CPU-scale synthetic corpus binding for
    smoke tests without HuggingFace hub network calls
  - `alc.nn.card.*` — Card-backed model metadata schema for
    save / load / register of models, weights, and deltas, with
    Card load-with-merge for LoRA branches
  - GPT-2 arch (tiny / medium / large presets), HF tokenizer, and
    `Dataset` trait for `alc.nn`
  - `nn-cuda` feature enables candle's CUDA backend on GPU hosts
    (RunPod A40 / A100). Both `candle-core/cuda` and `candle-nn/cuda`
    are enabled together as an invariant (see Fixed below).
  - Lua smoke examples: `nn_full_ft` / `nn_lora` / `nn_distill`.
    Rust smoke examples: `nn_medium_lora_gpu_smoke` (instrumented with
    per-step tracing + base-frozen VarMap verify).
  - In-VM `role="nn"` fast path on `alc.llm`: when a caller passes
    `role = "nn", model = <name>`, the request is dispatched
    synchronously to the algocline-nn model registry without a Host
    round-trip (compiled out on default builds).
- `cargo aidoc` setup for LLM-facing rustdoc distribution under
  `docs/aidoc/`. Landed `llms.txt` (5.1 KB summary) + `llms-full.txt`
  (46 KB full dump) + 52 per-crate / per-module markdown files.
  `just aidoc-gen` builds artifacts and `just aidoc-check` runs the
  strict-lint gate (`cargo aidoc --check --strict`, 0 findings after
  4 crate-root + 4 module rustdoc gap closures on `algocline-app` /
  `-core` / `-engine` / `-mcp` + `pool::error` / `pool::protocol` /
  `core::metrics` / `core::recent_log`). Wire into CI going forward.
- Phase 3-F: `alc.llm` opts now accepts a `card_context` field that
  injects prior Card summaries as an XML-like prefix
  (`<past_cards>...</past_cards>`) in the system prompt. Two resolution
  forms are supported (MVP): a Card id string
  (`card_context = "cot_20260718_a3f9c1"`) resolves the single Card, and
  a table (`card_context = { pkg = "cot", limit = 5 }`) fetches the most
  recent N Cards for the given pkg in `created_at` descending order. The
  emitted block uses a fixed template (1 line per Card, `pkg=` /
  `card_id=` / optional `[run.status=...]` / optional `Rating <val>` /
  optional `reason=...`) and is prefixed onto any existing `system`
  string, or used as the sole system prompt when `system` is unset.
  Resolution failures and empty results are silent no-op (no LuaError,
  no prefix) so existing `alc.llm` call sites remain regression-free.
  `alc.llm_batch` and fork-child VMs are out of scope for this phase.
- Phase 2-E: added `tracing::error` emit at Card store write points
  (`write_new_card` / `write_samples_text` / `write_aliases`) for
  MCP-blind observability of persistence failures. Each write-side `?`
  propagate is wrapped in `.map_err(|e| { tracing::error!(target:
  "alc.card", ...); e })` so the underlying I/O error still surfaces to
  the caller (Lua via `LuaError::external`) while operators get a
  structured `alc.card`-targeted diagnostic without needing an active
  Card session (`LogSink`) or the fan-out subscriber to be attached.
  The write-once conflict path in `write_samples_with_store` also emits
  an `error` entry so both underlying I/O failures and immutability
  violations show up in the same log target. Additive change: `?`
  propagate contract unchanged (rmcp / MCP wire shape identical), no
  new subscriber layer, no `LogSinkLayer`. Companion sweep for the read
  paths (`load_full` / `find_with_store` / `get_with_store` /
  `get_by_alias_with_store` / `read_samples_with_store`) is tracked as
  a separate carry issue and only marked with `TODO(issue:e60cd19d)`
  in this phase.
- Phase 2-D: added `LogSinkCardSubscriber` that fans out `CardEvent`
  (`Created` / `Appended` / `SamplesWritten` / `AliasesWritten`) into
  per-session `LogSink` ring buffers, closing the "Card writes → Log
  observability" Domain link. Each `Executor::start_session` (and
  `start_session_with_env`) registers the session's own `LogSink` on the
  process-wide `LogSinkCardSubscriber` via a new `register_log_sink` RAII
  API; the returned `LogSinkRegistration` guard is dropped when the
  `Session` is dropped so per-session lifecycle is closed. `CardEventBus`
  gains a new `add_subscriber` production API used by the singleton to
  self-register on first access. The subscriber uses the same
  snapshot-clone-then-iterate pattern as the existing publish path so
  registrations and unregistrations from other threads never deadlock a
  fan-out in flight. Additive change: existing subscribers
  (`FileCardSubscriber` seeded from `ALC_CARD_SINKS`) are unaffected.
- Phase 1-B: added optional `run` field to `alc.card.create` /
  `alc.card.append`, gated by `[setting.card].run` (default false, opt-in
  per session). When the setting is off, calls carrying a `run` field
  become no-op and return Lua `nil` without touching the card store or
  publishing a `CardEvent`; calls without `run` are unaffected. The
  section shape is `{ status: "succeeded"|"failed"|"skipped",
  reason?: string, action?: string }`; invalid status tokens raise a Lua
  error naming all three accepted values regardless of the gate state.
- `justfile` recipes to resync the vendored Lua packages from local
  upstream checkouts: `just vendor-alc-shapes` for the alc_shapes
  copy under `crates/algocline-app/src/service/gendoc/alc_shapes/`,
  `just vendor-evalframe` for the evalframe copy under
  `crates/algocline-engine/src/vendor/evalframe/`, and `just
  vendor-sync` to run both. Each recipe wipes the destination
  directory, copies every `.lua` file preserving subdirectory
  structure, and prepends a two-line origin marker naming the
  package, version, and resync recipe. Source directories default
  to local checkouts under `$HOME/projects/` and can be overridden
  per invocation with `ALC_SHAPES_SRC=<abs path>` or
  `EVALFRAME_SRC=<abs path>`.
- `alc.eval` simple form now accepts a per-case `rubric` field that
  overrides the default `llm_rubric` grader rubric for that case only.
  Cases without `rubric` continue to use the built-in three-axis default
  (factual accuracy / frontier reach / depth of reasoning), so existing
  scenarios are unaffected. The override is stored on `case.context`
  under the reserved key `_alc_rubric`; the vendored evalframe source
  remains verbatim.
- `alc_status` pending-query projection now exposes an optional `role`
  field. It is included by the `preview` and `full` presets so callers
  can distinguish judge pauses (`role = "grader"`) from strategy pauses
  without paying for the full prompt body. Custom `pending_filter`
  objects opt in via `{ "role": true }`; the `meta` preset stays minimal
  and does not project `role`.
- Vendored evalframe v0.4.0 into `algocline-engine`, enabling `alc.eval(...)`
  calls from user Lua code (via `alc_run`) without requiring
  `~/.algocline/packages/evalframe/` to be installed. The vendored modules
  are registered on `package.preload` at session VM init (including fork
  children and the `alc_pkg_test` sandbox) alongside the `std` global shim
  that evalframe expects. MCP `alc_eval` tool behavior is unchanged.
- LLM-as-Judge graders are now reachable from `alc.eval` simple form via the
  grader names `llm_rubric`, `llm_yes_no`, and `llm_factuality`. Each name
  auto-wires the matching evalframe scorer (`linear_1_5` for rubric /
  factuality, `bool` for yes/no) and a judge provider built on `alc.llm`;
  `llm_rubric` uses a default three-axis rubric (factual accuracy / frontier
  reach / depth of reasoning) when no rubric is supplied. Judge calls set an
  optional `role = "grader"` on `alc.llm`, which is forwarded verbatim on the
  `needs_response` payload (additive — the field is absent when unset) so a
  host can route grading calls to a different model than the strategy under
  test. The evalframe package itself is unchanged; the provider is built in the
  prelude.
- `alc.llm(prompt, opts)` and `alc.llm_batch` items accept an optional
  `role` string. When set it is forwarded verbatim on the paused-session
  `needs_response` JSON (additive; unknown keys are ignored for forward
  compatibility). Existing single-argument `alc.llm(prompt)` calls are
  unaffected.
- `@alc-eval` agent (`plugins/alc/agents/alc-eval.md`): paused-session runner
  for measurement / AnyModel routing. Answers each pending `alc.llm` query
  as the LLM itself — the model chosen at dispatch time via the Agent tool
  `model` parameter is the model under test — feeds answers via
  `alc_continue` (batch feed preferred), and loops until the session
  completes / errors / hits `max_rounds` / or a query falls outside pure
  text generation (`escalated`). Return contract is one block
  (`### Session` / `### Result` / `### Observations`) with `model:` echoed
  from the caller-passed label (never guessed) and no fabricated `usage`
  attached to `alc_continue`. Journal append defaults on and is skippable
  with `journal: off`; append uses an append-only recipe path (never the
  Read-full → Write pattern that caused the 2026-06-10 incident
  `94f692ef`). Enables per-model boost measurement without any engine
  change by pairing the runner with `alc_eval` scenarios and picking the
  runner's `model` per experiment (e.g. sonnet vs opus). Smoke-verified
  across three runs (verify_select × sonnet / opus + refine_loop ×
  sonnet, all `completed` with `pass@1 1.00`) plus one post-restart
  dispatch under the registered name `alc:alc-eval`.
- `/alc-wake` skill body: the Worker query paths section now documents
  `@alc-eval` alongside `@alc-adviser` / `@alc-refiner` / `/alc-build`,
  and the Responsibility split table lists the four Agent workers.
- `plugins/alc/README.md`: Components table adds an Agent (eval) row for
  `@alc-eval`; the Journal Backbone note updated to cite all four write
  owners (adviser / coder / refiner / eval).
- Root `README.md`: plugin bundle description enumerates the four
  agents (`@alc-adviser` / `@alc-coder` / `@alc-refiner` / `@alc-eval`).

### Changed

### Deprecated

### Removed

### Fixed

- `algocline-nn`: route `LayerNorm` through the backward-safe slow path
  in `crates/algocline-nn/src/arch/gpt2.rs`. `candle-nn` 0.11's fused
  `LayerNorm::forward` fast path dispatches through `apply_op3_no_bwd`,
  which yields a tensor with `BackpropOp::none()` — severing the
  autograd graph at every LayerNorm. Any trainable `Var` upstream of a
  LayerNorm (i.e. every transformer-block parameter — the tied `wte` LM
  head is the only exception because it participates only downstream of
  the final LN) receives no gradient during backward, so full FT and
  LoRA both silently fail to learn while forward + loss + optimizer.step
  still run without error. A local `apply_slow_layer_norm` helper
  routes all three call sites (`Block::ln_1`, `Block::ln_2`,
  `Gpt2Model::ln_f`) through `candle_nn::ops::layer_norm_slow`, which
  decomposes the LN into basic ops (`sub` / `sqr` / `sum_keepdim` /
  `div` / `sqrt` / `mul` / `add`), each of which has a proper backward
  implementation. Numerical output matches the fast path bit-exactly.
  Two CPU regression tests
  (`run_lora_ft_reduces_loss_on_overfit_corpus` /
  `run_lora_ft_updates_lora_weights` in
  `crates/algocline-nn/tests/lora_merge_equivalence.rs`) plus GPU A40
  verify (LoRA 200-step loss reduction, Full-FT 200-step sanity,
  192-tensor LoRA A/B before/after diff, base-frozen invariant) cover
  the fix. Tracked upstream as huggingface/candle#3011 (issue) and
  huggingface/candle#3613 (fix PR); this workaround is a permanent
  shim as long as the pinned `candle-nn` version predates the upstream
  fix.
- `algocline-nn`: propagate `candle-nn/cuda` in the `nn-cuda` feature.
  The feature previously enabled only `candle-core/cuda`, leaving
  `candle-nn` without its own cuda feature. `candle-nn` 0.11's
  `LayerNorm` is a `CustomOp3` whose `cuda_fwd` is
  `#[cfg(feature = "cuda")]`-gated, so a `candle-core/cuda`-only build
  still dispatched CustomOp on CPU and failed at runtime with
  `Candle("no cuda implementation for layer-norm")`. Enable
  `candle-nn/cuda` in the same feature and keep the feature pair in
  sync when bumping candle.

### Security

### Internal

## [0.45.0] - 2026-07-11

### Added

- `alc.llm(prompt, { cache_breakpoint = "..." })` and `alc.llm_batch({ { prompt = ..., cache_breakpoint = "..." } })` accept an optional opaque `cache_breakpoint` hint string. The engine forwards it verbatim on the paused-session JSON as the top-level `cache_breakpoint` field (single-query) or as a per-query `cache_breakpoint` field inside `queries[]` (batch). Absent when not set. The MCP host is responsible for mapping it to the provider-specific prompt-cache API (e.g. Anthropic `cache_control` blocks). Hosts that do not implement prompt caching MUST ignore the field. `LlmQuery::cache_breakpoint` in `algocline-core` is `Option<String>` with `skip_serializing_if = "Option::is_none"` so JSON shape stays additive. `PendingFilter::cache_breakpoint` (bool) controls projection into `Snapshot::pending`; enabled by `PendingFilter::preset_full()`.
- Two new MCP tools for `recipe_trace` observability:
  - `alc_trace_query`: scan every Card sample sidecar for rows carrying a `.trace` object (produced by the bundled `recipe_trace` pkg via `M.card_row(...)`) and return matches. Filters: `pkg`, `min_calls` / `max_calls` (over `.trace.total_calls`), `min_ms` / `max_ms` (over `.trace.total_trace_ms`), `completed`; page with `offset` + `limit`. Rows without `.trace.total_calls` are silently skipped without consuming `offset` so paging is stable across mixed-trace Card stores. Response is a JSON array of `{ card_id, pkg, sample_index, trace: {total_calls, total_trace_ms, completed}, case? }`.
  - `alc_trace_diff`: compare two individual trace rows. Params: `{a_card_id, a_sample_index?, b_card_id, b_sample_index?}` (missing indices default to 0). Response includes raw `a_trace` / `b_trace` triples plus a `delta` object with `calls_delta`, `ms_delta`, and `ms_per_call_delta` (all `b - a` semantics). Typed errors for missing card, out-of-range sample index, and rows without a `.trace` object.

  Both tools are read-only and re-use the existing `FileCardStore` — no new persistence layer is introduced. `EngineApi::trace_query` and `trace_diff` are defined with default `Err(...)` impls so downstream `EngineApi` impls that do not care about traces continue to compile unchanged.

### Changed

- Bump bundled source algocline-bundled-packages v0.24.0 → v0.29.1
- Bump `fuzzy-parser` dependency 0.1 → 0.3 in `algocline-engine`. The `distance` module (`similarity`, `Algorithm::JaroWinkler`) used by `alc.match_enum` / `alc.match_bool` bridge functions is unchanged; v0.3 adds `extract` / `repair` / `import` modules that are not yet surfaced through the Lua bridge.

### Deprecated

### Removed

### Fixed

### Security

### Internal

## [0.44.4] - 2026-06-21

### Added

### Changed

### Deprecated

### Removed

### Fixed

- `bridge::data` sentinel sweep: all 15 sibling `lua.to_value(&...)` callsites within `crates/algocline-engine/src/bridge/data.rs` (`alc.state.*`, `alc.card.*`, `alc.stats.*`, `alc.alias.*`, etc.) are now funneled through a new internal `to_lua_value` helper that disables `serialize_none_to_null` / `serialize_unit_to_null`. Previously these callsites used mlua's default `to_value`, which surfaces `serde_json::Value::Null` (and `Option::None`-shaped values inside typed structs) as a lightuserdata sentinel — the same root cause that v0.44.3 fixed for `alc.json_decode`. Downstream `if v.x then ...` truthy checks on nullable fields retrieved via `alc.state.get` etc. now skip the branch as expected. `alc.json_decode` is also re-routed through the same helper for a single SoT. Resolves issue ff6372af (sibling of db041966).

### Security

### Internal

## [0.44.3] - 2026-06-21

### Added

### Changed

### Deprecated

### Removed

### Fixed

- `alc.json_decode`: JSON `null` now decodes to Lua `nil` instead of an mlua lightuserdata sentinel. Previously `if obj.x then ...` truthy checks would proceed for nullable fields and downstream operations (e.g. `io.open(value)`) would crash when receiving the sentinel. Array elements decoded from `null` are now nil at that index; `#arr` is preserved at the original JSON length (mlua/Lua 5.4 array part semantics). Consumers iterating with `for i = 1, #arr do ... if arr[i] then` remain safe; `ipairs()` stops at the first nil hole as per standard Lua semantics. Aligns implementation with the existing `bridge_test.lua` API contract (`alc.json_decode("null") == nil`). Resolves issue db041966.

### Security

### Internal

## [0.44.2] - 2026-06-14

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Internal

- Internal: extract `pool_*_impl` inherent methods + co-located tests from `service/engine_api_impl.rs` into new `service/pool.rs` (mechanical move, no behavior change). GH Issue #7.

## [0.44.1] - 2026-06-14

### Changed

- Bundled packages: `algocline-swarm-frame` bumped from `v0.10.0` to `v0.11.0`. v0.11.0 adds the new `swarm_state_method` v0.1.0 package (Phase C of the state-primitive 2-layer split — see `docs/state-management.md`); `hub_index.json` grows from 5 to 6 entries. v0.10.0 backfill: CHANGELOG entries for the already-shipped `verdict_loop_plugin` v0.1.0 and `swarm_frame_algocline` v0.2.0 → v0.3.0 step lifecycle + ctx.dispatch additions. Run `alc update` to pick up the new collection.

## [0.44.0] - 2026-06-14

### Added

- `alc_state_delete` MCP tool: delete a state key in a namespace with atomic `.bak` backup; returns `{"ok":true,"existed":<bool>}` where `existed` signals prior key presence (idempotent: second call returns `existed:false` without error).
- `alc_state_list` MCP tool: list state keys in a namespace (returns `{"keys":[...]}`, sorted alphabetically, `.bak`/`.tmp` excluded).
- `alc_state_reset` MCP tool: reset `completed_steps` entries and data fields in an orch state file, creating an atomic `.bak` backup.
- `alc_state_set` MCP tool: write or overwrite a state key in a namespace with atomic `.bak` backup on overwrite; returns `{"ok":true}`.
- `alc_state_show` MCP tool: return the full JSON content of a state file by namespace and key.

### Notes

The `alc_state_*` tools (Phase A: list/show/reset, landed in merge commit `15e58bf`; Phase B:
set/delete) follow the 0.37.0 rollback boundary: all parameters use namespace-generic `String` /
`serde_json::Value` arguments (no `OrchState`, `CodingState`, `DispatchKey`, or other
application-domain type). Phase B adds `state_set` and `state_delete` to the `EngineApi` trait
with namespace-generic signatures. This is permitted under the 2026-06-13 narrowed constraint
update to `docs/state-management.md` (L148–163): constraint #2 was narrowed from "no new
`EngineApi` trait methods" to "no application-term leak in trait method signatures". The
`state_set(namespace: String, key: String, value: serde_json::Value)` and
`state_delete(namespace: String, key: String)` signatures contain only primitive types — the same
precedent as the 0.23.0 / 0.24.0 BREAKING additions.

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.43.0] - 2026-06-13

### Added

- `alc-adviser` agent: substrate cross-check step (3b) — before composing a
  design proposal, the adviser reads `plugins/alc/skills/alc-wake/SKILL.md`
  for the canonical Swarm framework primitive list, the workspace `alc.toml`
  / `alc.local.toml` `[packages]` section, and each in-use substrate's
  `init.lua` from `~/.algocline/packages/<sub>/`. Every gap finding emitted
  in `### Design Proposal` is now paired with either a literal substrate
  primitive path or the explicit phrase `no primitive applies`, preventing
  downstream re-invention of existing primitives (e.g., state persistence,
  dispatcher routing, verdict-loop gating) (GH #3).
- `alc-refiner` agent: substrate cross-check step (2b) — analogous to the
  adviser cross-check but Read/Write only (no MCP tools). Refiner proposals
  now carry a paired substrate primitive path or `no primitive applies` per
  proposed change, with `### Refiner Proposal` blocked on any of (a-0) Skill
  Read / (a) workspace `alc.toml` Read / (b) substrate `init.lua` Read
  failures (GH #3).
- `alc-coder` agent scaffold: `.stylua.toml` write step — the coder now
  writes a default `.stylua.toml` (`column_width = 100`,
  `indent_type = "Spaces"`, `indent_width = 4`) at `<pkg_root>/<name>/`
  during package scaffold, idempotent on existing files. Aligns scaffold
  output with the algocline-wide Lua formatter convention (GH #5).

### Fixed

- `alc-adviser` / `alc-refiner` agents: journal append SOP no longer
  recommends the `Read` full content → `Write` whole file pattern.
  Documentation now points to append-only Tool / Recipe paths (e.g.,
  `mcp__lds__recipe_run(recipe="journal-append", ...)` or
  `Bash(printf ... >> path)` enclosed in a recipe). Prevents recurrence of
  the 2026-06-10 incident (`94f692ef`) in which ~950 lines of `journal.md`
  were lost via Read-full → Write truncation. Frontmatter `tools:` entry
  removal and physical append-only Tool wiring are tracked separately.

## [0.42.2] - 2026-06-12

### Fixed

- `alc_advice`: resolve variant-scope linked packages from `alc.local.toml` before
  the global-tier install check, allowing worktree-scoped strategies to be applied
  without manual `alc_pkg_install` (#2).
- alc_advice: propagate variants slice into resolve_pkg_type_lua library guard so variant-scope linked packages pass the type-detect Lua VM (#2).

## [0.42.1] - 2026-06-12

### Security

- `alc_card_publish`: Derive pkg name from the on-disk locator instead of the Card TOML body to prevent CWE-22 path traversal. A malicious Card with `pkg.name = "../arbitrary"` (landable via `card_install`) could previously escape the staging tempdir on `create_dir_all` and read outside the card store root on `fs::copy`. The body-side `pkg.name` is now intentionally ignored; `pkg_name` is derived from `CardStore::find_card_locator(card_id).parent().file_name()` and passed through the canonical `algocline_engine::card::validate_name` guard. The silent `"unknown"` fallback is also removed.

## [0.42.0] - 2026-06-12

### Added

- `alc_card_publish`: New MCP tool that publishes a Card to a hub repository — runs `git clone` of `target_repo` to a staging dir, copies the Card files in, runs `git add` / `commit` / `push origin HEAD`, then calls `alc_hub_reindex`. Inputs: `card_id`, `target_repo` (URL: http/https/file/git@/ssh; pkg slug reserved for future versions), `commit_message` (optional; defaults to `"publish card {card_id}"`). Outputs: `{ published_url, commit_hash, reindex_status: { ok, output?, error? } }`. Push failures due to missing credentials return a typed `CardPublishError::MissingCredentials` with guidance text derived from the host's actual state (gh auth status, SSH key presence, git config user.{name,email}, origin remote). Push success and reindex failure are surfaced as independent fields — a successful push is never rolled back when only reindex fails. E2E tests in `tests/e2e.rs` cover (a) happy path, (b) push failure as typed error, (c) reindex failure isolated. (Issue #1)
- `alc_info`: Response now includes a `gh_credentials` field with per-component status — `gh_auth` (gh CLI availability + logged-in flag), `ssh_keys` (presence of `~/.ssh/id_*` candidates), `git_config` (user.name / user.email), and `origin_remote` (current project's origin remote URL). The same diagnostic helper backs the `alc_card_publish` credential error message so guidance text is grounded in actual host state rather than a generic hint. Additive, non-breaking.

## [0.41.3] - 2026-06-09

### Changed

- Bundled packages: `algocline-swarm-frame` bumped from `v0.9.0` to `v0.10.0`. Run `alc update` to pick up the new collection.

## [0.41.2] - 2026-06-07

### Added

- `alc_pkg_test` sandbox now mirrors the full `alc.*` primitive surface that `alc_run` exposes (stateless helpers like `alc.json_encode`, `alc.fingerprint`, `alc.fuzzy.*` are callable directly from specs; stateful helpers `alc.state.*` / `alc.card.*` are backed by a per-VM in-memory tempdir). A Pure-Lua mock layer adds `with_alc(overrides, fn)` for scoped overrides, `alc_mock.install/restore` for `before_each` setup, and `alc.spy(name, default_fn?)` for call observation. External-I/O entries (`alc.llm`, `alc.llm_batch`, `alc.fork`) are present as stubs that error with a `mock required: alc.<name>` message until `with_alc` overrides them. The invariant `production primitive surface ⊆ test sandbox primitive surface` is enforced at test time by `crates/algocline-engine/tests/bridge_sandbox_parity.rs`. Fixes the asymmetry that forced packages to embed inline pure-Lua JSON encoders to keep spec tests passing. New public exports on `algocline-engine`: `bridge` module is now `pub`, with `bridge::install_for_pkg_test(&Lua) -> LuaResult<()>` and `bridge::PRELUDE` available to crate consumers. Additive, non-breaking. See [docs/lua-stdlib.md §Test Sandbox](docs/lua-stdlib.md) for spec-author API. (Issue 7dc77cc7)

### Changed

- Bundled packages: `algocline-swarm-frame` bumped from `v0.8.0` to `v0.9.0`. v0.9.0 adds control-flow combinators on top of v0.8.0: `swarm_frame.sequence` (ordered Handler list with short-circuit on non-DONE), `swarm_frame.loop` (bounded iteration with predicate exit), `swarm_frame.branch` (single-shot conditional dispatch), and `swarm_frame.verdict_loop` (retry-on-FAIL gate). `combinator_demo` 0.1.0 ships as a new example package (minimal `verdict_loop` driver). lshape schemas `SwarmFrame.{SequenceOpts, LoopOpts, BranchOpts, VerdictLoopOpts, Handler}` are registered in `default_registry`. Mechanism/policy split: the Engine owns iteration / short-circuit / `cp_state` idempotent persistence; `parser` / `cond` / `fix` remain the caller's domain. Additive, non-breaking. Run `alc update` to pick up the new collection.

## [0.41.1] - 2026-05-31

### Added

- `alc.fmt(fmt, ...)` and `alc.log_fmt(level, fmt, ...)` Layer 1 prelude combinators: safe `string.format` drop-ins for LLM-derived metrics. Integer specs apply half-away-from-zero rounding (`1.5 -> 2`, `-1.5 -> -2`), NaN/+Inf/-Inf are rewritten to `%s` with `"NaN"`/`"Inf"`/`"-Inf"` substitutions, string-shaped numbers re-coerce via `tonumber`, and `%s` + `nil` falls back to `"<nil>"`. Other specs (`%s %f %.Nf %q %g` ...) pass through identically. Additive, non-breaking.

### Changed

- Bundled packages: `algocline-swarm-frame` bumped from `v0.7.0` to `v0.8.0`. The collection now ships three packages: `swarm_aggregate_plugin` 0.1.0 (new — bridges dmad / moa / reconcile aggregate algorithms onto `swarm_frame_algocline.make_dispatcher`), `swarm_frame` 0.8.0 (artifact_store + 4-method backend + summarize on top of v0.7.0), and `swarm_frame_algocline` 0.2.0 (Token & Prompt round-trip primitive with format mode + resolve_task_dir). Run `alc update` to pick up the new collection.

## [0.41.0] - 2026-05-29

### Changed — **BREAKING**: removed explicit M.meta.type declaration

- `M.meta.type` explicit declaration is removed (bug fix: Lua strategy packages have no bin/lib distinction). Package type is now always auto-detected from `type(pkg.run) == "function"` via VM eval.
- `alc_pkg_doctor` no longer emits the `unmarked_library` bucket (the explicit-type suggestion is obsolete). v0.40.0-only addition; 0 packages used the feature.
- `TypeSource::Explicit` variant removed (breaking for Rust API consumers / trait implementors only; wire `"explicit"` in legacy hub_index.json deserializes to `null` gracefully). v0.40.0-only addition; 0 packages used explicit type.
- `alc_pkg_list`: `warnings` field with explicit-type suggestion removed.
- hub `build_index` now derives type via VM eval (single source of truth) instead of text scan.

## [0.40.0] - 2026-05-29

### Added

- `alc-adviser` agent: Design Consultation mode. When the query expresses a build intent ("I want to build ...", "How should I combine ..."), the adviser switches from lookup mode to design consultation, searching existing packages as building blocks, finding reference implementations, and returning a `### Design Proposal` with architecture sketch and package combination suggestions. No Rust changes required — the mode uses existing MCP tools (`alc_pkg_list`, `alc_hub_search`, `alc_hub_info`).
- `PkgEntity` now carries a `type_source` field (`"explicit"` | `"auto_detected_runnable"` | `"auto_detected_library"`) recording the provenance of `pkg_type`. Backward-compatible via `#[serde(default)]`; legacy entries deserialize as `None`. Path A (`parse_from_init_lua`) and Path B (`LUA_TYPE_AUTODETECT` Lua snippet) both populate `type_source` independently.
- `alc_pkg_doctor`: new `unmarked_library` diagnostic bucket (11 total). Flags packages auto-detected as libraries (no `M.run`, no `M.meta.type`) and suggests adding `M.meta.type = "library"` for explicit declaration. Packages with `M.meta.type` explicitly set are unaffected; legacy entries without `type_source` are never flagged.
- `alc_pkg_list`: each package entry now includes an optional `warnings` array. When a package is auto-detected as a library, the entry carries a suggestion to declare `M.meta.type = "library"` explicitly. Field is omitted for entries without warnings.

### Fixed

- `alc_pkg_doctor`: alive symlinks under `~/.algocline/packages/` that are not registered in `installed.json`, `alc.toml`, or `alc.local.toml` are now reported (additive entries into `unmarked_library` or `unregistered_pkg`; JSON shape unchanged).
- `alc_pkg_doctor`: alive-symlink `type_source` detection now uses the same `eval_simple` + `LUA_TYPE_AUTODETECT` runtime path as `alc_pkg_list`, eliminating a static-parse divergence that caused 5 auto-detected library packages to be misclassified as `unregistered_pkg` instead of `unmarked_library`.

## [0.39.0] - 2026-05-28

### Added

- `PkgType` enum (`"runnable"` | `"library"`) introduced in `algocline-core`. Package authors can declare `M.meta.type = "runnable"` or `M.meta.type = "library"` in `init.lua`. When `M.meta.type` is absent the type is auto-detected at runtime: the Lua VM path uses `type(pkg.run) == "function"` (canonical; used by `alc_pkg_list`, `alc_advice`, and `alc_eval`); the offline `build_index` / `alc_hub_reindex` path uses a Rust text-scan (`detect_has_run`) as a mirror. A package without `M.run` defaults to `Library`.
- `pkg_type` field added to `PkgEntity` (serialized as `"type"` on the wire, `null` for legacy entries) and `ManifestEntry` (`installed.json`). Both are backward-compatible via `#[serde(default)]`.
- `alc_pkg_list` now includes the `type` field in each package entry.
- `alc_hub_reindex` / `alc_hub_search` now include the `type` field in `hub_index.json` entries and search results.
- `alc_advice`: library packages (resolved `type = "library"`) are rejected before `M.run` is invoked. The error message names the package and suggests `alc_run` with custom import code as an alternative.
- `alc_eval`: library packages are rejected before any LLM call is initiated. The error message names the package and suggests using a runnable package instead.
- Lua auto-detect snippet (`LUA_TYPE_AUTODETECT` constant in `resolve.rs`) is the single shared source for the `type(pkg.run)` detection logic injected by all runtime paths, preventing per-path divergence.
- `parse_meta` internal return type refactored from a 5-element tuple to a named `ParsedMeta` struct, making the parser self-documenting and future field additions non-breaking.

### Fixed

- `alc_hub_search`: pre-type-system `hub_index.json` entries (no `"type"` field) no longer produce `"type": null` in search results. `merge()` in `hub.rs` now defaults a missing `pkg_type` to `PkgType::Runnable` on both the remote-index path and the local-only fallback path. Packages with an explicit `"type"` value are unaffected. (backward compat fix: `hub.rs` `merge()`)

### Security

- Session ID generation now uses `rand::rng()` (ChaCha12 CSPRNG) instead of `RandomState`-based hashing (`gen_session_id` in `algocline-engine` and `gen_pool_sid` in pool dispatch). The previous `RandomState` path provided no cryptographic randomness guarantee and derived entropy solely from the per-process ASLR seed.
- App-layer session IDs (`alc-sess-*`) now include a `rand::rng().random::<u32>()` suffix in addition to the millisecond timestamp, eliminating guaranteed collisions for concurrent calls within the same millisecond.

## [0.38.6] - 2026-05-27

### Changed

- Bump `BUNDLED_SOURCES` tag for `algocline-swarm-frame` from v0.5.0 to v0.7.0 (5 spike examples + test infrastructure)

## [0.38.5] - 2026-05-26

### Added

- `M.meta.tags` support: package authors can declare `M.meta.tags = {"tag1", "tag2"}` in `init.lua`. Tags are parsed during `alc_hub_reindex`, projected into `hub_index.json`, and searchable via `alc_hub_search` (case-insensitive substring match against tags).
- `alc-coder` / `alc-refiner` agents now accept optional enrichment inputs (`reference_docs`, `negative_examples` for coder; `existing_tracker`, `reference_docs` for refiner) to provide richer context during package implementation and review.

### Fixed

- `alc-coder` agent now explicitly prohibits generating `LICENSE` files from LLM memory, preventing incorrect license text from being written into packages.
- `alc_hub_search` tag projection: fixed field inclusion so that `M.meta.tags` values appear correctly in search results.

## [0.38.4] - 2026-05-26

### Added

- `crates/algocline-app/src/service/lua/gendoc/docs/lint.lua` — new `W_META_LEGACY_M_VERSION` warning. When a pkg's `init.lua` defines a top-level `M.VERSION` field, `alc_hub_gendoc` now surfaces a non-blocking warning naming the canonical form (`M.meta.version` per `pkg-author-conventions.md` §2.1). The flag is collected by `extract.lua` (new `pkg_info.identity.legacy_m_version : boolean`) so lint emits regardless of whether `M.VERSION` and `M.meta.version` are kept in sync. Safe-to-remove when no external reference exists.

### Docs

- `docs/pkg-author-conventions.md` §2.1 — added `alc_shapes_compat` to the `M.meta` table as a **Recommended** field, closing a long-standing doc / implementation drift. The field has been first-class in `gendoc.rs` since the alc_shapes compat era (extract + SemVer-range validate + bundled-alc_shapes range mismatch error + undeclared warning) and is included in the `pkg_scaffold` template; only the convention doc was missing it. A Notes paragraph documents the existing gendoc enforcement behaviour (range mismatch → error; absent → warning).
- `docs/pkg-author-conventions.md` §5 — added `W_META_LEGACY_M_VERSION` row to the lint rules table.

### Internal

- `crates/algocline-engine/src/execution/registry.rs` — added `SessionRegistryV2::spawn_gc_task(ttl: Duration, interval: Duration)`, mirroring the legacy `SessionRegistry::spawn_gc_task` eviction contract (Option A: legacy parity). The GC task runs at the given `interval`, holds a write lock for the full check-and-remove sequence, and evicts only sessions that are both (a) idle beyond `ttl` (wall-clock ms via the new `last_active` field) and (b) have zero live broadcast subscribers (`bus_tx.receiver_count() == 0`), making eviction safe under concurrent `observe()` calls (Crux: concurrent GC eviction safety).
- `crates/algocline-engine/src/execution/record.rs` — added `pub(crate) last_active: Arc<AtomicI64>` to `SessionRecord`. Updated at driver-loop entry via `Ordering::Relaxed` (single writer = driver\_loop, single reader = GC), matching the `Session.last_activity_ms` pattern in legacy `session.rs` (Crux: legacy parity eviction behavior). Resolves debt #2.
- `crates/algocline-engine/src/execution/driver.rs` — `driver_loop` now accepts and stores the `last_active: Arc<AtomicI64>` parameter, writing `now_ms()` on entry to seed the GC idle clock.
- `crates/algocline-app/src/service/mod.rs` — `AppService::new` now wires `execution_registry.spawn_gc_task(Duration::from_secs(10800), Duration::from_secs(60))` immediately after constructing the V2 registry, matching the legacy 3-hour TTL / 60-second interval. Resolves debt #4.
- `crates/algocline-app/src/service/execution_service_impl.rs` — `observe_sink_free_when_no_subscribers` test rewritten to construct `SessionRegistryV2` directly with a sub-second TTL (100 ms) and interval (50 ms), then assert `ObserveError::NotFound` after a deterministic sleep that guarantees at least one GC tick post-TTL expiry. The previous dual-accept (`Ok` or `NotFound`) is replaced by a strict `NotFound` assertion (Crux: TTL override for test determinism). Resolves debt #3.

- `crates/algocline-engine/src/bridge/fork.rs` — removed redundant `.into_iter()` call on `results` (a `Vec<Option<Result<...>>>`) in the `strategy_names.iter().zip(results.into_iter()).enumerate()` chain; `zip` accepts `impl IntoIterator` directly, so the explicit conversion was flagged as `clippy::useless_conversion`. Iteration order, item count, and `(name, result)` pairing are unchanged. Verified that `results` satisfies `IntoIterator` (not the `Iterator` trait itself) at the call site, confirming the implicit conversion path through `zip`'s bound produces identical item types.
- `crates/algocline-app/src/service/logging.rs` — replaced `sort_by(|a, b| b.1.cmp(&a.1))` with `sort_by_key(|b| std::cmp::Reverse(b.1))` to clear a `clippy::unnecessary_sort_by` warning found in the same workspace-wide audit pass. Sort order (newest first) unchanged.
- `crates/algocline-engine/src/execution/registry.rs` — `SessionRegistryV2::observe()`:
  added `tracing::warn!(target = "session.observe", ...)` in the `try_read()` `Err` arm
  to surface lock contention as an observable log event. `NotFound` return is preserved
  (no breaking change). Resolves debt #1 (tracing-missing-on-err, Outline §1-2-6).
- `crates/algocline-engine/src/execution/driver.rs` — introduced `pub(crate) struct DriverContext` to group the six shared `Arc`/token arguments of `driver_loop` (`state`, `bus_tx`, `cancel_token`, `resp_txs`, `last_active`, `metrics`). Field declaration order reproduces the original flat-argument drop order so that `Arc` reference-count sequencing is identical before and after the refactor (Crux: Arc ownership order preserved). `driver_loop` signature narrows from seven arguments to three (`ctx: DriverContext`, `exec_task`, `llm_rx`); the `mut`-owned `exec_task` and `llm_rx` remain flat to avoid borrow-split complexity inside the `select!` loop. All call sites in `driver.rs` updated to `ctx.*` field access (mechanical rename, no logic change).
- `crates/algocline-engine/src/execution/registry.rs` — `spawn_v2()` Arc-clone block condensed into a single `DriverContext { ... }` constructor; `driver_loop` invocation updated to the three-argument form. No behaviour change; pure internal refactor with no public API surface affected (`DriverContext` is `pub(crate)`).

- `crates/algocline-core/src/metrics.rs` — added `ExecutionMetrics::usage_aggregate() -> Option<TokenUsage>`. Returns `None` when `llm_calls == 0` (no LLM call occurred in this run) and `Some(TokenUsage { prompt_tokens, completion_tokens })` otherwise, preserving the `None` vs `Some(zero-valued)` wire-shape invariant. Mutex poison returns `None` with `tracing::warn!`, matching the existing lock-guard convention in `metrics.rs`.
- `crates/algocline-engine/src/session.rs` — `Session::into_driver_parts` now moves `ExecutionMetrics` out as a fourth tuple element instead of dropping it. All v2 callers updated accordingly.
- `crates/algocline-engine/src/execution/registry.rs` — `SessionRegistryV2::spawn_v2` wraps the metrics in `Arc<ExecutionMetrics>` and clones it into both `DriverContext.metrics` and `SessionRecord.metrics`, establishing a single shared counter per session that persists across `alc_continue` calls.
- `crates/algocline-engine/src/execution/driver.rs` — `DriverContext` gained a `pub(crate) metrics: Arc<ExecutionMetrics>` field. The v2 `Done` branch now returns `usage: ctx.metrics.usage_aggregate()` instead of the previous `usage: None` hardcode. Three test-only `DriverContext` constructors updated with `metrics: Arc::new(ExecutionMetrics::new())`.
- `crates/algocline-engine/src/execution/record.rs` — `SessionRecord` gained a `pub(crate) metrics: Arc<ExecutionMetrics>` field to enable `resume` to feed per-response token usage into the same accumulator as the driver loop.
- `crates/algocline-engine/src/execution/driver.rs` — v2 `Paused` branch (`llm_rx.recv()` arm) now calls `ctx.metrics.create_observer().on_paused(&queries)` after building the `Vec<LlmQuery>` from the outgoing requests. This increments `SessionStatus.llm_calls`, which is the gate for `usage_aggregate()` to return `Some(...)` rather than `None`.
- `crates/algocline-engine/src/execution/registry.rs` — `SessionRegistryV2::resume` now threads per-response `usage` from `ResumePayload` through to `record.metrics.create_observer().on_response_fed(...)` for both `Single` and `Batch` variants. The intermediate collection is widened from `Vec<(String, String)>` to `Vec<(String, String, Option<TokenUsage>)>` to carry usage to the observer call.
- `crates/algocline-app/src/service/tests/unit.rs` — added integration test `test_usage_accumulates_across_alc_continue` verifying that usage sums across consecutive `alc_continue` calls within the same run-id rather than resetting.
- `tests/e2e.rs` — added `test_alc_run_usage_populated` end-to-end test asserting that `alc_run` with an `alc.llm()` call returns a non-null `usage` field in the `ExecutionResult`.

### Added

- `.github/workflows/ci.yml` — GitHub Actions CI workflow for all pushes and pull requests. Runs `just ci` (`fmt-check lua-fmt-check clippy test check-invariants check-agent-index`) on `ubuntu-latest` with `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2` for dependency caching, and stylua 2.3.1 pinned via `cargo install --locked --version 2.3.1 stylua`. On test failure, insta `.snap.new` files are uploaded as a GitHub Actions artifact (`if: failure()`) to aid snapshot regression diagnosis.
- `README.md` — CI status badge linking to the new `ci.yml` workflow.

### Notes

- Patch bump (0.38.3 → 0.38.4). New `W_META_LEGACY_M_VERSION` lint warning is additive (no existing pkg emits an error from it). The `usage_aggregate()` addition on `ExecutionMetrics` is additive `pub fn`. `SessionRegistryV2::spawn_gc_task` is additive on a `pub` struct (`spec/usage` path was already public). The `DriverContext` refactor is `pub(crate)` only — no SemVer-visible surface change. CI workflow + Lua lint integration are repository infrastructure, not consumer-facing API.

## [0.38.3] - 2026-05-23

### Changed

- `plugins/alc/agents/alc-coder.md` — added `### Spec Authoring Conventions` section after Driver Loop step 5, codifying two `mlua-lspec` runtime-shape regularities the static check cannot catch: top-of-file `lust` destructuring, and the `M.spec.entries` stub form for smoke / no-`alc.llm` packages. Includes the `lust-global-assumption` anti-pattern explaining the one guaranteed retry these omissions cause.
- `plugins/alc/skills/alc-wake/SKILL.md` — Post-impl install procedure now branches by `/alc-build --location=` value. Global mode keeps the existing `cp → /tmp → git init → alc_pkg_install` synthetic-single-pkg path. Collection mode (new) covers (B-1) collection-local use via the 0.38.2 `alc_pkg_test project_root` fix without any install step, and (B-2) global registration via `alc_pkg_link` (in-tree dev) or `alc_pkg_install url=file://<collection_root>` (published-source semantics). Adds an explicit guard against forking the SoT by `cp`'ing to `/tmp`.

### Notes

- Patch bump (0.38.2 → 0.38.3). Plugin-side documentation (Agent / Skill body) only — no Rust source changes. Required because `plugins/alc/` ships as part of the Claude Code Plugin Marketplace catalog; the marketplace consumer view of this repo needs the doc updates surfaced at a tagged release.

## [0.38.2] - 2026-05-23

### Fixed

- `alc_pkg_test` `pkg=<name>` mode now honors the `project_root` argument for package discovery. Previously the parameter only fed `auto_search_paths` (the `require()` resolution side) and was ignored when locating `<pkg>/init.lua` itself, so packages living in a user collection repo (`/alc-build --location=collection` layout: any repo with `alc.toml` at its root holding flat `<root>/<pkg>/` directories) could not be tested via `pkg=<name>` and required the `code_file=<absolute spec> search_paths=[<root>]` workaround. The new search order is `<project_root>/<name>/init.lua` → cwd-ancestor `alc.local.toml` (variant) → `~/.algocline/packages/<name>/init.lua` (global); existing global / variant paths are preserved unchanged. (Surfaced by the alc-coder dispatch verifying the `--location=collection` mechanism in `plugins/alc/`.)

### Notes

- Patch bump (0.38.1 → 0.38.2). Bug fix only; no API additions or removals. The `pkg_resolve_init_path` signature added an optional `project_root` parameter — this is a `pub(crate)` function so it is not a SemVer-visible change for crates.io consumers.

## [0.38.1] - 2026-05-21

### Added

- `plugins/alc/` — Claude Code Plugin Marketplace plugin for algocline package development (adviser / coder / refiner agents + `alc-build` / `alc-wake` skills). Bundled `mcp.json` registers `algocline` and `git-reader` MCP servers automatically when the plugin is enabled. Distributed under MIT OR Apache-2.0 (matches the crate license).
- `.claude-plugin/marketplace.json` — Marketplace catalog so this repository can be registered as a Claude Code Plugin Marketplace source. Users can install bundled plugins with `/plugin marketplace add ynishi/algocline` followed by `/plugin install alc@algocline`.

### Changed

- README MCP register example simplified to the minimum form (`{"command": "alc"}`), with a one-line note that `env` is only needed when one of the documented variables is set.
- Bundled `algocline` entry in `plugins/alc/mcp.json` aligned with the same minimum form.

### Docs

- Documented the `ALC_CARD_SINKS` environment variable in README (Card subscriber backends as a `|`-separated URI list; v1 only accepts `file:///absolute/path`; duplicate URIs are first-wins; malformed entries are logged and skipped). The variable has been implemented and tested in `algocline-engine` for a while but was missing from the public docs.
- Added a Quick-start subsection in README that explains how to install algocline as a Claude Code plugin (`/plugin marketplace add ynishi/algocline` followed by `/plugin install alc@algocline`), so users discover the bundled plugin path without diving into `plugins/alc/`.

### Notes

- Patch bump (0.38.0 -> 0.38.1). The only crates.io-scoped change is the README docs addition for `ALC_CARD_SINKS`; binary / library APIs are unchanged. The `plugins/alc/` and `.claude-plugin/` additions are distributed via the Claude Code Plugin Marketplace and are not part of the crates.io publish set.

## [0.38.0] - 2026-05-21

### Changed

- Bumped bundled `algocline-swarm-frame` from `v0.3.0` to `v0.5.0` in `BUNDLED_SOURCES` (`src/init.rs`). Adds `swarm_aggregate_plugin` and Rich Verdict 2-layer separation. `alc init` / `alc update` will pull the new tag.
- `alc_eval`: scenario-side `provider` field now takes precedence over the auto-wired algocline provider. Falls back to auto-wire (`ef.providers.algocline { strategy = "..." }`) when scenario does not specify one. Enables 0-cost replay providers (e.g. `ef.providers.recorded { log_path = "~/.algocline/logs/s-<id>.json" }`) and `mock.recording` / `mock.map` via inline scenario without changing the MCP wire shape. The `strategy` argument is still required — borrow any installed package name (e.g. `cot`) when running a pure-replay smoke; the borrowed strategy module is loaded but never invoked because the scenario-side provider supersedes it. Smoke: `alc_eval(strategy="cot", scenario=<inline with provider=ef.providers.recorded {...}>)` returns `response.model = "recorded"`, `stats.auto.llm_calls = 0`. (Lua-level change in `crates/algocline-engine/src/prelude.lua`; no MCP wire change.)

### Added

- `alc_pkg_doctor` gains a tenth bucket `unregistered_pkg`: physical directories under `~/.algocline/packages/` that contain `init.lua` but are not registered in `installed.json`, `alc.toml`, or `alc.local.toml`. The `suggestion` field for this bucket is an `array<string>` (Clippy-style multi-line with install / link / remove / source-note options) rather than the `string` used by all other buckets. `target_filter = Some(name)` for a physically-present-but-unregistered package now routes to this bucket instead of returning an error. Path-dep entries in `alc.toml` / `alc.local.toml` that point inside `packages_dir` are skipped via canonical path comparison to avoid false positives.
- `alc_pkg_test` gains `auto_search_paths` parameter (optional `bool`, default `true`) and two new response fields. When `true` (or omitted), the Lua VM's `package.path` is automatically prepended with directories resolved from all three registry sources: installed packages under `~/.algocline/packages/`, `alc.toml` linked-global path entries, and `alc.local.toml` linked-variant path entries. Auto-resolved directories are placed before any caller-supplied `search_paths` without deduplicating or reordering caller entries. Set `auto_search_paths: false` to suppress auto-resolution entirely (zero auto-resolved paths injected, registry I/O skipped). New response fields: `resolved_search_paths` — array of `{name, search_dir, source}` objects mapping each resolved package to its directory and source (`"installed"` / `"alc.toml"` / `"alc.local.toml"`); always present as an empty array when `auto_search_paths` is false. `search_path_warnings` — array of strings surfacing canonicalization or I/O errors encountered during resolution; present only when non-empty (additive). (breaking for trait implementors only; `EngineApi::pkg_test` gains one parameter)
- `[setting.<target>]` tables in `alc.toml` / `alc.local.toml` / `~/.algocline/config.toml` — schemaless per-target configuration. Target names and field names are arbitrary; algocline core does not validate or enforce any per-target schema.
- MCP tool `alc_setting_resolve(target?)` — returns `{resolved, sources}` with per-field layer attribution (`env > project > global`) in a single call. Omit `target` to receive all configured targets.
- `alc init` now creates `~/.algocline/config.toml` from a commented template when the file is absent. Re-runs are idempotent (existing file is never overwritten).
- `alc info` JSON output gains a `settings` field surfacing the resolved `[setting.*]` tables and per-field source attribution. Parse errors are surfaced as `settings_error` (non-fatal; info remains usable as a doctor tool).

### Notes

- Breaking for `EngineApi` trait implementors only (a new required method `setting_resolve` is added; only `AppService` exists upstream). MCP wire shape is additive.
- env override naming: `ALC_SETTING_<TARGET>_<FIELD>` (uppercase snake). No legacy short aliases (e.g. `ALC_JOURNAL_PATH`) are provided.

## [0.37.0] - 2026-05-17

### Changed: bundled swarm-frame bumped to `v0.3.0`

`BUNDLED_SOURCES` in `src/init.rs` now points `algocline-swarm-frame` at tag `v0.3.0` (was `v0.2.0`). `alc init` and `alc update` will pull the new tag on next invocation.

### Added

- `alc_pkg_test`: Run mlua-lspec tests against a package's spec directory (`<pkg>/spec/*_spec.lua`), a single `code_file`, or inline `code`. Returns JSON `{passed, failed, pending, total, duration_ms, spec_files[]}` with per-file and per-test breakdown. `pkg` / `code_file` / `code` are mutually exclusive — exactly one must be provided. (breaking for trait implementors only; `EngineApi::pkg_test` has no default impl)
- **swarm-frame v0.2.0 as bundled Hub Collection**: Added `https://github.com/ynishi/algocline-swarm-frame` at tag `v0.2.0` as a third bundled source, auto-installed via `alc init` / `alc update`. Contributes 3 swarm orchestration packages (`swarm_frame`, `swarm_frame_algocline`, `swarm_aggregate_plugin`) to the algocline Hub. The BUNDLED_SOURCES entry carries `tag = "v0.2.0"` (pinned release); the AUTO_INSTALL_SOURCES entry tracks the URL without a tag (main-branch resolution). See `docs/pkg-author-conventions.md` for the collection-only install convention.
- `alc.env` stdlib (Phase 1) — host-owned readonly env snapshot exposed to the Lua VM.
  - `alc.env.KEY` returns the env value as a string, or nil if absent.
  - `alc.env:get(key, default)` returns the value or a fallback default.
  - `alc.env:use{"FOO", "BAR"}` declare-at-use proxy: undeclared keys return nil regardless of env contents.
  - Source priority (strict three-layer merge): `inject > dotenv > os.env`. OS env access is opt-in via `ctx.env.allow_os = true`.
  - dotenv file parsing via the `dotenvy` crate (new dependency); path supplied as absolute path in `ctx.env.dotenv`.
  - Optional allowlist filter via `alc.toml [env.allow]`; omitting the section allows all resolved keys.
  - Writes always error: `alc.env.X = 1` raises `"alc.env is readonly"` at the Lua runtime level.
  - Additive: existing `ctx` fields (e.g. `qwen_env`) are unchanged.
- `alc_pkg_doctor`: added two new diagnostic buckets — `missing_meta` (installed pkg with init.lua but no `M.meta.name`) and `missing_hub_index` (Collection-mode project_root with 2+ pkg dirs but missing hub_index.json). JSON output now contains seven top-level arrays (additive; existing five buckets unchanged).
- `alc_pkg_doctor`: added `stale_cache` bucket (8th verdict) — emits one entry per `~/.algocline/hub_cache/{hash}.json` file whose mtime exceeds `CACHE_TTL_SECS` (3600s). Same TTL discipline as `HubCacheLookup::Stale`, surfaced read-only. JSON output now contains eight top-level arrays (additive; existing seven buckets unchanged). `target_filter=Some` skips the entire pass (hub cache files are not per-package).
- `alc_pkg_doctor`: added `spec_missing` bucket (9th verdict) — emits one entry per installed pkg whose `spec/` directory exists but contains zero `*_spec.lua` files. Aligns with `alc_pkg_test`'s spec discovery convention. Narrow scope: pkgs without a `spec/` directory are silently skipped (opt-in). JSON output now contains nine top-level arrays (additive; existing eight buckets unchanged).
- `docs/pkg-author-conventions.md`: added §10 Cards quick reference (`alc_card_find` predicate cheatsheet + AND/OR/NOT examples) and §11 Pre-publish verification workflow (4-step pre-push checklist `alc_pkg_test` → `alc_hub_dist` → `alc_pkg_doctor` → `alc_hub_search local_indices` + failure-mode table).
- `alc.state.list(namespace) -> string[]`: List keys under the dispatched state layout `{state_root}/{namespace}/*.json`. Returns sorted, `.bak`/`.tmp` excluded.
- `alc.state.show(namespace, key) -> table`: Read full JSON content of `{state_root}/{namespace}/{key}.json`. Typed error (message contains `not found`) when the key is absent.
- `alc.state.reset(namespace, key, opts?) -> table`: Atomically mutate the state file. Creates `{key}.json.bak`, removes specified steps from `data.completed_steps` and fields from `data`, writes via tempfile + rename. Returns `{ok, backup_path, steps_removed, fields_removed}`.
- `StateError` enum (`thiserror`-derived) in `algocline-engine::state` with variants `KeyNotFound`, `UnsafeSegment`, `IoBackup`, `IoWrite`, `Serde`, `ShapeInvalid`. Used by the three new `JsonFileStore` inherent methods (`list_dispatched`, `show_dispatched`, `reset_dispatched_with_backup`).

### Notes

- **No MCP tool** is added (no `alc_state_list` / `alc_state_show` / `alc_state_reset`). Callers run these via `mcp__algocline__alc_run` with a short Lua snippet.
- **No `EngineApi` trait method** is added. The implementation lives entirely inside the `algocline-engine` crate as inherent methods on `JsonFileStore`.
- Replaces the rolled-back design (`alc_state_*` MCP tool + `OrchState` types) which leaked application terminology into the shared `algocline-core` / `-engine` / `-app` / `-mcp` crates. The new design keeps all application terms in caller packages.
- External consumer documentation (state.json editing SOP in downstream orchestration projects) will be updated separately to point at the Lua snippet path; that change lives outside this repo.

## [0.36.0] - 2026-05-15

### Changed: bundled-packages bumped to `v0.24.0`

`BUNDLED_SOURCES` in `src/init.rs` now points `algocline-bundled-packages`
at tag `v0.24.0` (was `v0.23.0`). `alc init` and the `AUTO_INSTALL_SOURCES`
seed picked up by `pkg_install` will fetch this newer tag.

### Changed — **BREAKING**: `pkg_install` single-package mode removed

`SourceKind::Single` and the staging `init.lua` auto-detection in `pkg_install`
have been removed. Hub publishing now requires `hub_index.json` (1 entry is
sufficient for single-package repos).

**Before** (v0.35.x and earlier):

```
# Single-package repo (init.lua at repo root) auto-detected:
alc_pkg_install({ url: "github.com/user/my-pkg" })
# → installed as ~/.algocline/packages/my-pkg/
# Response: { mode: "single", installed: ["my-pkg"] }
```

**After** (v0.36.0+):

```
# Collection-layout repo with hub_index.json:
alc_pkg_install({ url: "github.com/user/my-pkg-collection" })
# → installed as ~/.algocline/packages/<each-subdir>/
# Response: { mode: "collection", installed: [...] }
```

**Migrate (1-pkg authors)**:

1. Move `init.lua` from repo root to `<repo>/<pkg_name>/init.lua` (nested layout).
2. Add a minimal `alc.toml` at repo root with a `[hub]` section.
3. Run `alc_hub_dist` from a Claude Code / rmcp MCP session:
   ```
   alc_hub_dist(
     source_dir = ".",
     output_path = "hub_index.json",
     projections = ["hub", "narrative"]
   )
   ```
4. `git add hub_index.json docs/ && git commit && git push`.

See `docs/pkg-author-conventions.md §7` for the full step-by-step guide.

### Changed: evalframe v0.4.0 integrated into Hub as a Collection source

`evalframe` v0.4.0 now ships `hub_index.json` at the repo root and uses the
standard Collection layout (`evalframe/<pkg>/init.lua`). It is re-integrated
into `BUNDLED_SOURCES` (tag `v0.4.0`) and `AUTO_INSTALL_SOURCES` so that
`alc init` and `pkg_install` auto-routes pick it up alongside
`algocline-bundled-packages`.

`evalframe` packages are listed as system packages (`SYSTEM_PACKAGES`) and are
therefore excluded from `pkg_list` user-facing output while still being
discoverable via `alc_hub_search`.

### Added: `alc_hub_search` accepts `local_indices` parameter

`alc_hub_search` now accepts an optional `local_indices: Vec<String>` field.
Each path is read as a `hub_index.json` file and its packages are merged into
the search results after the remote-fetched results (same deduplication via
`seen_names`). Parse or I/O failures are surfaced as `warnings` entries in the
response rather than hard errors, so a missing or malformed local file does not
abort the search. This is useful for pre-push verification of a new
`hub_index.json` before committing.

## [0.35.0] - 2026-05-14

### Added

- **`ExecutionService` trait — pure Service-layer execution API** (`algocline-core::execution`).
  A new `ExecutionService` trait with six verbs (`spawn`, `state`, `resume`, `cancel`,
  `observe`, `await_terminal`) provides a wire-concept-free API surface for session
  lifecycle management. All types live under `algocline-core::execution` with zero
  dependency on `rmcp::*`, `progressToken`, `_meta`, or any other MCP wire concept,
  enforcing invariant 1 (transport-independent domain layer) at the type-system level.
  The trait coexists with the legacy `EngineApi` trait; `AppService` implements both.

- **`SessionRegistryV2` + cooperative cancellation (4-checkpoint driver loop)**
  (`algocline-engine::execution`). A new `SessionRecord` struct bundles a per-session
  `broadcast::Sender<ProgressEvent>` (capacity 256), a `CancellationToken`, and a
  `JoinHandle` for the background driver task. `SessionRegistryV2` manages a
  `HashMap<SessionId, Arc<SessionRecord>>` with a clone-then-release lock pattern to
  avoid holding the map lock across `.await` points (K-4). The driver loop checks
  cancellation at exactly four checkpoints:

  - **A** — before the Lua chunk begins (`cancel_token.is_cancelled()`)
  - **B** — before publishing a pause notification (`cancel_token.is_cancelled()`)
  - **C** — at the resume entry path (`cancel_token.is_cancelled()`)
  - **D** — long-IO wait via `tokio::select!` racing `cancel_token.cancelled()`
    against `exec_task` / `llm_rx.recv()`

  No `JoinHandle::abort()` or process-kill path is introduced; shutdown is always
  cooperative. Sessions return immediately from `spawn` while the driver runs in the
  background (`tokio::spawn`), resolving debt #1778655047-40955 (blocking
  `wait_event` before `SessionId` return).

- **`ProgressEvent` broadcast bus — sink-free multi-observer fan-out**
  (`algocline-core::execution::progress`). A `tokio::sync::broadcast` channel with
  capacity 256 carries a tagged enum of seven variants (`StateTransition`,
  `LlmCallBegin`, `LlmCallEnd`, `PauseRequested`, `ResumeReceived`,
  `CancelRequested`, `Tick`). The `observe` verb returns a `broadcast::Receiver`
  wrapped in `ObserverHandle` as a synchronous call — no pre-registered sink is
  required. Zero-observer sessions accept `send()` calls without error (broadcast
  standard behaviour). Slow observers receive `RecvError::Lagged(n)` rather than
  silent drops, preserving loss observability. Multiple independent subscribers each
  receive the full event stream without affecting one another.

- **`alc_v2_run`, `alc_v2_state`, `alc_v2_resume`, `alc_v2_cancel` MCP tools**
  (`algocline-mcp`). Four new MCP tools that expose the `ExecutionService` API over
  the MCP wire with zero wire-concept leakage into the Service layer. All four tools
  coexist with the legacy `alc_run` / `alc_continue` path under a `_v2` prefix;
  neither the wire shape nor the behaviour of existing tools is altered.

  - **`alc_v2_run`** — spawns a new execution session via `ExecutionService::spawn`.
    Accepts `code` (Lua source), optional `session_id` (caller-supplied idempotency
    key), and optional `ctx` (JSON context). When the caller's MCP request carries
    `_meta.progressToken`, a `ProgressForwarder` task is spawned to stream
    `ProgressEvent` notifications back to the caller via
    `Peer::notify_progress`; if no token is present no forwarder is spawned and no
    progress notification is emitted for that request (Crux invariant 2).
    Returns `{ session_id, status, result?, error? }`.

  - **`alc_v2_state`** — returns the current `ExecutionStateV2` snapshot for a
    session (`ExecutionService::state`). Read-only; annotated
    `read_only_hint = true, idempotent_hint = true`.

  - **`alc_v2_resume`** — resumes a paused session by injecting a `ResumePayload`
    (`Single { response, usage?, query_id }` or `Batch(Vec<QueryResponse>)`) via
    `ExecutionService::resume`. Returns the updated session state.

  - **`alc_v2_cancel`** — cancels a session via `ExecutionService::cancel` with
    `CancelCode::User`. Annotated `idempotent_hint = true`; cancelling a session
    that is already at a terminal state is a no-op.

- **`ReqIdRegistry` — adapter-owned request-id ↔ session-id mapping**
  (`algocline-mcp::req_registry`). A new `ReqIdRegistry` struct wraps
  `Arc<RwLock<HashMap<RequestId, SessionId>>>` and provides `insert` / `lookup` /
  `remove_by_session` / `remove_by_request` with a clone-then-release lock pattern
  that never holds the lock across an `.await` point (K-4). The registry is owned
  exclusively by the MCP adapter crate; no wire type (`RequestId`, `ProgressToken`,
  `_meta.*`) crosses the crate boundary into `algocline-core` or `algocline-app`
  (Crux invariant 1). Entry lifetime: inserted on `alc_v2_run` success, removed
  unconditionally by a background `await_terminal` task on session completion, and
  also removed on `alc_v2_cancel` success.

- **`ProgressForwarder` — per-token progress bridge**
  (`algocline-mcp::progress_forwarder`). A free function
  `spawn_progress_forwarder(execution, peer, sid, token)` that calls
  `ExecutionService::observe` to obtain a `broadcast::Receiver<ProgressEvent>` and
  then forwards each event to the MCP caller via `peer.notify_progress`. Slow-reader
  `RecvError::Lagged(n)` is forwarded as a synthetic `{"kind":"lagged","n":n}`
  notification so the caller can detect message loss without silently dropping events.
  `RecvError::Closed` and a `notify_progress` send error both cause the task to exit
  cleanly without panic. Spawned if and only if `_meta.progressToken` is `Some`
  (Crux invariant 2).

- **`ServerHandler::on_cancelled` override — cooperative cancellation from MCP**
  (`algocline-mcp::AlcService`). Overrides the default empty `on_cancelled`
  implementation. When the MCP client sends a `notifications/cancelled` message, the
  handler resolves `request_id → SessionId` via `ReqIdRegistry::lookup` and then
  calls `ExecutionService::cancel` with `CancelCode::User` (Crux invariant 3). A
  lookup miss (mapping already removed or request never registered) is logged at
  `DEBUG` and treated as a no-op. No `JoinHandle::abort()` or direct channel-close
  path is introduced; cancellation is always routed through `ExecutionService`.

- **`impl ExecutionService for AppService`**
  (`algocline-app::service::execution_service_impl`). Wires the six trait verbs to
  the new `SessionRegistryV2`. The legacy `AppService::run` / `EngineApi` path is
  untouched; the new `Arc<SessionRegistryV2>` field added to `AppService` is additive.

- **Value types for execution module** (`algocline-core::execution`).
  Approximately 20 public types exported from `algocline-core`: `SessionId`,
  `SessionSpec`, `ExecutionStateV2`, `ExecutionStateTag`, `PauseInfo`, `ResumePayload`,
  `ResumeOutcome`, `CancelReason`, `CancelCode`, `CancelInfo`, `ExecutionResult`,
  `FailureInfo`, `TerminalOutcome`, `ProgressEvent`, `ObserverHandle`, and seven closed
  error enums (`SpawnError`, `StateError`, `ResumeError`, `CancelError`, `ObserveError`,
  `AwaitError`, `ObserverRecvError`). All types implement `serde::Serialize +
  serde::Deserialize`. Error enums use `thiserror::Error` with `#[from]` converters for
  upstream error types. No `#[non_exhaustive]` (wire boundary adapter is responsible
  for string conversion before crossing the MCP boundary).

### Changed

- **Bump `rmcp` 1.5 → 1.7** across the workspace (`Cargo.toml`,
  `crates/algocline-mcp/Cargo.toml`). rmcp 1.7.0 ships
  [rust-sdk#843](https://github.com/modelcontextprotocol/rust-sdk/pull/843)
  which flattens `PromptMessageContent::Resource` so embedded resource
  serialization matches the MCP spec (`{ "type": "resource", "resource":
  { uri, mimeType, text } }` instead of the previous doubly-nested
  `content.resource.resource.{uri,...}` shape). This unblocks Phase
  1.x-D embedded resource support in `prompts/get` responses (no Phase
  1.x-D wire change in this release; the bump only lifts the upstream
  technical blocker). No source changes required in algocline — the
  current MCP server compiles against 1.7 unchanged.

- **`await_terminal` no longer busy-polls — JoinHandle take-and-await**
  (`algocline-engine::execution`). The previous implementation looped
  `state.lock().await → match → tokio::task::yield_now().await`
  indefinitely until the state transitioned to a terminal variant.
  `yield_now()` consumes no CPU but keeps the task in the scheduler's
  ready queue, so N parallel `await_terminal` callers occupy N worker
  slots that legitimate work cannot use. `SessionRecord.join_handle`
  is now `Mutex<Option<JoinHandle<()>>>` and `await_terminal` takes the
  handle, `.await`-s it directly, then reads the (now-terminal) state
  once via the existing mutex. Single-awaiter discipline is documented;
  a second concurrent caller that observes `None` falls through to a
  direct state read, returning `AwaitError::Joined("...concurrent
  awaiter race...")` only in the rare race where the state has not yet
  been written. Current production usage is single-caller, so the race
  branch is purely defensive. No API change. Regression coverage:
  `await_terminal_returns_done_for_trivial_script`,
  `await_terminal_does_not_panic_on_second_concurrent_caller`.

- **Remove `SessionRecord._sentinel_rx` (sink-free invariant clarified)**
  (`algocline-engine::execution::record`). The sentinel
  `broadcast::Receiver<ProgressEvent>` was held only to ensure
  `bus_tx.send(...)` returned `Ok` when 0 user observers were subscribed,
  but every production call site already uses `let _ = bus_tx.send(...)`
  to absorb the result. The sentinel only added cost — a 256-deep buffer
  per session with `ProgressEvent` clones cycling under `Lagged` eviction
  — without protecting anything that wasn't already protected by the
  `let _ = ...` pattern. The Crux R3 (sink-free fan-out) doc comment in
  `record.rs` is reworded to reflect the actual invariant: "caller is not
  crashed by 0 observers", not "`send` always returns `Ok`". The matching
  unit test is renamed `bus_tx_send_succeeds_with_zero_observers` →
  `bus_tx_does_not_crash_caller_with_zero_observers` with the
  `assert!(result.is_ok())` removed (the new contract is panic-freedom).
  No API change at `ExecutionService::observe`; existing observers
  continue to receive the full event stream with the same late-subscribe
  semantics.

- **`lagged_emits_wrapper_event` test flush is now deterministic**
  (`algocline-mcp::progress_forwarder`). Removed the 50ms wall-clock
  `tokio::time::sleep` that preceded the duplex-transport drop. The
  duplex pipe buffer preserves bytes the server side has already
  written past the server-side drop, so `drop(running) → read_to_end`
  on the client side drains them without a timing-dependent delay.
  Bumps the `read_to_end` timeout 300ms → 500ms so the overall
  wall-clock upper bound is unchanged on slow CI runners. Test-only
  change; no production code touched.

### Fixed

- **MCP `alc_run.ctx` / `alc_advice.opts` / `alc_eval.strategy_opts`
  schema now declares `type: object`** instead of the unconstrained
  schemars default for `serde_json::Value`. Some MCP clients infer an
  unconstrained field as a free-form string and JSON-stringify it
  before sending, which previously caused the Lua-side `ctx` global
  to arrive as `type(ctx) == "string"` and broke pkgs that required
  a table (`swarm_frame.normalize_ctx: ctx must be a table (got
  string)`). With the schema constrained to object, conforming
  clients send the value as a real JSON object and the existing
  `lua.to_value` path materialises a Lua table as documented. Tracked
  in `tests/e2e.rs::test_alc_run_ctx_schema_is_object`. No runtime
  type change — Rust deserialisation still accepts `serde_json::Value`.

- **Defensive server-side normalisation for non-conforming clients.**
  When `alc_run.ctx` / `alc_advice.opts` / `alc_eval.strategy_opts`
  arrive as a `Value::String` whose body parses as a JSON object or
  array, the server reparses the payload before injecting into the
  Lua VM (`AppService::run` / `::advice` / `::eval`). Conforming
  clients (which already send a real object) are unaffected; legacy
  callers that still stringify the field — including downstream
  consumer Skills that defensively called
  `alc.json_decode(ctx)` to recover from the bug — keep working
  without code changes. Regression coverage:
  `tests/e2e.rs::test_alc_run_ctx_stringified_json_normalized_to_table`.

- **Prevent `state_before` nesting in `CancelInfo` on `Paused → Cancelled`**
  (`algocline-engine::execution::driver`). When a session was cancelled
  while in the `Paused` state, two `Cancelled` transitions could fire in
  sequence — first from `SessionRegistryV2::cancel` (which transitions
  immediately so a paused driver does not hang waiting on a resume),
  then from `driver_loop` checkpoint D once `cancel_token` fires. Both
  paths called `build_cancel_info` independently and the second call
  read the now-`Cancelled` state, producing
  `Cancelled(state_before=Cancelled(state_before=Paused))` instead of
  `Cancelled(state_before=Paused)`. The fix is two cooperating changes:
  (1) `transition_state` is now idempotent on terminal → terminal
  transitions (first transition wins — state is not overwritten and no
  `StateTransition` event is emitted), and (2) `build_cancel_info` is
  nested-aware (when the current state is already `Cancelled`, the new
  `state_before` inherits the inner `state_before` rather than wrapping
  the outer `Cancelled`). Either change alone breaks the regression;
  both together also future-proof against new cancel paths that may add
  similar double-transition scenarios. Regression coverage:
  `transition_state_terminal_is_idempotent`,
  `build_cancel_info_does_not_nest_on_already_cancelled_state`.
  Verified on the wire via `alc_v2_run` paused → `alc_v2_cancel` →
  `alc_v2_state` — `state_before` is now the original `Paused` snapshot,
  not a nested `Cancelled` wrapper.

## [0.34.0] - 2026-05-10

### Changed

- **BUNDLED_VERSION bump: `algocline-bundled-packages` v0.22.1 → v0.23.0**
  (`src/init.rs` `BUNDLED_SOURCES`). Picks up the bundled
  `card_analysis` package release — the analyzer pkg dispatched by
  default from the new `alc_card_analyze` MCP tool. Bundled v0.23.0
  promotes the POC analyzer (previously available only at
  `~/.algocline/packages/card_analysis/`) into the bundled fleet
  with `M.spec` (input + result shapes locked to the host
  `CardAnalyzeResult` typed contract), `S.instrument(M, "run")`
  wrapper for runtime shape validation, and `tests/test_card_analysis.lua`
  (5 cases / 6 tests covering happy path / input validation / empty
  samples sentinel / 4 failure heuristic paths + no-signal fallback /
  LLM unparseable fallback). `evalframe` source remains pinned at
  v0.3.0. Bundled release notes: `algocline-bundled-packages` v0.23.0
  CHANGELOG / README.

### Added

- **`alc_card_analyze` MCP tool — Card auto-analyzer**.
  Loads a Card body and its samples sidecar host-side and dispatches
  them to a Lua analyzer pkg via `require(pkg).run(ctx)` (default
  `pkg = "card_analysis"`). Sister tool to `alc_advice`: `alc_advice`
  runs a generic strategy over a free-form task; `alc_card_analyze`
  runs an analyzer over a Card. The Card schema (Tier 1 body + Tier 2
  `samples.jsonl` sidecar) is owned by the host so analyzer pkgs only
  deal with prompt construction + `alc.llm` + hint shaping.

  ctx shape passed to the pkg's `M.run(ctx)` — enumerated in the MCP
  tool description so pkg authors do not need to read Rust source:

  ```jsonc
  {
    "card_id": "<id>",
    "card":    <full Card body, same shape as alc_card_get>,
    "samples": [<sidecar rows, same shape as alc_card_samples>]
  }
  ```

  `DEFAULT_CARD_ANALYZE_PKG = "card_analysis"` (constant in
  `algocline-app/src/service/card.rs`) is an **IF promise**, not a
  bundled hard dependency: any installed pkg of that name (bundled,
  project-variant, or user-installed) satisfies the dispatch. The
  analyzer pkg is currently expected to live at
  `~/.algocline/packages/card_analysis/`; bundled-packages migration
  and `alc_shapes` spec are tracked as follow-ups.

  Internally `card_analyze` reuses the existing `AppService::advice`
  execution path, so auto-install bundled fallback /
  `start_and_tick` / response warning splicing all work uniformly
  with strategy dispatch.

  **Typed output contract**: the host deserializes the pkg's
  `ctx.result` through `CardAnalyzeResult` before constructing the
  MCP response. Required fields: `pattern: String`,
  `suggested_change: String`, `confidence: f64`. Optional:
  `failure_count: u64`, `sample_count: u64`. Pkg output that fails
  to deserialize returns a typed error instead of passing freeform
  JSON to the caller. The response shape is flat —
  `{ status, result: { pattern, suggested_change, confidence, ... }, stats }`
  — not double-nested.

  Trait surface: `EngineApi::card_analyze(card_id, pkg)` is added
  with a default-impl returning a clean error
  ("not implemented by this EngineApi impl"), so external trait
  implementors are not broken (breaking for trait implementors only;
  MCP wire shape is purely additive). Issue #1778367957-70837.

## [0.33.1] - 2026-05-09

### Changed

- **BUNDLED_VERSION bump: `algocline-bundled-packages` v0.21.0 → v0.22.1**
  (`src/init.rs` `BUNDLED_SOURCES`). Picks up the bundled-packages
  doc-convention work (V1 SSoT / Projection rule, §4.1 PkgName +
  StyledName 1-line summary form, narrative SSoT decommission of
  `M.docs.narrative`, bundled adoption guide). `evalframe` source
  remains pinned at v0.3.0.

- **`docs/pkg-author-conventions.md` §4.1 — 1-line summary casing
  convention added**. PkgName-only form
  (`{PkgName} — {verb phrase}`) is retained when the pkg directory
  name is the canonical form; new PkgName + StyledName form
  (`{PkgName}({StyledName}) — {verb phrase}`) covers pkgs with a
  paper-cited abbreviation (CoT, UCB, MCTS, etc.). Provides a reading
  aid without breaking directory / SSoT consistency. The §4.5 LuaCATS
  example is updated to `cot(CoT) — ...` to demonstrate the form.

- **`docs/pkg-author-conventions.md` upgraded to V1** —
  single-source-of-truth expansion. New §0 (SSoT / Projection global
  rule) and §1 (Publish patterns and lint scope) introduce the
  Source-vs-Projection discipline and the Bundled / Community /
  Private publish-mode taxonomy, together with the
  `Required` / `Recommended` / `Optional` field status legend. The
  existing top-level pkg shape, docstring narrative, and docstring
  style sections are renumbered (§2-§4); §5 Lint rules table is
  expanded to all 11 active codes plus 2 planned V1 codes
  (`E_META_MISSING_INPUT_SHAPE`, `E_PARAM_MISSING_DESCRIBE`); §6
  Migration is split into §6.1 (legacy `M.docs.narrative` removal,
  retained from V0) and §6.2 (V1 conventions adoption). Parameter
  SSoT location is corrected from the stale `M.meta.input_shape`
  phrasing to the canonical `M.spec.entries.{entry}.input` (matches
  `extract.lua` and the `cot/init.lua` reference). All
  cross-references to `algocline-bundled-packages/docs/docstring-convention.md`
  removed; this convention doc is now the canonical specification
  (the bundled-side doc is scheduled for removal in a separate
  change). Internal issue IDs dropped from public-facing prose. The
  2 planned V1 lint rules are spec-only at this revision; their
  implementation is a follow-up change. No code changes — pure spec
  document update.

### Added

- **Bundled-packages adoption guide for the M.docs SSOT migration**
  (`docs/bundled-packages-adoption-guide.md`). Operator-facing
  playbook for migrating an entire bundled-packages repo (currently
  117 pkgs at tag v0.21.0) onto the `M.docs` spec key shipped in
  v0.33.0. Complements `docs/pkg-author-conventions.md` (the
  per-pkg authoring spec): the convention doc tells you *what* to
  write, this guide tells you *how to migrate an existing fleet*.
  Covers field-by-field reference for `M.docs.{narrative,
  schema_version}`, the B-1 / B-2 / B-3 sub-step strategy
  (mechanical declaration insert / Diátaxis section restructuring
  / docstring discipline), the `alc_pkg_doctor narrative_issues`
  bucket as an adoption progress meter, edge cases (custom
  narrative paths, partial adoption, `alc init --force` overwrite
  caveat, future schema_version expansion), and a 9-row
  cross-reference table back into algocline-core / algocline-app /
  algocline-mcp source paths so bundled maintainers do not need to
  trace the impl themselves. Backs the off-repo follow-up issue
  #1778197753.

## [0.33.0] - 2026-05-08

### Added

- **`alc_pkg_doctor` narrative_issues bucket — M.docs SSOT lint (L-1)**
  (`crates/algocline-app/src/service/pkg/doctor.rs`,
  `tests/e2e.rs`, #1778197805 L-1). New `narrative_issues` top-level
  bucket on the `alc_pkg_doctor` JSON output. Two `kind`/`severity`
  classifications:

  - `declared_missing` (severity `warn`): the pkg declares
    `M.docs.narrative = "<path>"` but the file is absent. Surfaces
    typos and forgot-to-install-narrative cases.
  - `unmigrated` (severity `info`): the pkg has no `M.docs`
    declaration but a convention `narrative.md` exists — the
    bundled adoption signal for #1778197753 (machine-readable
    progress tracking of the SSOT migration).

  Pkgs whose `M.docs.narrative` resolves to an existing file, and
  pkgs with neither declaration nor convention narrative, stay
  silent (clean state). The pass piggy-backs on
  `EngineApi::pkg_resolve_narrative_path` (introduced in #1778112139)
  so the resolution semantics are identical to the
  `alc://packages/{name}/narrative` resource consumer.

  L-2 (Diátaxis section heading lint) and L-3 (docstring 1-line
  summary lint) remain out of this release; they require markdown
  and LuaCATS parser dependencies respectively and will be
  evaluated after bundled adoption (#1778197753) signals demand.

- **`M.docs` spec key — narrative SSOT mechanism for pkg authors**
  (`crates/algocline-app/src/service/lua/gendoc/docs/{entity_schemas,pkg_info,extract}.lua`,
  `crates/algocline-app/src/service/pkg/read.rs`,
  `crates/algocline-mcp/src/resources.rs`,
  `crates/algocline-core/src/engine_api.rs`,
  `docs/pkg-author-conventions.md`,
  #1778112139). Pkg authors can now declare narrative location and
  metadata via a top-level `M.docs` table:

  ```lua
  M.docs = {
      narrative      = "narrative.md",  -- pkg-dir-relative path
      schema_version = 1,
  }
  ```

  - `EngineApi::pkg_resolve_narrative_path(name)` returns the
    declared path (or `None` for convention fallback). `..` and
    leading `/` are rejected as a path-traversal guard.
  - `alc://packages/{name}/narrative` (#1777052474) consults
    `M.docs.narrative` first; falls back to the convention path
    `<pkg>/narrative.md` for pkgs without `M.docs`.
  - gendoc `extract.lua` populates `PkgInfo.docs` from `M.docs`,
    making the SSOT visible to downstream lint / projection paths.
  - `docs/pkg-author-conventions.md` codifies the full author
    surface: `M.meta` / `M.spec` / `M.docs` / `M.run` layering,
    Diátaxis section structure for `narrative.md`, and rustdoc-flavoured
    docstring 1-line summary discipline (arXiv 2510.26130 evidence).

  Backward compatible — `M.docs` is optional, all existing pkgs
  continue to work via convention fallback. bundled-packages
  117-pkg adoption is tracked in a separate issue (#1778197753),
  and lint enforcement in `alc_pkg_doctor` is tracked in
  #1778197805.

- **`alc://packages/{name}/narrative` MCP resource — bundled-pkg
  Stdpkg reference** (`src/init.rs`,
  `crates/algocline-mcp/src/resources.rs`,
  #1777052474). New resource template that exposes per-pkg narrative
  Markdown for bundled packages. `alc init` /
  `pkg_install AUTO_INSTALL_SOURCES` now copies
  `<source>/docs/narrative/{name}.md` to
  `~/.algocline/packages/{name}/narrative.md` (silent skip when the
  source has no narrative for that pkg). `ResourceCatalog` reads
  the file directly as `text/markdown`; missing narrative returns
  -32002 ResourceNotFound.

  **Scope (re-framed from V2 cluster)**: bundled-only Stdpkg
  reference. Personal-package gendoc-output narrative is
  intentionally not surfaced via Resources — schema completion is
  already covered by `alc_pkgs.d.lua` (#1777032565), and external
  narrative indexing belongs to services like Context7. The
  companion `list_changed` notification issue (#1777052497) was
  closed at the same time because the dynamic-pkg mutation surface
  it depended on is now out of Resource scope.

- **`alc_session_new` — optional MCP-connection session pin**
  (`crates/algocline-app/src/service/session.rs` (new),
  `crates/algocline-app/src/service/mod.rs`,
  `crates/algocline-app/src/service/project.rs`,
  `crates/algocline-mcp/src/service.rs`,
  `crates/algocline-core/src/engine_api.rs`,
  #1776627475). New MCP tool that pins a `project_root` (and
  optional `mode` ∈ `"default"` | `"test"`) for the current MCP
  connection. Subsequent tool calls without an explicit
  `project_root` resolve via P > **S** > E > W (per-call > session
  pin > `ALC_PROJECT_ROOT` > cwd ancestor walk). Activation is
  optional — when no session is activated, behaviour matches
  the legacy 3-layer chain (P > E > W) exactly.

  **Why**: AI agents working across multiple worktrees were
  forgetting per-call `project_root` arguments, causing variant-
  scope `pkg_install` calls to leak into the global manifest. The
  pin makes the project context implicit at the connection level
  while preserving per-call override semantics for debug / probe
  flows.

  **Lifetime**: per-MCP-connection. For stdio MCP transport (the
  only transport algocline currently supports), one process serves
  one connection, so the pin lives on `AppService` and drops with
  the process. No explicit `alc_session_end` — closing the MCP
  connection (or activating again) replaces the pin.

  **Modes**: `"default"` matches legacy resolution. `"test"` is a
  hint for downstream tools to apply stricter isolation; the
  current release records the mode but does not yet branch on it
  (consumers like scenario test runners can opt in incrementally).

  Backward compatible — every existing tool continues to accept an
  explicit `project_root` argument, and the env var / cwd walk
  fallbacks are preserved.



## [0.32.0] - 2026-05-07

### Fixed

- **Pool dispatch / client lib tests pass on systems lacking `/bin/true`
  and `/bin/false`** (`crates/algocline-app/src/pool/dispatch.rs`,
  `crates/algocline-app/src/pool/client.rs`, #1778084517). The two
  spawn-tests now invoke `Command::new("true"|"false")` so tokio's PATH
  lookup resolves to `/usr/bin/...` on the macOS sandbox layouts where
  `/bin/{true,false}` is absent. The handshake-timeout test held its
  server-side `Inner` across the sleep so async-move actually captures
  the writer/reader; previously the server socket dropped immediately
  and the client received `ResponseParse(EOF)` rather than the expected
  `Handshake` timeout.

### Added

- **`alc_status` list path now merges pool sessions**
  (`crates/algocline-app/src/service/status.rs`,
  `tests/e2e_pool.rs`, #1778084339). Calling `alc_status` without a
  `session_id` previously enumerated only `SessionRegistry` snapshots;
  `host_mode=true` sessions (which live in `pool_registry`) were
  invisible. The list path now fetches `pool_registry.read().await`
  live entries and merges them into the returned `sessions` array
  with the same shape used by the single-session fallback (marked
  `pool: true`). `SessionRegistry` takes precedence on sid collision
  (defensive — host_mode design avoids collisions). The compact
  `{"active_sessions":0,"sessions":[]}` wire format is preserved when
  both registries are empty (snapshot regression guard).

- **`alc_status include_history=true` fetches pool conversation_history
  via IPC** (`crates/algocline-app/src/pool/protocol.rs`,
  `src/pool_worker.rs`, `crates/algocline-app/src/service/status.rs`,
  #1778084344). The pool wire protocol's `Status` request was extended
  from a unit variant to a struct with `include_history: bool` (with
  `#[serde(default)]` for legacy `{"op":"status"}` wire compatibility);
  the response carries an optional `conversation_history` field
  (`#[serde(default, skip_serializing_if = "Option::is_none")]`). When a
  caller queries `alc_status` for a paused pool session with
  `include_history: true`, the service performs a one-shot UDS
  round-trip to the worker and injects the active session's
  `conversation_history` into the response. IPC failures surface as an
  additive `history_warning` field per the Service-layer
  Error-propagation rule rather than dropping the status reply itself.
  The list path is intentionally not extended (would require N IPC
  round-trips); that is tracked separately if needed.

### Changed

- **`alc.state` per-key file dispatch — `default.json` SSoT relief**
  (`crates/algocline-engine/src/state.rs`, #1776868812). The Rust
  `JsonFileStore` backend now writes keys shaped `{prefix}:{id}` (with
  safe segments — ASCII alphanumerics + `_-.`) to per-key files at
  `{root}/{prefix}/{id}.json` instead of bundling them into the legacy
  single `{ns}.json` (typically `default.json`). The shape matches what
  `flow.state.lua` (algocline-bundled-packages) already produces — keys
  like `flow_orch:abc-123` — so no Lua-side change is required.

  Read-side falls back to the legacy `{ns}.json` for entries written
  before this change, and `incr` / `set_nx` honour both files so
  semantics stay consistent across the dispatch boundary. Existing
  `default.json` is **not migrated** — it remains a legacy reader
  fallback that gradually shadows out as keys are re-set into
  dispatched files (per-task new sessions get the per-file layout
  from the first write).

  Keys without `:` (or with unsafe characters in either segment)
  bypass dispatch and continue using the legacy single-file path.

### Added

- **`alc_hub_dist projections=["luacats"]` — per-pkg IF types projection
  (`types/alc_pkgs.d.lua`)** (#1777032565). The existing luacats branch
  emitted only `alc_shapes` *registered* shapes
  (`types/alc_shapes.d.lua`); each pkg's `M.spec.entries.run.{input, result}`
  was invisible to LuaLS / LuaCATS consumers, so `alc.run("cot", ctx)`
  did not get ctx / return-value completion. The projection now also
  emits `types/alc_pkgs.d.lua` containing one entry per pkg shape:

  - inline shapes (`T.shape({...})`) become `---@class AlcPkgInput_<pkg>`
    / `---@class AlcPkgResult_<pkg>` blocks with one `---@field` per
    schema entry.
  - ref shapes (`T.ref("voted")`) become `---@alias AlcPkgResult_<pkg>
    AlcResultVoted` (resolves into the existing alc_shapes class).

  `gen_docs.lua` writes both `types/alc_shapes.d.lua` and
  `types/alc_pkgs.d.lua` in a single `--luacats` pass. Output is
  deterministic (pkgs sorted by name) and Issue A
  (`#1777032506`, embed + distribute alc_shapes.d.lua) remains the
  prerequisite — no schema or distribution change to alc_shapes.d.lua.

  New helpers in `alc_shapes.luacats`:
  `M.gen_pkgs(pkg_specs)` (top-level driver) and `M.alias_for(class_name,
  ref_schema, class_prefix)` (per-ref alias renderer).

- **`alc.stats.llm_calls()` — Lua-side access to the auto-counted session
  LLM call total** (`crates/algocline-engine/src/bridge/data.rs`,
  `crates/algocline-core/src/metrics.rs`). The `SessionStatus.llm_calls`
  counter has been incremented automatically on every paused-cycle
  complete since 0.29.0 (alc_status v2), but Lua scripts could only
  observe it via an external `alc_status` MCP round-trip. Recipes /
  ingredients can now read it directly:

  ```lua
  local before = alc.stats.llm_calls()
  -- ... do work that may invoke alc.llm() on multiple branches ...
  local count = alc.stats.llm_calls() - before
  ```

  This obviates the `total_llm_calls = total_llm_calls + 1` pattern
  manually maintained across recipe/ingredient packages (52 occurrences
  across 7 files in algocline-bundled-packages at the time of this
  change). New `StatsHandle` (`pub use algocline_core::StatsHandle`)
  exposes a read-only handle on `Arc<Mutex<SessionStatus>>`; mirrors the
  `BudgetHandle` pattern.

  Migration of bundled recipes (`sc`, `recipe_safe_panel`,
  `recipe_deep_panel`, `orch_adaptive`, `orch_gatephase`, `orch_escalate`,
  `particle_infer`) to the new API is tracked in a separate issue and
  ships in algocline-bundled-packages.

## [0.31.2] - 2026-05-07

### Fixed

- **Test isolation: `AppConfig::default()` no longer pollutes `cwd/.algocline/`**
  (`crates/algocline-app/src/service/config.rs`,
  `crates/algocline-app/src/service/transcript.rs`). The `#[cfg(test)]`
  `Default` impl previously returned a relative `app_dir = ".algocline"`,
  which caused `AppService::new()` startup GC to write
  `crates/algocline-app/.algocline/state/pool/registry.json` whenever
  `cargo test -p algocline-app --lib` was run from the crate dir. The
  default now leaks a fresh `tempfile::tempdir()` per call, so every
  `..Default::default()` callsite (22+ in unit/transcript tests) is
  automatically rooted at an isolated tempdir. `transcript::tests::config_with_log_dir`
  also dropped its hand-rolled `AppDir::new(PathBuf::from(".algocline"))`
  in favour of `..AppConfig::default()`.

- **`alc_status` — pool sessions are now visible when queried by `session_id`**
  (`crates/algocline-app/src/service/status.rs`). Previously `alc_status`
  consulted only the in-memory `SessionRegistry`; sessions started with
  `alc_run host_mode=true` live exclusively in `pool_registry` (registry.json)
  and were reported as `session 'sid' not found (may have completed)`. The MCP
  wire returned this plain-text error and `e2e_pool::test_pool_paused_session_visible_in_status`
  panicked on JSON parse. `AppService::status()` now falls back to
  `pool_registry` on `SessionRegistry` miss and returns `{status: "needs_response",
  session_id, pool: true, pid, sock, version, created_at}` for live pool entries.
  `include_history` is ignored on the pool path (worker-side IPC fetch is
  out of scope for this fix).

- **`alc init` — Collection packages now install all sub-files** (`src/init.rs`
  `copy_package`). Previously only `init.lua` was copied for Collection-layout
  packages, leaving sibling files (`mc.lua`, `stats.lua`, `sweep.lua`,
  `eval.lua`, `search.lua`, `stop.lua`, etc.) behind and causing `alc_pkg_doctor`
  to report `incomplete_pkg` for every multi-file bundled package (e.g.
  `abm/optimize`). The fix replaces the single-file copy with a full directory
  tree copy (`copy_dir`) matching the pattern already used by the MCP
  `alc_pkg_install` path.

- **`alc_pkg_install` — `force` parameter now propagates end-to-end**
  (`crates/algocline-mcp/src/service.rs`, `crates/algocline-core/src/engine_api.rs`,
  `crates/algocline-app/src/service/pkg/install.rs`). Previously `PkgInstallParams`
  had no `force` field; the MCP wire boundary silently dropped any caller-supplied
  value. `PkgInstallParams.force: Option<bool>` is now an additive field
  (`#[serde(default)]`), wired through `EngineApi::pkg_install` and
  `AppService::pkg_install_typed` to the install path. Passing `force: true` via
  the MCP tool now overwrites an already-installed Collection package instead of
  skipping it. (breaking for trait implementors only; MCP wire shape is additive)

## [0.31.1] - 2026-05-06

### Changed

- **BUNDLED_VERSION: `algocline-bundled-packages` v0.20.0 → v0.21.0**.
  `src/init.rs` `BUNDLED_SOURCES` tag bumped. New packages picked up by
  `alc init` / `alc update`:
  - `slm_mux` (Selection, Pure Computation) — SLM-MUX confidence-based
    per-model selection + complementarity-driven K-subset selection
    (arXiv:2510.05077, ICLR 2026 Poster).
  - `solve_verify_split` (Orchestration, Pure Computation) — compute
    allocator splitting test-time budget between solver and verifier
    (arXiv:2504.01005, COLM 2025).
  - `particle_infer`, `isp_aggregate` — additional inference/aggregation
    primitives.
  - `alc_shapes.M.slm_muxed` result shape registered.
  evalframe stays at v0.3.0. MCP-side `AUTO_INSTALL_SOURCES`
  (`crates/algocline-app/src/service/resolve.rs`) is unchanged (main-branch
  pinning, separate channel from CLI bundled tag).

## [0.31.0] - 2026-05-06

### Added

- **`alc` CLI: `clap` derive + `-h` / `--help` / `-V` / `--version` support**.
  Replaces the previous hand-rolled `args.get(1) == "..."` dispatch in
  `src/main.rs` with a `#[derive(Parser)]` `Cli` struct and a `Commands` enum
  (`Init`, `Update`). Default behaviour (no subcommand) still falls through to
  MCP stdio server mode, so `alc` invoked by an MCP client (e.g. Claude Code)
  is unchanged. Existing `alc init` / `alc update` flags (`--force`, `--dev`,
  …) are forwarded verbatim via `trailing_var_arg = true` + `allow_hyphen_values
  = true`, preserving full backward compatibility. `alc -h` / `alc --help`
  now print the subcommand list and a one-paragraph note that the default
  mode is MCP server (intended to be launched by an MCP client) and that
  ad-hoc shell access to MCP tools is best done through an `agent-block`
  harness that calls the MCP server directly. `alc -V` / `alc --version`
  print `alc <version>`. Three new e2e tests
  (`test_cli_help_short`, `test_cli_help_long`, `test_cli_version`) exercise
  these dry-run code paths without spawning the stdio MCP server.

- **Pool worker POC: long-lived `alc_run` sessions that survive MCP server
  restarts** (`host_mode=true`). When `alc_run` is called with
  `host_mode: true`, the session runs inside a dedicated worker subprocess
  (`1 session = 1 process`, each with its own mlua VM) rather than inside the
  MCP server process. The worker persists independently of the MCP server
  lifecycle so that a `paused` session survives a Claude Code (MCP client)
  restart and can be resumed via `alc_continue` after reconnection. The
  default (`host_mode` absent or `false`) is fully unchanged — all existing
  `alc_run` / `alc_continue` calls continue to run in-MCP with no behaviour
  difference.

  Key components added:

  - **`PoolError` + pool wire protocol** (`crates/algocline-app/src/pool/`):
    `PoolError` (`thiserror` enum covering `Connect`, `RegistryCorrupted`,
    `Spawn`, `Handshake`, `SessionNotFound`, `VersionMismatch`),
    `PoolRequest` / `PoolResponse` serde enums (JSON line IPC), and a
    version-handshake type.

  - **`PoolClient`** (`crates/algocline-app/src/pool/client.rs`): Unix domain
    socket client. Connects to a worker's UDS path, sends length-prefixed JSON
    lines, and receives responses. Cancel-safe design: on cancellation the
    connection is dropped and re-established to avoid partial-line buffer
    corruption.

  - **`--pool-worker` hidden subcommand** (`src/pool_worker.rs` +
    `src/main.rs`): the binary can be spawned as a worker subprocess with
    `alc --pool-worker --sid <sid> --sock <path>`. The worker starts its own
    `tokio` runtime, accepts exactly one IPC client, initialises one mlua VM
    via `AsyncIsle::spawn`, and dispatches `run` / `continue` / `stop` IPC
    messages in a loop. 1 session = 1 worker process; sessions are fully
    isolated.

  - **`PoolRegistry`** (`crates/algocline-app/src/pool/registry.rs`):
    persistent session registry at `~/.algocline/state/pool/registry.json`
    (path overridable via `ALC_POOL_STATE_DIR`). Entries are written atomically
    via `tempfile::NamedTempFile::persist` (`rename(2)`). On `AppService::new`,
    `scan_and_gc` evicts orphaned entries using `libc::kill(pid, 0)` (ESRCH
    detection). An advisory lock (`fs2::FileExt::lock_exclusive`) on
    `~/.algocline/pool/registry.lock` serialises concurrent writers.

  - **`AppService` host_mode dispatch** (`crates/algocline-app/src/service/`):
    when `host_mode=true`, `AppService::run` spawns a worker subprocess via
    `tokio::process::Command::new(current_exe())` with `pre_exec(setsid)` and
    registers the session in `PoolRegistry`. `alc_continue` / `alc_continue_batch`
    auto-detect the routing (pool vs in-MCP) from `PoolRegistry::lookup`.
    Registry I/O is wrapped in `spawn_blocking` to avoid blocking the async
    executor.

  - **Worker idle timeout + SIGTERM graceful shutdown**: workers respect
    `ALC_POOL_IDLE_TIMEOUT` (default `1800` s; `0` = no timeout). After the
    timeout elapses with no IPC activity the worker removes itself from
    `registry.json` and exits. SIGTERM is caught via
    `tokio::signal::unix::signal(SignalKind::terminate())` and triggers the
    same cleanup path.

  - **MCP tools `alc_pool_ensure` / `alc_pool_status` / `alc_pool_stop`**:
    three new MCP tools for pool lifecycle management.
    - `alc_pool_ensure` (`idempotent_hint=true`): ensures a pool worker is
      running for a session, spawning one if absent. Returns `{sid, sock,
      pid, status}`.
    - `alc_pool_status` (`read_only_hint=false, idempotent_hint=true`): queries
      live registry entries; returns the list of active pool sessions with PID /
      socket path / version. (`read_only_hint` corrected from `true` — the
      implementation runs GC + `registry.json` write on every call; see Fixed.)
    - `alc_pool_stop` (`destructive_hint=true`): sends SIGTERM to a named
      session's worker and removes the registry entry. Guards against
      `pid == 0` to prevent accidental process-group signals.

  Registry corruption is propagated as `PoolError::RegistryCorrupted` (not
  silently defaulted) in accordance with the project's Result propagation
  policy.

### Fixed

- **Pool worker tracing initialisation** (`src/main.rs`): the
  `Commands::PoolWorker` arm now calls `setup_tracing(config.log_dir.as_deref())?`
  before entering `pool_worker::run`. Previously the 14 `tracing::warn!` /
  `info!` / `error!` calls inside `src/pool_worker.rs` were dead because the
  subscriber was never registered in the worker subprocess. Worker stderr now
  emits structured log lines under `RUST_LOG`.

- **`alc_run` `#[tool]` description** (`crates/algocline-mcp/src/service.rs`):
  added a sentence documenting the `host_mode` parameter — "Pass
  `host_mode: true` to route the session through the persistent worker pool
  (`1 session = 1 process`); the worker survives MCP server restarts and can
  be resumed via `alc_continue`." Previously the tool description was silent
  about `host_mode`, violating K-49 (new option must be reflected in `#[tool]`
  doc).

- **`alc_run` cache-reload failure now surfaced to MCP caller**
  (`crates/algocline-app/src/service/run.rs`): when `PoolRegistry::load_or_default`
  fails after a `run_via_pool` call, the error is now propagated as
  `pool_cache_reload_warning` in the wire response JSON (additive field, never
  omitted when a reload error occurs). Previously the `Err(e)` arm only emitted
  `tracing::warn!`, which reaches the MCP server's stderr log but is invisible
  to the MCP caller UI — a silent data loss in violation of the project's Result
  propagation policy.

- **`alc_pool_status` annotation corrected** (`crates/algocline-mcp/src/service.rs`):
  `read_only_hint` changed from `true` to `false`. The implementation calls
  `scan_and_gc` and writes `registry.json` on every invocation, making
  `read_only_hint = true` a K-111 annotation mismatch. `idempotent_hint = true`
  and `open_world_hint = false` are unchanged.

- **Pool state directory permissions tightened to 0700 / 0600**
  (`crates/algocline-app/src/pool/registry.rs`, `crates/algocline-app/src/pool/dispatch.rs`,
  `src/pool_worker.rs`): the pool state directory (`~/.algocline/state/pool/`),
  `registry.json`, `registry.lock`, and worker UDS socket files are now created
  with mode `0700` / `0600` (owner-only) instead of the umask-default `0755` /
  `0644`. This eliminates a multi-user RCE attack surface on shared hosts.
  Permission setting uses `std::os::unix::fs::PermissionsExt::set_mode` under
  `#[cfg(unix)]`; Windows builds are unaffected.

- **Worker process zombie reaping** (`crates/algocline-app/src/pool/dispatch.rs`):
  `spawn_worker` is migrated from `std::process::Command` to
  `tokio::process::Command`. The returned `tokio::process::Child` handle is
  immediately handed to `tokio::spawn(async move { let _ = child.wait().await; })`
  so that the kernel zombie entry is reaped after the worker exits. Previously
  the `Child` handle was dropped without calling `wait`, leading to zombie
  accumulation across idle-timeout / SIGTERM cycles.

- **`PoolClient::connect` handshake recv bounded to 10 s**
  (`crates/algocline-app/src/pool/client.rs`): the handshake `recv_line` call
  is now wrapped in `tokio::time::timeout(Duration::from_secs(10), …)`. A
  timeout is returned as `PoolError::Handshake("handshake recv timeout (10s)")`.
  Previously a worker that never sent its handshake response caused
  `RunningService::cancel` to hang indefinitely. The four `tests/e2e_pool.rs`
  integration tests that were `#[ignore]`d for this reason are now enabled and
  run by default.

### Changed

- **`EngineApi::run` gains `host_mode: Option<bool>` parameter** (breaking for
  trait implementors only; MCP wire shape is additive). External crates
  implementing `EngineApi` must add this parameter. All in-repo implementations
  (`AppService`, `engine_default_err!` macro, `FakeEngine`, `NotFoundEngine`)
  are updated in the same commit.

- **`RunParams` gains `host_mode: Option<bool>` field** (MCP wire shape is
  additive; existing callers that omit the field receive `host_mode=false`
  default behaviour, which is the unchanged in-MCP path).

## [0.30.0] - 2026-04-26

### Added

- **MCP Resource `alc://hub/index`**: new fixed resource that exposes the
  aggregated hub package catalog as a single `application/json` read. Merges
  cached `hub_index.json` data across all registered hub sources; individual
  source failures are surfaced in a `"warnings"` array field in the response
  JSON rather than failing the whole read (best-effort aggregate). Stale
  cache entries (>1h) and registry-load failures also surface as warnings,
  preserving partial diagnostic information across the MCP wire boundary.
  On a clean install with no cached sources the response is
  `{"schema_version":"hub_index/v0","packages":[]}`. Backed by the new
  `EngineApi::hub_index_aggregate` trait method and
  `AppService::aggregate_index` service method.

- **MCP `completion/complete` for resource template arguments**: `ServerHandler::complete`
  is now implemented in `algocline-mcp`, enabling IDE-style tab-completion of resource
  template variables (e.g. `@alc:alc://packages/<TAB>` auto-completes with installed
  package names in Claude Code). Supported argument slots:
  - `alc://packages/{name}/…` → installed package names (from `pkg_list`)
  - `alc://cards/{card_id}` → eval card IDs (from `card_list`)
  - `alc://scenarios/{name}` → scenario names (from `scenario_list`)
  - `alc://eval/{result_id}` → eval result IDs (from `eval_history`)
  - `alc://logs/{…}` → empty (no session-list API)
  All results are prefix-filtered by the typed value, capped at 100 entries
  (`has_more: true` + `total` when truncated). `ref/prompt` references return an
  empty result without error. `ServerCapabilities` now declares `completions`.
  New public API surface: `extract_template_vars` (RFC 6570 Level-1 variable
  parser) and `complete_resource_arg` on `ResourceCatalog` (for external crates
  that build on the MCP resource layer).

### Changed

- **`EngineApi::hub_index_aggregate` added** (breaking for trait implementors only;
  MCP wire shape is additive). External crates implementing `EngineApi` must add this
  method. The default return type is `Result<String, String>` (JSON string), consistent
  with all other hub trait methods.
- **Bump `rmcp` 0.16.0 → 1.5.0**. rmcp 1.x marks most model structs
  `#[non_exhaustive]`, so direct struct expressions (`ReadResourceResult`,
  `CompleteResult`, `ServerInfo`, `ReadResourceRequestParams`,
  `CallToolRequestParams`, `CompleteRequestParams`) were migrated to
  constructor / `Default` + field-mutation patterns. MCP wire shape unchanged;
  breaking only for direct rmcp consumers reading internal types in this
  workspace.

### Fixed

- **MCP Resources: error code for resource-not-found cases now returns `-32002`** (was
  `-32602 invalid_params` at 8 sites in `resources.rs`). Affected paths: segment
  mismatch in `read_types`, file-not-found in `read_types`, `pkg not found` in
  `read_packages/meta`, and wildcard arms in `read_packages`, `read_cards`,
  `read_scenarios`, `read_eval`, and `read_logs`. URI parse errors / unknown service /
  malformed query still correctly return `-32602 invalid_params`.
- **README**: corrected `alc://packages/{name}` template URI rows — the actual
  templates served are `alc://packages/{name}/init.lua` and
  `alc://packages/{name}/meta` (the bare `alc://packages/{name}` form returns
  `resource not found`).
- **`alc://hub/index` registry-load failure**: previously the registry-load
  step's `?` early-return discarded already-accumulated `warnings` entries
  (e.g. `config.toml hub.collection_url:` parse warnings). Now degrades to
  `Ok` with the failure surfaced as a warning, symmetric with per-source
  cache-corrupt handling.
- **`alc://hub/index` stale cache no longer silently empty**: `load_cached`
  now distinguishes `NotPresent` / `Stale(HubIndex)` / `Fresh(HubIndex)` /
  `Corrupt(String)`. Stale entries (>1h) still merge their data and emit a
  `hub cache stale (>3600s)` warning instead of being indistinguishable from
  a fresh-install empty result.
- **`completion/complete` for `eval_id`**: prefix filter now applied to the
  full eval history rather than after the source-side limit. Matches outside
  the newest 100 entries are now visible and `has_more` accurately reflects
  truncation.

### Improved

- **MCP Resources: `title` field populated for all fixed resources and resource templates**
  via extended `make_resource` / `make_template` helpers. Fixed resources
  (`alc://types/alc.d.lua`, `alc://types/alc_shapes.d.lua`) now carry human-readable
  titles visible in MCP client UIs. All 7 resource templates similarly include titles.

### Removed

- **Dead `chrono` direct dep + `option_env!("VERGEN_BUILD_TIMESTAMP")` path**
  in `algocline-mcp`. The `annotations.lastModified` Some-branch was
  unreachable in standard builds (no vergen wired), and an earlier note about
  populating `lastModified` from a build-time timestamp did not actually
  apply on shipped binaries. The shape always emitted `annotations: None`,
  which is preserved.

## [0.29.1] - 2026-04-26

### Fixed

- **Log directory now honors `ALC_HOME`**: `AppConfig::resolve_log_dir`
  branch 2 (`{app_dir}/logs`) is derived from the resolved `AppDir`
  instead of hard-coding `dirs::home_dir().join(".algocline").join("logs")`.
  Setting `ALC_HOME=/custom` now resolves logs under `/custom/logs`,
  matching every other Service-layer path (packages / cards / state /
  evals). Previous behavior left logs at `~/.algocline/logs` even when
  `ALC_HOME` was set — a latent inconsistency. `LogDirSource::Home`
  Display label is unchanged (`~/.algocline/logs`) since it documents
  the unset-default semantics, not the resolved path.
- This also eliminates the last `dirs::home_dir()` direct read in the
  Service layer; the only remaining HOME / `ALC_HOME` reads under
  `crates/algocline-app/src/service/` are inside `AppConfig::resolve_app_dir`
  (the documented single resolution point).

## [0.29.0] - 2026-04-26

### Added

- **`alc_status` v2 — long-running session observability**: session detail
  JSON now includes 5 additive fields for diagnosing in-flight execution
  without external instrumentation:
  - `phase` (string enum: `starting` / `running` / `llm_pending` /
    `continuing` / `completed` / `failed` / `aborted`)
  - `started_at` / `last_activity_at` / `elapsed_ms` (unix ms;
    `last_activity_at` is the primary stuck-detection signal)
  - `tokens` (cumulative `prompt_total` / `response_total` / `total`
    plus optional `current_query` snapshot during paused LLM calls)
  - `recent_logs` (per-session ring buffer, cap=20; captures both
    `alc.lua.print` and `alc.log` channels)
  - `conversation_history` (LLM query records, cap=10; opt-in via
    `StatusParams.include_history` to preserve high-frequency polling
    cost). Existing `alc_status` callers keep working — all new fields
    are additive Option types.
- New `algocline-core::recent_log` module (`LogEntry` / `LogSink`,
  `Arc<Mutex<VecDeque>>` ring buffer) and `ExecutionObserver::on_log`
  trait method. `MetricsObserver` routes log entries to the per-session
  `LogSink`; `Session` exposes the sink to the Lua bridge via
  `BridgeConfig`. (breaking for crate-internal `ExecutionMetrics::snapshot`
  callers — `include_history: bool` parameter added; MCP wire shape is
  additive.)

## [0.28.0] - 2026-04-26

### Changed

- **Service-layer error propagation**: typed `ServiceError` / `ProjectFilesError` /
  `LockError` / `TranscriptError` / `PkgListError` / `HubRegistriesError` /
  `InstalledManifestStoreError` hierarchy is now used end-to-end inside
  `crates/algocline-app/src/service/`. `String` as `E` is confined to the MCP
  wire boundary (`engine_api_impl.rs`), where typed errors are flattened via
  `.map_err(|e| e.to_string())`. `with_exclusive_lock` is now generic over
  `E: From<LockError>`, eliminating the prior `String` round-trip that fused
  Lock-acquisition failures with `alc.toml` corruption into a single variant.
  (breaking for direct `service::*` consumers; MCP wire shape is unchanged.)
- **Silent-drop sites surfaced**: best-effort enrichment / projection writes
  that previously used `let _ = ...` or `tracing::warn!`-and-drop now surface
  warnings to the caller through additive `warnings` / `transcript_warning`
  fields in MCP response JSON. Affected: `hub_info` card-store reads,
  transcript meta projection writes, `run` session-strategy mutex poison.
  CLI / library context lacks a server-framework exception catcher, so
  `tracing::warn!` alone is insufficient — both the log and the user-facing
  response now carry the diagnostic.

### Fixed

- **`pkg_read_init_lua` honours explicit `project_root`**: previously called
  `resolve_project_root(None)` unconditionally, ignoring the AppService
  context. Tests worked around this by setting `ALC_PROJECT_ROOT` env, which
  raced across parallel tests because env names are process-global. Signature
  now takes `Option<&Path>` (matching `pkg_list` / `resolve_extra_lib_paths`),
  and tests pass `tmp.path()` directly. (breaking for direct callers;
  MCP wire boundary passes `None` for backwards-compatible behaviour.)
- **Card-store test isolation**: `crates/algocline-engine/src/card.rs` test
  module previously shared a process-wide `OnceLock<FileCardStore>` keyed
  by nanosecond-timestamp `unique_pkg()`. Nanosecond collisions caused
  flaky failures in `find_where_*`, `get_by_alias_*`, etc. All 29 tests now
  use per-test `FileCardStore::new(tempdir)`.
- **Crate-wide env-var test serialisation**: `with_env_var(key, val, f)`
  helper consolidated in `crates/algocline-app/src/service/test_support.rs`
  with a single static `Mutex`. `config.rs::app_dir_env_overrides_home`
  (which races against any test calling `AppConfig::default()`) now routes
  through this shared lock instead of bare `std::env::set_var`.

## [0.27.1] - 2026-04-25

### Fixed

- `JsonFileStore`: per-namespace lock to prevent lost updates under concurrent
  `alc.state.*` calls within the same process. The previous implementation
  noted "Safe within one process" but assumed single-threaded execution and was
  vulnerable to lost updates across tokio tasks (non-locking read-modify-write).
  A `std::sync::Mutex` per namespace now serialises the load → mutate →
  atomic-rename cycle. Multi-process safety still requires a backend with native
  `INCR` (Redis) or transactions (SQLite).
- Lua standard `print` is now redirected to `tracing::info!(target = "alc.lua.print", ...)`.
  Previously, raw `print(...)` from user strategies wrote directly to process
  stdout and corrupted the rmcp JSON-RPC transport, manifesting as
  `serde error expected value at line 1 column 1` in agent-block ↔ algocline
  sessions with ~50% reproducibility.

### Changed

- Bundled-packages tag bumped from `v0.19.0` to `v0.20.0`. v0.20.0 adds new packages (`particle_infer`, `isp_aggregate`) and includes the missing `abm/{mc,stats,sweep}.lua` + `abm/frame/` + `optimize/{eval,search,stop}.lua` files that `alc_pkg_doctor` previously flagged as `incomplete_pkg`.

## [0.27.0] - 2026-04-25

### Added

- `alc_pkg_doctor`: detect `incomplete_pkg` defect when a package's `init.lua`
  requires sibling submodules (`pkg.sub`) but the corresponding `sub.lua` /
  `sub/init.lua` is missing. Static string-literal `require` only (parenthesised
  form: `require("pkg.sub")` / `require('pkg.sub')`); dynamic / non-quoted forms
  are out of scope for MVP.

- **MCP Resources capability** — `algocline-mcp` now advertises `resources`
  alongside `tools`. Service-layer read-only paths are projected as MCP
  resources under the `alc://` scheme:
  - Fixed (`resources/list`): `alc://types/alc.d.lua`, `alc://types/alc_shapes.d.lua`
  - Templates (`resources/templates/list`): `alc://packages/{name}/init.lua`,
    `alc://packages/{name}/meta`, `alc://cards/{card_id}`,
    `alc://cards/{card_id}/samples`, `alc://scenarios/{name}`,
    `alc://eval/{result_id}`, `alc://logs/{session_id}`
  - Pagination via query string for samples/logs (`?offset=N&limit=M`).
  - MIME: `.lua` → `text/x-lua`, JSON → `application/json`.
  - See `docs/mcp-resources.md` for the full catalog and `@alc:<uri>` mention examples.

  Out of scope for this release (candidates for follow-up):
  - `alc://hub/index` — pending canonical AppDir path for `hub_reindex` default output.
  - `alc://packages/{name}/narrative` — `hub_gendoc` currently emits to an external `out_dir`.
  - `list_changed` notifications and `resources/subscribe` — static capability only in V1.

### Fixed

- **MCP Resources — pagination caps**: `?limit=` and `?offset=` on
  `alc://cards/{id}/samples` and `alc://logs/{session_id}` are now capped
  (10,000 / 10,000,000 / 10,000 / 1,000,000 respectively). Values above the
  cap return `invalid_params`. Prevents MCP-DoS via unbounded allocation.
- **MCP Resources — ID reserved-char rejection**: card_id, session_id, eval
  result_id, package name, and scenario name are now validated at the MCP
  boundary. URI-reserved characters (`& = ? / % SPACE`) return `invalid_params`.
- **MCP Resources — eval ID strict validation**: `alc://eval/{result_id}` now
  validates `^[A-Za-z0-9-]+_\d+$` instead of a bare `contains('_')` check.
- **MCP Resources — `read_types` explicit match**: only `alc.d.lua` and
  `alc_shapes.d.lua` are accepted as file names; any multi-segment or
  unrecognized path returns `invalid_params`.
- **MCP Resources — `parse_query` `=` in value**: removed erroneous rejection
  of query values containing `=` (e.g. `?key=a=b` is now accepted; `split_once`
  already splits on the first `=` only).
- **`pkg_read_init_lua` — malformed `alc.local.toml` propagation**: a corrupt
  local TOML file now returns `Err` (surfaced to MCP caller) instead of being
  silently swallowed via `tracing::warn!`. (CLAUDE.md §Service 層の Error 伝播規律)
- **MCP Resources — `pkg_meta` trait method**: extracted `EngineApi::pkg_meta`
  to avoid a JSON round-trip in the MCP layer; `read_packages` meta arm now
  delegates to the new method. (breaking for `EngineApi` implementors only)
- **MCP Resources — tempdir leak in test helper**: `make_fake_catalog` now
  returns `(ResourceCatalog, TempDir)` instead of leaking via `mem::forget`.
- **MCP Resources — 4 missing E2E smoke tests**: added
  `test_mcp_resource_read_card`, `test_mcp_resource_read_card_samples_pagination`,
  `test_mcp_resource_read_eval_detail`, and `test_mcp_resource_read_logs_pagination`.

### Changed

- `EngineApi` trait: added `pkg_read_init_lua(name)` (breaking for `EngineApi`
  implementors only; MCP wire shape is additive).
- `AlcService::new` now takes `Arc<AppDir>` as a second argument (additive
  constructor change for library embedders; MCP wire shape unchanged).
  `AppConfig::app_dir()` accessor can be used to obtain the handle before
  casting `Arc<AppService>` to `Arc<dyn EngineApi>`.
- `EngineApi` trait: added `pkg_meta(name)` method so the MCP `read_packages`
  meta arm avoids a JSON round-trip via `pkg_list` (breaking for `EngineApi`
  implementors only; MCP wire shape is additive).

### Internal

- Extracted `engine_default_err!` macro in `algocline-mcp` test module to
  deduplicate `EngineApi` test stub boilerplate (45-method trait — eliminates
  per-stub edits when adding new methods). Applied to `NoopEngine`;
  `FakeEngine` builder and `NotFoundEngine` overrides remain handwritten.

## [0.26.3] - 2026-04-24

### Changed

- `alc init` now also distributes `alc_shapes.d.lua` to `~/.algocline/types/` alongside `alc.d.lua`, enabling zero-config LuaCats completion for `alc_shapes` combinators (T.string, T.number, ...) and registered shapes. `pkg_install` response adds a new `alc_shapes_types_path` field (additive; existing `types_path` unchanged).

## [0.26.2] - 2026-04-24

### Fixed

- **`alc_pkg_install` no longer aborts on pre-existing `pkg_link` dev
  symlinks.** In collection install mode, any entry whose destination
  at `~/.algocline/packages/<name>` already points outside the packages
  base (the canonical `pkg_link` case) previously produced a hard
  `Path '<name>' escapes base directory` error from `ContainedPath::child`
  and aborted the entire batch. The symlink is now detected via
  `symlink_metadata()` before the containment check, routed to a new
  `skipped_symlinks: [<name>, ...]` field in the response JSON, emitted
  as a `tracing::warn!` line, and the installer continues with the
  remaining pkgs. Operators run `alc_pkg_unlink <name>` if they want
  the git-clone copy to replace the dev link.

## [0.26.1] - 2026-04-24

### Changed

- **BUNDLED_VERSION bump**: `BUNDLED_SOURCES` for
  `algocline-bundled-packages` moved from `v0.18.0` → `v0.19.0`.
  `v0.19.0` is the first bundled release that tracks the v0.26 core
  migration — Lua `context7_config.lua` / `devin_wiki_config.lua`
  retired, project-specific content migrated to `alc.toml`
  `[hub] / [hub.context7] / [hub.devin]` sections, and the `just dist`
  pipeline streamlined to a single `alc_hub_dist` MCP call without
  `config_path`. `alc init` now pulls the v0.26-ready bundled repo.

## [0.26.0] - 2026-04-24

### Added

- **`luacats` projection for `alc_hub_dist` / `alc_hub_gendoc`**: emits
  `source_dir/types/alc_shapes.d.lua` from the embedded alc_shapes SSoT
  via `S.LuaCats.gen`. This unifies bundled-side `just gen-shapes`
  behavior into core so any third-party package author can regenerate
  lua-ls type definitions in a single `alc_hub_dist` call. Opt-in:
  caller must include `"luacats"` in `projections`.
- **pkg compat contract (`M.meta.alc_shapes_compat`)**: packages can declare
  a semver range of alc_shapes versions they are known to work with
  (e.g. `alc_shapes_compat = ">=0.25.0, <0.26"`). On load the
  `alc_hub_dist` / `alc_hub_gendoc` path reads the declared range and
  rejects out-of-range loads with a typed `ShapesCompatViolation
  { pkg_name, declared_range, actual_version, hint }` error surfaced via
  the MCP wire response. Malformed ranges return `ShapesCompatMalformed`
  with the `semver` parse error. Undeclared packages continue loading
  with a warning (backward compat; the bundled pkg set will be migrated
  in a separate commit in the same v0.26 release window).
  Drift is resolved **declaratively**: core vendors alc_shapes (v0.25.1
  refactor `f9d6260`), and pkg authors declare their supported range.
  No runtime `allow_mirror` / source_dir-mirror preload path is
  introduced — a pkg author who needs a patched alc_shapes is expected
  to fork core or upstream the patch rather than run a parallel mirror.
- **`alc_pkg_scaffold` MCP tool**: generates a minimal package skeleton
  at `<target_dir>/<name>/init.lua` with an `M.meta` / `M.spec.entries.run`
  / `M.run` template and a pre-filled `alc_shapes_compat` range derived
  from the embedded alc_shapes version (e.g. current core 0.25.1 →
  `">=0.25.0, <0.26"`). Optional `category` / `description` params flow
  into `M.meta`. Typed errors on name validation failure
  (`PkgScaffoldError::NameInvalid`) or pre-existing init.lua
  (`AlreadyExists`) are surfaced via the MCP wire response. Companions
  the v0.26 pkg compat contract (commit `bce529a`): scaffolded packages
  are ecosystem-ready the moment they're created.
- **`narrative` and `llms` projections for `alc_hub_dist` / `alc_hub_gendoc`**:
  these projections were previously rejected with `unknown projection` even
  though the embedded `gen_docs.lua` already emits `docs/narrative/{pkg}.md`,
  `docs/llms.txt`, and `docs/llms-full.txt` unconditionally when
  `lint_only=false`. The Rust-side allowlist now accepts `"narrative"` and
  `"llms"` as valid `projections` values. No `gen_docs.lua` argv change is
  needed (approach A): the files are produced whenever any non-lint-only
  projection set is requested, regardless of whether `narrative`/`llms` appear
  in the list. The MCP doc string for `alc_hub_gendoc` / `alc_hub_dist` is
  also updated to list all accepted projection values, including the previously
  undocumented `luacats`.

### Changed — **BREAKING**: `alc_hub_gendoc` `config_path` retires `.lua`; new `alc.toml [hub.*]` sections

- `config_path` no longer accepts `.lua` files.  Passing a `.lua` path returns
  `gendoc: config_path extension '.lua' is no longer supported; use .toml`.
- When `projections` includes `"context7"` or `"devin"` and `config_path` is
  omitted, the project root's `alc.toml` is auto-explored.  New subsections
  `[hub]`, `[hub.context7]`, `[hub.devin]` are recognized.  See
  `docs/hub-gendoc-config.md` for the full schema.
- C7 / Devin rules and repo notes now ship as core defaults embedded in
  `algocline`.  User config may append (`extra_rules` / `extra_repo_notes`),
  override wholesale (`rules_override` / `repo_notes_override`), or read from
  an external file (`rules_file` / `repo_notes_file`).  `rules_file` and
  `rules_override` are mutually exclusive (typed error on both set).
- Migration: move `context7_config.lua` / `devin_wiki_config.lua` contents
  into the repo's `alc.toml` under the new subsections.  Callers using an
  explicit flat `config_path=*.toml` with `[context7]` / `[devin]` sections
  continue to work unchanged.

### Fixed

- **`alc_hub_gendoc` / `alc_hub_dist` context7 projection: `projectTitle` was
  always `null`.**  `HubProjectionConfig::to_context7_toml` now populates
  `projectTitle` from the resolved name chain
  (`[hub.<proj>].name > [hub].name > hub_index repo basename >
  DEFAULT_NAME_FALLBACK`) and `description` from the resolved description
  chain (`[hub.context7].description > [hub].description >
  DEFAULT_C7_DESCRIPTION`).  The precedence chains were already computed;
  the output keys were simply never written.  Devin identity fields
  (`project_name`, `description`) are not emitted: the DeepWiki schema
  (<https://docs.devin.ai/work-with-devin/deepwiki>) only documents
  `repo_notes` and `pages` as recognised top-level keys.

### Changed

## [0.25.1] - 2026-04-23

### Changed

- **Embedded `hub_gendoc` / `alc_hub_dist`: `alc_shapes` Lua library now fully
  vendored.** `algocline-app` now embeds the full `alc_shapes` module set
  (`init.lua` / `t.lua` / `reflect.lua` / `check.lua` / `instrument.lua` /
  `luacats.lua` / `spec_resolver.lua`) via `include_str!`, eliminating the
  disk fallback path in `register_preloads`. The `source_dir`'s on-disk
  `alc_shapes/` directory is no longer consulted at runtime. Non-breaking:
  MCP wire shape unchanged; `source_dir` is still accepted. Any third-party
  package author can now invoke `alc_hub_dist` against their own source tree
  without vendoring `alc_shapes` themselves.
- **`alc_shapes` version pinning**: the embedded `alc_shapes` now declares
  `M.VERSION` and the `hub_gendoc` / `alc_hub_dist` resolver reads it from
  any on-disk mirror at `source_dir/alc_shapes/init.lua`. When the mirror's
  `M.VERSION` differs from the embedded constant, dist fails with a typed
  `ShapesVersionMismatch { embedded, mirror, hint }` error surfaced via the
  MCP wire response — catches drift between core's vendored copy and a
  downstream repository's mirror at the earliest possible point.

### Removed

- `gendoc_bundled_smoke.rs` — the `#[ignore]`d bundled-packages parity smoke
  test is retired in favour of a plain rmcp-based E2E
  (`test_alc_hub_dist_fixture` in `tests/e2e.rs`) that runs against an in-repo
  fixture (`tests/fixtures/hub_dist_sample/`) and is CI-default.

### Fixed

- **Embedded `hub_gendoc` parity with standalone Lua**: `register_preloads`
  now accepts `source_dir` and loads `alc_shapes` / `alc_shapes.t` from disk
  when available (`{source_dir}/alc_shapes/init.lua`), falling back to
  in-binary stubs for fixture-only runs. Also adds a disk-backed preload test
  (`T.ref("voted")` via `tools.docs.projections`) and hardens
  `T.discriminated` by copying `variants` before schema construction.
- **Embedded `hub_gendoc` / `alc_hub_dist`**: extend the in-process
  `alc_shapes` / `alc_shapes.t` stubs so bundled package `init.lua`
  modules load under `gen_docs` (real trees use `local T = S.T`,
  `S.instrument`, `:describe`, and `T.boolean` / `T.table`). Fixes
  `dist` / `gendoc` failures against full `algocline-bundled-packages`
  that were unrelated to hub TOML config. Also fixes Lua closure
  ordering so `describe` captures `make_schema` as a local.

## [0.25.0] - 2026-04-23

### Added

- `alc_hub_gendoc` MCP tool — generate human-readable docs (narrative /
  hub / llms / context7 / devin projections) from a `hub_index.json`.
  Embeds `gen_docs.lua` + docs modules from algocline-bundled-packages so
  downstream hub repos no longer need to vendor the generator. Bundled-
  specific `context7_config.lua` / `devin_wiki_config.lua` are injected
  via the `config_path` argument.
- `alc_hub_dist` MCP tool — facade that runs `alc_hub_reindex` followed
  by `alc_hub_gendoc` in sequence, returning a composed
  `{ reindex, gendoc, preset_catalog_version, preset? }` response. Optional `preset` expands into
  primitive `alc_hub_gendoc` arguments (builtin `Current` recipes +
  optional `alc.toml` overrides under `[hub.dist.presets.<name>]`).
  Successful responses always include `preset_catalog_version`; when
  `preset` is used, an additional `preset` object includes the resolved
  primitive args for observability. `alc_info` also reports
  `preset_catalog_version`.
  Fails fast on reindex error; on gendoc failure the error text embeds
  the (already-succeeded) reindex JSON so the caller sees both outcomes.

### Changed

- `alc init` bundled-packages tag bumped `v0.17.0` → `v0.18.0`
  - `evalframe` stays at `v0.3.0`
  - v0.18.0 ships `hub_index.json` reindexed against the new typed
    `PackageSource` wire shape (see next entry). Older v0.17.0 still loads
    via the read-compat shim.

### Changed — `source` field wire format

The `source` field in `alc_hub_info` / `alc_hub_search` responses and in the
on-disk `~/.algocline/hub_index.json` / `~/.algocline/installed.json` files
changes from a plain string to a tagged object. `alc_info` / `alc_pkg_list`
outputs follow suit.

**Before (0.24.x)**

```json
"source": ""
"source": "installed"
"source": "bundled"
"source": "~/pkgs/foo"
"source": "https://github.com/owner/repo"
```

**After (0.25.0)**

```json
"source": {"type": "unknown"}
"source": {"type": "installed"}
"source": {"type": "bundled", "collection": null}
"source": {"type": "path", "path": "~/pkgs/foo"}
"source": {"type": "git",  "url":  "https://github.com/owner/repo", "rev": null}
```

Variants: `unknown` / `installed` / `bundled {collection}` / `path {path}` /
`git {url, rev}`.

### Migration

**Existing users: mostly no-op.**
`~/.algocline/installed.json` and `~/.algocline/hub_index.json` stored in the
legacy string form still load in 0.25.0 (a read-compat shim absorbs them).
Each entry is rewritten to the tagged form the next time `alc_pkg_install` /
`alc_pkg_remove` / `alc_hub_reindex` touches it. Legacy `""` entries now
surface in `alc_pkg_repair` as `unrepairable` with
`reason: "source unknown (legacy entry; run alc_hub_reindex)"`; run
`alc_hub_reindex` once to clear them.

**MCP consumers that read `source` from `alc_hub_info` / `alc_hub_search`:
update required.** Replace the string-typed branch with an object dispatch:

```lua
local src = response.pkg.source  -- table in 0.25.0+
if     src.type == "git"      then return src.url
elseif src.type == "path"     then return src.path
elseif src.type == "bundled"  then return src.collection   -- nil or string
else                                return nil             -- installed / unknown
end
```

For dual-compat with 0.24.x, append `elseif type(src) == "string" then ...`.

**Bundled packages repo:** `algocline-bundled-packages/hub_index.json` will
ship already reindexed in v0.18.0. Running 0.25.0 against v0.17.0 still
works through the read shim.

## [0.24.1] - 2026-04-19

### Changed

- `alc init` bundled-packages tag bumped `v0.16.0` → `v0.17.0`
  - `evalframe` stays at `v0.3.0`

## [0.24.0] - 2026-04-19

### Changed

- `alc init` bundled-packages tag bumped `v0.15.0` → `v0.16.0`
  - `evalframe` stays at `v0.3.0`

### Added

- **`alc_pkg_remove` `scope` parameter**: reintroduced with manifest-only
  semantics. Accepts `"project"` (default, existing behavior — removes
  from `alc.toml` + `alc.lock`), `"global"` (removes the entry from
  `~/.algocline/installed.json`), or `"all"` (both, lenient — succeeds
  if either scope had the entry).
  - Physical files in `~/.algocline/packages/{name}/` are **never**
    deleted by any scope. This is different from the historical
    0.14.0 `scope="global"` (removed in 0.15.0), which deleted the
    cache directory and was retired for safety. The name is reused
    because the new semantics are safe (manifest edit only) and
    mirror `PkgLinkScope`'s existing `scope` pattern.
  - Closes the "orphan `installed.json` entry has no tool cleanup
    path" gap that forced manual JSON edits (e.g. after e2e test
    source tempdirs vanish, leaving dead manifest entries).
- **`PkgRemoveScope` enum** in `algocline-mcp`: snake_case-serialized
  `project` / `global` / `all`. Unknown values are rejected by the
  schema rather than silently defaulted.
- **`EngineApi::pkg_remove`**: gains a `scope: Option<String>`
  parameter (breaking for trait implementors; the in-tree `AppService`
  impl is updated accordingly).

## [0.23.1] - 2026-04-19

### Changed

- `alc init` bundled-packages tag bumped `v0.14.0` → `v0.15.0`
  - 106 packages available (adds SoT docs pipeline + `alc_shapes` DSL extensions on bundled side; algocline core is unchanged — `hub_index.json` wire format `hub_index/v0` and flat `M.meta` structure are compatible)

## [0.23.0] - 2026-04-19

### Added

- `alc_status`: new `pending_filter` parameter for field-level projection of paused `LlmQuery`
  - String preset: `"meta"` (no prompt), `"preview"` (truncated prompt, default `ALC_PROMPT_PREVIEW_CHARS=200`), `"full"` (full prompt)
  - Custom object: `{ include_prompt: bool, prompt_max_chars: usize, include_batch_items: bool, include_opts: bool, include_payload: bool }`
  - UTF-8 safe truncation via `chars().take(N)` (never splits multi-byte sequences)
  - Unknown preset → typed error (no silent fallback)
  - Omit / `null` → legacy shape (full query fidelity)
- `ALC_PROMPT_PREVIEW_CHARS` env var: configure the preview char cap (default 200). Setting `0` yields empty previews — for no prompt at all, use the `"meta"` preset instead
- `algocline-engine`: public `PendingFilter`, `PromptProjection`, and `DEFAULT_PROMPT_PREVIEW_CHARS` for downstream host integration
- E2E coverage (`tests/e2e.rs`): 6 new rmcp-based tests — preset meta / preview+full / unknown preset error / bad-shape error / custom object filter / paused session projection
- `alc_pkg_list` / `alc_hub_search`: unified list-tool knobs (`limit` / `sort` / `filter` / `fields` / `verbose`)
  - `verbose = "summary"` (default) | `"full"`; `fields` > `verbose` when both set
  - `sort`: `"-key,key2"` DSL; default `pkg_list="-active,-installed_at"`, `hub_search="-installed,name"`
  - `filter`: key-value exact match; when `filter` and legacy `category` / `installed_only` conflict on the same key, **`filter` wins** (explicit parameter priority). `hub_search`'s `category` / `installed_only` fold into the `filter` map only when the key is not already set
  - `limit`: `null` → tool default (50); `0` → **no limit / return all** (empty-means-all idiom); negative → clamped to 0 (also "no limit")
- `hub_search`: `SearchResult.docstring_matched: Option<bool>` — hit source flag (true only when the query hit docstring alone)
- Size regression guard: default summary output capped < 15KB (50 entries × ~224 chars/entry ≈ 11 KB realistic ceiling, ~34% headroom)

### Changed

- `EngineApi::status` trait gains `pending_filter: Option<serde_json::Value>` (breaking for trait implementors only; MCP wire shape is additive)
- `Session::snapshot` / `SessionRegistry::list_snapshots`: signatures gain `Option<&PendingFilter>` for server-side projection
- `alc_pkg_list` summary preset fields (verbatim): `name`, `scope`, `version`, `active`, `resolved_source_path`, `resolved_source_kind`
- `alc_pkg_list` full preset fields (verbatim): summary + `install_source`, `installed_at`, `updated_at`, `override_paths`, `overrides`, `linked`, `link_target`, `broken`, `path`, `source`, `source_type`, `meta`, `error`
- `alc_hub_search` summary preset fields (verbatim): `name`, `version`, `description`, `category`, `installed`, `docstring_matched`
- `alc_hub_search` full preset fields (verbatim): summary + `source`, `card_count`, `best_card`, `docstring`
- `hub_search`: `SearchResult.docstring` is now `skip_serializing` (internal signal for matching; projected only when the field set includes `"docstring"` — via `fields=["docstring"]` or `verbose="full"`)

### Preset drift policy

- Add field to preset = **minor** version bump
- Remove/rename field = **major** version bump
- Both must list the full preset verbatim in the CHANGELOG `Changed` section

### Fixed

- `alc_pkg_list` / `alc_hub_search`: large 1-line JSON output (63K–68K chars) no longer exceeds Claude Code context limits at default `verbose="summary"`

## [0.22.0] - 2026-04-18

### Added — Card sink fan-out (`ALC_CARD_SINKS`) + backfill

Card mutations (`create` / `append` / `write_samples` / alias writes /
`import_from_dir`) now fan out to every subscriber URI listed in the
`ALC_CARD_SINKS` env var (pipe-separated). The primary store remains
authoritative; subscriber failures are isolated — primary writes always
succeed, and per-subscriber health is exposed via `alc_stats` under the
new `card_sinks` field (`ok` / `err` counters keyed by event kind +
`last_error`).

Registered sinks can be backfilled with the new tool:

```lua
local r = alc.card.sink_backfill({
  sink    = "file:///tmp/mirror",
  dry_run = false,
})
-- r = { sink, pushed = {...}, skipped = {...}, failed = {...}, pushed_samples = {...} }
```

`alc_card_sink_backfill` pushes every Card from the primary store to one
registered subscriber URI. Drift-safe: cards already present on the sink
are skipped (never overwritten). Bypasses bus fan-out so in-sync peers
see no duplicate Created events. Supports `dry_run` for preview.

Tool annotations: `destructive_hint = false`, `idempotent_hint = true`,
`open_world_hint = false`.

### Added — `SubscriberStats` exposed via `alc_stats`

`alc_stats` now includes a `card_sinks` array with one row per
registered subscriber:

```json
"card_sinks": [
  {
    "sink": "file:///tmp/mirror",
    "ok":  { "created": 12, "appended": 3, "samples": 5, "aliases": 1 },
    "err": { "created": 0,  "appended": 0, "samples": 0, "aliases": 0 },
    "last_error": null
  }
]
```

All four event-kind keys are always present (may be 0). `last_error`
populates the latest failing publish with `{ kind, msg, ts_ms }`.

## [0.21.0] - 2026-04-17

### Changed

- `alc_pkg_repair`: LocalPath sources whose source directory is missing, or exists but has no `init.lua` at root, are now classified as `unrepairable` (`kind: "installed_missing"`) with an actionable `reason` / `suggestion`, instead of landing in `failed` with the misleading "'name' parameter is only supported for single-package dirs" message. Matches the policy already used for Bundled / Path sources: states detectable before attempting install belong in `unrepairable`; `failed` is reserved for runtime errors during an actual install attempt.
- `alc_pkg_install` (local path): now rejects a missing source directory up front with a clear `"Source directory does not exist: {path}"` error instead of falling through to the collection-mode branch and producing the misleading `'name' parameter` error.
- `alc init`: bundled `algocline-bundled-packages` tag bumped `v0.12.0` → `v0.14.0`. `evalframe` stays at `v0.3.0`.

## [0.20.1] - 2026-04-16

### Added

- `alc_pkg_list`: entries now include `resolved_source_path` (canonical absolute dir), `resolved_source_kind` (installed/linked/local_path/bundled), and `override_paths` (shadowed same-name pkg paths) for LLM agent source access.

### Fixed

- `alc_pkg_list`: project `installed` / `git` / `bundled` entries no longer list their own backing directory (`packages_dir()/{name}`) as a `override_paths` self-shadow. Only genuinely distinct same-name packages (e.g. a project `path` vendor dir overriding a global install) now appear in `override_paths`.

### Changed

- `alc_pkg_list` (internal): meta merge ordering tightened so every host-authoritative field (`error`, `linked`, `link_target`, `broken`, …) is uniformly protected from Lua `meta.*` clobbering. Output JSON shape is unchanged for conforming packages; packages whose `meta` illicitly shadowed these names now correctly return the host value.
- `alc_pkg_list` (internal): `resolved_source_kind` is now a typed enum internally (`Installed`/`Linked`/`LocalPath`/`Bundled`); wire format is identical (snake_case strings).

## [0.20.0] - 2026-04-13

### Added — `alc_card_samples` / `alc.card.read_samples` gains `where`

The per-case sidecar reader now accepts the same nested-object `where`
DSL as `alc_card_find`, evaluated against each JSONL row. `offset` is
applied after filtering (Prisma/SQL convention), so paging the matched
subset is predictable.

```lua
local failures = alc.card.read_samples(card_id, {
  where  = { passed = false, score = { lt = 0.5 } },
  offset = 0,
  limit  = 20,
})
```

Pure addition — calls without `where` keep previous semantics.

### Added — `alc_card_lineage` / `alc.card.lineage`

New lineage walker that traverses Card ancestry/descendants via the
`metadata.prior_card_id` convention. Directions: `"up"` (ancestors,
default), `"down"` (descendants), `"both"`. Optional `depth` cap,
`include_stats`, and `relation_filter` for following only edges with
specific `prior_relation` values. Returns `{ root, nodes, edges,
truncated }` where `nodes[*].depth` is signed (0 root, negative
ancestor, positive descendant).

Also documents `[strategy_params]` and `metadata.prior_card_id` /
`metadata.prior_relation` as recognized Card schema conventions.

### Changed — **BREAKING**: `alc_card_find` / `alc.card.find` DSL

Two breaking changes, plus one additive field. 0.x allows breaks —
migrate callers before upgrading.

**1. Filter fields → `where` object (Prisma-style)**

All ad-hoc filter fields (`scenario`, `model`, `min_pass_rate`) are
removed. Use a nested `where` object that walks Card sections, with
implicit equality on scalars and reserved operators on leaf objects.

```lua
-- Before
alc.card.find({ pkg = "foo", scenario = "bar", min_pass_rate = 0.5 })

-- After
alc.card.find({
    pkg = "foo",
    where = {
        scenario = { name = "bar" },
        stats = { pass_rate = { gte = 0.5 } },
    },
})
```

Operators: `eq ne lt lte gt gte in nin exists contains starts_with`,
plus logical `_and` / `_or` / `_not` at any level.

**2. `sort` → `order_by` (dotted path, descending prefix, multi-key)**

`sort = "pass_rate"` is replaced with `order_by` — a dotted-path string
(`"stats.pass_rate"`), a `-` prefix for descending, or an array for
multi-key sort with tiebreakers.

```lua
-- Before
sort = "pass_rate"

-- After
order_by = "-stats.pass_rate"
-- or
order_by = { "-stats.pass_rate", "created_at" }
```

**Added**: `offset` for pagination (pure addition, non-breaking).

Missing-field semantics: `eq`/`lt`/etc. evaluate false on missing
fields; `ne`/`nin` evaluate true; `exists: false` matches only when
the field is absent.

See `docs/lua-stdlib.md#alccardfind` for the full reference.

## [0.19.0] - 2026-04-13

### Added

- **`alc_hub_info`**: Show detailed information for a single package — metadata, all Cards, aliases, and aggregated stats (card count, eval count, best pass rate). Looks up remote indices first, falls back to local `init.lua` parse.
- **`collection_url` support**: New `[hub].collection_url` in `~/.algocline/config.toml` adds a Tier 0 aggregated index URL, fetched before per-source registries.

### Fixed

- **Path traversal guard** in `hub_info`: reject package names containing `..`, `/`, or `\`.
- **Duplicate `card::list` call** in `hub_info`: reuse a single call for both JSON output and stats.
- **`count_evals_for_pkg` ordering**: two-pass collection eliminates `read_dir` iteration-order dependency.

### Changed

- Enriched module-level RustDoc for `card.rs` (Card schema, design principles, storage layout) and `hub.rs` (staged design, index schema, 4-tier discovery, caching).

## [0.18.0] - 2026-04-12

### Added — Hub: Package Discovery & Search

Registry-based remote index discovery with per-source caching.

- **`alc_hub_search`**: Search packages across remote Hub indices + local install state. Index URLs are auto-discovered from hub registries (populated by `pkg_install` / `card_install`), the installed-packages manifest, and bundled seeds. Results include `installed: true/false`, descriptions, categories, and source URLs.
- **`alc_hub_reindex`**: Generate a hub index from a packages directory. New `source_dir` parameter enables pure metadata extraction from a repo checkout (no manifest or card data mixed in) for CI publishing.
- **Hub registries** (`~/.algocline/hub_registries.json`): Persistent registry of source URLs, auto-populated on `pkg_install` and `card_install`. Atomic writes via tempfile + rename.
- **Per-source cache** (`~/.algocline/hub_cache/{hash}.json`): Each remote index cached independently with 1-hour TTL using FNV-1a URL hashing.

### Changed

- Bump `algocline-bundled-packages` to v0.11.2 (adds `hub_index.json`)

## [0.17.1] - 2026-04-12

### Changed

- Bump `algocline-bundled-packages` from v0.11.0 to v0.11.1
  (Optimizer Card support)

## [0.17.0] - 2026-04-12

### Added — `alc.eval()` Lua function

Evalframe facade exposed as a first-class Lua function in prelude.
Accepts string scenario names or inline tables, wires the algocline
provider automatically, and optionally emits a Card on completion.

### Changed — `alc_eval` MCP tool delegates to `alc.eval()`

The MCP `alc_eval` tool now delegates to the prelude `alc.eval()`
function instead of hand-building evalframe Lua code. Card emission
is handled Lua-side, removing Rust-side `maybe_save_card`.
`eval_compare` shares the `STD_SHIM` constant with `eval`.

### Added — Card schema v0 (frozen)

Immutable run-result snapshots stored as TOML under
`~/.algocline/cards/{pkg}/{card_id}.toml`. The full v0 surface is now
considered frozen — future additions land behind a `v1` schema bump.

**v0 schema**:
- REQUIRED: `schema_version`, `card_id`, `created_at`, `pkg.name`
- Everything else is OPTIONAL and auto-injected when derivable
- `card_id` format: `{pkg}_{model_short}_{YYYYMMDDTHHMMSS}_{hash6}`
- Low-hex `hash6` (DJB2 last 6 chars) to avoid top-bit collisions
- `param_fingerprint` auto-computed from `[params]` when present

**Lua API (`alc.card.*`)**:
- `create(table)` — write a new Card (immutable)
- `get(card_id)` / `get_by_alias(name)` — fetch full Card
- `list(filter?)` / `find(query?)` — summaries with sort / filter
- `append(card_id, fields)` — additive-only annotation
- `alias_set(name, card_id, opts?)` / `alias_list(filter?)` — mutable aliases
- `write_samples(card_id, samples)` / `read_samples(card_id, opts?)` —
  write-once per-case JSONL sidecar

**MCP tools (host-side read surface)**:
- `alc_card_list` / `alc_card_get` / `alc_card_find`
- `alc_card_alias_list` / `alc_card_alias_set` / `alc_card_get_by_alias`
- `alc_card_append`
- `alc_card_samples` (per-case sidecar read with `offset` / `limit` paging)

**`alc_eval` integration**: Opt-in `auto_card=true` emits a Card from
the eval result on completion, and when per-case rows are present
dumps them to a `{card_id}.samples.jsonl` sidecar.

**Examples**: `examples/cards/prompt_ab_demo.lua` — a self-contained
6-trial prompt sweep exercising create / find / alias_set / append
end-to-end with no LLM calls.

## [0.15.1] - 2026-04-09

### Added

- **mlua-mathlib v0.3.0**: Upgraded from v0.2. Adds 22 new `alc.math` functions:
  - Hypothesis testing: `welch_t_test`, `mann_whitney_u`, `chi_squared_test`, `ks_test`
  - Ranking & IR metrics: `rank`, `spearman_correlation`, `kendall_tau`, `ndcg`, `mrr`
  - Information theory: `entropy`, `kl_divergence`, `js_divergence`, `cross_entropy`
  - Special functions: `logsumexp`, `logit`, `expit`
  - Time series: `moving_average`, `ewma`, `autocorrelation`
  - Combinatorics: `permutations`
  - RNG: `shuffle`, `sample_with_replacement`

## [0.15.0] - 2026-04-09

### Added

- **`alc_init` MCP tool**: Initialize project — creates `alc.toml` in the project root if absent. Equivalent to `alc init` for project-scoped setup via MCP
- **`alc_update` MCP tool**: Update installed packages declared in `alc.toml` — re-installs each entry from its recorded source URL and updates `alc.lock`
- **`alc_migrate` MCP tool**: Migrate legacy `alc.lock` (v1 `local_dir` entries) to the new `alc.toml` + `alc.lock` schema. Generates `alc.toml` from existing lock entries and rewrites `alc.lock` to the new format
- **`alc_pkg_unlink` MCP tool**: Remove a symlink created by `alc_pkg_link`. Rejects real directories (only symlinks are removed) to prevent accidental deletion of installed packages
- **`alc.toml`**: New project-level package declaration file. Declares packages with `name`, `source`, and optional `version`. Used as the source of truth for project-local package management
- **`alc.toml`-based project discovery**: Project root is now detected by walking up the directory tree to find `alc.toml` (previously `alc.lock`). `alc.lock` remains the resolved lockfile written by install/link operations
- **Lock mismatch warning**: Detects drift between `alc.toml` declarations and `alc.lock` resolved entries. Warns when packages declared in `alc.toml` are absent from `alc.lock` or vice versa
- **`PackageSource::Installed` / `PackageSource::Path`**: Renamed variants replacing `LocalCopy` and `LocalDir` respectively. `Installed` = package installed to cache from a URL; `Path` = symlinked local directory
- **`alc.toml` auto-append on install**: `alc_pkg_install` automatically appends the installed package to `alc.toml` when a project root is detected
- **Symlink-based `alc_pkg_link`**: Rewrites `pkg_link` to create a symlink inside `~/.algocline/packages/` pointing to the local directory. Removes the containment check entirely. `pkg_list` reports `linked`, `link_target`, and `broken` fields for symlink entries
- **Source provenance in `alc_pkg_list`**: Each entry now shows a `from` field indicating the install source (URL, path, or bundled)

### Changed

- **`alc_pkg_remove`**: Unified to remove from `alc.toml` + `alc.lock` only — cache directory is never deleted. The `scope` parameter is removed; removal always targets the project-local declaration
- **`alc_pkg_list`**: Project scope now reads from `alc.toml` (declarations) merged with `alc.lock` (resolved version/source), instead of reading `local_dir` entries directly from `alc.lock`
- **`PkgRemoveParams`**: `scope` field replaced by `version` (optional, for disambiguation)
- **`PkgLinkParams`**: `project_root` field removed; project root is auto-detected via `alc.toml` walk
- **`EngineApi` trait**: Removed `scope` from `pkg_remove`; added `alc_init`, `alc_update`, `alc_migrate`, `pkg_unlink` methods
- **`lockfile.rs`**: `LockPackage` loses `linked_at` field; gains `version: Option<String>`. `resolve_local_dir_paths` renamed to `resolve_path_entries` with containment check removed
- **`project.rs`**: `walk_up_for_lockfile` renamed to `walk_up_for_alc_toml`
- **`detect_legacy_format`**: Migrated from string-contains to TOML structural parsing to prevent false positives on package names containing `linked_at` or `local_dir`
- **Test helper consolidation**: Extracted duplicated `make_app_service` / `with_fake_home` into shared `test_support` module

### Fixed

- **`pkg_link` / `pkg_unlink` tests**: Replaced `Handle::block_on()` inside `#[tokio::test]` (runtime nesting panic) with `FakeHome` RAII guard pattern that allows direct `.await`. All 10 previously broken tests now pass
- **`eval_auto_installs_evalframe_on_missing` test**: Added `rt.enter()` guard for `AppService::new()` which calls `spawn_gc_task` requiring a runtime context; added `HOME_MUTEX` serialization to prevent env var races with `FakeHome` tests
- **Dead code cleanup**: Removed unused `resolve_installed_paths`, `resolve_abs`, and `#[allow(dead_code)]` annotations

## [0.14.0] - 2026-04-09

### Added

- **`alc_pkg_link`**: Link a local directory as a project-local package without copying. Records the path in `alc.lock`. Supports single package and collection layouts. Idempotent — re-linking updates the existing entry
- **`alc.lock`**: Project-local lockfile schema (version=1) for managing project-scoped package references. Stores `local_dir` entries pointing to on-disk paths
- **Project-local package resolution**: `alc.lock` `local_dir` entries are resolved as high-priority `FsResolver`s, taking precedence over `ALC_PACKAGES_PATH` and global `~/.algocline/packages/`. Enables per-project package overrides without modifying global state
- **`project_root` parameter**: `alc_run`, `alc_advice`, `alc_pkg_list`, `alc_pkg_remove` accept optional `project_root` to activate project-local package resolution. Auto-detected via `ALC_PROJECT_ROOT` env or `alc.lock` ancestor walk when omitted
- **`scope` parameter**: `alc_pkg_list` and `alc_pkg_remove` accept `scope` (`"project"` / `"global"`) for explicit scope targeting
- **`PackageSource` enum**: Type-safe representation of package origins (Git / LocalCopy / LocalDir / Bundled) with legacy string inference for backward compatibility

### Changed

- **`BUNDLED_VERSION`**: Updated bundled-packages from `v0.9.0` to `v0.11.0`
- **`EngineApi` trait**: `run` and `advice` gain `project_root: Option<String>` parameter; `pkg_list` gains `project_root`; `pkg_remove` gains `project_root` and `scope` (breaking for trait implementors)
- **`pkg.rs` → `pkg/` module**: Split monolithic `pkg.rs` into `pkg/install.rs`, `pkg/list.rs`, `pkg/remove.rs`, `pkg/tests.rs` submodules

### Fixed

- **Lua injection prevention**: Package names are whitelist-validated before interpolation into Lua source in `pkg_list` meta evaluation
- **Path containment**: `pkg_link` canonicalizes and containment-checks `LocalDir` paths so `alc.lock` cannot reference paths outside `project_root`
- **Atomic lockfile writes**: `save_lockfile` uses `NamedTempFile` + `persist` to prevent readers from observing half-written `alc.lock`
- **`eval_simple` require cache**: Clears `package.loaded[name]` before meta evaluation to avoid stale cached modules across calls

## [0.13.0] - 2026-04-04

### Added

- **`alc.llm_json(prompt, opts?)`**: LLM call with automatic JSON parsing and 1-retry repair. Uses `alc.json_extract` for 3-stage fallback parsing; on failure, retries with previous output included so the model can fix rather than regenerate
- **`alc.math`**: Numeric computing namespace (44 functions) via mlua-mathlib v0.2.0 — RNG, distribution sampling (Normal, Beta, Gamma, Poisson, Binomial, etc.), descriptive statistics, CDF/PPF, special functions (erf, gamma, beta, digamma, factorial), transforms (softmax, histogram, Wilson CI)
- **`docs/lua-stdlib.md`**: `alc.math` section with full API reference
- **`types/alc.d.lua`**: LuaCats type definitions for all `alc.math.*` functions

### Changed

- **`BUNDLED_VERSION`**: Updated bundled-packages from `v0.7.0` to `v0.9.0`, evalframe from `v0.1.0` to `v0.3.0`
- **Dependencies**: mlua-mathlib `0.1` → `0.2`

## [0.12.1] - 2026-04-02

### Fixed

- **`alc.match_bool`**: Add word boundary check to prevent false positives (e.g. `"ok"` in `"token"`, `"pass"` in `"bypass"`, `"no"` in `"innovation"`)
- **`alc.match_enum`**: Fuzzy fallback now splits text into words and compares per-word instead of whole-text, enabling typo detection in long LLM responses

### Added

- **`docs/lua-stdlib.md`**: Type Support section — LuaCats setup and `lua-language-server --check` CI integration guide

## [0.12.0] - 2026-04-02

### Added

- **`alc.match_enum(text, candidates, opts?)`**: Fuzzy enum matcher for LLM output. Case-insensitive substring match with Jaro-Winkler fuzzy fallback (Layer 0, powered by `fuzzy-parser` crate)
- **`alc.match_bool(text)`**: Yes/no normalizer for LLM responses. Returns `true`, `false`, or `nil` based on last-occurring affirmative/negative keyword (Layer 0)
- **`alc.parse_number(text, pattern?)`**: Extract numbers from LLM output with optional Lua pattern (Layer 1 Prelude)
- **Host token tracking**: `alc_continue` accepts optional `usage` field with `prompt_tokens` / `completion_tokens`. Tracked as `TokenSource::Host` in `ExecutionMetrics`, providing accurate token counts instead of character-based estimates
- **`max_tokens` budget**: Host can set `max_tokens` in `alc_run` context (`ctx._max_tokens`). When budget is exhausted, subsequent `alc.llm()` calls fail with a budget error
- **`alc init` / `alc update`**: Distributes `alc.d.lua` LuaCats type stub to `~/.algocline/types/alc.d.lua` on every run. Enables editor completion (Lua Language Server, `lua_ls`) for all `alc.*` StdLib functions. If `.luarc.json` is absent from the current directory, a setup tip is printed to stderr
- **MCP server startup**: Automatically distributes `alc.d.lua` on each server start, so the type stub is always up-to-date after `cargo install`
- **`alc_pkg_install` response**: Added `types_path` field — absolute path to the installed `alc.d.lua` stub — so MCP clients can surface the location without an extra tool call

### Changed

- **`alc_advice` `task` parameter**: Now optional (`Option<String>`). Packages that don't use `ctx.task` (e.g. `factscore`, `optimize`, `lineage`) can be called with `opts` alone
- **`EngineApi::advice` trait**: `task` parameter changed from `String` to `Option<String>` (breaking for trait implementors)
- **`EngineApi::continue_single` trait**: Added `usage: Option<TokenUsage>` parameter (breaking for trait implementors)

## [0.11.1] - 2026-04-01

### Changed

- **`alc_log_view`**: Added `max_chars` parameter for detail mode (default: 100KB). Truncates transcript from oldest rounds when response exceeds limit. Set `max_chars=0` for unlimited

## [0.11.0] - 2026-03-30

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.6.0` to `v0.7.0`

### Fixed

- **Clippy warnings**: Removed redundant closure in `spec.rs`, replaced `assert_eq!(…, true)` with `assert!()` in unit tests

## [0.10.0] - 2026-03-25

### Added

- **`alc.fork(strategies, ctx, opts?)`**: Parallel multi-VM strategy execution (Layer 0). Spawns N independent Lua VMs, each running one strategy with the same context. LLM requests from all children are batched through the parent's channel for true LLM parallelism. Strategy names validated (alphanumeric + underscore only)
- **`alc.cache(prompt, opts?)`**: Session-scoped memoized LLM call (Layer 1). Returns cached response for repeated identical prompts. FIFO eviction at 256 entries. Supports `cache_key` override and `cache_skip` bypass. `alc.cache_info()` / `alc.cache_clear()` for introspection
- **`alc.parallel(items, prompt_fn, opts?)`**: Batch-parallel LLM calls over an array (Layer 1). Transforms each item into a prompt via `prompt_fn`, sends all as a single `alc.llm_batch()` call. Optional `post_fn` for response post-processing
- **`QueryId::fork(vm_index, seq)`**: Fork-specific query ID format (`f-{vm}-{seq}`) for child VM LLM request tracking
- **`query_id` auto-resolve**: `alc_continue` without explicit `query_id` now auto-resolves when exactly one query is pending. Returns error for zero or multiple pending queries
- **`query_id` in response**: Single-query `needs_response` now includes `query_id` field for explicit identification

### Changed

- **`EngineApi` trait**: Extracted transport-independent API trait from `AppService` into `algocline-core`. MCP handler now operates through `Arc<dyn EngineApi>`, enabling future remote (socket/HTTP) implementations without depending on the concrete `AppService`
- **`FeedResult`, `ExecutionResult`, `TerminalState`**: Added `Serialize` derive for future transport serialization (HTTP/gRPC)
- **`BridgeConfig`**: Added `lib_paths` field for package search paths (needed by `alc.fork` to setup child VMs)
- **`bridge` module split**: Extracted `ForkEvent`, `ForkQuery`, `register_fork` into `bridge/fork.rs` submodule (bridge.rs 1249 → mod.rs 934 + fork.rs 345)

## [0.9.0] - 2026-03-24

### Added

- **Budget control**: `ctx.budget` with `max_llm_calls` and `max_elapsed_ms` limits. `alc.budget_remaining()` (Layer 0) returns remaining capacity, `alc.budget_check()` (Layer 1) provides boolean guard for optional LLM calls. Budget is enforced at `alc.llm()` / `alc.llm_batch()` call time
- **Token estimation**: `TokenCount` and `TokenSource` types for prompt/response token tracking in `ExecutionMetrics`
- **Progress reporting**: `alc.progress(step, total, msg?)` for structured step tracking, readable via `alc_status`
- **`alc_status`**: MCP tool to query active session status — state, metrics snapshot, progress, and strategy name. Omit `session_id` to list all active sessions
- **`alc.pipe(strategies, ctx, opts?)`**: Sequential pipeline combinator. Chains multiple strategies, passing each stage's result as the next stage's `ctx.task`. Supports both `require()`-based strategies and inline functions. Records `pipe_history` for debugging

### Changed

- **`BridgeConfig` struct**: Replaced growing parameter list in `bridge::register()` with a single config struct holding `llm_tx`, `ns`, `custom_metrics`, `budget`, and `progress` handles
- **Handle-based metrics access**: `CustomMetrics`, `Budget`, `Progress` now accessed via cloneable Handle types instead of `Arc<Mutex<T>>` directly

## [0.8.0] - 2026-03-24

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.4.0` to `v0.5.0` (9 new packages: 5 orchestration — orch_fixpipe, orch_gatephase, orch_adaptive, orch_nver, orch_escalate; 3 routing — router_daao, router_semantic, router_capability; 1 optimization — optimize)

## [0.7.1] - 2026-03-22

### Fixed

- **Per-session VM isolation**: Each `alc_run` / `alc_advice` call now spawns a dedicated Lua VM. Previously, all sessions shared a single VM, causing global namespace pollution (`alc`, `ctx`, `package.loaded`) between concurrent sessions. This eliminates coroutine cross-contamination when running multiple strategies in parallel

### Changed

- **`package.loaded` clearing removed**: No longer needed since each session starts with a fresh VM

## [0.7.0] - 2026-03-22

### Added

- **`alc_stats`**: Aggregate usage stats across all logged sessions. Per-strategy counts, averages (elapsed_ms, llm_calls, rounds), and totals. Optional `strategy` filter and `days` time window
- **`alc_info`**: Diagnostic tool showing server configuration — resolved log directory (with source), tracing mode, packages directory, and version. Similar to `mise doctor`
- **Strategy tracking**: Session logs (`.json` and `.meta.json`) now record `strategy` name for all advice/eval sessions, enabling per-strategy analytics

### Changed

- **`AppConfig`**: Replaced `TranscriptConfig` with centralized `AppConfig` resolved from environment variables. Single resolution point for all configuration
- **Log directory fallback chain**: `ALC_LOG_DIR` env → `~/.algocline/logs` → `$XDG_STATE_HOME/algocline/logs` → `<cwd>/algocline-logs` → None (stderr-only). Sandbox/container environments now preserve file logging via cwd fallback
- **Tracing**: Unified `setup_tracing` into single function accepting `Option<&Path>`. File + stderr when log dir available, stderr-only otherwise
- **Crate dependencies**: Removed `algocline-engine` dependency from `algocline-mcp` — accepts `AppService` directly

### Refactored

- **`algocline-app::service`**: Split 3099-line monolithic `service.rs` into domain-based module directory (`service/config.rs`, `path.rs`, `resolve.rs`, `transcript.rs`, `eval_store.rs`, `run.rs`, `eval.rs`, `pkg.rs`, `logging.rs`, `scenario.rs`, `tests/`). No API changes

## [0.6.0] - 2026-03-20

### Added

- **`alc.json_extract(raw)`**: Extract JSON object/array from LLM output. Handles raw JSON, markdown fences (` ```json ``` `), and embedded JSON within surrounding text via balanced brace/bracket iteration
- **`alc.state.update(key, fn, default?)`**: Single-operation read-modify-write for state. Reads current value, applies transform function, writes back
- **`alc.llm_safe(prompt, opts, default)`**: Non-throwing LLM wrapper. Returns default on failure instead of raising, logs warning. For optional enrichment where failure should not abort the pipeline
- **`alc.fingerprint(str)`**: Text normalization + DJB2 hash (8-char hex). For deduplication, not cryptography
- **`alc.tuning(defaults, ctx, opts?)`**: Config merge with deep-merge support for dict-like nested tables, shallow-replace for arrays/scalars. Supports `opts.prefix` for namespaced overrides, strips `_schema` key (reserved for Layer 2 parameter metadata)

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.3.0` to `v0.4.0` (6 new strategy packages: s2a, plan_solve, rstar, faithful, moa, bot)

### Fixed

- **`alc.json_extract`**: Iterate all balanced brace/bracket pairs via `gmatch` instead of first-match-only. Fixes false-negative when non-JSON balanced text precedes valid JSON
- **`alc.fingerprint`**: DJB2 modulo corrected from `0xFFFFFFFF` (2^32-1) to `0x100000000` (2^32) per standard specification
- **`alc.tuning`**: Warn and fall back to defaults when `opts.prefix` value exists but is not a table, preventing silent unintended overrides from top-level ctx keys

## [0.5.0] - 2026-03-18

### Added

- **Scenario management**: `alc_scenario_list`, `alc_scenario_show`, `alc_scenario_install` tools for managing reusable eval scenarios in `~/.algocline/scenarios/`
- **`scenario_name` parameter**: `alc_eval` now accepts `scenario_name` to load installed scenarios by name (e.g. `"math_basic"`), in addition to existing `scenario` (inline) and `scenario_file` (path)
- **Bundled scenarios**: `alc init` / `alc_pkg_install` automatically installs scenarios from `scenarios/` subdirectory in package collections
- **Resilience pattern**: `DirEntryFailures` type alias for batch I/O operations that collect per-entry failures instead of aborting. JSON responses include `"failures"` field for diagnostics

### Changed

- **`BUNDLED_VERSION`**: Updated from `0.2.0` to `v0.3.0` (includes 9 new strategy packages, robust_qa, and 3 bundled eval scenarios)

## [0.4.0] - 2026-03-17

### Added

- **`alc_eval`**: Evaluate a strategy against a scenario with test cases and graders. Accepts inline Lua (`scenario`) or file path (`scenario_file`) with a strategy name. Strategy is auto-wired as provider via `ef.providers.algocline`
- **`alc_eval_history`**: List past eval results with optional strategy filter, sorted newest-first
- **`alc_eval_detail`**: View a specific eval result by ID in full detail
- **`alc_eval_compare`**: Compare two eval results with Welch's t-test for statistical significance via evalframe's `stats.welch_t`
- **Eval persistence**: Results automatically saved to `~/.algocline/evals/` with full JSON result + lightweight meta files for fast listing
- **`alc.time()`**: Wall-clock primitive for evalframe latency tracking
- **evalframe**: Bundled as a system dependency, auto-installed on first `alc_eval` / `alc_eval_compare` use
- **Multi-source bundled installation**: `alc init` now supports multiple source repositories (Collection and Single kinds) instead of a single URL. `--dev` mode searches local sibling directories

### Changed

- **`alc_pkg_list`**: System packages (evalframe) excluded from listing to avoid require errors and declutter output
- **Lua string escaping**: Fixed escaping for newlines/carriage returns in bridge layer

## [0.3.0] - 2026-03-15

### Added

- **`underspecified` flag**: New domain primitive on `LlmQuery` for marking prompts whose preconditions depend on intent/goal definitions outside the current context. Same serde pattern as `grounded` flag
- **`alc.specify()`**: Layer 1 prelude convenience wrapper that sets `underspecified = true`, pairing with `alc.ground()` / `grounded` pattern
- **Bundled packages v0.2.0**: 15 new packages including intent understanding (ambig, prism, intent_discovery, intent_belief), reasoning strategies (ucb, panel, cot, sc, reflect, calibrate, contrastive, meta_prompt, factscore, cove), and combinators (deliberate, pre_mortem)

### Changed

- **`BUNDLED_VERSION`**: Updated from `0.1.0` to `0.2.0`

## [0.2.1] - 2026-03-15

### Changed

- **`alc init` versioning**: Decoupled bundled packages version from algocline's own `CARGO_PKG_VERSION`. Introduced `BUNDLED_VERSION` constant (`0.1.0`) so the two can evolve independently
- **`alc init` transport**: Replaced GitHub Releases tarball download with `git clone --branch v{BUNDLED_VERSION}`, eliminating the need for release asset management

### Removed

- **`review` package**: Removed from bundled package list (poor output quality)

## [0.2.0] - 2026-03-15

### Added

- **Transcript logging**: Full prompt/response transcript saved to `~/.algocline/logs/{session_id}.json` with lightweight `.meta.json` summaries for fast listing
- **Session notes**: `alc_note` tool to annotate completed sessions with feedback/observations; notes persisted in log files with `notes_count` tracked in meta
- **Log viewer**: `alc_log_view` tool to list sessions (from meta files) or view full transcript detail
- **Auto stats**: `rounds`, `total_prompt_chars`, `total_response_chars` tracked automatically via `MetricsObserver`
- **Transcript in stats**: `transcript_to_json()` on `ExecutionMetrics` for structured prompt/response history (excluded from `to_json()` stats output)
- **Local package install**: `alc_pkg_install` accepts absolute local paths, copying directly without git clone; supports both single packages and collections with overwrite semantics for dev workflow
- **Collection install**: Package repositories with `*/init.lua` subdirs are detected as collections and each subdir installed as a separate package
- **Test suite**: 151 tests across all crates — unit tests, property-based tests (proptest), path traversal rejection, chunk function invariants, state machine transitions

### Changed

- **Package architecture**: Standard packages extracted to separate `algocline-bundled-packages` repository; `alc_advice` auto-installs from GitHub if requested package is missing
- **MSRV**: Updated from 1.77 to 1.88

## [0.1.0] - 2026-03-01

### Added

- Initial release
- MCP server with `alc_run`, `alc_continue`, `alc_advice` tools
- Three-layer Lua StdLib: Layer 0 (Rust primitives), Layer 1 (Lua prelude), Layer 2 (packages via `require()`)
- `alc.llm()` / `alc.llm_batch()` — coroutine-based async LLM calls
- `alc.json_encode` / `alc.json_decode` — serde_json bridge
- `alc.log()` — tracing bridge
- `alc.state` — persistent key-value store (`~/.algocline/state/`)
- `alc.chunk()` — text segmentation (lines/chars with overlap)
- `alc.stats` — custom metrics recording
- Prelude combinators: `alc.map`, `alc.reduce`, `alc.vote`, `alc.filter`
- Package management: `alc_pkg_list`, `alc_pkg_install`, `alc_pkg_remove`
- `alc init` — bundled package installer (GitHub Releases + local fallback)
- Domain model: `ExecutionState` state machine with `PendingQueries` join barrier
- `ExecutionObserver` trait for cross-cutting concerns
- `SessionRegistry` for concurrent session management
- `ContainedPath` for path traversal prevention
- Coroutine-based execution via `mlua-isle` (non-blocking `alc.llm()`)
