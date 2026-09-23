# algocline-nn::sampling

Next-token samplers for the inference path.

Consumes a `[vocab]`-shaped logits row (one position, one batch item)
and returns the sampled token id. Callers responsible for the outer
loop — the sampler holds no cache, no state beyond its own RNG, and
no dependency on the specific adapter that produced the logits.

# The Sampler 3-layer plan

- **Layer 1 (this module)** — [`Sampler`] trait + Rust default
  implementations ([`GreedySampler`] / [`TemperatureSampler`] /
  [`TopKTopPSampler`]). Every consumer starts here.
- **Layer 2 ([`constraint`] / [`json_schema`])** — filters that mask
  logits before a Layer 1 sampler picks, wired in through
  [`ConstrainedSampler`], itself a `Sampler`. Four [`Constraint`]s
  have landed: [`StopTokensConstraint`] (termination only),
  [`AllowListConstraint`] (a fixed legal set, masking only),
  [`RegexConstraint`] (anchored full-match over a tokenizer's surface
  strings) and [`JsonSchemaConstraint`] (a JSON schema compiled to a
  regex and enforced through the previous one). GBNF grammars are a
  future addition behind the same trait — and the one that unlocks
  recursive schemas, which no regular language can express.
- **Layer 3 (engine side: `alc.nn.sampler` / `alc.nn.constraint`)** —
  Lua factories that build the types above, compose them, and let a
  Lua function *be* a `Sampler`. No scheduling primitive lives here:
  the generation loop is already Lua-side, so swapping the active
  sampler per position is plain Lua control flow over two handles.
  The only thing this layer needed from Layer 1 was the ability to
  erase the sampler's concrete type — see the
  `impl Sampler for Box<dyn Sampler + Send>` below.

Layer 2 / 3 attach as additional `impl Sampler` types (including the
Lua-callback bridge on the engine side) without changing the trait.
That is the entire point of Layer 1.

# Determinism

Every stochastic sampler carries its own [`StdRng`]. A caller that
needs save/load reproducibility supplies a fixed seed at
construction; the RNG state advances by exactly one draw per
[`Sampler::sample`] call, so two runs with the same seed and same
logits stream reproduce the same tokens.

[`Sampler::sample`] takes `&mut self` to allow the RNG state to
advance; a caller sharing a sampler across generation loops (unlikely
given the state semantics) is expected to serialise access
themselves.

## Types

- `GreedySampler` — Argmax: pick the highest-scoring token.
- `TemperatureSampler` — Temperature-scaled multinomial sampler.
- `TopKTopPSampler` — Nucleus (top-p) + top-k truncation, then temperature-scaled

## Traits

- `Sampler` — Next-token sampler.

