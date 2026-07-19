# algocline 0.45.0

# algocline — LLM amplification engine

algocline provides a Lua execution environment for structurally enhancing LLM
reasoning. Strategies are Pure Lua modules that call `alc.*` primitives
to orchestrate multi-step LLM interactions.

## Architecture: Three-Layer StdLib

```text
Layer 0: Runtime Primitives (Rust → alc.*)
│  Injected by bridge.rs into every Lua session.
│  These are the foundational building blocks that cannot be
│  expressed in Pure Lua (I/O, LLM calls, serialization).
│
│  alc.llm(prompt, opts?)       — Host LLM call via MCP Sampling
│  alc.json_encode / json_decode — serde_json bridge
│  alc.log(level, msg)          — tracing bridge
│  alc.state.get/set/keys/delete — persistent key-value store
│  alc.chunk(text, opts?)       — text segmentation
│
Layer 1: Prelude Combinators (Lua → alc.*)
│  Loaded from prelude.lua (embedded via include_str!).
│  Higher-order functions that compose Layer 0 primitives.
│  Auto-injected into alc.* namespace alongside Layer 0.
│
│  alc.map(items, fn)          — transform each element
│  alc.reduce(items, fn, init) — fold to single value
│  alc.vote(answers)           — majority aggregation
│  alc.filter(items, fn)       — conditional selection
│
Layer 2: Bundled Packages (require() from ~/.algocline/packages/)
   Installed to ~/.algocline/packages/ via `alc init`.
   Each is a self-contained Lua module built on Layer 0/1.
   Loaded explicitly via require("{name}").

   explore  — UCB1 hypothesis space exploration    [selection]
   panel    — multi-perspective deliberation       [synthesis]
   chain    — iterative chain-of-thought           [reasoning]
   ensemble — independent sampling + majority vote [aggregation]
   verify   — draft-verify-revise cycle            [validation]
```

**Design rationale**: Layer 0/1 form the built-in library — always
available, no explicit import needed. Layer 2 packages are bundled
but opt-in via `require()`, analogous to how `tokio` relates to `std`
in the Rust ecosystem.

