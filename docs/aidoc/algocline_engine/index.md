# algocline-engine 0.45.0

Lua strategy execution engine.

Owns the [`Session`] lifecycle (running / needs_response / completed),
the [`Executor`] loop that drives Lua coroutines and mediates
`alc.llm()` pauses, the [`FileCardStore`] for structured session
artifacts, [`bridge`] modules that expose Rust-backed globals
(`alc.*`) to Lua, and the resolver factory that wires strategies
to LLM providers.

## Modules

- [`bridge`](bridge.md): Layer 0: Runtime Primitives
- [`card`](card.md): Card storage — immutable run-result snapshots.
- [`execution`](execution.md): `algocline-engine::execution` — v2 execution module.
- [`session`](session.md): Session-based Lua execution with pause/resume on alc.llm() calls.
- [`state`](state.md): Persistent key-value state backed by JSON files.

