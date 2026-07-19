# algocline-engine::bridge

Layer 0: Runtime Primitives

Registers Rust-backed functions into the `alc.*` Lua namespace.
These provide capabilities that cannot be expressed in Pure Lua:
I/O (state), serialization (json), host communication (llm),
and text processing (chunk).

All functions registered here are available in every Lua session
without explicit `require()`.

## Functions

- `install_for_pkg_test` — Install the production `alc.*` primitive surface plus the mock layer
- `register` — Register all Layer 0 runtime primitives onto the given table.

## Types

- `BridgeConfig` — All handles needed by Layer 0 runtime primitives.

## Constants

- `PRELUDE` — Layer 1 prelude (also used by fork to setup child VMs).

