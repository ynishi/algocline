# Examples

Scripts that drive the public `alc.*` surface end to end. Each one is
self-contained and is meant to be run through `alc_run` (`code_file`
form) or the Rust harnesses under `crates/algocline-engine/tests/`.

| Path | What it exercises |
|---|---|
| `cards/` | Card store: an A/B prompt comparison and a sweep replay through `alc.card.*` |
| `nn_full_ft_smoke.lua` / `nn_lora_smoke.lua` / `nn_distill_smoke.lua` | The three `alc.nn.trainer` paths on a synthetic corpus (driven by `tests/nn_smoke_test.rs`) |
| `nn_fullft_shakespeare_e2e.lua` / `nn_lora_shakespeare_e2e.lua` / `nn_distill_llm_teacher_e2e.lua` | The same paths on a real corpus; see `docs/nn-e2e-runbook.md` |
| `nn_medium_gpu_bridge_smoke.lua` | The `medium` preset over the Lua bridge on a GPU pod |

The GameAI application (a card-duel / boss-duel NPC whose play style is
a tuned small model) no longer lives here. It is an application built
on the platform rather than part of it, so it moved out of this
repository; its history was split off intact (`git subtree split -P
examples/gameai`) to be published as its own repository. The platform
contracts it exercised — train → Card → gated decode, and the
checkpoint hook feeding a metric — are fenced in-tree by
`tests/nn_gate_smoke.rs` and `tests/nn_ckpt_hook_e2e.rs` over the
game-free `pick` fixture in `tests/lua/fixtures/`.
