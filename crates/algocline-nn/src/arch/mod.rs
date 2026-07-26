//! Architecture presets for `alc.nn.preset.*`.
//!
//! Each preset is a pure function of a config → runnable candle module.
//! Phase 1 ships GPT-2 (medium / large) here; TinyLlama (Phase 2) will
//! land alongside as `arch::tinyllama` on the same layout.
//!
//! The Lua bridge in `algocline-engine` (`bridge/nn_card.rs`) builds a
//! [`Gpt2Config`] from user opts, then calls [`Gpt2Model::new`] (random
//! init) or [`Gpt2Model::from_pretrained`] (HF warm-start; design §12
//! Q5) to obtain the model handle.
//!
//! Weight naming convention follows the reference GPT-2 checkpoints
//! (nanoGPT / `openai-community/gpt2*`) so a downloaded safetensors
//! bundle can be loaded through [`candle_nn::VarBuilder::from_mmaped_safetensors`]
//! with the same variable names used at construction time.

pub mod adapter;
pub mod gpt2;
pub mod lora;
pub mod moe;
pub mod tinyllama;

/// Softmax over the last dimension through the backward-safe basic-op
/// composition (`candle_nn::ops::softmax`), never the fused
/// `ops::softmax_last_dim`.
///
/// Rationale: candle-nn 0.11's `softmax_last_dim` is a `CustomOp1`
/// registered via `apply_op1_no_bwd` — its output carries
/// `BackpropOp::none()`, so backward silently treats it as a constant
/// and every parameter whose only gradient path runs through it
/// receives no gradient. Same cliff family as the LayerNorm fast path
/// (`gpt2::apply_slow_layer_norm`) and the TinyLlama RMSNorm / RoPE
/// shims. Discovered by `tests/moe_grad_coverage.rs`: the MoE router's
/// only gradient path is its softmax, so the severing that attention
/// masks (the fused / parallel V path keeps projection gradients
/// non-zero) showed up as a hard "missing from GradStore".
///
/// `ops::softmax` composes `max_keepdim` / `broadcast_sub` / `exp` /
/// `sum_keepdim` / `broadcast_div`, each with a proper backward.
pub(crate) fn softmax_last_dim_slow(
    xs: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    candle_nn::ops::softmax(xs, candle_core::D::Minus1)
}

pub use gpt2::{Gpt2Config, Gpt2Model};
pub use lora::{max_abs_diff_f32, LoraConfig, LoraLinear, LoraWrappable};
pub use moe::MoeConfig;
pub use tinyllama::{TinyLlamaConfig, TinyLlamaModel};
