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

pub use gpt2::{Gpt2Config, Gpt2Model};
pub use lora::{max_abs_diff_f32, LoraConfig, LoraLinear};
