//! Spec-driven customization points for the GPT-2 stack (Phase 1).
//!
//! Generalizes the seam the dense-MoE block opened: instead of one
//! `Option` field per experiment, [`Gpt2Custom`] collects the
//! architecture axes a block can deviate from the GPT-2 reference on,
//! and `Block::new` builds from the spec. `Gpt2Config::custom = None`
//! keeps the stock GPT-2 behaviour bit-for-bit (same ops, same VarMap
//! names).
//!
//! Phase 1 axes (VarMap-light: names stay identical or only gain
//! entries):
//!
//! - **activation** — GELU (reference) / ReLU / SiLU, plus the gated
//!   variants SwiGLU / GeGLU from Shazeer 2020 (arXiv:2002.05202).
//!   Gated variants add a `mlp.c_gate` projection (the activated
//!   branch; `mlp.c_fc` stays the linear branch, Llama's `up_proj`).
//! - **norm** — LayerNorm (reference) / RMSNorm (Zhang & Sennrich
//!   2019, arXiv:1910.07467). RMSNorm keeps the `ln_*.weight` names
//!   and simply has no bias, and goes through the backward-safe
//!   `rms_norm_slow` shim like TinyLlama.
//! - **residual topology** — sequential (reference) / parallel
//!   attention + MLP (GPT-J / PaLM, arXiv:2204.02311):
//!   `y = x + attn(ln_1(x)) + ff(ln_2(x))`.
//! - **mlp_ratio** — the MLP expansion factor (reference 4).
//!
//! Phase 2 (positional encodings, GQA, sliding-window attention,
//! untied head, Post-LN) extends this struct; `Default` lets existing
//! constructions survive field additions.
//!
//! All axes are experiment equipment, so a config that sets `custom`
//! is **random-init only**: the pretrained loaders and the merged
//! exporter refuse it (same guard family as MoE).

use candle_core::Result as CandleResult;

/// MLP activation. `Gelu` is the GPT-2 reference; the gated variants
/// compute `act(c_gate(x)) * c_fc(x)` (Shazeer 2020).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// GPT-2 reference.
    #[default]
    Gelu,
    /// Rectified linear unit.
    Relu,
    /// Sigmoid-weighted linear unit (`x * sigmoid(x)`).
    Silu,
    /// `silu(c_gate(x)) * c_fc(x)` — Llama-family convention.
    SwiGlu,
    /// `gelu(c_gate(x)) * c_fc(x)`.
    GeGlu,
}

impl Activation {
    /// Gated variants carry the extra `mlp.c_gate` projection.
    pub fn is_gated(&self) -> bool {
        matches!(self, Self::SwiGlu | Self::GeGlu)
    }
}

/// Block / final normalization kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormKind {
    /// GPT-2 reference (affine weight + bias).
    #[default]
    LayerNorm,
    /// Weight-only RMS normalization (no mean subtraction, no bias).
    RmsNorm,
}

/// How the block combines its two halves with the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualKind {
    /// GPT-2 reference: `x' = x + attn(ln_1(x)); y = x' + ff(ln_2(x'))`.
    #[default]
    Sequential,
    /// GPT-J / PaLM: `y = x + attn(ln_1(x)) + ff(ln_2(x))` — both
    /// halves read the same input; one residual write.
    Parallel,
}

/// Architecture customization spec. `Default` reproduces the GPT-2
/// reference on every axis, so `custom: Some(Gpt2Custom::default())`
/// builds the same network shape as `custom: None` (it still trips the
/// random-init-only guard — the guard keys on presence, not content,
/// for simplicity).
#[derive(Debug, Clone, Default)]
pub struct Gpt2Custom {
    /// MLP activation.
    pub act: Activation,
    /// Normalization kind (applies to `ln_1` / `ln_2` / `ln_f`).
    pub norm: NormKind,
    /// Residual topology.
    pub residual: ResidualKind,
    /// MLP expansion factor (`hidden = mlp_ratio * dim`). Reference 4.
    /// `0` means "reference" and is normalized to 4 by
    /// [`Self::mlp_hidden`] — but prefer writing 4 explicitly.
    pub mlp_ratio: usize,
}

impl Gpt2Custom {
    /// MLP hidden width for a model of width `dim`.
    pub fn mlp_hidden(&self, dim: usize) -> usize {
        let ratio = if self.mlp_ratio == 0 {
            4
        } else {
            self.mlp_ratio
        };
        ratio * dim
    }

    /// Validate the invariants the builder assumes.
    pub fn validate(&self) -> CandleResult<()> {
        // All Phase 1 axes are closed enums; only the numeric knob can
        // go out of range, and `0` is normalized rather than rejected
        // so `Gpt2Custom::default()` stays a valid spec.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_reference() {
        let c = Gpt2Custom::default();
        assert_eq!(c.act, Activation::Gelu);
        assert_eq!(c.norm, NormKind::LayerNorm);
        assert_eq!(c.residual, ResidualKind::Sequential);
        assert_eq!(c.mlp_hidden(16), 64); // ratio 0 normalizes to 4
    }

    #[test]
    fn gated_flag() {
        assert!(Activation::SwiGlu.is_gated());
        assert!(Activation::GeGlu.is_gated());
        assert!(!Activation::Gelu.is_gated());
        assert!(!Activation::Relu.is_gated());
        assert!(!Activation::Silu.is_gated());
    }

    #[test]
    fn mlp_hidden_uses_ratio() {
        let c = Gpt2Custom {
            mlp_ratio: 3,
            ..Default::default()
        };
        assert_eq!(c.mlp_hidden(16), 48);
    }
}
