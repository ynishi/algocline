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
//! Phase 2 axes (VarMap entries move or the attention wiring changes):
//!
//! - **position** — learned `wpe` (reference) / RoPE (Su 2021,
//!   arXiv:2104.09864) / ALiBi (Press 2022, arXiv:2108.12409) / NoPE
//!   (Kazemnejad 2023, arXiv:2305.19466). Every non-learned variant
//!   drops the `wpe` Var; RoPE reuses TinyLlama's backward-safe
//!   `apply_rope` shim, ALiBi adds a constant per-head score bias.
//! - **GQA** — `kv_heads` shrinks the fused `c_attn` projection to
//!   `[dim + 2·kv·head_dim, dim]` and shares each KV head across
//!   `heads / kv_heads` query heads via `repeat_kv` (Ainslie 2023,
//!   arXiv:2305.13245).
//! - **sliding-window attention** — `window` bands the causal mask so
//!   position `i` attends to `(i - w, i]` (Mistral 2023).
//! - **untied head** — an independent `lm_head.weight` Var instead of
//!   reusing `wte`.
//! - **Post-LN** — norm after the sublayer + residual add (Xiong et
//!   al. 2020, arXiv:2002.04745) instead of the Pre-LN reference. Its
//!   known training instability is a probe subject, not a defect.
//!   Post-LN combined with the parallel residual topology has no
//!   canonical wiring and is rejected at build time.
//!
//! All axes are experiment equipment, so a config that sets `custom`
//! keeps the HuggingFace-hub loaders shut: [`super::gpt2::Gpt2Model::from_pretrained`]
//! and the merged exporter refuse it (same guard family as MoE). Bundles
//! written by this crate's own trainer (`VarMap::save`) *are* loadable —
//! they carry exactly the Vars the spec declares — which is why every
//! type in this module is `Serialize` / `Deserialize`: a Card records
//! its spec so the load path can rebuild the identical config.
//!
//! The serde representation is the same lowercase vocabulary the Lua
//! bridge accepts (`"swiglu"` / `"rmsnorm"` / `"preln"` / `"nope"` /
//! ...), so a Card's `custom` table reads the way the caller wrote it.

use candle_core::Result as CandleResult;
use serde::{Deserialize, Serialize};

/// MLP activation. `Gelu` is the GPT-2 reference; the gated variants
/// compute `act(c_gate(x)) * c_fc(x)` (Shazeer 2020).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NormKind {
    /// GPT-2 reference (affine weight + bias).
    #[default]
    LayerNorm,
    /// Weight-only RMS normalization (no mean subtraction, no bias).
    RmsNorm,
}

/// How the block combines its two halves with the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResidualKind {
    /// GPT-2 reference: `x' = x + attn(ln_1(x)); y = x' + ff(ln_2(x'))`.
    #[default]
    Sequential,
    /// GPT-J / PaLM: `y = x + attn(ln_1(x)) + ff(ln_2(x))` — both
    /// halves read the same input; one residual write.
    Parallel,
}

/// Where the block norms sit relative to the sublayers (Xiong et al.
/// 2020, arXiv:2002.04745).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NormPlacement {
    /// GPT-2 reference: norm the sublayer *input*
    /// (`x + attn(ln_1(x))`).
    #[default]
    PreLn,
    /// Original-Transformer placement: norm the residual *sum*
    /// (`ln_1(x + attn(x))`). Known to be harder to train at depth —
    /// observing that instability is the point of shipping it.
    PostLn,
}

/// How the model injects position information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PosKind {
    /// GPT-2 reference: learned absolute embedding (`wpe`), added to
    /// the token embedding.
    #[default]
    Learned,
    /// Rotary embedding applied to Q / K inside every attention
    /// (Su 2021). No `wpe` Var.
    Rope,
    /// Constant per-head linear score bias (Press 2022). No `wpe` Var.
    Alibi,
    /// No positional information at all (Kazemnejad 2023's NoPE). The
    /// causal mask is the only order signal. No `wpe` Var.
    ///
    /// Serialized as `"nope"` (the paper's acronym and the Lua
    /// bridge's spelling), not the Rust variant name.
    #[serde(rename = "nope")]
    NoPos,
}

/// Architecture customization spec. `Default` reproduces the GPT-2
/// reference on every axis, so `custom: Some(Gpt2Custom::default())`
/// builds the same network shape as `custom: None` (it still trips the
/// random-init-only guard — the guard keys on presence, not content,
/// for simplicity).
///
/// Every field carries `#[serde(default)]` so a Card written before an
/// axis existed still deserializes into the reference value for that
/// axis; `Option` fields are omitted when absent so a spec table only
/// lists the axes the caller actually set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Gpt2Custom {
    /// MLP activation.
    #[serde(default)]
    pub act: Activation,
    /// Normalization kind (applies to `ln_1` / `ln_2` / `ln_f`).
    #[serde(default)]
    pub norm: NormKind,
    /// Residual topology.
    #[serde(default)]
    pub residual: ResidualKind,
    /// MLP expansion factor (`hidden = mlp_ratio * dim`). Reference 4.
    /// `0` means "reference" and is normalized to 4 by
    /// [`Self::mlp_hidden`] — but prefer writing 4 explicitly.
    #[serde(default)]
    pub mlp_ratio: usize,
    /// Norm placement relative to the sublayers.
    #[serde(default)]
    pub placement: NormPlacement,
    /// Positional-information kind.
    #[serde(default)]
    pub pos: PosKind,
    /// `Some(k)` = grouped-query attention with `k` KV heads
    /// (`heads % k == 0` required, checked at build where `heads` is
    /// known). `None` = MHA (reference). `Some(heads)` is
    /// shape-identical to MHA; `Some(1)` is MQA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<usize>,
    /// `Some(w)` = sliding-window causal mask (`w ≥ 1`): position `i`
    /// attends to `(i - w, i]`. `None` = full causal (reference).
    /// `w ≥ ctx` degenerates to full causal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    /// `true` = independent `lm_head.weight` Var. `false` (reference)
    /// = LM head tied to `wte`. (Design §4 sketched this as
    /// `tied_head: bool`; inverted so `Default` stays the reference on
    /// every axis.)
    #[serde(default)]
    pub untied_head: bool,
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

    /// Validate the invariants the builder assumes. Checks that need
    /// the surrounding [`super::gpt2::Gpt2Config`] (GQA divisibility,
    /// RoPE even head_dim) live in `Gpt2Model::new`.
    pub fn validate(&self) -> CandleResult<()> {
        // Phase 1 axes are closed enums; `mlp_ratio == 0` is
        // normalized rather than rejected so `Gpt2Custom::default()`
        // stays a valid spec.
        if self.kv_heads == Some(0) {
            return Err(candle_core::Error::Msg(
                "gpt2 custom: kv_heads must be ≥ 1 (use None for MHA)".into(),
            ));
        }
        if self.window == Some(0) {
            return Err(candle_core::Error::Msg(
                "gpt2 custom: window must be ≥ 1 (use None for full causal)".into(),
            ));
        }
        if self.placement == NormPlacement::PostLn && self.residual == ResidualKind::Parallel {
            return Err(candle_core::Error::Msg(
                "gpt2 custom: Post-LN with the parallel residual topology has no \
                 canonical wiring (the two norms cannot both sit on the single \
                 residual write); pick one axis"
                    .into(),
            ));
        }
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
        assert_eq!(c.placement, NormPlacement::PreLn);
        assert_eq!(c.pos, PosKind::Learned);
        assert_eq!(c.kv_heads, None);
        assert_eq!(c.window, None);
        assert!(!c.untied_head);
        c.validate().expect("default spec is valid");
    }

    #[test]
    fn validate_rejects_zero_knobs() {
        let kv0 = Gpt2Custom {
            kv_heads: Some(0),
            ..Default::default()
        };
        assert!(kv0.validate().unwrap_err().to_string().contains("kv_heads"));
        let w0 = Gpt2Custom {
            window: Some(0),
            ..Default::default()
        };
        assert!(w0.validate().unwrap_err().to_string().contains("window"));
    }

    #[test]
    fn validate_rejects_post_ln_parallel() {
        let c = Gpt2Custom {
            placement: NormPlacement::PostLn,
            residual: ResidualKind::Parallel,
            ..Default::default()
        };
        let msg = c.validate().unwrap_err().to_string();
        assert!(msg.contains("Post-LN"), "unexpected error: {msg}");
    }

    #[test]
    fn gated_flag() {
        assert!(Activation::SwiGlu.is_gated());
        assert!(Activation::GeGlu.is_gated());
        assert!(!Activation::Gelu.is_gated());
        assert!(!Activation::Relu.is_gated());
        assert!(!Activation::Silu.is_gated());
    }

    /// The serde vocabulary must equal the strings the Lua bridge
    /// accepts (`bridge/nn_card.rs` `parse_custom_spec`) — a Card
    /// records the spec so a caller reads back what they wrote, and a
    /// hand-authored Card table uses the documented spelling.
    #[test]
    fn serde_vocabulary_matches_lua_spelling() {
        let cases: &[(&str, serde_json::Value)] = &[
            ("gelu", serde_json::to_value(Activation::Gelu).unwrap()),
            ("relu", serde_json::to_value(Activation::Relu).unwrap()),
            ("silu", serde_json::to_value(Activation::Silu).unwrap()),
            ("swiglu", serde_json::to_value(Activation::SwiGlu).unwrap()),
            ("geglu", serde_json::to_value(Activation::GeGlu).unwrap()),
            (
                "layernorm",
                serde_json::to_value(NormKind::LayerNorm).unwrap(),
            ),
            ("rmsnorm", serde_json::to_value(NormKind::RmsNorm).unwrap()),
            (
                "sequential",
                serde_json::to_value(ResidualKind::Sequential).unwrap(),
            ),
            (
                "parallel",
                serde_json::to_value(ResidualKind::Parallel).unwrap(),
            ),
            ("preln", serde_json::to_value(NormPlacement::PreLn).unwrap()),
            (
                "postln",
                serde_json::to_value(NormPlacement::PostLn).unwrap(),
            ),
            ("learned", serde_json::to_value(PosKind::Learned).unwrap()),
            ("rope", serde_json::to_value(PosKind::Rope).unwrap()),
            ("alibi", serde_json::to_value(PosKind::Alibi).unwrap()),
            ("nope", serde_json::to_value(PosKind::NoPos).unwrap()),
        ];
        for (expected, actual) in cases {
            assert_eq!(actual, &serde_json::json!(expected));
        }
    }

    #[test]
    fn serde_roundtrips_non_default_spec() {
        let spec = Gpt2Custom {
            act: Activation::SwiGlu,
            norm: NormKind::RmsNorm,
            residual: ResidualKind::Parallel,
            mlp_ratio: 3,
            placement: NormPlacement::PreLn,
            pos: PosKind::Rope,
            kv_heads: Some(1),
            window: Some(4),
            untied_head: true,
        };
        let json = serde_json::to_value(&spec).expect("serialize");
        let back: Gpt2Custom = serde_json::from_value(json.clone()).expect("deserialize");
        assert_eq!(serde_json::to_value(&back).expect("re-serialize"), json);
        assert_eq!(back.act, Activation::SwiGlu);
        assert_eq!(back.pos, PosKind::Rope);
        assert_eq!(back.kv_heads, Some(1));
        assert_eq!(back.window, Some(4));
        assert!(back.untied_head);
    }

    /// An empty table deserializes to the reference spec, and absent
    /// `Option` axes are omitted rather than written as `null`.
    #[test]
    fn serde_defaults_and_omits_absent_options() {
        let spec: Gpt2Custom = serde_json::from_value(serde_json::json!({})).expect("deserialize");
        assert_eq!(spec.act, Activation::Gelu);
        assert_eq!(spec.pos, PosKind::Learned);
        assert_eq!(spec.kv_heads, None);
        assert_eq!(spec.window, None);

        let json = serde_json::to_value(&spec).expect("serialize");
        assert!(json.get("kv_heads").is_none(), "got: {json}");
        assert!(json.get("window").is_none(), "got: {json}");
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
