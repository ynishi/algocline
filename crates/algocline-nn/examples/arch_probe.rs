//! Generic architecture A/B probe for the GPT-2 customization axes.
//!
//! The composable counterpart of `init_loss_probe` (init scale) and
//! `moe_router_probe` (router behaviour): give it a list of
//! `(label, Gpt2Config)` arms and it trains each one on the same
//! 8-row cycled corpus for the same steps, over n independent draws,
//! and reports the CE loss min/median/max at fixed checkpoints plus
//! the trainable parameter count per arm (so param-count mismatches
//! between arms are visible instead of silently biasing the read).
//!
//! First measured pairs (Phase 1 of the custom-points issue):
//!
//!   - SwiGLU vs GELU — Shazeer 2020 (arXiv:2002.05202) reports GLU
//!     variants beating plain activations on transformer LM loss. The
//!     SwiGLU arm uses `mlp_ratio = 3` against the reference 4 to
//!     keep the parameter counts close (gated MLP carries 3 matrices
//!     per ratio unit; exact matching would need the paper's 2/3
//!     fractional ratio, which the integer knob does not express).
//!   - RMSNorm vs LayerNorm — Zhang & Sennrich 2019 (arXiv:1910.07467)
//!     reports parity-or-better quality at lower cost.
//!
//! Phase 2 pairs (position / norm-placement axes; design §5 initial
//! candidates):
//!
//!   - RoPE vs learned wpe — Su 2021 (arXiv:2104.09864). Param count
//!     drops by ctx·dim (no wpe Var).
//!   - NoPE vs learned wpe — Kazemnejad 2023 (arXiv:2305.19466)
//!     reports causal-mask-only decoders still learn order.
//!   - Post-LN vs Pre-LN — Xiong et al. 2020 (arXiv:2002.04745)
//!     predicts Post-LN trains worse without warmup; the instability
//!     itself is the observation target.
//!
//! As with the sibling probes, a claim that does not reproduce at
//! this tiny scale is recorded as-is, not massaged.
//!
//! Run: cargo run -p algocline-nn --example arch_probe --release

use algocline_nn::arch::{
    Activation, Gpt2Config, Gpt2Custom, Gpt2Model, NormKind, NormPlacement, PosKind,
};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const STEPS: usize = 120;
const DRAWS: usize = 5;
const ROWS: usize = 8;
const SEQ: usize = 32;

fn base_cfg(custom: Option<Gpt2Custom>) -> Gpt2Config {
    Gpt2Config {
        layers: 4,
        heads: 4,
        dim: 256,
        ctx: 64,
        vocab: 50257,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom,
    }
}

fn run_arm(cfg: &Gpt2Config) {
    let checkpoints = [0usize, 40, 80, STEPS - 1];
    let mut per_draw: Vec<Vec<f32>> = Vec::with_capacity(DRAWS);
    let mut param_count: usize = 0;

    for draw in 0..DRAWS {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(cfg, vb).expect("build model");
        param_count = vm
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum();

        // Same memorizable-but-not-in-one-step corpus as the sibling
        // probes.
        let corpus: Vec<Vec<u32>> = (0..ROWS)
            .map(|r| {
                (0..SEQ)
                    .map(|i| ((r * 7919 + i * 977) % cfg.vocab) as u32)
                    .collect()
            })
            .collect();

        let mut opt = AdamW::new(
            vm.all_vars(),
            ParamsAdamW {
                lr: 3e-4,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        let mut picked = Vec::with_capacity(checkpoints.len());
        for step in 0..STEPS {
            let ids = &corpus[step % ROWS];
            let inputs = Tensor::from_slice(&ids[..SEQ - 1], (1, SEQ - 1), &cfg.device).unwrap();
            let targets = Tensor::from_slice(&ids[1..SEQ], (1, SEQ - 1), &cfg.device).unwrap();
            let logits = model.forward(&inputs).unwrap();
            let loss = HardLabelDistillLoss::new()
                .compute(&logits, &targets, None)
                .unwrap();
            let val: f32 = loss.to_scalar().unwrap();
            if checkpoints.contains(&step) {
                picked.push(val);
            }
            opt.backward_step(&loss).unwrap();
        }

        let line: Vec<String> = checkpoints
            .iter()
            .zip(&picked)
            .map(|(s, v)| format!("s{s}={v:.4}"))
            .collect();
        println!("  draw {draw}: {}", line.join("  "));
        per_draw.push(picked);
    }

    println!("  params: {param_count}");
    println!("  min/median/max over {DRAWS} draws:");
    for (i, step) in checkpoints.iter().enumerate() {
        let mut vals: Vec<f32> = per_draw.iter().map(|d| d[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "    step {step:>3}: min={:.4} median={:.4} max={:.4}",
            vals[0],
            vals[vals.len() / 2],
            vals[vals.len() - 1]
        );
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "algocline_nn=info".into()),
        )
        .init();

    let arms: Vec<(&str, Gpt2Config)> = vec![
        ("baseline (GELU + LayerNorm, ratio 4)", base_cfg(None)),
        (
            "SwiGLU (ratio 3, ~param-matched)",
            base_cfg(Some(Gpt2Custom {
                act: Activation::SwiGlu,
                mlp_ratio: 3,
                ..Default::default()
            })),
        ),
        (
            "RMSNorm (GELU, ratio 4)",
            base_cfg(Some(Gpt2Custom {
                norm: NormKind::RmsNorm,
                mlp_ratio: 4,
                ..Default::default()
            })),
        ),
        (
            "RoPE (no wpe)",
            base_cfg(Some(Gpt2Custom {
                pos: PosKind::Rope,
                ..Default::default()
            })),
        ),
        (
            "NoPE (no position info)",
            base_cfg(Some(Gpt2Custom {
                pos: PosKind::NoPos,
                ..Default::default()
            })),
        ),
        (
            "Post-LN (learned wpe)",
            base_cfg(Some(Gpt2Custom {
                placement: NormPlacement::PostLn,
                ..Default::default()
            })),
        ),
    ];

    let c = &arms[0].1;
    println!(
        "arch probe: layers={} dim={} vocab={}, {ROWS} rows cycled, seq={SEQ}, \
         lr 3e-4, {STEPS} steps, {DRAWS} draws\n",
        c.layers, c.dim, c.vocab
    );

    for (label, cfg) in &arms {
        println!("--- {label}: {DRAWS} draws ---");
        run_arm(cfg);
        println!();
    }
}
