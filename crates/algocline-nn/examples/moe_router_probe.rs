//! Router-behaviour probe for the dense-MoE GPT-2 block.
//!
//! Backs the load-balancing aux loss with a measurement instead of a
//! citation — the same "probe it, don't quote it" discipline as
//! `init_loss_probe.rs` (INIT_STDEV) and the same n-draw + checkpoint
//! min/median/max reporting shape.
//!
//! A/B: `α = 0.01` (Switch Transformer §2.2 coefficient) vs `α = 0`
//! (no load-balancing pressure), n=5 independent draws each, on a tiny
//! MoE stack (2 layers / 4 experts / top-2 routing / dim 256 / CPU)
//! memorizing an 8-row cycled corpus. Observed per checkpoint:
//!
//!   - masked-free CE loss (same corpus recipe as the residual A/B)
//!   - expert utilization: fraction of tokens whose top-1 expert is i,
//!     averaged over layers — collapse concentrates this on one expert
//!   - routing entropy: mean `H(probs)` in nats (uniform over 4
//!     experts = ln 4 ≈ 1.386; collapse → 0)
//!   - the raw aux value `E · Σ f_i P_i` (uniform routing = 1.0)
//!
//! The paper claim under test: without the aux term the router is free
//! to collapse onto few experts; with it, utilization stays spread. If
//! the direction does not reproduce at this scale, that result is
//! recorded as-is (CHANGELOG), not massaged.
//!
//! Run: cargo run -p algocline-nn --example moe_router_probe --release

use algocline_nn::arch::{Gpt2Config, Gpt2Model, MoeConfig};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const N_EXPERTS: usize = 4;
const STEPS: usize = 120;
const DRAWS: usize = 5;
const ROWS: usize = 8;
const SEQ: usize = 32;

fn probe_cfg(alpha: f64) -> Gpt2Config {
    Gpt2Config {
        layers: 2,
        heads: 4,
        dim: 256,
        ctx: 64,
        vocab: 50257,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: Some(MoeConfig {
            alpha,
            ..MoeConfig::new(N_EXPERTS)
        }),
    }
}

/// Checkpoint observation, aggregated over the model's MoE layers.
struct Obs {
    ce: f32,
    aux: f32,
    entropy: f32,
    util: Vec<f32>,
}

impl Obs {
    /// Largest single-expert share of top-1 assignments — the collapse
    /// indicator (1/E when perfectly spread, → 1.0 when collapsed).
    fn max_share(&self) -> f32 {
        self.util.iter().cloned().fold(0.0, f32::max)
    }
}

/// Utilization + entropy from the per-layer router probabilities
/// (`[1, S, E]` each), averaged over layers.
fn router_stats(probs_per_layer: &[Tensor]) -> (f32, Vec<f32>) {
    let mut entropy_sum = 0.0f32;
    let mut util = vec![0.0f32; N_EXPERTS];
    let layers = probs_per_layer.len().max(1);
    for probs in probs_per_layer {
        let flat: Vec<Vec<f32>> = probs.squeeze(0).unwrap().to_vec2().unwrap(); // [S][E]
        let s = flat.len() as f32;
        for row in &flat {
            let h: f32 = -row
                .iter()
                .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
                .sum::<f32>();
            entropy_sum += h / s;
            let top1 = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            util[top1] += 1.0 / s;
        }
    }
    let entropy = entropy_sum / layers as f32;
    for u in util.iter_mut() {
        *u /= layers as f32;
    }
    (entropy, util)
}

fn run_arm(label: &str, alpha: f64) {
    let cfg = probe_cfg(alpha);
    let checkpoints = [0usize, 40, 80, STEPS - 1];
    println!("--- {label}: {DRAWS} draws ---");

    let mut per_draw: Vec<Vec<Obs>> = Vec::with_capacity(DRAWS);
    for draw in 0..DRAWS {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vb).expect("build moe model");

        // Same memorizable-but-not-in-one-step corpus as the residual
        // A/B in init_loss_probe.
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

        let mut picked: Vec<Obs> = Vec::with_capacity(checkpoints.len());
        for step in 0..STEPS {
            let ids = &corpus[step % ROWS];
            let inputs = Tensor::from_slice(&ids[..SEQ - 1], (1, SEQ - 1), &cfg.device).unwrap();
            let targets = Tensor::from_slice(&ids[1..SEQ], (1, SEQ - 1), &cfg.device).unwrap();

            let (logits, aux, probs) = model.forward_with_router_probs(&inputs).unwrap();
            let aux = aux.expect("MoE model returns aux");
            let ce = HardLabelDistillLoss::new()
                .compute(&logits, &targets, None)
                .unwrap();

            if checkpoints.contains(&step) {
                let (entropy, util) = router_stats(&probs);
                picked.push(Obs {
                    ce: ce.to_scalar().unwrap(),
                    aux: aux.to_scalar().unwrap(),
                    entropy,
                    util,
                });
            }

            let total = if alpha == 0.0 {
                ce
            } else {
                (ce + (aux * alpha).unwrap()).unwrap()
            };
            opt.backward_step(&total).unwrap();
        }

        let line: Vec<String> = checkpoints
            .iter()
            .zip(&picked)
            .map(|(s, o)| {
                format!(
                    "s{s}: ce={:.3} aux={:.3} H={:.3} max_share={:.2}",
                    o.ce,
                    o.aux,
                    o.entropy,
                    o.max_share()
                )
            })
            .collect();
        let last_util: Vec<String> = picked
            .last()
            .unwrap()
            .util
            .iter()
            .map(|u| format!("{u:.2}"))
            .collect();
        println!(
            "  draw {draw}: {}  final util=[{}]",
            line.join("  |  "),
            last_util.join(", ")
        );
        per_draw.push(picked);
    }

    println!("  min/median/max over {DRAWS} draws:");
    for (i, step) in checkpoints.iter().enumerate() {
        let summary = |f: &dyn Fn(&Obs) -> f32| -> (f32, f32, f32) {
            let mut vals: Vec<f32> = per_draw.iter().map(|d| f(&d[i])).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (vals[0], vals[vals.len() / 2], vals[vals.len() - 1])
        };
        let (ce_lo, ce_med, ce_hi) = summary(&|o: &Obs| o.ce);
        let (h_lo, h_med, h_hi) = summary(&|o: &Obs| o.entropy);
        let (ms_lo, ms_med, ms_hi) = summary(&|o: &Obs| o.max_share());
        println!(
            "    step {step:>3}: ce {ce_lo:.3}/{ce_med:.3}/{ce_hi:.3}  \
             H {h_lo:.3}/{h_med:.3}/{h_hi:.3}  max_share {ms_lo:.2}/{ms_med:.2}/{ms_hi:.2}"
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

    let cfg = probe_cfg(0.01);
    let moe = cfg.moe.as_ref().unwrap();
    println!(
        "moe router probe: layers={} dim={} experts={} top_k={} vocab={}, \
         {ROWS} rows cycled, seq={SEQ}, lr 3e-4, {STEPS} steps, {DRAWS} draws",
        cfg.layers, cfg.dim, moe.n_experts, moe.top_k, cfg.vocab
    );
    // The aux the model returns is summed over its MoE layers, so the
    // uniform-routing baseline is `layers × 1.0`.
    println!(
        "uniform entropy = ln({N_EXPERTS}) = {:.3}, uniform max_share = {:.2}, \
         uniform aux = {:.1} ({} layers × 1.0)\n",
        (N_EXPERTS as f32).ln(),
        1.0 / N_EXPERTS as f32,
        cfg.layers as f32,
        cfg.layers
    );

    run_arm("alpha = 0.01 (Switch §2.2)", 0.01);
    println!();
    run_arm("alpha = 0 (no load balancing)", 0.0);
}
