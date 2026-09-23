//! Parameter initialisation a seed decides.
//!
//! Two runs of the same config differ in two places: the order the rows
//! arrive in, and the numbers the parameters start at. The first is
//! [`DatasetOpts::seed`](crate::train::DatasetOpts::seed). This is the
//! second.
//!
//! # Why not `Device::set_seed`
//!
//! candle has one, and on CUDA and Metal it works. The CPU backend
//! answers `cannot seed the CPU rng with set_seed` and draws from the
//! thread RNG (`candle-core 0.11`, `cpu_backend/mod.rs`), so on the
//! device most development happens on there is nothing to seed. A run
//! whose reproducibility depended on the backend would be repeatable on
//! the GPU and not on the laptop, which is the worse of the two
//! failures: the discrepancy only shows up once the two are compared.
//!
//! # What this does instead
//!
//! [`seeded_var_builder`] hands the model a [`VarBuilder`] whose
//! backing store draws the initial values itself, from a seeded
//! [`StdRng`], honouring the [`Init`] hint each parameter was declared
//! with. The architectures are untouched: they already say what
//! distribution each parameter wants — `Randn { stdev: INIT_STDEV }`
//! for GPT-2's embeddings, Kaiming for a linear — and this reads that
//! declaration rather than restating it. A separate walker that re-drew
//! parameters after construction would be a second copy of every
//! architecture's initialisation, free to drift from the first.
//!
//! # What is still not deterministic
//!
//! Initialisation and row order, both fixed here, are what a caller
//! controls. They are not everything:
//!
//! - **Reduction order on the GPU.** Floating-point addition is not
//!   associative, and CUDA kernels do not promise a fixed summation
//!   order between runs. Two identically-seeded runs can diverge in the
//!   last bits and then, through a few thousand steps, visibly.
//! - **cuDNN algorithm selection**, which can vary with available
//!   memory.
//! - **Dropout and any other sampling inside a forward pass**, which
//!   draw from the device RNG this does not reach.
//!
//! This is the field's ordinary position rather than a shortfall
//! peculiar to this crate — PyTorch ships the same knobs and declines
//! the same guarantee ("completely reproducible results are not
//! guaranteed across PyTorch releases, individual commits, or different
//! platforms"). What matters is that the part a caller can control is
//! controllable, and that the rest is written down.

use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::init::NormalOrUniform;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder, VarMap};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A [`VarBuilder`] over `vm` whose fresh parameters are drawn from
/// `seed`.
///
/// Registration still goes through `vm`, so the caller holds the same
/// map it would have held, and everything downstream — the optimizer,
/// the checkpoint, `restore_into` — sees no difference. Only the
/// numbers a parameter starts at change, and only for parameters this
/// builder creates: a name the map already carries is returned as it
/// stands, which is what makes a partially-populated map (a resumed or
/// partially-loaded model) behave the same as before.
///
/// Draw order follows the order the architecture asks for its
/// parameters in, which is fixed by the code, so the same seed gives
/// the same model.
pub fn seeded_var_builder<'a>(
    vm: &VarMap,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> VarBuilder<'a> {
    let backend = SeededVarMap {
        // `VarMap` is `Arc`-backed, so this clone shares the caller's
        // map rather than shadowing it.
        vm: vm.clone(),
        rng: Mutex::new(StdRng::seed_from_u64(seed)),
    };
    VarBuilder::from_backend(Box::new(backend), dtype, device.clone())
}

/// A [`VarMap`] that draws its fresh values from an RNG it owns.
struct SeededVarMap {
    vm: VarMap,
    /// Behind a mutex because [`SimpleBackend`] is `Send + Sync` and
    /// `get` takes `&self`. Construction is single-threaded in every
    /// caller here, so the lock is never contended; it is what the
    /// trait bound costs, not a concurrency design.
    rng: Mutex<StdRng>,
}

impl SimpleBackend for SeededVarMap {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        if self.vm.contains_tensor(name) {
            // Already registered — a load path put it there, and the
            // hint is not consulted for an existing tensor (the same
            // rule `VarMap::get` follows).
            return self.vm.get(s, name, h, dtype, dev);
        }
        // Register through the map so the `Var` is the caller's, then
        // overwrite the value it was created with. `Init::Const(0.)`
        // rather than `h`, so the throwaway creation does not draw from
        // the device RNG this exists to bypass.
        let placeholder = self.vm.get(s.clone(), name, Init::Const(0.), dtype, dev)?;
        let values = self.draw(&s, &h, dtype, dev)?;
        let data = self
            .vm
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("seeded init: VarMap lock poisoned".into()))?;
        let var = data.get(name).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "seeded init: `{name}` vanished from the VarMap between registering and filling it"
            ))
        })?;
        var.set(&values)?;
        drop(data);
        // The tensor the caller gets has to be the one the `Var` now
        // holds, not the zeros handed back at registration.
        let _ = placeholder;
        Ok(values)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> CandleResult<Tensor> {
        SimpleBackend::get_unchecked(&self.vm, name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.vm.contains_tensor(name)
    }
}

impl SeededVarMap {
    /// The tensor `h` describes, drawn from this backend's RNG.
    ///
    /// Mirrors `Init::var` term for term — the same distributions, the
    /// same Kaiming bound and gain, read from candle's own
    /// `FanInOut::for_shape` and `NonLinearity::gain` so a change on
    /// that side does not quietly leave this behind.
    fn draw(&self, s: &Shape, h: &Init, dtype: DType, dev: &Device) -> CandleResult<Tensor> {
        match h {
            Init::Const(v) => Tensor::ones(s.clone(), DType::F32, dev)?
                .affine(*v, 0.)?
                .to_dtype(dtype),
            Init::Uniform { lo, up } => self.uniform(s, *lo, *up, dtype, dev),
            Init::Randn { mean, stdev } => self.normal(s, *mean, *stdev, dtype, dev),
            Init::Kaiming {
                dist,
                fan,
                non_linearity,
            } => {
                let fan = fan.for_shape(s);
                let std = non_linearity.gain() / (fan as f64).sqrt();
                match dist {
                    NormalOrUniform::Uniform => {
                        let bound = 3f64.sqrt() * std;
                        self.uniform(s, -bound, bound, dtype, dev)
                    }
                    NormalOrUniform::Normal => self.normal(s, 0., std, dtype, dev),
                }
            }
        }
    }

    /// `elem_count` draws from `U(lo, up)`.
    fn uniform(
        &self,
        s: &Shape,
        lo: f64,
        up: f64,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        let n = s.elem_count();
        let mut rng = self.lock_rng()?;
        let span = up - lo;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                let u: f64 = StandardUniform.sample(&mut *rng);
                (lo + u * span) as f32
            })
            .collect();
        Tensor::from_vec(values, s.clone(), dev)?.to_dtype(dtype)
    }

    /// `elem_count` draws from `N(mean, stdev)`, by Box–Muller.
    ///
    /// Written out rather than taken from `rand_distr`: the crate is not
    /// a dependency here, and the transform is three lines whose
    /// correctness is checkable by reading it.
    fn normal(
        &self,
        s: &Shape,
        mean: f64,
        stdev: f64,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        let n = s.elem_count();
        let mut rng = self.lock_rng()?;
        let mut values: Vec<f32> = Vec::with_capacity(n);
        while values.len() < n {
            // `u1` is drawn on (0, 1]: `ln(0)` is negative infinity,
            // and a single such draw would put an infinity in the
            // parameter.
            let raw: f64 = StandardUniform.sample(&mut *rng);
            let u1: f64 = 1.0 - raw;
            let u2: f64 = StandardUniform.sample(&mut *rng);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            values.push((mean + stdev * r * theta.cos()) as f32);
            if values.len() < n {
                values.push((mean + stdev * r * theta.sin()) as f32);
            }
        }
        Tensor::from_vec(values, s.clone(), dev)?.to_dtype(dtype)
    }

    fn lock_rng(&self) -> CandleResult<std::sync::MutexGuard<'_, StdRng>> {
        self.rng
            .lock()
            .map_err(|_| candle_core::Error::Msg("seeded init: RNG lock poisoned".into()))
    }
}

/// Names and shapes of everything a map holds, for a test that wants to
/// compare two builds parameter by parameter.
#[cfg(test)]
fn flat(vm: &VarMap) -> std::collections::HashMap<String, Vec<f32>> {
    let data = vm.data().lock().unwrap();
    data.iter()
        .map(|(name, var)| {
            let t = var
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            (name.clone(), t.to_vec1::<f32>().unwrap())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::Gpt2Config;
    use crate::arch::Gpt2Model;

    fn tiny_cfg() -> Gpt2Config {
        Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        }
    }

    fn build_with_seed(seed: u64) -> VarMap {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vs = seeded_var_builder(&vm, seed, cfg.dtype, &cfg.device);
        let _ = Gpt2Model::new(&cfg, vs).expect("build gpt2");
        vm
    }

    /// The point of the whole module: same seed, same model.
    #[test]
    fn the_same_seed_builds_the_same_parameters() {
        let a = flat(&build_with_seed(11));
        let b = flat(&build_with_seed(11));
        assert_eq!(a.len(), b.len());
        assert!(!a.is_empty(), "the model must register some parameters");
        for (name, values) in &a {
            assert_eq!(
                b.get(name),
                Some(values),
                "`{name}` differs between two builds under the same seed"
            );
        }
    }

    /// And a different seed is allowed to build a different one —
    /// otherwise the knob would be inert and the test above would pass
    /// on a constant.
    #[test]
    fn a_different_seed_builds_different_parameters() {
        let a = flat(&build_with_seed(11));
        let b = flat(&build_with_seed(12));
        assert!(
            a.iter().any(|(name, values)| b.get(name) != Some(values)),
            "two seeds produced identical parameters everywhere"
        );
    }

    /// The distributions the architecture asked for are the ones drawn:
    /// GPT-2's embeddings are declared `Randn { stdev: 0.02 }`, and a
    /// draw an order of magnitude wide would saturate the softmax at
    /// step 0 — the failure `INIT_STDEV` exists to prevent.
    #[test]
    fn the_declared_distribution_is_the_one_drawn() {
        let vm = build_with_seed(3);
        let values = flat(&vm);
        let wte = values
            .iter()
            .find(|(name, _)| name.contains("wte"))
            .map(|(_, v)| v.clone())
            .expect("the token embedding is registered");
        let n = wte.len() as f32;
        let mean = wte.iter().sum::<f32>() / n;
        let var = wte.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let stdev = var.sqrt();
        assert!(
            (stdev - crate::arch::gpt2::INIT_STDEV as f32).abs() < 0.01,
            "wte stdev {stdev} is not the declared {}",
            crate::arch::gpt2::INIT_STDEV
        );
        assert!(mean.abs() < 0.01, "wte mean {mean} is not near zero");
    }

    /// Norm weights are declared `Const(1.0)` and biases `Const(0.0)`;
    /// a seeded build must not turn either into a draw.
    #[test]
    fn constants_stay_constant_under_a_seed() {
        let vm = build_with_seed(5);
        for (name, values) in flat(&vm) {
            if name.ends_with(".bias") {
                assert!(
                    values.iter().all(|v| *v == 0.0),
                    "`{name}` must stay at zero"
                );
            }
        }
    }
}
