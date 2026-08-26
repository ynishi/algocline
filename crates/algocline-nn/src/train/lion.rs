//! Lion — the sign-momentum optimizer from *Symbolic Discovery of
//! Optimization Algorithms* (Chen et al., arXiv:2302.06675).
//!
//! # The update
//!
//! Writing `interp(x, y, a) = (1 - a)·x + a·y`, one step is
//!
//! ```text
//! update   = sign(interp(g, m, β₁))
//! w        = w - lr·update - lr·λ·w
//! m        = interp(g, m, β₂)
//! ```
//!
//! Two properties follow from the `sign`, and both matter to a caller
//! coming from AdamW:
//!
//! - **Every element moves by exactly `lr`.** There is no per-parameter
//!   scaling, so the learning rate is the step size rather than an
//!   upper bound on it. The paper's guidance is a learning rate 3–10×
//!   *smaller* than AdamW's and a decoupled weight decay 3–10× *larger*,
//!   so that the effective decay `lr·λ` lands in the same place.
//! - **One state tensor instead of two.** Lion keeps a momentum and no
//!   second moment, which is the memory argument for it.
//!
//! The momentum update reads the *pre-step* momentum, the same value
//! the update direction was computed from — not the one the step just
//! produced. Both lines above use `m` on the right-hand side, and
//! swapping their order silently changes the algorithm into something
//! with no published behaviour.
//!
//! # Precision
//!
//! An FP32 master copy is kept for parameters that are not already F32.
//! With a sign update every element moves by exactly `lr`, so a BF16
//! parameter whose ulp exceeds `lr` would round every step away and
//! stall at its initial value while the loss curve looked merely flat.
//! F32 parameters are updated in place — the master would be a copy of
//! the thing it mirrors.

use candle_core::backprop::GradStore;
use candle_core::{DType, Result as CandleResult, Tensor, Var};
use candle_nn::optim::Optimizer;

/// Lion's hyperparameters.
///
/// Defaults are the paper's: `β₁ = 0.9`, `β₂ = 0.99`. `lr` and
/// `weight_decay` have no defensible default — see the module docs on
/// the 3–10× relationship to AdamW — so they carry AdamW-shaped
/// placeholders that a caller is expected to replace.
#[derive(Debug, Clone, Copy)]
pub struct ParamsLion {
    /// Step size. Every element moves by exactly this much per step.
    pub lr: f64,
    /// Interpolation weight for the update direction.
    pub beta1: f64,
    /// Interpolation weight for the momentum EMA.
    pub beta2: f64,
    /// Decoupled weight decay `λ`, applied as `-lr·λ·w`.
    pub weight_decay: f64,
}

impl Default for ParamsLion {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// Per-parameter state: the live `Var`, its FP32 master when the live
/// one is not F32, and the momentum.
struct SlotLion {
    var: Var,
    /// FP32 master, or `None` when `var` is already F32 and is updated
    /// in place.
    master: Option<Tensor>,
    /// FP32 momentum (`m`).
    momentum: Tensor,
}

/// Lion over candle `Var`s.
///
/// Constructed through the [`Optimizer`] trait so the training loop can
/// hold it interchangeably with [`candle_nn::AdamW`] and
/// [`crate::train::mixed::MixedAdamW`].
pub struct Lion {
    slots: Vec<SlotLion>,
    params: ParamsLion,
}

impl Optimizer for Lion {
    type Config = ParamsLion;

    fn new(vars: Vec<Var>, params: ParamsLion) -> CandleResult<Self> {
        let slots = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                if var.dtype() == DType::F16 {
                    return Err(candle_core::Error::Msg(
                        "Lion: f16 parameters need loss scaling, which is not implemented \
                         — use bf16 (same exponent range as f32) or f32"
                            .into(),
                    ));
                }
                let momentum = Tensor::zeros(var.shape(), DType::F32, var.device())?;
                let master = if var.dtype() == DType::F32 {
                    None
                } else {
                    // `detach` cuts the master free of the autograd tape
                    // so holding it across steps retains no graph nodes.
                    Some(var.as_tensor().to_dtype(DType::F32)?.detach())
                };
                Ok(SlotLion {
                    var,
                    master,
                    momentum,
                })
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self { slots, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        let lr = self.params.lr;
        let lr_lambda = lr * self.params.weight_decay;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        for slot in self.slots.iter_mut() {
            let Some(g) = grads.get(slot.var.as_tensor()) else {
                continue;
            };
            let g = g.to_dtype(DType::F32)?.detach();

            // update = sign(interp(g, m, β₁)) — the momentum read here
            // is the pre-step one, which is also what the EMA below
            // reads.
            let blended = ((&slot.momentum * beta1)? + (&g * (1.0 - beta1))?)?;
            let update = sign(&blended)?;

            // w = w - lr·update - lr·λ·w, with the decay decoupled from
            // the gradient rather than folded into it.
            let current = match &slot.master {
                Some(master) => master.clone(),
                None => slot.var.as_tensor().detach(),
            };
            let next = (&current * (1f64 - lr_lambda))?;
            let next = (next - (update * lr)?)?;

            // m = interp(g, m, β₂), from the same pre-step momentum.
            let next_m = ((&slot.momentum * beta2)? + (&g * (1.0 - beta2))?)?;

            slot.var.set(&next.to_dtype(slot.var.dtype())?)?;
            if slot.master.is_some() {
                slot.master = Some(next.detach());
            }
            slot.momentum = next_m.detach();
        }
        Ok(())
    }
}

impl Lion {
    /// Current hyperparameters.
    pub fn params(&self) -> &ParamsLion {
        &self.params
    }
}

impl std::fmt::Debug for Lion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lion")
            .field("params", &self.params)
            .field("slots", &self.slots.len())
            .finish()
    }
}

/// Elementwise `sign`, as `(x > 0) - (x < 0)`.
///
/// candle 0.11 has no `sign` op, and the comparisons return `u8`, so
/// both sides are cast before the subtraction. Zero maps to zero, which
/// is what leaves a parameter with no gradient signal untouched by the
/// step rather than nudged in an arbitrary direction.
fn sign(x: &Tensor) -> CandleResult<Tensor> {
    let pos = x.gt(0f64)?.to_dtype(DType::F32)?;
    let neg = x.lt(0f64)?.to_dtype(DType::F32)?;
    pos - neg
}

#[cfg(test)]
mod tests {
    //! What is worth pinning is the algorithm's identity: the sign, the
    //! decoupled decay, and the fact that both right-hand sides read
    //! the same pre-step momentum. A transposition of the last two
    //! lines still trains — it just stops being Lion.

    use super::*;
    use candle_core::Device;

    fn var(values: &[f32]) -> Var {
        Var::from_slice(values, values.len(), &Device::Cpu).expect("var")
    }

    /// Drive one step with a chosen gradient.
    ///
    /// `GradStore` cannot be built directly (its constructor is private
    /// to candle), so the gradient is produced rather than inserted:
    /// `d/dw Σ(w · g) = g`, which makes the linear objective below an
    /// exact way to ask for one.
    fn step_with(opt: &mut Lion, v: &Var, grad: &[f32]) {
        let g = Tensor::from_slice(grad, grad.len(), &Device::Cpu).expect("grad");
        let loss = (v.as_tensor() * &g).expect("mul").sum_all().expect("sum");
        let grads = loss.backward().expect("backward");
        opt.step(&grads).expect("step");
    }

    fn values(v: &Var) -> Vec<f32> {
        v.as_tensor().to_vec1::<f32>().expect("to_vec1")
    }

    /// From a zero momentum, `interp(g, 0, β₁) = (1-β₁)·g`, whose sign
    /// is the sign of `g`. So the first step moves every element by
    /// exactly `lr` against its gradient, whatever the magnitudes.
    #[test]
    fn the_first_step_moves_every_element_by_exactly_lr() {
        let v = var(&[0.0, 0.0, 0.0, 0.0]);
        let mut opt = Lion::new(
            vec![v.clone()],
            ParamsLion {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .expect("lion");

        // Gradients spanning four orders of magnitude and both signs.
        step_with(&mut opt, &v, &[1000.0, 0.001, -1000.0, -0.001]);
        let got = values(&v);
        let want = [-0.1, -0.1, 0.1, 0.1];
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "element {i}: got {g}, want {w} — the step size is lr, not a function of |g|"
            );
        }
    }

    /// A zero gradient leaves the parameter alone rather than pushing
    /// it in whichever direction the sign of zero happened to take.
    #[test]
    fn a_zero_gradient_moves_nothing() {
        let v = var(&[0.5, -0.5]);
        let mut opt = Lion::new(
            vec![v.clone()],
            ParamsLion {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .expect("lion");
        step_with(&mut opt, &v, &[0.0, 0.0]);
        let got = values(&v);
        assert!((got[0] - 0.5).abs() < 1e-6, "{got:?}");
        assert!((got[1] + 0.5).abs() < 1e-6, "{got:?}");
    }

    /// Decoupled: with no gradient at all the parameter still shrinks
    /// by `lr·λ` of itself, which is what "decoupled" means and what
    /// separates it from folding `λ·w` into the gradient (where the
    /// sign would flatten it to a constant `lr` step).
    #[test]
    fn weight_decay_is_decoupled_from_the_sign() {
        let v = var(&[1.0, 100.0]);
        let mut opt = Lion::new(
            vec![v.clone()],
            ParamsLion {
                lr: 0.1,
                weight_decay: 0.5,
                ..Default::default()
            },
        )
        .expect("lion");
        step_with(&mut opt, &v, &[0.0, 0.0]);
        let got = values(&v);
        // w · (1 - lr·λ) = w · 0.95
        assert!((got[0] - 0.95).abs() < 1e-6, "{got:?}");
        assert!(
            (got[1] - 95.0).abs() < 1e-3,
            "{got:?} — the decay scales with w, so it is not a flat lr step"
        );
    }

    /// Both update lines read the pre-step momentum. Two steps with the
    /// same gradient make that checkable: after step 1 the momentum is
    /// `(1-β₂)·g`, so step 2's blend is
    /// `β₁·(1-β₂)·g + (1-β₁)·g`, which has the same sign as `g` — and
    /// the parameter moves by `lr` again rather than stalling.
    #[test]
    fn the_momentum_carries_between_steps() {
        let v = var(&[0.0]);
        let mut opt = Lion::new(
            vec![v.clone()],
            ParamsLion {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.0,
            },
        )
        .expect("lion");
        step_with(&mut opt, &v, &[1.0]);
        assert!((values(&v)[0] + 0.1).abs() < 1e-6);
        step_with(&mut opt, &v, &[1.0]);
        assert!((values(&v)[0] + 0.2).abs() < 1e-6, "{:?}", values(&v));

        // Two steps in, the EMA holds only `1 - β₂² ≈ 0.02` of the
        // gradient, so the fresh term (weight `1 - β₁ = 0.1`) is the
        // larger of the two and a flip flips the update. Momentum at
        // β₂ = 0.99 is a slow average, not a short one.
        step_with(&mut opt, &v, &[-1.0]);
        assert!(
            (values(&v)[0] + 0.1).abs() < 1e-6,
            "early on the fresh gradient wins, got {:?}",
            values(&v)
        );

        // Given enough steps for the EMA to fill, the same single flip
        // does not turn the update around: the blend is
        // `β₁·m + (1-β₁)·g ≈ 0.9·1 - 0.1 > 0`. That resistance is what
        // the momentum is for, and it is invisible until `m` is large.
        let v2 = var(&[0.0]);
        let mut opt2 = Lion::new(
            vec![v2.clone()],
            ParamsLion {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.0,
            },
        )
        .expect("lion");
        for _ in 0..500 {
            step_with(&mut opt2, &v2, &[1.0]);
        }
        let before = values(&v2)[0];
        step_with(&mut opt2, &v2, &[-1.0]);
        assert!(
            (values(&v2)[0] - (before - 0.1)).abs() < 1e-4,
            "a filled momentum must absorb one flipped gradient: {before} -> {:?}",
            values(&v2)
        );
    }

    /// The paper's defaults, pinned so a later edit to
    /// `Default::default` is a decision rather than a drift.
    #[test]
    fn the_defaults_are_the_papers() {
        let p = ParamsLion::default();
        assert!((p.beta1 - 0.9).abs() < f64::EPSILON);
        assert!((p.beta2 - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn f16_parameters_are_refused_by_name() {
        let v = Var::zeros(4, DType::F16, &Device::Cpu).expect("f16 var");
        let err = Lion::new(vec![v], ParamsLion::default()).expect_err("f16 must be refused");
        let text = err.to_string();
        assert!(text.contains("Lion") && text.contains("f16"), "{text}");
    }
}
