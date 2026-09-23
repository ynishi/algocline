//! Mixed-precision AdamW (design §7.1: BF16 weights / activations +
//! FP32 optimizer states).
//!
//! candle-nn's stock [`candle_nn::AdamW`] keeps its first/second
//! moments in the *parameter's* dtype, so handing it BF16 `Var`s
//! silently runs the whole optimizer state in BF16 — the classic
//! mixed-precision failure mode where small updates round to zero at
//! the parameter's magnitude and training stalls without erroring.
//! [`MixedAdamW`] implements the standard master-weights recipe
//! instead:
//!
//! - the model's `Var`s stay in their low precision (BF16) and keep
//!   driving forward / backward,
//! - a private FP32 master copy of every parameter plus FP32 moment
//!   tensors receive the AdamW update (gradients are upcast per step),
//! - the updated master is cast back down and written into the `Var`
//!   so the next forward sees the new weights.
//!
//! The update math mirrors `candle_nn::AdamW::step` term for term
//! (decoupled weight decay applied as `θ · (1 − lr·λ)`, bias-corrected
//! moments) so the FP32-on-FP32 special case is numerically identical
//! to the stock optimizer — pinned by the parity test below.
//!
//! F16 is deliberately not accepted: its 5-bit exponent needs loss
//! scaling to keep gradients from flushing to zero, and no scaler
//! ships here. BF16 shares FP32's 8-bit exponent, which is why it
//! trains without a scaler.

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{DType, Result as CandleResult, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use candle_nn::ParamsAdamW;

use super::optstate::{param_name, slot_tensor};

/// Per-parameter optimizer state: the live low-precision `Var` plus
/// its FP32 master / moment tensors.
struct SlotAdamW {
    /// The model parameter (BF16 in the mixed path). Forward /
    /// backward run against this; the optimizer writes the downcast
    /// master back into it after every step.
    var: Var,
    /// FP32 master copy of the parameter — the value AdamW actually
    /// updates, so sub-BF16-ulp updates accumulate instead of
    /// rounding away.
    master: Tensor,
    /// FP32 first moment (`m`).
    first_moment: Tensor,
    /// FP32 second moment (`v`).
    second_moment: Tensor,
}

/// AdamW with FP32 master weights over low-precision parameters.
///
/// Constructed through the [`Optimizer`] trait (`Optimizer::new`) so
/// the training loop can hold it interchangeably with the stock
/// [`candle_nn::AdamW`].
pub struct MixedAdamW {
    slots: Vec<SlotAdamW>,
    params: ParamsAdamW,
    step_t: usize,
}

impl Optimizer for MixedAdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> CandleResult<Self> {
        let slots = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                if var.dtype() == DType::F16 {
                    return Err(candle_core::Error::Msg(
                        "MixedAdamW: f16 parameters need loss scaling, which is not \
                         implemented — use bf16 (same exponent range as f32) or f32"
                            .into(),
                    ));
                }
                let shape = var.shape();
                let device = var.device();
                // `detach` cuts the master free of the autograd tape so
                // holding it across steps does not retain graph nodes.
                let master = var.as_tensor().to_dtype(DType::F32)?.detach();
                let first_moment = Tensor::zeros(shape, DType::F32, device)?;
                let second_moment = Tensor::zeros(shape, DType::F32, device)?;
                Ok(SlotAdamW {
                    var,
                    master,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            slots,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for slot in self.slots.iter_mut() {
            let Some(g) = grads.get(slot.var.as_tensor()) else {
                continue;
            };
            // Upcast the gradient once; every term below is FP32.
            let g = g.to_dtype(DType::F32)?.detach();
            let next_m = ((&slot.first_moment * beta1)? + (&g * (1.0 - beta1))?)?;
            let next_v = ((&slot.second_moment * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let next_master = (&slot.master * (1f64 - lr_lambda))?;
            let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
            let next_master = (next_master - (adjusted_grad * lr)?)?;
            // Write the downcast master into the live parameter, then
            // persist the FP32 state — `detach` on every stored tensor
            // keeps step-to-step op chains from accumulating.
            slot.var.set(&next_master.to_dtype(slot.var.dtype())?)?;
            slot.master = next_master.detach();
            slot.first_moment = next_m.detach();
            slot.second_moment = next_v.detach();
        }
        Ok(())
    }
}

impl MixedAdamW {
    /// Current AdamW hyperparameters.
    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    /// Steps applied so far. The bias corrections read it, so a resume
    /// that restored the moments but not this would apply a correction
    /// for step 1 to moments that are hundreds of steps old.
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    /// Write every slot's master and moments into `out`, keyed
    /// `<parameter name>.{master,m,v}`.
    ///
    /// # Errors
    ///
    /// When a parameter the optimizer holds is absent from `names` —
    /// its state would have no name to be stored under, and skipping it
    /// would produce a file that restores into a partly-fresh
    /// optimizer.
    pub fn write_state(
        &self,
        names: &HashMap<TensorId, String>,
        out: &mut HashMap<String, Tensor>,
    ) -> Result<(), String> {
        for slot in &self.slots {
            let name = param_name(names, slot.var.as_tensor().id())?;
            out.insert(format!("{name}.master"), slot.master.clone());
            out.insert(format!("{name}.m"), slot.first_moment.clone());
            out.insert(format!("{name}.v"), slot.second_moment.clone());
        }
        Ok(())
    }

    /// Read the slots back out of a loaded state, and set the step
    /// count the bias corrections read.
    ///
    /// All or nothing: the first slot that is missing or shaped wrong
    /// ends the restore with an error and the optimizer is left
    /// untouched, because a half-restored optimizer trains and reports
    /// a loss exactly like a working one.
    pub fn read_state(
        &mut self,
        names: &HashMap<TensorId, String>,
        src: &HashMap<String, Tensor>,
        step: usize,
    ) -> Result<(), String> {
        // Built to completion before anything is written, so a refusal
        // leaves the live optimizer as it was.
        let mut staged = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let name = param_name(names, slot.var.as_tensor().id())?;
            staged.push((
                slot_tensor(src, name, "master", &slot.master)?,
                slot_tensor(src, name, "m", &slot.first_moment)?,
                slot_tensor(src, name, "v", &slot.second_moment)?,
            ));
        }
        for (slot, (master, m, v)) in self.slots.iter_mut().zip(staged) {
            slot.master = master;
            slot.first_moment = m;
            slot.second_moment = v;
        }
        self.step_t = step;
        Ok(())
    }
}

impl std::fmt::Debug for MixedAdamW {
    // Manual impl: slots hold `Var` / `Tensor`, whose Debug output
    // would dump values; the useful state is the size and progress.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixedAdamW")
            .field("slots", &self.slots.len())
            .field("step_t", &self.step_t)
            .field("lr", &self.params.lr)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::AdamW;

    fn quadratic_grad(var: &Var, target: &Tensor) -> CandleResult<GradStore> {
        // loss = mean((x - target)^2) — differentiable toy objective.
        let diff = (var.as_tensor() - target)?;
        let loss = diff.sqr()?.mean_all()?;
        loss.backward()
    }

    /// FP32-on-FP32, MixedAdamW must match candle-nn's AdamW step for
    /// step: same seed values, same grads, same hyperparams.
    #[test]
    fn f32_parity_with_stock_adamw() {
        let dev = Device::Cpu;
        let init = vec![0.5f32, -1.25, 2.0, 0.03125];
        let target = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &dev).unwrap();

        let var_stock = Var::new(init.clone(), &dev).unwrap();
        let var_mixed = Var::new(init, &dev).unwrap();
        let params = ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut stock = AdamW::new(vec![var_stock.clone()], params.clone()).unwrap();
        let mut mixed = MixedAdamW::new(vec![var_mixed.clone()], params).unwrap();

        for _ in 0..5 {
            let g_stock = quadratic_grad(&var_stock, &target).unwrap();
            let g_mixed = quadratic_grad(&var_mixed, &target).unwrap();
            stock.step(&g_stock).unwrap();
            mixed.step(&g_mixed).unwrap();
        }

        let a: Vec<f32> = var_stock.as_tensor().to_vec1().unwrap();
        let b: Vec<f32> = var_mixed.as_tensor().to_vec1().unwrap();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "element {i} diverged: stock={x} mixed={y}"
            );
        }
    }

    /// A restored optimizer takes the same next step the original would
    /// have. This is the whole claim behind saving the state: without
    /// the moments and the step count, the step after a restart is the
    /// step of a fresh run.
    #[test]
    fn a_restored_optimizer_continues_the_step_the_original_would_have_taken() {
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.05,
            weight_decay: 0.1,
            ..Default::default()
        };
        let target = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &dev).unwrap();
        let init = vec![0.5f32, -1.25, 2.0, 0.03125];

        // One optimizer runs five steps and keeps going; the other runs
        // the same five, is saved, and is replaced by a fresh optimizer
        // that reads the state back.
        let vm_a = VarMap::new();
        let _ = vm_a
            .get(4, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var_a = vm_a.data().lock().unwrap()["w"].clone();
        var_a
            .set(&Tensor::new(init.clone(), &dev).unwrap())
            .unwrap();

        let vm_b = VarMap::new();
        let _ = vm_b
            .get(4, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var_b = vm_b.data().lock().unwrap()["w"].clone();
        var_b.set(&Tensor::new(init, &dev).unwrap()).unwrap();

        let mut a = MixedAdamW::new(vec![var_a.clone()], params.clone()).unwrap();
        let mut b = MixedAdamW::new(vec![var_b.clone()], params.clone()).unwrap();
        for _ in 0..5 {
            a.step(&quadratic_grad(&var_a, &target).unwrap()).unwrap();
            b.step(&quadratic_grad(&var_b, &target).unwrap()).unwrap();
        }

        let names = crate::train::optstate::names_by_tensor_id(&vm_b);
        let mut tensors = HashMap::new();
        b.write_state(&names, &mut tensors).unwrap();
        let saved_step = b.step_count();
        assert_eq!(saved_step, 5);
        drop(b);

        let mut restored = MixedAdamW::new(vec![var_b.clone()], params).unwrap();
        restored.read_state(&names, &tensors, saved_step).unwrap();

        a.step(&quadratic_grad(&var_a, &target).unwrap()).unwrap();
        restored
            .step(&quadratic_grad(&var_b, &target).unwrap())
            .unwrap();

        let after_a: Vec<f32> = var_a.as_tensor().to_vec1().unwrap();
        let after_b: Vec<f32> = var_b.as_tensor().to_vec1().unwrap();
        for (i, (x, y)) in after_a.iter().zip(after_b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-7,
                "element {i}: uninterrupted={x} restored={y}"
            );
        }
    }

    /// A state that does not cover every parameter is refused, and the
    /// optimizer it was read into is left as it was.
    #[test]
    fn an_incomplete_state_is_refused_and_changes_nothing() {
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let _ = vm
            .get(2, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var = vm.data().lock().unwrap()["w"].clone();
        var.set(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap())
            .unwrap();
        let mut opt = MixedAdamW::new(vec![var.clone()], ParamsAdamW::default()).unwrap();
        let target = Tensor::zeros(2, DType::F32, &dev).unwrap();
        opt.step(&quadratic_grad(&var, &target).unwrap()).unwrap();
        let before: Vec<f32> = opt.slots[0].first_moment.to_vec1().unwrap();

        let names = crate::train::optstate::names_by_tensor_id(&vm);
        let mut partial = HashMap::new();
        opt.write_state(&names, &mut partial).unwrap();
        partial.remove("w.v");

        let err = opt.read_state(&names, &partial, 1).unwrap_err();
        assert!(err.contains("w.v"), "message: {err}");
        let after: Vec<f32> = opt.slots[0].first_moment.to_vec1().unwrap();
        assert_eq!(before, after, "a refused restore must not have written");
    }

    /// BF16 parameters keep their dtype, and repeated small updates    /// BF16 parameters keep their dtype, and repeated small updates
    /// accumulate through the FP32 master instead of rounding away at
    /// BF16 precision.
    #[test]
    fn bf16_params_learn_through_f32_master() {
        let dev = Device::Cpu;
        let var = Var::from_tensor(
            &Tensor::new(&[1.0f32, -2.0, 0.5], &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        )
        .unwrap();
        let target = Tensor::zeros(3, DType::BF16, &dev).unwrap();
        let params = ParamsAdamW {
            lr: 1e-2,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut opt = MixedAdamW::new(vec![var.clone()], params).unwrap();

        let loss_at = |v: &Var| -> f32 {
            (v.as_tensor() - &target)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_scalar()
                .unwrap()
        };
        let initial = loss_at(&var);
        for _ in 0..100 {
            let grads = quadratic_grad(&var, &target).unwrap();
            opt.step(&grads).unwrap();
        }
        let trained = loss_at(&var);
        assert_eq!(var.dtype(), DType::BF16, "params must stay bf16");
        assert!(
            trained < initial * 0.5,
            "loss must decrease: initial={initial} trained={trained}"
        );

        // Master retains sub-BF16 precision: casting it down and back
        // up loses information, so master ≠ round-trip unless the run
        // happened to land exactly on BF16 grid points for every
        // element — with 100 AdamW steps at lr 1e-2 that never happens.
        let slot = &opt.slots[0];
        let roundtrip = slot
            .master
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let max_gap: f32 = (&slot.master - &roundtrip)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            max_gap > 0.0,
            "master should carry precision below the bf16 grid"
        );
    }

    /// F16 parameters are refused at construction with a directional
    /// message (no loss scaler ships here).
    #[test]
    fn f16_params_refused_at_construction() {
        let dev = Device::Cpu;
        let var = Var::from_tensor(
            &Tensor::new(&[1.0f32], &dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap(),
        )
        .unwrap();
        let err = MixedAdamW::new(vec![var], ParamsAdamW::default()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("f16") && msg.contains("loss scaling"),
            "unexpected error: {msg}"
        );
    }
}
