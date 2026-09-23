//! The forward and backward of a gradient-checkpointed step.
//!
//! See [`crate::arch::blockwise`] for what checkpointing is and which
//! models can be driven this way. This module is the other half: the
//! two-phase pass the training loop runs in place of
//! `forward → loss → backward`.
//!
//! # The two phases
//!
//! **Forward.** Embed, then walk the blocks, detaching each block's
//! output before it becomes the next block's input. The detach is what
//! frees the memory: a detached tensor carries no operation, so the
//! graph that produced it — every intermediate inside that block —
//! becomes unreachable and is dropped. What is kept is one
//! `[batch, seq, dim]` tensor per block, the inputs, which is the
//! checkpoint. The head and the loss run on the last one, tracked
//! normally.
//!
//! **Backward.** Run the loss's own backward, which reaches the head's
//! parameters and stops at the last checkpoint — leaving, in the grad
//! store, the gradient with respect to that checkpoint. Then walk the
//! blocks backwards: re-run one block on its stored input (rebuilding
//! exactly the intermediates the forward threw away), and push the
//! incoming gradient through it.
//!
//! # Pushing a gradient into a sub-graph
//!
//! candle's `Tensor::backward` seeds the graph with ones, so it answers
//! `∂(Σ y)/∂θ` and there is no argument for "start from this gradient
//! instead". The surrogate `Σ(y ⊙ g)` supplies it:
//!
//! ```text
//! ∂/∂θ Σᵢ yᵢ·gᵢ  =  Σᵢ gᵢ · ∂yᵢ/∂θ  =  Jᵀg
//! ```
//!
//! which is exactly the chained gradient, for `g` treated as a
//! constant — and it is, having come out of a grad store already
//! detached. The same backward also writes `∂/∂x Σ(y ⊙ g)` for the
//! block's input `x`, which is the gradient the next block down
//! receives.
//!
//! # Why the checkpoint is re-entered as a `Var`
//!
//! Because a plain detached tensor does not work, and the difference is
//! silent. candle's graph walk marks a node as tracking gradients only
//! if a `Var` is reachable below it, and skips the rest entirely — so
//! every operation computed from a non-`Var` leaf alone is never
//! visited by the backward pass. A gradient does land on such a leaf
//! when its immediate consumer is tracked, which makes the arrangement
//! look workable on a one-operation example; it stops there. Measured:
//! `layer_norm_slow` over a non-`Var` input leaves that input with no
//! gradient at all, because the mean and variance it computes first
//! depend on nothing else.
//!
//! Each checkpoint is therefore re-entered as a temporary
//! [`Var`](candle_core::Var) during the backward phase. A `Var` is a
//! tracked leaf, so the block's recomputation tracks gradients
//! throughout and its own gradient — the one the next block down
//! receives — is written where it can be read. The `Var` lives for one
//! block's recomputation and is dropped with it; it is never registered
//! in any `VarMap` and no optimizer ever sees it.
//!
//! Both halves of that are asserted in this module's tests rather than
//! inferred, because a checkpointed run whose gradients quietly stop
//! partway produces a loss curve that looks like a slow one.

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Result as CandleResult, Tensor, TensorId, Var};
use candle_nn::VarMap;

use crate::arch::blockwise::Checkpointable;

/// Run one checkpointed step and return `(loss value, gradients)`.
///
/// `loss_of` turns the logits into the scalar loss — supplied by the
/// caller rather than taken as a `Loss` + targets, so the training
/// loop keeps its own arrangement of masks and target shifts in one
/// place.
///
/// `scale` is the caller's pre-backward factor (the `1 / grad_accum`
/// of the accumulating path). It multiplies the loss before any
/// backward runs, so every gradient this returns is already scaled —
/// the same arithmetic the uncheckpointed path performs, in the same
/// order.
///
/// The returned store holds gradients for the model's parameters. It
/// also holds gradients for the checkpoints themselves, which no
/// optimizer reads and which are dropped with the store.
pub fn checkpointed_step<F>(
    model: &dyn Checkpointable,
    xs: &Tensor,
    loss_of: F,
    scale: f64,
) -> CandleResult<(f32, GradStore)>
where
    F: FnOnce(&Tensor) -> CandleResult<Tensor>,
{
    model.checkpointable().map_err(candle_core::Error::Msg)?;

    // ── forward, keeping only the block inputs ──────────────────────
    let blocks = model.block_count();
    let mut inputs: Vec<Tensor> = Vec::with_capacity(blocks);
    let mut h = model.embed_input(xs)?;
    for index in 0..blocks {
        // Detached, so the block's own graph is the only thing the
        // recomputation needs and the forward's is dropped.
        let input = h.detach();
        inputs.push(input.clone());
        h = model.block_forward(index, &input)?.detach();
    }

    // The head and the loss keep their graph: it is one block's worth
    // of intermediates and the backward starts here. The last
    // checkpoint enters it as a `Var` for the reason in the module
    // doc — as a plain tensor the head's backward would not reach it.
    let last = Var::from_tensor(&h)?;
    let logits = model.head_forward(last.as_tensor())?;
    let loss = loss_of(&logits)?;
    let loss_value: f32 = loss.to_scalar()?;
    let scaled = (&loss * scale)?;
    let mut grads = scaled.backward()?;

    // ── backward, recomputing one block at a time ───────────────────
    // The gradient flowing into the last checkpoint, written by the
    // head's backward.
    let mut incoming = grads.get(last.as_tensor()).cloned().ok_or_else(|| {
        candle_core::Error::Msg(
            "checkpointed_step: the head's backward left no gradient on the last checkpoint; \
             the loss does not depend on the model's output"
                .into(),
        )
    })?;

    for index in (0..blocks).rev() {
        // A temporary tracked leaf holding the stored input. Dropped at
        // the end of this iteration; never registered anywhere.
        let input = Var::from_tensor(&inputs[index])?;
        let recomputed = model.block_forward(index, input.as_tensor())?;
        // Σ(y ⊙ g) — see the module doc. `g` is detached already, so
        // the surrogate's graph is the block's alone.
        let surrogate = recomputed.mul(&incoming)?.sum_all()?;
        let block_grads = surrogate.backward()?;
        incoming = block_grads.get(input.as_tensor()).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "checkpointed_step: block {index} left no gradient on its input; the block \
                 does not read it"
            ))
        })?;
        grads.extend(block_grads)?;
    }

    // The embedding is below the first checkpoint and was not part of
    // any block's graph, so it gets its own pass with the gradient that
    // came out of block 0.
    let embedded = model.embed_input(xs)?;
    let surrogate = embedded.mul(&incoming)?.sum_all()?;
    grads.extend(surrogate.backward()?)?;

    Ok((loss_value, grads))
}

/// How many of `vm`'s variables `grads` holds a gradient for.
///
/// Exists because [`max_grad_gap`] answers `0.0` when the two stores
/// are *both* empty, which is a comparison that passes without
/// comparing anything — and did, silently, while a bug upstream was
/// producing no gradients at all. A test asserting the gap must assert
/// this too.
pub fn grad_coverage(vm: &VarMap, grads: &GradStore) -> usize {
    let names = crate::train::optstate::names_by_tensor_id(vm);
    names
        .keys()
        .filter(|id| grads.get_id(**id).is_some())
        .count()
}

/// Sum of `|a - b|`'s maximum over every variable both stores hold, for
/// a test comparing a checkpointed step against an ordinary one.
///
/// Answers `0.0` for two stores that hold nothing, so a caller has to
/// pair it with [`grad_coverage`]; the pair is what makes "the two
/// agree" mean something.
///
/// Lives here rather than in the test module because both this crate's
/// tests and a downstream one want the same comparison, and a second
/// copy would be free to disagree about which variables count.
pub fn max_grad_gap(vm: &VarMap, a: &GradStore, b: &GradStore) -> CandleResult<f32> {
    let names: HashMap<TensorId, String> = crate::train::optstate::names_by_tensor_id(vm);
    let mut worst = 0.0f32;
    for id in names.keys() {
        match (a.get_id(*id), b.get_id(*id)) {
            (Some(ga), Some(gb)) => {
                let gap: f32 = (ga - gb)?
                    .abs()?
                    .max_all()?
                    .to_dtype(candle_core::DType::F32)?
                    .to_scalar()?;
                worst = worst.max(gap);
            }
            (None, None) => {}
            // One store has a gradient the other does not: that is a
            // whole parameter's worth of difference, not a numeric one.
            _ => return Ok(f32::INFINITY),
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    /// The trap this module is built around, pinned from both sides.
    ///
    /// A gradient *does* land on a non-`Var` operand when its immediate
    /// consumer is tracked — which is what makes a plain detached
    /// checkpoint look workable on a one-operation example. It stops
    /// there: anything computed from that leaf alone is never visited
    /// by the backward walk, so a real block leaves it with nothing.
    ///
    /// Asserted against candle directly, because the day the second
    /// half starts holding, the `Var` round-trip below becomes
    /// unnecessary — and the day the first half stops, a checkpointed
    /// run would return silently truncated gradients instead of
    /// failing.
    #[test]
    fn a_non_var_leaf_gets_a_gradient_one_operation_deep_and_no_further() {
        let dev = Device::Cpu;
        let w = Var::new(&[2.0f32, 3.0], &dev).unwrap();
        let x = Tensor::new(&[5.0f32, 7.0], &dev).unwrap(); // not a Var
        let y = w.as_tensor().mul(&x).unwrap();
        let grads = y.sum_all().unwrap().backward().unwrap();

        let dw: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        assert_eq!(dw, vec![5.0, 7.0], "∂Σ(w·x)/∂w = x");
        let dx: Vec<f32> = grads
            .get(&x)
            .expect("the immediate consumer writes it")
            .to_vec1()
            .unwrap();
        assert_eq!(dx, vec![2.0, 3.0], "∂Σ(w·x)/∂x = w");

        // Now with operations that read the leaf and nothing else
        // first — a normalisation's mean and variance. Those are never
        // walked, so the chain back to the leaf is not walked either.
        let leaf = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &dev).unwrap();
        let alpha = Var::new(&[1.0f32, 1.0, 1.0, 1.0], &dev).unwrap();
        let beta = Var::new(&[0.0f32, 0.0, 0.0, 0.0], &dev).unwrap();
        let normed =
            candle_nn::ops::layer_norm_slow(&leaf, alpha.as_tensor(), beta.as_tensor(), 1e-5)
                .unwrap();
        let grads = normed.sum_all().unwrap().backward().unwrap();
        assert!(
            grads.get(alpha.as_tensor()).is_some(),
            "the parameters are reached"
        );
        assert!(
            grads.get(&leaf).is_none(),
            "and the leaf is not — which is why a checkpoint is re-entered as a Var"
        );

        // The same leaf as a `Var` is reached.
        let leaf = Var::from_tensor(&leaf).unwrap();
        let normed = candle_nn::ops::layer_norm_slow(
            leaf.as_tensor(),
            alpha.as_tensor(),
            beta.as_tensor(),
            1e-5,
        )
        .unwrap();
        let grads = normed.sum_all().unwrap().backward().unwrap();
        assert!(
            grads.get(leaf.as_tensor()).is_some(),
            "a tracked leaf receives the gradient the block has to pass down"
        );
    }

    /// And the surrogate injects an incoming gradient: `Σ(y ⊙ g)`
    /// backwards to `Jᵀg` rather than to `Jᵀ1`.
    #[test]
    fn the_surrogate_pushes_the_gradient_it_was_given() {
        let dev = Device::Cpu;
        let w = Var::new(&[2.0f32, 3.0], &dev).unwrap();
        let x = Tensor::new(&[5.0f32, 7.0], &dev).unwrap();
        let g = Tensor::new(&[10.0f32, 100.0], &dev).unwrap();

        let y = w.as_tensor().mul(&x).unwrap();
        let grads = y.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
        let dw: Vec<f32> = grads.get(w.as_tensor()).unwrap().to_vec1().unwrap();
        assert_eq!(dw, vec![50.0, 700.0], "∂Σ(w·x·g)/∂w = x·g");
    }

    /// The claim a checkpointed run rests on: it is the same run. Every
    /// parameter's gradient has to match the ordinary backward's, and
    /// the loss has to be the same number.
    #[test]
    fn a_checkpointed_step_produces_the_gradients_the_ordinary_one_does() {
        use crate::arch::gpt2::Gpt2Config;
        use crate::arch::{seeded_var_builder, Gpt2Model};
        use crate::train::loss::{CrossEntropyLoss, Loss};

        let cfg = Gpt2Config {
            layers: 3,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vs = seeded_var_builder(&vm, 7788, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();

        let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let inputs = Tensor::from_slice(&row[..5], (1, 5), &cfg.device).unwrap();
        let targets = Tensor::from_slice(&row[1..], (1, 5), &cfg.device).unwrap();
        let ce = CrossEntropyLoss::new();
        // A scale other than 1, so a path that forgot to apply it is
        // caught rather than coinciding.
        let scale = 0.5f64;

        let logits = model.forward(&inputs).unwrap();
        let plain_loss = ce.compute(&logits, &targets, None).unwrap();
        let plain_value: f32 = plain_loss.to_scalar().unwrap();
        let plain_grads = (&plain_loss * scale).unwrap().backward().unwrap();

        let (value, ckpt_grads) =
            checkpointed_step(&model, &inputs, |l| ce.compute(l, &targets, None), scale).unwrap();

        assert!(
            (value - plain_value).abs() < 1e-6,
            "loss differs: {value} vs {plain_value}"
        );
        // Both paths must actually have produced gradients: comparing
        // two empty stores agrees perfectly and proves nothing.
        let parameters = vm.data().lock().unwrap().len();
        assert_eq!(
            grad_coverage(&vm, &plain_grads),
            parameters,
            "the ordinary backward left some parameter without a gradient"
        );
        assert_eq!(
            grad_coverage(&vm, &ckpt_grads),
            parameters,
            "the checkpointed backward left some parameter without a gradient"
        );
        let gap = max_grad_gap(&vm, &plain_grads, &ckpt_grads).unwrap();
        assert!(
            gap.is_finite(),
            "one path produced a gradient for a parameter the other did not"
        );
        assert!(gap < 1e-5, "gradients diverged by {gap}");
    }

    /// The blockwise decomposition is the forward: embed → blocks →
    /// head has to equal `forward`, or the recomputation would be of a
    /// different model.
    #[test]
    fn walking_the_blocks_reproduces_the_ordinary_forward() {
        use crate::arch::gpt2::Gpt2Config;
        use crate::arch::{seeded_var_builder, Gpt2Model};

        let cfg = Gpt2Config {
            layers: 3,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vs = seeded_var_builder(&vm, 31, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &cfg.device).unwrap();

        let mut h = model.embed_input(&ids).unwrap();
        for i in 0..model.block_count() {
            h = model.block_forward(i, &h).unwrap();
        }
        let walked = model.head_forward(&h).unwrap();
        let direct = model.forward(&ids).unwrap();
        let gap: f32 = (walked - direct)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            gap < 1e-6,
            "the blockwise walk diverged from forward by {gap}"
        );
    }

    /// A model that cannot be driven blockwise is refused before the
    /// forward runs, not answered with a differently-trained run.
    #[test]
    fn a_model_that_refuses_checkpointing_is_not_run() {
        struct Refuses;
        impl Checkpointable for Refuses {
            fn block_count(&self) -> usize {
                1
            }
            fn embed_input(&self, _xs: &Tensor) -> CandleResult<Tensor> {
                panic!("the guard must run before the forward")
            }
            fn block_forward(&self, _i: usize, _h: &Tensor) -> CandleResult<Tensor> {
                panic!("the guard must run before the forward")
            }
            fn head_forward(&self, _h: &Tensor) -> CandleResult<Tensor> {
                panic!("the guard must run before the forward")
            }
            fn checkpointable(&self) -> Result<(), String> {
                Err("not this one".into())
            }
        }

        let xs = Tensor::zeros((1, 2), DType::U32, &Device::Cpu).unwrap();
        let err = checkpointed_step(&Refuses, &xs, |t| t.sum_all(), 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not this one"), "{err}");
    }
}
