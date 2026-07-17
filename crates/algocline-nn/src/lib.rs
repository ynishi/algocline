//! algocline-nn — thin candle wrapper for the `alc.nn` Lua surface.
//!
//! # Architecture
//!
//! This crate is the Host(Rust) side of the alc.nn layer boundary. The design
//! intent is:
//!
//! - **Host owns the heavy state**: tensors, the autograd graph, parameters
//!   (`candle_nn::VarMap`), optimizer state, and the `GradStore` all live in
//!   Rust. Lua never holds a `Var` lifetime — only opaque handles.
//! - **Lua owns composition and loops**: model assembly, the training loop, lr
//!   schedule, and batching are written in Lua. Rust exposes only thin wraps of
//!   individual candle ops; it does not embed loop / schedule / batching logic.
//! - **core stays clean**: `algocline-core` never depends on candle or tensor
//!   types. This crate is an optional, feature-gated dependency of the engine
//!   (`nn` feature, default off) so the default MCP build stays light.
//!
//! # L1 spike scope
//!
//! Phase L1 is a spike: it validates the riskiest unknowns (candle link
//! interference, GradStore key access, and `mlua::UserData` tensor exposure)
//! and lands a minimal primitive. It is not the full op set.
//!
//! In Step 1 this crate only links `candle-core` (CPU) to confirm there is no
//! link interference with the mlua-vendored workspace. Later steps add
//! `candle-nn` (VarMap / autograd / optimizer) and the `mlua` UserData surface.

#![warn(missing_docs)]

use candle_core::{Device, Tensor};

/// Link-gate probe used by the Step 1 spike.
///
/// Constructs a tiny CPU tensor and returns its element count. The only purpose
/// is to force `candle-core` to actually link and execute in the algocline
/// workspace so the spike can confirm there is no link interference with the
/// mlua-vendored build.
pub fn probe() -> candle_core::Result<usize> {
    let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    Ok(t.elem_count())
}

// ─── alc.nn Lua surface ──────────────────────────────────────────────────────

use mlua::prelude::*;

/// Lua `UserData` view of a candle [`Tensor`].
///
/// The candle `Tensor` is held by value; because a candle `Tensor` is an
/// `Arc`-backed handle, moving it into the UserData is a cheap clone of the
/// handle (not the data). Lua sees an opaque handle and can only call the
/// exposed methods — it never owns tensor storage or (later) a `Var` lifetime,
/// which stays Host-owned per the design's layer boundary.
///
/// L1 exposes a single element-wise op (`add`) plus read-back helpers; the full
/// op set is a later step.
pub(crate) struct AlcTensor(Tensor);

impl mlua::UserData for AlcTensor {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // add(other) — element-wise addition, returns a new AlcTensor.
        methods.add_method("add", |_, this, other: mlua::UserDataRef<AlcTensor>| {
            let sum = this
                .0
                .add(&other.0)
                .map_err(|e| LuaError::external(format!("alc.nn tensor:add: {e}")))?;
            Ok(AlcTensor(sum))
        });

        // dims() — shape as a Lua array of integers.
        methods.add_method("dims", |_, this, ()| Ok(this.0.dims().to_vec()));

        // to_vec() — flatten to a 1-D Lua array of numbers (read-back for tests
        // and simple inspection).
        methods.add_method("to_vec", |_, this, ()| {
            let flat = this
                .0
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| LuaError::external(format!("alc.nn tensor:to_vec: {e}")))?;
            Ok(flat)
        });
    }
}

/// Build the `alc.nn` module table.
///
/// Mirrors the `register_math` convention (`bridge/mod.rs`): the engine calls
/// this behind the `nn` feature and sets the result as `alc.nn`. The returned
/// table currently exposes a single constructor, `nn.tensor(data)`, which
/// builds a 1-D CPU `f32` tensor from a Lua array of numbers.
pub fn module(lua: &Lua) -> LuaResult<LuaTable> {
    let nn = lua.create_table()?;

    let tensor = lua.create_function(|_, data: Vec<f32>| {
        let len = data.len();
        let t = Tensor::from_vec(data, (len,), &Device::Cpu)
            .map_err(|e| LuaError::external(format!("alc.nn.tensor: {e}")))?;
        Ok(AlcTensor(t))
    })?;
    nn.set("tensor", tensor)?;

    Ok(nn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

    #[test]
    fn probe_links_candle_cpu() {
        assert_eq!(probe().unwrap(), 3);
    }

    /// Step 2 spike: prove the design's "Host owns VarMap + autograd graph +
    /// GradStore" claim holds on real candle CPU.
    ///
    /// Flow: build a tiny linear layer whose weights are `Var`s in a `VarMap`
    /// → forward (matmul + bias) → scalar loss → `loss.backward()` → pull the
    /// per-variable gradient out of the returned `GradStore` by keying on the
    /// `Var`'s inner tensor. This confirms the exact GradStore key-access
    /// spelling that was previously an open question in the design.
    #[test]
    fn varmap_backward_gradstore_per_var() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Tiny linear layer: weight (1, 2) + bias (1,), registered as Vars.
        let linear = candle_nn::linear(2, 1, vb.pp("ln")).unwrap();

        // Forward on a fixed input, then a scalar loss.
        let x = Tensor::new(&[[1.0f32, 2.0]], &device).unwrap(); // (1, 2)
        let y = linear.forward(&x).unwrap(); // (1, 1)
        let loss = y.sqr().unwrap().mean_all().unwrap(); // scalar

        // Backward → GradStore.
        let grads = loss.backward().unwrap();

        // Per-variable gradient extraction: key the GradStore on each Var's
        // inner tensor. Every parameter Var must have a gradient.
        let vars = varmap.all_vars();
        assert!(!vars.is_empty(), "VarMap must own the layer's parameters");
        for var in &vars {
            let g = grads
                .get(var.as_tensor())
                .expect("each Var must have a gradient in the GradStore");
            assert_eq!(
                g.dims(),
                var.as_tensor().dims(),
                "gradient shape must match its Var"
            );
        }
    }

    /// Step 2 spike: the fused optimizer path (`backward_step`) mutates the
    /// VarMap-owned parameters in place. Confirms optimizer state is Host-owned
    /// and a single training step actually moves the weights.
    #[test]
    fn optimizer_step_updates_weights() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let linear = candle_nn::linear(2, 1, vb.pp("ln")).unwrap();

        // Snapshot the weight-sum before the step.
        let before = weight_sum(&varmap);

        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        let x = Tensor::new(&[[1.0f32, 2.0]], &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let loss = y.sqr().unwrap().mean_all().unwrap();

        // Fused backward + parameter update.
        opt.backward_step(&loss).unwrap();

        let after = weight_sum(&varmap);
        assert!(
            (after - before).abs() > 1e-7,
            "optimizer step must change the weights (before={before}, after={after})"
        );
    }

    /// Step 3 spike: `AlcTensor` UserData is reachable from Lua. Build two
    /// tensors via `nn.tensor(...)`, add them with `t:add(u)`, and read the
    /// result back with `t:to_vec()`. Confirms the mlua exposure of a candle
    /// tensor (the third of the design's open questions).
    #[test]
    fn lua_tensor_add_roundtrip() {
        let lua = mlua::Lua::new();
        let nn = module(&lua).unwrap();
        lua.globals().set("nn", nn).unwrap();

        let out: Vec<f32> = lua
            .load(
                r#"
                local a = nn.tensor({ 1, 2, 3 })
                local b = nn.tensor({ 10, 20, 30 })
                local c = a:add(b)
                return c:to_vec()
            "#,
            )
            .eval()
            .unwrap();

        assert_eq!(out, vec![11.0, 22.0, 33.0]);

        let dims: Vec<usize> = lua
            .load("return nn.tensor({ 1, 2, 3, 4 }):dims()")
            .eval()
            .unwrap();
        assert_eq!(dims, vec![4]);
    }

    /// Sum of all parameter values across the VarMap (test helper).
    fn weight_sum(varmap: &VarMap) -> f32 {
        varmap
            .all_vars()
            .iter()
            .map(|v| {
                v.as_tensor()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            })
            .sum()
    }
}
