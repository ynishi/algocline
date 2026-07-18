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

use candle_core::Var;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use mlua::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ─── NN Store (save/load DI) ─────────────────────────────────────────────────

/// Error returned by an [`NnStore`] operation.
///
/// String-wrapping so a concrete implementation can carry its own error type
/// (fs, network, database, blob store) without leaking it across the boundary.
#[derive(Debug)]
pub struct NnStoreError(String);

impl NnStoreError {
    /// Wrap any displayable error message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for NnStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NnStoreError {}

/// Storage backend for `alc.nn.save` / `alc.nn.load` — the injection point for
/// where saved model bundles live.
///
/// The Host injects a concrete implementation via [`install_store`]; the nn
/// crate itself has no awareness of the process's app dir, `$HOME`, or any
/// environment variable. Implementations resolve a model `name` to a concrete
/// on-disk path; the tensor serialization (safetensors dump / load) stays
/// inside the nn crate.
///
/// Typical wiring: the engine constructs [`FsStore`] with a root chosen from
/// its AppDir (e.g. `<app_dir>/nn`) and calls [`install_store`]. Tests inject
/// an [`FsStore`] over a temp dir, or their own mock impl.
pub trait NnStore: Send + Sync {
    /// Return a writable path for `name`. Implementations create parent
    /// directories as needed so the caller can hand the path straight to the
    /// safetensors writer.
    fn save_path(&self, name: &str) -> Result<PathBuf, NnStoreError>;

    /// Return the read path for `name`. Implementations do not need to check
    /// existence — the caller surfaces missing-file errors from the loader.
    fn load_path(&self, name: &str) -> Result<PathBuf, NnStoreError>;
}

/// Filesystem-rooted default [`NnStore`].
///
/// Root is provided at construction time so the nn crate never inspects
/// `$HOME` or env directly. `name` is restricted to `[A-Za-z0-9_.-]` and
/// cannot contain `..`, so it cannot escape the root.
pub struct FsStore {
    root: PathBuf,
}

impl FsStore {
    /// Create a store rooted at `root`. Each `name` maps to
    /// `<root>/<name>.safetensors`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, name: &str) -> Result<PathBuf, NnStoreError> {
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
            || name.contains("..")
        {
            return Err(NnStoreError::new(format!(
                "invalid model name {name:?} (allowed: [A-Za-z0-9_.-], no '..')"
            )));
        }
        Ok(self.root.join(format!("{name}.safetensors")))
    }
}

impl NnStore for FsStore {
    fn save_path(&self, name: &str) -> Result<PathBuf, NnStoreError> {
        let p = self.resolve(name)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| NnStoreError::new(format!("create {parent:?}: {e}")))?;
        }
        Ok(p)
    }

    fn load_path(&self, name: &str) -> Result<PathBuf, NnStoreError> {
        self.resolve(name)
    }
}

/// App-data handle placed on the Lua VM by [`install_store`].
struct NnStoreHandle(Arc<dyn NnStore>);

/// Register a concrete [`NnStore`] on the Lua VM.
///
/// Must be called before `alc.nn.save` / `alc.nn.load` are used; otherwise
/// both closures return an "no NN store registered" error. Re-installing
/// overwrites the previous store.
pub fn install_store(lua: &Lua, store: Arc<dyn NnStore>) -> LuaResult<()> {
    lua.set_app_data(NnStoreHandle(store));
    Ok(())
}

/// Per-VM `alc.nn` model registry.
///
/// Holds Lua closures that the `alc.llm` bridge can dispatch to when a caller
/// passes `role="nn"` with a `model=<name>`. The registry lives on the VM as
/// app data, so its lifetime is tied to the Lua session and never crosses VMs.
/// Only the `alc.llm` in-process routing needs to reach across the crate
/// boundary, so this is the sole engine-visible export.
///
/// Values are `mlua::Function` (each entry is a Lua-side forward closure).
/// The Send + Mutex wrapping is only there to satisfy mlua's `send` feature
/// on `set_app_data`; the VM itself always runs on a single thread, so there
/// is never any real cross-thread traffic through this cell.
pub struct NnModelRegistry(pub Arc<Mutex<HashMap<String, mlua::Function>>>);

impl NnModelRegistry {
    /// Look up a registered model closure by name.
    ///
    /// Returns `None` if no model with `name` is registered on this VM.
    pub fn get(&self, name: &str) -> Option<mlua::Function> {
        self.0.lock().ok()?.get(name).cloned()
    }
}

/// Extract a candle [`Tensor`] from a Lua userdata that is either an
/// [`AlcTensor`] (an intermediate activation) or an [`AlcVar`] (a trainable
/// parameter). This lets Lua write ops uniformly over both operand kinds, e.g.
/// `w:mul(x)` where `w` is a Var and `x` is a Tensor.
fn tensor_of(ud: &mlua::AnyUserData) -> LuaResult<Tensor> {
    if let Ok(t) = ud.borrow::<AlcTensor>() {
        Ok(t.0.clone())
    } else if let Ok(v) = ud.borrow::<AlcVar>() {
        Ok(v.0.as_tensor().clone())
    } else {
        Err(LuaError::external(
            "alc.nn: expected an alc.nn tensor or var operand",
        ))
    }
}

/// Apply a binary candle op to `lhs` and the tensor behind `rhs`, wrapping the
/// result as a new [`AlcTensor`]. The candle op builds the autograd graph.
fn binop(
    lhs: &Tensor,
    rhs: &mlua::AnyUserData,
    op: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
    name: &str,
) -> LuaResult<AlcTensor> {
    let r = tensor_of(rhs)?;
    let out = op(lhs, &r).map_err(|e| LuaError::external(format!("alc.nn {name}: {e}")))?;
    Ok(AlcTensor(out))
}

/// Register the tensor-op methods shared by [`AlcTensor`] and [`AlcVar`].
///
/// `get` projects the receiver to its backing tensor. Every op returns a new
/// [`AlcTensor`] (a plain activation); ops on a `Var` still track gradients
/// because the returned tensor keeps the candle graph edge back to the `Var`.
fn add_tensor_ops<T, M>(methods: &mut M, get: fn(&T) -> &Tensor)
where
    T: 'static,
    M: mlua::UserDataMethods<T>,
{
    methods.add_method("add", move |_, this, other: mlua::AnyUserData| {
        binop(get(this), &other, |a, b| a.add(b), "tensor:add")
    });
    methods.add_method("mul", move |_, this, other: mlua::AnyUserData| {
        binop(get(this), &other, |a, b| a.mul(b), "tensor:mul")
    });
    methods.add_method("sub", move |_, this, other: mlua::AnyUserData| {
        binop(get(this), &other, |a, b| a.sub(b), "tensor:sub")
    });
    methods.add_method("matmul", move |_, this, other: mlua::AnyUserData| {
        binop(get(this), &other, |a, b| a.matmul(b), "tensor:matmul")
    });
    methods.add_method("sqr", move |_, this, ()| {
        let out = get(this)
            .sqr()
            .map_err(|e| LuaError::external(format!("alc.nn tensor:sqr: {e}")))?;
        Ok(AlcTensor(out))
    });
    methods.add_method("mean", move |_, this, ()| {
        let out = get(this)
            .mean_all()
            .map_err(|e| LuaError::external(format!("alc.nn tensor:mean: {e}")))?;
        Ok(AlcTensor(out))
    });
    methods.add_method("dims", move |_, this, ()| Ok(get(this).dims().to_vec()));
    methods.add_method("to_vec", move |_, this, ()| {
        let flat = get(this)
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| LuaError::external(format!("alc.nn tensor:to_vec: {e}")))?;
        Ok(flat)
    });
}

/// Lua `UserData` view of a candle [`Tensor`] — an intermediate activation.
///
/// The candle `Tensor` is `Arc`-backed, so moving it into the UserData is a
/// cheap handle clone (not a data copy). Lua sees an opaque handle and calls
/// the shared tensor ops; it never owns tensor storage.
pub(crate) struct AlcTensor(Tensor);

impl mlua::UserData for AlcTensor {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        add_tensor_ops(methods, |this: &AlcTensor| &this.0);
    }
}

/// Lua `UserData` view of a trainable candle [`Var`] (a parameter).
///
/// The `Var` is the Host-owned trainable state: gradients flow back to it and
/// the optimizer updates it in place. Lua holds only this handle and never owns
/// the `Var` lifetime beyond the handle, matching the design's layer boundary
/// (parameters live in Rust, composition lives in Lua). Shares the same op set
/// as [`AlcTensor`] so a Var can be used directly as an operand.
pub(crate) struct AlcVar(Var);

impl mlua::UserData for AlcVar {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        add_tensor_ops(methods, |this: &AlcVar| this.0.as_tensor());
    }
}

/// Lua `UserData` for a candle optimizer over Host-owned [`Var`]s.
///
/// Holds the optimizer state (momentum etc.) in Rust. `backward_step(loss)` is
/// the fused path: candle runs `loss.backward()` into a `GradStore` and applies
/// one update step to the owned Vars in a single call.
pub(crate) struct AlcOptimizer(AdamW);

impl mlua::UserData for AlcOptimizer {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("backward_step", |_, this, loss: mlua::AnyUserData| {
            let l = tensor_of(&loss)?;
            this.0
                .backward_step(&l)
                .map_err(|e| LuaError::external(format!("alc.nn opt:backward_step: {e}")))?;
            Ok(())
        });
    }
}

/// Build the `alc.nn` module table.
///
/// Mirrors the `register_math` convention (`bridge/mod.rs`): the engine calls
/// this behind the `nn` feature and sets the result as `alc.nn`. Exposes the
/// thin candle primitives the layer boundary requires — tensor / var / adamw
/// plus the shared tensor ops — so Lua can write a model and a training loop.
pub fn module(lua: &Lua) -> LuaResult<LuaTable> {
    let nn = lua.create_table()?;

    // Per-VM model registry: dispatched by the alc.llm bridge when a caller
    // passes role="nn". Installed idempotently — if the VM already has one
    // (e.g. install_for_pkg_test also called `module()`), reuse it so registry
    // lookups from the llm bridge see the same entries.
    if lua.app_data_ref::<NnModelRegistry>().is_none() {
        lua.set_app_data(NnModelRegistry(Arc::new(Mutex::new(HashMap::new()))));
    }

    // nn.tensor(data) — a 1-D CPU f32 tensor from a Lua array of numbers.
    let tensor = lua.create_function(|_, data: Vec<f32>| {
        let len = data.len();
        let t = Tensor::from_vec(data, (len,), &Device::Cpu)
            .map_err(|e| LuaError::external(format!("alc.nn.tensor: {e}")))?;
        Ok(AlcTensor(t))
    })?;
    nn.set("tensor", tensor)?;

    // nn.var(n, init) — a trainable 1-D parameter of length n filled with init.
    // The Var is Host-owned; Lua receives a handle usable as an operand.
    let var = lua.create_function(|_, (n, init): (usize, f32)| {
        let t = Tensor::from_vec(vec![init; n], (n,), &Device::Cpu)
            .map_err(|e| LuaError::external(format!("alc.nn.var: {e}")))?;
        let v = Var::from_tensor(&t).map_err(|e| LuaError::external(format!("alc.nn.var: {e}")))?;
        Ok(AlcVar(v))
    })?;
    nn.set("var", var)?;

    // nn.adamw(vars, lr) — AdamW over the given Vars (optimizer state Host-owned).
    let adamw = lua.create_function(|_, (vars, lr): (mlua::Table, f64)| {
        let mut collected: Vec<Var> = Vec::new();
        for pair in vars.sequence_values::<mlua::AnyUserData>() {
            let ud = pair?;
            let v = ud
                .borrow::<AlcVar>()
                .map_err(|_| LuaError::external("alc.nn.adamw: expected a list of alc.nn vars"))?;
            collected.push(v.0.clone());
        }
        let opt = AdamW::new(
            collected,
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )
        .map_err(|e| LuaError::external(format!("alc.nn.adamw: {e}")))?;
        Ok(AlcOptimizer(opt))
    })?;
    nn.set("adamw", adamw)?;

    // nn.register(name, forward_fn) — register a Lua closure as an alc.llm
    // responder. The engine's alc.llm bridge dispatches to it when a call
    // passes `role="nn"` and `model=<name>`, so a trained model becomes an
    // in-process LLM the strategy addresses via the standard `alc.llm` API.
    let register = lua.create_function(|lua, (name, forward): (String, mlua::Function)| {
        let registry = lua
            .app_data_ref::<NnModelRegistry>()
            .ok_or_else(|| LuaError::external("alc.nn.register: registry missing"))?;
        registry
            .0
            .lock()
            .map_err(|e| LuaError::external(format!("alc.nn.register: poisoned lock: {e}")))?
            .insert(name, forward);
        Ok(())
    })?;
    nn.set("register", register)?;

    // nn.save(vars, name) — dump the given named Vars via the installed
    // NnStore. `vars` is a Lua table keyed by string (e.g. `{ w = w, b = b }`);
    // each key becomes a safetensors entry name. The Host's NnStore chose the
    // on-disk location; this crate owns the tensor → bytes serialization
    // (candle safetensors), so callers can swap the backend without touching
    // the dump path.
    let save = lua.create_function(|lua, (vars, name): (mlua::Table, String)| {
        let store = lua
            .app_data_ref::<NnStoreHandle>()
            .ok_or_else(|| LuaError::external("alc.nn.save: no NN store registered"))?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for pair in vars.pairs::<String, mlua::AnyUserData>() {
            let (k, ud) = pair?;
            let v = ud
                .borrow::<AlcVar>()
                .map_err(|_| LuaError::external("alc.nn.save: table values must be alc.nn vars"))?;
            map.insert(k, v.0.as_tensor().clone());
        }
        let path = store
            .0
            .save_path(&name)
            .map_err(|e| LuaError::external(format!("alc.nn.save: {e}")))?;
        candle_core::safetensors::save(&map, &path)
            .map_err(|e| LuaError::external(format!("alc.nn.save: {e}")))?;
        Ok(())
    })?;
    nn.set("save", save)?;

    // nn.load(name) — restore Vars via the installed NnStore. Returns a Lua
    // table keyed by the safetensors entry names produced by `save`. Each
    // value is a fresh Host-owned Var initialised from the stored tensor.
    let load = lua.create_function(|lua, name: String| {
        let store = lua
            .app_data_ref::<NnStoreHandle>()
            .ok_or_else(|| LuaError::external("alc.nn.load: no NN store registered"))?;
        let path = store
            .0
            .load_path(&name)
            .map_err(|e| LuaError::external(format!("alc.nn.load: {e}")))?;
        let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
            .map_err(|e| LuaError::external(format!("alc.nn.load: {e}")))?;
        let out = lua.create_table()?;
        for (k, t) in tensors {
            let v = Var::from_tensor(&t)
                .map_err(|e| LuaError::external(format!("alc.nn.load: {e}")))?;
            out.set(k, AlcVar(v))?;
        }
        Ok(out)
    })?;
    nn.set("load", load)?;

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

    /// End-to-end: a tiny linear model is *trained* from Lua through the real
    /// candle path, then "called" on a new input. Learns y = 2x + 1 with
    /// `nn.var` parameters + `nn.adamw` + `opt:backward_step`, all composed in
    /// Lua (Rust exposes only the primitives). Asserts the trained model
    /// predicts x=5 -> ~11.
    #[test]
    fn lua_trains_linear_model_and_predicts() {
        let lua = mlua::Lua::new();
        let nn = module(&lua).unwrap();
        lua.globals().set("nn", nn).unwrap();

        let pred: Vec<f32> = lua
            .load(
                r#"
                local w = nn.var(1, 0.0)   -- weight, starts at 0
                local b = nn.var(1, 0.0)   -- bias, starts at 0
                local opt = nn.adamw({ w, b }, 0.1)
                local xs = { 0, 1, 2, 3, 4 }
                local ys = { 1, 3, 5, 7, 9 }   -- y = 2x + 1
                for _ = 1, 400 do
                    for i = 1, #xs do
                        local x = nn.tensor({ xs[i] })
                        local target = nn.tensor({ ys[i] })
                        local out = w:mul(x):add(b)          -- w*x + b
                        local loss = out:sub(target):sqr():mean()
                        opt:backward_step(loss)
                    end
                end
                -- call the trained model on a new input
                return w:mul(nn.tensor({ 5 })):add(b):to_vec()
            "#,
            )
            .eval()
            .expect("lua training loop");

        // y = 2*5 + 1 = 11
        assert!(
            (pred[0] - 11.0).abs() < 0.2,
            "trained model should predict ~11 for x=5, got {}",
            pred[0]
        );
    }

    /// DI roundtrip: train a linear model, save through an injected
    /// [`FsStore`] over a temp dir, load it in a fresh Lua VM, and verify the
    /// restored Vars produce the same prediction. Confirms the store trait
    /// carries the whole dump / restore path — no `$HOME` / env lookup, no
    /// path leakage into the nn crate.
    #[test]
    fn lua_save_load_roundtrip_via_store() {
        let root = std::env::temp_dir().join("alc-nn-save-load-roundtrip");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let store: Arc<dyn NnStore> = Arc::new(FsStore::new(&root));

        // Train + save in VM A.
        let lua_a = mlua::Lua::new();
        install_store(&lua_a, Arc::clone(&store)).unwrap();
        let nn_a = module(&lua_a).unwrap();
        lua_a.globals().set("nn", nn_a).unwrap();
        let pred_before: Vec<f32> = lua_a
            .load(
                r#"
                local w = nn.var(1, 0.0)
                local b = nn.var(1, 0.0)
                local opt = nn.adamw({ w, b }, 0.1)
                local xs = { 0, 1, 2, 3, 4 }
                local ys = { 1, 3, 5, 7, 9 }
                for _ = 1, 400 do
                    for i = 1, #xs do
                        local x = nn.tensor({ xs[i] })
                        local target = nn.tensor({ ys[i] })
                        local loss = w:mul(x):add(b):sub(target):sqr():mean()
                        opt:backward_step(loss)
                    end
                end
                nn.save({ w = w, b = b }, "roundtrip")
                return w:mul(nn.tensor({ 5 })):add(b):to_vec()
            "#,
            )
            .eval()
            .expect("train + save");
        assert!(root.join("roundtrip.safetensors").exists());

        // Load in a fresh VM B with only the store re-installed.
        let lua_b = mlua::Lua::new();
        install_store(&lua_b, Arc::clone(&store)).unwrap();
        let nn_b = module(&lua_b).unwrap();
        lua_b.globals().set("nn", nn_b).unwrap();
        let pred_after: Vec<f32> = lua_b
            .load(
                r#"
                local m = nn.load("roundtrip")
                return m.w:mul(nn.tensor({ 5 })):add(m.b):to_vec()
            "#,
            )
            .eval()
            .expect("load + predict");

        assert!(
            (pred_before[0] - pred_after[0]).abs() < 1e-4,
            "prediction must survive save/load (before={}, after={})",
            pred_before[0],
            pred_after[0]
        );
    }

    /// save/load without an installed store must return a clear Lua error, so
    /// callers can't silently write to a nn-crate-chosen default location.
    #[test]
    fn save_without_installed_store_errors() {
        let lua = mlua::Lua::new();
        let nn = module(&lua).unwrap();
        lua.globals().set("nn", nn).unwrap();
        let err = lua
            .load(
                r#"
                local w = nn.var(1, 0.0)
                nn.save({ w = w }, "noop")
            "#,
            )
            .exec()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no NN store registered"),
            "unexpected error: {err}"
        );
    }

    /// FsStore rejects names that could escape the store root.
    #[test]
    fn fs_store_rejects_path_traversal() {
        let root = std::env::temp_dir().join("alc-nn-traversal-test");
        let store = FsStore::new(&root);
        assert!(store.save_path("../evil").is_err());
        assert!(store.save_path("a/b").is_err());
        assert!(store.save_path("").is_err());
        assert!(store.save_path("ok_name-1.v2").is_ok());
    }

    /// Sum of all parameter values across the VarMap (test helper).
    fn weight_sum(varmap: &VarMap) -> f32 {
        varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().sum_all().unwrap().to_scalar::<f32>().unwrap())
            .sum()
    }
}
