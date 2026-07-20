//! `alc.nn.wrap_lora` bridge — Layer 5b S1 Lua bind for the arch-neutral
//! LoRA wrap surface established by Layer 2 (`Gpt2Model::wrap_lora` /
//! `TinyLlamaModel::wrap_lora`) and Layer 4b (`NnHandle` dispatch).
//!
//! Registers a single Lua entry:
//!
//! ```text
//! alc.nn.wrap_lora(base_handle, opts) -> NnHandle
//! ```
//!
//! Self-consumable — a caller can wrap and inspect the resulting
//! [`NnHandle`] without ever running the trainer. Layer 5b S2
//! (`alc.nn.trainer.run_lora_ft`) consumes the same wrap surface.
//!
//! # Invariants
//!
//! All errors surface as loud [`LuaError::external`] with prefix
//! `alc.nn.wrap_lora:` — no silent fallback, no `warn!` swallow (per
//! `CLAUDE.md §Service 層の Error 伝播規律`). Every shape listed in the
//! Layer 5b design doc §2.1 has a matching branch in this module.
//!
//! 1. Base handle validation refuses (a) non-userdata / non-`NnHandle`
//!    userdata, (b) already-wrapped handles (double-wrap protection
//!    via [`NnHandle::is_lora_wrapped`]), (c) inference-only
//!    architectures (Llama).
//! 2. `target_modules` validation routes through the underlying arch's
//!    canonical target set ([`LoraConfig::default_targets`] for GPT-2,
//!    [`TinyLlamaModel::default_lora_targets`] for TinyLlama). The
//!    bridge does not hard-code an arch → target-set mapping locally;
//!    both sources live in the `algocline-nn` crate and stay the SoT.
//! 3. `dropout` is validated in `[0.0, 1.0)` even though the current
//!    LoRA forward path ignores it (schema stability for a future
//!    dropout ship — see [`LoraConfig`] docs).
//! 4. File body is `#[cfg(feature = "nn")]` at the module level via
//!    `bridge/mod.rs`; the default build never links this file.

use algocline_nn::arch::{LoraConfig, TinyLlamaModel};
use mlua::prelude::*;

use super::nn_card::{
    wrap_gpt2_lora_bridge, wrap_tinyllama_lora_bridge, Gpt2Handle, LlamaHandle, NnHandle,
    TinyLlamaHandle,
};

/// Register `alc.nn.wrap_lora` onto the pre-existing `alc.nn` table.
///
/// Must be called after [`super::register_nn`] (which populates the
/// `alc.nn` sub-table). Signature mirrors the sibling
/// [`super::nn_card::register_nn_card`]: obtain the `alc.nn` sub-table
/// through `alc_table.get("nn")` inside the function so the caller
/// only has to hand the outer `alc` table.
pub(super) fn register_nn_wrap(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let nn_table: LuaTable = alc_table.get("nn")?;

    let wrap_lora = lua.create_function(
        |_lua, (base, opts): (LuaValue, LuaTable)| -> LuaResult<NnHandle> {
            wrap_lora_impl(&base, opts)
        },
    )?;
    nn_table.set("wrap_lora", wrap_lora)?;
    Ok(())
}

fn wrap_lora_impl(base: &LuaValue, opts: LuaTable) -> LuaResult<NnHandle> {
    // 1. Reject non-userdata bases up front — design §2.1 row 1
    //    ("expected NnHandle, got <type>"). Typed at LuaValue so the
    //    Lua-side type name is available for the error message.
    let ud = match base {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.wrap_lora: expected NnHandle, got {}",
                base.type_name()
            )));
        }
    };

    // 2. Downcast to NnHandle (arch-neutral) or one of the typed
    //    Handles (backward-compat for callers that reach for
    //    `alc.nn.preset.gpt2` / `.tinyllama` directly). Same discipline
    //    as `nn_card::merge_lora_impl`.
    let handle: NnHandle = if let Ok(nn) = ud.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = ud.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = ud.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = ud.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.wrap_lora: expected NnHandle, got unknown userdata \
             (Gpt2Handle / TinyLlamaHandle / LlamaHandle also accepted)",
        ));
    };

    // 3. Double-wrap protection.
    if handle.is_lora_wrapped() {
        return Err(LuaError::external(
            "alc.nn.wrap_lora: handle is already LoRA-wrapped; drop the wrap or \
             start from a base handle",
        ));
    }

    // 4. Refuse arch families outside the wrap-capable set (Llama
    //    adapter is inference-only — same posture as Layer 4b
    //    `load_wrap`). Done before opts validation so a Llama caller
    //    sees the directional arch error rather than a rank/alpha
    //    schema error that also applies.
    if let NnHandle::Llama(_) = handle {
        return Err(LuaError::external(format!(
            "alc.nn.wrap_lora: architecture {} is not LoRA-wrappable \
             (only gpt2 / tinyllama families are supported)",
            handle.arch()
        )));
    }

    let arch = handle.arch();

    // 5. Extract + validate scalar opts.
    let rank = extract_rank(&opts)?;
    let alpha = extract_alpha(&opts)?;
    let dropout = extract_dropout(&opts)?;

    // 6. Resolve target_modules (per-arch default when nil) and
    //    validate every entry against the arch's canonical set.
    let target_modules = resolve_and_validate_targets(&opts, arch)?;

    // 7. Build LoraConfig + dispatch per NnHandle variant.
    let mut cfg = LoraConfig::with_targets(rank, alpha, target_modules);
    cfg.dropout = dropout;

    match handle {
        NnHandle::Gpt2(base_gpt2) => {
            let wrapped = wrap_gpt2_lora_bridge(&base_gpt2, &cfg)?;
            Ok(NnHandle::Gpt2(wrapped))
        }
        NnHandle::TinyLlama(base_tll) => {
            let wrapped = wrap_tinyllama_lora_bridge(&base_tll, &cfg)?;
            Ok(NnHandle::TinyLlama(wrapped))
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the assertion
            // rather than silently invoking a non-existent wrap path.
            unreachable!("Llama variant guarded above")
        }
    }
}

fn extract_rank(opts: &LuaTable) -> LuaResult<usize> {
    let raw: Option<i64> = opts.get("rank")?;
    let n = raw.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.wrap_lora: opts.rank must be a positive integer")
    })?;
    Ok(n as usize)
}

fn extract_alpha(opts: &LuaTable) -> LuaResult<f32> {
    let raw: Option<f64> = opts.get("alpha")?;
    let v = raw.filter(|v| *v > 0.0).ok_or_else(|| {
        LuaError::external("alc.nn.wrap_lora: opts.alpha must be a positive number")
    })?;
    Ok(v as f32)
}

fn extract_dropout(opts: &LuaTable) -> LuaResult<f32> {
    // Absent → default 0.0 (design §1.1). Present → must be in
    // `[0.0, 1.0)` (exclusive upper).
    let raw: Option<f64> = opts.get("dropout")?;
    let v = raw.unwrap_or(0.0);
    if !(0.0..1.0).contains(&v) {
        return Err(LuaError::external(
            "alc.nn.wrap_lora: opts.dropout must be in [0.0, 1.0)",
        ));
    }
    Ok(v as f32)
}

fn resolve_and_validate_targets(opts: &LuaTable, arch: &str) -> LuaResult<Vec<String>> {
    // The wrap-capable arch dispatch is already guarded by
    // `wrap_lora_impl` step 4; treat an unknown arch here as an
    // internal invariant break so a future variant addition does not
    // silently fall through to a Rust-side error.
    let known = canonical_targets_for(arch).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.wrap_lora: architecture {arch} is not LoRA-wrappable \
             (only gpt2 / tinyllama families are supported)"
        ))
    })?;

    let raw: LuaValue = opts.get("target_modules")?;
    match raw {
        LuaValue::Nil => Ok(known),
        LuaValue::Table(tbl) => {
            let entries: Vec<String> = tbl
                .sequence_values::<String>()
                .collect::<LuaResult<Vec<_>>>()?;
            if entries.is_empty() {
                return Err(LuaError::external(
                    "alc.nn.wrap_lora: opts.target_modules must be non-empty \
                     (or nil for the per-arch default)",
                ));
            }
            for entry in &entries {
                if !known.iter().any(|k| k == entry) {
                    let known_list = known.join(", ");
                    return Err(LuaError::external(format!(
                        "alc.nn.wrap_lora: unknown target module {entry:?} for arch {arch} \
                         (known: [{known_list}])"
                    )));
                }
            }
            Ok(entries)
        }
        other => Err(LuaError::external(format!(
            "alc.nn.wrap_lora: opts.target_modules must be an array of strings \
             (or nil for the per-arch default); got {}",
            other.type_name()
        ))),
    }
}

/// Return the arch's canonical LoRA target-module set. Layer 5b
/// discipline: read the set from the `algocline-nn` crate (SoT)
/// rather than duplicate the list here — see design §1.3.
///
/// Adding a new LoRA-capable arch = new arm + widen the wrap-capable
/// dispatch in [`wrap_lora_impl`] (step 4 arch refusal + step 7 match).
/// A follow-up refactor could push this into `nn_card::ArchOps` as a
/// new function-pointer slot (§Q4-A extension) so the arch table
/// stays the single grep-able add site; kept inline for L5b-S1 to
/// respect the "do not modify nn_card.rs beyond minimal widening"
/// scope.
fn canonical_targets_for(arch: &str) -> Option<Vec<String>> {
    match arch {
        "gpt2" => Some(LoraConfig::default_targets()),
        "tinyllama" => Some(TinyLlamaModel::default_lora_targets()),
        _ => None,
    }
}

#[cfg(test)]
mod wrap_lora_bridge_tests {
    //! Layer 5b S1 — bridge integration tests for
    //! `alc.nn.wrap_lora`.
    //!
    //! Axis A1/A2 exercise the GPT-2 + TinyLlama happy paths (arch
    //! dispatch + base-freeze byte-identical invariant). Axis B1-B5
    //! exercise the config schema refusals (rank / alpha /
    //! target_modules / dropout).
    //!
    //! All tests use the CPU/F32 `gpt2-tiny` / `tinyllama-tiny`
    //! micro shapes — same discipline as `nn_card::merge_lora_bridge_tests`
    //! (no HF hub download, no >1s train step).
    use super::super::nn_card::{build_gpt2_handle, build_tinyllama_handle};
    use super::*;
    use mlua::Lua;
    use serde_json::json;

    fn opts_table(lua: &Lua, v: serde_json::Value) -> LuaTable {
        use mlua::LuaSerdeExt;
        let val = lua.to_value(&v).expect("to_value");
        match val {
            LuaValue::Table(t) => t,
            _ => unreachable!("json object must serialise to Lua table"),
        }
    }

    /// Build a `gpt2-tiny` base handle in-memory (pretrained=false so
    /// the base VarMap is present). Returns tempdir + base handle +
    /// Lua VM; caller keeps the tempdir alive for the test duration.
    fn setup_gpt2_base_scaffold() -> (tempfile::TempDir, Gpt2Handle, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("build gpt2 tiny base");
        (tmp, base, lua)
    }

    fn setup_tinyllama_base_scaffold() -> (tempfile::TempDir, TinyLlamaHandle, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let base = build_tinyllama_handle("tinyllama-tiny", Some(&base_opts), &nn_dir)
            .expect("build tinyllama-tiny base");
        (tmp, base, lua)
    }

    /// Byte-snapshot every tensor in the given VarMap as a flat
    /// `Vec<f32>`. Mirrors the discipline used by
    /// `algocline-nn/tests/tinyllama_lora_ft.rs::run_lora_ft_tinyllama_leaves_base_weights_bit_identical`.
    fn snapshot_varmap(vm: &candle_nn::VarMap) -> Vec<Vec<f32>> {
        vm.all_vars()
            .iter()
            .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
            .collect()
    }

    // ─── Axis A — happy paths ──────────────────────────────────────

    #[test]
    fn wrap_lora_gpt2_happy_path_returns_wrapped_handle() {
        let (_tmp, base, lua) = setup_gpt2_base_scaffold();
        let base_vm = base.varmap().expect("gpt2 tiny base carries a VarMap");
        let before = snapshot_varmap(&base_vm);
        let base_var_count = base_vm.all_vars().len();

        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, json!({ "rank": 4, "alpha": 8.0 }));
        let wrapped = wrap_lora_impl(&LuaValue::UserData(base_ud), opts).expect("wrap_lora");
        assert!(
            wrapped.is_lora_wrapped(),
            "wrap_lora must set has_lora=true"
        );
        assert_eq!(wrapped.arch(), "gpt2");

        // Base VarMap byte-identical after wrap. `wrap_lora` allocates
        // a fresh LoRA VarMap and moves each base linear into its
        // frozen `LoraLinear::base` slot — the base map's tensors
        // must remain untouched.
        let after = snapshot_varmap(&base_vm);
        assert_eq!(
            base_vm.all_vars().len(),
            base_var_count,
            "base VarMap var count changed during wrap_lora"
        );
        assert_eq!(before.len(), after.len(), "base VarMap length changed");
        for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            assert_eq!(
                b, a,
                "base VarMap tensor #{i} changed during wrap_lora (must stay frozen)"
            );
        }
    }

    #[test]
    fn wrap_lora_tinyllama_happy_path_returns_wrapped_handle() {
        let (_tmp, base, lua) = setup_tinyllama_base_scaffold();
        let base_vm = base.varmap().expect("tinyllama tiny base carries a VarMap");
        let before = snapshot_varmap(&base_vm);
        let base_var_count = base_vm.all_vars().len();

        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();

        let opts = opts_table(&lua, json!({ "rank": 4, "alpha": 8.0 }));
        let wrapped = wrap_lora_impl(&LuaValue::UserData(base_ud), opts).expect("wrap_lora");
        assert!(
            wrapped.is_lora_wrapped(),
            "wrap_lora must set has_lora=true"
        );
        assert_eq!(wrapped.arch(), "tinyllama");

        let after = snapshot_varmap(&base_vm);
        assert_eq!(
            base_vm.all_vars().len(),
            base_var_count,
            "base VarMap var count changed during wrap_lora"
        );
        assert_eq!(before.len(), after.len(), "base VarMap length changed");
        for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            assert_eq!(
                b, a,
                "base VarMap tensor #{i} changed during wrap_lora (must stay frozen)"
            );
        }
    }

    // ─── Axis B — config schema refusals ───────────────────────────

    // Helper: NnHandle does not implement Debug (Handle wraps
    // Arc<Mutex<Model>> whose Debug is expensive / unavailable);
    // use match to extract the error string rather than
    // `.unwrap_err()` which would demand Debug on the Ok side.
    // Same pattern as `nn_card::merge_lora_bridge_tests`.
    fn expect_err(result: LuaResult<NnHandle>) -> String {
        match result {
            Ok(_) => panic!("expected wrap_lora_impl to fail; got Ok(NnHandle)"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn wrap_lora_refuses_zero_rank() {
        let (_tmp, base, lua) = setup_gpt2_base_scaffold();
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let opts = opts_table(&lua, json!({ "rank": 0, "alpha": 8.0 }));
        let msg = expect_err(wrap_lora_impl(&LuaValue::UserData(base_ud), opts));
        assert!(
            msg.contains("alc.nn.wrap_lora:") && msg.contains("opts.rank"),
            "expected rank error, got: {msg}"
        );
    }

    #[test]
    fn wrap_lora_refuses_missing_alpha() {
        let (_tmp, base, lua) = setup_gpt2_base_scaffold();
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        // Provide rank but omit alpha.
        let opts = opts_table(&lua, json!({ "rank": 4 }));
        let msg = expect_err(wrap_lora_impl(&LuaValue::UserData(base_ud), opts));
        assert!(
            msg.contains("alc.nn.wrap_lora:") && msg.contains("opts.alpha"),
            "expected alpha error, got: {msg}"
        );
    }

    #[test]
    fn wrap_lora_refuses_empty_target_modules_array() {
        let (_tmp, base, lua) = setup_gpt2_base_scaffold();
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let opts = opts_table(
            &lua,
            json!({ "rank": 4, "alpha": 8.0, "target_modules": [] }),
        );
        let msg = expect_err(wrap_lora_impl(&LuaValue::UserData(base_ud), opts));
        assert!(
            msg.contains("alc.nn.wrap_lora:") && msg.contains("opts.target_modules"),
            "expected target_modules error, got: {msg}"
        );
    }

    #[test]
    fn wrap_lora_refuses_unknown_target_module_per_arch() {
        // Design §4.2 B4: pass `up` (a gpt2-only target) to a
        // tinyllama base to force the bridge to route validation
        // through the arch's canonical set rather than a static
        // superset.
        let (_tmp, base, lua) = setup_tinyllama_base_scaffold();
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();
        let opts = opts_table(
            &lua,
            json!({
                "rank": 4,
                "alpha": 8.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up"],
            }),
        );
        let msg = expect_err(wrap_lora_impl(&LuaValue::UserData(base_ud), opts));
        assert!(
            msg.contains("alc.nn.wrap_lora:")
                && msg.contains("unknown target module")
                && msg.contains("tinyllama"),
            "expected per-arch unknown-target error, got: {msg}"
        );
    }

    #[test]
    fn wrap_lora_refuses_dropout_out_of_range() {
        let (_tmp, base, lua) = setup_gpt2_base_scaffold();
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        // 1.0 is at the exclusive upper bound → refused.
        let opts = opts_table(&lua, json!({ "rank": 4, "alpha": 8.0, "dropout": 1.0 }));
        let msg = expect_err(wrap_lora_impl(&LuaValue::UserData(base_ud), opts));
        assert!(
            msg.contains("alc.nn.wrap_lora:") && msg.contains("opts.dropout"),
            "expected dropout error, got: {msg}"
        );
    }
}
