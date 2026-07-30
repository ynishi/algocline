//! Shared opts extraction + error conversion for the `alc.nn.trainer.*`
//! and `alc.nn.wrap_lora` Lua surfaces (feature `nn`).
//!
//! Single source of truth for the training-config extractors and the
//! [`TrainError`] -> [`LuaError`] conversion consumed by both trainer
//! surface families:
//!
//! - [`super::nn_card`] — `alc.nn.trainer.{full_ft, lora, distill}`:
//!   every training-config key is optional and falls back to the
//!   [`FullFtConfig`] crate default ([`extract_full_ft_opts`]).
//! - [`super::nn_trainer`] — `alc.nn.trainer.{run_lora_ft, run_full_ft,
//!   run_distill}`: `lr` / `batch` / `steps` are required and
//!   `warmup` / `schedule` are validated up front
//!   ([`extract_run_train_cfg`]).
//!
//! The two contracts stay distinct (that is a Lua-visible difference);
//! only the implementations are shared.
//!
//! The LoRA opts schema ([`extract_lora_cfg`] +
//! [`canonical_targets_for`]) is shared on the same terms by
//! [`super::nn_wrap`] (`alc.nn.wrap_lora`) and
//! [`super::nn_trainer`] (`alc.nn.trainer.run_lora_ft`), and the
//! distillation-loss selector ([`extract_distill_loss_kind`]) by
//! `alc.nn.trainer.distill` and `alc.nn.trainer.run_distill`.
//!
//! # Prefix as an argument
//!
//! Every entry takes the caller's error `prefix` as its first argument
//! rather than hard-coding one, so the loud-error contract (one error
//! prefix per Lua surface) survives the consolidation. The precedent is
//! [`super::nn_card::guard_base_dtype_for_training`], which already
//! serves four trainer entrypoints off a single `fn_name` argument.

use algocline_nn::arch::{LoraConfig, TinyLlamaModel};
use algocline_nn::train::{DistillLossKind, FullFtConfig, ScheduleKind, TrainError};
use mlua::prelude::*;

/// Extract a [`FullFtConfig`] from an opts table, applying the crate's
/// defaults for any missing key. Rejects zero-sized values at the
/// boundary (matches the training-loop's own early exit shape).
///
/// This is the optional-field contract used by the
/// `alc.nn.trainer.{full_ft, lora, distill}` entries: a missing key is
/// not an error, and `opts = nil` yields [`FullFtConfig::default`].
/// Surfaces that require the caller to spell the hyperparameters out
/// use [`extract_run_train_cfg`] instead.
///
/// `prefix` is prepended to every validation error (see the module
/// docs on prefix-as-argument).
pub(super) fn extract_full_ft_opts(
    prefix: &str,
    opts: Option<&LuaTable>,
) -> LuaResult<FullFtConfig> {
    let mut cfg = FullFtConfig::default();
    let Some(t) = opts else {
        return Ok(cfg);
    };
    if let Some(v) = t.get::<Option<f64>>("lr")? {
        cfg.lr = v;
    }
    if let Some(v) = t.get::<Option<usize>>("batch_size")? {
        cfg.batch_size = v;
    }
    if let Some(v) = t.get::<Option<usize>>("steps")? {
        cfg.steps = v;
    }
    if let Some(v) = t.get::<Option<usize>>("warmup")? {
        cfg.warmup = v;
    }
    if let Some(v) = t.get::<Option<String>>("schedule")? {
        cfg.schedule = parse_schedule(prefix, &v)?;
    }
    apply_optional_overrides(&mut cfg, t)?;
    if cfg.batch_size == 0 {
        return Err(LuaError::external(format!(
            "{prefix}: batch_size must be >= 1"
        )));
    }
    Ok(cfg)
}

/// Extract a validated [`FullFtConfig`] from `opts` for the
/// `alc.nn.trainer.run_*` surfaces.
///
/// Strict contract (Layer 5b/5c design §1.2): `lr` (finite, > 0),
/// `batch` (> 0) and `steps` (> 0) are required; `warmup` is optional
/// and defaults to 0 (a negative value is a loud error rather than a
/// silent clamp); `schedule` is optional, defaults to
/// `"CosineWithWarmup"` and accepts only `"CosineWithWarmup"` /
/// `"Constant"` — anything else is refused with the caller-supplied
/// value in the message so a typo is debuggable at the surface.
///
/// The pass-through fields (`grad_accum` / `weight_decay` /
/// `ckpt_every` / `ckpt_keep`) are read by
/// [`apply_optional_overrides`], the same helper
/// [`extract_full_ft_opts`] uses, so both surfaces see one field set.
///
/// The caller's Lua table is read-only here: the pre-consolidation
/// implementations canonicalised field names by writing back into the
/// caller's table (`opts.set("batch_size", …)`) before delegating;
/// the [`FullFtConfig`] is now assembled directly, which drops that
/// observable side effect without changing the returned value.
pub(super) fn extract_run_train_cfg(prefix: &str, opts: &LuaTable) -> LuaResult<FullFtConfig> {
    let lr: Option<f64> = opts.get("lr")?;
    let lr = lr.filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| {
        LuaError::external(format!("{prefix}: opts.lr must be a positive number"))
    })?;

    let batch: Option<i64> = opts.get("batch")?;
    let batch = batch.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external(format!("{prefix}: opts.batch must be a positive integer"))
    })? as usize;

    let steps: Option<i64> = opts.get("steps")?;
    let steps = steps.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external(format!("{prefix}: opts.steps must be a positive integer"))
    })? as usize;

    // Optional warmup — must be >= 0 (integer). `i64` lets a negative
    // value surface the design §2.2 loud error rather than silently
    // clamp to 0.
    let warmup = match opts.get::<Option<i64>>("warmup")? {
        Some(v) if v < 0 => {
            return Err(LuaError::external(format!(
                "{prefix}: opts.warmup must be >= 0"
            )));
        }
        Some(v) => v as usize,
        None => 0,
    };

    // Optional schedule — design §1.2 default "CosineWithWarmup".
    let schedule = match opts
        .get::<Option<String>>("schedule")?
        .as_deref()
        .unwrap_or("CosineWithWarmup")
    {
        "CosineWithWarmup" => ScheduleKind::CosineWithWarmup,
        "Constant" => ScheduleKind::Constant,
        other => {
            return Err(LuaError::external(format!(
                "{prefix}: opts.schedule must be one of \
                 \"CosineWithWarmup\" / \"Constant\" (got {other:?})"
            )));
        }
    };

    let mut cfg = FullFtConfig {
        lr,
        batch_size: batch,
        steps,
        warmup,
        schedule,
        ..FullFtConfig::default()
    };
    apply_optional_overrides(&mut cfg, opts)?;
    Ok(cfg)
}

/// Apply the pass-through [`FullFtConfig`] fields that both surface
/// families accept verbatim (no surface-specific validation):
/// `grad_accum` / `weight_decay` / `ckpt_every` / `ckpt_keep`.
///
/// Keeping the reads here is what makes each of those keys appear
/// exactly once in the repo — the pre-consolidation `run_*` surfaces
/// reached the same keys indirectly (they delegated to the
/// `full_ft`-side extractor after rewriting the caller table), and the
/// doc claim that they were "NOT exposed" did not match the code.
///
/// Wrong-type input surfaces as an mlua type-mismatch error (no
/// prefix rewrite) exactly as it did before, so no `prefix` argument
/// is threaded through here.
fn apply_optional_overrides(cfg: &mut FullFtConfig, opts: &LuaTable) -> LuaResult<()> {
    if let Some(v) = opts.get::<Option<usize>>("grad_accum")? {
        cfg.grad_accum = v;
    }
    if let Some(v) = opts.get::<Option<f64>>("weight_decay")? {
        cfg.weight_decay = v;
    }
    if let Some(v) = opts.get::<Option<usize>>("ckpt_every")? {
        cfg.ckpt_every = v;
    }
    if let Some(v) = opts.get::<Option<usize>>("ckpt_keep")? {
        cfg.ckpt_keep = v;
    }
    Ok(())
}

/// Parse the `alc.nn.trainer.{full_ft, lora, distill}` schedule
/// vocabulary (`"cosine"` / `"cosine_with_warmup"` / `"constant"`).
///
/// The `run_*` surfaces use their own two-value vocabulary
/// (`"CosineWithWarmup"` / `"Constant"`, see
/// [`extract_run_train_cfg`]) — the two spellings are Lua-visible
/// contracts and are deliberately kept apart.
fn parse_schedule(prefix: &str, s: &str) -> LuaResult<ScheduleKind> {
    match s {
        "cosine" | "cosine_with_warmup" => Ok(ScheduleKind::CosineWithWarmup),
        "constant" => Ok(ScheduleKind::Constant),
        other => Err(LuaError::external(format!(
            "{prefix}: unknown schedule '{other}' \
             (expected 'cosine' or 'constant')"
        ))),
    }
}

/// Extract a validated [`LoraConfig`] from a LoRA opts table.
///
/// Strict contract shared by `alc.nn.wrap_lora` ([`super::nn_wrap`])
/// and `alc.nn.trainer.run_lora_ft` ([`super::nn_trainer`]):
///
/// - `rank` — required, positive integer.
/// - `alpha` — required, positive number.
/// - `dropout` — optional (default `0.0`), must be in `[0.0, 1.0)`.
///   Validated even though the current LoRA forward path ignores it,
///   so the schema is stable for a future dropout ship.
/// - `target_modules` — optional; `nil` resolves to the arch's
///   canonical set ([`canonical_targets_for`]), an explicit array must
///   be non-empty and every entry must belong to that set.
///
/// `arch` is the [`super::nn_card::NnHandle::arch`] string. Both
/// callers already refuse non-wrap-capable architectures before
/// reaching here, so the unknown-arch arm only fires when a new
/// [`super::nn_card::NnHandle`] variant is added without widening
/// [`canonical_targets_for`] — a loud error rather than a fall-through
/// into a Rust-side failure.
///
/// The looser `alc.nn.trainer.lora` extractor
/// (`super::nn_card::extract_lora_cfg`) is a *different* Lua-visible
/// contract (no arch check, no dropout range check, map-shaped
/// `target_modules` accepted) and deliberately stays in that module.
pub(super) fn extract_lora_cfg(prefix: &str, arch: &str, opts: &LuaTable) -> LuaResult<LoraConfig> {
    let rank: Option<i64> = opts.get("rank")?;
    let rank = rank.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external(format!("{prefix}: opts.rank must be a positive integer"))
    })? as usize;

    let alpha: Option<f64> = opts.get("alpha")?;
    let alpha = alpha.filter(|v| *v > 0.0).ok_or_else(|| {
        LuaError::external(format!("{prefix}: opts.alpha must be a positive number"))
    })? as f32;

    let dropout: Option<f64> = opts.get("dropout")?;
    let dropout = dropout.unwrap_or(0.0);
    if !(0.0..1.0).contains(&dropout) {
        return Err(LuaError::external(format!(
            "{prefix}: opts.dropout must be in [0.0, 1.0)"
        )));
    }
    let dropout = dropout as f32;

    let known = canonical_targets_for(arch).ok_or_else(|| {
        LuaError::external(format!(
            "{prefix}: architecture {arch} is not LoRA-wrappable \
             (only gpt2 / tinyllama families are supported)"
        ))
    })?;

    let raw: LuaValue = opts.get("target_modules")?;
    let target_modules: Vec<String> = match raw {
        LuaValue::Nil => known,
        LuaValue::Table(tbl) => {
            let entries: Vec<String> = tbl
                .sequence_values::<String>()
                .collect::<LuaResult<Vec<_>>>()?;
            if entries.is_empty() {
                return Err(LuaError::external(format!(
                    "{prefix}: opts.target_modules must be non-empty \
                     (or nil for the per-arch default)"
                )));
            }
            for entry in &entries {
                if !known.iter().any(|k| k == entry) {
                    let known_list = known.join(", ");
                    return Err(LuaError::external(format!(
                        "{prefix}: unknown target module {entry:?} for arch {arch} \
                         (known: [{known_list}])"
                    )));
                }
            }
            entries
        }
        other => {
            return Err(LuaError::external(format!(
                "{prefix}: opts.target_modules must be an array of strings \
                 (or nil for the per-arch default); got {}",
                other.type_name()
            )));
        }
    };

    let mut cfg = LoraConfig::with_targets(rank, alpha, target_modules);
    cfg.dropout = dropout;
    Ok(cfg)
}

/// Return the arch's canonical LoRA target-module set, or `None` when
/// the architecture is not LoRA-wrappable.
///
/// The `algocline-nn` crate owns the lists (Layer 5b design §1.3); the
/// bridge only routes. Adding a LoRA-capable arch = a new arm here
/// plus widening the wrap-capable dispatch in
/// [`super::nn_wrap`] / [`super::nn_trainer`].
pub(super) fn canonical_targets_for(arch: &str) -> Option<Vec<String>> {
    match arch {
        "gpt2" => Some(LoraConfig::default_targets()),
        "tinyllama" => Some(TinyLlamaModel::default_lora_targets()),
        _ => None,
    }
}

/// Extract the distillation-loss selector from `opts.loss_kind`.
///
/// Shared by `alc.nn.trainer.distill` ([`super::nn_card`], which
/// passes an optional opts table) and
/// `alc.nn.trainer.run_distill` ([`super::nn_trainer`], which always
/// has one). A missing key — or a missing table — defaults to
/// `"ce"`; an unknown value is refused rather than silently falling
/// back, and wrong-type input (`loss_kind = true`) surfaces as an
/// mlua type-mismatch error for the same reason.
pub(super) fn extract_distill_loss_kind(
    prefix: &str,
    opts: Option<&LuaTable>,
) -> LuaResult<DistillLossKind> {
    let raw = match opts {
        Some(t) => t
            .get::<Option<String>>("loss_kind")?
            .unwrap_or_else(|| "ce".to_string()),
        None => "ce".to_string(),
    };
    match raw.as_str() {
        "ce" => Ok(DistillLossKind::Ce),
        other => Err(LuaError::external(format!(
            "{prefix}: unknown loss_kind '{other}' (expected 'ce')"
        ))),
    }
}

/// Translate a [`TrainError`] into a `prefix`-tagged [`LuaError`].
///
/// Variant-by-variant so each error carries an actionable hint
/// (design §2.2 shape); unknown / future variants route through
/// `Display` so a `TrainError` addition stays a loud runtime error
/// rather than a compile break.
pub(super) fn train_err_to_lua(prefix: &str, e: TrainError) -> LuaError {
    let msg = match e {
        TrainError::ZeroSteps => format!("{prefix}: zero steps"),
        TrainError::LeaseHeld => {
            format!("{prefix}: training lease already active on this VM")
        }
        TrainError::DatasetExhausted { seen, requested } => format!(
            "{prefix}: dataset exhausted after {seen} steps \
             (requested {requested})"
        ),
        TrainError::Ckpt(inner) => format!("{prefix}: checkpoint: {inner}"),
        TrainError::Candle(inner) => format!("{prefix}: candle: {inner}"),
        other => format!("{prefix}: {other}"),
    };
    LuaError::external(msg)
}

#[cfg(test)]
mod tests {
    //! Extractor unit tests moved verbatim from `nn_card::trainer_tests`
    //! together with the functions under test (only the `prefix`
    //! argument is new).
    use super::*;
    use mlua::Lua;

    fn opts_from(lua: &Lua, pairs: &[(&str, LuaValue)]) -> LuaTable {
        let t = lua.create_table().expect("create opts table");
        for (k, v) in pairs {
            t.set(*k, v.clone()).expect("set opt field");
        }
        t
    }

    #[test]
    fn full_ft_opts_defaults_when_empty() {
        let lua = Lua::new();
        let cfg = extract_full_ft_opts("alc.nn.trainer", None).expect("None -> defaults");
        let default = FullFtConfig::default();
        assert_eq!(cfg.lr, default.lr);
        assert_eq!(cfg.batch_size, default.batch_size);
        assert_eq!(cfg.steps, default.steps);

        let empty = lua.create_table().unwrap();
        let cfg2 =
            extract_full_ft_opts("alc.nn.trainer", Some(&empty)).expect("empty table -> defaults");
        assert_eq!(cfg2.lr, default.lr);
        assert_eq!(cfg2.batch_size, default.batch_size);
    }

    #[test]
    fn full_ft_opts_partial_merges_with_defaults() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[
                ("lr", LuaValue::Number(1e-3)),
                ("steps", LuaValue::Integer(42)),
            ],
        );
        let cfg = extract_full_ft_opts("alc.nn.trainer", Some(&opts)).expect("partial merge");
        assert!((cfg.lr - 1e-3).abs() < 1e-12, "lr override");
        assert_eq!(cfg.steps, 42, "steps override");
        // Unset fields keep the crate default.
        let d = FullFtConfig::default();
        assert_eq!(cfg.batch_size, d.batch_size);
        assert_eq!(cfg.warmup, d.warmup);
    }

    #[test]
    fn full_ft_opts_reject_zero_batch_size() {
        let lua = Lua::new();
        let opts = opts_from(&lua, &[("batch_size", LuaValue::Integer(0))]);
        let err = extract_full_ft_opts("alc.nn.trainer", Some(&opts)).expect_err("zero batch_size");
        assert!(
            err.to_string().contains("batch_size must be >= 1"),
            "message: {err}"
        );
    }

    #[test]
    fn full_ft_opts_reads_pass_through_fields() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[
                ("grad_accum", LuaValue::Integer(2)),
                ("weight_decay", LuaValue::Number(0.5)),
                ("ckpt_every", LuaValue::Integer(7)),
                ("ckpt_keep", LuaValue::Integer(1)),
            ],
        );
        let cfg = extract_full_ft_opts("alc.nn.trainer", Some(&opts)).expect("pass-through");
        assert_eq!(cfg.grad_accum, 2);
        assert!((cfg.weight_decay - 0.5).abs() < 1e-12);
        assert_eq!(cfg.ckpt_every, 7);
        assert_eq!(cfg.ckpt_keep, 1);
    }

    #[test]
    fn schedule_parser_accepts_known_and_rejects_unknown() {
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "cosine").unwrap(),
            ScheduleKind::CosineWithWarmup
        ));
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "cosine_with_warmup").unwrap(),
            ScheduleKind::CosineWithWarmup
        ));
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "constant").unwrap(),
            ScheduleKind::Constant
        ));
        let err = parse_schedule("alc.nn.trainer", "linear").expect_err("unknown");
        assert!(err.to_string().contains("linear"), "message: {err}");
    }

    #[test]
    fn run_train_cfg_requires_lr_batch_steps() {
        let lua = Lua::new();
        let empty = lua.create_table().unwrap();
        let err = extract_run_train_cfg("alc.nn.trainer.run_full_ft", &empty)
            .expect_err("missing required fields");
        assert!(
            err.to_string()
                .contains("alc.nn.trainer.run_full_ft: opts.lr must be a positive number"),
            "message: {err}"
        );

        let no_batch = opts_from(&lua, &[("lr", LuaValue::Number(1e-4))]);
        let err = extract_run_train_cfg("alc.nn.trainer.run_full_ft", &no_batch)
            .expect_err("missing batch");
        assert!(err.to_string().contains("opts.batch"), "message: {err}");

        let no_steps = opts_from(
            &lua,
            &[
                ("lr", LuaValue::Number(1e-4)),
                ("batch", LuaValue::Integer(2)),
            ],
        );
        let err = extract_run_train_cfg("alc.nn.trainer.run_full_ft", &no_steps)
            .expect_err("missing steps");
        assert!(err.to_string().contains("opts.steps"), "message: {err}");
    }

    #[test]
    fn run_train_cfg_happy_path_maps_fields_without_mutating_opts() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[
                ("lr", LuaValue::Number(1e-4)),
                ("batch", LuaValue::Integer(2)),
                ("steps", LuaValue::Integer(3)),
                ("warmup", LuaValue::Integer(1)),
                ("grad_accum", LuaValue::Integer(4)),
            ],
        );
        let cfg = extract_run_train_cfg("alc.nn.trainer.run_lora_ft", &opts).expect("happy path");
        assert!((cfg.lr - 1e-4).abs() < 1e-12);
        assert_eq!(cfg.batch_size, 2);
        assert_eq!(cfg.steps, 3);
        assert_eq!(cfg.warmup, 1);
        assert!(matches!(cfg.schedule, ScheduleKind::CosineWithWarmup));
        // Pass-through field stays reachable (the pre-consolidation
        // delegation chain reached it too).
        assert_eq!(cfg.grad_accum, 4);
        // The caller's table is left untouched — no `batch_size` /
        // `schedule` write-back.
        assert!(opts.get::<Option<i64>>("batch_size").unwrap().is_none());
        assert!(opts.get::<Option<String>>("schedule").unwrap().is_none());
    }

    #[test]
    fn run_train_cfg_rejects_negative_warmup_and_unknown_schedule() {
        let lua = Lua::new();
        let base: &[(&str, LuaValue)] = &[
            ("lr", LuaValue::Number(1e-4)),
            ("batch", LuaValue::Integer(2)),
            ("steps", LuaValue::Integer(3)),
        ];

        let mut with_warmup = base.to_vec();
        with_warmup.push(("warmup", LuaValue::Integer(-1)));
        let opts = opts_from(&lua, &with_warmup);
        let err = extract_run_train_cfg("alc.nn.trainer.run_distill", &opts)
            .expect_err("negative warmup");
        assert!(
            err.to_string()
                .contains("alc.nn.trainer.run_distill: opts.warmup must be >= 0"),
            "message: {err}"
        );

        let mut with_schedule = base.to_vec();
        with_schedule.push((
            "schedule",
            LuaValue::String(lua.create_string("cosine").unwrap()),
        ));
        let opts = opts_from(&lua, &with_schedule);
        let err = extract_run_train_cfg("alc.nn.trainer.run_distill", &opts)
            .expect_err("lower-case schedule is not the run_* vocabulary");
        assert!(err.to_string().contains("must be one of"), "message: {err}");
    }

    #[test]
    fn train_err_to_lua_uses_prefix_and_variant_hint() {
        let err = train_err_to_lua("alc.nn.trainer", TrainError::ZeroSteps);
        assert!(
            err.to_string().contains("alc.nn.trainer: zero steps"),
            "message: {err}"
        );
        let err = train_err_to_lua(
            "alc.nn.trainer.run_full_ft",
            TrainError::DatasetExhausted {
                seen: 1,
                requested: 2,
            },
        );
        assert!(
            err.to_string()
                .contains("alc.nn.trainer.run_full_ft: dataset exhausted after 1 steps"),
            "message: {err}"
        );
    }

    // ─── LoRA opts schema ──────────────────────────────────────────

    /// Minimal valid LoRA opts (`rank` + `alpha`), extended by the
    /// caller when a specific field is under test.
    fn lora_opts(lua: &Lua, extra: &[(&str, LuaValue)]) -> LuaTable {
        let mut pairs: Vec<(&str, LuaValue)> = vec![
            ("rank", LuaValue::Integer(4)),
            ("alpha", LuaValue::Number(8.0)),
        ];
        pairs.extend_from_slice(extra);
        opts_from(lua, &pairs)
    }

    #[test]
    fn lora_cfg_requires_positive_rank_and_alpha() {
        let lua = Lua::new();
        let empty = lua.create_table().unwrap();
        let err = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &empty).expect_err("missing rank");
        assert!(
            err.to_string()
                .contains("alc.nn.wrap_lora: opts.rank must be a positive integer"),
            "message: {err}"
        );

        let zero_rank = opts_from(
            &lua,
            &[
                ("rank", LuaValue::Integer(0)),
                ("alpha", LuaValue::Number(8.0)),
            ],
        );
        let err = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &zero_rank).expect_err("zero rank");
        assert!(err.to_string().contains("opts.rank"), "message: {err}");

        let no_alpha = opts_from(&lua, &[("rank", LuaValue::Integer(4))]);
        let err = extract_lora_cfg("alc.nn.trainer.run_lora_ft", "gpt2", &no_alpha)
            .expect_err("missing alpha");
        assert!(
            err.to_string()
                .contains("alc.nn.trainer.run_lora_ft: opts.alpha must be a positive number"),
            "message: {err}"
        );
    }

    #[test]
    fn lora_cfg_dropout_defaults_to_zero_and_rejects_out_of_range() {
        let lua = Lua::new();
        let cfg = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &lora_opts(&lua, &[]))
            .expect("dropout default");
        assert_eq!(cfg.dropout, 0.0);

        let ok = lora_opts(&lua, &[("dropout", LuaValue::Number(0.05))]);
        let cfg = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &ok).expect("in-range dropout");
        assert!((cfg.dropout - 0.05).abs() < 1e-6);

        // 1.0 is the exclusive upper bound → refused.
        let high = lora_opts(&lua, &[("dropout", LuaValue::Number(1.0))]);
        let err = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &high).expect_err("dropout 1.0");
        assert!(
            err.to_string()
                .contains("alc.nn.wrap_lora: opts.dropout must be in [0.0, 1.0)"),
            "message: {err}"
        );

        let negative = lora_opts(&lua, &[("dropout", LuaValue::Number(-0.1))]);
        let err =
            extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &negative).expect_err("negative dropout");
        assert!(err.to_string().contains("opts.dropout"), "message: {err}");
    }

    #[test]
    fn lora_cfg_rejects_unknown_arch() {
        let lua = Lua::new();
        let err = extract_lora_cfg("alc.nn.wrap_lora", "llama", &lora_opts(&lua, &[]))
            .expect_err("llama is inference-only");
        assert!(
            err.to_string().contains(
                "alc.nn.wrap_lora: architecture llama is not LoRA-wrappable \
                 (only gpt2 / tinyllama families are supported)"
            ),
            "message: {err}"
        );
    }

    #[test]
    fn lora_cfg_defaults_targets_per_arch() {
        let lua = Lua::new();
        let cfg = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &lora_opts(&lua, &[]))
            .expect("gpt2 defaults");
        assert_eq!(cfg.target_modules, LoraConfig::default_targets());
        assert_eq!(cfg.rank, 4);
        assert!((cfg.alpha - 8.0).abs() < 1e-6);

        let cfg = extract_lora_cfg("alc.nn.wrap_lora", "tinyllama", &lora_opts(&lua, &[]))
            .expect("tinyllama defaults");
        assert_eq!(
            cfg.target_modules,
            TinyLlamaModel::default_lora_targets(),
            "the per-arch default must come from the arch, not a superset"
        );
    }

    #[test]
    fn lora_cfg_validates_explicit_targets_against_the_arch_set() {
        let lua = Lua::new();
        let gpt2_targets = canonical_targets_for("gpt2").expect("gpt2 is wrappable");
        let first = gpt2_targets.first().expect("non-empty default set").clone();

        // A single entry from the arch's own set is accepted.
        let list = lua.create_table().unwrap();
        list.set(1, first.clone()).unwrap();
        let opts = lora_opts(&lua, &[("target_modules", LuaValue::Table(list))]);
        let cfg = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &opts).expect("known target");
        assert_eq!(cfg.target_modules, vec![first]);

        // Empty array → refused (nil is the way to ask for the default).
        let empty = lua.create_table().unwrap();
        let opts = lora_opts(&lua, &[("target_modules", LuaValue::Table(empty))]);
        let err = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &opts).expect_err("empty array");
        assert!(
            err.to_string().contains(
                "alc.nn.wrap_lora: opts.target_modules must be non-empty \
                 (or nil for the per-arch default)"
            ),
            "message: {err}"
        );

        // A gpt2-only target against a tinyllama base → per-arch refusal
        // naming the offending entry, the arch and the known set.
        let cross = lua.create_table().unwrap();
        cross.set(1, "up").unwrap();
        let opts = lora_opts(&lua, &[("target_modules", LuaValue::Table(cross))]);
        let err = extract_lora_cfg("alc.nn.trainer.run_lora_ft", "tinyllama", &opts)
            .expect_err("gpt2-only target on tinyllama");
        let msg = err.to_string();
        assert!(
            msg.contains(
                "alc.nn.trainer.run_lora_ft: unknown target module \"up\" for arch tinyllama"
            ) && msg.contains("known: ["),
            "message: {msg}"
        );

        // Wrong shape (not an array) → refused with the Lua type name.
        let opts = lora_opts(&lua, &[("target_modules", LuaValue::Boolean(true))]);
        let err = extract_lora_cfg("alc.nn.wrap_lora", "gpt2", &opts).expect_err("boolean targets");
        assert!(
            err.to_string().contains(
                "alc.nn.wrap_lora: opts.target_modules must be an array of strings \
                 (or nil for the per-arch default); got boolean"
            ),
            "message: {err}"
        );
    }

    #[test]
    fn canonical_targets_known_and_unknown_arches() {
        assert_eq!(
            canonical_targets_for("gpt2"),
            Some(LoraConfig::default_targets())
        );
        assert_eq!(
            canonical_targets_for("tinyllama"),
            Some(TinyLlamaModel::default_lora_targets())
        );
        assert!(canonical_targets_for("llama").is_none());
        assert!(canonical_targets_for("").is_none());
    }

    // ─── Distillation loss selector ────────────────────────────────
    //
    // Moved from `nn_card::trainer_tests` together with the function
    // (only the `prefix` argument is new).

    #[test]
    fn distill_loss_kind_defaults_to_ce_and_rejects_unknown() {
        assert!(matches!(
            extract_distill_loss_kind("alc.nn.trainer.distill", None).unwrap(),
            DistillLossKind::Ce
        ));
        let lua = Lua::new();
        let empty = lua.create_table().unwrap();
        assert!(matches!(
            extract_distill_loss_kind("alc.nn.trainer.run_distill", Some(&empty)).unwrap(),
            DistillLossKind::Ce
        ));
        let opts = opts_from(
            &lua,
            &[(
                "loss_kind",
                LuaValue::String(lua.create_string("ce").unwrap()),
            )],
        );
        assert!(matches!(
            extract_distill_loss_kind("alc.nn.trainer.distill", Some(&opts)).unwrap(),
            DistillLossKind::Ce
        ));
        let bad = opts_from(
            &lua,
            &[(
                "loss_kind",
                LuaValue::String(lua.create_string("kl_soft").unwrap()),
            )],
        );
        let err = extract_distill_loss_kind("alc.nn.trainer.run_distill", Some(&bad))
            .expect_err("unknown loss");
        assert!(
            err.to_string().contains(
                "alc.nn.trainer.run_distill: unknown loss_kind 'kl_soft' (expected 'ce')"
            ),
            "message: {err}"
        );
    }

    #[test]
    fn distill_loss_kind_rejects_wrong_type_input() {
        let lua = Lua::new();
        // `loss_kind = true` (boolean) must surface as a Lua
        // type-mismatch error, not silently fall back to "ce".
        // (Integers coerce to strings in mlua, so booleans are used to
        // provoke the type check.)
        let opts = opts_from(&lua, &[("loss_kind", LuaValue::Boolean(true))]);
        let err = extract_distill_loss_kind("alc.nn.trainer.distill", Some(&opts))
            .expect_err("wrong-type loss_kind");
        let msg = err.to_string();
        assert!(!msg.is_empty(), "type-mismatch error should have a message");
    }
}
