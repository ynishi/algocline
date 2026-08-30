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

use std::path::PathBuf;

use algocline_nn::arch::{LoraConfig, TinyLlamaModel};
use algocline_nn::train::{
    Candidate, CkptControl, CkptFlow, CkptHook, CkptInfo, DistillLossKind, FullFtConfig, KeepMark,
    OptimizerKind, ScheduleKind, TrainError,
};
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
    apply_optional_overrides(prefix, &mut cfg, t)?;
    if cfg.batch_size == 0 {
        return Err(LuaError::external(format!(
            "{prefix}: batch_size must be >= 1"
        )));
    }
    Ok(cfg)
}

/// Extract an optional `on_ckpt` Lua callback from an opts table and
/// wrap it as a Rust-side [`CkptHook`].
///
/// Kept separate from [`extract_full_ft_opts`] because the hook is not
/// part of [`FullFtConfig`]'s owned data (a `Box<dyn FnMut>` breaks the
/// `#[derive(Debug, Clone)]` cascade and pushes the borrow story into
/// every downstream consumer). The full-fine-tune bridge extracts the
/// hook alongside the config and passes both to `run_full_ft` as
/// independent arguments.
///
/// The returned closure holds the Lua callback via a [`WeakLua`] +
/// [`LuaFunction`] pair, mirroring the sampler bridge
/// ([`crate::bridge::nn_sampler::LuaSamplerBridge`]) — mlua 0.11's
/// `send` feature makes both `Send`, so the resulting hook satisfies
/// the `Send` bound on [`CkptHook`]. A dropped Lua VM surfaces as a
/// `TrainError::Hook` rather than a `panic!`.
///
/// Lua return-value contract (loud on anything else):
/// - `nil` or `"continue"` → carry on, hold nothing
/// - `"break"` → stop now, hold nothing
/// - `"keep"` → hold this checkpoint, carry on
/// - `{ action = …, keep = … }` → the two axes stated separately, for
///   the one combination the strings cannot say (hold *and* stop)
/// - any other value → `TrainError::Hook` naming the offending value
///
/// A Lua-side `error(...)` inside the callback also surfaces as
/// `TrainError::Hook` with the raised message. Either way the run ends
/// there: no terminal `<prefix>.safetensors` is written and no
/// Checkpoint comes back, so the weights a caller can still reach are
/// the rotating `<prefix>-step<N>.safetensors` files plus any step the
/// hook had already kept (see
/// [`algocline_nn::train::TrainError::Hook`] docs).
///
/// # `ckpt_every` cross-check
///
/// The training loop only fires the hook when `ckpt_every > 0`
/// ([`algocline_nn::train::FullFtConfig::ckpt_every`] defaults to `0`,
/// i.e. mid-run checkpoints disabled), so an `on_ckpt` supplied without
/// a positive `ckpt_every` is a silent no-op: the caller would see a
/// clean run and never a single fire. That pairing is refused here,
/// loudly, for every surface that extracts the hook.
pub(super) fn extract_on_ckpt_hook(
    prefix: &str,
    lua: &Lua,
    opts: Option<&LuaTable>,
) -> LuaResult<Option<CkptHook>> {
    let Some(t) = opts else {
        return Ok(None);
    };
    let Some(callback): Option<LuaFunction> = t.get("on_ckpt")? else {
        return Ok(None);
    };

    // Cross-check against the sibling `ckpt_every` key read by
    // `apply_optional_overrides` — a hook that can never fire is a
    // configuration error, not a no-op default.
    let ckpt_every: usize = t.get::<Option<usize>>("ckpt_every")?.unwrap_or(0);
    if ckpt_every == 0 {
        return Err(LuaError::external(format!(
            "{prefix}: opts.on_ckpt requires opts.ckpt_every > 0 \
             (the hook would never fire)"
        )));
    }

    let weak = lua.weak();
    let prefix_owned = prefix.to_string();
    let hook: CkptHook = Box::new(move |info: &CkptInfo| -> Result<CkptControl, String> {
        let lua = weak
            .try_upgrade()
            .ok_or_else(|| format!("{prefix_owned}: Lua state owning on_ckpt is gone"))?;
        let table = ckpt_info_to_lua(&lua, info)
            .map_err(|e| format!("{prefix_owned}: cannot build info table: {e}"))?;
        let returned: LuaValue = callback
            .call(table)
            .map_err(|e| format!("{prefix_owned}: on_ckpt callback failed: {e}"))?;
        parse_ckpt_control(&prefix_owned, &returned)
    });
    Ok(Some(hook))
}

/// Build a Lua table mirroring the [`CkptInfo`] fields so the callback
/// can index into it by name (`info.step`, `info.ckpt_path`, …).
fn ckpt_info_to_lua(lua: &Lua, info: &CkptInfo) -> LuaResult<LuaTable> {
    let t = lua.create_table()?;
    t.set("step", info.step)?;
    // `ckpt_path` is UTF-8 in practice (`CheckpointStore` builds it
    // from ASCII prefix + numeric step + `.safetensors`), but the file
    // system can technically return non-UTF-8 bytes on Linux — fall
    // back to a lossy display so the callback still sees a string.
    let path_string = info
        .ckpt_path
        .to_str()
        .map(String::from)
        .unwrap_or_else(|| info.ckpt_path.to_string_lossy().into_owned());
    t.set("ckpt_path", path_string)?;
    t.set("train_loss", info.train_loss)?;
    t.set("lr", info.lr)?;
    t.set("grad_norm", info.grad_norm)?;
    t.set("elapsed_ms", info.elapsed_ms)?;
    t.set("min_train_loss", info.min_train_loss)?;
    Ok(t)
}

/// Project a run's held checkpoints into a Lua array.
///
/// The other half of the hook ABI: [`ckpt_info_to_lua`] carries a
/// checkpoint *into* the hook, this carries back out the ones the hook
/// asked to keep. One entry per [`Candidate`], in the order it asked:
///
/// ```lua
/// { step = 40, ckpt_path = "/…/run-step40.safetensors",
///   train_loss = 1.83, reason = "tier-2" }
/// ```
///
/// `reason` is absent when the hook did not give one. Every path is
/// pinned for the life of the run, so each entry still resolves when
/// the caller reads it.
pub(super) fn candidates_to_lua(lua: &Lua, candidates: &[Candidate]) -> LuaResult<LuaTable> {
    let out = lua.create_table()?;
    for (i, c) in candidates.iter().enumerate() {
        let entry = lua.create_table()?;
        entry.set("step", c.step)?;
        entry.set("ckpt_path", c.ckpt_path.to_string_lossy().into_owned())?;
        entry.set("train_loss", c.train_loss)?;
        if let Some(reason) = &c.reason {
            entry.set("reason", reason.as_str())?;
        }
        out.set(i + 1, entry)?;
    }
    Ok(out)
}

/// Map a Lua-side `on_ckpt` return value to a [`CkptControl`].
///
/// Kept out of [`extract_on_ckpt_hook`] so the parser is testable
/// without spinning up a full trainer run.
///
/// The string forms cover the three decisions a hook makes most of the
/// time. The table form exists because the hook is really answering two
/// questions — does the run go on, and is this checkpoint worth holding
/// — and one string cannot say "hold this one and stop", which is how a
/// successful search ends.
fn parse_ckpt_control(prefix: &str, value: &LuaValue) -> Result<CkptControl, String> {
    match value {
        LuaValue::Nil => Ok(CkptControl::CONTINUE),
        LuaValue::String(s) => match s.to_str().as_deref() {
            Ok("continue") => Ok(CkptControl::CONTINUE),
            Ok("break") => Ok(CkptControl::BREAK),
            Ok("keep") => Ok(CkptControl::keep(None)),
            Ok(other) => Err(format!(
                "{prefix}: on_ckpt must return 'continue' | 'break' | 'keep' | nil \
                 or a table, got {other:?}"
            )),
            Err(e) => Err(format!(
                "{prefix}: on_ckpt returned a non-UTF-8 string: {e}"
            )),
        },
        LuaValue::Table(t) => parse_ckpt_control_table(prefix, t),
        other => Err(format!(
            "{prefix}: on_ckpt must return string, table or nil, got {}",
            other.type_name()
        )),
    }
}

/// Parse the table form: `{ action = "continue"|"break"|nil,
/// keep = true|false|"<reason>"|nil }`.
///
/// `keep` as a string carries the caller's own note straight through to
/// the candidate record, so a band label survives into the run outcome
/// without the trainer needing to know what a band is.
///
/// A key that is neither of those two is refused. Ignoring it would
/// make `{ actoin = "break", keep = "cleared" }` a run that holds the
/// checkpoint and then trains to `steps` anyway, with a candidates list
/// that looks exactly as the caller intended — nothing downstream could
/// tell that decision apart from the one that was asked for. A
/// misspelled *value* is already refused by name; a misspelled *key*
/// silently inverting a decision is the same defect one level up.
fn parse_ckpt_control_table(prefix: &str, t: &LuaTable) -> Result<CkptControl, String> {
    let mut unknown: Vec<String> = Vec::new();
    for pair in t.pairs::<LuaValue, LuaValue>() {
        let (key, _) = pair.map_err(|e| format!("{prefix}: on_ckpt table is unreadable: {e}"))?;
        match &key {
            LuaValue::String(s) => match s.to_str() {
                Ok(name) if matches!(name.as_ref(), "action" | "keep") => {}
                Ok(name) => unknown.push(format!("{name:?}")),
                Err(_) => unknown.push("<non-UTF-8 key>".to_string()),
            },
            other => unknown.push(format!("<{} key>", other.type_name())),
        }
    }
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!(
            "{prefix}: on_ckpt table has unknown key(s) {}; only 'action' and 'keep' are read",
            unknown.join(", ")
        ));
    }

    let action: Option<mlua::String> = t
        .get("action")
        .map_err(|e| format!("{prefix}: on_ckpt table field 'action' is unreadable: {e}"))?;
    let flow = match action.as_ref().map(|s| s.to_str()).transpose() {
        Ok(None) => CkptFlow::Continue,
        Ok(Some(s)) => match s.as_ref() {
            "continue" => CkptFlow::Continue,
            "break" => CkptFlow::Break,
            other => {
                return Err(format!(
                    "{prefix}: on_ckpt table field 'action' must be \
                     'continue' | 'break' | nil, got {other:?}"
                ))
            }
        },
        Err(e) => {
            return Err(format!(
                "{prefix}: on_ckpt table field 'action' is not UTF-8: {e}"
            ))
        }
    };

    let keep_value: LuaValue = t
        .get("keep")
        .map_err(|e| format!("{prefix}: on_ckpt table field 'keep' is unreadable: {e}"))?;
    let keep = match keep_value {
        LuaValue::Nil | LuaValue::Boolean(false) => None,
        LuaValue::Boolean(true) => Some(KeepMark { reason: None }),
        LuaValue::String(s) => match s.to_str() {
            Ok(reason) => Some(KeepMark {
                reason: Some(reason.to_string()),
            }),
            Err(e) => {
                return Err(format!(
                    "{prefix}: on_ckpt table field 'keep' is not UTF-8: {e}"
                ))
            }
        },
        other => {
            return Err(format!(
                "{prefix}: on_ckpt table field 'keep' must be boolean, string or nil, got {}",
                other.type_name()
            ))
        }
    };

    Ok(CkptControl { flow, keep })
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
    // CamelCase here, snake_case in the sibling family: see
    // [`parse_schedule`] on why the two are not merged.
    let schedule = match opts
        .get::<Option<String>>("schedule")?
        .as_deref()
        .unwrap_or("CosineWithWarmup")
    {
        "CosineWithWarmup" => ScheduleKind::CosineWithWarmup,
        "Constant" => ScheduleKind::Constant,
        "Linear" => ScheduleKind::Linear,
        "WarmupStableDecay" => ScheduleKind::WarmupStableDecay,
        other => {
            return Err(LuaError::external(format!(
                "{prefix}: opts.schedule must be one of \
                 \"CosineWithWarmup\" / \"Constant\" / \"Linear\" / \
                 \"WarmupStableDecay\" (got {other:?})"
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
    apply_optional_overrides(prefix, &mut cfg, opts)?;
    Ok(cfg)
}

/// Apply the pass-through [`FullFtConfig`] fields that both surface
/// families accept verbatim (no surface-specific validation):
/// `grad_accum` / `weight_decay` / `ckpt_every` / `ckpt_keep` /
/// `init_from` / `mask_disallowed_logits`.
///
/// Keeping the reads here is what makes each of those keys appear
/// exactly once in the repo — the pre-consolidation `run_*` surfaces
/// reached the same keys indirectly (they delegated to the
/// `full_ft`-side extractor after rewriting the caller table), and the
/// doc claim that they were "NOT exposed" did not match the code.
///
/// Wrong-type input surfaces as an mlua type-mismatch error (no
/// prefix rewrite) exactly as it did before. `prefix` is threaded
/// through for the one field that carries a validation rule of its
/// own (`init_from`, which refuses an empty path), so that refusal
/// reads like every other error from the caller's surface.
fn apply_optional_overrides(
    prefix: &str,
    cfg: &mut FullFtConfig,
    opts: &LuaTable,
) -> LuaResult<()> {
    if let Some(v) = opts.get::<Option<usize>>("grad_accum")? {
        cfg.grad_accum = v;
    }
    if let Some(v) = opts.get::<Option<f64>>("weight_decay")? {
        cfg.weight_decay = v;
    }
    if let Some(v) = opts.get::<Option<String>>("optimizer")? {
        cfg.optimizer = parse_optimizer(prefix, &v)?;
    }
    if let Some(v) = opts.get::<Option<f64>>("min_lr")? {
        cfg.min_lr = v;
    }
    if let Some(v) = opts.get::<Option<usize>>("decay_steps")? {
        cfg.decay_steps = Some(v);
    }
    if let Some(v) = opts.get::<Option<f64>>("beta1")? {
        cfg.beta1 = v;
    }
    if let Some(v) = opts.get::<Option<f64>>("beta2")? {
        cfg.beta2 = v;
    }
    if let Some(v) = opts.get::<Option<f64>>("eps")? {
        cfg.eps = v;
    }
    if let Some(v) = opts.get::<Option<usize>>("ckpt_every")? {
        cfg.ckpt_every = v;
    }
    if let Some(v) = opts.get::<Option<usize>>("ckpt_keep")? {
        cfg.ckpt_keep = v;
    }
    // `init_from` names a checkpoint the model's variables are restored
    // from before the first step. An empty string is refused rather
    // than read as "no checkpoint": the two spellings for "do not
    // resume" would then be `nil` and `""`, and a caller whose path
    // variable came back empty would get a fresh run under a config
    // that says it resumed. The restore itself is strict — anything
    // short of a complete one is a `TrainError::Restore` and the run
    // does not start.
    if let Some(v) = opts.get::<Option<String>>("init_from")? {
        if v.is_empty() {
            return Err(LuaError::external(format!(
                "{prefix}: opts.init_from must be a non-empty checkpoint path \
                 (omit the key to train from the handle as built)"
            )));
        }
        cfg.init_from = Some(PathBuf::from(v));
    }
    // Whether the loss scores each target among the ids its position
    // allowed instead of among the whole vocabulary. Independent of
    // whether the model is *handed* those ids (that is the training
    // entry point's business), so both switches are readable here and
    // neither implies the other.
    if let Some(v) = opts.get::<Option<bool>>("mask_disallowed_logits")? {
        cfg.mask_disallowed_logits = v;
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
/// The snake_case vocabulary, used by the `full_ft` / `lora` / `distill`
/// family.
///
/// The sibling `run_*` family spells the same values in CamelCase and
/// keeps its own reader below. The split is a Lua-visible difference
/// between the two contracts (see the module docs), so a name is added
/// to both rather than one vocabulary being widened to swallow the
/// other.
fn parse_schedule(prefix: &str, s: &str) -> LuaResult<ScheduleKind> {
    ScheduleKind::parse(s).ok_or_else(|| {
        LuaError::external(format!(
            "{prefix}: unknown schedule '{s}' (expected one of {})",
            ScheduleKind::NAMES.join(" / ")
        ))
    })
}

/// Read `opts.optimizer`, naming the alternatives on a miss.
fn parse_optimizer(prefix: &str, s: &str) -> LuaResult<OptimizerKind> {
    OptimizerKind::parse(s).ok_or_else(|| {
        LuaError::external(format!(
            "{prefix}: unknown optimizer '{s}' (expected one of {})",
            OptimizerKind::NAMES.join(" / ")
        ))
    })
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
        TrainError::Hook(inner) => format!("{prefix}: {inner}"),
        // `opts.init_from` named a checkpoint the variables could not
        // be restored from. The hint names the option rather than the
        // Rust field, because that is what the caller edits.
        TrainError::Restore(inner) => format!(
            "{prefix}: opts.init_from could not be restored: {inner}; \
             the run did not start"
        ),
        TrainError::InitFromUnsupported => format!(
            "{prefix}: opts.init_from is not supported by this surface; \
             restore the base handle before wrapping it"
        ),
        // The two channel mismatches. Both are a disagreement between
        // how the model was built and what the dataset carries, so the
        // hint points at the pair rather than at either side.
        TrainError::MissingConditions { rows } => format!(
            "{prefix}: the model declares a conditioning table but a batch of {rows} row(s) \
             carried no conditions; attach them when the dataset is built"
        ),
        TrainError::UnexpectedConditions { rows, conds } => format!(
            "{prefix}: a batch of {rows} row(s) carries {conds} condition(s) but the model \
             was built without a conditioning table; add cond_slots to the preset opts \
             or drop the conditions from the dataset"
        ),
        TrainError::MissingAllowedSets { rows, needed } => format!(
            "{prefix}: a batch of {rows} row(s) carries no allowed-id sets, which this run \
             requires ({needed}); attach them when the dataset is built"
        ),
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
    fn opts_read_init_from_and_the_loss_mask_switch() {
        let lua = Lua::new();
        let path = LuaValue::String(lua.create_string("/tmp/run-step40.safetensors").unwrap());
        let opts = opts_from(
            &lua,
            &[
                ("init_from", path),
                ("mask_disallowed_logits", LuaValue::Boolean(true)),
            ],
        );
        let cfg = extract_full_ft_opts("alc.nn.trainer", Some(&opts)).expect("channel opts");
        assert_eq!(
            cfg.init_from.as_deref(),
            Some(std::path::Path::new("/tmp/run-step40.safetensors"))
        );
        assert!(cfg.mask_disallowed_logits);

        // Both surfaces read the same field set, so the strict
        // extractor has to see them too.
        let strict = opts_from(
            &lua,
            &[
                ("lr", LuaValue::Number(1e-4)),
                ("batch", LuaValue::Integer(2)),
                ("steps", LuaValue::Integer(3)),
                (
                    "init_from",
                    LuaValue::String(lua.create_string("/tmp/base.safetensors").unwrap()),
                ),
            ],
        );
        let cfg = extract_run_train_cfg("alc.nn.trainer.run_full_ft", &strict).expect("strict");
        assert_eq!(
            cfg.init_from.as_deref(),
            Some(std::path::Path::new("/tmp/base.safetensors"))
        );
        // Absent keys keep the crate default, which is "no resume" and
        // "score against the whole vocabulary".
        let d = FullFtConfig::default();
        assert!(d.init_from.is_none());
        assert!(!d.mask_disallowed_logits);
    }

    #[test]
    fn opts_refuse_an_empty_init_from_path() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[(
                "init_from",
                LuaValue::String(lua.create_string("").unwrap()),
            )],
        );
        let err = extract_full_ft_opts("alc.nn.trainer.full_ft", Some(&opts))
            .expect_err("an empty init_from must be refused");
        assert!(
            err.to_string()
                .contains("alc.nn.trainer.full_ft: opts.init_from must be a non-empty"),
            "message: {err}"
        );
    }

    #[test]
    fn train_err_to_lua_points_channel_errors_at_the_option() {
        let err = train_err_to_lua(
            "alc.nn.trainer.run_full_ft",
            TrainError::InitFromUnsupported,
        );
        assert!(
            err.to_string().contains("opts.init_from is not supported"),
            "message: {err}"
        );

        let err = train_err_to_lua(
            "alc.nn.trainer.run_full_ft",
            TrainError::UnexpectedConditions { rows: 2, conds: 2 },
        );
        assert!(
            err.to_string().contains("cond_slots"),
            "the hint must name the preset option that adds the table: {err}"
        );

        let err = train_err_to_lua(
            "alc.nn.trainer.run_full_ft",
            TrainError::MissingAllowedSets {
                rows: 2,
                needed: "the model input",
            },
        );
        assert!(
            err.to_string().contains("the model input"),
            "the hint must carry which switch asked for the sets: {err}"
        );
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
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "linear").unwrap(),
            ScheduleKind::Linear
        ));
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "wsd").unwrap(),
            ScheduleKind::WarmupStableDecay
        ));
        assert!(matches!(
            parse_schedule("alc.nn.trainer", "warmup_stable_decay").unwrap(),
            ScheduleKind::WarmupStableDecay
        ));

        // CamelCase belongs to the sibling `run_*` family and is not
        // read here — the two vocabularies stay apart.
        let err = parse_schedule("alc.nn.trainer", "Linear").expect_err("wrong vocabulary");
        assert!(err.to_string().contains("Linear"), "message: {err}");

        let err = parse_schedule("alc.nn.trainer", "triangular").expect_err("unknown");
        assert!(err.to_string().contains("triangular"), "message: {err}");
        // The refusal names the alternatives rather than only the miss.
        assert!(err.to_string().contains("cosine"), "message: {err}");
    }

    #[test]
    fn optimizer_parser_accepts_known_and_names_the_alternatives() {
        assert!(matches!(
            parse_optimizer("alc.nn.trainer", "adamw").unwrap(),
            OptimizerKind::AdamW
        ));
        assert!(matches!(
            parse_optimizer("alc.nn.trainer", "lion").unwrap(),
            OptimizerKind::Lion
        ));
        let err = parse_optimizer("alc.nn.trainer", "sgd").expect_err("unknown");
        let text = err.to_string();
        assert!(text.contains("sgd") && text.contains("lion"), "{text}");
    }

    /// The knobs that were reachable in Rust and not from Lua: every
    /// one of them now arrives, and the defaults are unchanged when the
    /// caller says nothing.
    #[test]
    fn the_optimizer_and_schedule_knobs_reach_the_config() {
        let lua = Lua::new();
        let cfg = extract_full_ft_opts("alc.nn.trainer", None).expect("defaults");
        assert_eq!(cfg.optimizer, OptimizerKind::AdamW);
        assert!(cfg.min_lr.abs() < f64::EPSILON);
        assert_eq!(cfg.decay_steps, None);
        assert!((cfg.beta1 - 0.9).abs() < 1e-12);
        assert!((cfg.beta2 - 0.999).abs() < 1e-12);
        assert!((cfg.eps - 1e-8).abs() < 1e-20);

        let t = lua.create_table().expect("opts");
        t.set("optimizer", "lion").unwrap();
        t.set("schedule", "wsd").unwrap();
        t.set("min_lr", 1e-5).unwrap();
        t.set("decay_steps", 250usize).unwrap();
        t.set("beta1", 0.95).unwrap();
        t.set("beta2", 0.99).unwrap();
        t.set("eps", 1e-6).unwrap();
        let cfg = extract_full_ft_opts("alc.nn.trainer", Some(&t)).expect("opts");
        assert_eq!(cfg.optimizer, OptimizerKind::Lion);
        assert_eq!(cfg.schedule, ScheduleKind::WarmupStableDecay);
        assert!((cfg.min_lr - 1e-5).abs() < 1e-20);
        assert_eq!(cfg.decay_steps, Some(250));
        assert!((cfg.beta1 - 0.95).abs() < 1e-12);
        assert!((cfg.beta2 - 0.99).abs() < 1e-12);
        assert!((cfg.eps - 1e-6).abs() < 1e-20);
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
    fn parse_ckpt_control_accepts_nil_and_continue_and_break() {
        let lua = Lua::new();
        assert_eq!(
            parse_ckpt_control("prefix", &LuaValue::Nil).unwrap(),
            CkptControl::CONTINUE
        );
        let s_cont = LuaValue::String(lua.create_string("continue").unwrap());
        assert_eq!(
            parse_ckpt_control("prefix", &s_cont).unwrap(),
            CkptControl::CONTINUE
        );
        let s_break = LuaValue::String(lua.create_string("break").unwrap());
        assert_eq!(
            parse_ckpt_control("prefix", &s_break).unwrap(),
            CkptControl::BREAK
        );
    }

    /// `"keep"` holds the checkpoint without stopping the run — the
    /// ordinary middle of a checkpoint search.
    #[test]
    fn parse_ckpt_control_accepts_keep_string() {
        let lua = Lua::new();
        let s_keep = LuaValue::String(lua.create_string("keep").unwrap());
        let control = parse_ckpt_control("prefix", &s_keep).unwrap();
        assert_eq!(control.flow, CkptFlow::Continue);
        assert_eq!(control.keep, Some(KeepMark { reason: None }));
    }

    /// The combination no single string can express: hold this one and
    /// stop. This is what an earlier ABI had no room for, and why the
    /// hook used to copy the file out from under the rotation itself.
    #[test]
    fn parse_ckpt_control_table_says_keep_and_break_together() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        t.set("action", "break").unwrap();
        t.set("keep", "tier-3").unwrap();
        let control = parse_ckpt_control("prefix", &LuaValue::Table(t)).unwrap();
        assert_eq!(control.flow, CkptFlow::Break);
        assert_eq!(
            control.keep,
            Some(KeepMark {
                reason: Some("tier-3".to_string())
            }),
            "a string `keep` carries the caller's own label through"
        );
    }

    /// An omitted `action` means continue, and `keep = true` marks the
    /// checkpoint without a reason.
    #[test]
    fn parse_ckpt_control_table_defaults_action_to_continue() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        t.set("keep", true).unwrap();
        let control = parse_ckpt_control("prefix", &LuaValue::Table(t)).unwrap();
        assert_eq!(control.flow, CkptFlow::Continue);
        assert_eq!(control.keep, Some(KeepMark { reason: None }));

        let empty = lua.create_table().unwrap();
        assert_eq!(
            parse_ckpt_control("prefix", &LuaValue::Table(empty)).unwrap(),
            CkptControl::CONTINUE,
            "an empty table is the same decision as nil"
        );
    }

    /// `keep = false` is a decision, not a marker: it must not pin.
    #[test]
    fn parse_ckpt_control_table_treats_keep_false_as_no_keep() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        t.set("keep", false).unwrap();
        let control = parse_ckpt_control("prefix", &LuaValue::Table(t)).unwrap();
        assert_eq!(control.keep, None);
    }

    /// A misspelled key must not read as a decision the caller did not
    /// make. `{ actoin = "break", keep = … }` would otherwise hold the
    /// checkpoint and train on to `steps`, and the candidates list
    /// would look exactly right.
    #[test]
    fn parse_ckpt_control_table_refuses_unknown_keys() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        t.set("actoin", "break").unwrap();
        t.set("keep", "cleared").unwrap();
        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Table(t)).unwrap_err();
        assert!(
            err.contains("alc.nn.trainer") && err.contains("actoin") && err.contains("unknown key"),
            "message must name prefix + the offending key: {err}"
        );

        // The mirror case: the flow is right, the keep is silently lost.
        let t = lua.create_table().unwrap();
        t.set("action", "break").unwrap();
        t.set("kept", "x").unwrap();
        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Table(t)).unwrap_err();
        assert!(err.contains("kept"), "{err}");

        // A non-string key is just as unreadable an intent.
        let t = lua.create_table().unwrap();
        t.set(1, "break").unwrap();
        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Table(t)).unwrap_err();
        assert!(err.contains("unknown key"), "{err}");
    }

    #[test]
    fn parse_ckpt_control_table_rejects_bad_action_and_keep_types() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        t.set("action", "halt").unwrap();
        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Table(t)).unwrap_err();
        assert!(
            err.contains("alc.nn.trainer") && err.contains("action") && err.contains("halt"),
            "message must name prefix + field + offending value: {err}"
        );

        let t = lua.create_table().unwrap();
        t.set("keep", 3).unwrap();
        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Table(t)).unwrap_err();
        assert!(
            err.contains("keep") && err.contains("integer"),
            "message must name the field and the type it got: {err}"
        );
    }

    #[test]
    fn parse_ckpt_control_rejects_other_strings_and_wrong_types() {
        let lua = Lua::new();
        let bad_str = LuaValue::String(lua.create_string("halt").unwrap());
        let err = parse_ckpt_control("alc.nn.trainer", &bad_str).unwrap_err();
        assert!(
            err.contains("alc.nn.trainer") && err.contains("on_ckpt") && err.contains("halt"),
            "message must name prefix + surface + offending value: {err}"
        );

        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Boolean(true)).unwrap_err();
        assert!(
            err.contains("must return string, table or nil") && err.contains("boolean"),
            "wrong-type message must state the expected shape + observed type: {err}"
        );

        let err = parse_ckpt_control("alc.nn.trainer", &LuaValue::Integer(1)).unwrap_err();
        assert!(
            err.contains("must return string, table or nil") && err.contains("integer"),
            "wrong-type message must state the observed type: {err}"
        );
    }

    #[test]
    fn extract_on_ckpt_hook_returns_none_when_key_absent() {
        let lua = Lua::new();
        let hook = extract_on_ckpt_hook("alc.nn.trainer", &lua, None).unwrap();
        assert!(hook.is_none(), "opts=None must yield hook=None");

        let empty = lua.create_table().unwrap();
        let hook = extract_on_ckpt_hook("alc.nn.trainer", &lua, Some(&empty)).unwrap();
        assert!(
            hook.is_none(),
            "opts table without on_ckpt must yield hook=None"
        );
    }

    #[test]
    fn extract_on_ckpt_hook_refuses_hook_without_positive_ckpt_every() {
        // `ckpt_every` defaults to 0 (mid-run checkpoints disabled), so
        // an `on_ckpt` without it registers a hook that can never fire.
        // Both the absent and the explicit-zero spellings must be loud.
        let lua = Lua::new();
        let callback = lua
            .create_function(|_, _: LuaValue| Ok(()))
            .expect("create on_ckpt stub");

        // `CkptHook` is a boxed closure (no `Debug`), so `expect_err`
        // is unavailable — unwrap the error arm by hand.
        fn expect_refusal(result: LuaResult<Option<CkptHook>>, what: &str) -> LuaError {
            match result {
                Ok(_) => panic!("{what}"),
                Err(e) => e,
            }
        }

        let no_ckpt_every = opts_from(&lua, &[("on_ckpt", LuaValue::Function(callback.clone()))]);
        let err = expect_refusal(
            extract_on_ckpt_hook("alc.nn.trainer.run_full_ft", &lua, Some(&no_ckpt_every)),
            "on_ckpt without ckpt_every must be refused",
        );
        assert!(
            err.to_string().contains("alc.nn.trainer.run_full_ft:")
                && err.to_string().contains("opts.ckpt_every > 0"),
            "message must name the surface + the missing key: {err}"
        );

        let zero_ckpt_every = opts_from(
            &lua,
            &[
                ("on_ckpt", LuaValue::Function(callback.clone())),
                ("ckpt_every", LuaValue::Integer(0)),
            ],
        );
        let err = expect_refusal(
            extract_on_ckpt_hook("alc.nn.trainer.full_ft", &lua, Some(&zero_ckpt_every)),
            "on_ckpt with ckpt_every = 0 must be refused",
        );
        assert!(
            err.to_string().contains("alc.nn.trainer.full_ft:")
                && err.to_string().contains("opts.ckpt_every > 0"),
            "message: {err}"
        );

        // Positive `ckpt_every` keeps the hook.
        let ok = opts_from(
            &lua,
            &[
                ("on_ckpt", LuaValue::Function(callback)),
                ("ckpt_every", LuaValue::Integer(2)),
            ],
        );
        let hook = extract_on_ckpt_hook("alc.nn.trainer.run_full_ft", &lua, Some(&ok))
            .expect("ckpt_every = 2 must yield a hook");
        assert!(hook.is_some(), "positive ckpt_every must yield Some(hook)");
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
