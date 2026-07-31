//! `alc.nn.sampler` / `alc.nn.constraint` — Layer 3 of the sampler plan
//! (feature `nn`).
//!
//! Builds the `algocline_nn::sampling` types from Lua, composes them, and
//! lets a Lua function be a sampler in its own right:
//!
//! ```text
//! alc.nn.sampler.greedy()                                  -> Sampler
//! alc.nn.sampler.temperature(temperature, seed)            -> Sampler
//! alc.nn.sampler.top_k_top_p(top_k, top_p, temp, seed)     -> Sampler
//! alc.nn.sampler.lua(function(logits) ... end)             -> Sampler
//! alc.nn.sampler.constrained(sampler, constraint)          -> Sampler
//!
//! alc.nn.constraint.stop_tokens({ id, ... })               -> Constraint
//! alc.nn.constraint.allow_list({ id, ... })                -> Constraint
//! alc.nn.constraint.regex(pattern, vocab)                  -> Constraint
//! alc.nn.constraint.json_schema(schema, vocab)             -> Constraint
//!
//! sampler:sample(logits)  -> token id
//! sampler:is_done()       -> bool
//! sampler:reset()
//! ```
//!
//! # Why there is no schedule primitive
//!
//! The generation loop already lives in Lua (`nn_gen.rs`), so swapping
//! the active sampler mid-generation needs no new API — it is an `if`:
//!
//! ```lua
//! local greedy = alc.nn.sampler.greedy()
//! local temp   = alc.nn.sampler.temperature(0.8, 42)
//! local s = h:generate_session(alc.nn.tokenize("gpt2", prompt))
//! for _ = 1, 64 do
//!     local logits = s:next_logits()
//!     -- the "schedule" is plain Lua: sharp opening, softer tail
//!     local active = s:position() < 20 and greedy or temp
//!     s:append(active:sample(logits))
//! end
//! ```
//!
//! A Rust-side scheduler would have to re-express that control flow in a
//! configuration language, and every question it could answer
//! ("temperature ramp", "switch on a stop word") is one line of Lua here.
//!
//! # Move semantics on composition
//!
//! [`ConstrainedSampler`] *owns* its inner sampler, and a sampler owns
//! its RNG. Two Lua handles pointing at one RNG would interleave draws
//! from generations that each believe they are reproducible from their
//! seed — a reproducibility bug that reports itself as nothing at all.
//! `alc.nn.sampler.constrained` therefore **consumes** both arguments:
//! the inner sampler and the constraint are moved out of their handles,
//! and any later use of those handles is a loud error rather than a
//! surprise alias. Build a second sampler instead of sharing one.
//!
//! Rebuilding is therefore the intended shape for a per-decision legal
//! mask, not a workaround for it:
//!
//! ```lua
//! for turn, legal_ids in ipairs(decisions) do
//!     -- one fresh chain per decision; the seed is derived explicitly
//!     -- so the turn stays reproducible on its own
//!     local s = alc.nn.sampler.constrained(
//!         alc.nn.sampler.temperature(0.8, base_seed + turn),
//!         alc.nn.constraint.allow_list(legal_ids)
//!     )
//!     local id = s:sample(session:next_logits())
//!     session:append(id)
//! end
//! ```
//!
//! There is deliberately no API to hand a live constraint a new id list:
//! a mutable legal set would make a draw depend on call order instead of
//! on the seed and the prefix.
//!
//! # Custom samplers see masked logits
//!
//! A Lua callback wrapped by `alc.nn.sampler.constrained` receives the
//! row *after* masking, with forbidden ids at `-inf`. It is expected to
//! respect that (`logits:top(n)` naturally does — masked entries sort
//! last); the constraint layer does not re-check the inner sampler's
//! answer, exactly as it does not for the Rust impls.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, TryLockError};

use algocline_nn::sampling::{
    AllowListConstraint, ConstrainedSampler, Constraint, GreedySampler, JsonSchemaConstraint,
    RegexConstraint, Sampler, StopTokensConstraint, TemperatureSampler, TopKTopPSampler,
};
use candle_core::{Result as CandleResult, Tensor};
use mlua::prelude::*;
use mlua::LuaSerdeExt;

use super::nn_gen::{load_tokenizer, LogitsHandle};

/// Type-erased sampler. See the `Box<dyn Sampler + Send>` impl in
/// `algocline_nn::sampling` for why erasure is possible at all.
type BoxedSampler = Box<dyn Sampler + Send>;

/// Type-erased constraint, same rationale.
type BoxedConstraint = Box<dyn Constraint + Send>;

/// The one constrained shape the bridge builds. Spelled out once so the
/// erasure boxes appear in a single place.
type ErasedConstrained = ConstrainedSampler<BoxedSampler, BoxedConstraint>;

/// A sampler as Lua holds it.
///
/// The two arms exist because `is_done` / `reset` are inherent methods on
/// [`ConstrainedSampler`], not part of the [`Sampler`] trait — the trait
/// deliberately has no notion of termination. Erasing a constrained
/// sampler into a plain `Box<dyn Sampler>` would therefore lose exactly
/// the two methods a generation loop polls.
enum ErasedSampler {
    /// Any Layer 1 sampler, or a Lua callback.
    Plain(BoxedSampler),
    /// A Layer 2 composition, kept distinguishable so `is_done` / `reset`
    /// stay reachable.
    Constrained(ErasedConstrained),
}

impl ErasedSampler {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        match self {
            Self::Plain(sampler) => sampler.sample(logits),
            Self::Constrained(sampler) => sampler.sample(logits),
        }
    }

    /// Whether the constraint considers the generated prefix terminal.
    ///
    /// An unconstrained sampler has no opinion about termination and no
    /// prefix to form one from, so it answers `false` forever: a loop
    /// polling `is_done` on a plain sampler is bounded by its own token
    /// budget, which is the pre-existing behaviour of every Layer 1
    /// sampler.
    fn is_done(&self) -> bool {
        match self {
            Self::Plain(_) => false,
            Self::Constrained(sampler) => sampler.is_done(),
        }
    }

    /// Drop the generated prefix so the sampler can drive another
    /// generation. A no-op on a plain sampler, which holds no prefix —
    /// deliberately *not* an error, so a loop that resets between
    /// generations works with either kind of sampler.
    fn reset(&mut self) {
        if let Self::Constrained(sampler) = self {
            sampler.reset();
        }
    }
}

/// Lua-facing sampler handle.
///
/// `Option` encodes the move semantics from the module doc: `None` means
/// the sampler was moved into a [`ConstrainedSampler`] and this handle is
/// spent. The `Mutex` is there for mlua's `send` feature (the VM is
/// single-threaded) and doubles as the re-entrancy guard — a Lua callback
/// that tries to re-enter the sampler that invoked it finds the slot
/// locked and gets an error instead of a deadlock.
pub(super) struct SamplerHandle {
    inner: Mutex<Option<ErasedSampler>>,
}

impl SamplerHandle {
    fn new(sampler: ErasedSampler) -> Self {
        Self {
            inner: Mutex::new(Some(sampler)),
        }
    }

    /// Wrap a concrete Layer 1 sampler (or the Lua bridge).
    fn plain<S: Sampler + Send + 'static>(sampler: S) -> Self {
        Self::new(ErasedSampler::Plain(Box::new(sampler)))
    }

    fn sample(&self, logits: &Tensor) -> LuaResult<u32> {
        const ENTRY: &str = "alc.nn sampler:sample";
        let mut slot = lock_slot(&self.inner, ENTRY, "sampler")?;
        let sampler = slot.as_mut().ok_or_else(|| moved_sampler(ENTRY))?;
        sampler
            .sample(logits)
            .map_err(|e| LuaError::external(format!("{ENTRY}: {e}")))
    }

    fn is_done(&self) -> LuaResult<bool> {
        const ENTRY: &str = "alc.nn sampler:is_done";
        let slot = lock_slot(&self.inner, ENTRY, "sampler")?;
        let sampler = slot.as_ref().ok_or_else(|| moved_sampler(ENTRY))?;
        Ok(sampler.is_done())
    }

    fn reset(&self) -> LuaResult<()> {
        const ENTRY: &str = "alc.nn sampler:reset";
        let mut slot = lock_slot(&self.inner, ENTRY, "sampler")?;
        let sampler = slot.as_mut().ok_or_else(|| moved_sampler(ENTRY))?;
        sampler.reset();
        Ok(())
    }
}

impl mlua::UserData for SamplerHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // `add_method` (shared borrow) rather than `add_method_mut`: a
        // Lua sampler callback runs *inside* this call, and a mutable
        // userdata borrow would make any unrelated access to the same
        // handle from that callback fail with mlua's borrow error
        // instead of the specific message the mutex guard produces.
        methods.add_method("sample", |_, this, logits: LuaUserDataRef<LogitsHandle>| {
            this.sample(logits.tensor())
        });
        methods.add_method("is_done", |_, this, ()| this.is_done());
        methods.add_method("reset", |_, this, ()| this.reset());
    }
}

/// Lua-facing constraint handle.
///
/// Opaque: a constraint answers questions about a prefix the sampler
/// owns, so there is nothing to ask it from Lua. Its whole purpose is to
/// be handed to `alc.nn.sampler.constrained`, which consumes it — hence
/// the same `Option` slot as [`SamplerHandle`].
pub(super) struct ConstraintHandle {
    inner: Mutex<Option<BoxedConstraint>>,
}

impl ConstraintHandle {
    fn new<C: Constraint + Send + 'static>(constraint: C) -> Self {
        Self {
            inner: Mutex::new(Some(Box::new(constraint))),
        }
    }
}

impl mlua::UserData for ConstraintHandle {}

/// A Lua function acting as a [`Sampler`].
///
/// Holds a [`WeakLua`] rather than a `Lua`: the handle owning this bridge
/// lives *in* that VM, so a strong reference would be a cycle keeping the
/// whole state alive for as long as the sampler exists.
struct LuaSamplerBridge {
    lua: WeakLua,
    callback: LuaFunction,
}

impl Sampler for LuaSamplerBridge {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        let vocab = logits.dims1()?;
        let lua = self
            .lua
            .try_upgrade()
            .ok_or_else(|| msg("alc.nn.sampler.lua: the Lua state owning this callback is gone"))?;
        // A fresh handle per call, wrapping whatever row reached us —
        // under a constraint that is the *masked* row, which is what the
        // callback must see to respect the mask. `Tensor` clone is a
        // refcount bump, not a copy.
        let handle = lua
            .create_userdata(LogitsHandle::from_tensor(logits.clone()))
            .map_err(|e| {
                msg(format!(
                    "alc.nn.sampler.lua: cannot pass logits to Lua: {e}"
                ))
            })?;
        let returned: LuaValue = self
            .callback
            .call(handle)
            .map_err(|e| msg(format!("alc.nn.sampler.lua: callback failed: {e}")))?;
        token_from_lua(returned, vocab)
    }
}

/// Validate what a Lua sampler callback returned.
///
/// The contract every [`Sampler`] upholds is "the returned id is a valid
/// vocab index"; a Lua callback cannot be trusted to, so it is checked
/// here. Absorbing a bad answer (clamping, rounding) would put a token
/// nobody asked for into the stream and leave the actual bug — an
/// off-by-one over a `top(n)` table, a forgotten `math.floor` — to
/// surface as gibberish output much later.
fn token_from_lua(value: LuaValue, vocab: usize) -> CandleResult<u32> {
    let id = match &value {
        LuaValue::Integer(id) => *id,
        // Lua 5.4 keeps `3.0` a float even though it names an integer.
        // Accepting an integral float is not laxity: arithmetic on ids
        // (`#t / 2`, a division) produces one, and the fractional case
        // below still fails loudly.
        LuaValue::Number(n) if n.is_finite() && n.fract() == 0.0 => *n as i64,
        other => {
            return Err(msg(format!(
                "alc.nn.sampler.lua: callback must return an integer token id, got {}",
                other.type_name()
            )))
        }
    };
    let bound = i64::try_from(vocab).map_err(|e| {
        msg(format!(
            "alc.nn.sampler.lua: vocab {vocab} out of range: {e}"
        ))
    })?;
    if id < 0 || id >= bound {
        return Err(msg(format!(
            "alc.nn.sampler.lua: callback returned token id {id}, \
             which is outside the vocabulary (0..{vocab})"
        )));
    }
    u32::try_from(id).map_err(|e| {
        msg(format!(
            "alc.nn.sampler.lua: token id {id} does not fit in u32: {e}"
        ))
    })
}

/// Register `alc.nn.sampler.*` and `alc.nn.constraint.*`.
///
/// `nn_dir` is only read when a constraint resolves its vocabulary from a
/// tokenizer preset name; it goes through the same
/// `<nn_dir>/tokenizers` cache as `alc.nn.tokenize`, so a constraint and
/// the prompt it constrains always agree on what a token id means.
pub(super) fn register_sampler_ns(
    lua: &Lua,
    nn_table: &LuaTable,
    nn_dir: PathBuf,
) -> LuaResult<()> {
    let sampler_ns = lua.create_table()?;

    let greedy = lua.create_function(|_, ()| Ok(SamplerHandle::plain(GreedySampler)))?;
    sampler_ns.set("greedy", greedy)?;

    // `seed` is required rather than defaulting to entropy: a default
    // seed would make every stochastic generation irreproducible by
    // omission, and the reproducibility guarantee is the reason the
    // samplers carry their own RNG at all.
    let temperature = lua.create_function(|_, (temperature, seed): (f32, u64)| {
        Ok(SamplerHandle::plain(TemperatureSampler::new(
            temperature,
            seed,
        )))
    })?;
    sampler_ns.set("temperature", temperature)?;

    let top_k_top_p = lua.create_function(
        |_, (top_k, top_p, temperature, seed): (Option<usize>, Option<f32>, f32, u64)| {
            Ok(SamplerHandle::plain(TopKTopPSampler::new(
                top_k,
                top_p,
                temperature,
                seed,
            )))
        },
    )?;
    sampler_ns.set("top_k_top_p", top_k_top_p)?;

    let lua_sampler = lua.create_function(|lua, callback: LuaFunction| {
        Ok(SamplerHandle::plain(LuaSamplerBridge {
            lua: lua.weak(),
            callback,
        }))
    })?;
    sampler_ns.set("lua", lua_sampler)?;

    let constrained = lua.create_function(
        |_,
         (inner, constraint): (
            LuaUserDataRef<SamplerHandle>,
            LuaUserDataRef<ConstraintHandle>,
        )| { constrained_impl(&inner, &constraint) },
    )?;
    sampler_ns.set("constrained", constrained)?;

    nn_table.set("sampler", sampler_ns)?;

    let constraint_ns = lua.create_table()?;

    let stop_tokens = lua.create_function(|_, ids: Vec<u32>| {
        Ok(ConstraintHandle::new(StopTokensConstraint::new(ids)))
    })?;
    constraint_ns.set("stop_tokens", stop_tokens)?;

    // No vocabulary argument: an allow list names token ids directly,
    // so it needs no surface strings to reason about — which is also why
    // it works with a tokenizer this bridge has never seen.
    let allow_list = lua.create_function(|_, ids: Vec<u32>| {
        const ENTRY: &str = "alc.nn.constraint.allow_list";
        let constraint = AllowListConstraint::new(ids)
            .map_err(|e| LuaError::external(format!("{ENTRY}: {e}")))?;
        Ok(ConstraintHandle::new(constraint))
    })?;
    constraint_ns.set("allow_list", allow_list)?;

    let regex_dir = nn_dir.clone();
    let regex = lua.create_function(move |_, (pattern, vocab): (String, LuaValue)| {
        const ENTRY: &str = "alc.nn.constraint.regex";
        let vocab = resolve_vocab(ENTRY, vocab, &regex_dir)?;
        let constraint = RegexConstraint::new(&pattern, vocab)
            .map_err(|e| LuaError::external(format!("{ENTRY}: {e}")))?;
        Ok(ConstraintHandle::new(constraint))
    })?;
    constraint_ns.set("regex", regex)?;

    let schema_dir = nn_dir;
    let json_schema = lua.create_function(move |lua, (schema, vocab): (LuaTable, LuaValue)| {
        const ENTRY: &str = "alc.nn.constraint.json_schema";
        let schema: serde_json::Value = lua.from_value(LuaValue::Table(schema)).map_err(|e| {
            LuaError::external(format!("{ENTRY}: schema is not convertible to JSON: {e}"))
        })?;
        let vocab = resolve_vocab(ENTRY, vocab, &schema_dir)?;
        let constraint = JsonSchemaConstraint::new(&schema, vocab)
            .map_err(|e| LuaError::external(format!("{ENTRY}: {e}")))?;
        Ok(ConstraintHandle::new(constraint))
    })?;
    constraint_ns.set("json_schema", json_schema)?;

    nn_table.set("constraint", constraint_ns)?;

    Ok(())
}

/// Move both arguments into one [`ConstrainedSampler`].
///
/// Both slots are inspected before either is emptied, so a call naming a
/// spent handle leaves the *other* argument intact and reusable. Taking
/// first and validating second would consume a live sampler on the way to
/// reporting someone else's mistake.
fn constrained_impl(
    inner: &SamplerHandle,
    constraint: &ConstraintHandle,
) -> LuaResult<SamplerHandle> {
    const ENTRY: &str = "alc.nn.sampler.constrained";
    let mut sampler_slot = lock_slot(&inner.inner, ENTRY, "sampler")?;
    let mut constraint_slot = lock_slot(&constraint.inner, ENTRY, "constraint")?;
    match (sampler_slot.is_some(), constraint_slot.is_some()) {
        (true, true) => {}
        (false, _) => return Err(moved_sampler(ENTRY)),
        (_, false) => return Err(moved_constraint(ENTRY)),
    }

    // Both slots were verified above, so neither `take` can observe
    // `None`; the `else` arm exists because this crate does not use
    // `unwrap` in library code, not because it is reachable.
    let (Some(sampler), Some(constraint)) = (sampler_slot.take(), constraint_slot.take()) else {
        return Err(LuaError::external(format!(
            "{ENTRY}: internal error: a handle emptied itself between the check and the move"
        )));
    };

    // Stacking works because a `ConstrainedSampler` is itself a
    // `Sampler`: constraining an already-constrained sampler applies the
    // outer mask first and the inner one last.
    let sampler: BoxedSampler = match sampler {
        ErasedSampler::Plain(sampler) => sampler,
        ErasedSampler::Constrained(sampler) => Box::new(sampler),
    };
    Ok(SamplerHandle::new(ErasedSampler::Constrained(
        ConstrainedSampler::new(sampler, constraint),
    )))
}

/// Resolve a constraint's vocabulary.
///
/// Two shapes, because the two callers are different: a preset name is
/// what production code passes (the surface strings must come from the
/// very tokenizer that produced the prompt), while a table of strings is
/// what a test or a hand-rolled tokenizer needs — and inventing a fake
/// preset on disk to reach the first path would be the alternative.
fn resolve_vocab(entry: &str, value: LuaValue, nn_dir: &Path) -> LuaResult<Vec<String>> {
    match value {
        LuaValue::String(preset) => {
            let preset = preset.to_str()?;
            let tokenizer = load_tokenizer(entry, &preset, nn_dir)?;
            tokenizer
                .vocab_strings()
                .map_err(|e| LuaError::external(format!("{entry}: vocabulary: {e}")))
        }
        LuaValue::Table(table) => {
            let mut vocab = Vec::with_capacity(table.raw_len());
            for (index, surface) in table.sequence_values::<String>().enumerate() {
                vocab.push(surface.map_err(|e| {
                    LuaError::external(format!(
                        "{entry}: vocab[{}] must be a string: {e}",
                        index + 1
                    ))
                })?);
            }
            Ok(vocab)
        }
        other => Err(LuaError::external(format!(
            "{entry}: vocab must be a tokenizer preset name or a table of \
             surface strings indexed by token id, got {}",
            other.type_name()
        ))),
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

/// Lock a handle's slot, turning both failure modes into Lua errors.
///
/// `WouldBlock` is not contention — the VM is single-threaded, so the
/// only way to meet a held lock is re-entrancy: a Lua sampler callback
/// reaching back into the handle whose `sample` is still on the stack.
/// Blocking there would deadlock the VM, so it reports instead.
fn lock_slot<'a, T>(
    slot: &'a Mutex<Option<T>>,
    entry: &str,
    what: &str,
) -> LuaResult<MutexGuard<'a, Option<T>>> {
    match slot.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::WouldBlock) => Err(LuaError::external(format!(
            "{entry}: this {what} is already in use further up the call stack; \
             a Lua sampler callback cannot re-enter the sampler that invoked it"
        ))),
        Err(TryLockError::Poisoned(e)) => Err(LuaError::external(format!(
            "{entry}: {what} state is poisoned: {e}"
        ))),
    }
}

fn moved_sampler(entry: &str) -> LuaError {
    LuaError::external(format!(
        "{entry}: this sampler was moved into a constrained sampler; \
         use the constrained sampler, or build a fresh one"
    ))
}

fn moved_constraint(entry: &str) -> LuaError {
    LuaError::external(format!(
        "{entry}: this constraint was moved into a constrained sampler; \
         build a fresh one instead of reusing it"
    ))
}

fn msg(text: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(text.into())
}

#[cfg(test)]
mod tests {
    //! Value-level assertions that `tests/nn_bridge_smoke.rs` cannot
    //! make: it drives the Lua surface but cannot reach a `Tensor` or an
    //! `ErasedSampler`, both of which are crate-internal.

    use super::*;
    use candle_core::Device;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (values.len(),), &Device::Cpu).expect("logits row")
    }

    /// A plain sampler answers `is_done() == false` forever and treats
    /// `reset` as a no-op, so a generation loop can poll both without
    /// caring whether a constraint is attached.
    #[test]
    fn plain_sampler_is_never_done_and_survives_reset() {
        let handle = SamplerHandle::plain(GreedySampler);
        assert!(!handle.is_done().expect("is_done"));
        assert_eq!(handle.sample(&logits(&[0.1, 3.2, 0.5])).expect("sample"), 1);
        assert!(!handle.is_done().expect("is_done after sampling"));
        handle.reset().expect("reset must be a no-op, not an error");
        assert_eq!(
            handle.sample(&logits(&[0.1, 3.2, 0.5])).expect("sample"),
            1,
            "reset must leave a plain sampler usable"
        );
    }

    /// The move is observable from both sides: the composed sampler works
    /// and the source handle is spent. This is the invariant that keeps
    /// one RNG from being driven by two Lua handles.
    #[test]
    fn constrained_consumes_both_handles() {
        let inner = SamplerHandle::plain(GreedySampler);
        let constraint = ConstraintHandle::new(StopTokensConstraint::new(vec![1]));
        let composed = constrained_impl(&inner, &constraint).expect("compose");

        assert_eq!(
            composed.sample(&logits(&[0.1, 3.2, 0.5])).expect("sample"),
            1
        );
        assert!(composed.is_done().expect("is_done"), "stop token reached");
        composed.reset().expect("reset");
        assert!(!composed.is_done().expect("is_done after reset"));

        let err = inner
            .sample(&logits(&[0.1, 3.2, 0.5]))
            .expect_err("a moved-from sampler must not sample")
            .to_string();
        assert!(err.contains("moved"), "unexpected error: {err}");

        // `SamplerHandle` is deliberately not `Debug` (it owns an opaque
        // sampler), so the error is read out by hand rather than through
        // `expect_err`.
        let err = match constrained_impl(&inner, &constraint) {
            Ok(_) => panic!("composing two spent handles must fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("moved"), "unexpected error: {err}");
    }

    /// A failed composition must not eat the live argument: naming a
    /// spent sampler leaves the constraint reusable, and vice versa.
    #[test]
    fn a_failed_composition_leaves_the_other_argument_intact() {
        let spent = SamplerHandle::plain(GreedySampler);
        let eaten = ConstraintHandle::new(StopTokensConstraint::new(vec![1]));
        constrained_impl(&spent, &eaten).expect("first composition");

        let live = ConstraintHandle::new(StopTokensConstraint::new(vec![2]));
        assert!(
            constrained_impl(&spent, &live).is_err(),
            "the sampler is spent, so the composition must fail"
        );

        let fresh = SamplerHandle::plain(GreedySampler);
        constrained_impl(&fresh, &live).expect("the constraint must survive a failed composition");
    }

    /// The callback contract is checked, not assumed: a fractional
    /// number, a non-number, and an out-of-range id are all caller bugs
    /// that must surface at the call rather than as a strange token.
    #[test]
    fn lua_callback_return_values_are_validated() {
        assert_eq!(
            token_from_lua(LuaValue::Integer(3), 8).expect("integer id"),
            3
        );
        assert_eq!(
            token_from_lua(LuaValue::Number(3.0), 8).expect("integral float id"),
            3,
            "Lua 5.4 keeps 3.0 a float; it still names token 3"
        );
        for bad in [
            LuaValue::Number(1.5),
            LuaValue::Nil,
            LuaValue::Boolean(true),
            LuaValue::Integer(-1),
            LuaValue::Integer(8),
        ] {
            assert!(
                token_from_lua(bad.clone(), 8).is_err(),
                "{bad:?} must be rejected"
            );
        }
    }
}
