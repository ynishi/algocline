//! `alc.nn` generation sessions — the Lua-facing decode loop surface
//! (feature `nn`).
//!
//! Sits beside `nn_card.rs` (preset / card / data / trainer bindings)
//! and owns everything the token-by-token inference path needs:
//!
//! ```text
//! handle:generate_session(prompt_tokens, opts?) -> GenSession
//! session:next_logits()                  -> LogitsHandle
//! session:append(token_id)
//! session:tokens()                       -> { id, ... }
//! session:position()                     -> n
//! logits:vocab()                         -> n
//! logits:top(n)                          -> { { id = i, value = v }, ... }
//! logits:argmax()                        -> id
//! alc.nn.tokenize(preset, text)          -> { id, ... }
//! alc.nn.detokenize(preset, ids)         -> string
//! alc.nn.chat_prompt(preset, messages)   -> string
//! ```
//!
//! The samplers that consume a `LogitsHandle` live next door in
//! `nn_sampler.rs`.
//!
//! # Why a session rather than a bare `forward`
//!
//! [`LlamaAdapter`] carries a built-in KV cache, so exposing its
//! `forward` straight to Lua would mean two Lua-side loops sharing one
//! cache: their keys and values would interleave and each loop would
//! read a context the other polluted, with no error anywhere. A
//! [`GenSession`] instead owns a cache obtained from
//! [`LlamaAdapter::new_cache`] while the weights stay shared through the
//! `Arc`. Two sessions over one handle therefore cannot mix state at
//! all — the isolation is structural, not a convention the caller has to
//! remember. The bare handle-level `forward` is deliberately *not*
//! bound.
//!
//! # Sessions over trainable arches (GPT-2 / TinyLlama)
//!
//! The trainable arch models expose no KV cache — their `forward` is
//! the training-loop full-sequence pass. Their sessions therefore run
//! on a **stateless backend**: every `next_logits` re-forwards the full
//! token history and slices the final position's row. That is O(n²)
//! over the generation length, which is acceptable for the model sizes
//! the train side targets (tiny/small presets, smoke-scale ctx) and
//! buys the same Lua surface as the Llama session — a decode loop
//! written against one handle kind runs unchanged against the others.
//! The history is capped at the model's context window; exceeding it is
//! a loud session-level error rather than a positional-embedding
//! failure surfacing from candle.
//!
//! # Optional input channels
//!
//! A GPT-2 model built with one of the optional input channels
//! (`alc.nn.preset.gpt2("custom", { cond_slots = N })` or
//! `{ allowed_input = true }`) reads that channel on every forward, so
//! a session over it takes the channel's value in `opts`:
//! `{ cond = <row> }` or `{ allowed = { id, ... } }`. Both directions
//! of disagreement between the option and the model are refused — see
//! [`extract_channel`], which also says why the *silent* direction (a
//! table the caller says nothing about) is the one that matters.
//!
//! `opts.allowed` is the model's input channel, not a decode
//! constraint: it is added to the residual stream before the model
//! answers. Restricting what the sampler may *pick* is
//! `alc.nn.constraint.allow_list` in `nn_sampler.rs`, and the two are
//! independent — a model can be told what is available and still rank
//! an unavailable id first.
//!
//! # Why the loop lives in Lua
//!
//! The session exposes one forward step (`next_logits`) and one
//! commit step (`append`); deciding *which* token to append is the
//! caller's. That keeps sampling strategy, stop conditions, and
//! mid-generation control flow in Lua where they are cheap to iterate
//! on. A Rust-side `generate(...)` convenience can be added later
//! without changing this surface.
//!
//! # Why logits are opaque
//!
//! [`LogitsHandle`] wraps the `[vocab]` f32 row without exposing its
//! values wholesale to Lua. Materialising a vocab-sized float table per
//! token would cost more than the forward itself for a real vocabulary;
//! the sampler bindings consume the tensor Rust-side instead. What Lua
//! can ask for is the *ranking* — `top(n)` and `argmax()` — which is what
//! a hand-written sampler actually reads and which costs one pass over
//! the row.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use algocline_nn::arch::adapter::{InferenceAdapter, LlamaAdapter, LlamaCache};
use algocline_nn::arch::{AllowedSets, CondIndex, Gpt2Model, TinyLlamaModel};
use algocline_nn::tokenizer::{HfTokenizer, Message};
use algocline_nn::train::DeviceView;
use candle_core::{DType, Device, Tensor};
use mlua::prelude::*;

use super::nn_card::{Gpt2Handle, LlamaHandle, NnHandle, TinyLlamaHandle};

/// The forward path a [`GenSession`] drives.
///
/// `Llama` forwards incrementally through a per-session KV cache. The
/// trainable arches (`Gpt2` / `TinyLlama`) have no KV cache — their
/// `forward` is the training-loop full-sequence pass — so their arm
/// re-forwards the whole token history each step instead (see the
/// module-level "Sessions over trainable arches" section for the cost
/// trade-off).
enum SessionBackend {
    Llama {
        /// Shared, read-only weights. Cloning the `Arc` is what lets
        /// many sessions run against one loaded model.
        adapter: Arc<LlamaAdapter>,
        /// This session's private KV cache.
        ///
        /// The `Mutex` is here for the same reason as the one inside
        /// [`LlamaAdapter`]: mlua's `send` feature requires the
        /// `UserData` to be `Send`. The VM is single-threaded, so there
        /// is no real contention on it.
        cache: Mutex<LlamaCache>,
    },
    /// Stateless full-history re-forward over the shared trainable
    /// model. Session isolation is trivial here — there is no
    /// per-session state beyond the token history the session already
    /// owns.
    Gpt2(Arc<Mutex<Gpt2Model>>),
    /// Same stateless discipline as `Gpt2`.
    TinyLlama(Arc<Mutex<TinyLlamaModel>>),
}

/// The optional input channel a session feeds the model on every
/// forward, alongside the tokens.
///
/// Fixed for the session's lifetime. A decode loop that could change
/// the channel between steps would produce a sequence no single
/// forward pass explains, and there is no way to read back from the
/// tokens which step ran under what — so the value is taken once, at
/// construction, next to the prompt it belongs with.
///
/// The variants are mutually exclusive because the model's are: a spec
/// setting both channels is refused at build time
/// ([`algocline_nn::arch::Gpt2Custom::validate`]), since no forward
/// pass delivers both.
#[derive(Debug)]
enum SessionChannel {
    /// No channel — the plain forward.
    None,
    /// A row of the model's conditioning table, added at every
    /// position of every step.
    Cond(CondIndex),
    /// The ids the model may pick from. Flat: the same set is handed
    /// to every position of every step.
    ///
    /// # Flat, and what that leaves out
    ///
    /// The training-side channel is per position — each position
    /// carries the set that was available there
    /// ([`algocline_nn::arch::AllowedSets`]). A decode loop could
    /// supply the same shape, but it would have to restate the whole
    /// history's sets on every step, and the sets for the positions
    /// already forwarded cannot have changed. What a decoder actually
    /// knows fresh is the set at the position it is about to answer,
    /// and threading that through without restating the rest needs a
    /// session-level "advance with this set" call rather than an opts
    /// key. That is out of scope here: this option covers the case
    /// where one set holds for the whole generation.
    Allowed(Vec<u32>),
}

/// The optional channels a model was built with, read off its config.
///
/// Carried as a small value so the session can be constructed from one
/// lock acquisition: the device, the channels and the spec all come out
/// of the same config read.
#[derive(Debug, Clone, Copy, Default)]
struct DeclaredChannels {
    /// Rows of the model's conditioning table, or `None` when it has
    /// none.
    cond_slots: Option<usize>,
    /// Whether the model reads an allowed-id set at every position.
    allowed_input: bool,
}

impl DeclaredChannels {
    /// Read the channels a GPT-2 model declares.
    fn of_gpt2(model: &Gpt2Model) -> Self {
        let Some(spec) = model.config().custom.as_ref() else {
            return Self::default();
        };
        Self {
            cond_slots: spec.cond_slots,
            allowed_input: spec.allowed_input,
        }
    }
}

/// One in-flight generation over a model handle.
///
/// Holds the full token history (prompt included) plus the count of
/// tokens already forwarded, so a caller cannot get the `index_pos`
/// bookkeeping wrong: `next_logits` derives it from the session's own
/// state rather than taking it as an argument.
pub(super) struct GenSession {
    backend: SessionBackend,
    /// The input channel every forward of this session carries.
    channel: SessionChannel,
    /// Device the input tensors must be built on. Captured at
    /// construction — a model's device never changes after load.
    device: Device,
    /// Vocabulary bound every caller-supplied token id is checked
    /// against.
    vocab: usize,
    /// Model context window. Enforced on the stateless arms, whose
    /// full-history re-forward would otherwise surface a positional
    /// embedding error from deep inside candle.
    ctx: usize,
    /// Prompt tokens followed by every token the caller appended.
    tokens: Vec<u32>,
    /// How many entries of `tokens` are already forwarded (for the
    /// Llama arm: already in the KV cache). Equals the `index_pos` the
    /// next forward must start at.
    forwarded: usize,
}

impl GenSession {
    /// Validate a caller-supplied prompt: non-empty, every id within
    /// `vocab`.
    ///
    /// The prompt must be non-empty: a session with nothing to forward
    /// could not answer `next_logits`, and silently accepting one would
    /// only defer the failure to a confusing place.
    fn validate_prompt(prompt: &[i64], vocab: usize) -> LuaResult<Vec<u32>> {
        if prompt.is_empty() {
            return Err(LuaError::external(
                "alc.nn generate_session: prompt_tokens is empty; \
                 provide at least one token id to forward",
            ));
        }
        prompt
            .iter()
            .enumerate()
            .map(|(i, id)| check_token(*id, vocab, &format!("prompt_tokens[{}]", i + 1)))
            .collect()
    }

    /// Start a session over `prompt` against a Llama adapter.
    fn new_llama(
        adapter: Arc<LlamaAdapter>,
        prompt: &[i64],
        opts: Option<&LuaTable>,
    ) -> LuaResult<Self> {
        let meta = adapter.meta();
        let tokens = Self::validate_prompt(prompt, meta.vocab)?;
        // The adapter architectures carry no channel table, so the two
        // opts keys have nowhere to land here.
        let channel = extract_channel(opts, DeclaredChannels::default(), meta.vocab)?;
        let cache = adapter
            .new_cache()
            .map_err(|e| LuaError::external(format!("alc.nn generate_session: kv cache: {e}")))?;
        Ok(Self {
            backend: SessionBackend::Llama {
                adapter,
                cache: Mutex::new(cache),
            },
            channel,
            device: meta.device,
            vocab: meta.vocab,
            ctx: meta.ctx,
            tokens,
            forwarded: 0,
        })
    }

    /// Start a session over `prompt` against a trainable GPT-2 model.
    ///
    /// `opts` may name the input channel the model was built with —
    /// `cond` (a row of its conditioning table) or `allowed` (the ids
    /// it may pick from). Both are checked against what the model
    /// actually carries; see [`extract_channel`].
    pub(super) fn new_gpt2(
        model: Arc<Mutex<Gpt2Model>>,
        vocab: usize,
        ctx: usize,
        prompt: &[i64],
        opts: Option<&LuaTable>,
    ) -> LuaResult<Self> {
        let tokens = Self::validate_prompt(prompt, vocab)?;
        let (device, declared) = {
            let guard = model.lock().map_err(|e| {
                LuaError::external(format!("alc.nn generate_session: model lock: {e}"))
            })?;
            (guard.device().clone(), DeclaredChannels::of_gpt2(&guard))
        };
        let channel = extract_channel(opts, declared, vocab)?;
        Ok(Self {
            backend: SessionBackend::Gpt2(model),
            channel,
            device,
            vocab,
            ctx,
            tokens,
            forwarded: 0,
        })
    }

    /// Start a session over `prompt` against a trainable TinyLlama
    /// model.
    pub(super) fn new_tinyllama(
        model: Arc<Mutex<TinyLlamaModel>>,
        vocab: usize,
        ctx: usize,
        prompt: &[i64],
        opts: Option<&LuaTable>,
    ) -> LuaResult<Self> {
        let tokens = Self::validate_prompt(prompt, vocab)?;
        let device = model
            .lock()
            .map_err(|e| LuaError::external(format!("alc.nn generate_session: model lock: {e}")))?
            .device()
            .clone();
        // TinyLlama has no channel-table axis, so a channel key here is
        // refused rather than dropped.
        let channel = extract_channel(opts, DeclaredChannels::default(), vocab)?;
        Ok(Self {
            backend: SessionBackend::TinyLlama(model),
            channel,
            device,
            vocab,
            ctx,
            tokens,
            forwarded: 0,
        })
    }

    /// Build the `[1, len]` input tensor for the full token history and
    /// run one stateless forward, returning the final position's
    /// `[1, vocab]` row.
    ///
    /// Shared by the `Gpt2` / `TinyLlama` arms of `next_logits`; the
    /// per-arm closure only supplies the model's inherent `forward`.
    fn full_history_row(
        &self,
        forward: impl FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    ) -> LuaResult<Tensor> {
        let n = self.tokens.len();
        if n > self.ctx {
            return Err(LuaError::external(format!(
                "alc.nn session:next_logits: session history ({n} tokens) exceeds \
                 the model context window ({ctx}); trainable-arch sessions \
                 re-forward the full history and cannot generate past ctx",
                ctx = self.ctx
            )));
        }
        let input = Tensor::from_slice(&self.tokens, (1, n), &self.device)
            .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?;
        // `[1, n, vocab]` full-sequence logits → keep only the final
        // position's `[1, vocab]` row, matching the Llama adapter's
        // LastToken output shape so the common tail below is shared.
        let full = forward(&input)
            .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?;
        full.narrow(1, n - 1, 1)
            .and_then(|t| t.squeeze(1))
            .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))
    }

    /// Forward every token appended since the last call and return the
    /// next-token logits row.
    ///
    /// Calling this twice without an intervening `append` is an error
    /// rather than a re-run of the previous step: there is nothing new
    /// to forward, and returning the previous row would hide a caller
    /// bug (a decode loop that forgot to commit its sampled token would
    /// spin forever producing the same distribution).
    fn next_logits(&mut self) -> LuaResult<LogitsHandle> {
        let pending = &self.tokens[self.forwarded..];
        if pending.is_empty() {
            return Err(LuaError::external(
                "alc.nn session:next_logits: no pending tokens; \
                 call session:append(token_id) first",
            ));
        }

        let logits = match &self.backend {
            SessionBackend::Llama { adapter, cache } => {
                let input = Tensor::from_slice(pending, (1, pending.len()), &self.device)
                    .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?;
                let mut cache = cache.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: kv cache lock poisoned: {e}"
                    ))
                })?;
                adapter
                    .forward_with_cache(&input, self.forwarded, &mut cache)
                    .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?
            }
            SessionBackend::Gpt2(model) => {
                let guard = model.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: model lock poisoned: {e}"
                    ))
                })?;
                match &self.channel {
                    SessionChannel::None => self.full_history_row(|input| guard.forward(input))?,
                    SessionChannel::Cond(index) => {
                        // One row, so one condition. The session is
                        // batch-1 by construction.
                        let conds = [*index];
                        self.full_history_row(|input| guard.forward_conditioned(input, &conds))?
                    }
                    SessionChannel::Allowed(ids) => self.full_history_row(|input| {
                        // The same set at every position of the one
                        // row, rebuilt each step because the history
                        // the sets have to cover grows with it. Built
                        // inside the closure so the context-window
                        // check runs first.
                        let positions = input.dims()[1];
                        let sets = vec![vec![ids.clone(); positions]];
                        let allowed = AllowedSets::new(&sets, input.device())?;
                        guard.forward_allowed(input, &allowed)
                    })?,
                }
            }
            SessionBackend::TinyLlama(model) => {
                let guard = model.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: model lock poisoned: {e}"
                    ))
                })?;
                self.full_history_row(|input| guard.forward(input))?
            }
        };
        // Advance only after a successful forward: a failed step leaves
        // the session where it was so the caller can react without the
        // position silently running ahead of the cache.
        self.forwarded = self.tokens.len();

        // Every arm above lands on `[1, vocab]` and this session always
        // forwards batch 1, so drop the batch axis to reach the
        // `[vocab]` f32 row the sampler layer takes.
        let row = logits
            .squeeze(0)
            .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?;
        let row = if row.dtype() == DType::F32 {
            row
        } else {
            row.to_dtype(DType::F32).map_err(|e| {
                LuaError::external(format!("alc.nn session:next_logits: logits to f32: {e}"))
            })?
        };
        Ok(LogitsHandle { inner: row })
    }

    /// Commit `token` as the next token of this generation.
    ///
    /// The token is only queued here; it reaches the forward pass on the
    /// following `next_logits` call.
    fn append(&mut self, token: i64) -> LuaResult<()> {
        let id = check_token(token, self.vocab, "token_id")?;
        self.tokens.push(id);
        Ok(())
    }
}

/// Error prefix for the session-construction surface.
const GEN_SESSION_ERR_PREFIX: &str = "alc.nn generate_session";

/// Resolve the `cond` / `allowed` opts against the channels the model
/// declares.
///
/// Both directions of disagreement are refused, and neither is
/// cosmetic:
///
/// - A key the model has no table for is a caller who believes they
///   are conditioning a model that has nothing to condition on. Only
///   the allowed-id side is caught by the forward pass itself; a
///   `cond` handed to a model without the table would be caught there
///   too, but the message would name a Rust entry point rather than
///   the option that was written.
/// - A table the caller says nothing about is the silent direction. A
///   conditioning table left unfed adds the zero vector at every
///   position, so the model runs in a state it never trained in and
///   the output looks exactly as ordinary as a correct run. That is
///   the case this refusal exists for.
///
/// `vocab` bounds the `allowed` ids: an id outside it is refused here
/// rather than at the embedding lookup, where the message would name
/// neither the option nor the bound.
fn extract_channel(
    opts: Option<&LuaTable>,
    declared: DeclaredChannels,
    vocab: usize,
) -> LuaResult<SessionChannel> {
    let (cond, allowed) = match opts {
        Some(t) => (
            t.get::<Option<u32>>("cond").map_err(|e| {
                LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.cond must be a conditioning-table row \
                     (a non-negative integer): {e}"
                ))
            })?,
            t.get::<Option<Vec<u32>>>("allowed").map_err(|e| {
                LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.allowed must be an array of token ids: {e}"
                ))
            })?,
        ),
        None => (None, None),
    };

    if cond.is_some() && allowed.is_some() {
        return Err(LuaError::external(format!(
            "{GEN_SESSION_ERR_PREFIX}: opts.cond and opts.allowed name two different \
             channels and no model carries both; pass one"
        )));
    }

    match (cond, allowed) {
        (Some(row), None) => {
            let slots = declared.cond_slots.ok_or_else(|| {
                LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.cond was given but this model has no \
                     conditioning table; build it with \
                     alc.nn.preset.gpt2('custom', {{ cond_slots = N }})"
                ))
            })?;
            let index = CondIndex::new(row, slots).map_err(|e| {
                LuaError::external(format!("{GEN_SESSION_ERR_PREFIX}: opts.cond: {e}"))
            })?;
            Ok(SessionChannel::Cond(index))
        }
        (None, Some(ids)) => {
            if !declared.allowed_input {
                return Err(LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.allowed was given but this model has no \
                     allowed-id table; build it with \
                     alc.nn.preset.gpt2('custom', {{ allowed_input = true }})"
                )));
            }
            if ids.is_empty() {
                return Err(LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.allowed is empty, which says every id is \
                     unavailable; the model would answer as though it had no allowed-id \
                     channel at all"
                )));
            }
            for (i, id) in ids.iter().enumerate() {
                check_token(i64::from(*id), vocab, &format!("opts.allowed[{}]", i + 1))?;
            }
            Ok(SessionChannel::Allowed(ids))
        }
        (None, None) => {
            if declared.cond_slots.is_some() {
                return Err(LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: this model was built with a conditioning table \
                     and was trained with a condition at every position; generating without \
                     one runs it in a state it never trained in and the output would look no \
                     different — pass opts.cond = <row>"
                )));
            }
            if declared.allowed_input {
                return Err(LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: this model was built with an allowed-id table \
                     and reads one at every position; pass opts.allowed = {{ id, ... }}"
                )));
            }
            Ok(SessionChannel::None)
        }
        // Refused above.
        (Some(_), Some(_)) => unreachable!("both channels guarded above"),
    }
}

/// Validate one caller-supplied token id against the model's vocabulary.
///
/// Out-of-range ids are rejected here rather than at the embedding
/// lookup so the message names the offending value and the bound,
/// instead of surfacing as a candle index error several layers down.
fn check_token(id: i64, vocab: usize, what: &str) -> LuaResult<u32> {
    let bound = i64::try_from(vocab)
        .map_err(|e| LuaError::external(format!("alc.nn: vocab size {vocab} out of range: {e}")))?;
    if id < 0 || id >= bound {
        return Err(LuaError::external(format!(
            "alc.nn: {what} = {id} is outside the model vocabulary (0..{vocab})"
        )));
    }
    u32::try_from(id)
        .map_err(|e| LuaError::external(format!("alc.nn: {what} = {id} is not a token id: {e}")))
}

impl mlua::UserData for GenSession {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method_mut("next_logits", |_, this, ()| this.next_logits());
        methods.add_method_mut("append", |_, this, token: i64| this.append(token));
        methods.add_method("tokens", |_, this, ()| Ok(this.tokens.clone()));
        methods.add_method("position", |_, this, ()| Ok(this.tokens.len()));
    }
}

/// Mostly opaque `[vocab]` f32 logits row.
///
/// The values are never marshalled to Lua wholesale — see the module
/// doc. What Lua does get are the two *ranking* answers a hand-written
/// sampler needs ([`Self::top`], [`Self::argmax`]), each of which costs
/// one pass over the row instead of a vocab-sized table allocation. The
/// Rust-side sampler bindings read [`Self::tensor`] directly.
pub(super) struct LogitsHandle {
    inner: Tensor,
}

impl LogitsHandle {
    /// Wrap a `[vocab]` f32 row.
    ///
    /// Used by the sampler bridge to hand a *masked* row to a Lua
    /// callback: the callback must see what the constraint left standing,
    /// not the untouched row the session produced.
    pub(super) fn from_tensor(inner: Tensor) -> Self {
        Self { inner }
    }

    /// The wrapped `[vocab]` f32 row, in the shape
    /// `algocline_nn::sampling::Sampler` expects.
    ///
    /// Kept as an accessor (rather than letting a sibling module reach
    /// into the field) so the invariant "a `LogitsHandle` is always a
    /// valid sampler input" stays stated where the value is produced.
    pub(super) fn tensor(&self) -> &Tensor {
        &self.inner
    }

    /// The `n` highest-scoring tokens, best first, as
    /// `{ { id = <token id>, value = <logit> }, ... }`.
    ///
    /// # Ordering
    ///
    /// Sorted by descending logit with the lower token id winning ties,
    /// so the answer is stable across runs. Comparison is `f32::total_cmp`
    /// rather than `partial_cmp`: a constraint mask writes `-inf` into the
    /// row, and a total order keeps those entries at the bottom instead of
    /// leaving their placement to an `unwrap_or(Equal)` fallback.
    ///
    /// # Errors
    ///
    /// `n == 0` and `n > vocab` both error rather than clamping. A caller
    /// asking for zero candidates, or for more than exist, has a bug that
    /// a silently shortened list would hide — `top(k)` feeding a sampler
    /// that then indexes `[k]` would fail somewhere else entirely.
    fn top(&self, lua: &Lua, n: usize) -> LuaResult<LuaTable> {
        let values: Vec<f32> = self
            .inner
            .to_vec1()
            .map_err(|e| LuaError::external(format!("alc.nn logits:top: {e}")))?;
        let vocab = values.len();
        if n == 0 {
            return Err(LuaError::external(
                "alc.nn logits:top: n must be at least 1",
            ));
        }
        if n > vocab {
            return Err(LuaError::external(format!(
                "alc.nn logits:top: n = {n} exceeds the vocabulary size {vocab}"
            )));
        }

        let mut ranked: Vec<(usize, f32)> = values.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let out = lua.create_table_with_capacity(n, 0)?;
        for (rank, (id, value)) in ranked.into_iter().take(n).enumerate() {
            let entry = lua.create_table_with_capacity(0, 2)?;
            let id = u32::try_from(id).map_err(|e| {
                LuaError::external(format!(
                    "alc.nn logits:top: token id {id} is not a u32: {e}"
                ))
            })?;
            entry.set("id", id)?;
            entry.set("value", value)?;
            out.set(rank + 1, entry)?;
        }
        Ok(out)
    }

    /// The highest-scoring token id.
    ///
    /// Delegates to the same candle `argmax` the Rust-side
    /// `GreedySampler` uses, so a Lua sampler built on this agrees with
    /// `alc.nn.sampler.greedy()` token for token — including on the
    /// tie-breaking rule, which is candle's to define.
    fn argmax(&self) -> LuaResult<u32> {
        self.inner
            .argmax(0)
            .and_then(|idx| idx.to_scalar::<u32>())
            .map_err(|e| LuaError::external(format!("alc.nn logits:argmax: {e}")))
    }
}

impl mlua::UserData for LogitsHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("vocab", |_, this, ()| {
            this.inner
                .dims1()
                .map_err(|e| LuaError::external(format!("alc.nn logits: {e}")))
        });
        methods.add_method("top", |lua, this, n: usize| this.top(lua, n));
        methods.add_method("argmax", |_, this, ()| this.argmax());
    }
}

/// Register `handle:generate_session(prompt_tokens, opts?)` on the
/// Llama handle's method table.
///
/// Lives here rather than in `nn_card.rs` so the whole generation
/// surface stays in one module; `nn_card.rs` only calls this from its
/// `impl UserData for LlamaHandle`.
pub(super) fn add_generate_session_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<LlamaHandle>,
{
    methods.add_method(
        "generate_session",
        |_, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            GenSession::new_llama(this.adapter(), &prompt, opts.as_ref())
        },
    );
}

/// Register `handle:generate_session(prompt_tokens, opts?)` on the
/// trainable GPT-2 handle's method table (stateless full-history
/// backend).
pub(super) fn add_gpt2_generate_session_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<Gpt2Handle>,
{
    methods.add_method(
        "generate_session",
        |_, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            GenSession::new_gpt2(
                this.model(),
                this.vocab(),
                this.ctx(),
                &prompt,
                opts.as_ref(),
            )
        },
    );
}

/// Register `handle:generate_session(prompt_tokens, opts?)` on the
/// trainable TinyLlama handle's method table (stateless full-history
/// backend).
pub(super) fn add_tinyllama_generate_session_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<TinyLlamaHandle>,
{
    methods.add_method(
        "generate_session",
        |_, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            GenSession::new_tinyllama(
                this.model(),
                this.vocab(),
                this.ctx(),
                &prompt,
                opts.as_ref(),
            )
        },
    );
}

/// Register `handle:generate_session(prompt_tokens, opts?)` on the
/// arch-neutral [`NnHandle`] union, fanning out per variant.
///
/// This is what closes the "train → save Card → `load_handle` →
/// generate" loop from Lua: `alc.nn.card.load_handle` returns an
/// `NnHandle`, so without this registration a reloaded model could not
/// generate no matter which arch it wraps. It is also the path a
/// channel-carrying model reaches after a reload, which is why the
/// `opts` table is threaded through every variant rather than only the
/// GPT-2 one — the two other arms refuse a channel key instead of
/// dropping it.
pub(super) fn add_nn_handle_generate_session_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<NnHandle>,
{
    methods.add_method(
        "generate_session",
        |_, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            let opts = opts.as_ref();
            match this {
                NnHandle::Llama(h) => GenSession::new_llama(h.adapter(), &prompt, opts),
                NnHandle::Gpt2(h) => {
                    GenSession::new_gpt2(h.model(), h.vocab(), h.ctx(), &prompt, opts)
                }
                NnHandle::TinyLlama(h) => {
                    GenSession::new_tinyllama(h.model(), h.vocab(), h.ctx(), &prompt, opts)
                }
            }
        },
    );
}

/// Roles a message may carry.
///
/// An allowlist rather than a pass-through: a chat template branches on
/// the role string, and an unrecognised one (a typo'd `"assistent"`, a
/// `"human"` borrowed from another API) usually falls through every
/// branch and vanishes from the prompt. Silently dropping a turn is the
/// worst failure this surface has, so the typo is refused here instead.
const CHAT_ROLES: [&str; 4] = ["system", "user", "assistant", "tool"];

/// Register `alc.nn.tokenize` / `alc.nn.detokenize` /
/// `alc.nn.chat_prompt`.
///
/// All three resolve the tokenizer through [`HfTokenizer::load_cached`]
/// against `<nn_dir>/tokenizers`, the same cache directory the
/// `alc.nn.data.*` producers use, so a preset downloaded by either path
/// is reused by the other.
pub(super) fn register_gen_ns(lua: &Lua, nn_table: &LuaTable, nn_dir: PathBuf) -> LuaResult<()> {
    let tokenize_dir = nn_dir.clone();
    let tokenize = lua.create_function(
        move |_lua, (preset, text): (String, String)| -> LuaResult<Vec<u32>> {
            let tok = load_tokenizer("alc.nn.tokenize", &preset, &tokenize_dir)?;
            tok.encode(&text)
                .map_err(|e| LuaError::external(format!("alc.nn.tokenize: {e}")))
        },
    )?;
    nn_table.set("tokenize", tokenize)?;

    let detokenize_dir = nn_dir.clone();
    let detokenize = lua.create_function(
        move |_lua, (preset, ids): (String, Vec<u32>)| -> LuaResult<String> {
            let tok = load_tokenizer("alc.nn.detokenize", &preset, &detokenize_dir)?;
            tok.decode(&ids)
                .map_err(|e| LuaError::external(format!("alc.nn.detokenize: {e}")))
        },
    )?;
    nn_table.set("detokenize", detokenize)?;

    let chat_dir = nn_dir;
    let chat_prompt = lua.create_function(
        move |_lua, (preset, messages): (String, LuaTable)| -> LuaResult<String> {
            let messages = parse_messages("alc.nn.chat_prompt", &messages)?;
            let tok = load_tokenizer("alc.nn.chat_prompt", &preset, &chat_dir)?;
            // `add_generation_prompt` is fixed to true: what a caller
            // wants from this entry point is a prompt to continue. The
            // transcript-only form (false) is what a trainer wants, and
            // it arrives with the rest of the training-side options
            // rather than as a bare positional flag here.
            tok.apply_chat_template(&messages, true)
                .map_err(|e| LuaError::external(format!("alc.nn.chat_prompt: {e}")))
        },
    )?;
    nn_table.set("chat_prompt", chat_prompt)?;

    Ok(())
}

/// Convert the Lua `{ { role = ..., content = ... }, ... }` array into
/// [`Message`]s.
///
/// Every rejection names the offending index and what was expected: a
/// conversation is assembled by a strategy loop, so "one of the turns is
/// wrong" without a position is close to useless. Missing fields and
/// wrong types are refused rather than defaulted — a turn with an empty
/// role renders into a prompt the model was never tuned on, which shows
/// up as a bad answer instead of an error.
fn parse_messages(entry: &str, messages: &LuaTable) -> LuaResult<Vec<Message>> {
    let mut out = Vec::with_capacity(messages.raw_len());
    for (index, value) in messages.clone().sequence_values::<LuaValue>().enumerate() {
        let position = index + 1;
        let value = value?;
        let turn = value.as_table().ok_or_else(|| {
            LuaError::external(format!(
                "{entry}: messages[{position}] must be a table with 'role' and 'content', got {}",
                value.type_name()
            ))
        })?;
        let role: String = field(entry, turn, position, "role")?;
        let content: String = field(entry, turn, position, "content")?;
        if !CHAT_ROLES.contains(&role.as_str()) {
            return Err(LuaError::external(format!(
                "{entry}: messages[{position}].role = '{role}' is not a chat role \
                 (expected one of {})",
                CHAT_ROLES.join(", ")
            )));
        }
        out.push(Message { role, content });
    }
    Ok(out)
}

/// Read one required string field off a message table.
fn field(entry: &str, turn: &LuaTable, position: usize, name: &str) -> LuaResult<String> {
    match turn.get::<LuaValue>(name)? {
        LuaValue::String(s) => Ok(s.to_str()?.to_string()),
        LuaValue::Nil => Err(LuaError::external(format!(
            "{entry}: messages[{position}] is missing '{name}'"
        ))),
        other => Err(LuaError::external(format!(
            "{entry}: messages[{position}].{name} must be a string, got {}",
            other.type_name()
        ))),
    }
}

/// Resolve a tokenizer preset against the session's nn directory.
///
/// An empty `nn_dir` means the session was built without a store
/// (`BridgeConfig::nn_dir` doc). Refuse up front instead of letting
/// `load_cached` create a relative `tokenizers/` directory next to
/// whatever the process CWD happens to be.
///
/// Shared with `nn_sampler.rs`, whose constraints resolve a vocabulary
/// from the same preset names and must land in the same cache directory
/// — a constraint built against a different tokenizer than the one that
/// produced the prompt would mask the wrong ids.
pub(super) fn load_tokenizer(
    entry: &str,
    preset: &str,
    nn_dir: &std::path::Path,
) -> LuaResult<HfTokenizer> {
    if nn_dir.as_os_str().is_empty() {
        return Err(LuaError::external(format!(
            "{entry}: this session has no nn directory configured, \
             so the tokenizer cache cannot be resolved"
        )));
    }
    HfTokenizer::load_cached(preset, &nn_dir.join("tokenizers"))
        .map_err(|e| LuaError::external(format!("{entry}: {e}")))
}

#[cfg(test)]
mod tests {
    //! Value-level assertions on the session.
    //!
    //! These live in-crate because [`LogitsHandle`] is opaque by design:
    //! `tests/nn_bridge_smoke.rs` (a separate crate) can drive the Lua
    //! surface but cannot read a logits row, so the "two sessions do not
    //! mix" claim is verified here where the tensor is reachable. The
    //! Lua-facing loop and its error paths are covered there.

    use super::*;
    use algocline_nn::arch::adapter::LlamaAdapterConfig;
    use candle_nn::{VarBuilder, VarMap};

    /// Tiny random-weight adapter (2 layers / vocab 64 / ctx 16). The
    /// weights are fixed for the returned adapter's lifetime, which is
    /// what lets the tests below compare logits for equality.
    fn tiny_adapter() -> Arc<LlamaAdapter> {
        let cfg = LlamaAdapterConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        Arc::new(LlamaAdapter::load(vb, cfg).expect("load tiny Llama"))
    }

    /// Greedy step: read the row, pick the argmax, commit it.
    fn step(session: &mut GenSession) -> (u32, Vec<f32>) {
        let logits = session.next_logits().expect("next_logits");
        let row: Vec<f32> = logits.tensor().to_vec1().expect("logits row");
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        let next = u32::try_from(best).expect("token id fits u32");
        session
            .append(i64::from(next))
            .expect("append sampled token");
        (next, row)
    }

    fn run(adapter: Arc<LlamaAdapter>, prompt: &[i64], steps: usize) -> Vec<(u32, Vec<f32>)> {
        let mut s = GenSession::new_llama(adapter, prompt, None).expect("session");
        (0..steps).map(|_| step(&mut s)).collect()
    }

    fn max_gap(a: &[(u32, Vec<f32>)], b: &[(u32, Vec<f32>)]) -> f32 {
        assert_eq!(a.len(), b.len(), "stream length mismatch");
        let mut worst = 0.0f32;
        for ((_, ra), (_, rb)) in a.iter().zip(b) {
            assert_eq!(ra.len(), rb.len(), "vocab mismatch");
            for (x, y) in ra.iter().zip(rb) {
                worst = worst.max((x - y).abs());
            }
        }
        worst
    }

    /// Two sessions over one handle advanced in lockstep must produce
    /// exactly what each produces alone. This is the bridge-level
    /// counterpart of the adapter test: the isolation has to survive the
    /// `Arc<LlamaAdapter>` sharing that `generate_session` performs.
    #[test]
    fn two_sessions_over_one_handle_do_not_mix() {
        let adapter = tiny_adapter();
        let prompt_a: [i64; 3] = [1, 2, 3];
        let prompt_b: [i64; 3] = [10, 11, 12];

        let solo_a = run(Arc::clone(&adapter), &prompt_a, 4);
        let solo_b = run(Arc::clone(&adapter), &prompt_b, 4);

        let mut a =
            GenSession::new_llama(Arc::clone(&adapter), &prompt_a, None).expect("session a");
        let mut b =
            GenSession::new_llama(Arc::clone(&adapter), &prompt_b, None).expect("session b");
        let mut mixed_a = Vec::new();
        let mut mixed_b = Vec::new();
        for _ in 0..4 {
            mixed_a.push(step(&mut a));
            mixed_b.push(step(&mut b));
        }

        assert!(
            max_gap(&mixed_a, &solo_a) < 1e-5,
            "session A drifted when interleaved with B"
        );
        assert!(
            max_gap(&mixed_b, &solo_b) < 1e-5,
            "session B drifted when interleaved with A"
        );
        // Non-vacuity: if both prompts produced the same distributions,
        // contamination would be undetectable and this proves nothing.
        assert!(
            max_gap(&solo_a, &solo_b) > 1e-4,
            "test prompts must produce different logits"
        );
    }

    /// The row handed to the sampler layer is `[vocab]` f32 — the exact
    /// contract `algocline_nn::sampling::Sampler` validates. A leftover
    /// batch axis would be rejected there, one call site further away.
    #[test]
    fn logits_row_is_vocab_shaped_f32() {
        let adapter = tiny_adapter();
        let vocab = adapter.vocab();
        let mut session = GenSession::new_llama(adapter, &[1, 2, 3], None).expect("session");
        let logits = session.next_logits().expect("next_logits");
        assert_eq!(logits.tensor().dims(), &[vocab]);
        assert_eq!(logits.tensor().dtype(), DType::F32);
    }

    /// The session tracks its own `index_pos`: after the prompt is
    /// forwarded, only the appended token is pending.
    #[test]
    fn position_and_forwarded_track_appends() {
        let adapter = tiny_adapter();
        let mut session = GenSession::new_llama(adapter, &[1, 2, 3], None).expect("session");
        assert_eq!(session.tokens.len(), 3);
        assert_eq!(session.forwarded, 0);

        session.next_logits().expect("prompt forward");
        assert_eq!(session.forwarded, 3);

        session.append(7).expect("append");
        assert_eq!(session.tokens, vec![1, 2, 3, 7]);
        assert_eq!(session.forwarded, 3, "append must not forward by itself");

        session.next_logits().expect("incremental forward");
        assert_eq!(session.forwarded, 4);
    }

    // ─── Channel opts ─────────────────────────────────────────────

    fn channel_opts(lua: &Lua, pairs: &[(&str, LuaValue)]) -> LuaTable {
        let t = lua.create_table().expect("opts table");
        for (k, v) in pairs {
            t.set(*k, v.clone()).expect("set opt");
        }
        t
    }

    fn declares_cond(slots: usize) -> DeclaredChannels {
        DeclaredChannels {
            cond_slots: Some(slots),
            allowed_input: false,
        }
    }

    fn declares_allowed() -> DeclaredChannels {
        DeclaredChannels {
            cond_slots: None,
            allowed_input: true,
        }
    }

    #[test]
    fn a_model_with_no_channel_and_no_opts_runs_plain() {
        let channel =
            extract_channel(None, DeclaredChannels::default(), 64).expect("no channel anywhere");
        assert!(matches!(channel, SessionChannel::None));
    }

    #[test]
    fn cond_resolves_against_the_declared_table() {
        let lua = Lua::new();
        let opts = channel_opts(&lua, &[("cond", LuaValue::Integer(1))]);
        let channel = extract_channel(Some(&opts), declares_cond(3), 64).expect("row 1 of 3");
        match channel {
            SessionChannel::Cond(index) => assert_eq!(index.row(), 1),
            other => panic!("expected a conditioned session, got {other:?}"),
        }

        // Outside the table the model actually has: a crossed pair
        // rather than a typo, and refused as one.
        let opts = channel_opts(&lua, &[("cond", LuaValue::Integer(7))]);
        let err =
            extract_channel(Some(&opts), declares_cond(3), 64).expect_err("row 7 of a 3-row table");
        assert!(err.to_string().contains("outside"), "message: {err}");
    }

    #[test]
    fn cond_against_a_model_without_the_table_is_refused() {
        let lua = Lua::new();
        let opts = channel_opts(&lua, &[("cond", LuaValue::Integer(0))]);
        let err = extract_channel(Some(&opts), DeclaredChannels::default(), 64)
            .expect_err("no conditioning table to select from");
        let msg = err.to_string();
        assert!(
            msg.contains("no conditioning table") && msg.contains("cond_slots"),
            "message must name the missing table and how to build one: {msg}"
        );
    }

    /// The silent direction: the model carries the table and the
    /// caller says nothing, so the channel contributes nothing and the
    /// output looks exactly as ordinary as a correct run.
    #[test]
    fn a_declared_channel_left_unfed_is_refused() {
        let err = extract_channel(None, declares_cond(2), 64)
            .expect_err("a conditioning table left unfed");
        assert!(
            err.to_string().contains("opts.cond"),
            "message must name the option that feeds it: {err}"
        );

        let err =
            extract_channel(None, declares_allowed(), 64).expect_err("an allowed table left unfed");
        assert!(
            err.to_string().contains("opts.allowed"),
            "message must name the option that feeds it: {err}"
        );
    }

    #[test]
    fn allowed_ids_are_checked_against_the_vocabulary() {
        let lua = Lua::new();
        let ids = lua.create_table().expect("ids table");
        ids.set(1, 3).expect("set");
        ids.set(2, 9).expect("set");
        let opts = channel_opts(&lua, &[("allowed", LuaValue::Table(ids))]);
        let channel = extract_channel(Some(&opts), declares_allowed(), 16).expect("ids in range");
        match channel {
            SessionChannel::Allowed(ids) => assert_eq!(ids, vec![3, 9]),
            other => panic!("expected an allowed-id session, got {other:?}"),
        }

        let ids = lua.create_table().expect("ids table");
        ids.set(1, 99).expect("set");
        let opts = channel_opts(&lua, &[("allowed", LuaValue::Table(ids))]);
        let err = extract_channel(Some(&opts), declares_allowed(), 16)
            .expect_err("an id outside the vocabulary");
        assert!(
            err.to_string().contains("opts.allowed[1]"),
            "message must name the offending entry: {err}"
        );

        // An empty set says every id is unavailable, which the model
        // cannot be told apart from having no channel at all.
        let empty = lua.create_table().expect("ids table");
        let opts = channel_opts(&lua, &[("allowed", LuaValue::Table(empty))]);
        let err =
            extract_channel(Some(&opts), declares_allowed(), 16).expect_err("an empty allowed set");
        assert!(err.to_string().contains("empty"), "message: {err}");
    }

    #[test]
    fn the_two_channels_cannot_be_combined() {
        let lua = Lua::new();
        let ids = lua.create_table().expect("ids table");
        ids.set(1, 1).expect("set");
        let opts = channel_opts(
            &lua,
            &[
                ("cond", LuaValue::Integer(0)),
                ("allowed", LuaValue::Table(ids)),
            ],
        );
        let err = extract_channel(Some(&opts), declares_cond(2), 16)
            .expect_err("no model carries both channels");
        assert!(err.to_string().contains("two different"), "message: {err}");
    }
}
