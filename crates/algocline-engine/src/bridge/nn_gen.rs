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
//! handle:embed(tokens, opts?)            -> { number, ... }
//! alc.nn.logits.mix(a, b, beta, opts?)   -> LogitsHandle
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
//! These carry a KV cache of their own
//! ([`algocline_nn::arch::KvCache`], one per session, obtained from the
//! model's `new_cache`), so a session forwards only the tokens appended
//! since its last step and attends over the rest. Generating `n` tokens
//! runs the model over `n` positions rather than `1 + 2 + … + n`.
//!
//! Until this landed they re-forwarded the whole history every step —
//! quadratic in the generation length, and every step recomputing keys
//! and values from weights that had not changed. The Lua surface is
//! unchanged either way; what changed is what it costs.
//!
//! The history is still capped at the model's context window, counted
//! against what the cache holds, and exceeding it is a loud error
//! rather than a positional-embedding failure surfacing from candle.
//!
//! # Optional input channels
//!
//! A GPT-2 model built with one of the optional input channels
//! (`alc.nn.preset.gpt2("custom", { cond_slots = N })` or
//! `{ allowed_input = true }`) reads that channel on every forward, so
//! a session over it takes the channel's value in `opts`:
//! `{ cond = <row> }` or `{ allowed = { id, ... } }`. `cond` also
//! accepts `{ w0, w1, ... }` — one weight per row of the table — which
//! conditions on the combination those weights describe rather than on
//! a single row. Both directions
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
use algocline_nn::arch::KvCache;
use algocline_nn::arch::{AllowedSets, CondIndex, Gpt2Model, TinyLlamaModel};
use std::path::Path;

use algocline_nn::gguf::{export_gguf, parse_precision, GgufArch, GgufSpec, PRECISION_NAMES};
use algocline_nn::pooling::{pool, Pooling};
use algocline_nn::sampling::{beam_search, Beam, BeamModel, BeamOptions};
use algocline_nn::tokenizer::{HfTokenizer, Message};
use algocline_nn::train::DeviceView;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use mlua::prelude::*;

use super::nn_card::{Gpt2Handle, LlamaHandle, NnHandle, TinyLlamaHandle};

/// The forward path a [`GenSession`] drives.
///
/// Every arm forwards incrementally through a per-session KV cache:
/// the adapter through its own, the trainable arches through
/// [`KvCache`] (see the module-level "Sessions over trainable arches"
/// section). What differs between them is only where the cache comes
/// from.
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
    /// Incremental forward over the shared trainable model, through
    /// this session's own KV cache. Two sessions over one handle share
    /// the weights and nothing else, the same isolation the Llama arm
    /// has.
    Gpt2 {
        model: Arc<Mutex<Gpt2Model>>,
        cache: Mutex<KvCache>,
    },
    /// Same discipline as `Gpt2`.
    TinyLlama {
        model: Arc<Mutex<TinyLlamaModel>>,
        cache: Mutex<KvCache>,
    },
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
    /// A weighted combination of *every* row of the model's
    /// conditioning table, added at every position of every step.
    ///
    /// The rows the model trained on are the table's own; a combination
    /// of them is a point the training never visited, which is the
    /// point of having it at decode time. Whether the model's behaviour
    /// varies smoothly between two rows is a question about the trained
    /// model — this only makes the question askable.
    CondWeights(Vec<f32>),
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
    /// Model context window. Enforced on the trainable arms by
    /// [`GenSession::cached_step`], so the refusal names the session
    /// and its history rather than surfacing a tensor dimension from
    /// deep inside candle. The Llama adapter caps itself.
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
        let (device, declared, cache) = {
            let guard = model.lock().map_err(|e| {
                LuaError::external(format!("alc.nn generate_session: model lock: {e}"))
            })?;
            (
                guard.device().clone(),
                DeclaredChannels::of_gpt2(&guard),
                guard.new_cache(),
            )
        };
        let channel = extract_channel(opts, declared, vocab)?;
        Ok(Self {
            backend: SessionBackend::Gpt2 {
                model,
                cache: Mutex::new(cache),
            },
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
        let cache = {
            let guard = model.lock().map_err(|e| {
                LuaError::external(format!("{GEN_SESSION_ERR_PREFIX}: model lock: {e}"))
            })?;
            guard.new_cache()
        };
        Ok(Self {
            backend: SessionBackend::TinyLlama {
                model,
                cache: Mutex::new(cache),
            },
            channel,
            device,
            vocab,
            ctx,
            tokens,
            forwarded: 0,
        })
    }

    /// Forward this step's pending tokens through the caller's cached
    /// entry point and return the final position's `[1, vocab]` row.
    ///
    /// Shared by the `Gpt2` / `TinyLlama` arms of `next_logits`; the
    /// per-arm closure supplies the model's `*_with_cache` call. The
    /// context window is checked here rather than left to the model, so
    /// the message names the session and its history instead of a
    /// tensor dimension.
    ///
    /// `on_failure` is called when the forward fails, before the error
    /// is returned: a model that failed part way down its layer stack
    /// has pushed entries for the layers it reached and not for the
    /// rest, so the cache no longer describes any sequence. Dropping it
    /// is what makes the session's "a failed step leaves the session
    /// where it was" true — otherwise a retry forwards the same tokens
    /// into already-extended layers.
    fn cached_step(
        &self,
        forward: impl FnOnce(&Tensor) -> candle_core::Result<Tensor>,
        on_failure: impl FnOnce(),
    ) -> LuaResult<Tensor> {
        let pending = &self.tokens[self.forwarded..];
        let total = self.tokens.len();
        if total > self.ctx {
            return Err(LuaError::external(format!(
                "alc.nn session:next_logits: session history ({total} tokens) exceeds \
                 the model context window ({ctx}); a session cannot generate past ctx",
                ctx = self.ctx
            )));
        }
        let input = Tensor::from_slice(pending, (1, pending.len()), &self.device)
            .map_err(|e| LuaError::external(format!("alc.nn session:next_logits: {e}")))?;
        // `[1, pending, vocab]` — this step's positions only, so the
        // last row is the next-token distribution. Same shape as the
        // Llama adapter's LastToken output, which is what lets the tail
        // of `next_logits` be shared.
        let step = forward(&input).map_err(|e| {
            on_failure();
            LuaError::external(format!("alc.nn session:next_logits: {e}"))
        })?;
        let last = step.dims()[1] - 1;
        step.narrow(1, last, 1)
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
            SessionBackend::Gpt2 { model, cache } => {
                let guard = model.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: model lock poisoned: {e}"
                    ))
                })?;
                let mut cache = cache.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: kv cache lock poisoned: {e}"
                    ))
                })?;
                // The cache is behind one `&mut` borrow, so the reset
                // is expressed as a flag the arm below acts on rather
                // than as a closure that would need a second one.
                let mut failed = false;
                let row = self.cached_step(
                    |input| match &self.channel {
                        SessionChannel::None => guard.forward_with_cache(input, &mut cache),
                        SessionChannel::Cond(index) => {
                            // One row, so one condition. The session is
                            // batch-1 by construction.
                            let conds = [*index];
                            guard.forward_conditioned_with_cache(input, &conds, &mut cache)
                        }
                        SessionChannel::CondWeights(weights) => {
                            // One combination for the whole forward; the
                            // session is batch-1 either way.
                            guard.forward_cond_weighted_with_cache(input, weights, &mut cache)
                        }
                        SessionChannel::Allowed(ids) => {
                            // The same set at every position of this step's
                            // tokens. Only this step's: the positions
                            // already in the cache read their sets when
                            // they were forwarded.
                            let positions = input.dims()[1];
                            let sets = vec![vec![ids.clone(); positions]];
                            let allowed = AllowedSets::new(&sets, input.device())?;
                            guard.forward_allowed_with_cache(input, &allowed, &mut cache)
                        }
                    },
                    || failed = true,
                );
                if failed {
                    cache.reset();
                }
                row?
            }
            SessionBackend::TinyLlama { model, cache } => {
                let guard = model.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: model lock poisoned: {e}"
                    ))
                })?;
                let mut cache = cache.lock().map_err(|e| {
                    LuaError::external(format!(
                        "alc.nn session:next_logits: kv cache lock poisoned: {e}"
                    ))
                })?;
                let mut failed = false;
                let row = self.cached_step(
                    |input| guard.forward_with_cache(input, &mut cache),
                    || failed = true,
                );
                if failed {
                    cache.reset();
                }
                row?
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
///
/// `cond` arrives in one of two forms ([`CondOpt`]). The weighted form
/// is checked against the same declared row count as the row form, and
/// an all-zero weighting is refused for the reason an unfed table is:
/// it adds the zero vector, so the model runs in a state it never
/// trained in and the output looks no different.
fn extract_channel(
    opts: Option<&LuaTable>,
    declared: DeclaredChannels,
    vocab: usize,
) -> LuaResult<SessionChannel> {
    let (cond, allowed) = match opts {
        Some(t) => (
            extract_cond(t)?,
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
        (Some(cond), None) => {
            let slots = declared.cond_slots.ok_or_else(|| {
                LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.cond was given but this model has no \
                     conditioning table; build it with \
                     alc.nn.preset.gpt2('custom', {{ cond_slots = N }})"
                ))
            })?;
            match cond {
                CondOpt::Row(row) => {
                    let index = CondIndex::new(row, slots).map_err(|e| {
                        LuaError::external(format!("{GEN_SESSION_ERR_PREFIX}: opts.cond: {e}"))
                    })?;
                    Ok(SessionChannel::Cond(index))
                }
                CondOpt::Weights(weights) => {
                    if weights.len() != slots {
                        return Err(LuaError::external(format!(
                            "{GEN_SESSION_ERR_PREFIX}: opts.cond holds {} weight(s) for a \
                             {slots}-row conditioning table; as a table it is one weight per \
                             row, in row order",
                            weights.len()
                        )));
                    }
                    if weights.iter().all(|w| *w == 0.0) {
                        return Err(LuaError::external(format!(
                            "{GEN_SESSION_ERR_PREFIX}: every weight in opts.cond is zero, which \
                             adds nothing — the same state this model runs in when the channel \
                             is left unfed, and the reason an unfed one is refused above"
                        )));
                    }
                    Ok(SessionChannel::CondWeights(weights))
                }
            }
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

/// What `opts.cond` said, before it is checked against the model.
///
/// Two forms, because there are two things a caller can mean by "the
/// condition": one of the table's rows, or a combination of all of
/// them. They are distinguished by Lua type rather than by a second
/// key — a number is a row and a table is a weight vector — so neither
/// form can be written in a way the other would accept.
enum CondOpt {
    /// A row index, the form that has always been accepted.
    Row(u32),
    /// One weight per row of the table, in row order.
    Weights(Vec<f32>),
}

/// Read `opts.cond` in either of its forms.
///
/// The number path re-reads the key through mlua's own conversion so
/// that what an integer `cond` accepts is exactly what it accepted
/// before this second form existed — including the integral-float case
/// (`cond = 1.0`), which arithmetic on a row index produces.
fn extract_cond(opts: &LuaTable) -> LuaResult<Option<CondOpt>> {
    let raw: LuaValue = opts
        .get("cond")
        .map_err(|e| LuaError::external(format!("{GEN_SESSION_ERR_PREFIX}: opts.cond: {e}")))?;
    match raw {
        LuaValue::Nil => Ok(None),
        LuaValue::Table(weights) => extract_cond_weights(&weights).map(Some),
        _ => opts
            .get::<u32>("cond")
            .map(|row| Some(CondOpt::Row(row)))
            .map_err(|e| {
                LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.cond must be a conditioning-table row \
                     (a non-negative integer) or an array of one weight per row: {e}"
                ))
            }),
    }
}

/// Read the array form of `opts.cond`.
///
/// Every element is required to be a number that survives the trip to
/// `f32` finite: a weight that arrived as `1e300` would become an
/// infinity before the model ever saw it, and the caller who wrote it
/// would have no way to see where it turned. What the *model's* dtype
/// makes of a weight is checked where the dtype is known, in
/// `Gpt2Model::forward_cond_weighted`.
///
/// Each element is taken as a [`LuaValue`] and matched, for the reason
/// [`mix_beta`] is: mlua's own conversion would also accept a numeric
/// *string*, and a weight that arrived as text is a caller who built it
/// by concatenation, which is a different bug than a weight out of
/// range.
///
/// The length is not checked here — the number of rows belongs to the
/// model, and [`extract_channel`] is where the two meet.
fn extract_cond_weights(weights: &LuaTable) -> LuaResult<CondOpt> {
    let len = weights.raw_len();
    if len == 0 {
        return Err(LuaError::external(format!(
            "{GEN_SESSION_ERR_PREFIX}: opts.cond is an empty table; as a table it names one \
             weight per conditioning row, and no weights name no condition"
        )));
    }
    let mut out = Vec::with_capacity(len);
    for i in 1..=len {
        let raw: LuaValue = weights.get(i).map_err(|e| {
            LuaError::external(format!(
                "{GEN_SESSION_ERR_PREFIX}: opts.cond[{i}] must be a number: {e}"
            ))
        })?;
        let value = match raw {
            LuaValue::Integer(v) => v as f64,
            LuaValue::Number(v) => v,
            other => {
                return Err(LuaError::external(format!(
                    "{GEN_SESSION_ERR_PREFIX}: opts.cond[{i}] must be a number, got {}",
                    other.type_name()
                )))
            }
        };
        let weight = value as f32;
        if !weight.is_finite() {
            return Err(LuaError::external(format!(
                "{GEN_SESSION_ERR_PREFIX}: opts.cond[{i}] = {value} is not a finite weight as an \
                 f32; it would reach every position of every step"
            )));
        }
        out.push(weight);
    }
    Ok(CondOpt::Weights(out))
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

/// Error prefix for the `alc.nn.logits.mix` surface.
const LOGITS_MIX_ERR_PREFIX: &str = "alc.nn.logits.mix";

/// The space a mixture's coefficients act in.
///
/// The two are different operations on the same pair of rows, and which
/// one a caller wants depends on what they are mixing *for* — see
/// [`mix_impl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixSpace {
    /// Mix the distributions: `β·softmax(a) + (1-β)·softmax(b)`.
    Prob,
    /// Mix the logits: `β·a + (1-β)·b`.
    Log,
}

impl MixSpace {
    /// Resolve the `opts.space` string.
    ///
    /// An unrecognised name is refused rather than falling back to the
    /// default: the two spaces produce different distributions, and a
    /// typo'd `"probs"` silently taking the arithmetic path would be a
    /// caller who believes they measured one operation and measured the
    /// other.
    fn parse(name: &str) -> LuaResult<Self> {
        match name {
            "prob" => Ok(Self::Prob),
            "log" => Ok(Self::Log),
            other => Err(LuaError::external(format!(
                "{LOGITS_MIX_ERR_PREFIX}: opts.space = '{other}' is not a mixing space \
                 (expected 'prob' or 'log')"
            ))),
        }
    }
}

/// Read a `[vocab]` f32 row off a handle and refuse the values that
/// have no image under either mixture.
///
/// `-inf` is accepted: it is what a constraint mask writes into a row,
/// and it means the same thing in both spaces (an id with no
/// probability). `NaN` and `+inf` are refused — the softmax of a row
/// containing `+inf` is `0/0` at every other entry, and a `NaN` reaching
/// [`LogitsHandle::top`] would sort to one end of `total_cmp`'s order
/// (above `+inf` or below `-inf`, by its sign bit) rather than being
/// ignored, and that order is the ranking a sampler acts on.
fn mix_row(handle: &LogitsHandle, side: &str) -> LuaResult<Vec<f32>> {
    let row: Vec<f32> = handle
        .tensor()
        .to_vec1()
        .map_err(|e| LuaError::external(format!("{LOGITS_MIX_ERR_PREFIX}: reading {side}: {e}")))?;
    if let Some((index, value)) = row
        .iter()
        .enumerate()
        .find(|(_, v)| v.is_nan() || **v == f32::INFINITY)
    {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: {side}[{index}] is {value}; a mixture of it is undefined \
             (-inf is accepted — that is what a constraint mask writes)"
        )));
    }
    Ok(row)
}

/// The row's largest entry, refusing a row that has no finite one.
///
/// A row of nothing but `-inf` (or an empty row) denotes no
/// distribution: every id has been struck out, so there is nothing left
/// to weight. Both mixing spaces need that refused — [`softmax_f64`]
/// because the shift below needs a finite maximum, and the log space
/// because `β·-inf + (1-β)·-inf` would carry the dead row through.
fn finite_max(row: &[f32], side: &str) -> LuaResult<f32> {
    let max = row.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v));
    if !max.is_finite() {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: {side} has no finite entry, so it denotes no \
             distribution to mix"
        )));
    }
    Ok(max)
}

/// The distribution a row denotes, in f64.
///
/// Shifted by the row's maximum before exponentiating, so the largest
/// term is `exp(0) = 1` and the denominator is at least 1 — the division
/// below cannot be by zero.
fn softmax_f64(row: &[f32], side: &str) -> LuaResult<Vec<f64>> {
    let max = f64::from(finite_max(row, side)?);
    let mut weights: Vec<f64> = row.iter().map(|v| (f64::from(*v) - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    Ok(weights)
}

/// `alc.nn.logits.mix(a, b, beta, opts?) -> logits`
///
/// Combine two rows into one, with `beta` the weight on `a`.
///
/// ```text
/// local mixed = alc.nn.logits.mix(from_one, from_other, 0.7)
/// local id = mixed:argmax()          -- or hand it to a sampler
/// ```
///
/// The result is an ordinary logits handle, so everything that consumes
/// one — `argmax`, `top`, the samplers, a constrained sampler — consumes
/// this too.
///
/// # The two spaces
///
/// `opts.space = "prob"` (the default) mixes the *distributions*:
/// `softmax(a)` and `softmax(b)` are combined as
/// `β·pA + (1-β)·pB`, and the result is stored as the log of that
/// mixture so the downstream softmax recovers it. `"log"` mixes the
/// logits themselves, `β·a + (1-β)·b`, which is a geometric mixture of
/// the two distributions after normalisation.
///
/// The default is `"prob"` because the two answer different questions
/// and the arithmetic one is the question a caller mixing *behaviours*
/// is asking. Measured on a 200-state sweep of β (2026-08-25): under the
/// arithmetic mixture the share of steps agreeing with the first source
/// moved monotonically with β and at most 14% of steps agreed with
/// neither source, while under the geometric mixture 33% of steps at
/// β = 0.5 agreed with neither. That is the geometric mixture behaving
/// as it is supposed to — it concentrates where both sources put mass,
/// which is a consensus, not a blend — so `"log"` stays available and
/// stays a deliberate choice.
///
/// # What this does not claim
///
/// Mixing per step is not the same as mixing over whole sequences. What
/// fraction of a *generated sequence* follows either source is a
/// property of the models and the states they reach, not of β, and
/// nothing here establishes a relation between the two. β is the weight
/// on one step's distribution; anything beyond that is measured, not
/// assumed.
///
/// # Errors
///
/// - The rows are over different vocabularies.
/// - `beta` is not a number, is not finite, or is outside `[0, 1]`.
/// - `opts.space` is a string other than `"prob"` or `"log"`.
/// - Either row holds a `NaN` or a `+inf`, or has no finite entry at
///   all (see [`mix_row`] and [`finite_max`]). Both rows are checked at
///   every `beta`, the endpoints included.
/// - The geometric mixture masks every id, which happens when the two
///   rows leave no id unmasked in common: the result would denote no
///   distribution, and the arithmetic space is where a union is meant.
fn mix_impl(a: &LogitsHandle, b: &LogitsHandle, beta: f64, space: MixSpace) -> LuaResult<Tensor> {
    let vocab_a = a
        .tensor()
        .dims1()
        .map_err(|e| LuaError::external(format!("{LOGITS_MIX_ERR_PREFIX}: a: {e}")))?;
    let vocab_b = b
        .tensor()
        .dims1()
        .map_err(|e| LuaError::external(format!("{LOGITS_MIX_ERR_PREFIX}: b: {e}")))?;
    if vocab_a != vocab_b {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: a is over {vocab_a} ids and b over {vocab_b}; mixing them \
             would add together the scores of tokens that do not denote the same thing"
        )));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: beta = {beta} is outside [0, 1]; beta is the weight on a \
             and (1 - beta) the weight on b"
        )));
    }

    // Both rows are read and checked before anything is returned, at
    // every `beta` including the endpoints. The refusals are a statement
    // about the arguments — a row carrying a `NaN` or a `+inf`, or one
    // with nothing left unmasked, is not a distribution — and a
    // statement that held only for `beta` strictly inside the interval
    // would be one the caller could not act on.
    let row_a = mix_row(a, "a")?;
    let row_b = mix_row(b, "b")?;
    finite_max(&row_a, "a")?;
    finite_max(&row_b, "b")?;

    // The endpoints of the log-space mixture are the row they name,
    // returned rather than computed. That is the convention the
    // operation carries, not a workaround: the geometric family
    // `p ∝ pA^β·pB^(1-β)` lives on `supp(pA) ∩ supp(pB)` for every β in
    // (0, 1) and on `supp(pA)` alone at β = 1 under `x⁰ = 1`, so the
    // value at an endpoint genuinely differs from the limit approaching
    // it. Computing it here would also read `0 · -inf`, which is `NaN`.
    if space == MixSpace::Log {
        if beta == 1.0 {
            return Ok(a.tensor().clone());
        }
        if beta == 0.0 {
            return Ok(b.tensor().clone());
        }
    }

    let mixed: Vec<f32> = match space {
        MixSpace::Prob => {
            let pa = softmax_f64(&row_a, "a")?;
            let pb = softmax_f64(&row_b, "b")?;
            pa.iter()
                .zip(&pb)
                .map(|(x, y)| (beta * x + (1.0 - beta) * y).ln() as f32)
                .collect()
        }
        MixSpace::Log => row_a
            .iter()
            .zip(&row_b)
            .map(|(x, y)| (beta * f64::from(*x) + (1.0 - beta) * f64::from(*y)) as f32)
            .collect(),
    };

    // The same refusal the inputs get, applied to the result, so that
    // what comes out of a mixture can go back into one. Only the
    // geometric space can reach it: an arithmetic mixture of two
    // distributions sums to 1, so some id always keeps mass, while
    // `β·a + (1-β)·b` is `-inf` wherever either side is — and if the two
    // rows leave no id unmasked in common, every id is.
    if !mixed.iter().any(|v| v.is_finite()) {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: the mixture masks every id — a and b mask every id the \
             other keeps, and a geometric mixture lives on the ids both rows allow; \
             opts.space = 'prob' mixes over the union instead"
        )));
    }

    // Built on `a`'s device: the values crossed to the host to be mixed
    // either way, and one of the two devices has to be picked when they
    // differ.
    Tensor::from_vec(mixed, vocab_a, a.tensor().device())
        .map_err(|e| LuaError::external(format!("{LOGITS_MIX_ERR_PREFIX}: {e}")))
}

/// Read `beta` off the Lua stack.
///
/// Taken as a `LuaValue` rather than as an `f64` so the refusal names
/// the argument: mlua's own conversion would also accept a numeric
/// *string*, and a `beta` that arrived as text is a caller who built it
/// by concatenation and has a different bug than the one a mixing
/// weight out of range describes.
fn mix_beta(value: &LuaValue) -> LuaResult<f64> {
    let beta = match value {
        LuaValue::Integer(i) => *i as f64,
        LuaValue::Number(n) => *n,
        other => {
            return Err(LuaError::external(format!(
                "{LOGITS_MIX_ERR_PREFIX}: beta must be a number in [0, 1], got {}",
                other.type_name()
            )))
        }
    };
    if !beta.is_finite() {
        return Err(LuaError::external(format!(
            "{LOGITS_MIX_ERR_PREFIX}: beta = {beta} is not a finite number"
        )));
    }
    Ok(beta)
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
/// trainable GPT-2 handle's method table.
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

/// Register `handle:embed(tokens, opts?)` on the trainable GPT-2
/// handle's method table.
pub(super) fn add_gpt2_embed_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<Gpt2Handle>,
{
    methods.add_method(
        "embed",
        |_, this, (tokens, opts): (Vec<i64>, Option<LuaTable>)| {
            embed_gpt2(this, &tokens, opts.as_ref())
        },
    );
}

/// Register `handle:embed(tokens, opts?)` on the trainable TinyLlama
/// handle's method table.
pub(super) fn add_tinyllama_embed_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<TinyLlamaHandle>,
{
    methods.add_method(
        "embed",
        |_, this, (tokens, opts): (Vec<i64>, Option<LuaTable>)| {
            embed_tinyllama(this, &tokens, opts.as_ref())
        },
    );
}

/// Register `handle:embed(tokens, opts?)` on the union handle — what a
/// Card reloaded through `alc.nn.card.load_handle` carries.
pub(super) fn add_nn_handle_embed_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<NnHandle>,
{
    methods.add_method(
        "embed",
        |_, this, (tokens, opts): (Vec<i64>, Option<LuaTable>)| {
            let opts = opts.as_ref();
            match this {
                NnHandle::Gpt2(h) => embed_gpt2(h, &tokens, opts),
                NnHandle::TinyLlama(h) => embed_tinyllama(h, &tokens, opts),
                // The adapter architectures expose logits and nothing
                // else: their forward returns the head's output and
                // there is no hidden state to reach behind it. Refused
                // by name rather than answered with the logits row,
                // which is a different quantity of a different width.
                NnHandle::Llama(_) => Err(LuaError::external(format!(
                    "{EMBED_ERR_PREFIX}: the llama adapter exposes logits only, so it has no \
                     hidden state to pool; embed with a gpt2 / tinyllama handle"
                ))),
            }
        },
    );
}

/// The embedding of `tokens` under a GPT-2 handle.
fn embed_gpt2(this: &Gpt2Handle, tokens: &[i64], opts: Option<&LuaTable>) -> LuaResult<Vec<f32>> {
    let ids = validate_embed_input(tokens, this.vocab(), this.ctx())?;
    let pooling = extract_pooling(opts)?;
    let model = this.model();
    let guard = model
        .lock()
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: model lock: {e}")))?;
    let input = Tensor::from_slice(&ids, (1, ids.len()), guard.device())
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))?;
    let hidden = guard
        .hidden(&input)
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))?;
    pooled_row(&hidden, pooling)
}

/// The embedding of `tokens` under a TinyLlama handle.
fn embed_tinyllama(
    this: &TinyLlamaHandle,
    tokens: &[i64],
    opts: Option<&LuaTable>,
) -> LuaResult<Vec<f32>> {
    let ids = validate_embed_input(tokens, this.vocab(), this.ctx())?;
    let pooling = extract_pooling(opts)?;
    let model = this.model();
    let guard = model
        .lock()
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: model lock: {e}")))?;
    let input = Tensor::from_slice(&ids, (1, ids.len()), guard.device())
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))?;
    let hidden = guard
        .hidden(&input)
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))?;
    pooled_row(&hidden, pooling)
}

/// Error prefix for the embedding surface.
const EMBED_ERR_PREFIX: &str = "alc.nn handle:embed";

/// Error prefix for the beam-search surface.
const BEAM_ERR_PREFIX: &str = "alc.nn handle:beam_search";

/// A [`BeamModel`] over a closure, so each architecture contributes
/// five lines rather than an impl.
struct ClosureBeam<F>(F);

impl<F> BeamModel for ClosureBeam<F>
where
    F: FnMut(&[u32]) -> candle_core::Result<Vec<f32>>,
{
    fn next_log_probs(&mut self, tokens: &[u32]) -> candle_core::Result<Vec<f32>> {
        (self.0)(tokens)
    }
}

/// Read `{ beams, max_new, length_penalty, eos }` off a Lua table.
fn beam_options(opts: Option<&LuaTable>, vocab: usize) -> LuaResult<BeamOptions> {
    let mut out = BeamOptions::default();
    let Some(t) = opts else {
        return Ok(out);
    };
    if let Some(v) = t.get::<Option<usize>>("beams")? {
        // Bounded, because the cost is `beams²` sequence clones per
        // step and the number comes from Lua: an unbounded one is a
        // hang rather than an error. The ceiling is far above any
        // useful width — published work stops well before 50.
        if v == 0 || v > MAX_BEAMS {
            return Err(LuaError::external(format!(
                "{BEAM_ERR_PREFIX}: opts.beams must be between 1 and {MAX_BEAMS} (got {v})"
            )));
        }
        out.beams = v;
    }
    if let Some(v) = t.get::<Option<usize>>("max_new")? {
        if v == 0 {
            return Err(LuaError::external(format!(
                "{BEAM_ERR_PREFIX}: opts.max_new must be at least 1; a search that generates \
                 nothing returns the prompt"
            )));
        }
        out.max_new = v;
    }
    if let Some(v) = t.get::<Option<f32>>("length_penalty")? {
        // A negative exponent inverts the ranking and a non-finite one
        // makes every comparison fall through to `Ordering::Equal`,
        // which leaves the "best" beam to sort stability.
        if !v.is_finite() || v < 0.0 {
            return Err(LuaError::external(format!(
                "{BEAM_ERR_PREFIX}: opts.length_penalty must be finite and >= 0 (got {v}); \
                 0 leaves raw score sums and 1 is the mean per token"
            )));
        }
        out.length_penalty = v;
    }
    if let Some(v) = t.get::<Option<i64>>("eos")? {
        out.eos = Some(check_token(v, vocab, "opts.eos")?);
    }
    Ok(out)
}

/// Widest beam this surface accepts.
///
/// Not a property of the search — a bound on a number that arrives from
/// Lua and multiplies the work quadratically.
const MAX_BEAMS: usize = 64;

/// Project the search's beams into the Lua array a caller gets back.
fn beams_to_lua(lua: &Lua, beams: Vec<Beam>) -> LuaResult<LuaTable> {
    let out = lua.create_table()?;
    for (i, beam) in beams.into_iter().enumerate() {
        let row = lua.create_table()?;
        row.set("tokens", beam.tokens)?;
        row.set("score", beam.score)?;
        row.set("finished", beam.finished)?;
        out.set(i + 1, row)?;
    }
    Ok(out)
}

/// The last position's log-probabilities from a full-sequence forward.
///
/// Log-probabilities rather than logits because the search sums them
/// across steps, and summing unnormalised scores compares sequences
/// under different normalisers.
fn last_log_probs(logits: &Tensor) -> candle_core::Result<Vec<f32>> {
    use candle_core::IndexOp;
    let t = logits.dims()[1];
    let row = logits.i((0, t - 1))?.to_dtype(DType::F32)?;
    candle_nn::ops::log_softmax(&row, 0)?.to_vec1()
}

/// Register `handle:beam_search(prompt, opts?)` on the GPT-2 handle.
pub(super) fn add_gpt2_beam_search_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<Gpt2Handle>,
{
    methods.add_method(
        "beam_search",
        |lua, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            let vocab = this.vocab();
            // Options first: the window check reads `max_new`.
            let options = beam_options(opts.as_ref(), vocab)?;
            let ids = validate_beam_prompt(&prompt, vocab, this.ctx(), options.max_new)?;
            let model = this.model();
            let guard = model
                .lock()
                .map_err(|e| LuaError::external(format!("{BEAM_ERR_PREFIX}: model lock: {e}")))?;
            let device = guard.device().clone();
            let mut beam_model = ClosureBeam(|tokens: &[u32]| {
                let input = Tensor::from_slice(tokens, (1, tokens.len()), &device)?;
                last_log_probs(&guard.forward(&input)?)
            });
            let beams = beam_search(&mut beam_model, &ids, &options)
                .map_err(|e| LuaError::external(format!("{BEAM_ERR_PREFIX}: {e}")))?;
            beams_to_lua(lua, beams)
        },
    );
}

/// Register `handle:beam_search(prompt, opts?)` on the TinyLlama
/// handle.
pub(super) fn add_tinyllama_beam_search_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<TinyLlamaHandle>,
{
    methods.add_method(
        "beam_search",
        |lua, this, (prompt, opts): (Vec<i64>, Option<LuaTable>)| {
            let vocab = this.vocab();
            // Options first: the window check reads `max_new`.
            let options = beam_options(opts.as_ref(), vocab)?;
            let ids = validate_beam_prompt(&prompt, vocab, this.ctx(), options.max_new)?;
            let model = this.model();
            let guard = model
                .lock()
                .map_err(|e| LuaError::external(format!("{BEAM_ERR_PREFIX}: model lock: {e}")))?;
            let device = guard.device().clone();
            let mut beam_model = ClosureBeam(|tokens: &[u32]| {
                let input = Tensor::from_slice(tokens, (1, tokens.len()), &device)?;
                last_log_probs(&guard.forward(&input)?)
            });
            let beams = beam_search(&mut beam_model, &ids, &options)
                .map_err(|e| LuaError::external(format!("{BEAM_ERR_PREFIX}: {e}")))?;
            beams_to_lua(lua, beams)
        },
    );
}

/// Check a beam-search prompt, leaving room for what it will generate.
///
/// The window is checked against prompt plus budget rather than against
/// the prompt alone: a search that would run past `ctx` half way
/// through is better refused before it starts than after it has spent
/// the forwards — which is why `max_new` is a parameter here and the
/// options are read before this is called.
fn validate_beam_prompt(
    prompt: &[i64],
    vocab: usize,
    ctx: usize,
    max_new: usize,
) -> LuaResult<Vec<u32>> {
    if prompt.is_empty() {
        return Err(LuaError::external(format!(
            "{BEAM_ERR_PREFIX}: prompt_tokens is empty; there is no sequence to continue"
        )));
    }
    if prompt.len() >= ctx {
        return Err(LuaError::external(format!(
            "{BEAM_ERR_PREFIX}: a prompt of {} fills the model context window ({ctx}), \
             leaving nothing to generate",
            prompt.len()
        )));
    }
    if prompt.len() + max_new > ctx {
        return Err(LuaError::external(format!(
            "{BEAM_ERR_PREFIX}: a prompt of {} plus max_new = {max_new} would reach {} \
             positions, past the model context window ({ctx})",
            prompt.len(),
            prompt.len() + max_new
        )));
    }
    prompt
        .iter()
        .enumerate()
        .map(|(i, id)| check_token(*id, vocab, &format!("prompt_tokens[{}]", i + 1)))
        .collect()
}

/// Error prefix for the GGUF export surface.
const GGUF_ERR_PREFIX: &str = "alc.nn handle:export_gguf";

/// The shape and weights a GGUF export reads off a handle.
///
/// Built by each handle type, so this module needs none of their
/// private fields and they need none of the export's vocabulary beyond
/// this one struct.
pub(super) struct GgufSource {
    /// Architecture family, as the handle names it.
    pub family: &'static str,
    /// Full `family-variant`, recorded as the model's name.
    pub variant: String,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub ctx: usize,
    pub vocab: usize,
    /// RoPE base, for architectures that rotate.
    pub rope_theta: Option<f32>,
    /// The weights, or `None` for a handle whose tensors live behind an
    /// mmap this bridge never named.
    pub varmap: Option<Arc<VarMap>>,
}

/// Register `handle:export_gguf(path, opts?)` on the GPT-2 handle.
pub(super) fn add_gpt2_export_gguf_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<Gpt2Handle>,
{
    methods.add_method(
        "export_gguf",
        |lua, this, (path, opts): (String, Option<LuaTable>)| {
            export_gguf_impl(lua, this.gguf_source(), &path, opts.as_ref())
        },
    );
}

/// Register `handle:export_gguf(path, opts?)` on the TinyLlama handle.
pub(super) fn add_tinyllama_export_gguf_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<TinyLlamaHandle>,
{
    methods.add_method(
        "export_gguf",
        |lua, this, (path, opts): (String, Option<LuaTable>)| {
            export_gguf_impl(lua, this.gguf_source()?, &path, opts.as_ref())
        },
    );
}

/// Register `handle:export_gguf(path, opts?)` on the inference-only
/// adapter handle, where it refuses.
///
/// Registered rather than left absent so the caller gets the reason
/// instead of `attempt to call a nil value`.
pub(super) fn add_llama_export_gguf_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<LlamaHandle>,
{
    methods.add_method(
        "export_gguf",
        |_, _this, (_path, _opts): (String, Option<LuaTable>)| -> LuaResult<LuaTable> {
            Err(LuaError::external(format!(
                "{GGUF_ERR_PREFIX}: the llama adapter holds its weights behind an mmap this \
                 bridge never named, so there is nothing here to write; export from a \
                 trainable handle"
            )))
        },
    );
}

/// Register `handle:export_gguf(path, opts?)` on the union handle —
/// what a Card reloaded through `alc.nn.card.load_handle` carries.
pub(super) fn add_nn_handle_export_gguf_method<M>(methods: &mut M)
where
    M: mlua::UserDataMethods<NnHandle>,
{
    methods.add_method(
        "export_gguf",
        |lua, this, (path, opts): (String, Option<LuaTable>)| {
            let source = match this {
                NnHandle::Gpt2(h) => h.gguf_source(),
                NnHandle::TinyLlama(h) => h.gguf_source()?,
                NnHandle::Llama(_) => {
                    return Err(LuaError::external(format!(
                        "{GGUF_ERR_PREFIX}: the llama adapter holds its weights behind an \
                         mmap this bridge never named, so there is nothing here to write; \
                         export from a trainable handle"
                    )))
                }
            };
            export_gguf_impl(lua, source, &path, opts.as_ref())
        },
    );
}

/// Write this handle's weights out as GGUF, and report what was
/// written.
fn export_gguf_impl(
    lua: &Lua,
    source: GgufSource,
    path: &str,
    opts: Option<&LuaTable>,
) -> LuaResult<LuaTable> {
    let precision = match opts
        .map(|t| t.get::<Option<String>>("precision"))
        .transpose()?
    {
        Some(Some(name)) => parse_precision(&name).ok_or_else(|| {
            LuaError::external(format!(
                "{GGUF_ERR_PREFIX}: unknown precision '{name}' (expected one of: {})",
                PRECISION_NAMES.join(" / ")
            ))
        })?,
        _ => GgmlDType::F32,
    };
    // A path to an HF `tokenizer.json`, not a preset name: a
    // `UserData` method has no view of the app directory the preset
    // cache lives under. The cache is at
    // `<app>/nn/tokenizers/<preset>.json`, which is what a caller
    // wanting the preset's own vocabulary points at.
    //
    // Refused if it does not exist, rather than writing a file with no
    // vocabulary in it — llama.cpp will not load one, and the failure
    // would surface there instead of here.
    let tokenizer = match opts
        .map(|t| t.get::<Option<String>>("tokenizer"))
        .transpose()?
    {
        Some(Some(file)) => {
            let file = PathBuf::from(file);
            if !file.is_file() {
                return Err(LuaError::external(format!(
                    "{GGUF_ERR_PREFIX}: opts.tokenizer names no file at {}; pass the path to \
                     an HF tokenizer.json (the preset cache is <app>/nn/tokenizers/<preset>.json)",
                    file.display()
                )));
            }
            Some(file)
        }
        _ => None,
    };

    let (spec, varmap) = gguf_spec(source)?;
    let report = export_gguf(
        &varmap,
        &spec,
        precision,
        tokenizer.as_deref(),
        Path::new(path),
    )
    .map_err(|e| LuaError::external(format!("{GGUF_ERR_PREFIX}: {e}")))?;

    let out = lua.create_table()?;
    out.set("path", path)?;
    out.set("tensors", report.tensors)?;
    out.set("metadata", report.metadata)?;
    out.set("tokenizer", report.tokenizer)?;
    out.set("architecture", spec.arch.name())?;
    Ok(out)
}

/// A handle's source to the export spec and the weights.
///
/// Refused where the handle cannot answer: a handle built
/// `pretrained = true` holds its tensors behind an mmap this bridge
/// never named, so there is nothing here to write under the names GGUF
/// wants. Exporting those means loading the weights into a
/// from-scratch handle first, and the message says so rather than
/// writing an empty file.
fn gguf_spec(source: GgufSource) -> LuaResult<(GgufSpec, Arc<VarMap>)> {
    let arch = match source.family {
        "gpt2" => GgufArch::Gpt2,
        "tinyllama" => GgufArch::Llama,
        other => {
            return Err(LuaError::external(format!(
                "{GGUF_ERR_PREFIX}: no GGUF key set for architecture family `{other}`"
            )))
        }
    };
    if source.variant.contains("custom") {
        return Err(LuaError::external(format!(
            "{GGUF_ERR_PREFIX}: a custom architecture's shape is not one GGUF has a key set \
             for — its feed-forward ratio, norm kind and position scheme are this crate's \
             own, and a reader would assemble the reference graph instead"
        )));
    }
    let varmap = source.varmap.clone().ok_or_else(|| {
        LuaError::external(format!(
            "{GGUF_ERR_PREFIX}: this handle was built with pretrained = true and carries no \
             VarMap, so its tensors have no names here; load the weights into a \
             from-scratch handle to export them"
        ))
    })?;
    Ok((
        GgufSpec {
            arch,
            layers: source.layers,
            heads: source.heads,
            kv_heads: source.kv_heads,
            dim: source.dim,
            ffn_dim: source.ffn_dim,
            ctx: source.ctx,
            vocab: source.vocab,
            eps: 1e-5,
            rope_theta: source.rope_theta,
            name: source.variant,
        },
        varmap,
    ))
}

/// Check an embedding input: non-empty, in vocabulary, inside the
/// context window.
///
/// The window is checked here rather than left to the forward pass for
/// the same reason the session checks it: the message then names the
/// input the caller wrote instead of a tensor dimension.
fn validate_embed_input(tokens: &[i64], vocab: usize, ctx: usize) -> LuaResult<Vec<u32>> {
    if tokens.is_empty() {
        return Err(LuaError::external(format!(
            "{EMBED_ERR_PREFIX}: tokens is empty; an embedding of nothing is not a vector \
             of zeros, it is a question with no subject"
        )));
    }
    if tokens.len() > ctx {
        return Err(LuaError::external(format!(
            "{EMBED_ERR_PREFIX}: {} tokens exceed the model context window ({ctx}); \
             split the input or embed a window of it",
            tokens.len()
        )));
    }
    tokens
        .iter()
        .enumerate()
        .map(|(i, id)| check_token(*id, vocab, &format!("tokens[{}]", i + 1)))
        .collect()
}

/// Read `opts.pooling`, defaulting to the mean.
fn extract_pooling(opts: Option<&LuaTable>) -> LuaResult<Pooling> {
    let Some(t) = opts else {
        return Ok(Pooling::default());
    };
    let Some(name) = t.get::<Option<String>>("pooling")? else {
        return Ok(Pooling::default());
    };
    Pooling::parse(&name).ok_or_else(|| {
        LuaError::external(format!(
            "{EMBED_ERR_PREFIX}: unknown pooling '{name}' (expected one of: {})",
            Pooling::NAMES.join(" / ")
        ))
    })
}

/// Pool a `[1, seq, dim]` hidden state into the flat Lua array a caller
/// gets back.
///
/// A plain array of numbers rather than an opaque handle: an embedding
/// is consumed by whatever the caller already has — a distance, a store,
/// a file — and none of those would take a handle this crate defines.
fn pooled_row(hidden: &Tensor, pooling: Pooling) -> LuaResult<Vec<f32>> {
    let pooled = pool(hidden, pooling, None)
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))?;
    pooled
        .squeeze(0)
        .and_then(|row| row.to_dtype(DType::F32))
        .and_then(|row| row.to_vec1::<f32>())
        .map_err(|e| LuaError::external(format!("{EMBED_ERR_PREFIX}: {e}")))
}

/// Register `handle:generate_session(prompt_tokens, opts?)` on the
/// trainable TinyLlama handle's method table.
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
/// `alc.nn.chat_prompt`, plus the `alc.nn.logits.*` namespace.
///
/// The first three resolve the tokenizer through
/// [`HfTokenizer::load_cached`] against `<nn_dir>/tokenizers`, the same
/// cache directory the `alc.nn.data.*` producers use, so a preset
/// downloaded by either path is reused by the other.
///
/// `alc.nn.logits` holds the operations that take logits rows and
/// return one — currently [`mix_impl`]. It sits beside `alc.nn.sampler`
/// (which turns a row into a token) rather than inside it: a mixture is
/// still a row, and it may be mixed again, masked, or ranked before
/// anything samples from it.
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

    let logits_ns = lua.create_table()?;
    let mix = lua.create_function(
        |_,
         (a, b, beta, opts): (
            LuaUserDataRef<LogitsHandle>,
            LuaUserDataRef<LogitsHandle>,
            LuaValue,
            Option<LuaTable>,
        )| {
            let beta = mix_beta(&beta)?;
            let space = match opts {
                Some(t) => match t.get::<Option<String>>("space").map_err(|e| {
                    LuaError::external(format!(
                        "{LOGITS_MIX_ERR_PREFIX}: opts.space must be a string \
                         ('prob' or 'log'): {e}"
                    ))
                })? {
                    Some(name) => MixSpace::parse(&name)?,
                    None => MixSpace::Prob,
                },
                None => MixSpace::Prob,
            };
            mix_impl(&a, &b, beta, space).map(LogitsHandle::from_tensor)
        },
    )?;
    logits_ns.set("mix", mix)?;
    nn_table.set("logits", logits_ns)?;

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
    use candle_core::IndexOp;
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

    /// Vocabulary and context window of [`tiny_gpt2`], restated where
    /// a session is built because `new_gpt2` takes both from the
    /// caller (the handle carries them in the bridge proper).
    const TINY_GPT2_VOCAB: usize = 32;
    const TINY_GPT2_CTX: usize = 16;

    /// A tiny GPT-2 with fixed weights, so two sessions over it can be
    /// compared for equality.
    fn tiny_gpt2() -> Arc<Mutex<Gpt2Model>> {
        use algocline_nn::arch::Gpt2Config;
        let cfg = Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: TINY_GPT2_CTX,
            vocab: TINY_GPT2_VOCAB,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vs = algocline_nn::arch::seeded_var_builder(&vm, 31337, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).expect("build gpt2 tiny");
        Arc::new(Mutex::new(model))
    }

    fn run_gpt2(
        model: Arc<Mutex<Gpt2Model>>,
        prompt: &[i64],
        steps: usize,
    ) -> Vec<(u32, Vec<f32>)> {
        let mut s = GenSession::new_gpt2(model, TINY_GPT2_VOCAB, TINY_GPT2_CTX, prompt, None)
            .expect("session");
        (0..steps).map(|_| step(&mut s)).collect()
    }

    /// The cached session returns, at every step, the row a full
    /// re-forward of the history returns. This is the bridge-level
    /// statement of the claim the cache rests on — the arch-level one
    /// is `gpt2::tests::a_cached_decode_matches_the_full_re_forward`,
    /// and this one additionally covers the session's own bookkeeping
    /// (what it forwards, and which row of the step it reads).
    #[test]
    fn a_gpt2_session_returns_what_a_full_re_forward_would() {
        let model = tiny_gpt2();
        let prompt: [i64; 3] = [1, 2, 3];
        let produced = run_gpt2(Arc::clone(&model), &prompt, 4);

        let mut history: Vec<u32> = prompt.iter().map(|id| *id as u32).collect();
        for (step_index, (sampled, row)) in produced.iter().enumerate() {
            let n = history.len();
            let input = Tensor::from_slice(&history, (1, n), &Device::Cpu).expect("input");
            let full = {
                let guard = model.lock().unwrap();
                guard.forward(&input).expect("full forward")
            };
            let reference: Vec<f32> = full
                .i((0, n - 1))
                .unwrap()
                .to_vec1()
                .expect("reference row");
            let gap = row
                .iter()
                .zip(&reference)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                gap < 2e-4,
                "step {step_index}: session row diverged by {gap}"
            );
            history.push(*sampled);
        }
    }

    /// Two GPT-2 sessions over one handle hold separate caches: run in
    /// lockstep they produce exactly what each produces alone. Before
    /// the cache the arm was stateless and this was trivially true;
    /// with per-session state it is the property that has to be kept.
    #[test]
    fn two_gpt2_sessions_over_one_handle_do_not_mix() {
        let model = tiny_gpt2();
        let prompt_a: [i64; 3] = [1, 2, 3];
        let prompt_b: [i64; 3] = [10, 11, 12];

        let solo_a = run_gpt2(Arc::clone(&model), &prompt_a, 4);
        let solo_b = run_gpt2(Arc::clone(&model), &prompt_b, 4);

        let mut a = GenSession::new_gpt2(
            Arc::clone(&model),
            TINY_GPT2_VOCAB,
            TINY_GPT2_CTX,
            &prompt_a,
            None,
        )
        .unwrap();
        let mut b = GenSession::new_gpt2(
            Arc::clone(&model),
            TINY_GPT2_VOCAB,
            TINY_GPT2_CTX,
            &prompt_b,
            None,
        )
        .unwrap();
        let mut mixed_a = Vec::new();
        let mut mixed_b = Vec::new();
        for _ in 0..4 {
            mixed_a.push(step(&mut a));
            mixed_b.push(step(&mut b));
        }

        assert!(
            max_gap(&solo_a, &mixed_a) < 1e-6,
            "session a saw session b's history"
        );
        assert!(
            max_gap(&solo_b, &mixed_b) < 1e-6,
            "session b saw session a's history"
        );
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

    // ─── Weighted cond opts ───────────────────────────────────────

    fn weights_opt(lua: &Lua, weights: &[LuaValue]) -> LuaTable {
        let list = lua.create_table().expect("weights table");
        for (i, w) in weights.iter().enumerate() {
            list.set(i + 1, w.clone()).expect("set weight");
        }
        channel_opts(lua, &[("cond", LuaValue::Table(list))])
    }

    /// The array form resolves against the same declared row count the
    /// integer form does, and reaches the session as a combination
    /// rather than as a row.
    #[test]
    fn cond_as_an_array_becomes_a_weighted_channel() {
        let lua = Lua::new();
        let opts = weights_opt(
            &lua,
            &[LuaValue::Number(0.25), LuaValue::Number(0.75), lua_zero()],
        );
        let channel = extract_channel(Some(&opts), declares_cond(3), 64).expect("three weights");
        match channel {
            SessionChannel::CondWeights(weights) => assert_eq!(weights, vec![0.25, 0.75, 0.0]),
            other => panic!("expected a weighted session, got {other:?}"),
        }
    }

    /// An integer `cond` still means a row after the array form was
    /// added — including the integral-float spelling that arithmetic on
    /// a row index produces.
    #[test]
    fn the_integer_form_is_unchanged_by_the_array_form() {
        let lua = Lua::new();
        for value in [LuaValue::Integer(1), LuaValue::Number(1.0)] {
            let opts = channel_opts(&lua, &[("cond", value.clone())]);
            match extract_channel(Some(&opts), declares_cond(3), 64).expect("row 1 of 3") {
                SessionChannel::Cond(index) => assert_eq!(index.row(), 1),
                other => panic!("expected a row selection for {value:?}, got {other:?}"),
            }
        }
    }

    /// The weights and the table have to agree on how many rows there
    /// are; a short list would otherwise silently name a prefix.
    #[test]
    fn cond_weights_must_cover_every_row() {
        let lua = Lua::new();
        let opts = weights_opt(&lua, &[LuaValue::Number(1.0), lua_zero()]);
        let err = extract_channel(Some(&opts), declares_cond(3), 64)
            .expect_err("two weights for three rows");
        let msg = err.to_string();
        assert!(
            msg.contains("2 weight(s)") && msg.contains("3-row"),
            "message must name both counts: {msg}"
        );
    }

    /// All-zero weights add the zero vector — the unfed state the
    /// channel exists to refuse.
    #[test]
    fn all_zero_cond_weights_are_refused() {
        let lua = Lua::new();
        let opts = weights_opt(&lua, &[lua_zero(), lua_zero()]);
        let err = extract_channel(Some(&opts), declares_cond(2), 64)
            .expect_err("nothing to condition on");
        assert!(
            err.to_string()
                .contains("every weight in opts.cond is zero"),
            "message: {err}"
        );
    }

    /// A weight that is not a number, and one that no `f32` can hold.
    #[test]
    fn cond_weights_must_be_finite_numbers() {
        let lua = Lua::new();
        let text = lua.create_string("half").expect("string");
        let opts = weights_opt(&lua, &[LuaValue::String(text), lua_zero()]);
        let err = extract_channel(Some(&opts), declares_cond(2), 64).expect_err("a non-number");
        assert!(
            err.to_string().contains("opts.cond[1]"),
            "message must name the entry: {err}"
        );

        let opts = weights_opt(&lua, &[lua_zero(), LuaValue::Number(1e300)]);
        let err = extract_channel(Some(&opts), declares_cond(2), 64).expect_err("an overflow");
        assert!(
            err.to_string().contains("opts.cond[2]"),
            "message must name the entry: {err}"
        );
    }

    /// A numeric string is refused, the way `beta` refuses one: mlua's
    /// own conversion would have taken it, and a weight that arrived as
    /// text is a caller who built it by concatenation.
    #[test]
    fn cond_weights_refuse_a_numeric_string() {
        let lua = Lua::new();
        let text = lua.create_string("0.5").expect("string");
        let opts = weights_opt(&lua, &[LuaValue::String(text), lua_zero()]);
        let err = extract_channel(Some(&opts), declares_cond(2), 64).expect_err("a numeric string");
        let msg = err.to_string();
        assert!(
            msg.contains("opts.cond[1]") && msg.contains("must be a number"),
            "message must name the entry and what it is not: {msg}"
        );

        // The integer form of a weight is still a weight.
        let opts = weights_opt(&lua, &[LuaValue::Integer(1), lua_zero()]);
        match extract_channel(Some(&opts), declares_cond(2), 64).expect("integer weights") {
            SessionChannel::CondWeights(weights) => assert_eq!(weights, vec![1.0, 0.0]),
            other => panic!("expected a weighted session, got {other:?}"),
        }
    }

    /// An empty table is neither a row nor a combination.
    #[test]
    fn an_empty_cond_table_is_refused() {
        let lua = Lua::new();
        let opts = weights_opt(&lua, &[]);
        let err = extract_channel(Some(&opts), declares_cond(2), 64).expect_err("no weights");
        assert!(err.to_string().contains("empty table"), "message: {err}");
    }

    /// Lua numeric literal `0.0`, spelled once.
    fn lua_zero() -> LuaValue {
        LuaValue::Number(0.0)
    }

    // ─── alc.nn.logits.mix ────────────────────────────────────────

    fn logits_of(values: &[f32]) -> LogitsHandle {
        LogitsHandle::from_tensor(
            Tensor::from_slice(values, values.len(), &Device::Cpu).expect("logits row"),
        )
    }

    fn mixed_row(a: &[f32], b: &[f32], beta: f64, space: MixSpace) -> Vec<f32> {
        mix_impl(&logits_of(a), &logits_of(b), beta, space)
            .expect("mix")
            .to_vec1()
            .expect("row")
    }

    fn distribution(row: &[f32]) -> Vec<f64> {
        softmax_f64(row, "row").expect("a distribution")
    }

    fn worst_gap(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()))
    }

    /// In probability space the result *denotes* the arithmetic mixture:
    /// the stored row is its log, so the softmax a sampler applies
    /// recovers `β·pA + (1-β)·pB` exactly. The endpoints are the two
    /// input distributions, which is the claim `β` is worth having.
    #[test]
    fn prob_space_denotes_the_arithmetic_mixture() {
        let a = [2.0f32, 0.0, -1.0, 0.5];
        let b = [-1.0f32, 3.0, 0.25, 0.0];
        let (pa, pb) = (distribution(&a), distribution(&b));

        for beta in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let mixed = distribution(&mixed_row(&a, &b, beta, MixSpace::Prob));
            let expected: Vec<f64> = pa
                .iter()
                .zip(&pb)
                .map(|(x, y)| beta * x + (1.0 - beta) * y)
                .collect();
            let gap = worst_gap(&mixed, &expected);
            assert!(
                gap < 1e-6,
                "beta {beta} was {gap} off the arithmetic mixture"
            );
        }
    }

    /// In log space the result *is* the arithmetic mixture of the two
    /// rows, and the endpoints are the rows themselves — returned
    /// unchanged rather than computed, because `0 · -inf` is `NaN`.
    #[test]
    fn log_space_mixes_the_rows_and_returns_the_endpoints_intact() {
        let a = [2.0f32, f32::NEG_INFINITY, -1.0];
        let b = [-1.0f32, 3.0, 0.25];

        assert_eq!(mixed_row(&a, &b, 1.0, MixSpace::Log), a.to_vec());
        assert_eq!(mixed_row(&a, &b, 0.0, MixSpace::Log), b.to_vec());

        let mixed = mixed_row(&a, &b, 0.25, MixSpace::Log);
        let expected = |x: f64, y: f64| 0.25 * x + 0.75 * y;
        assert!((f64::from(mixed[0]) - expected(2.0, -1.0)).abs() < 1e-6);
        // A masked entry stays masked at every interior weight: the
        // sampler must not be handed a candidate the constraint struck
        // out.
        assert_eq!(mixed[1], f32::NEG_INFINITY);
        assert!((f64::from(mixed[2]) - expected(-1.0, 0.25)).abs() < 1e-6);
    }

    /// Two rows masked to disjoint id sets have no geometric mixture:
    /// every entry of `β·a + (1-β)·b` is `-inf`, and such a row denotes
    /// no distribution. It is refused rather than handed on — an
    /// `argmax` over it would return candle's tie-break index, a
    /// plausible-looking token drawn from nothing.
    #[test]
    fn log_space_refuses_a_mixture_of_disjoint_supports() {
        let neg = f32::NEG_INFINITY;
        let a = [0.0f32, 1.0, neg, neg];
        let b = [neg, neg, 0.5, 2.0];
        for beta in [0.25, 0.5, 0.75] {
            let err = mix_impl(&logits_of(&a), &logits_of(&b), beta, MixSpace::Log)
                .expect_err("nothing left unmasked in common");
            let msg = err.to_string();
            assert!(
                msg.contains("masks every id") && msg.contains("'prob'"),
                "message must name the cause and the space that mixes over the union: {msg}"
            );
        }

        // Non-vacuity: one shared unmasked id and the same rows mix.
        let b_shared = [0.0f32, neg, 0.5, 2.0];
        let mixed = mixed_row(&a, &b_shared, 0.5, MixSpace::Log);
        assert!(
            mixed[0].is_finite(),
            "id 0 is open on both sides: {mixed:?}"
        );
    }

    /// The endpoints return the row they name, but they read both rows
    /// first: a row that denotes no distribution is refused at every
    /// `beta`, in both spaces. A refusal that held only strictly inside
    /// the interval would be one a caller could not act on.
    #[test]
    fn mix_checks_both_rows_at_the_endpoints_too() {
        let neg = f32::NEG_INFINITY;
        let good = [0.0f32, 1.0];
        for space in [MixSpace::Log, MixSpace::Prob] {
            for beta in [0.0, 1.0] {
                let err = mix_impl(&logits_of(&good), &logits_of(&[f32::NAN, 1.0]), beta, space)
                    .expect_err("NaN in b");
                assert!(err.to_string().contains("b[0]"), "message: {err}");

                let err = mix_impl(&logits_of(&[neg, neg]), &logits_of(&good), beta, space)
                    .expect_err("a row with nothing left");
                assert!(
                    err.to_string().contains("no finite entry"),
                    "message: {err}"
                );
            }
        }
    }

    /// A masked id keeps zero probability in the arithmetic mixture
    /// exactly when the *other* row masks it too — a mask on one side
    /// only is a candidate the other side still offers, weighted down
    /// rather than removed.
    #[test]
    fn prob_space_keeps_a_mask_only_where_both_sides_agree() {
        let a = [0.0f32, f32::NEG_INFINITY, 1.0];
        let b = [0.0f32, f32::NEG_INFINITY, 1.0];
        assert_eq!(mixed_row(&a, &b, 0.5, MixSpace::Prob)[1], f32::NEG_INFINITY);

        let b_open = [0.0f32, 0.5, 1.0];
        let mixed = mixed_row(&a, &b_open, 0.5, MixSpace::Prob);
        assert!(
            mixed[1].is_finite(),
            "the unmasked side still offers id 1: {mixed:?}"
        );
    }

    /// Two rows over different vocabularies index different things, so
    /// adding their scores together is refused.
    #[test]
    fn mix_refuses_rows_of_different_widths() {
        let err = mix_impl(
            &logits_of(&[0.0, 1.0]),
            &logits_of(&[0.0, 1.0, 2.0]),
            0.5,
            MixSpace::Prob,
        )
        .expect_err("2 ids against 3");
        let msg = err.to_string();
        assert!(
            msg.contains("2 ids") && msg.contains("3"),
            "message must name both widths: {msg}"
        );
    }

    /// `beta` is a weight, and the values that are not one are refused
    /// rather than clamped: a caller who computed 1.4 has a bug that
    /// clamping to 1.0 would answer with a plausible-looking row.
    #[test]
    fn mix_refuses_a_beta_outside_the_unit_interval() {
        for beta in [-0.1, 1.4, f64::INFINITY] {
            let err = mix_impl(
                &logits_of(&[0.0, 1.0]),
                &logits_of(&[1.0, 0.0]),
                beta,
                MixSpace::Prob,
            )
            .expect_err("beta outside [0, 1]");
            assert!(err.to_string().contains("beta"), "message: {err}");
        }
    }

    /// `beta` off the Lua stack: numbers pass, everything else is
    /// refused by name — including the numeric string mlua's own
    /// conversion would have accepted.
    #[test]
    fn mix_beta_takes_numbers_only() {
        let lua = Lua::new();
        assert_eq!(mix_beta(&LuaValue::Number(0.5)).expect("a number"), 0.5);
        assert_eq!(mix_beta(&LuaValue::Integer(1)).expect("an integer"), 1.0);

        let text = lua.create_string("0.5").expect("string");
        let err = mix_beta(&LuaValue::String(text)).expect_err("a numeric string");
        assert!(err.to_string().contains("beta must be a number"), "{err}");

        let err = mix_beta(&LuaValue::Number(f64::NAN)).expect_err("NaN");
        assert!(err.to_string().contains("finite"), "{err}");
    }

    /// An unrecognised space is refused rather than defaulted: the two
    /// produce different distributions, so a typo that fell through to
    /// the default would be a measurement of the wrong operation.
    #[test]
    fn mix_refuses_an_unknown_space() {
        assert_eq!(MixSpace::parse("prob").expect("prob"), MixSpace::Prob);
        assert_eq!(MixSpace::parse("log").expect("log"), MixSpace::Log);
        let err = MixSpace::parse("probs").expect_err("a typo");
        assert!(
            err.to_string().contains("not a mixing space"),
            "message: {err}"
        );
    }

    /// A row that denotes no distribution stops here. `NaN` in
    /// particular would sort above every real candidate in
    /// [`LogitsHandle::top`], which is the ranking a sampler acts on.
    #[test]
    fn mix_refuses_rows_it_cannot_read_as_distributions() {
        let err = mix_impl(
            &logits_of(&[f32::NAN, 1.0]),
            &logits_of(&[0.0, 1.0]),
            0.5,
            MixSpace::Prob,
        )
        .expect_err("NaN in a");
        assert!(err.to_string().contains("a[0]"), "message: {err}");

        let err = mix_impl(
            &logits_of(&[0.0, 1.0]),
            &logits_of(&[0.0, f32::INFINITY]),
            0.5,
            MixSpace::Prob,
        )
        .expect_err("+inf in b");
        assert!(err.to_string().contains("b[1]"), "message: {err}");

        let all_masked = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        let err = mix_impl(
            &logits_of(&all_masked),
            &logits_of(&[0.0, 1.0]),
            0.5,
            MixSpace::Prob,
        )
        .expect_err("a row with nothing left");
        assert!(
            err.to_string().contains("no finite entry"),
            "message: {err}"
        );
    }
}
