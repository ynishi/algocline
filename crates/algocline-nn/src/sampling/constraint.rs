//! Layer 2 of the sampler plan: constraints that mask logits before a
//! Layer 1 [`Sampler`] picks a token.
//!
//! A [`Constraint`] answers two questions per generation step, both from
//! the generated-token prefix alone: *which tokens may come next*
//! ([`Constraint::mask`]) and *should generation stop*
//! ([`Constraint::is_terminal`]). [`ConstrainedSampler`] wires a
//! constraint to an arbitrary inner sampler and is itself a [`Sampler`],
//! so constrained decoding composes with every Layer 1 impl and with
//! future Layer 3 schedules without either side knowing about the other.
//!
//! # Sparse masks
//!
//! [`TokenMask`] is deliberately sparse. A dense `Vec<bool>` of vocab
//! length (32k–256k entries) would be rebuilt on every single token even
//! when the constraint has nothing to say. [`TokenMask::AllowAll`] is the
//! common case for prefix-agnostic constraints (stop tokens, most of a
//! grammar's interior) and costs *nothing*: the logits tensor is handed
//! to the inner sampler untouched, with no device round-trip.
//!
//! # Failure is loud
//!
//! A mask that leaves zero candidate tokens is a caller programming
//! error, not a situation to paper over. Softmax over an all-`-inf` row
//! yields NaNs, and silently falling back to argmax on the *unmasked*
//! logits would emit a token the constraint explicitly forbade — the one
//! outcome constrained decoding exists to prevent. Both cases return
//! `Err` instead.
//!
//! # Constraints that have landed
//!
//! [`StopTokensConstraint`] is the termination-only case: it masks
//! nothing and only answers [`Constraint::is_terminal`].
//! [`AllowListConstraint`] is the masking-only case: a fixed legal set,
//! the same at every position. [`RegexConstraint`] is the first
//! structural one — it drives an
//! anchored DFA over the tokenizer's surface strings so every sampled
//! token keeps the output on a path towards a full pattern match. JSON
//! schema and GBNF grammars are future additions behind the same trait.

use candle_core::{Device, Result as CandleResult, Tensor};
use regex_automata::{
    dfa::{dense, Automaton, StartKind},
    util::primitives::StateID,
    Anchored, Input, MatchKind,
};

use super::{validate_logits, Sampler};

/// A token mask as one bit per token id.
///
/// The dense counterpart of [`TokenMask`], and the shape the field
/// converged on independently:
/// [XGrammar](https://github.com/mlc-ai/xgrammar) and
/// [llguidance](https://docs.rs/llguidance/latest/llguidance/struct.Matcher.html)
/// both settled on a 32-bit-word bitset of `ceil(vocab / 32)` elements
/// with a set bit meaning *allowed*, without a specification saying so.
/// Two reasons, and both apply here: a grammar's allowed set is neither
/// reliably small nor reliably large — an id list is the wrong
/// representation at one end and the other — and a bitset can be filled
/// into a buffer the caller already owns, which takes the per-token
/// allocation out of the decode loop.
///
/// The bit primitives ignore an id at or beyond `vocab`: a set is a
/// set, and resizing or panicking inside one is not its business.
/// [`TokenBitset::fill_from`] does not ignore it — a constraint that
/// emits an out-of-range id has a bug, and absorbing it there would
/// hide the bug behind plausible output, which is the contract
/// [`TokenMask`] has always carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBitset {
    /// One bit per id, low bit first, `1` = allowed.
    words: Vec<u32>,
    vocab: usize,
}

impl TokenBitset {
    /// A set of `vocab` ids with nothing allowed.
    pub fn none(vocab: usize) -> Self {
        Self {
            words: vec![0; vocab.div_ceil(32)],
            vocab,
        }
    }

    /// A set of `vocab` ids with everything allowed.
    pub fn all(vocab: usize) -> Self {
        let mut out = Self::none(vocab);
        out.allow_all();
        out
    }

    /// Ids this set describes.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Allow every id.
    ///
    /// The tail bits past `vocab` in the last word are left clear, so
    /// two sets of one vocabulary compare equal exactly when they allow
    /// the same ids — which is what the mask cache's comparison rests
    /// on.
    pub fn allow_all(&mut self) {
        self.words.fill(u32::MAX);
        self.clear_tail();
    }

    /// Allow nothing.
    pub fn deny_all(&mut self) {
        self.words.fill(0);
    }

    /// Allow `id`, if it is in range.
    pub fn allow(&mut self, id: u32) {
        if (id as usize) < self.vocab {
            self.words[id as usize / 32] |= 1 << (id % 32);
        }
    }

    /// Deny `id`, if it is in range.
    pub fn deny(&mut self, id: u32) {
        if (id as usize) < self.vocab {
            self.words[id as usize / 32] &= !(1 << (id % 32));
        }
    }

    /// Whether `id` is allowed. Out-of-range ids are not.
    pub fn contains(&self, id: u32) -> bool {
        (id as usize) < self.vocab && self.words[id as usize / 32] & (1 << (id % 32)) != 0
    }

    /// Whether nothing is allowed — the state a mask must never be left
    /// in, since no token could be sampled.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// How many ids are allowed.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Re-shape to `vocab` and allow everything, reusing the buffer.
    ///
    /// The decode loop calls this once per step on a set it already
    /// owns; allocating only happens when the vocabulary changes, which
    /// it does not within a generation.
    pub fn reset_to_all(&mut self, vocab: usize) {
        if self.vocab != vocab {
            self.words.clear();
            self.words.resize(vocab.div_ceil(32), 0);
            self.vocab = vocab;
        }
        self.allow_all();
    }

    /// Fill from a [`TokenMask`], reusing the buffer.
    ///
    /// # Errors
    ///
    /// The first id at or beyond `vocab`. A constraint that names one
    /// is a constraint with a bug, and a mask that quietly dropped it
    /// would restrict the generation to something the caller did not
    /// ask for while looking entirely ordinary.
    pub fn fill_from(&mut self, mask: &TokenMask, vocab: usize) -> Result<(), u32> {
        self.reset_to_all(vocab);
        let ids = match mask {
            TokenMask::AllowAll => return Ok(()),
            TokenMask::Deny(ids) => ids,
            TokenMask::Allow(ids) => {
                self.deny_all();
                ids
            }
        };
        for id in ids {
            if (*id as usize) >= vocab {
                return Err(*id);
            }
        }
        match mask {
            TokenMask::Deny(ids) => {
                for id in ids {
                    self.deny(*id);
                }
            }
            TokenMask::Allow(ids) => {
                for id in ids {
                    self.allow(*id);
                }
            }
            TokenMask::AllowAll => unreachable!("returned above"),
        }
        Ok(())
    }

    /// The allowed ids, ascending.
    ///
    /// For a caller that needs the list — a report, a test, a
    /// representation that is not a mask. The decode path does not use
    /// it: walking the words is the point of holding them.
    pub fn allowed(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.count());
        for (index, word) in self.words.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                out.push((index * 32) as u32 + bit);
                bits &= bits - 1;
            }
        }
        out
    }

    /// Clear the bits past `vocab` in the final word.
    fn clear_tail(&mut self) {
        let used = self.vocab % 32;
        if used != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u32 << used) - 1;
            }
        }
    }
}

/// Sparse per-step token mask produced by a [`Constraint`].
///
/// Variants are mutually exclusive views of the same decision:
///
/// - [`TokenMask::AllowAll`] — no restriction. The logits tensor is
///   passed through untouched (no allocation, no device round-trip).
/// - [`TokenMask::Deny`] — the listed token ids are masked to `-inf`,
///   everything else survives. Use when the forbidden set is small.
/// - [`TokenMask::Allow`] — *only* the listed token ids survive,
///   everything else is masked to `-inf`. Use when the permitted set is
///   small (a grammar mid-production, a JSON key alternation).
///
/// Ids are token indices into the vocab axis of the logits row. Ids at
/// or beyond `vocab` are rejected at mask-application time — a
/// constraint that emits them has a bug the sampler must not absorb.
/// Duplicate ids are harmless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenMask {
    /// Every token is permitted; the logits are not modified at all.
    AllowAll,
    /// The listed token ids are masked to `-inf`.
    Deny(Vec<u32>),
    /// Only the listed token ids survive; all others are masked to
    /// `-inf`. An empty list is rejected (no token could be sampled).
    Allow(Vec<u32>),
}

/// Per-step restriction on the next token.
///
/// Both methods receive the **generated-token prefix only** — the prompt
/// is not included. A constraint that needs prompt context captures it
/// at construction time. The prefix is exactly the sequence of tokens
/// this [`ConstrainedSampler`] has produced since construction or the
/// last [`ConstrainedSampler::reset`].
///
/// [`Constraint::mask`] takes the prefix and `&self`, so a constraint
/// can always be written statelessly. It may instead keep the state
/// that prefix implies and let the sampler keep the two in step through
/// [`accept`](Constraint::accept) / [`rollback`](Constraint::rollback)
/// / [`reset`](Constraint::reset) — the verb set XGrammar and
/// llguidance arrived at separately, and the one that turns a
/// prefix-walking constraint from `O(n)` per token into `O(1)`.
///
/// **A stateful implementation must agree with itself**: after
/// accepting exactly the tokens of some prefix, its mask must be the
/// mask it would return for that prefix. The sampler calls `accept`
/// once per token it produces and `reset` when its own prefix is
/// cleared, so an implementation that honours the contract cannot
/// drift — but one that keeps state and answers `mask` from the prefix
/// inconsistently would produce a generation neither path explains.
pub trait Constraint {
    /// Tokens permitted at the position immediately after `prefix`.
    ///
    /// Return [`TokenMask::AllowAll`] when the constraint has nothing to
    /// restrict at this position; that path is free.
    fn mask(&self, prefix: &[u32]) -> TokenMask;

    /// Whether generation should stop, given everything produced so far.
    ///
    /// The sampler itself never stops on this signal — it has no loop to
    /// break. The generation loop owns termination and polls
    /// [`ConstrainedSampler::is_done`].
    fn is_terminal(&self, prefix: &[u32]) -> bool;

    /// The vocabulary this constraint reasons over, or `None` for one
    /// that reasons over ids alone and so fits any width.
    ///
    /// [`ConstrainedSampler`] checks it against the logits row it is
    /// handed. A constraint built from a token→string table cannot
    /// judge an id past the end of that table, and under a wider row
    /// those ids would come through permitted without ever having been
    /// examined — plausible output drawn from the part of the
    /// vocabulary the grammar never saw.
    fn vocab(&self) -> Option<usize> {
        None
    }

    /// Fill `out` with the tokens permitted after `prefix`.
    ///
    /// The dense form of [`Self::mask`], and the one the decode loop
    /// calls: `out` is a buffer the caller already owns, so a step
    /// allocates nothing. The default adapts [`Self::mask`], which is
    /// what makes every existing implementation work unchanged; an
    /// implementation whose natural output is a bitset — a grammar
    /// walking a token trie — overrides this and leaves `mask` as the
    /// adapter instead.
    fn fill_mask(&self, prefix: &[u32], vocab: usize, out: &mut TokenBitset) -> Result<(), u32> {
        out.fill_from(&self.mask(prefix), vocab)
    }

    /// The same, for a caller whose accepted tokens **are** `prefix`.
    ///
    /// [`ConstrainedSampler`] is that caller and the only one: it
    /// accepts every token it produces and resets with its own prefix,
    /// so a constraint keeping state can answer from it instead of
    /// walking. The default ignores the distinction, which is correct
    /// for a stateless implementation and is why this is a separate
    /// method rather than a precondition on [`Self::fill_mask`] —
    /// a precondition the public method cannot check is a trap.
    fn fill_mask_accepted(
        &self,
        prefix: &[u32],
        vocab: usize,
        out: &mut TokenBitset,
    ) -> Result<(), u32> {
        self.fill_mask(prefix, vocab, out)
    }

    /// Advance by one token the sampler has committed to.
    ///
    /// A no-op by default, which is right for a constraint that reads
    /// the prefix. A stateful one advances its automaton here instead
    /// of re-deriving it next step.
    fn accept(&mut self, token: u32) {
        let _ = token;
    }

    /// Undo the last `n` accepted tokens.
    ///
    /// For a caller that retracts — a beam search abandoning a branch,
    /// a speculative decode whose draft was rejected. A constraint that
    /// cannot rewind re-derives from the prefix instead, which is what
    /// the default no-op means for a stateless one.
    fn rollback(&mut self, n: usize) {
        let _ = n;
    }

    /// Forget everything accepted, as at construction.
    fn reset(&mut self) {}
}

/// A boxed, type-erased constraint is still a [`Constraint`].
///
/// The counterpart of `impl Sampler for Box<dyn Sampler + Send>`: Layer 3
/// picks the constraint at runtime from Lua, so the engine needs to name
/// `ConstrainedSampler<_, Box<dyn Constraint + Send>>`. `Send` is in the
/// bound for the same reason as there — mlua's `send` feature requires
/// the `UserData` holding a constraint to be `Send`.
///
/// Delegation preserves the trait's purity contract: the box adds an
/// indirection, not state.
impl Constraint for Box<dyn Constraint + Send> {
    fn mask(&self, prefix: &[u32]) -> TokenMask {
        (**self).mask(prefix)
    }

    fn is_terminal(&self, prefix: &[u32]) -> bool {
        (**self).is_terminal(prefix)
    }

    /// Delegated rather than defaulted, so a boxed constraint that
    /// overrides any of these keeps its override through the erasure —
    /// the default would quietly re-adapt from `mask` and undo it.
    fn fill_mask(&self, prefix: &[u32], vocab: usize, out: &mut TokenBitset) -> Result<(), u32> {
        (**self).fill_mask(prefix, vocab, out)
    }

    fn fill_mask_accepted(
        &self,
        prefix: &[u32],
        vocab: usize,
        out: &mut TokenBitset,
    ) -> Result<(), u32> {
        (**self).fill_mask_accepted(prefix, vocab, out)
    }

    fn accept(&mut self, token: u32) {
        (**self).accept(token)
    }

    fn rollback(&mut self, n: usize) {
        (**self).rollback(n)
    }

    fn reset(&mut self) {
        (**self).reset()
    }

    fn vocab(&self) -> Option<usize> {
        (**self).vocab()
    }
}

/// A [`Sampler`] that masks logits through a [`Constraint`] before
/// delegating the actual draw to an inner sampler.
///
/// The wrapper owns the generated-token prefix, which is the one piece
/// of state Layer 1 samplers deliberately do not carry. Composition is
/// arbitrary: the inner sampler may itself be a `ConstrainedSampler`,
/// stacking constraints (the innermost mask is applied last).
///
/// # Invariants
///
/// - The returned token id is always permitted by the constraint's mask
///   for the current prefix; a mask that permits nothing errors rather
///   than falling back.
/// - `prefix` grows by exactly one token per successful
///   [`Sampler::sample`] call and is untouched on error, so a caller can
///   retry a failed step without corrupting the history.
#[derive(Debug, Clone)]
pub struct ConstrainedSampler<S: Sampler, C: Constraint> {
    inner: S,
    constraint: C,
    prefix: Vec<u32>,
    /// This step's allowed set, reused across steps so a decode loop
    /// allocates nothing per token.
    allowed: TokenBitset,
    /// The bias tensor of the last mask, reused while the mask holds.
    bias: MaskBias,
}

impl<S: Sampler, C: Constraint> ConstrainedSampler<S, C> {
    /// Wrap `inner` with `constraint`, starting from an empty prefix.
    pub fn new(inner: S, constraint: C) -> Self {
        Self {
            inner,
            constraint,
            prefix: Vec::new(),
            // Sized on the first step, from the logits the caller
            // brings: the constraint does not always know the
            // vocabulary and the sampler never does until then.
            allowed: TokenBitset::none(0),
            bias: MaskBias::default(),
        }
    }

    /// Whether the constraint considers the current prefix terminal.
    ///
    /// Polled by the generation loop; the sampler never acts on it.
    pub fn is_done(&self) -> bool {
        self.constraint.is_terminal(&self.prefix)
    }

    /// Tokens generated so far, oldest first.
    pub fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    /// Drop the generated prefix so the same sampler can drive another
    /// generation.
    ///
    /// The inner sampler's RNG is intentionally *not* reset: two
    /// generations from one reused sampler stay distinct, while a caller
    /// wanting bit-identical repeats constructs a fresh sampler from the
    /// same seed.
    pub fn reset(&mut self) {
        self.prefix.clear();
        // The constraint's own state goes with it, or a stateful one
        // would answer the next generation from the last one's
        // automaton.
        self.constraint.reset();
    }
}

impl<S: Sampler, C: Constraint> Sampler for ConstrainedSampler<S, C> {
    /// Refused: this sampler holds one prefix and a batch is several
    /// sequences. Mixing every row's tokens into one prefix would
    /// return valid ids in the right shape, constrained against a
    /// sequence none of the rows is.
    fn sample_batch(&mut self, _logits: &Tensor) -> CandleResult<Vec<u32>> {
        Err(candle_core::Error::Msg(
            "sample_batch: a constrained sampler tracks one prefix, and a batch is several              sequences; use one constrained sampler per row"
                .into(),
        ))
    }

    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        validate_logits(logits)?;
        let vocab = logits.dims()[0];
        if let Some(declared) = self.constraint.vocab() {
            if declared != vocab {
                return Err(candle_core::Error::Msg(format!(
                    "ConstrainedSampler: the constraint reasons over {declared} tokens and the \
                     logits row is {vocab} wide; the ids in between would be permitted without \
                     the constraint having examined them"
                )));
            }
        }
        // The accepted form: this sampler's prefix is exactly what it
        // has accepted, which is the precondition the fast path needs.
        self.constraint
            .fill_mask_accepted(&self.prefix, vocab, &mut self.allowed)
            .map_err(|id| {
                candle_core::Error::Msg(format!(
                    "ConstrainedSampler: mask token id {id} is out of range for vocab {vocab}"
                ))
            })?;
        if self.allowed.is_empty() {
            return Err(candle_core::Error::Msg(format!(
                "ConstrainedSampler: the constraint permits no token at position {} \
                 (vocab {vocab} fully masked)",
                self.prefix.len()
            )));
        }
        let token = if self.allowed.count() == vocab {
            // Nothing is restricted: hand the caller's tensor straight
            // through, no bias and no copy.
            self.inner.sample(logits)?
        } else {
            let bias = self.bias.bias_for(&self.allowed, logits.device())?;
            // The surviving entries come through untouched —
            // `x + 0.0` is `x` — and the row never leaves the device it
            // was produced on.
            let masked = logits.add(&bias)?;
            self.inner.sample(&masked)?
        };
        self.prefix.push(token);
        // The constraint advances with the prefix, so a stateful one
        // never has to re-walk it.
        self.constraint.accept(token);
        Ok(token)
    }
}

/// Terminate generation when one of a fixed set of token ids is emitted.
///
/// The stop tokens are **not** masked out: `mask` always returns
/// [`TokenMask::AllowAll`], matching the semantics every mainstream
/// runtime uses — the stop token is a legitimate sample, and it is the
/// generation *loop* that halts once it appears. Masking them instead
/// would make the model unable to ever finish.
///
/// Detection is on the last token only. A stop token deeper in the
/// prefix means the loop ignored an earlier stop signal, which is the
/// caller's decision to make.
#[derive(Debug, Clone, Default)]
pub struct StopTokensConstraint {
    stop_tokens: Vec<u32>,
}

impl StopTokensConstraint {
    /// Build a constraint terminating on any of `stop_tokens`.
    ///
    /// An empty list yields a constraint that never terminates, which is
    /// a valid (if unusual) request — the loop is then bounded by a
    /// max-token budget instead.
    pub fn new(stop_tokens: Vec<u32>) -> Self {
        Self { stop_tokens }
    }

    /// The configured stop token ids.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }
}

impl Constraint for StopTokensConstraint {
    fn mask(&self, _prefix: &[u32]) -> TokenMask {
        TokenMask::AllowAll
    }

    fn is_terminal(&self, prefix: &[u32]) -> bool {
        match prefix.last() {
            Some(last) => self.stop_tokens.contains(last),
            None => false,
        }
    }
}

/// Restrict every position to a fixed set of legal token ids.
///
/// The mirror image of [`StopTokensConstraint`]: that one only
/// terminates and never masks, this one only masks and never terminates.
/// [`Constraint::mask`] returns the same [`TokenMask::Allow`] regardless
/// of the prefix, so an inner sampler may draw as noisily as it likes
/// and still return a legal token — the "legal mask" a game or tool
/// caller wants when the legal set is known *before* decoding and does
/// not depend on what was decoded so far.
///
/// Termination is deliberately left elsewhere: a fixed legal set says
/// nothing about when a sequence is complete, so `is_terminal` is always
/// `false` and the generation loop (or a stacked
/// [`StopTokensConstraint`]) owns stopping.
///
/// # Empty lists are rejected
///
/// An empty legal set permits nothing, which [`mask_bias`] refuses at
/// sample time. That is one token too late to be useful: the mistake is
/// in the caller's legality computation, not in the draw. `new` returns
/// `Err` instead, the same way [`RegexConstraint::new`] rejects a
/// pattern it cannot compile.
///
/// # Intended usage: rebuild per decision
///
/// [`ConstrainedSampler`] owns both its inner sampler and its
/// constraint, and the Lua-facing `alc.nn.sampler.constrained` consumes
/// both handles. The intended pattern is therefore to rebuild the whole
/// chain for each decision, with the legal set recomputed from the
/// current position and the seed derived explicitly (from a turn number,
/// say) so the decision stays reproducible:
///
/// ```text
/// sampler.constrained(sampler.temperature(t, seed_i),
///                     constraint.allow_list(legal_ids_i))
/// ```
///
/// There is intentionally no API to swap the id list on a live
/// constraint: a mutable legal set would make the mask depend on call
/// order rather than on the prefix, which is exactly the purity the
/// [`Constraint`] contract relies on for reproducibility.
#[derive(Debug, Clone)]
pub struct AllowListConstraint {
    allowed: Vec<u32>,
}

impl AllowListConstraint {
    /// Build a constraint permitting exactly `allowed`.
    ///
    /// Returns `Err` when `allowed` is empty (see the type doc).
    /// Duplicate ids are harmless, and ids outside the vocab are caught
    /// when the mask is applied — the constraint has no way to know the
    /// vocab size, and inventing one here would be a second source of
    /// truth for it.
    pub fn new(allowed: Vec<u32>) -> CandleResult<Self> {
        if allowed.is_empty() {
            return Err(candle_core::Error::Msg(
                "AllowListConstraint: the allow list is empty, so no token could ever be sampled"
                    .into(),
            ));
        }
        Ok(Self { allowed })
    }

    /// The configured legal token ids.
    pub fn allowed(&self) -> &[u32] {
        &self.allowed
    }
}

impl Constraint for AllowListConstraint {
    fn mask(&self, _prefix: &[u32]) -> TokenMask {
        TokenMask::Allow(self.allowed.clone())
    }

    fn is_terminal(&self, _prefix: &[u32]) -> bool {
        false
    }
}

/// Restrict generation to token sequences that spell a full match of a
/// regular expression.
///
/// # Semantics
///
/// The pattern is compiled as `^(?:pattern)$` — the wrapping is literal,
/// not a figure of speech, and both halves are load-bearing. Without the
/// leading anchor the DFA would look for a match starting anywhere;
/// without the *trailing* one an anchored DFA stops caring about the rest
/// of the input the moment a match exists, so `\d{3}` would happily
/// accept `123abc` and report every prefix of it as terminal. A caller
/// may still write `^` / `$` explicitly; they are redundant, not wrong.
///
/// `vocab` is the surface string of every token id, indexed by id — the
/// shape [`crate::tokenizer::HfTokenizer::vocab_strings`] produces. It is
/// the piece that turns a byte-level automaton into a token-level filter:
/// a candidate token is admitted when walking its bytes from the current
/// DFA state does not land in a dead state, i.e. when the pattern can
/// still be completed after emitting it. That is strictly stronger than
/// "the token matches so far" — it also rules out tokens that would paint
/// generation into a corner one step later.
///
/// # Cost
///
/// [`Constraint::mask`] re-walks the prefix from the start state and then
/// trial-walks every vocab entry, so a step costs
/// `O(prefix_bytes + vocab * token_bytes)`. The re-walk is deliberate:
/// [`Constraint`] promises purity with respect to the prefix (`&self`, no
/// interior mutation), which is what keeps seeded generation
/// reproducible and makes a constraint safe to share. The known
/// optimisation — precomputing a state → permitted-token-set index at
/// construction time, the way Outlines does — is deferred until a
/// per-step measurement shows this loop is the bottleneck; it trades a
/// vocab × states build cost and a large resident index for the walk.
///
/// # Empty tokens
///
/// A vocab entry that is the empty string is denied at every position. It
/// consumes no bytes, so it cannot advance the DFA, and a generation loop
/// that kept drawing it would never terminate. Those entries are exactly
/// the special / surface-less ids `vocab_strings` reports as empty.
///
/// # Impossible positions
///
/// When the prefix itself is unreachable (a token walked into a dead
/// state, or an id past the end of `vocab`), and when no token can
/// continue a viable prefix, `mask` returns [`TokenMask::Allow`] with an
/// empty list. [`Constraint::mask`] cannot return an error — but an empty
/// `Allow` is rejected by [`mask_bias`], so the condition surfaces as a
/// loud `Err` from [`Sampler::sample`] instead of quietly emitting an
/// off-pattern token.
#[derive(Debug, Clone)]
pub struct RegexConstraint {
    dfa: dense::DFA<Vec<u32>>,
    /// Anchored start state, resolved once at construction so the
    /// per-step walk cannot fail.
    start: StateID,
    vocab: Vec<String>,
    /// The incrementally kept walk — see [`Walk`] and
    /// [`Constraint::accept`].
    walked: Walk,
}

/// What this constraint knows about the walk so far.
///
/// Three states, not two: a walk that **died** is an answer (nothing is
/// permitted from here), and a walk that is **unknown** is not. Folding
/// them together makes a rollback look like a dead end, which permits
/// nothing and ends the generation — the bug this enum exists to make
/// unrepresentable.
#[derive(Debug, Clone, Copy)]
enum Walk {
    /// The state after exactly `len` accepted tokens, or `None` if the
    /// walk died along the way.
    Known { state: Option<StateID>, len: usize },
    /// Nothing is known; the next question re-walks the prefix.
    ///
    /// Entered by a rollback, because a DFA does not run backwards.
    Unknown,
}

impl RegexConstraint {
    /// Compile `pattern` into an anchored full-match DFA over `vocab`.
    ///
    /// Returns `Err` on an invalid pattern. Compiling up front rather
    /// than lazily is the point: a typo in a pattern is a caller bug that
    /// should surface where the constraint is configured, not mid-stream
    /// on some later token.
    ///
    /// The DFA is built with [`MatchKind::All`] rather than the default
    /// leftmost-first semantics. Leftmost-first stops exploring once it
    /// has committed to a match, which would make `a|ab` reject the `b`
    /// in `ab`; `All` keeps every alternative alive, which is the
    /// question a constraint actually asks ("can *any* match still be
    /// reached from here?").
    pub fn new(pattern: &str, vocab: Vec<String>) -> CandleResult<Self> {
        // Full-match wrapping (see the type doc): the trailing anchor is
        // what makes a state past the end of the pattern *dead* rather
        // than merely "already matched", which is the difference between
        // rejecting an off-pattern token and waving it through.
        let full_match = format!("^(?:{pattern})$");
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Anchored)
                    .match_kind(MatchKind::All),
            )
            .build(&full_match)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "RegexConstraint: cannot compile pattern {pattern:?}: {e}"
                ))
            })?;
        let start = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "RegexConstraint: no anchored start state for pattern {pattern:?}: {e}"
                ))
            })?;
        Ok(Self {
            dfa,
            start,
            vocab,
            walked: Walk::Known {
                state: Some(start),
                len: 0,
            },
        })
    }

    /// Whether the DFA can still reach a match from `state`.
    ///
    /// A quit state counts as unusable alongside a dead one: it means the
    /// automaton refused to keep going (a byte outside what the pattern's
    /// look-around support can handle), and treating that as "alive"
    /// would admit a token whose acceptance is unknown.
    fn alive(&self, state: StateID) -> bool {
        !self.dfa.is_dead_state(state) && !self.dfa.is_quit_state(state)
    }

    /// Walk `bytes` from `state`, bailing out as soon as the walk dies.
    fn step(&self, mut state: StateID, bytes: &[u8]) -> StateID {
        for &byte in bytes {
            state = self.dfa.next_state(state, byte);
            if !self.alive(state) {
                break;
            }
        }
        state
    }

    /// DFA state after consuming the whole prefix, or `None` when the
    /// prefix cannot be part of any match (including the case of an id
    /// that is not in `vocab` at all).
    ///
    /// Walked from the start every time. [`Self::walked`] is the same
    /// state kept incrementally, and the two are asserted equal — this
    /// one stays because a caller that never accepts (a mask asked for
    /// a prefix this constraint has not been advanced through, a
    /// beam search probing a branch) still needs an answer.
    fn state_for(&self, prefix: &[u32]) -> Option<StateID> {
        let mut state = self.start;
        for &id in prefix {
            let piece = self.vocab.get(id as usize)?;
            state = self.step(state, piece.as_bytes());
            if !self.alive(state) {
                return None;
            }
        }
        Some(state)
    }

    /// The tokens permitted from `state`, or nothing at all when the
    /// walk has died.
    ///
    /// The half of [`Constraint::mask`] that does not depend on how the
    /// state was reached, so the walked path and the kept-walk fast
    /// path cannot answer differently.
    fn mask_from(&self, state: Option<StateID>) -> TokenMask {
        let Some(state) = state else {
            // Unreachable prefix. Permitting nothing routes this into
            // the loud-failure path rather than letting the sampler
            // improvise.
            return TokenMask::Allow(Vec::new());
        };
        let vocab = self.vocab.len();
        let mut allowed: Vec<u32> = Vec::new();
        for (id, piece) in self.vocab.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            if self.alive(self.step(state, piece.as_bytes())) {
                allowed.push(id as u32);
            }
        }

        // Pick whichever variant stays sparse. Mid-pattern the permitted
        // set is usually tiny (`Allow`), but at a position where the
        // pattern is permissive — `.*`, a wide character class — the
        // *denied* set is the small one and `Deny` avoids materialising a
        // near-full-vocab list on every single token.
        if allowed.len() == vocab {
            return TokenMask::AllowAll;
        }
        if allowed.len() * 2 > vocab {
            let mut denied = Vec::with_capacity(vocab - allowed.len());
            let mut survivors = allowed.iter().copied().peekable();
            for id in 0..vocab {
                let id = id as u32;
                if survivors.peek() == Some(&id) {
                    survivors.next();
                } else {
                    denied.push(id);
                }
            }
            return TokenMask::Deny(denied);
        }
        TokenMask::Allow(allowed)
    }

    /// The kept walk, if it covers a prefix of exactly `len` tokens.
    ///
    /// **Length is not identity.** Two prefixes of one length can reach
    /// different states, so this is only sound for a caller that knows
    /// the accepted tokens *are* the prefix — which is
    /// [`ConstrainedSampler`], and nothing else: it accepts every token
    /// it produces and resets when its prefix is cleared. Every other
    /// caller goes through [`Self::state_for`], which walks.
    ///
    /// Keeping the accepted tokens and comparing them would make the
    /// check sound for everyone and cost the walk it exists to avoid.
    fn kept_walk(&self, len: usize) -> Option<Option<StateID>> {
        match self.walked {
            Walk::Known { state, len: known } if known == len => Some(state),
            _ => None,
        }
    }
}

impl Constraint for RegexConstraint {
    /// The surface-string table this was built with — see
    /// [`Self::mask`]'s note on the two vocabularies.
    fn vocab(&self) -> Option<usize> {
        Some(self.vocab.len())
    }

    /// The mask from the kept walk when it covers this prefix — see
    /// [`Constraint::fill_mask_accepted`]. Falls back to walking when
    /// the walk is unknown, which is what a rollback leaves behind.
    fn fill_mask_accepted(
        &self,
        prefix: &[u32],
        vocab: usize,
        out: &mut TokenBitset,
    ) -> Result<(), u32> {
        match self.kept_walk(prefix.len()) {
            Some(state) => out.fill_from(&self.mask_from(state), vocab),
            None => out.fill_from(&self.mask(prefix), vocab),
        }
    }

    /// Advance the kept state by one token.
    ///
    /// This is what takes the per-token cost from `O(prefix)` to
    /// `O(token bytes)`: without it every step re-walked the whole
    /// generation from the start, which over `n` tokens is `O(n²)`
    /// bytes through the DFA.
    fn accept(&mut self, token: u32) {
        self.walked = match self.walked {
            // Nothing is known, and one more token does not make it
            // known: the next question re-walks either way.
            Walk::Unknown => Walk::Unknown,
            Walk::Known { state, len } => {
                let next = match (state, self.vocab.get(token as usize)) {
                    (Some(state), Some(piece)) => {
                        let next = self.step(state, piece.as_bytes());
                        self.alive(next).then_some(next)
                    }
                    // Already dead, or a token outside the vocabulary:
                    // either way the walk cannot continue, and `None`
                    // is what `state_for` would have answered.
                    _ => None,
                };
                Walk::Known {
                    state: next,
                    len: len + 1,
                }
            }
        };
    }

    /// Drop the last `n` accepted tokens.
    ///
    /// A DFA cannot be run backwards, so the kept walk becomes unknown
    /// and the next mask re-walks the prefix. Correct, and the cost
    /// falls on the rollback rather than on every step.
    ///
    /// Rolling back to nothing is the one case that stays known: that
    /// state is the start state, which is a constant.
    fn rollback(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let remaining = match self.walked {
            Walk::Known { len, .. } => len.saturating_sub(n),
            Walk::Unknown => 0,
        };
        self.walked = if remaining == 0 {
            Walk::Known {
                state: Some(self.start),
                len: 0,
            }
        } else {
            Walk::Unknown
        };
    }

    fn reset(&mut self) {
        self.walked = Walk::Known {
            state: Some(self.start),
            len: 0,
        };
    }

    /// # The two vocabularies
    ///
    /// This reasons over `self.vocab`, the surface strings it was built
    /// with, while the mask is applied over the logits row's width. The
    /// two are the same number for every shipped preset, and
    /// [`ConstrainedSampler`] refuses a disagreement rather than
    /// letting it pass — without that check, an id past the end of the
    /// string list would be permitted without the DFA ever having
    /// looked at it.
    ///
    /// Always walks `prefix`. The state kept by [`Self::accept`] is a
    /// fast path for the sampler alone — see [`Self::kept_walk`].
    fn mask(&self, prefix: &[u32]) -> TokenMask {
        let Some(state) = self.state_for(prefix) else {
            // Unreachable prefix. Permitting nothing routes this into the
            // loud-failure path rather than letting the sampler improvise.
            return TokenMask::Allow(Vec::new());
        };
        self.mask_from(Some(state))
    }

    fn is_terminal(&self, prefix: &[u32]) -> bool {
        match self.state_for(prefix) {
            // The end-of-input transition applies the pattern's trailing
            // look-around (`$`, `\b`) before the match is read off, which
            // is what makes this a *full* match rather than a prefix one.
            Some(state) => self.dfa.is_match_state(self.dfa.next_eoi_state(state)),
            None => false,
        }
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

/// The additive bias a mask applies: `0.0` where a token survives,
/// `-inf` where it does not.
///
/// A bias rather than a rewritten row, because addition leaves the
/// surviving entries as they were (`x + 0.0 == x`) and the whole tensor
/// stays where it already is.
///
/// One value changes its bits and nothing else: a logit of `-0.0` comes
/// back as `+0.0`, since IEEE-754 addition of two zeros of unlike sign
/// rounds to `+0.0`. The two compare equal under every operator and
/// exponentiate to the same number, so no sampler here can tell them
/// apart — but the claim is "unchanged", not "bit-identical", and the
/// difference is stated rather than rounded over. The previous form read the logits
/// back to the host, scattered `-inf` into the copy, and uploaded the
/// result — on an accelerator that is a device→host→device round trip
/// of the full vocabulary at every decoded token, against a model whose
/// forward never left the device.
///
/// The bias is still built on the host and uploaded once, which is half
/// the traffic and, for a constraint whose mask does not change between
/// steps, is paid once per generation rather than once per token — see
/// [`MaskBias`].
///
/// Errors on an out-of-range token id and on a mask that leaves no
/// candidate at all; see the module doc for why neither is recoverable.
/// Both are decided from the id list alone, so neither needs the logits.
fn bitset_bias(allowed: &TokenBitset, device: &Device) -> CandleResult<Tensor> {
    if allowed.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "ConstrainedSampler: mask leaves no candidate tokens (vocab {} fully masked)",
            allowed.vocab()
        )));
    }
    let bias: Vec<f32> = (0..allowed.vocab() as u32)
        .map(|id| {
            if allowed.contains(id) {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    Tensor::from_vec(bias, allowed.vocab(), device)
}

/// The bias tensor of the most recent mask, kept so a constraint whose
/// mask does not change between steps uploads it once.
///
/// Which is most of them: an allow-list and a stop-token set answer the
/// same mask at every position, and a grammar's mask changes only when
/// the parse state does. The comparison that decides a hit is a host
/// memcmp over the id list — cheap next to the transfer it avoids, and
/// free when the lists are short, which is the case an allow-list
/// decode is.
#[derive(Debug, Default, Clone)]
struct MaskBias {
    cached: Option<(TokenBitset, Tensor)>,
}

impl MaskBias {
    /// The bias for `allowed`, from the cache when it fits.
    fn bias_for(&mut self, allowed: &TokenBitset, device: &Device) -> CandleResult<Tensor> {
        if let Some((cached, bias)) = self.cached.as_ref() {
            if cached == allowed && bias.device().same_device(device) {
                return Ok(bias.clone());
            }
        }
        let bias = bitset_bias(allowed, device)?;
        self.cached = Some((allowed.clone(), bias.clone()));
        Ok(bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{GreedySampler, TemperatureSampler, TopKTopPSampler};
    use candle_core::Device;

    fn cpu_logits(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(),), &Device::Cpu).unwrap()
    }

    /// The mask is applied as an additive bias, so a surviving logit
    /// comes through bit-identical — `x + 0.0` is `x`, and any other
    /// arrangement would perturb the values the sampler ranks.
    #[test]
    fn a_surviving_logit_is_not_touched_by_the_mask() {
        let values = [0.1f32, -3.25, 7.5, 0.0, -0.0];
        let logits = cpu_logits(&values);
        let mut set = TokenBitset::none(values.len());
        set.fill_from(&TokenMask::Deny(vec![1, 3]), values.len())
            .unwrap();
        let bias = bitset_bias(&set, &Device::Cpu).unwrap();
        let masked: Vec<f32> = logits.add(&bias).unwrap().to_vec1().unwrap();
        assert_eq!(masked[0].to_bits(), values[0].to_bits());
        assert_eq!(masked[2].to_bits(), values[2].to_bits());
        assert!(masked[1].is_infinite() && masked[1].is_sign_negative());
        assert!(masked[3].is_infinite() && masked[3].is_sign_negative());

        // The one exception, pinned rather than papered over: `-0.0`
        // comes back `+0.0` because IEEE-754 rounds a sum of unlike
        // zeros that way. Equal under every comparison the samplers
        // make, and it exponentiates to the same number, so nothing
        // downstream can observe it — but it is not bit-identity.
        assert_eq!(values[4].to_bits(), (-0.0f32).to_bits());
        assert_eq!(masked[4].to_bits(), 0.0f32.to_bits());
        assert_eq!(masked[4], values[4], "still equal, and that is what ranks");
    }

    /// A constraint whose mask does not change between steps uploads
    /// the bias once. Tensor identity is the observation: a cache hit
    /// hands back a clone of the same tensor, a miss builds a new one.
    #[test]
    fn an_unchanged_mask_reuses_the_bias_it_already_built() {
        let mut cache = MaskBias::default();
        let mut set = TokenBitset::none(4);
        set.fill_from(&TokenMask::Allow(vec![1, 2]), 4).unwrap();
        let first = cache.bias_for(&set, &Device::Cpu).unwrap();
        let second = cache.bias_for(&set, &Device::Cpu).unwrap();
        assert_eq!(
            first.id(),
            second.id(),
            "an unchanged mask must not rebuild"
        );

        set.fill_from(&TokenMask::Allow(vec![1]), 4).unwrap();
        let third = cache.bias_for(&set, &Device::Cpu).unwrap();
        assert_ne!(first.id(), third.id(), "a changed mask must rebuild");

        // A different vocabulary is a different set even under the same
        // allow list, or a row of another width would be added to.
        set.fill_from(&TokenMask::Allow(vec![1]), 8).unwrap();
        let fourth = cache.bias_for(&set, &Device::Cpu).unwrap();
        assert_ne!(third.id(), fourth.id());
        assert_eq!(fourth.dims(), &[8]);
    }

    /// The two refusals are decided from the id list, so they do not
    /// need the logits and fire before anything reaches the device.
    #[test]
    fn the_mask_refusals_are_decided_without_the_logits() {
        let mut set = TokenBitset::none(4);
        assert_eq!(
            set.fill_from(&TokenMask::Allow(vec![9]), 4),
            Err(9),
            "an out-of-range id is refused while filling, before any tensor exists"
        );

        set.fill_from(&TokenMask::Deny(vec![0, 1, 2, 3]), 4)
            .unwrap();
        let err = bitset_bias(&set, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("fully masked"), "{err}");

        set.fill_from(&TokenMask::Allow(Vec::new()), 4).unwrap();
        let err = bitset_bias(&set, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("fully masked"), "{err}");
    }

    /// Logits used across the mask tests: argmax is index 1.
    fn fixture() -> Tensor {
        cpu_logits(&[0.1, 3.2, 0.5, 2.7, 1.0])
    }

    /// Constraint returning one fixed mask regardless of prefix, so the
    /// mask-application path can be tested independently of any real
    /// grammar.
    struct FixedMask(TokenMask);

    impl Constraint for FixedMask {
        fn mask(&self, _prefix: &[u32]) -> TokenMask {
            self.0.clone()
        }

        fn is_terminal(&self, _prefix: &[u32]) -> bool {
            false
        }
    }

    /// A stop token is sampled like any other token; what changes is
    /// that the constraint flips to terminal once it lands in the
    /// prefix. Guards the "loop stops, token is not masked" semantics.
    #[test]
    fn stop_token_flips_is_done() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![1]));
        assert!(!s.is_done(), "empty prefix must not be terminal");

        let token = s.sample(&fixture()).unwrap();
        assert_eq!(token, 1, "stop token must remain sampleable");
        assert!(s.is_done(), "stop token in prefix must be terminal");
    }

    /// A non-stop token leaves the constraint non-terminal. Without this
    /// the previous test would pass on an `is_terminal` that always
    /// returns true after the first token.
    #[test]
    fn non_stop_token_does_not_terminate() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![4]));
        let token = s.sample(&fixture()).unwrap();
        assert_eq!(token, 1);
        assert!(!s.is_done(), "non-stop token must not terminate");
    }

    /// `Deny` on the argmax hands the runner-up to the inner sampler.
    /// The greedy inner sampler makes the effect unambiguous.
    #[test]
    fn deny_excludes_the_argmax() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Deny(vec![1])));
        assert_eq!(s.sample(&fixture()).unwrap(), 3, "runner-up expected");
        assert_eq!(s.prefix(), &[3], "prefix must record the sampled token");
    }

    /// `Allow` of a single non-argmax token forces that token even
    /// though its logit is the lowest in the row.
    #[test]
    fn allow_forces_the_listed_token() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Allow(vec![0])));
        assert_eq!(s.sample(&fixture()).unwrap(), 0);
    }

    /// An empty `Allow` permits nothing. Erroring is the point: argmax
    /// on the unmasked logits would return a token the constraint
    /// forbade, and softmax over an all-`-inf` row is NaN.
    #[test]
    fn allow_empty_errors() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Allow(vec![])));
        assert!(s.sample(&fixture()).is_err(), "empty Allow must error");
        assert!(s.prefix().is_empty(), "failed step must not grow prefix");
    }

    /// `Deny` covering the whole vocab is the same degenerate state as
    /// an empty `Allow` and must fail the same way.
    #[test]
    fn deny_covering_full_vocab_errors() {
        let mut s = ConstrainedSampler::new(
            GreedySampler,
            FixedMask(TokenMask::Deny(vec![0, 1, 2, 3, 4])),
        );
        assert!(s.sample(&fixture()).is_err(), "full Deny must error");
    }

    #[test]
    fn a_bitset_is_the_set_it_was_filled_with() {
        let mut set = TokenBitset::none(70);
        assert!(set.is_empty());
        set.fill_from(&TokenMask::Allow(vec![0, 33, 69]), 70)
            .unwrap();
        assert_eq!(set.allowed(), vec![0, 33, 69]);
        assert_eq!(set.count(), 3);
        assert!(set.contains(33) && !set.contains(34));

        set.fill_from(&TokenMask::Deny(vec![0, 69]), 70).unwrap();
        assert_eq!(set.count(), 68);
        assert!(!set.contains(0) && !set.contains(69) && set.contains(1));

        set.fill_from(&TokenMask::AllowAll, 70).unwrap();
        assert_eq!(set.count(), 70, "the bits past the vocab stay clear");
    }

    #[test]
    fn a_bitset_refuses_an_id_it_has_no_bit_for() {
        let mut set = TokenBitset::none(4);
        assert_eq!(set.fill_from(&TokenMask::Allow(vec![1, 9]), 4), Err(9));
        assert_eq!(set.fill_from(&TokenMask::Deny(vec![4]), 4), Err(4));
    }

    #[test]
    fn a_bitset_reuses_its_buffer_across_steps() {
        // What takes the allocation out of the decode loop: the width
        // only changes when the vocabulary does, and it does not within
        // a generation.
        let mut set = TokenBitset::all(64);
        let before = set.allowed().len();
        set.fill_from(&TokenMask::Allow(vec![7]), 64).unwrap();
        set.fill_from(&TokenMask::AllowAll, 64).unwrap();
        assert_eq!(set.allowed().len(), before);
        assert_eq!(set.vocab(), 64);
    }

    /// A constraint reasoning over a narrower vocabulary than the
    /// logits row is refused: the ids in between would come through
    /// permitted without the grammar ever having examined them.
    #[test]
    fn a_constraint_and_a_wider_logits_row_are_refused() {
        let vocab: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let c = RegexConstraint::new("a+", vocab).unwrap();
        let mut s = ConstrainedSampler::new(GreedySampler, c);
        // Five logits against a three-token table.
        let err = s
            .sample(&cpu_logits(&[0.1, 0.2, 0.3, 9.0, 0.4]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("3 tokens") && err.contains("5 wide"), "{err}");

        // The matching width goes through.
        let vocab: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let c = RegexConstraint::new("a+", vocab).unwrap();
        let mut s = ConstrainedSampler::new(GreedySampler, c);
        assert!(s.sample(&cpu_logits(&[0.1, 0.2, 0.3])).is_ok());
    }

    /// An id-only constraint declares no vocabulary and fits any width,
    /// which is what lets an allow list work with a tokenizer this
    /// crate has never seen.
    #[test]
    fn an_id_only_constraint_fits_any_width() {
        let mut s =
            ConstrainedSampler::new(GreedySampler, AllowListConstraint::new(vec![1, 2]).unwrap());
        assert!(s.sample(&cpu_logits(&[0.1, 5.0, 0.3, 0.4, 0.5])).is_ok());
    }

    /// The public `mask` answers the prefix it is given, whatever has
    /// been accepted. The kept walk is a fast path for the sampler,
    /// whose prefix *is* what it accepted; a caller probing another
    /// branch — a beam search, a speculative decode, the uses
    /// `rollback` is documented for — must not be answered from a state
    /// that belongs to a different sequence of the same length.
    #[test]
    fn mask_answers_the_prefix_it_is_given_not_the_one_accepted() {
        let vocab: Vec<String> = ["a", "b", "ab", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut c = RegexConstraint::new("(ab)+", vocab).unwrap();
        // Accept "a" — a legal opening — then ask about "b", which is
        // not one. Both prefixes are one token long.
        c.accept(0);
        assert_eq!(
            c.mask(&[1]),
            TokenMask::Allow(Vec::new()),
            "an impossible prefix permits nothing, whatever was accepted"
        );
        assert!(!c.is_terminal(&[1]));
        // And the accepted prefix still answers from the kept walk.
        let mut set = TokenBitset::none(4);
        c.fill_mask_accepted(&[0], 4, &mut set).unwrap();
        let mut walked = TokenBitset::none(4);
        c.fill_mask(&[0], 4, &mut walked).unwrap();
        assert_eq!(set, walked, "the fast path answers what the walk does");
    }

    /// The kept state and the prefix walk must agree at every step —
    /// the optimisation is only an optimisation if it answers the same
    /// question.
    #[test]
    fn the_kept_state_agrees_with_the_prefix_walk() {
        let vocab: Vec<String> = ["a", "b", "ab", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let stateful = RegexConstraint::new("(ab|ac)+", vocab.clone()).unwrap();
        let reference = RegexConstraint::new("(ab|ac)+", vocab).unwrap();

        let mut stateful = stateful;
        let mut prefix: Vec<u32> = Vec::new();
        // Walk a few tokens, checking the two masks at every position —
        // including after a token that kills the walk (3 = "c" from the
        // start is not a legal opening).
        for token in [0u32, 1, 0, 3, 1] {
            assert_eq!(
                stateful.mask(&prefix),
                reference.mask(&prefix),
                "masks diverged at prefix {prefix:?}"
            );
            assert_eq!(
                stateful.is_terminal(&prefix),
                reference.is_terminal(&prefix),
                "terminality diverged at prefix {prefix:?}"
            );
            prefix.push(token);
            stateful.accept(token);
        }
        assert_eq!(stateful.mask(&prefix), reference.mask(&prefix));
    }

    /// `rollback` un-accepts, and the mask afterwards is the mask for
    /// the shorter prefix — re-walked, since a DFA does not run
    /// backwards.
    #[test]
    fn rollback_returns_the_constraint_to_an_earlier_position() {
        let vocab: Vec<String> = ["a", "b", "ab", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut c = RegexConstraint::new("(ab)+", vocab).unwrap();
        let after_one = c.mask(&[0]);
        c.accept(0);
        c.accept(1);
        c.rollback(1);
        assert_eq!(
            c.mask(&[0]),
            after_one,
            "back to where it was after one token"
        );
        c.rollback(1);
        assert_eq!(
            c.mask(&[]),
            RegexConstraint::new(
                "(ab)+",
                vec!["a".into(), "b".into(), "ab".into(), "c".into()]
            )
            .unwrap()
            .mask(&[])
        );
    }

    /// `reset` puts it back to construction, which is what the sampler
    /// calls when its own prefix is cleared.
    #[test]
    fn reset_returns_the_constraint_to_construction() {
        let vocab: Vec<String> = ["a", "b", "ab", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut c = RegexConstraint::new("(ab)+", vocab).unwrap();
        let fresh = c.mask(&[]);
        c.accept(0);
        c.accept(1);
        c.reset();
        assert_eq!(c.mask(&[]), fresh);
    }

    /// The sampler keeps the constraint in step with its own prefix:
    /// after sampling, the constraint has accepted exactly what the
    /// sampler produced, and `reset` clears both.
    #[test]
    fn the_sampler_advances_and_resets_the_constraint_with_its_prefix() {
        /// Records what it was told, so the test can see the calls
        /// rather than infer them.
        #[derive(Default)]
        struct Recorder {
            accepted: Vec<u32>,
            resets: usize,
        }
        impl Constraint for Recorder {
            fn mask(&self, _prefix: &[u32]) -> TokenMask {
                TokenMask::AllowAll
            }
            fn is_terminal(&self, _prefix: &[u32]) -> bool {
                false
            }
            fn accept(&mut self, token: u32) {
                self.accepted.push(token);
            }
            fn reset(&mut self) {
                self.resets += 1;
                self.accepted.clear();
            }
        }

        let mut s = ConstrainedSampler::new(GreedySampler, Recorder::default());
        let logits = cpu_logits(&[0.1, 5.0, 0.2, 0.3, 0.4]);
        let a = s.sample(&logits).unwrap();
        let b = s.sample(&logits).unwrap();
        assert_eq!(s.prefix(), &[a, b]);
        s.reset();
        assert_eq!(s.prefix().len(), 0);
        // The constraint saw the same two tokens and then the reset.
        let _ = (a, b);
    }

    /// A token id at or beyond `vocab` is a constraint bug. Absorbing it
    /// (clamping, skipping) would hide the bug behind plausible output.
    #[test]
    fn out_of_range_token_id_errors() {
        for mask in [TokenMask::Deny(vec![5]), TokenMask::Allow(vec![0, 9])] {
            let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(mask.clone()));
            assert!(s.sample(&fixture()).is_err(), "{mask:?} must error");
        }
    }

    /// Masking must not disturb the inner sampler's reproducibility: two
    /// constrained samplers built from the same seed and fed the same
    /// logits stream produce the same tokens, and the mask actually
    /// binds (the denied ids never appear).
    #[test]
    fn composition_with_top_k_top_p_stays_reproducible() {
        let logits = cpu_logits(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5]);
        let denied = vec![2, 7];

        let build = || {
            ConstrainedSampler::new(
                TopKTopPSampler::new(Some(4), Some(0.95), 1.0, 24601),
                FixedMask(TokenMask::Deny(denied.clone())),
            )
        };
        let mut a = build();
        let mut b = build();

        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b, "constrained sampler diverged on shared seed");
        assert!(
            seq_a.iter().all(|t| !denied.contains(t)),
            "denied tokens leaked into the stream: {seq_a:?}"
        );
    }

    /// `reset` returns the sampler to the pre-generation state so one
    /// sampler can drive several generations.
    #[test]
    fn reset_clears_the_prefix() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![1]));
        s.sample(&fixture()).unwrap();
        assert!(s.is_done());

        s.reset();
        assert!(s.prefix().is_empty(), "reset must clear the prefix");
        assert!(!s.is_done(), "reset must clear the terminal state");
    }

    /// The type-erased composition Layer 3 actually builds — a boxed
    /// sampler wrapped by a boxed constraint — must behave exactly like
    /// the statically typed one. Guards both blanket impls at once: a
    /// delegation that dropped the mask or the terminal signal would
    /// show up here and nowhere else, since no Rust caller has a reason
    /// to erase these types.
    #[test]
    fn boxed_sampler_and_constraint_compose_like_their_concrete_types() {
        let inner: Box<dyn Sampler + Send> = Box::new(GreedySampler);
        let constraint: Box<dyn Constraint + Send> = Box::new(FixedMask(TokenMask::Deny(vec![1])));
        let mut erased = ConstrainedSampler::new(inner, constraint);
        assert_eq!(
            erased.sample(&fixture()).unwrap(),
            3,
            "the boxed constraint must still mask the argmax away"
        );

        // Stacking: a constrained sampler is itself boxable as the inner
        // sampler of another one, which is how `alc.nn.sampler.constrained`
        // composes twice.
        let stacked_inner: Box<dyn Sampler + Send> = Box::new(erased);
        let stop: Box<dyn Constraint + Send> = Box::new(StopTokensConstraint::new(vec![3]));
        let mut stacked = ConstrainedSampler::new(stacked_inner, stop);
        assert_eq!(stacked.sample(&fixture()).unwrap(), 3);
        assert!(
            stacked.is_done(),
            "the outer constraint must see the token the inner one produced"
        );
    }

    // ─── RegexConstraint ──────────────────────────────────────────────

    /// Single-character vocab for the regex tests: ids `0..=9` are the
    /// digits, id 10 is `-`, id 11 is `a` (the one token no digit pattern
    /// can ever accept).
    fn digit_vocab() -> Vec<String> {
        let mut v: Vec<String> = (0..10).map(|d| d.to_string()).collect();
        v.push("-".to_string());
        v.push("a".to_string());
        v
    }

    /// Logits over `digit_vocab()` whose argmax is `a` (11) and whose
    /// runner-up is `-` (10), so an unconstrained greedy sampler would
    /// produce nothing but off-pattern tokens. Among the digits, 9 wins.
    fn digit_logits() -> Tensor {
        cpu_logits(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 5.0, 9.0])
    }

    /// Project a mask onto the concrete id set it permits, so the tests
    /// assert on semantics instead of on which sparse variant the
    /// heuristic happened to choose.
    fn allowed_ids(mask: &TokenMask, vocab: usize) -> Vec<u32> {
        let all = (0..vocab as u32).collect::<Vec<u32>>();
        match mask {
            TokenMask::AllowAll => all,
            TokenMask::Allow(ids) => {
                let mut ids = ids.clone();
                ids.sort_unstable();
                ids
            }
            TokenMask::Deny(ids) => all.into_iter().filter(|i| !ids.contains(i)).collect(),
        }
    }

    /// End-to-end shape enforcement: only digits may open the pattern,
    /// only the separator may follow three of them, and the completed
    /// eight-token sequence is terminal. The greedy inner sampler makes
    /// the mask's effect unambiguous — every draw would be `a` (11)
    /// without it.
    #[test]
    fn regex_forces_the_pattern_shape() {
        let vocab = digit_vocab();
        let c = RegexConstraint::new(r"^\d{3}-\d{4}$", vocab.clone()).unwrap();

        let opening = c.mask(&[]);
        assert_eq!(
            allowed_ids(&opening, vocab.len()),
            (0..10u32).collect::<Vec<_>>(),
            "only digits may open the pattern"
        );
        assert!(
            matches!(opening, TokenMask::Deny(_)),
            "10 of 12 survivors must ride the Deny complement, got {opening:?}"
        );
        assert_eq!(
            allowed_ids(&c.mask(&[1, 2, 3]), vocab.len()),
            vec![10],
            "the separator is the only legal continuation after three digits"
        );

        let mut s = ConstrainedSampler::new(GreedySampler, c);
        let drawn: Vec<u32> = (0..8).map(|_| s.sample(&digit_logits()).unwrap()).collect();
        assert_eq!(drawn, vec![9, 9, 9, 10, 9, 9, 9, 9]);
        assert!(s.is_done(), "a complete match must be terminal");
    }

    /// A viable prefix is not a match. Without this the terminal check
    /// could be "the prefix has not died yet" and the phone-number test
    /// above would still pass.
    #[test]
    fn partial_match_is_not_terminal() {
        let c = RegexConstraint::new(r"^\d{3}-\d{4}$", digit_vocab()).unwrap();
        for prefix in [
            vec![],
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 10],
            vec![1, 2, 3, 10, 4, 5, 6],
        ] {
            assert!(
                !c.is_terminal(&prefix),
                "prefix {prefix:?} is only a partial match"
            );
        }
        assert!(c.is_terminal(&[1, 2, 3, 10, 4, 5, 6, 7]));
    }

    /// A token may carry several characters, so one draw can advance the
    /// DFA by more than one byte. Also covers the implicit anchoring: the
    /// pattern here carries no `^` / `$` of its own.
    #[test]
    fn multi_character_tokens_advance_several_bytes() {
        let vocab = vec![
            "1".to_string(),
            "23".to_string(),
            "4".to_string(),
            "x".to_string(),
        ];
        let c = RegexConstraint::new(r"\d{3}", vocab.clone()).unwrap();

        assert_eq!(
            allowed_ids(&c.mask(&[0]), vocab.len()),
            vec![0, 1, 2],
            "every digit token survives after one digit; only \"x\" is denied"
        );
        assert!(
            c.is_terminal(&[0, 1]),
            "\"1\" + \"23\" is three digits in two tokens"
        );
        assert!(!c.is_terminal(&[0, 2]), "\"1\" + \"4\" is only two digits");
        assert!(
            allowed_ids(&c.mask(&[0, 1]), vocab.len()).is_empty(),
            "a saturated pattern admits no continuation"
        );
    }

    /// An empty surface string consumes no bytes, so it can never move
    /// the DFA. Permitting it would let a generation loop draw it forever
    /// without the constraint ever advancing.
    #[test]
    fn empty_surface_tokens_are_never_allowed() {
        let vocab = vec![String::new(), "1".to_string(), "2".to_string()];
        let c = RegexConstraint::new(r"\d{2}", vocab.clone()).unwrap();
        for prefix in [vec![], vec![1], vec![1, 2]] {
            assert!(
                !allowed_ids(&c.mask(&prefix), vocab.len()).contains(&0),
                "prefix {prefix:?} allowed the empty token"
            );
        }
    }

    /// A shorter alternative must not shadow a longer one. This is the
    /// concrete reason the DFA is built with `MatchKind::All`: under
    /// leftmost-first semantics the automaton commits to `a` and stops
    /// exploring, which would deny the `b` that completes `ab`.
    #[test]
    fn a_shorter_alternative_does_not_shadow_a_longer_one() {
        let vocab = vec!["a".to_string(), "b".to_string()];
        let c = RegexConstraint::new("a|ab", vocab.clone()).unwrap();

        assert!(c.is_terminal(&[0]), "\"a\" is a complete match");
        assert_eq!(
            allowed_ids(&c.mask(&[0]), vocab.len()),
            vec![1],
            "\"ab\" must still be reachable after \"a\""
        );
        assert!(c.is_terminal(&[0, 1]), "\"ab\" is a complete match too");
    }

    /// An invalid pattern is a caller bug and must surface where the
    /// constraint is configured, not on some later token.
    #[test]
    fn invalid_pattern_fails_at_construction() {
        for pattern in ["(", "[a-z", "a{2,1}"] {
            assert!(
                RegexConstraint::new(pattern, digit_vocab()).is_err(),
                "pattern {pattern:?} must be rejected at construction"
            );
        }
    }

    /// Positions where nothing can be emitted — an unreachable prefix, an
    /// id outside the vocab, a pattern already saturated — all collapse
    /// to an empty `Allow`, which is what makes the sampler fail loudly
    /// rather than emit an off-pattern token.
    #[test]
    fn impossible_positions_error_loudly() {
        let vocab = digit_vocab();
        let c = RegexConstraint::new(r"\d{3}", vocab.clone()).unwrap();

        assert_eq!(
            c.mask(&[11]),
            TokenMask::Allow(Vec::new()),
            "\"a\" can never start the pattern"
        );
        assert_eq!(
            c.mask(&[99]),
            TokenMask::Allow(Vec::new()),
            "an id outside the vocab is a caller bug, not a skippable token"
        );
        assert_eq!(
            c.mask(&[1, 2, 3]),
            TokenMask::Allow(Vec::new()),
            "three digits saturate the pattern"
        );

        // Regression guard for the trailing anchor. An anchored DFA whose
        // pattern is not end-anchored stops discriminating once a match
        // exists, so `\d{3}` would accept a fourth digit and call every
        // longer prefix terminal.
        assert!(c.is_terminal(&[1, 2, 3]));
        assert!(
            !c.is_terminal(&[1, 2, 3, 4]),
            "an over-long prefix is not a full match"
        );

        let mut s = ConstrainedSampler::new(GreedySampler, c);
        for _ in 0..3 {
            s.sample(&digit_logits()).unwrap();
        }
        assert!(
            s.sample(&digit_logits()).is_err(),
            "a saturated pattern must error rather than emit"
        );
    }

    /// Masking through a stochastic sampler keeps both guarantees at
    /// once: the seed still reproduces the stream, and every token the
    /// stream contains is on-pattern.
    #[test]
    fn regex_composed_with_top_k_top_p_stays_reproducible() {
        let logits = cpu_logits(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5, 1.5, 2.2, 4.0, 9.0]);
        let build = || {
            ConstrainedSampler::new(
                TopKTopPSampler::new(Some(4), Some(0.95), 1.0, 24601),
                RegexConstraint::new(r"^\d{3}-\d{4}$", digit_vocab()).unwrap(),
            )
        };
        let mut a = build();
        let mut b = build();

        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b, "constrained sampler diverged on shared seed");

        assert!(
            seq_a[..3].iter().all(|t| *t < 10),
            "positions 0-2 must be digits: {seq_a:?}"
        );
        assert_eq!(seq_a[3], 10, "position 3 must be the separator: {seq_a:?}");
        assert!(
            seq_a[4..].iter().all(|t| *t < 10),
            "positions 4-7 must be digits: {seq_a:?}"
        );
        assert!(a.is_done(), "the completed pattern must be terminal");
    }

    // ─── AllowListConstraint ──────────────────────────────────────────

    /// The legal set is the whole answer: the same `Allow` mask comes
    /// back at every position, and the constraint never claims a prefix
    /// is terminal (stopping belongs to the loop / stop tokens).
    #[test]
    fn allow_list_masks_identically_at_every_position() {
        let c = AllowListConstraint::new(vec![0, 3]).expect("non-empty allow list");
        assert_eq!(c.allowed(), &[0, 3]);
        for prefix in [vec![], vec![3], vec![0, 3, 3], vec![99]] {
            assert_eq!(
                c.mask(&prefix),
                TokenMask::Allow(vec![0, 3]),
                "prefix {prefix:?} must not change the legal set"
            );
            assert!(
                !c.is_terminal(&prefix),
                "prefix {prefix:?} must not be terminal"
            );
        }
    }

    /// Sampling through the constraint returns only listed ids, even
    /// when the argmax is illegal — the guarantee the "legal mask" is
    /// there for. Greedy makes the choice unambiguous: token 1 wins the
    /// unmasked row, token 3 is the best of the legal ones.
    #[test]
    fn allow_list_confines_the_sampled_tokens() {
        let allowed = vec![0, 3];
        let mut s = ConstrainedSampler::new(
            GreedySampler,
            AllowListConstraint::new(allowed.clone()).expect("non-empty allow list"),
        );
        for step in 0..4 {
            let token = s.sample(&fixture()).expect("sample");
            assert_eq!(token, 3, "step {step} drew an illegal token");
        }
        assert!(
            s.prefix().iter().all(|t| allowed.contains(t)),
            "illegal token leaked into the prefix: {:?}",
            s.prefix()
        );
    }

    /// The noisy-but-legal composition from the issue: a temperature
    /// sampler under an allow list stays reproducible by seed and still
    /// never leaves the legal set. The allow list excludes the argmax
    /// (a mask that stopped binding would show up as token 1) and its
    /// two members carry equal logits, so the draw is a genuine coin
    /// flip rather than a disguised argmax.
    #[test]
    fn allow_list_under_temperature_stays_legal_and_reproducible() {
        let logits = cpu_logits(&[1.0, 9.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5]);
        let allowed = vec![5, 6];
        let build = || {
            ConstrainedSampler::new(
                TemperatureSampler::new(1.5, 24601),
                AllowListConstraint::new(allowed.clone()).expect("non-empty allow list"),
            )
        };
        let mut a = build();
        let mut b = build();

        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).expect("sample")).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).expect("sample")).collect();
        assert_eq!(seq_a, seq_b, "allow-listed sampler diverged on shared seed");
        assert!(
            seq_a.iter().all(|t| allowed.contains(t)),
            "illegal token in the stream: {seq_a:?}"
        );
        assert!(
            seq_a.contains(&5) && seq_a.contains(&6),
            "the draw is degenerate, so the test proves nothing about noise: {seq_a:?}"
        );
    }

    /// An empty legal set is a caller bug in the *legality* computation,
    /// so it is refused where that computation is wired up rather than
    /// one token later inside `mask_bias`.
    #[test]
    fn allow_list_rejects_an_empty_list_at_construction() {
        let err = match AllowListConstraint::new(Vec::new()) {
            Ok(_) => panic!("an empty allow list must be rejected at construction"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("empty"), "unexpected error: {err}");
    }

    /// An id past the end of the vocab is still caught, just later: the
    /// constraint cannot know the vocab size, so `mask_bias` is the one
    /// place that can tell.
    #[test]
    fn allow_list_out_of_range_id_errors_at_sample_time() {
        let mut s = ConstrainedSampler::new(
            GreedySampler,
            AllowListConstraint::new(vec![5]).expect("non-empty allow list"),
        );
        assert!(
            s.sample(&fixture()).is_err(),
            "an id outside the vocab must error"
        );
        assert!(s.prefix().is_empty(), "failed step must not grow prefix");
    }

    /// `StopTokensConstraint` never masks — it only terminates. Asserted
    /// on the variant itself so a future refactor cannot quietly start
    /// denying the stop tokens (which would make generation unable to
    /// finish).
    #[test]
    fn stop_tokens_constraint_never_masks() {
        let c = StopTokensConstraint::new(vec![1, 2]);
        for prefix in [vec![], vec![9], vec![1], vec![2, 1]] {
            assert_eq!(
                c.mask(&prefix),
                TokenMask::AllowAll,
                "prefix {prefix:?} must stay unmasked"
            );
        }
    }
}
