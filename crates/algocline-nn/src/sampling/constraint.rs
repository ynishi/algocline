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

use candle_core::{Result as CandleResult, Tensor};
use regex_automata::{
    dfa::{dense, Automaton, StartKind},
    util::primitives::StateID,
    Anchored, Input, MatchKind,
};

use super::{validate_logits, Sampler};

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
/// Implementations are expected to be pure with respect to the prefix
/// (`&self`, no interior mutation): the same prefix must yield the same
/// mask, which is what makes seeded generation reproducible end to end.
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
}

impl<S: Sampler, C: Constraint> ConstrainedSampler<S, C> {
    /// Wrap `inner` with `constraint`, starting from an empty prefix.
    pub fn new(inner: S, constraint: C) -> Self {
        Self {
            inner,
            constraint,
            prefix: Vec::new(),
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
    }
}

impl<S: Sampler, C: Constraint> Sampler for ConstrainedSampler<S, C> {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        let mask = self.constraint.mask(&self.prefix);
        let token = match mask {
            // Free path: hand the caller's tensor straight through.
            TokenMask::AllowAll => self.inner.sample(logits)?,
            _ => {
                let masked = apply_mask(logits, &mask)?;
                self.inner.sample(&masked)?
            }
        };
        self.prefix.push(token);
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
/// An empty legal set permits nothing, which [`apply_mask`] refuses at
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
/// `Allow` is rejected by [`apply_mask`], so the condition surfaces as a
/// loud `Err` from [`Sampler::sample`] instead of quietly emitting an
/// off-pattern token.
#[derive(Debug, Clone)]
pub struct RegexConstraint {
    dfa: dense::DFA<Vec<u32>>,
    /// Anchored start state, resolved once at construction so the
    /// per-step walk cannot fail.
    start: StateID,
    vocab: Vec<String>,
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
        Ok(Self { dfa, start, vocab })
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
}

impl Constraint for RegexConstraint {
    fn mask(&self, prefix: &[u32]) -> TokenMask {
        let Some(state) = self.state_for(prefix) else {
            // Unreachable prefix. Permitting nothing routes this into the
            // loud-failure path rather than letting the sampler improvise.
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

/// Build a new logits row with the masked-out entries set to `-inf`.
///
/// Round-trips through host memory (`to_vec1` → mutate → `from_vec`)
/// rather than composing tensor ops: the mask is sparse and the vocab
/// row is a single vector, so the scatter is cheaper than materialising
/// a full-width mask tensor. The result lands on the input's device.
///
/// Errors on an out-of-range token id and on a mask that leaves no
/// candidate at all; see the module doc for why neither is recoverable.
fn apply_mask(logits: &Tensor, mask: &TokenMask) -> CandleResult<Tensor> {
    validate_logits(logits)?;
    let mut values = logits.to_vec1::<f32>()?;
    let vocab = values.len();

    // `keep` starts at the variant's default answer and the id list
    // flips the exceptions, so both variants share one scatter loop.
    let (ids, kept_default) = match mask {
        TokenMask::AllowAll => return Ok(logits.clone()),
        TokenMask::Deny(ids) => (ids, true),
        TokenMask::Allow(ids) => {
            if ids.is_empty() {
                return Err(candle_core::Error::Msg(
                    "ConstrainedSampler: TokenMask::Allow with an empty token list leaves no candidate tokens".into(),
                ));
            }
            (ids, false)
        }
    };

    let mut keep = vec![kept_default; vocab];
    for &id in ids {
        let idx = id as usize;
        if idx >= vocab {
            return Err(candle_core::Error::Msg(format!(
                "ConstrainedSampler: mask token id {id} is out of range for vocab {vocab}"
            )));
        }
        keep[idx] = !kept_default;
    }

    if keep.iter().all(|k| !*k) {
        return Err(candle_core::Error::Msg(format!(
            "ConstrainedSampler: mask leaves no candidate tokens (vocab {vocab} fully masked)"
        )));
    }

    for (value, keep) in values.iter_mut().zip(keep) {
        if !keep {
            *value = f32::NEG_INFINITY;
        }
    }

    Tensor::from_vec(values, vocab, logits.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{GreedySampler, TemperatureSampler, TopKTopPSampler};
    use candle_core::Device;

    fn cpu_logits(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(),), &Device::Cpu).unwrap()
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
    /// one token later inside `apply_mask`.
    #[test]
    fn allow_list_rejects_an_empty_list_at_construction() {
        let err = match AllowListConstraint::new(Vec::new()) {
            Ok(_) => panic!("an empty allow list must be rejected at construction"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("empty"), "unexpected error: {err}");
    }

    /// An id past the end of the vocab is still caught, just later: the
    /// constraint cannot know the vocab size, so `apply_mask` is the one
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
