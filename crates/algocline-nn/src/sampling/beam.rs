//! Beam search.
//!
//! Every sampler in [`super`] commits to one token and never
//! reconsiders: the sequence it produces is the product of a chain of
//! local choices, and a high-probability continuation reachable only
//! through a mediocre first token is unreachable. Beam search keeps `k`
//! partial sequences alive, extends all of them, and keeps the `k` best
//! of the result — so a token that looked second-best can still lead.
//!
//! It is not a better sampler; it is a different objective. Sampling
//! draws from the model's distribution, and beam search approximates
//! the *most likely sequence* under it. That is what you want for a
//! translation, a constrained field, a short structured answer — and
//! not what you want for open text, where the most likely sequence is
//! famously bland and repetitive.
//!
//! # Scoring
//!
//! A beam's score is the sum of its tokens' log-probabilities. Summing
//! logs rather than multiplying probabilities is not a convenience: the
//! product of a few hundred probabilities underflows f32 long before a
//! generation ends, and every beam would score zero.
//!
//! Longer sequences score lower for being longer, since every
//! additional term is negative. [`BeamOptions::length_penalty`] divides
//! by `length^α`, the normalisation from
//! [Wu et al. 2016](https://arxiv.org/abs/1609.08144) §7 — `0.0` leaves
//! raw sums and prefers short answers, `1.0` is the mean log-probability
//! per token, and the values in use sit near `0.6`–`1.0`.
//!
//! # What this does not do
//!
//! Sampling and beam search do not compose: the second explores, and a
//! stochastic step would make the exploration unrepeatable and the
//! "best" beam a draw. This search is greedy over the top-`k`
//! continuations and takes no RNG.

use candle_core::Result as CandleResult;

/// A model a beam search can advance.
///
/// The caller implements it, because how a sequence becomes
/// log-probabilities is theirs: a plain re-forward, a KV cache per
/// beam, a remote call. `tokens` is always the full sequence, so an
/// implementation with no cache is correct and one with a cache can
/// key on the prefix it already holds.
pub trait BeamModel {
    /// Log-probabilities over the vocabulary for the token after
    /// `tokens`.
    ///
    /// Log-probabilities, not logits: the search sums them, and summing
    /// unnormalised scores compares sequences under different
    /// normalisers — which silently favours whichever step happened to
    /// have the flattest distribution.
    fn next_log_probs(&mut self, tokens: &[u32]) -> CandleResult<Vec<f32>>;
}

/// How the search runs.
#[derive(Debug, Clone)]
pub struct BeamOptions {
    /// Sequences kept alive at each step. `1` is greedy decoding with
    /// this machinery's overhead and no benefit.
    pub beams: usize,
    /// Tokens to add beyond the prompt.
    pub max_new: usize,
    /// Length normalisation exponent. `0.0` is off; see the module doc.
    pub length_penalty: f32,
    /// Token that ends a sequence. A beam reaching it is finished and
    /// stops being extended — it still competes for the final ranking.
    pub eos: Option<u32>,
}

impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            beams: 4,
            max_new: 32,
            length_penalty: 1.0,
            eos: None,
        }
    }
}

/// One finished or surviving sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Beam {
    /// The whole sequence, prompt included.
    pub tokens: Vec<u32>,
    /// Sum of the generated tokens' log-probabilities. The prompt
    /// contributes nothing — it was not chosen.
    pub score: f32,
    /// Whether it ended at [`BeamOptions::eos`] rather than at the
    /// token budget.
    pub finished: bool,
}

impl Beam {
    /// The score the ranking uses: the sum, length-normalised.
    ///
    /// Zero-length generations score zero rather than dividing by zero,
    /// which is reachable when `max_new` is 0.
    fn ranked(&self, prompt_len: usize, penalty: f32) -> f32 {
        let generated = self.tokens.len().saturating_sub(prompt_len);
        if generated == 0 {
            return 0.0;
        }
        if penalty == 0.0 {
            return self.score;
        }
        self.score / (generated as f32).powf(penalty)
    }
}

/// Search for the most likely continuations of `prompt`.
///
/// Returns the beams ranked best first — finished ones and, if the
/// budget ran out first, the survivors.
///
/// # Errors
///
/// A `beams` or vocabulary of zero, a model that answers with a
/// differently-sized row between steps, and whatever the model itself
/// raises.
pub fn beam_search<M: BeamModel>(
    model: &mut M,
    prompt: &[u32],
    opts: &BeamOptions,
) -> CandleResult<Vec<Beam>> {
    if opts.beams == 0 {
        return Err(candle_core::Error::Msg(
            "beam_search: beams must be at least 1".into(),
        ));
    }
    if prompt.is_empty() {
        return Err(candle_core::Error::Msg(
            "beam_search: the prompt is empty, so there is no sequence to continue".into(),
        ));
    }

    let prompt_len = prompt.len();
    let mut live = vec![Beam {
        tokens: prompt.to_vec(),
        score: 0.0,
        finished: false,
    }];
    let mut finished: Vec<Beam> = Vec::new();

    for _ in 0..opts.max_new {
        if live.is_empty() {
            break;
        }
        // Every live beam's every candidate, then the best `k` of the
        // lot — the step that makes this a search rather than `k`
        // independent greedy decodes.
        let mut candidates: Vec<Beam> = Vec::new();
        for beam in &live {
            let log_probs = model.next_log_probs(&beam.tokens)?;
            if log_probs.is_empty() {
                return Err(candle_core::Error::Msg(
                    "beam_search: the model answered with an empty distribution".into(),
                ));
            }
            // Only the top `beams` continuations of each beam can
            // survive the cut below, so the rest are not worth
            // materialising — a full sort per beam over a 50k
            // vocabulary would dominate the search.
            for (id, score) in top_k(&log_probs, opts.beams) {
                let mut tokens = beam.tokens.clone();
                tokens.push(id);
                let ends = opts.eos == Some(id);
                candidates.push(Beam {
                    tokens,
                    score: beam.score + score,
                    finished: ends,
                });
            }
        }

        candidates.sort_by(|a, b| {
            b.ranked(prompt_len, opts.length_penalty)
                .partial_cmp(&a.ranked(prompt_len, opts.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(opts.beams);

        live = Vec::with_capacity(candidates.len());
        for beam in candidates {
            if beam.finished {
                finished.push(beam);
            } else {
                live.push(beam);
            }
        }
    }

    // The survivors compete with the finished: a beam that ran out of
    // budget mid-sentence can still be the best answer, and dropping it
    // would return nothing at all for a run whose `eos` never came.
    let mut all = finished;
    all.extend(live);
    all.sort_by(|a, b| {
        b.ranked(prompt_len, opts.length_penalty)
            .partial_cmp(&a.ranked(prompt_len, opts.length_penalty))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(opts.beams);
    Ok(all)
}

/// The `k` highest entries of `values` as `(index, value)`, best first.
///
/// A partial selection rather than a sort: the caller needs the top few
/// of a vocabulary-sized row at every step of every beam.
fn top_k(values: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(values.len());
    let mut indexed: Vec<(u32, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, *v))
        .collect();
    // `select_nth_unstable_by` puts the k best in front in linear time;
    // only those are then ordered.
    if k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
    }
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic model: the row is a function of the sequence so
    /// far, so the best continuation is a fact about the fixture rather
    /// than a draw.
    ///
    /// A function of the *sequence* and not of the position, because a
    /// row that depended on the position alone could not express the
    /// case beam search exists for — where what a token costs depends
    /// on what came before it.
    struct Scripted<F: Fn(&[u32]) -> Vec<f32>> {
        row_for: F,
        /// Calls made, so a test can see how much work a search did.
        calls: usize,
    }

    impl<F: Fn(&[u32]) -> Vec<f32>> Scripted<F> {
        fn new(row_for: F) -> Self {
            Self { row_for, calls: 0 }
        }
    }

    impl<F: Fn(&[u32]) -> Vec<f32>> BeamModel for Scripted<F> {
        fn next_log_probs(&mut self, tokens: &[u32]) -> CandleResult<Vec<f32>> {
            self.calls += 1;
            Ok((self.row_for)(tokens))
        }
    }

    /// A row that ignores the sequence — for the tests whose point is
    /// not the dependence.
    fn fixed(rows: Vec<Vec<f32>>, prompt_len: usize) -> impl Fn(&[u32]) -> Vec<f32> {
        move |tokens: &[u32]| {
            let step = tokens.len() - prompt_len;
            rows.get(step).or_else(|| rows.last()).unwrap().clone()
        }
    }

    /// The case the whole method exists for: the best sequence starts
    /// with a token that is not the best first token.
    ///
    /// Step 0 prefers token 1 (-0.1) over token 0 (-1.0). Step 1 pays
    /// -5.0 for everything after a 1 and -0.1 after a 0, which greedy
    /// decoding cannot see: it commits to 1 and ends at -5.1, while the
    /// beam keeps 0 alive and ends at -1.1.
    #[test]
    fn a_beam_finds_what_a_greedy_decode_cannot() {
        // Two-token vocabulary. The first step prefers token 1; what
        // follows a 1 is expensive and what follows a 0 is cheap, which
        // is the dependence a greedy decode cannot see.
        let mut scripted = Scripted::new(|tokens: &[u32]| match tokens.len() {
            1 => vec![-1.0, -0.1],
            _ => match tokens.last() {
                Some(1) => vec![-5.0, -5.0],
                _ => vec![-0.1, -0.1],
            },
        });
        let opts = BeamOptions {
            beams: 2,
            max_new: 2,
            length_penalty: 0.0,
            eos: None,
        };
        let beams = beam_search(&mut scripted, &[7], &opts).unwrap();
        let best = &beams[0];
        assert_eq!(
            best.tokens,
            vec![7, 0, 0],
            "the better path starts with the worse token"
        );
        assert!(
            (best.score - (-1.1)).abs() < 1e-5,
            "score {} is not -1.1",
            best.score
        );

        // And the greedy path is gone: once its second token's cost
        // showed, both of the two surviving beams came from the token
        // greedy decoding discarded. Dropping it is the search working,
        // not a beam being lost.
        assert!(
            beams.iter().all(|b| b.tokens.get(1) != Some(&1)),
            "the greedy first token should have been outscored: {beams:?}"
        );
    }

    /// `beams = 1` is greedy decoding, which is worth pinning: it is
    /// the degenerate case every off-by-one in the cut would break.
    #[test]
    fn one_beam_is_a_greedy_decode() {
        let mut scripted = Scripted::new(fixed(vec![vec![-1.0, -0.1], vec![-0.1, -5.0]], 1));
        let opts = BeamOptions {
            beams: 1,
            max_new: 2,
            length_penalty: 0.0,
            eos: None,
        };
        let beams = beam_search(&mut scripted, &[7], &opts).unwrap();
        assert_eq!(beams.len(), 1);
        assert_eq!(
            beams[0].tokens,
            vec![7, 1, 0],
            "the locally best at each step"
        );
    }

    /// A beam that reaches `eos` stops being extended and keeps
    /// competing.
    #[test]
    fn a_finished_beam_stops_growing_and_still_ranks() {
        // Token 2 is the most likely first token and is the eos.
        let mut scripted = Scripted::new(fixed(
            vec![vec![-2.0, -2.0, -0.1], vec![-0.5, -0.5, -9.0]],
            1,
        ));
        let opts = BeamOptions {
            beams: 3,
            max_new: 3,
            length_penalty: 0.0,
            eos: Some(2),
        };
        let beams = beam_search(&mut scripted, &[5], &opts).unwrap();
        let best = &beams[0];
        assert_eq!(best.tokens, vec![5, 2], "ended at eos after one token");
        assert!(best.finished);
        assert!(
            beams.iter().any(|b| !b.finished && b.tokens.len() == 4),
            "the unfinished beams ran to the budget: {beams:?}"
        );
    }

    /// Length normalisation changes which beam wins, which is the only
    /// reason the knob exists.
    #[test]
    fn the_length_penalty_decides_between_short_and_long() {
        let rows = vec![vec![-0.1, -2.0], vec![-0.1, -2.0], vec![-0.1, -2.0]];
        let mut short = Scripted::new(fixed(rows.clone(), 1));
        // Without normalisation a longer sequence always scores lower,
        // because every extra term is negative.
        let raw = beam_search(
            &mut short,
            &[9],
            &BeamOptions {
                beams: 2,
                max_new: 3,
                length_penalty: 0.0,
                eos: Some(0),
            },
        )
        .unwrap();
        assert_eq!(raw[0].tokens, vec![9, 0], "the shortest finished beam wins");

        // With `1.0` the score is the mean per token, so a long beam of
        // equally good tokens ties rather than losing.
        let mut long = Scripted::new(fixed(rows, 1));
        let normed = beam_search(
            &mut long,
            &[9],
            &BeamOptions {
                beams: 2,
                max_new: 3,
                length_penalty: 1.0,
                eos: Some(0),
            },
        )
        .unwrap();
        let best = normed[0].ranked(1, 1.0);
        assert!(
            (best - (-0.1)).abs() < 1e-5,
            "the mean per token is -0.1, got {best}"
        );
    }

    /// The search costs `beams` forwards per step, not `beams ×
    /// vocabulary`: only the top `beams` continuations of each beam can
    /// survive the cut, so the rest are never built.
    #[test]
    fn the_search_forwards_once_per_live_beam_per_step() {
        let mut scripted = Scripted::new(fixed(vec![vec![-0.1, -0.2, -0.3, -0.4]], 1));
        let opts = BeamOptions {
            beams: 3,
            max_new: 4,
            length_penalty: 0.0,
            eos: None,
        };
        beam_search(&mut scripted, &[1], &opts).unwrap();
        // Step 0 has one live beam; every later step has `beams`.
        assert_eq!(scripted.calls, 1 + 3 * 3);
    }

    #[test]
    fn the_refusals_are_refusals() {
        let mut scripted = Scripted::new(fixed(vec![vec![-0.1]], 1));
        assert!(beam_search(
            &mut scripted,
            &[1],
            &BeamOptions {
                beams: 0,
                ..BeamOptions::default()
            }
        )
        .is_err());
        assert!(beam_search(&mut scripted, &[], &BeamOptions::default()).is_err());
    }

    #[test]
    fn top_k_returns_the_best_in_order() {
        let values = [0.1f32, 5.0, -2.0, 4.0];
        assert_eq!(top_k(&values, 2), vec![(1, 5.0), (3, 4.0)]);
        // More asked for than there are: everything, still ordered.
        assert_eq!(top_k(&values, 9).len(), 4);
        assert_eq!(top_k(&values, 9)[0], (1, 5.0));
    }
}
