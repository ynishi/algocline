//! Amplifying a condition token at decode time.
//!
//! # Why amplify at all
//!
//! Measured on the three-band chess checkpoint, the divergence between
//! what the model plays as a 1100 and as a 1900 is 0.0097 bits, against
//! 0.33 bits from a uniform draw. At the one position where the human
//! gap can be measured cleanly — the start position, where prefixes
//! repeat enough to count — humans differ by 0.0180 bits and the model
//! reproduces 0.0151 of it. The model is close to the ceiling; the
//! ceiling is just low. Two club players and two strong players open
//! with the same handful of moves.
//!
//! Fidelity to that ceiling is the right target when the goal is
//! predicting people. It is the wrong target when the goal is an
//! opponent that feels like a different player, because a difference
//! of 0.018 bits is not something anyone notices across a game.
//!
//! # The mechanism
//!
//! Classifier-free guidance, as carried over to autoregressive
//! language models by Sanchez et al. (arXiv:2306.17806): take the
//! logits the model produces under the condition, and extrapolate away
//! from a reference in that direction.
//!
//! ```text
//! guided = reference + gamma * (conditioned - reference)
//! ```
//!
//! `gamma = 1` returns the conditioned logits untouched, `gamma = 0`
//! collapses to the reference, and `gamma > 1` overshoots: whatever
//! the condition was doing, it does more of.
//!
//! # What stands in for "unconditional"
//!
//! Guidance normally contrasts against a model run with no condition
//! at all, which requires having trained with the condition dropped
//! out some of the time. This checkpoint never saw a row without a
//! band token, so there is no unconditional branch to contrast with.
//!
//! [`mean_logits`] substitutes the average across the bands the model
//! does carry. It is the population the bands were split out of, which
//! is the thing a band is a deviation from — so extrapolating away
//! from it amplifies exactly what makes a band that band. The
//! substitution is worth stating plainly: it is not the same object
//! the original paper contrasts against, and a checkpoint trained with
//! condition dropout would give a truer reference.
//!
//! # The known cost
//!
//! Guidance trades diversity for fidelity to the condition, and
//! overshooting is a documented failure mode — the literature calls it
//! over-guidance. Here it should show up as the legal-move mass
//! draining away as gamma climbs, since nothing constrains the
//! extrapolated logits to stay on moves that exist. Measure it rather
//! than assume a gamma.

/// Elementwise mean of several logit rows.
///
/// Used as the reference guidance extrapolates away from. Returns an
/// empty vector when given no rows.
///
/// # Panics
///
/// Panics if the rows are not all the same length — that would mean
/// two runs of the same model over the same vocabulary disagreed on
/// its size, which is a bug rather than an input error.
pub fn mean_logits(rows: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(
            row.len(),
            first.len(),
            "logit row {i} has length {} but row 0 has {}",
            row.len(),
            first.len()
        );
    }
    let n = rows.len() as f32;
    (0..first.len())
        .map(|j| rows.iter().map(|r| r[j]).sum::<f32>() / n)
        .collect()
}

/// Extrapolate `conditioned` away from `reference` by `gamma`.
///
/// `gamma = 1` is the identity, so a caller sweeping gamma gets the
/// unguided model for free at one end of the sweep.
///
/// # Panics
///
/// Panics on a length mismatch, for the same reason as
/// [`mean_logits`].
pub fn guide_logits(conditioned: &[f32], reference: &[f32], gamma: f32) -> Vec<f32> {
    assert_eq!(
        conditioned.len(),
        reference.len(),
        "conditioned has {} logits but reference has {}",
        conditioned.len(),
        reference.len()
    );
    conditioned
        .iter()
        .zip(reference.iter())
        .map(|(c, r)| r + gamma * (c - r))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_of_one_changes_nothing() {
        let cond = vec![1.0, -2.0, 0.5];
        let refr = vec![0.0, 1.0, 1.0];
        assert_eq!(guide_logits(&cond, &refr, 1.0), cond);
    }

    #[test]
    fn gamma_of_zero_collapses_to_the_reference() {
        let cond = vec![1.0, -2.0, 0.5];
        let refr = vec![0.0, 1.0, 1.0];
        assert_eq!(guide_logits(&cond, &refr, 0.0), refr);
    }

    #[test]
    fn gamma_above_one_overshoots_in_the_same_direction() {
        let cond = vec![2.0, 0.0];
        let refr = vec![1.0, 1.0];
        // Each coordinate moves twice as far from the reference.
        assert_eq!(guide_logits(&cond, &refr, 2.0), vec![3.0, -1.0]);
    }

    #[test]
    fn a_condition_that_did_nothing_stays_doing_nothing() {
        // Guidance cannot manufacture an effect that is not there:
        // when the condition equals the reference, every gamma agrees.
        let same = vec![0.3, 0.7, -1.0];
        for gamma in [0.0, 1.0, 4.0, 16.0] {
            assert_eq!(guide_logits(&same, &same, gamma), same);
        }
    }

    #[test]
    fn the_mean_of_one_row_is_that_row() {
        let rows = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(mean_logits(&rows), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_mean_averages_coordinatewise() {
        let rows = vec![vec![0.0, 4.0], vec![2.0, 0.0]];
        assert_eq!(mean_logits(&rows), vec![1.0, 2.0]);
    }

    #[test]
    fn no_rows_means_no_reference() {
        assert!(mean_logits(&[]).is_empty());
    }
}
