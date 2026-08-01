//! Unit tests for the four metric primitives + `validate` gates.
//!
//! Numerical assertions use hand-derived reference values (uniform vs
//! uniform → 0, entropy of uniform_n → ln(n), TVD between disjoint
//! one-hots → 1.0) so the tests double as an executable statement of
//! the primitive semantics.

use super::*;

/// Uniform distribution of length `n` (each entry = `1/n`).
fn uniform(n: usize) -> Vec<f32> {
    vec![1.0 / n as f32; n]
}

/// One-hot of length `n` with mass at index `i`.
fn one_hot(n: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    v[i] = 1.0;
    v
}

// ── Numerical correctness ────────────────────────────────────────────

#[test]
fn kl_self_is_zero() {
    let p = uniform(5);
    let d = kl(&p, &p).unwrap();
    assert!(d.abs() < 1e-5, "KL(p||p) must be ~0, got {d}");
}

#[test]
fn kl_disjoint_one_hot_is_infinity() {
    // p places mass where q has none -> +inf by definition.
    let p = one_hot(3, 0);
    let q = one_hot(3, 1);
    let d = kl(&p, &q).unwrap();
    assert!(d.is_infinite() && d > 0.0, "expected +inf, got {d}");
}

#[test]
fn kl_known_value_matches_hand_calc() {
    // p = [0.5, 0.5], q = [0.25, 0.75]
    // KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
    //    = 0.5*ln(2) + 0.5*ln(2/3)
    let p = vec![0.5f32, 0.5];
    let q = vec![0.25f32, 0.75];
    let expected = 0.5 * 2f32.ln() + 0.5 * (2.0f32 / 3.0).ln();
    let d = kl(&p, &q).unwrap();
    assert!(
        (d - expected).abs() < 1e-6,
        "KL mismatch: got {d}, expected {expected}"
    );
}

#[test]
fn js_symmetry() {
    let p = vec![0.1f32, 0.3, 0.6];
    let q = vec![0.5f32, 0.2, 0.3];
    let a = js(&p, &q).unwrap();
    let b = js(&q, &p).unwrap();
    assert!(
        (a - b).abs() < 1e-6,
        "JS must be symmetric: js(p,q)={a}, js(q,p)={b}"
    );
}

#[test]
fn js_disjoint_one_hot_reaches_ln2() {
    // Two disjoint one-hots max out JS at ln(2).
    let p = one_hot(4, 0);
    let q = one_hot(4, 1);
    let d = js(&p, &q).unwrap();
    let expected = 2f32.ln();
    assert!(
        (d - expected).abs() < 1e-6,
        "JS(disjoint one-hots) = ln(2), got {d}"
    );
}

#[test]
fn js_self_is_zero() {
    let p = vec![0.25f32, 0.25, 0.5];
    let d = js(&p, &p).unwrap();
    assert!(d.abs() < 1e-6, "JS(p,p) must be 0, got {d}");
}

#[test]
fn tvd_bounded() {
    // Disjoint one-hots: TVD = 0.5*(1+1) = 1.0.
    let p = one_hot(5, 0);
    let q = one_hot(5, 3);
    let d = tvd(&p, &q).unwrap();
    assert!(
        (d - 1.0).abs() < 1e-6,
        "TVD(disjoint one-hots) = 1.0, got {d}"
    );
}

#[test]
fn tvd_self_is_zero() {
    let p = uniform(6);
    let d = tvd(&p, &p).unwrap();
    assert!(d.abs() < 1e-6, "TVD(p,p) must be 0, got {d}");
}

#[test]
fn tvd_symmetric() {
    let p = vec![0.2f32, 0.3, 0.5];
    let q = vec![0.5f32, 0.4, 0.1];
    let a = tvd(&p, &q).unwrap();
    let b = tvd(&q, &p).unwrap();
    assert!((a - b).abs() < 1e-7, "TVD must be symmetric: {a} vs {b}");
}

#[test]
fn entropy_uniform_max() {
    for n in [4usize, 8] {
        let p = uniform(n);
        let h = entropy(&p).unwrap();
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-5,
            "entropy(uniform_{n}) = ln({n}) = {expected}, got {h}"
        );
    }
}

#[test]
fn entropy_one_hot_zero() {
    let p = one_hot(4, 2);
    let h = entropy(&p).unwrap();
    assert_eq!(h, 0.0, "entropy(one-hot) must be exactly 0, got {h}");
}

// ── Validation gates ─────────────────────────────────────────────────

#[test]
fn validate_empty() {
    let empty: Vec<f32> = vec![];
    assert_eq!(entropy(&empty), Err(MetricError::Empty));
    assert_eq!(kl(&empty, &empty), Err(MetricError::Empty));
    assert_eq!(js(&empty, &empty), Err(MetricError::Empty));
    assert_eq!(tvd(&empty, &empty), Err(MetricError::Empty));
}

#[test]
fn validate_negative() {
    // Sum happens to be 1.0 (0.5 + -0.2 + 0.7 = 1.0) so the negative
    // check must fire before the not-normalized check.
    let bad = vec![0.5f32, -0.2, 0.7];
    match entropy(&bad) {
        Err(MetricError::Negative { index, value }) => {
            assert_eq!(index, 1);
            assert!((value - -0.2).abs() < 1e-6, "unexpected value {value}");
        }
        other => panic!("expected Negative, got {other:?}"),
    }
}

#[test]
fn validate_not_normalized() {
    let bad = vec![0.3f32, 0.3, 0.3]; // sum = 0.9
    match entropy(&bad) {
        Err(MetricError::NotNormalized { sum, tol }) => {
            assert!((sum - 0.9).abs() < 1e-4, "unexpected sum {sum}");
            assert_eq!(tol, NORM_TOL);
        }
        other => panic!("expected NotNormalized, got {other:?}"),
    }
}

#[test]
fn validate_non_finite() {
    let bad = vec![0.5f32, f32::NAN, 0.5];
    match entropy(&bad) {
        Err(MetricError::NonFinite { index, value }) => {
            assert_eq!(index, 1);
            assert!(value.is_nan(), "expected NaN, got {value}");
        }
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

#[test]
fn kl_length_mismatch() {
    let p = vec![0.5f32, 0.5];
    let q = vec![0.25f32, 0.25, 0.5];
    match kl(&p, &q) {
        Err(MetricError::LengthMismatch { p: pl, q: ql }) => {
            assert_eq!(pl, 2);
            assert_eq!(ql, 3);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn tvd_length_mismatch() {
    let p = uniform(3);
    let q = uniform(4);
    assert!(matches!(
        tvd(&p, &q),
        Err(MetricError::LengthMismatch { p: 3, q: 4 })
    ));
}

#[test]
fn js_length_mismatch() {
    let p = uniform(2);
    let q = uniform(5);
    assert!(matches!(
        js(&p, &q),
        Err(MetricError::LengthMismatch { p: 2, q: 5 })
    ));
}
