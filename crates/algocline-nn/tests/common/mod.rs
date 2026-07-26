//! Shared helper for the gradient-coverage gates
//! (`gpt2_grad_coverage.rs` / `moe_grad_coverage.rs` /
//! `tinyllama_grad_coverage.rs` / `custom_grad_coverage.rs`).
//!
//! The gates share one mechanical core: after a single backward pass,
//! every Var registered in the model's VarMap must carry a non-zero,
//! finite gradient. Keeping the walk here means a new architecture
//! axis gets coverage by writing "build a config, run one backward,
//! call the helper with the pinned Var count" — instead of re-deriving
//! the loop (and its NaN guard) per test file.

use candle_core::backprop::GradStore;
use candle_nn::VarMap;

/// Assert that every Var in `vm` received a non-zero, finite gradient.
///
/// `expected_vars` pins the inventory so the per-Var loop cannot go
/// vacuous when the arch registration changes shape — a drifted count
/// fails loudly and the call site updates the pin deliberately.
pub fn assert_full_grad_coverage(vm: &VarMap, grads: &GradStore, expected_vars: usize) {
    let data = vm.data().lock().unwrap();
    assert_eq!(
        data.len(),
        expected_vars,
        "VarMap inventory drifted; update the pinned count at the call site"
    );

    let mut missing: Vec<String> = Vec::new();
    let mut zero: Vec<String> = Vec::new();
    for (name, var) in data.iter() {
        match grads.get(var.as_tensor()) {
            None => missing.push(name.clone()),
            Some(g) => {
                let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
                // The explicit NaN arm keeps a NaN gradient from
                // slipping through the `<=` comparison.
                if mag.is_nan() || mag <= 0.0 {
                    zero.push(format!("{name} (sum|g|={mag})"));
                }
            }
        }
    }
    missing.sort();
    zero.sort();
    assert!(
        missing.is_empty() && zero.is_empty(),
        "autograd coverage hole — the loss can still descend while these \
         parameters never learn.\n  missing from GradStore: {missing:?}\n  \
         zero/NaN gradient: {zero:?}"
    );
}
