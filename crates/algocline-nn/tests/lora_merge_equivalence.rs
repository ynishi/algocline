//! Integration test — LoRA merge equivalence.
//!
//! The subtask invariant for LoRA is that a wrapped forward
//!
//! ```text
//! y_lora = base(x) + scaling * lora_b(lora_a(x))
//! ```
//!
//! is element-wise close to a "merged" forward
//!
//! ```text
//! y_merged = Linear(base_weight + scaling * lora_b_weight @ lora_a_weight, base_bias)(x)
//! ```
//!
//! within 1e-4 across a small batch. That guarantee is what lets an
//! inference-only downstream consumer collapse the two low-rank
//! matrices back into a single Linear without behaviour drift.
//!
//! The test uses hand-set weights (rather than a `VarBuilder`
//! initialiser) so its outcome is fully deterministic regardless of
//! candle's RNG state.

use algocline_nn::arch::{max_abs_diff_f32, LoraConfig, LoraLinear};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

fn tensor2(vals: &[f32], rows: usize, cols: usize, dev: &Device) -> Tensor {
    Tensor::from_slice(vals, (rows, cols), dev).unwrap()
}

#[test]
fn wrapped_forward_matches_merged_linear_within_tolerance() {
    let device = Device::Cpu;

    // Base linear: 5 -> 4, deterministic values.
    let base_w = tensor2(
        &[
            0.10f32, 0.20, 0.30, 0.40, 0.50, //
            0.60, 0.70, 0.80, 0.90, 1.00, //
            1.10, 1.20, 1.30, 1.40, 1.50, //
            1.60, 1.70, 1.80, 1.90, 2.00, //
        ],
        4,
        5,
        &device,
    );
    let base_b = Tensor::from_slice(&[0.01f32, -0.02, 0.03, -0.04], (4,), &device).unwrap();
    let base = Linear::new(base_w, Some(base_b));

    // LoRA legs at rank = 2 (2 <= min(4, 5) so wrap must succeed).
    let a_w = tensor2(
        &[
            0.01f32, 0.02, 0.03, 0.04, 0.05, //
            0.06, 0.07, 0.08, 0.09, 0.10, //
        ],
        2,
        5,
        &device,
    );
    let b_w = tensor2(
        &[
            0.11f32, 0.12, //
            0.13, 0.14, //
            0.15, 0.16, //
            0.17, 0.18, //
        ],
        4,
        2,
        &device,
    );
    let lora_a = Linear::new(a_w, None);
    let lora_b = Linear::new(b_w, None);
    let scaling = LoraConfig::new(2, 4.0).scaling();

    let lora = LoraLinear::from_parts(base, lora_a, lora_b, scaling);

    // Build a plain merged Linear and compare its forward against the
    // LoRA-wrapped forward across a batch of inputs.
    let merged_w = lora.merged_weight().unwrap();
    let merged = Linear::new(merged_w, lora.base().bias().cloned());

    let xs = tensor2(
        &[
            0.5f32, -0.5, 1.5, -1.5, 2.5, //
            -2.5, 0.0, 3.0, -3.0, 0.25, //
            0.75, -0.75, 1.25, -1.25, 0.10, //
        ],
        3,
        5,
        &device,
    );

    let y_lora = lora.forward(&xs).unwrap();
    let y_merged = merged.forward(&xs).unwrap();
    assert_eq!(y_lora.dims(), y_merged.dims());
    let diff = max_abs_diff_f32(&y_lora, &y_merged).unwrap();
    assert!(
        diff < 1e-4,
        "LoRA / merged forwards diverged by {diff} (tolerance = 1e-4)"
    );
}

#[test]
fn merged_weight_shape_matches_base() {
    let device = Device::Cpu;
    let base_w = Tensor::zeros((6, 8), DType::F32, &device).unwrap();
    let base = Linear::new(base_w, None);
    let a_w = Tensor::zeros((3, 8), DType::F32, &device).unwrap();
    let b_w = Tensor::zeros((6, 3), DType::F32, &device).unwrap();
    let lora = LoraLinear::from_parts(
        base,
        Linear::new(a_w, None),
        Linear::new(b_w, None),
        LoraConfig::new(3, 6.0).scaling(),
    );
    let merged = lora.merged_weight().unwrap();
    assert_eq!(merged.dims(), &[6, 8]);
}

#[test]
fn zero_lora_deltas_leave_base_output_unchanged() {
    // If both LoRA matrices are zero, the wrap must be a no-op:
    // `y = base(x) + scaling * 0 = base(x)`.
    let device = Device::Cpu;
    let base_w = tensor2(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &device);
    let base_b = Tensor::from_slice(&[0.1f32, 0.2], (2,), &device).unwrap();
    let base = Linear::new(base_w, Some(base_b));
    let base_ref = Linear::new(base.weight().clone(), base.bias().cloned());

    let a_w = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
    let b_w = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
    let lora = LoraLinear::from_parts(
        base,
        Linear::new(a_w, None),
        Linear::new(b_w, None),
        LoraConfig::new(2, 4.0).scaling(),
    );

    let xs = tensor2(&[1.0f32, -1.0, 0.5, 0.5, 0.25, -0.25], 2, 3, &device);
    let y_lora = lora.forward(&xs).unwrap();
    let y_base = base_ref.forward(&xs).unwrap();
    let diff = max_abs_diff_f32(&y_lora, &y_base).unwrap();
    assert!(diff < 1e-6, "zero-delta LoRA changed base output by {diff}");
}
