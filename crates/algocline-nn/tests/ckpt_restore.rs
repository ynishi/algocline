//! Integration test for restoring a checkpoint into a trainable
//! `VarMap`.
//!
//! The interesting cases are the failures, and the reason they need
//! covering is narrower than it first looks. `candle_nn::VarMap::load`
//! is loud about most disagreements: it walks the *registered*
//! variables, so a variable the file lacks stops it with
//! `CannotFindTensor` and a shape that differs stops it with the
//! mismatch `Var::set` raises.
//!
//! What it does pass over in silence is an **empty map** — the loop
//! body never runs, `Ok(())` comes back, and a fully random
//! initialisation is reported as a resume — and tensors in the file
//! that the map does not register, which are dropped without a word.
//! Everything else here is about degree rather than silence: all the
//! disagreements instead of the first, a report naming them, and a
//! refusal that leaves the map exactly as it was rather than half
//! written. These tests pin each of those, and pin that the model built
//! before the restore is what ends up holding the restored weights.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{
    restore_into, restore_into_partial, run_full_ft, ApplyStage, CrossEntropyLoss, DatasetOpts,
    FullFtConfig, RestoreError, ScheduleKind, TokenizedDataset, TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tempfile::TempDir;

/// Build a tiny GPT-2 on CPU with a caller-chosen layer count and
/// vocabulary, together with the `VarMap` that owns its parameters.
///
/// Those two dials are what the mismatch tests vary: the vocabulary
/// changes a tensor's shape while leaving its name alone, and the layer
/// count changes the set of names while leaving every shared tensor
/// identical.
fn model(layers: usize, vocab: usize) -> (VarMap, Gpt2Model) {
    let cfg = Gpt2Config {
        layers,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: None,
    };
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let m = Gpt2Model::new(&cfg, vb).expect("build tiny gpt-2");
    (vm, m)
}

/// A `VarMap` whose two parameters carry names no checkpoint in this
/// file contains.
///
/// Stands in for the realistic version of that situation: a map holding
/// `base.*` pointed at a file whose tensors are `model.*`, or a LoRA
/// wrapper aimed at the wrong run's output.
fn foreign_varmap() -> VarMap {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    let _ = candle_nn::linear(4, 4, vb.pp("adapter")).expect("build linear");
    vm
}

/// Every parameter as raw bits, keyed by name.
///
/// Bits rather than floats so "bit-identical" in the round-trip
/// assertion is literal: two values that print the same but differ in
/// the last mantissa bit fail here, which is the point of comparing a
/// restore against its source at all.
fn weight_bits(vm: &VarMap) -> BTreeMap<String, Vec<u32>> {
    let data = vm.data().lock().expect("VarMap lock");
    data.iter()
        .map(|(name, var)| {
            let flat = var
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to f32 vec");
            (name.clone(), flat.into_iter().map(f32::to_bits).collect())
        })
        .collect()
}

fn names(vm: &VarMap) -> BTreeSet<String> {
    let data = vm.data().lock().expect("VarMap lock");
    data.keys().cloned().collect()
}

/// The model's own output on a fixed input, as raw bits.
///
/// Read through the `Gpt2Model` handle rather than the `VarMap`,
/// because that is the thing a resume has to change: `chess_bake`
/// builds its model and then restores into the map behind it, which
/// only works because `Var::set` writes through the shared storage
/// instead of swapping it. Comparing maps would never notice if that
/// stopped being true.
fn logits(m: &Gpt2Model, input: &Tensor) -> Vec<u32> {
    m.forward(input)
        .expect("forward")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to f32 vec")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// A repeating 8-token sequence, as in the Full FT integration test.
fn synthetic_corpus(rows: usize) -> Vec<Vec<u32>> {
    let base: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    std::iter::repeat_with(|| base.clone()).take(rows).collect()
}

/// Train `vm` for a few steps and return the path of the final bundle.
///
/// Training rather than saving a fresh initialisation matters for the
/// round trip: it is the optimiser's output that a resume has to
/// recover, and a random init would round-trip even through a restore
/// that quietly did nothing to a map initialised the same way.
fn train_and_save(vm: &VarMap, m: &Gpt2Model, dir: &Path) -> PathBuf {
    let mut dataset = TokenizedDataset::new(
        synthetic_corpus(200),
        DatasetOpts {
            batch_size: 1,
            ctx_len: 8,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );
    let ft_cfg = FullFtConfig {
        lr: 8e-3,
        batch_size: 1,
        grad_accum: 1,
        steps: 20,
        warmup: 2,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
    };
    run_full_ft(
        m,
        vm,
        &mut dataset,
        &ft_cfg,
        &CrossEntropyLoss::new(),
        dir,
        "restore",
        Arc::new(TrainingLease::new()),
        None,
    )
    .expect("training must complete");
    dir.join("restore.safetensors")
}

/// Copy a checkpoint with every tensor cast to `dtype`.
///
/// Cheaper and more direct than baking a model at another dtype: the
/// names and shapes stay exactly as they were, so the only thing left
/// for the restore to object to is the dtype itself.
fn recast(src: &Path, dst: &Path, dtype: DType) {
    let tensors = candle_core::safetensors::load(src, &Device::Cpu).expect("read the source");
    let cast: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, t)| (name, t.to_dtype(dtype).expect("cast")))
        .collect();
    candle_core::safetensors::save(&cast, dst).expect("write the recast copy");
}

#[test]
fn round_trip_restores_the_trained_weights_bit_for_bit() {
    let tmp = TempDir::new().unwrap();

    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());
    let saved = weight_bits(&trained_vm);

    // A second model of the same shape, holding its own random init.
    let (fresh_vm, _fresh) = model(2, 24);
    let before = weight_bits(&fresh_vm);
    assert_ne!(
        before, saved,
        "the two random initialisations must differ, otherwise the \
         round-trip assertion would pass on a restore that did nothing"
    );

    let report = restore_into(&fresh_vm, &ckpt).expect("same-shape restore must succeed");

    assert_eq!(weight_bits(&fresh_vm), saved);
    assert!(report.is_complete());
    assert_eq!(report.restored_count(), saved.len());
    assert_eq!(report.registered_count(), saved.len());
    assert_eq!(report.restored, saved.keys().cloned().collect::<Vec<_>>());
    assert!(report.absent_from_file.is_empty());
    assert!(report.unused_from_file.is_empty());
    assert_eq!(report.path, ckpt);
}

#[test]
fn the_model_built_before_the_restore_is_the_one_that_sees_it() {
    let tmp = TempDir::new().unwrap();

    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());
    let input = Tensor::from_vec(vec![1u32, 4, 9, 16], (1, 4), &Device::Cpu).expect("input tensor");
    let want = logits(&trained, &input);

    // Built before the restore and never rebuilt afterwards, which is
    // how `chess_bake` uses it: the model is constructed, then the
    // checkpoint goes into the map the model already borrows from.
    let (fresh_vm, fresh) = model(2, 24);
    let before = logits(&fresh, &input);
    assert_ne!(
        before, want,
        "two random initialisations must disagree on the same input, \
         otherwise this test would pass without a restore happening"
    );

    restore_into(&fresh_vm, &ckpt).expect("same-shape restore must succeed");

    assert_eq!(
        logits(&fresh, &input),
        want,
        "the model handle must compute with the restored weights; if it \
         still sees its initialisation, the restore reached the map but \
         not the tensors the model captured"
    );
}

#[test]
fn a_shape_that_disagrees_is_rejected_rather_than_half_loaded() {
    let tmp = TempDir::new().unwrap();

    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());

    // Same names, wider vocabulary: the embedding (and anything else
    // sized by the vocabulary) no longer describes the same parameter.
    let (wide_vm, _wide) = model(2, 32);
    let before = weight_bits(&wide_vm);

    let err = restore_into(&wide_vm, &ckpt).expect_err("a vocab change must not restore");
    match &err {
        RestoreError::Mismatch { mismatches, path } => {
            assert_eq!(path, &ckpt);
            assert!(
                !mismatches.is_empty(),
                "a Mismatch must name at least one tensor"
            );
            for m in mismatches {
                assert_ne!(
                    m.expected_shape, m.found_shape,
                    "{} was reported as a mismatch with identical shapes",
                    m.name
                );
            }
            // The message has to identify at least the first offending
            // tensor, otherwise the caller learns no more than "it
            // failed" and has to go looking for the difference by hand.
            let first = &mismatches[0].name;
            assert!(
                err.to_string().contains(first.as_str()),
                "the error message should name {first}: {err}"
            );
        }
        other => panic!("expected RestoreError::Mismatch, got {other:?}"),
    }

    assert_eq!(
        weight_bits(&wide_vm),
        before,
        "a rejected restore must leave every variable untouched"
    );
}

#[test]
fn a_dtype_that_disagrees_is_rejected_with_the_shapes_still_agreeing() {
    let tmp = TempDir::new().unwrap();

    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());
    let half = tmp.path().join("restore-f16.safetensors");
    recast(&ckpt, &half, DType::F16);

    // Every name and every shape lines up; only the dtype moved. A
    // check that compared shapes alone would wave this through and then
    // discover the problem half-way into writing the map.
    let (fresh_vm, _fresh) = model(2, 24);
    let before = weight_bits(&fresh_vm);

    let err = restore_into(&fresh_vm, &half)
        .expect_err("an F16 checkpoint must not load into an F32 map");
    match &err {
        RestoreError::Mismatch { mismatches, path } => {
            assert_eq!(path, &half);
            assert!(
                !mismatches.is_empty(),
                "a Mismatch must name at least one tensor"
            );
            for m in mismatches {
                assert_eq!(
                    m.expected_shape, m.found_shape,
                    "{} should differ in dtype only",
                    m.name
                );
                assert_eq!(m.expected_dtype, DType::F32, "{}", m.name);
                assert_eq!(m.found_dtype, DType::F16, "{}", m.name);
            }
        }
        other => panic!("expected RestoreError::Mismatch, got {other:?}"),
    }

    assert_eq!(
        weight_bits(&fresh_vm),
        before,
        "a rejected restore must leave every variable untouched"
    );
}

#[test]
fn a_variable_the_file_lacks_fails_strictly_and_passes_partially() {
    let tmp = TempDir::new().unwrap();

    let (small_vm, small) = model(2, 24);
    let ckpt = train_and_save(&small_vm, &small, tmp.path());
    let saved = weight_bits(&small_vm);

    // Three layers: layers 0 and 1 match the file, layer 2 has no
    // counterpart in it. This is the shape of a LoRA-wrapped map over a
    // base checkpoint — extra legs the file was never going to carry.
    let (deep_vm, _deep) = model(3, 24);
    let expected_absent: Vec<String> = names(&deep_vm)
        .difference(&names(&small_vm))
        .cloned()
        .collect();
    assert!(
        !expected_absent.is_empty(),
        "a third layer must register variables the two-layer file lacks"
    );
    let before_strict = weight_bits(&deep_vm);

    let err = restore_into(&deep_vm, &ckpt).expect_err("strict restore must refuse to skip vars");
    match &err {
        RestoreError::Incomplete {
            path,
            registered,
            absent,
        } => {
            assert_eq!(path, &ckpt);
            assert_eq!(*registered, names(&deep_vm).len());
            assert_eq!(absent, &expected_absent);
        }
        other => panic!("expected RestoreError::Incomplete, got {other:?}"),
    }
    assert!(
        err.to_string().contains("restore_into_partial"),
        "the strict error should point at the deliberate way through: {err}"
    );
    // Asserted here rather than after the partial call below: the
    // partial restore writes on purpose, so a comparison taken after it
    // would say nothing about whether `Incomplete` refused cleanly.
    assert_eq!(
        weight_bits(&deep_vm),
        before_strict,
        "an Incomplete verdict must leave every variable untouched, \
         including the ones the file does carry"
    );

    let untouched_before: BTreeMap<String, Vec<u32>> = before_strict
        .into_iter()
        .filter(|(name, _)| expected_absent.contains(name))
        .collect();

    let report = restore_into_partial(&deep_vm, &ckpt).expect("partial restore must succeed");

    assert!(!report.is_complete());
    assert_eq!(report.absent_from_file, expected_absent);
    assert_eq!(report.restored_count(), saved.len());
    assert_eq!(
        report.registered_count(),
        saved.len() + expected_absent.len()
    );
    assert!(report.unused_from_file.is_empty());

    // The shared variables took the file's values; the extra layer kept
    // the initialisation it was built with.
    let after = weight_bits(&deep_vm);
    for (name, bits) in &saved {
        assert_eq!(after.get(name), Some(bits), "{name} was not restored");
    }
    for (name, bits) in &untouched_before {
        assert_eq!(after.get(name), Some(bits), "{name} should not have moved");
    }
}

#[test]
fn the_partial_restore_is_still_strict_about_shapes() {
    let tmp = TempDir::new().unwrap();

    let (small_vm, small) = model(2, 24);
    let ckpt = train_and_save(&small_vm, &small, tmp.path());

    // Three layers *and* a wider vocabulary. The third layer is a
    // legitimate absence; the vocabulary is not a legitimate anything.
    // If the excuse for the former also covered the latter, "partial"
    // would be a way to opt out of the shape check.
    let (wide_deep_vm, _wide_deep) = model(3, 32);
    let before = weight_bits(&wide_deep_vm);

    let err = restore_into_partial(&wide_deep_vm, &ckpt)
        .expect_err("a partial restore must still reject a shape disagreement");
    match &err {
        RestoreError::Mismatch { mismatches, path } => {
            assert_eq!(path, &ckpt);
            assert!(!mismatches.is_empty());
        }
        other => panic!("expected RestoreError::Mismatch, got {other:?}"),
    }

    assert_eq!(
        weight_bits(&wide_deep_vm),
        before,
        "a rejected partial restore must leave every variable untouched"
    );
}

#[test]
fn a_map_sharing_no_name_with_the_file_is_refused_by_both_entry_points() {
    let tmp = TempDir::new().unwrap();
    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());
    let in_file_count = names(&trained_vm).len();

    let foreign = foreign_varmap();
    let registered_count = names(&foreign).len();
    let before = weight_bits(&foreign);

    // Strict: every registered name is absent, which `Incomplete`
    // already covers and reports in more detail.
    match restore_into(&foreign, &ckpt) {
        Err(RestoreError::Incomplete { absent, .. }) => {
            assert_eq!(absent.len(), registered_count)
        }
        other => panic!("expected RestoreError::Incomplete, got {other:?}"),
    }

    // Permissive: absence is excused one name at a time, so without a
    // separate check this would write nothing and report success.
    match restore_into_partial(&foreign, &ckpt) {
        Err(RestoreError::NothingRestored {
            path,
            registered,
            in_file,
        }) => {
            assert_eq!(path, ckpt);
            assert_eq!(registered, registered_count);
            assert_eq!(in_file, in_file_count);
        }
        other => panic!("expected RestoreError::NothingRestored, got {other:?}"),
    }

    assert_eq!(
        weight_bits(&foreign),
        before,
        "neither refusal may touch the map"
    );
}

#[test]
fn tensors_the_map_does_not_want_are_reported_not_rejected() {
    let tmp = TempDir::new().unwrap();

    // A three-layer checkpoint read by a two-layer model: every
    // registered variable is satisfied, and the third layer's tensors
    // are surplus. That is a legitimate partial read of a full model,
    // so it succeeds — but the report has to say what went unused.
    let (deep_vm, deep) = model(3, 24);
    let ckpt = train_and_save(&deep_vm, &deep, tmp.path());

    let (shallow_vm, _shallow) = model(2, 24);
    let expected_unused: Vec<String> = names(&deep_vm)
        .difference(&names(&shallow_vm))
        .cloned()
        .collect();

    let report = restore_into(&shallow_vm, &ckpt).expect("a superset checkpoint must restore");

    assert!(report.is_complete());
    assert_eq!(report.restored_count(), names(&shallow_vm).len());
    assert!(report.absent_from_file.is_empty());
    assert_eq!(report.unused_from_file, expected_unused);

    let saved = weight_bits(&deep_vm);
    for (name, bits) in weight_bits(&shallow_vm) {
        assert_eq!(saved.get(&name), Some(&bits), "{name} was not restored");
    }

    // The one-line summary is what a training log carries, so it has to
    // hold the numbers a reader would otherwise have to ask for: how
    // much was restored, out of how much, from where, and how much of
    // the file went by unused.
    let summary = report.summary();
    for fragment in [
        &format!(
            "{} of {}",
            report.restored_count(),
            report.registered_count()
        ),
        &format!("{} tensors in the file unused", expected_unused.len()),
        &format!(
            "{} left at their initial value",
            report.absent_from_file.len()
        ),
    ] {
        assert!(
            summary.contains(fragment.as_str()),
            "summary should contain {fragment:?}: {summary}"
        );
    }
    assert!(
        summary.contains("restore.safetensors"),
        "summary should name the checkpoint: {summary}"
    );
}

#[test]
fn an_empty_varmap_is_refused_instead_of_reporting_a_flawless_nothing() {
    let tmp = TempDir::new().unwrap();
    let (trained_vm, trained) = model(2, 24);
    let ckpt = train_and_save(&trained_vm, &trained, tmp.path());

    // Zero registered variables would otherwise restore zero of zero
    // and report success — the usual cause being a model built against
    // a different map than the one handed over here.
    let empty = VarMap::new();
    match restore_into(&empty, &ckpt) {
        Err(RestoreError::NoRegisteredVars { path }) => assert_eq!(path, ckpt),
        other => panic!("expected RestoreError::NoRegisteredVars, got {other:?}"),
    }
}

#[test]
fn a_mid_apply_failure_says_what_it_may_have_left_behind() {
    // Constructed rather than provoked: pass one verifies every shape
    // and dtype, so reaching `Apply` through the public entry points
    // means defeating the check that exists to prevent it. What the
    // variant has to get right is the wording — a caller reads it to
    // decide whether the map in front of them is usable — and that is
    // reachable directly.
    let path = PathBuf::from("/tmp/half.safetensors");
    let of = |stage: ApplyStage, written: Vec<String>| {
        RestoreError::Apply {
            path: path.clone(),
            stage,
            name: "h.0.attn.c_attn.weight".into(),
            written,
            message: "storage fault".into(),
        }
        .to_string()
    };

    // Reading touches no variable, so this one may promise the map is
    // intact.
    let load_first = of(ApplyStage::Load, Vec::new());
    assert!(
        load_first.contains("still holds exactly what it did"),
        "a failed read of the first tensor leaves the map alone: {load_first}"
    );
    assert!(
        !load_first.contains("rebuilt"),
        "nothing was written, so nothing needs rebuilding: {load_first}"
    );

    // Writing had already begun, and candle reports no boundary
    // between "refused" and "copied half of it", so this one may not.
    let set_first = of(ApplyStage::Set, Vec::new());
    assert!(
        set_first.contains("may hold part of the checkpoint"),
        "a failed write of the first tensor cannot claim the map is intact: {set_first}"
    );
    assert!(
        set_first.contains("rebuilt"),
        "the caller has to be told to rebuild: {set_first}"
    );

    // With variables behind it, both stages say the same thing.
    for stage in [ApplyStage::Load, ApplyStage::Set] {
        let later = of(stage, vec!["wte.weight".into(), "wpe.weight".into()]);
        assert!(
            later.contains("2 variable(s) had already been written"),
            "the count belongs in the message: {later}"
        );
        assert!(
            later.contains("wte.weight") && later.contains("wpe.weight"),
            "so do the names: {later}"
        );
        assert!(later.contains("rebuilt"), "and the verdict: {later}");
    }
}

#[test]
fn a_missing_checkpoint_names_the_path_it_could_not_read() {
    let tmp = TempDir::new().unwrap();
    let (vm, _m) = model(2, 24);
    let missing = tmp.path().join("not-written-yet.safetensors");

    match restore_into(&vm, &missing) {
        Err(RestoreError::Open { path, .. }) => assert_eq!(path, missing),
        other => panic!("expected RestoreError::Open, got {other:?}"),
    }
}

#[test]
fn a_file_that_is_not_a_checkpoint_is_an_open_error() {
    let tmp = TempDir::new().unwrap();
    let (vm, _m) = model(2, 24);
    // Present and readable, but not a safetensors container: an
    // operator pointing a resume at the training log rather than at the
    // weights. The path exists, so nothing before the header parse
    // notices.
    let junk = tmp.path().join("train.log");
    std::fs::write(&junk, b"[bake] training 20 steps at batch 1\n").expect("write the decoy");

    match restore_into(&vm, &junk) {
        Err(RestoreError::Open { path, .. }) => assert_eq!(path, junk),
        other => panic!("expected RestoreError::Open, got {other:?}"),
    }
}
