//! What a per-position bake does, end to end, on a tiny PGN.
//!
//! `chess_bake` is an example binary, so a test cannot call it. What it
//! can do is walk the same sequence through the same public API — read
//! PGN, band the rows, resolve each row's band against the model shape,
//! train, write the sidecar — which is what this does. Everything
//! specific to the example (argument parsing, the environment dials, the
//! validation curve) is out; everything the conditioning depends on is
//! in, and is reached the same way the example reaches it.
//!
//! Two things are pinned here that nothing else holds:
//!
//! - **Verification plan 02, Phase 0-7.** Two bakes under the two
//!   conditioning conventions land on the same checkpoint name, and the
//!   cross-encoding read is refused. The shape layer's half of that is
//!   already covered by
//!   `chess::a_second_bake_under_the_other_encoding_is_not_readable_as_the_first`;
//!   this is the version where the checkpoints are real and were
//!   actually trained.
//! - **A per-position run is resumable.** The convention adds one
//!   tensor, `cond_wte.weight`, and a resume is the one path that can
//!   quietly leave a tensor at its random initialisation.

use std::sync::Arc;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::corpus::{
    build_rows, ConditionBand, ConditionSpec, CorpusOptions, TeacherRow,
};
use algocline_nn::chess::pgn::PgnReader;
use algocline_nn::chess::train::{cond_table, row_conditions};
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::{CondEncoding, ModelShape, ShapeError, ShapeKind};
use algocline_nn::train::{
    restore_into, run_conditioned_ft, run_full_ft, Checkpoint, CrossEntropyLoss, DatasetOpts,
    FullFtConfig, ScheduleKind, TeacherCardDataset, TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tempfile::TempDir;

const CTX: usize = 16;
const BATCH: usize = 2;

fn bands() -> Vec<ConditionBand> {
    vec![
        ConditionBand::rating(1100, 1299, "<elo:1100-1299>"),
        ConditionBand::rating(1900, 2099, "<elo:1900-2099>"),
    ]
}

/// Six games, alternating between the two bands, each four plies long.
///
/// Small enough to keep the test in the millisecond range and real
/// enough to go through the PGN reader and the board replay rather than
/// through invented ids.
fn tiny_pgn() -> String {
    let games = [
        (1150, "1. e4 e5 2. Nf3 Nc6 1-0"),
        (2000, "1. d4 d5 2. c4 e6 0-1"),
        (1200, "1. c4 c5 2. Nc3 Nc6 1-0"),
        (1950, "1. e4 c5 2. Nf3 d6 0-1"),
        (1180, "1. d4 Nf6 2. c4 e6 1-0"),
        (2050, "1. Nf3 d5 2. g3 Nf6 1-0"),
    ];
    let mut out = String::new();
    for (elo, moves) in games {
        out.push_str(&format!("[WhiteElo \"{elo}\"]\n"));
        out.push_str(&format!("[BlackElo \"{elo}\"]\n"));
        out.push_str("[Termination \"Normal\"]\n");
        out.push_str(&format!("\n{moves}\n\n"));
    }
    out
}

/// The shape a bake of this corpus runs at, under `encoding`.
fn shape_for(vocab: &MoveVocab, encoding: CondEncoding) -> ModelShape {
    let mut shape = ModelShape::compact(vocab.model_vocab_size(), bands());
    shape.layers = 1;
    shape.heads = 2;
    shape.dim = 16;
    shape.ctx = CTX;
    shape.encoding = encoding;
    shape
}

/// The spec every row's band is an ordinal into, and the one
/// `cond_table` resolves against. One value, named in both places.
fn spec() -> ConditionSpec {
    ConditionSpec {
        key: "WhiteElo".to_string(),
        bands: bands(),
    }
}

fn corpus_rows(vocab: &MoveVocab) -> Vec<TeacherRow> {
    let opts = CorpusOptions {
        max_rows: 100,
        max_len: Some(CTX),
        conditions: vec![spec()],
        ..Default::default()
    };
    let mut reader = PgnReader::new(std::io::Cursor::new(tiny_pgn()));
    build_rows(&mut reader, vocab, &opts)
        .expect("the corpus builds")
        .into_teacher_rows()
        .expect("the corpus's row lists agree")
}

fn ft_cfg(steps: usize) -> FullFtConfig {
    FullFtConfig {
        lr: 1e-3,
        batch_size: BATCH,
        grad_accum: 1,
        steps,
        warmup: 1,
        schedule: ScheduleKind::Constant,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
    }
}

/// One bake: build the model, band the rows, train, write the sidecar.
///
/// The `init_from` argument is the example's `CHESS_INIT_FROM`, and is
/// applied exactly where the example applies it — after the model
/// exists and before training, through the strict restore.
fn bake(
    shape: &ModelShape,
    dir: &std::path::Path,
    prefix: &str,
    steps: usize,
    init_from: Option<&std::path::Path>,
) -> (Checkpoint, std::path::PathBuf) {
    let vocab = MoveVocab::new(&shape.band_tokens()).expect("vocabulary");
    let rows = corpus_rows(&vocab);
    assert!(
        rows.len() >= steps * BATCH + BATCH,
        "the tiny corpus has to cover {steps} step(s) at batch {BATCH}, and holds {}",
        rows.len()
    );

    let conditioned = shape.encoding == CondEncoding::EveryPosition;
    let conds = conditioned.then(|| {
        let table = cond_table(shape, &spec()).expect("the corpus bands are the model's");
        row_conditions(&rows, std::slice::from_ref(&table)).expect("every row carries a band")
    });

    let ds_opts = DatasetOpts {
        batch_size: BATCH,
        ctx_len: shape.ctx,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    };
    let ds = TeacherCardDataset::from_rows(TeacherRow::into_pairs(rows), ds_opts)
        .expect("rows are well formed");
    let mut ds = match conds {
        Some(conds) => ds.with_conditions(conds).expect("one per row"),
        None => ds,
    };

    let cfg = shape.config(Device::Cpu, DType::F32);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build the model");

    if let Some(path) = init_from {
        let report = restore_into(&vm, path).expect("the resume restores every variable");
        assert!(
            report.is_complete(),
            "a resume that left a variable at its initialisation is not a resume: {}",
            report.summary()
        );
        assert!(
            report.restored.iter().any(|n| n == "cond_wte.weight"),
            "the conditioning table is the tensor this convention adds, and it has to come \
             back: {}",
            report.summary()
        );
    }

    let loss = CrossEntropyLoss::new();
    let lease = Arc::new(TrainingLease::new());
    let ckpt = if conditioned {
        run_conditioned_ft(
            &model,
            &vm,
            &mut ds,
            &ft_cfg(steps),
            &loss,
            dir,
            prefix,
            lease,
            None,
        )
    } else {
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            &ft_cfg(steps),
            &loss,
            dir,
            prefix,
            lease,
            None,
        )
    }
    .expect("the bake completes");

    let ckpt_path = dir.join(format!("{prefix}.safetensors"));
    shape.save(&ckpt_path).expect("the sidecar is written");
    (ckpt, ckpt_path)
}

/// The step the refusal in `chess_bake` used to stand in for: a
/// per-position bake runs, lands a checkpoint, and the sidecar beside it
/// says so.
#[test]
fn a_per_position_bake_lands_a_checkpoint_that_says_every_position() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, CondEncoding::EveryPosition);

    let (ckpt, ckpt_path) = bake(&shape, tmp.path(), "perpos", 1, None);
    assert!(ckpt_path.is_file(), "the checkpoint has to be on disk");
    assert!(ckpt.train_loss.is_finite(), "got {}", ckpt.train_loss);

    // The sidecar is at the per-position name, which is not the one an
    // older build looks in.
    assert_eq!(
        ModelShape::path_for_kind(
            &ckpt_path,
            ShapeKind {
                encoding: CondEncoding::EveryPosition,
                legal_input: false,
            }
        )
        .file_name()
        .unwrap(),
        "perpos.shape2.json"
    );
    assert!(!ModelShape::path_for(&ckpt_path).exists());

    let back = ModelShape::load_as(&ckpt_path, CondEncoding::EveryPosition)
        .expect("a per-position reader reads it");
    assert_eq!(back.encoding, CondEncoding::EveryPosition);
    assert_eq!(back.bands, bands());

    // And the readers — every one of which is set up for the prefix
    // convention — refuse it.
    assert!(
        matches!(
            ModelShape::load_as(&ckpt_path, CondEncoding::Prefix),
            Err(ShapeError::EncodingMismatch { .. })
        ),
        "a prefix reader must refuse a per-position checkpoint"
    );
}

/// Verification plan 02, Phase 0-7, at the level of the bake: two runs
/// under the two conventions, the same checkpoint name, and a
/// cross-encoding read that is refused rather than answered.
///
/// The second bake's sweep is what makes it so. While both sidecars
/// existed, a prefix reader found the stale prefix one, agreed with
/// itself, and scored per-position weights.
#[test]
fn two_bakes_under_the_two_encodings_leave_one_readable_checkpoint() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");

    // First the prefix arm, at the name every reader already knows.
    let prefix_shape = shape_for(&vocab, CondEncoding::Prefix);
    let (_, ckpt_path) = bake(&prefix_shape, tmp.path(), "arm", 1, None);
    assert!(ModelShape::path_for(&ckpt_path).is_file());
    assert_eq!(
        ModelShape::load_as(&ckpt_path, CondEncoding::Prefix)
            .expect("a prefix reader reads a prefix bake")
            .encoding,
        CondEncoding::Prefix
    );

    // Then the per-position arm onto the same name. The weights are
    // replaced and so is the description of them.
    let perpos_shape = shape_for(&vocab, CondEncoding::EveryPosition);
    let (_, same_path) = bake(&perpos_shape, tmp.path(), "arm", 1, None);
    assert_eq!(same_path, ckpt_path);
    assert!(
        !ModelShape::path_for(&ckpt_path).exists(),
        "the prefix sidecar describes weights that are no longer there"
    );

    // The read that would otherwise have produced a full set of
    // plausible numbers.
    let err = ModelShape::load_as(&ckpt_path, CondEncoding::Prefix).unwrap_err();
    assert!(
        matches!(err, ShapeError::EncodingMismatch { .. }),
        "{err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("every-position"), "{msg}");
    assert!(msg.contains("prefix"), "{msg}");
}

/// A per-position run resumes from its own checkpoint, conditioning
/// table included.
///
/// This is the one path where the extra tensor could go missing without
/// a word: a permissive load would leave `cond_wte.weight` at its random
/// initialisation and report a resume. The strict restore inside `bake`
/// asserts it came back.
#[test]
fn a_per_position_run_resumes_from_its_own_checkpoint() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, CondEncoding::EveryPosition);

    let (_, first) = bake(&shape, tmp.path(), "run1", 1, None);
    let (resumed, second) = bake(&shape, tmp.path(), "run2", 1, Some(&first));

    assert!(second.is_file());
    assert!(resumed.train_loss.is_finite(), "got {}", resumed.train_loss);
    assert_eq!(
        ModelShape::load_as(&second, CondEncoding::EveryPosition)
            .expect("the resumed run's sidecar reads")
            .encoding,
        CondEncoding::EveryPosition,
        "a resumed run writes the convention it ran under"
    );
}

/// And a corpus banded for one model cannot be trained against another.
///
/// The ordinals would line up — both lists are `0..n` — so the refusal
/// has to come from the tokens.
#[test]
fn a_corpus_banded_for_another_model_is_refused() {
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let mut other = shape_for(&vocab, CondEncoding::EveryPosition);
    other.bands = vec![
        ConditionBand::rating(1500, 1699, "<elo:1500-1699>"),
        ConditionBand::rating(1900, 2099, "<elo:1900-2099>"),
    ];
    let err = cond_table(&other, &spec()).unwrap_err();
    assert!(
        err.to_string().contains("<elo:1100-1299>"),
        "the message has to name the band that has no row: {err}"
    );
}
