//! What a bake that hands the model its legality does, end to end, on
//! a tiny PGN.
//!
//! `chess_bake` is an example binary, so a test cannot call it. What it
//! can do is walk the same sequence through the same public API — read
//! PGN, build the legal-masked dataset, train through the entry point
//! that takes the sets as input, write the sidecar — which is what this
//! does. Everything specific to the example (argument parsing, the
//! environment dials, the validation curve) is out.
//!
//! Three things are pinned here that nothing else holds:
//!
//! - **The sets reach the model from a real corpus.** The unit tests
//!   feed the entry point hand-written batches; this feeds it the ones
//!   `LegalMaskedDataset` produces from replayed games, where the legal
//!   set at each position comes from the board rather than from a
//!   fixture.
//! - **A legality bake is resumable.** The convention adds one tensor,
//!   `legal_wte.weight`, and a resume is the one path that can quietly
//!   leave a tensor at its random initialisation.
//! - **The readers that cannot supply the input refuse it.**
//!   `chess_eval` and `chess_match` generate legal moves for the
//!   position in front of them, not for every position of a row, so a
//!   checkpoint of this kind is turned away rather than scored with the
//!   channel missing. That refusal is asserted against
//!   `open_reader_shape_as`, the entry point those two open checkpoints
//!   through, rather than against the predicate inside it — and against
//!   `open_reader_shape`, which is that gate on its own. An example
//!   rewritten to fetch its shape from `ModelShape::load_any` would
//!   compile and pass; that route is open, and is what `chess_bake`'s
//!   resume and the two readers below take. So this covers a way in
//!   rather than every way in.
//! - **The readers that can supply it line up with the training
//!   convention.** `chess_cond` and `chess_play` recover the sets by
//!   replaying the game, and what they recover has to sit at the same
//!   positions `LegalMaskedDataset` records for the same game. That is
//!   what the comparison pins — which positions carry a set, which are
//!   empty, where each list ends — and not the enumeration inside a
//!   position, which both sides reach by the same three calls in the
//!   same order.

use std::sync::Arc;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::corpus::{
    build_rows, ConditionBand, ConditionSpec, CorpusOptions, LegalMaskedDataset, TeacherRow,
};
use algocline_nn::chess::pgn::{move_from_uci_standard, uci_standard, PgnReader};
use algocline_nn::chess::vocab::{legal_ids, MoveVocab};
use algocline_nn::chess::window::{play_row, COND_PREFIX_LEN};
use algocline_nn::chess::{
    open_reader_shape, open_reader_shape_as, CondEncoding, ModelShape, ShapeError, ShapeKind,
};
use algocline_nn::train::{
    legal_input_sets, restore_into, run_full_ft, Checkpoint, CrossEntropyLoss, Dataset,
    DatasetOpts, FullFtConfig, ScheduleKind, TrainError, TrainingLease,
};
use algocline_nn::train::{run_legal_ft, TeacherCardDataset};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use cozy_chess::Board;
use tempfile::TempDir;

const CTX: usize = 16;
const BATCH: usize = 2;

fn bands() -> Vec<ConditionBand> {
    vec![ConditionBand::rating(1100, 2099, "<elo:1100-2099>")]
}

/// Six games, each four plies long, all inside the one band.
fn tiny_pgn() -> String {
    let games = [
        "1. e4 e5 2. Nf3 Nc6 1-0",
        "1. d4 d5 2. c4 e6 0-1",
        "1. c4 c5 2. Nc3 Nc6 1-0",
        "1. e4 c5 2. Nf3 d6 0-1",
        "1. d4 Nf6 2. c4 e6 1-0",
        "1. Nf3 d5 2. g3 Nf6 1-0",
    ];
    let mut out = String::new();
    for moves in games {
        out.push_str("[WhiteElo \"1500\"]\n[BlackElo \"1500\"]\n[Termination \"Normal\"]\n");
        out.push_str(&format!("\n{moves}\n\n"));
    }
    out
}

fn spec() -> ConditionSpec {
    ConditionSpec {
        key: "WhiteElo".to_string(),
        bands: bands(),
    }
}

/// The shape a bake of this corpus runs at.
fn shape_for(vocab: &MoveVocab, legal_input: bool) -> ModelShape {
    let mut shape = ModelShape::compact(vocab.model_vocab_size(), bands());
    shape.layers = 1;
    shape.heads = 2;
    shape.dim = 16;
    shape.ctx = CTX;
    shape.legal_input = legal_input;
    shape
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

fn ds_opts(ctx: usize) -> DatasetOpts {
    DatasetOpts {
        batch_size: BATCH,
        ctx_len: ctx,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    }
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

/// One bake: build the model, wrap the rows in the legal-masked
/// dataset, train through the entry point that hands the sets to the
/// model, write the sidecar.
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

    // `[BOS, band, moves..]` — two tokens before the first move, which
    // is what tells the dataset where the replay starts.
    let mut ds = LegalMaskedDataset::new(
        TeacherRow::into_pairs(rows),
        vocab.clone(),
        2,
        ds_opts(shape.ctx),
    );

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
            report.restored.iter().any(|n| n == "legal_wte.weight"),
            "the legality table is the tensor this convention adds, and it has to come \
             back: {}",
            report.summary()
        );
    }

    let ckpt = run_legal_ft(
        &model,
        &vm,
        &mut ds,
        &ft_cfg(steps),
        &CrossEntropyLoss::new(),
        dir,
        prefix,
        Arc::new(TrainingLease::new()),
        None,
    )
    .expect("the bake completes");

    let ckpt_path = dir.join(format!("{prefix}.safetensors"));
    shape.save(&ckpt_path).expect("the sidecar is written");
    (ckpt, ckpt_path)
}

/// A legality bake runs on rows a board replay produced, lands a
/// checkpoint, and the sidecar beside it says what it was trained
/// under — at a name an older build does not look in.
#[test]
fn a_legality_bake_lands_a_checkpoint_that_says_so() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, true);

    let (ckpt, ckpt_path) = bake(&shape, tmp.path(), "legal", 1, None);
    assert!(ckpt_path.is_file(), "the checkpoint has to be on disk");
    assert!(ckpt.train_loss.is_finite(), "got {}", ckpt.train_loss);

    assert_eq!(
        ModelShape::path_for_kind(
            &ckpt_path,
            ShapeKind {
                encoding: CondEncoding::Prefix,
                legal_input: true,
            }
        )
        .file_name()
        .unwrap(),
        "legal.shape-legal.json"
    );
    assert!(
        !ModelShape::path_for(&ckpt_path).exists(),
        "a build that predates this axis reads the bare name, and must find nothing"
    );

    let back = ModelShape::load_any(&ckpt_path).expect("the sidecar reads");
    assert!(back.legal_input);
    assert_eq!(back.encoding, CondEncoding::Prefix);

    // And the readers, none of which supplies a legality input, turn it
    // away rather than scoring the model without one.
    let err = back.require_no_legal_input(&ckpt_path).unwrap_err();
    assert!(
        matches!(err, ShapeError::LegalInputUnsupported { .. }),
        "{err:?}"
    );
}

/// The refusal reaches the readers that cannot supply the input.
///
/// The predicate is asserted above; what this asserts is the way in.
/// `chess_eval` and `chess_match` open a checkpoint with
/// `open_reader_shape_as`, and neither it nor the `open_reader_shape`
/// it is built on can be made to hand back a legality shape. While the
/// refusal was a second line at each call site, no test reached any of
/// those lines: examples are compiled by `cargo test` and not run, and
/// what was asserted was `require_no_legal_input` on its own.
///
/// This is a narrowing rather than a proof about all future readers.
/// `ModelShape::load_any` is `pub` and is what `chess_bake`'s resume and
/// the two readers that do supply the sets take, so a reader can be
/// written against it and be checked by nothing here. What it cannot do
/// is come through this function and skip the gate — and what it meets
/// instead, if it builds its rows with `BandBatch`, is a constructor
/// that will not take a legality shape without the sets.
#[test]
fn the_reader_entry_points_refuse_a_legality_checkpoint() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");

    let (_, legal) = bake(&shape_for(&vocab, true), tmp.path(), "legal", 1, None);
    for err in [
        open_reader_shape(&legal).unwrap_err(),
        open_reader_shape_as(&legal, CondEncoding::Prefix).unwrap_err(),
    ] {
        assert!(
            matches!(err, ShapeError::LegalInputUnsupported { .. }),
            "{err:?}"
        );
    }

    // And an ordinary checkpoint of the same corpus passes both, so the
    // refusal above is about the axis rather than about anything else
    // this bake writes.
    let ordinary = tmp.path().join("plain.safetensors");
    let plain_shape = shape_for(&vocab, false);
    {
        let cfg = plain_shape.config(Device::Cpu, DType::F32);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let _ = Gpt2Model::new(&cfg, vb).expect("build the model");
        vm.save(&ordinary).expect("weights on disk");
    }
    plain_shape.save(&ordinary).expect("the sidecar is written");
    assert!(
        !open_reader_shape(&ordinary)
            .expect("an ordinary checkpoint is readable")
            .legal_input
    );
    open_reader_shape_as(&ordinary, CondEncoding::Prefix)
        .expect("and readable by a prefix-only reader");
}

/// Ten plies of a real game, as the ids of its moves and the sets each
/// of them was drawn from — what a reader holds after replaying a game
/// to the position it is scoring.
///
/// Both sides walk a different pawn forward one square, so every ply
/// reaches a position no earlier ply reached and the legal sets differ
/// from one to the next. A knight shuffle was the first draft and is
/// wrong here for a reason worth recording: it returns to the same
/// position every four plies, so `at_ply[4]` and `at_ply[0]` are the
/// same set and the windowed test below could not tell a correct
/// mapping from one that ignores the window. The test caught it.
fn a_pawn_walk(vocab: &MoveVocab) -> (Vec<u32>, Vec<Vec<u32>>) {
    let game = [
        "a2a3", "a7a6", "b2b3", "b7b6", "c2c3", "c7c6", "d2d3", "d7d6", "e2e3", "e7e6",
    ];
    let mut board = Board::default();
    let mut ids = Vec::with_capacity(game.len());
    let mut at_ply = Vec::with_capacity(game.len() + 1);
    for (ply, uci) in game.iter().enumerate() {
        at_ply.push(legal_ids(&board, vocab));
        let mv = move_from_uci_standard(&board, uci)
            .unwrap_or_else(|| panic!("{uci} is not legal at ply {ply}"));
        let played = uci_standard(&board, mv);
        board.play_unchecked(mv);
        ids.push(vocab.id_of(&played).expect("move in the vocabulary"));
    }
    at_ply.push(legal_ids(&board, vocab));
    (ids, at_ply)
}

/// The sets `LegalMaskedDataset` records for one row, unpadded.
///
/// `ctx_len` is the row's own length, so nothing is padded or cut and
/// the list is exactly one entry per token — which is what makes it
/// comparable, position for position, against what a reader recovers.
fn dataset_sets(row: Vec<u32>, vocab: &MoveVocab) -> Vec<Vec<u32>> {
    let len = row.len();
    let mask = vec![1.0f32; len];
    let mut ds = LegalMaskedDataset::new(
        vec![(row, mask)],
        vocab.clone(),
        2,
        DatasetOpts {
            batch_size: 1,
            ctx_len: len,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );
    let batch = ds
        .next_batch()
        .expect("the dataset yields")
        .expect("a first batch");
    let mut allowed = batch
        .allowed_ids
        .expect("the legal-masked dataset carries sets");
    assert_eq!(allowed.len(), 1, "one row in, one row out");
    allowed.remove(0)
}

/// What a reader recovers sits where the training path recorded it.
///
/// The two walks are separate: `LegalMaskedDataset` replays the row it
/// was handed and also has to find the played move in order to advance
/// the board, while a reader replays the game as it walks it and keeps
/// `legal_ids` at each ply. On a row that fits whole in the context the
/// two lists can be compared directly — the reader's is the dataset's
/// plus one entry for the position past the end of the row, which is the
/// move the model is being asked for and which training sees as a target
/// rather than as a position.
///
/// So this is the check against the convention rather than against a
/// description of it: `allowed_ids[p]` is the set the token at `p` was
/// drawn from, on both sides; the shift that turns it into a model input
/// is applied by `legal_input_sets` on one side and by `BandBatch` on
/// the other, from the same offset.
///
/// What it does not check is the contents of a set. Both sides call
/// `generate_moves`, then `uci_standard`, then `id_of`, in that order,
/// so a defect in the generator's order or in either mapping is
/// duplicated identically and this still passes. The claim here is
/// alignment.
#[test]
fn the_readers_sets_are_the_ones_the_dataset_records() {
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let band = vocab.id_of(&bands()[0].token).expect("the band");

    // Ten plies behind `[BOS, band]` is twelve tokens, inside a context
    // of sixteen, so nothing here is windowed.
    let (moves, at_ply) = a_pawn_walk(&vocab);
    let window = play_row(Some(band), &moves, CTX).expect("a row");
    assert_eq!(window.len(), moves.len() + COND_PREFIX_LEN, "not windowed");

    let mine = window
        .legal_sets(&at_ply, moves.len())
        .expect("the history covers it");
    let theirs = dataset_sets(window.tokens().to_vec(), &vocab);

    assert_eq!(
        mine.len(),
        theirs.len() + 1,
        "the reader also answers at the position past the row"
    );
    assert!(
        theirs.iter().skip(COND_PREFIX_LEN).all(|s| !s.is_empty()),
        "every move of this row should carry the set it was drawn from, or the comparison \
         below is between two lists of empties"
    );
    for (p, (a, b)) in mine.iter().zip(&theirs).enumerate() {
        assert_eq!(a, b, "position {p}");
    }
    assert_eq!(
        mine[window.len()],
        at_ply[moves.len()],
        "and the extra entry is the position on the board"
    );
}

/// The same on a row the context window cut, which the test above
/// cannot distinguish from an identity mapping.
///
/// Ten moves behind a two-token prefix, in a context of eight, keeps six
/// of them: the row's first move is the game's fifth. A reader that
/// indexed the history by row position would hand that position the
/// opening — a set of twenty perfectly ordinary chess moves, at a
/// position where they are four plies stale.
///
/// The dataset is not the comparison here, because it never holds a row
/// of this shape. `LegalMaskedDataset` pads or truncates a row's **tail**
/// to the context — `row.chain(pad).take(ctx)` — so its rows always begin
/// at the game's first move. Dropping the front is what a reader does and
/// what training never does, so this regime exists on the reading side
/// only.
#[test]
fn a_windowed_row_takes_its_sets_from_where_the_window_starts() {
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let band = vocab.id_of(&bands()[0].token).expect("the band");

    let (moves, at_ply) = a_pawn_walk(&vocab);
    let narrow = 8;
    let window = play_row(Some(band), &moves, narrow).expect("a row");
    assert_eq!(window.len(), narrow, "this row has to be windowed");

    let kept = narrow - COND_PREFIX_LEN;
    let start = moves.len() - kept;
    assert_eq!(start, 4);

    let sets = window
        .legal_sets(&at_ply, moves.len())
        .expect("the history covers it");
    assert_eq!(
        sets[COND_PREFIX_LEN], at_ply[start],
        "the row's first move was drawn from the board after {start} moves"
    );
    assert_ne!(
        sets[COND_PREFIX_LEN], at_ply[0],
        "and not from the opening, which is what indexing by row position would give"
    );
    // The two differ as sets and not merely as objects, so the assertion
    // above is not resting on a comparison that could never fail.
    assert!(
        at_ply[start] != at_ply[0] && !at_ply[start].is_empty(),
        "these two plies offer the same moves, so this game cannot show the difference"
    );
    assert_eq!(
        sets[window.len()],
        at_ply[moves.len()],
        "the last entry is still the position on the board"
    );
}

/// The property the all-empty refusal in `LegalSets::window` relies on:
/// a batch of replayed games carries ids.
///
/// The refusal turns "no position of any row holds an id" into an
/// error, which is only safe if the training path does not produce one.
/// It does not here, and the reason is positional rather than
/// statistical — the window starts at position 1, a row's first move
/// sits at position 2 behind `[BOS, band]`, and the opening position
/// offers twenty moves — so this reads the first batch the dataset
/// yields and builds the input from it. What the reason rests on is a
/// `ctx` with room for a move past the prefix; `legal_input_sets` says
/// so, and a caller that took that room away would meet the refusal
/// rather than a silent channel.
#[test]
fn a_batch_of_replayed_games_carries_legal_ids() {
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let mut ds = LegalMaskedDataset::new(
        TeacherRow::into_pairs(corpus_rows(&vocab)),
        vocab.clone(),
        2,
        ds_opts(CTX),
    );
    let batch = ds
        .next_batch()
        .expect("the dataset yields")
        .expect("a first batch");
    let sets = legal_input_sets(&batch, CTX, &Device::Cpu)
        .expect("a batch of replayed games is not all-empty")
        .expect("the legal-masked dataset carries sets");
    assert_eq!(sets.rows(), BATCH);
    assert_eq!(sets.width(), CTX - 1);
    assert!(
        sets.widest() >= 20,
        "the opening position offers twenty moves, so the widest set covers them: got {}",
        sets.widest()
    );
}

/// A legality run resumes from its own checkpoint, table included.
///
/// This is the one path where the extra tensor could go missing without
/// a word: a permissive load would leave `legal_wte.weight` at its
/// random initialisation and report a resume. The strict restore inside
/// `bake` asserts it came back.
#[test]
fn a_legality_run_resumes_from_its_own_checkpoint() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, true);

    let (_, first) = bake(&shape, tmp.path(), "run1", 1, None);
    let (resumed, second) = bake(&shape, tmp.path(), "run2", 1, Some(&first));

    assert!(second.is_file());
    assert!(resumed.train_loss.is_finite(), "got {}", resumed.train_loss);
    assert!(
        ModelShape::load_any(&second)
            .expect("the resumed run's sidecar reads")
            .legal_input,
        "a resumed run writes the convention it ran under"
    );
}

/// The dataset that carries no legal sets cannot drive this entry
/// point, and says so rather than training the model without the
/// channel its checkpoint would record.
#[test]
fn a_legality_run_over_a_plain_dataset_is_refused() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, true);
    let rows = corpus_rows(&vocab);

    let mut ds = TeacherCardDataset::from_rows(TeacherRow::into_pairs(rows), ds_opts(shape.ctx))
        .expect("rows are well formed");
    let cfg = shape.config(Device::Cpu, DType::F32);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build the model");

    let err = run_legal_ft(
        &model,
        &vm,
        &mut ds,
        &ft_cfg(1),
        &CrossEntropyLoss::new(),
        tmp.path(),
        "no_sets",
        Arc::new(TrainingLease::new()),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, TrainError::MissingLegalSets { rows: BATCH }),
        "{err:?}"
    );
}

/// And the arm this one is meant to be compared against still runs:
/// the same corpus and the same legal sets, used only to mask the loss,
/// through the entry point that hands the model nothing.
///
/// The two arms differ in the model as well as the entry point — one
/// carries `legal_wte` and the other does not — which is why the shape
/// records it and every reader checks it.
#[test]
fn the_mask_only_arm_trains_on_the_same_batches() {
    let tmp = TempDir::new().unwrap();
    let vocab = MoveVocab::new(
        &bands()
            .iter()
            .map(|b| b.token.clone())
            .collect::<Vec<String>>(),
    )
    .expect("vocabulary");
    let shape = shape_for(&vocab, false);
    let rows = corpus_rows(&vocab);

    let mut ds = LegalMaskedDataset::new(
        TeacherRow::into_pairs(rows),
        vocab.clone(),
        2,
        ds_opts(shape.ctx),
    );
    let cfg = shape.config(Device::Cpu, DType::F32);
    assert!(
        cfg.custom.is_none(),
        "the mask-only arm is the reference architecture"
    );
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build the model");

    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut ds,
        &ft_cfg(1),
        &CrossEntropyLoss::new(),
        tmp.path(),
        "mask_only",
        Arc::new(TrainingLease::new()),
        None,
    )
    .expect("the legal sets are the loss mask's business on this path");
    assert!(ckpt.train_loss.is_finite(), "got {}", ckpt.train_loss);
    assert!(!vm.data().lock().unwrap().contains_key("legal_wte.weight"));
}
