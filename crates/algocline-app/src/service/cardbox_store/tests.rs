//! Pure-function tests for the cardbox mapping. Nothing here shells
//! out; the round trip against the real binary lives in
//! `tests/cardbox_store_roundtrip.rs`.

use super::*;
use algocline_engine::card::{parse_order_by, parse_where};

/// The literal `now` every `open_plan` test is handed, so the
/// `created_at` fallback is a value rather than a clock.
const FIXED_NOW: &str = "2026-09-21T12:00:00Z";

fn plan_for(input: Json, mode: OpenMode) -> WritePlan {
    open_plan(&input, FIXED_NOW, mode).expect("open_plan")
}

/// `--flag value` lookup over an argv.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn tag<'a>(plan: &'a WritePlan, key: &str) -> Option<&'a str> {
    plan.tags
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

// ─── v0 Card → CLI arguments ───────────────────────────────────

#[test]
fn open_requires_pkg_name() {
    let err = open_plan(
        &json!({ "params": { "a": 1 } }),
        FIXED_NOW,
        OpenMode::Lifecycle,
    )
    .expect_err("pkg.name is required");
    assert!(err.contains("pkg.name is required"), "{err}");
}

#[test]
fn open_carries_the_required_four_flags() {
    let plan = plan_for(
        json!({
            "pkg": { "name": "cot" },
            "scenario": { "name": "gsm8k" },
            "created_by": "alc@0.49.0",
        }),
        OpenMode::Lifecycle,
    );
    assert_eq!(plan.args[0], "open");
    assert_eq!(flag(&plan.args, "--pkg"), Some("cot"));
    assert_eq!(flag(&plan.args, "--scenario"), Some("gsm8k"));
    assert_eq!(flag(&plan.args, "--source"), Some("alc"));
    assert_eq!(flag(&plan.args, "--created-by"), Some("alc@0.49.0"));
}

#[test]
fn a_card_naming_no_scenario_gets_the_sentinel() {
    let plan = plan_for(json!({ "pkg": { "name": "cot" } }), OpenMode::Lifecycle);
    assert_eq!(flag(&plan.args, "--scenario"), Some("none"));
}

#[test]
fn created_by_defaults_to_this_algocline() {
    let plan = plan_for(json!({ "pkg": { "name": "cot" } }), OpenMode::Lifecycle);
    let expected = format!("alc@{}", env!("CARGO_PKG_VERSION"));
    assert_eq!(flag(&plan.args, "--created-by"), Some(expected.as_str()));
}

#[test]
fn params_go_to_params_verbatim() {
    let plan = plan_for(
        json!({ "pkg": { "name": "cot" }, "params": { "persona": { "moves_count": 3 } } }),
        OpenMode::Lifecycle,
    );
    let sent: Json = serde_json::from_str(flag(&plan.args, "--params").expect("--params"))
        .expect("params is JSON");
    assert_eq!(sent, json!({ "persona": { "moves_count": 3 } }));
}

/// The measured mapping that fails silently when it is wrong: metadata
/// belongs in the close's `stats.metadata`, never in `--params`.
#[test]
fn metadata_never_rides_in_params() {
    let card = json!({
        "pkg": { "name": "nn" },
        "params": { "lr": 0.001 },
        "metadata": { "nn": { "architecture": "tiny" } },
    });
    let plan = plan_for(card.clone(), OpenMode::Create);
    let sent: Json = serde_json::from_str(flag(&plan.args, "--params").expect("--params"))
        .expect("params is JSON");
    assert_eq!(
        sent,
        json!({ "lr": 0.001 }),
        "metadata in params makes the v0 predicate metadata.nn.architecture return 0 rows"
    );

    let outcome = CloseOutcome {
        status: RunStatus::Succeeded,
        stats: None,
        cost: None,
        error: None,
    };
    let close = close_plan(
        "c1",
        &outcome,
        closing_metadata(card.get("metadata")).as_ref(),
    )
    .expect("close_plan");
    let stats: Json = serde_json::from_str(flag(&close.args, "--stats").expect("--stats"))
        .expect("stats is JSON");
    assert_eq!(
        stats,
        json!({ "metadata": { "nn": { "architecture": "tiny" } } })
    );
}

#[test]
fn created_at_is_a_tag_and_falls_back_to_now() {
    let given = plan_for(
        json!({ "pkg": { "name": "cot" }, "created_at": "2026-01-02T03:04:05Z" }),
        OpenMode::Lifecycle,
    );
    assert_eq!(tag(&given, "created_at"), Some("2026-01-02T03:04:05Z"));

    let absent = plan_for(json!({ "pkg": { "name": "cot" } }), OpenMode::Lifecycle);
    assert_eq!(tag(&absent, "created_at"), Some(FIXED_NOW));
}

#[test]
fn lineage_splits_into_a_parent_flag_and_a_relation_tag() {
    let plan = plan_for(
        json!({
            "pkg": { "name": "cot" },
            "metadata": { "prior_card_id": "seed_1", "prior_relation": "sweep_variant" },
        }),
        OpenMode::Lifecycle,
    );
    assert_eq!(flag(&plan.args, "--parent"), Some("seed_1"));
    assert_eq!(tag(&plan, "lineage.relation"), Some("sweep_variant"));
}

#[test]
fn run_fields_become_tags_at_open() {
    let plan = plan_for(
        json!({
            "pkg": { "name": "cot" },
            "run": { "status": "succeeded", "flow": "coding_orch", "reason": "clean" },
        }),
        OpenMode::Lifecycle,
    );
    assert_eq!(tag(&plan, "run.flow"), Some("coding_orch"));
    assert_eq!(tag(&plan, "run.reason"), Some("clean"));
    assert_eq!(tag(&plan, "run.action"), None, "absent fields stay absent");
}

#[test]
fn card_id_is_passed_through_when_the_input_names_one() {
    let named = plan_for(
        json!({ "pkg": { "name": "nn" }, "card_id": "nn_pinned_1" }),
        OpenMode::Lifecycle,
    );
    assert_eq!(flag(&named.args, "--id"), Some("nn_pinned_1"));

    let unnamed = plan_for(json!({ "pkg": { "name": "nn" } }), OpenMode::Lifecycle);
    assert_eq!(flag(&unnamed.args, "--id"), None, "cardbox mints it");
}

#[test]
fn a_section_with_no_slot_is_refused_not_dropped() {
    let err = open_plan(
        &json!({ "pkg": { "name": "cot" }, "strategy_params": { "alpha": 0.7 } }),
        FIXED_NOW,
        OpenMode::Create,
    )
    .expect_err("strategy_params has no slot");
    assert!(err.contains("strategy_params"), "{err}");
    assert!(err.contains("refused rather than dropped"), "{err}");
}

#[test]
fn open_refuses_metadata_that_only_close_can_place() {
    let err = open_plan(
        &json!({ "pkg": { "name": "nn" }, "metadata": { "nn": { "architecture": "tiny" } } }),
        FIXED_NOW,
        OpenMode::Lifecycle,
    )
    .expect_err("non-lineage metadata is close-time data");
    assert!(err.contains("stats.metadata"), "{err}");

    // create() holds the whole Card, so it can place the same metadata.
    open_plan(
        &json!({ "pkg": { "name": "nn" }, "metadata": { "nn": { "architecture": "tiny" } } }),
        FIXED_NOW,
        OpenMode::Create,
    )
    .expect("create places metadata at close");
}

// ─── close ─────────────────────────────────────────────────────

fn outcome(status: RunStatus, error: Option<&str>) -> CloseOutcome {
    CloseOutcome {
        status,
        stats: None,
        cost: None,
        error: error.map(str::to_string),
    }
}

#[test]
fn succeeded_closes_ok() {
    let plan = close_plan("c1", &outcome(RunStatus::Succeeded, None), None).expect("close_plan");
    assert_eq!(plan.args, vec!["close", "c1", "--ok"]);
    assert_eq!(tag(&plan, "run.status"), Some("succeeded"));
}

#[test]
fn failed_closes_failed_and_carries_the_message() {
    let plan =
        close_plan("c1", &outcome(RunStatus::Failed, Some("timeout")), None).expect("close_plan");
    assert_eq!(
        plan.args,
        vec!["close", "c1", "--failed", "--error", "timeout"]
    );
    assert_eq!(tag(&plan, "run.status"), Some("failed"));
}

/// cardbox refuses `--failed` without a message; v0 allows `error =
/// nil`. A named absence is written rather than letting the close fail.
#[test]
fn failed_without_a_message_still_closes() {
    let plan = close_plan("c1", &outcome(RunStatus::Failed, None), None).expect("close_plan");
    assert_eq!(
        flag(&plan.args, "--error"),
        Some("run failed (no error message reported)")
    );
}

/// The one status cardbox has no flag for. It closes `--ok` because a
/// skipped run did not fail, and the tag is what keeps it from reading
/// back as `succeeded`.
#[test]
fn skipped_closes_ok_but_is_tagged_skipped() {
    let plan = close_plan("c1", &outcome(RunStatus::Skipped, None), None).expect("close_plan");
    assert_eq!(plan.args, vec!["close", "c1", "--ok"]);
    assert_eq!(tag(&plan, "run.status"), Some("skipped"));

    let succeeded =
        close_plan("c1", &outcome(RunStatus::Succeeded, None), None).expect("close_plan");
    assert_ne!(
        tag(&plan, "run.status"),
        tag(&succeeded, "run.status"),
        "skipped must stay distinguishable from succeeded"
    );
}

#[test]
fn stats_and_cost_ride_their_own_flags() {
    let o = CloseOutcome {
        status: RunStatus::Succeeded,
        stats: Some(json!({ "pass_rate": 0.8 })),
        cost: Some(json!({ "usd": 0.01 })),
        error: None,
    };
    let plan = close_plan("c1", &o, None).expect("close_plan");
    assert_eq!(flag(&plan.args, "--stats"), Some(r#"{"pass_rate":0.8}"#));
    assert_eq!(flag(&plan.args, "--cost"), Some(r#"{"usd":0.01}"#));
}

// ─── cardbox get JSON → v0 Card ────────────────────────────────

/// A closed Card as `cardbox get` really returns it, including the
/// `{}`-for-empty-list quirk on `parents` / `aliases` / `checkpoints`.
fn cb_closed() -> Json {
    json!({
        "aliases": {},
        "checkpoints": {},
        "closed_ms": 1790026549418i64,
        "cost": { "usd": 0.01 },
        "created_by": "alc@0.49.0",
        "evals": 3,
        "fingerprint": "a77603a009aef8ee",
        "id": "demo_sc1_20260921T213520_87354a",
        "opened_ms": 1790026520694i64,
        "params": { "persona": { "moves_count": 3 } },
        "parents": ["seed_1"],
        "pkg": "demo",
        "samples": { "batches": 1, "rows": 3 },
        "scenario": "sc1",
        "source": "alc",
        "state": "closed_ok",
        "stats": {
            "metadata": { "nn": { "candle": { "bundle_ref": "nn/demo" }, "architecture": "tiny" } },
            "pass_rate": 0.8
        },
        "tags": {
            "created_at": "2026-09-01T00:00:00Z",
            "lineage.relation": "sweep_variant",
            "run.flow": "coding_orch",
            "run.status": "skipped"
        }
    })
}

#[test]
fn get_rebuilds_the_v0_identity_sections() {
    let card = card_from_cardbox(&cb_closed()).expect("reconstruct");
    assert_eq!(card["schema_version"], json!("card/v0"));
    assert_eq!(card["card_id"], json!("demo_sc1_20260921T213520_87354a"));
    assert_eq!(card["pkg"], json!({ "name": "demo" }));
    assert_eq!(card["scenario"], json!({ "name": "sc1" }));
    assert_eq!(card["created_by"], json!("alc@0.49.0"));
    assert_eq!(card["param_fingerprint"], json!("a77603a009aef8ee"));
    assert_eq!(card["params"], json!({ "persona": { "moves_count": 3 } }));
    assert_eq!(card["cost"], json!({ "usd": 0.01 }));
}

/// `card_context` reads `created_at` and slices `[5..10]` out of it.
#[test]
fn created_at_comes_from_the_tag_not_from_opened_ms() {
    let card = card_from_cardbox(&cb_closed()).expect("reconstruct");
    assert_eq!(card["created_at"], json!("2026-09-01T00:00:00Z"));
    assert_eq!(
        card["created_at"].as_str().and_then(|s| s.get(5..10)),
        Some("09-01"),
        "card_context renders MM/DD from this slice"
    );
}

/// Only when the tag is absent — a Card this backend did not write.
#[test]
fn created_at_falls_back_to_the_event_time_when_untagged() {
    let mut cb = cb_closed();
    cb["tags"] = json!({});
    let card = card_from_cardbox(&cb).expect("reconstruct");
    assert_eq!(card["created_at"], json!("2026-09-21T21:35:20Z"));
}

/// `bridge/nn_card.rs` reads exactly this path.
#[test]
fn metadata_comes_back_out_of_stats_to_its_v0_position() {
    let card = card_from_cardbox(&cb_closed()).expect("reconstruct");
    assert_eq!(
        card.pointer("/metadata/nn/candle/bundle_ref"),
        Some(&json!("nn/demo"))
    );
    assert_eq!(
        card.pointer("/metadata/nn/architecture"),
        Some(&json!("tiny"))
    );
    assert_eq!(
        card.pointer("/stats/metadata"),
        None,
        "metadata must not be left behind in stats"
    );
    assert_eq!(card["stats"], json!({ "pass_rate": 0.8 }));
}

#[test]
fn lineage_comes_back_from_the_parent_and_the_tag() {
    let card = card_from_cardbox(&cb_closed()).expect("reconstruct");
    assert_eq!(
        card.pointer("/metadata/prior_card_id"),
        Some(&json!("seed_1"))
    );
    assert_eq!(
        card.pointer("/metadata/prior_relation"),
        Some(&json!("sweep_variant"))
    );
}

/// The round trip the `skipped` decision exists for.
#[test]
fn a_skipped_close_reads_back_as_skipped_not_succeeded() {
    let card = card_from_cardbox(&cb_closed()).expect("reconstruct");
    assert_eq!(card.pointer("/run/status"), Some(&json!("skipped")));
    assert_eq!(card.pointer("/run/flow"), Some(&json!("coding_orch")));
    assert_eq!(card.pointer("/cardbox/state"), Some(&json!("closed_ok")));
}

/// A Card cardbox closed without the tag still gets a status.
#[test]
fn state_is_the_fallback_when_no_status_tag_was_written() {
    let mut cb = cb_closed();
    cb["tags"] = json!({});
    cb["state"] = json!("closed_failed");
    cb["error"] = json!("boom");
    let card = card_from_cardbox(&cb).expect("reconstruct");
    assert_eq!(card.pointer("/run/status"), Some(&json!("failed")));
    assert_eq!(card["error"], json!("boom"));
}

#[test]
fn an_open_card_has_no_run_status_yet() {
    let cb = json!({
        "aliases": {}, "checkpoints": {}, "evals": 0,
        "id": "c1", "opened_ms": 1790026520694i64, "parents": {},
        "pkg": "demo", "samples": { "batches": 0, "rows": 0 },
        "scenario": "none", "source": "alc", "state": "open", "tags": {}
    });
    let card = card_from_cardbox(&cb).expect("reconstruct");
    assert_eq!(card.get("run"), None);
    assert_eq!(card.pointer("/cardbox/state"), Some(&json!("open")));
    assert_eq!(card.pointer("/cardbox/evals"), Some(&json!(0)));
    assert_eq!(
        card.get("scenario"),
        None,
        "the 'none' sentinel reads back as 'named no scenario'"
    );
}

/// What ran is known when the run starts, so an open Card says it even
/// though it has no outcome yet.
#[test]
fn an_open_card_still_reports_what_is_running() {
    let cb = json!({
        "aliases": {}, "checkpoints": {}, "evals": 0,
        "id": "c1", "opened_ms": 1790026520694i64, "parents": {},
        "pkg": "demo", "samples": { "batches": 0, "rows": 0 },
        "scenario": "none", "source": "alc", "state": "open",
        "tags": { "run.flow": "coding_orch", "run.reason": "seeded" }
    });
    let card = card_from_cardbox(&cb).expect("reconstruct");
    assert_eq!(card.pointer("/run/flow"), Some(&json!("coding_orch")));
    assert_eq!(card.pointer("/run/reason"), Some(&json!("seeded")));
    assert_eq!(
        card.pointer("/run/status"),
        None,
        "an open run has not succeeded, failed or been skipped yet"
    );
}

#[test]
fn the_empty_list_quirk_does_not_become_a_parent() {
    let mut cb = cb_closed();
    cb["parents"] = json!({});
    let card = card_from_cardbox(&cb).expect("reconstruct");
    assert_eq!(card.pointer("/metadata/prior_card_id"), None);
}

// ─── where / order_by ──────────────────────────────────────────

fn where_for(v: Json) -> Json {
    where_json(&parse_where(&v).expect("parse_where"), WhereMode::Cards).expect("render")
}

#[test]
fn an_equality_predicate_round_trips_to_the_same_json() {
    assert_eq!(
        where_for(json!({ "params": { "persona": { "moves_count": 3 } } })),
        json!({ "params": { "persona": { "moves_count": 3 } } })
    );
}

/// `compat find` is cardbox's own translation of this path, so it must
/// be handed over untouched — translating it here too would produce
/// `stats.stats.metadata`.
#[test]
fn metadata_is_handed_to_compat_untranslated() {
    assert_eq!(
        where_for(json!({ "metadata": { "nn": { "architecture": "tiny" } } })),
        json!({ "metadata": { "nn": { "architecture": "tiny" } } })
    );
}

#[test]
fn created_at_and_run_fields_are_translated_to_their_tags() {
    assert_eq!(
        where_for(json!({ "created_at": { "gte": "2026-09-01" } })),
        json!({ "tags": { "created_at": { "gte": "2026-09-01" } } })
    );
    assert_eq!(
        where_for(json!({ "run": { "flow": "coding_orch" } })),
        json!({ "tags": { "run.flow": "coding_orch" } })
    );
    assert_eq!(
        where_for(json!({ "run": { "status": "skipped" } })),
        json!({ "tags": { "run.status": "skipped" } }),
        "otherwise a skipped run would be unfindable"
    );
}

#[test]
fn scalar_sections_are_flattened_to_their_columns() {
    assert_eq!(
        where_for(json!({ "scenario": { "name": "gsm8k" } })),
        json!({ "scenario": "gsm8k" })
    );
    assert_eq!(
        where_for(json!({ "model": { "id": "opus" } })),
        json!({ "model": "opus" })
    );
}

#[test]
fn sibling_clauses_merge_instead_of_overwriting() {
    assert_eq!(
        where_for(json!({ "stats": { "pass_rate": { "gte": 0.8 }, "n": { "gte": 30 } } })),
        json!({ "stats": { "pass_rate": { "gte": 0.8 }, "n": { "gte": 30 } } })
    );
}

#[test]
fn or_and_not_are_refused_for_cards_and_allowed_for_rows() {
    let pred = parse_where(&json!({ "_or": [{ "a": 1 }, { "a": 2 }] })).expect("parse");
    let err = where_json(&pred, WhereMode::Cards).expect_err("compat find is AND-only");
    assert!(err.contains("AND-only"), "{err}");

    let rows = where_json(&pred, WhereMode::Rows).expect("rows take the full v0 DSL");
    assert_eq!(rows, json!({ "_or": [{ "a": 1 }, { "a": 2 }] }));
}

/// Sample rows are raw JSON objects, so their paths are not Card paths.
#[test]
fn row_paths_are_never_translated() {
    let pred = parse_where(&json!({ "created_at": "x" })).expect("parse");
    assert_eq!(
        where_json(&pred, WhereMode::Rows).expect("render"),
        json!({ "created_at": "x" })
    );
}

#[test]
fn prior_card_id_is_refused_with_a_pointer_to_lineage() {
    let pred = parse_where(&json!({ "metadata": { "prior_card_id": "seed_1" } })).expect("parse");
    let err = where_json(&pred, WhereMode::Cards).expect_err("a parent is an edge");
    assert!(err.contains("lineage"), "{err}");
}

#[test]
fn order_by_translates_and_carries_the_descending_dash() {
    let keys = parse_order_by(&json!("-created_at")).expect("parse");
    assert_eq!(
        order_by_flag(&keys).expect("flag"),
        Some("--order-by=-tags.created_at".to_string()),
        "this is what reproduces the file backend's ordering"
    );
    let asc = parse_order_by(&json!("stats.pass_rate")).expect("parse");
    assert_eq!(
        order_by_flag(&asc).expect("flag"),
        Some("--order-by=stats.pass_rate".to_string())
    );
    assert_eq!(order_by_flag(&[]).expect("flag"), None);
}

#[test]
fn a_second_sort_key_errors_rather_than_being_dropped() {
    let keys = parse_order_by(&json!(["-stats.pass_rate", "created_at"])).expect("parse");
    let err = order_by_flag(&keys).expect_err("compat find takes one key");
    assert!(err.contains("one key"), "{err}");
}

// ─── row projections ───────────────────────────────────────────

#[test]
fn a_compat_row_becomes_a_summary_carrying_its_tags() {
    let row = json!({
        "card_id": "demo_sc1_x", "pkg": "demo", "scenario": "sc1",
        "state": "closed_ok", "opened_ms": 1790026907079i64, "pass_rate": 0.8,
        "tags": { "created_at": "2026-09-01T00:00:00Z", "run.flow": "coding_orch" }
    });
    let s = summary_from_row(&row).expect("summary");
    assert_eq!(s.card_id, "demo_sc1_x");
    assert_eq!(s.pkg, "demo");
    assert_eq!(s.scenario.as_deref(), Some("sc1"));
    assert_eq!(s.pass_rate, Some(0.8));
    assert_eq!(
        s.created_at.as_deref(),
        Some("2026-09-01T00:00:00Z"),
        "the tag, not opened_ms"
    );
    assert_eq!(s.flow.as_deref(), Some("coding_orch"));
}

/// The row shape of a cardbox older than 0.1.2, and of a 0.1.2 Card
/// that was never tagged. Both leave the fields absent rather than
/// standing `opened_ms` in for a `created_at` it is not.
#[test]
fn a_row_without_the_tags_leaves_both_fields_unknown() {
    let no_tags_key = json!({
        "card_id": "demo_sc1_x", "pkg": "demo",
        "state": "open", "opened_ms": 1790026907079i64
    });
    let empty_tags = json!({
        "card_id": "demo_sc1_x", "pkg": "demo",
        "state": "open", "opened_ms": 1790026907079i64, "tags": {}
    });
    let other_tags = json!({
        "card_id": "demo_sc1_x", "pkg": "demo",
        "state": "open", "opened_ms": 1790026907079i64,
        "tags": { "run.status": "succeeded", "lineage.relation": "sweep_variant" }
    });
    for row in [no_tags_key, empty_tags, other_tags] {
        let s = summary_from_row(&row).unwrap_or_else(|| panic!("summary for {row}"));
        assert_eq!(s.created_at, None, "{row}");
        assert_eq!(s.flow, None, "{row}");
        assert_eq!(s.card_id, "demo_sc1_x");
    }
}

/// Either tag can be present without the other — a Card that set no
/// `run.flow` still carries the `created_at` every open writes.
#[test]
fn one_tag_present_does_not_conjure_the_other() {
    let dated = json!({
        "card_id": "a", "pkg": "demo",
        "tags": { "created_at": "2026-09-01T00:00:00Z" }
    });
    let s = summary_from_row(&dated).expect("summary");
    assert_eq!(s.created_at.as_deref(), Some("2026-09-01T00:00:00Z"));
    assert_eq!(s.flow, None);

    let flowed = json!({ "card_id": "a", "pkg": "demo", "tags": { "run.flow": "nn_bake" } });
    let s = summary_from_row(&flowed).expect("summary");
    assert_eq!(s.created_at, None);
    assert_eq!(s.flow.as_deref(), Some("nn_bake"));
}

#[test]
fn an_alias_row_becomes_a_v0_alias() {
    let row = json!({
        "name": "best", "card_id": "demo_sc1_x", "pkg": "demo",
        "bound_ms": 1790026614926i64, "note": "n"
    });
    let a = alias_from_row(&row).expect("alias");
    assert_eq!(a.name, "best");
    assert_eq!(a.card_id, "demo_sc1_x");
    assert_eq!(a.pkg.as_deref(), Some("demo"));
    assert_eq!(a.set_at, "2026-09-21T21:36:54Z");
    assert_eq!(a.note.as_deref(), Some("n"));
}

// ─── helpers ───────────────────────────────────────────────────

#[test]
fn the_empty_list_quirk_is_absorbed_in_one_place() {
    assert_eq!(as_list(Some(&json!({}))), Vec::<Json>::new());
    assert_eq!(as_list(None), Vec::<Json>::new());
    assert_eq!(as_list(Some(&json!(["a"]))), vec![json!("a")]);
}

#[test]
fn epoch_ms_renders_as_the_rfc3339_v0_expects() {
    assert_eq!(rfc3339_from_epoch_ms(0), "1970-01-01T00:00:00Z");
    assert_eq!(
        rfc3339_from_epoch_ms(1_790_026_520_694),
        "2026-09-21T21:35:20Z"
    );
    assert_eq!(
        rfc3339_from_epoch_ms(1_767_225_600_000),
        "2026-01-01T00:00:00Z"
    );
}

#[test]
fn a_relation_filter_cuts_the_subtree_behind_a_severed_edge() {
    let node = |id: &str, parent: Option<&str>, relation: Option<&str>| LineageNode {
        card_id: id.to_string(),
        pkg: "demo".into(),
        prior_card_id: parent.map(str::to_string),
        prior_relation: relation.map(str::to_string),
        depth: 0,
        stats: None,
    };
    // root ← child (sweep_variant) ← grandchild (sweep_variant),
    // and root ← other (reflection_of).
    let nodes = vec![
        node("root", None, None),
        node("child", Some("root"), Some("sweep_variant")),
        node("grandchild", Some("child"), Some("sweep_variant")),
        node("other", Some("root"), Some("reflection_of")),
        node("under_other", Some("other"), Some("sweep_variant")),
    ];
    let parents = BTreeMap::new();
    let keep = reachable_under_filter("root", &nodes, &parents, &["sweep_variant".to_string()]);
    assert!(keep.contains("root") && keep.contains("child") && keep.contains("grandchild"));
    assert!(!keep.contains("other"), "its edge fails the filter");
    assert!(
        !keep.contains("under_other"),
        "a passing edge behind a severed one is still unreachable"
    );
}

#[test]
fn append_refuses_a_free_form_section_by_name() {
    let msg = append_refusal("persona");
    assert!(msg.contains("persona"), "{msg}");
    assert!(msg.contains("review"), "names what does map: {msg}");
    assert!(msg.contains("stop matching"), "names the cost: {msg}");
}
