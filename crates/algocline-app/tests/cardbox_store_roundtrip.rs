//! Round-trips [`CardboxStore`] against the real `cardbox` binary in a
//! tempdir root.
//!
//! The pure mapping is unit-tested in the module itself; what only the
//! binary can answer is whether the arguments those tests assert on are
//! the arguments cardbox actually accepts, and whether what comes back
//! out reconstructs the Card that went in.
//!
//! Skips visibly when `cardbox` is not installed — the same shape as
//! `algocline-engine/tests/lua_unit_test.rs` skipping when `evalframe`
//! is absent, so a fresh checkout keeps `cargo test` green.

use std::path::PathBuf;
use std::process::Command;

use algocline_app::CardboxStore;
use algocline_engine::card::{
    parse_where, CardBackend, CloseOutcome, FindQuery, LineageQuery, RunStatus, SamplesQuery,
};
use serde_json::{json, Value as Json};

/// A store over a fresh tempdir, or `None` with a visible note when the
/// binary is missing. The `TempDir` is returned so it outlives the store.
fn store() -> Option<(CardboxStore, tempfile::TempDir)> {
    match Command::new("cardbox").arg("version").output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            eprintln!(
                "cardbox_store_roundtrip: skipped — `cardbox version` exited {}:\n  {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
            return None;
        }
        Err(e) => {
            eprintln!(
                "cardbox_store_roundtrip: skipped — cardbox is not on PATH ({e}).\n  \
                 Install it with `cargo install runcard` to run this test."
            );
            return None;
        }
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let store = CardboxStore::new(dir.path().to_path_buf(), PathBuf::from("cardbox"));
    Some((store, dir))
}

/// The seed a run is opened with.
fn seed(flow: &str) -> Json {
    json!({
        "pkg": { "name": "demo" },
        "scenario": { "name": "sc1" },
        "created_by": "alc@test",
        "created_at": "2026-09-01T00:00:00Z",
        "params": { "persona": { "moves_count": 3 }, "temperature": 0.7 },
        "model": { "id": "claude-opus-4-6" },
        "run": { "status": "succeeded", "flow": flow, "reason": "seeded" },
    })
}

/// open → write_samples → close(failed) → get → find → alias_set →
/// lineage, in one pass over one store.
#[test]
fn a_run_opens_writes_rows_closes_failed_and_reads_back() {
    let Some((store, _dir)) = store() else { return };

    // ── open ───────────────────────────────────────────────────
    let (card_id, locator) = store.open(seed("coding_orch")).expect("open");
    assert!(!card_id.is_empty());
    assert_eq!(
        locator,
        _dir.path(),
        "a cardbox Card is a row in a store, so the locator is that store"
    );

    let opened = store.get(&card_id).expect("get").expect("card exists");
    assert_eq!(opened["pkg"], json!({ "name": "demo" }));
    assert_eq!(opened["created_at"], json!("2026-09-01T00:00:00Z"));
    assert_eq!(opened.pointer("/run/flow"), Some(&json!("coding_orch")));
    assert_eq!(
        opened.pointer("/run/status"),
        None,
        "an open Card has not finished, so it has no outcome yet"
    );
    assert_eq!(opened.pointer("/cardbox/state"), Some(&json!("open")));

    // ── write_samples ──────────────────────────────────────────
    let rows = vec![
        json!({ "case": "c0", "passed": true, "score": 0.9 }),
        json!({ "case": "c1", "passed": false, "score": 0.1 }),
        json!({ "case": "c2", "passed": true, "score": 0.7 }),
    ];
    store
        .write_samples(&card_id, rows.clone())
        .expect("write_samples");

    let back = store
        .read_samples(&card_id, SamplesQuery::default())
        .expect("read_samples");
    assert_eq!(back, rows, "rows come back verbatim, in order");

    let filtered = store
        .read_samples(
            &card_id,
            SamplesQuery {
                offset: 0,
                limit: None,
                where_: Some(parse_where(&json!({ "passed": true })).expect("parse")),
            },
        )
        .expect("read_samples where");
    assert_eq!(filtered.len(), 2, "cardbox rows evaluates the v0 DSL");

    let paged = store
        .read_samples(
            &card_id,
            SamplesQuery {
                offset: 1,
                limit: Some(1),
                where_: None,
            },
        )
        .expect("read_samples paged");
    assert_eq!(
        paged,
        vec![json!({ "case": "c1", "passed": false, "score": 0.1 })]
    );

    // Write-once, as on the file backend — cardbox itself would append.
    let err = store
        .write_samples(&card_id, rows)
        .expect_err("samples are write-once");
    assert!(err.contains("write-once"), "{err}");

    // ── close(failed) ──────────────────────────────────────────
    let sealed = store
        .close(
            &card_id,
            CloseOutcome {
                status: RunStatus::Failed,
                stats: Some(json!({
                    "pass_rate": 0.67,
                    "metadata": { "nn": { "architecture": "tiny" } }
                })),
                cost: Some(json!({ "usd": 0.012 })),
                error: Some("timeout after 30s".into()),
            },
        )
        .expect("close");

    // ── get: the whole write mapping, inverted ─────────────────
    assert_eq!(sealed["card_id"], json!(card_id));
    assert_eq!(sealed.pointer("/run/status"), Some(&json!("failed")));
    assert_eq!(sealed.pointer("/run/flow"), Some(&json!("coding_orch")));
    assert_eq!(sealed.pointer("/run/reason"), Some(&json!("seeded")));
    assert_eq!(sealed["error"], json!("timeout after 30s"));
    assert_eq!(sealed["cost"], json!({ "usd": 0.012 }));
    assert_eq!(sealed["created_at"], json!("2026-09-01T00:00:00Z"));
    assert_eq!(sealed["model"], json!({ "id": "claude-opus-4-6" }));
    assert_eq!(sealed["scenario"], json!({ "name": "sc1" }));
    assert_eq!(
        sealed["params"],
        json!({ "persona": { "moves_count": 3 }, "temperature": 0.7 })
    );
    assert_eq!(sealed.pointer("/stats/pass_rate"), Some(&json!(0.67)));
    assert_eq!(
        sealed.pointer("/metadata/nn/architecture"),
        Some(&json!("tiny")),
        "metadata is written into stats.metadata and read back out of it"
    );
    assert_eq!(sealed.pointer("/stats/metadata"), None);
    assert_eq!(
        sealed.pointer("/cardbox/state"),
        Some(&json!("closed_failed"))
    );

    // ── find ───────────────────────────────────────────────────
    // The measured mapping: this predicate returns 0 rows if metadata
    // went anywhere but the close's stats.
    let hits = store
        .find(FindQuery {
            pkg: Some("demo".into()),
            where_: Some(
                parse_where(&json!({ "metadata": { "nn": { "architecture": "tiny" } } }))
                    .expect("parse"),
            ),
            ..FindQuery::default()
        })
        .expect("find metadata");
    assert_eq!(hits.len(), 1, "metadata.nn.architecture must be queryable");
    assert_eq!(hits[0].card_id, card_id);
    assert_eq!(hits[0].pkg, "demo");
    assert_eq!(hits[0].pass_rate, Some(0.67));

    // The two translated paths: created_at and run.* are tags.
    for predicate in [
        json!({ "created_at": { "gte": "2026-08-01" } }),
        json!({ "run": { "flow": "coding_orch" } }),
        json!({ "run": { "status": "failed" } }),
        json!({ "params": { "persona": { "moves_count": 3 } } }),
        json!({ "stats": { "pass_rate": { "gte": 0.5 } } }),
        json!({ "scenario": { "name": "sc1" } }),
    ] {
        let hits = store
            .find(FindQuery {
                pkg: Some("demo".into()),
                where_: Some(parse_where(&predicate).expect("parse")),
                ..FindQuery::default()
            })
            .unwrap_or_else(|e| panic!("find {predicate}: {e}"));
        assert_eq!(
            hits.len(),
            1,
            "find {predicate} matched {} rows",
            hits.len()
        );
    }

    // `list` is the same call without a predicate.
    let listed = store.list(Some("demo")).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].card_id, card_id);
    assert!(store.list(Some("nosuchpkg")).expect("list").is_empty());

    // ── alias_set ──────────────────────────────────────────────
    let alias = store
        .alias_set("best_on_demo", &card_id, None, Some("the only one"))
        .expect("alias_set");
    assert_eq!(alias.name, "best_on_demo");
    assert_eq!(alias.card_id, card_id);
    assert_eq!(alias.pkg.as_deref(), Some("demo"));
    assert_eq!(alias.note.as_deref(), Some("the only one"));
    assert!(
        alias.set_at.starts_with("20"),
        "set_at is RFC3339: {}",
        alias.set_at
    );

    assert_eq!(store.alias_list(None).expect("alias_list").len(), 1);
    assert_eq!(store.alias_list(Some("demo")).expect("alias_list").len(), 1);
    assert!(store
        .alias_list(Some("other"))
        .expect("alias_list")
        .is_empty());

    let by_alias = store
        .get_by_alias("best_on_demo")
        .expect("get_by_alias")
        .expect("bound");
    assert_eq!(by_alias["card_id"], json!(card_id));
    // A miss is Ok(None), not an error — cardbox exits 1 for both.
    assert!(store
        .get_by_alias("unbound")
        .expect("get_by_alias")
        .is_none());
    assert!(store.get("no-such-card").expect("get").is_none());

    // ── lineage ────────────────────────────────────────────────
    let mut child_seed = seed("coding_orch");
    child_seed["metadata"] = json!({
        "prior_card_id": card_id,
        "prior_relation": "sweep_variant",
    });
    let (child_id, _) = store.open(child_seed).expect("open child");
    store
        .close(
            &child_id,
            CloseOutcome {
                status: RunStatus::Skipped,
                stats: None,
                cost: None,
                error: None,
            },
        )
        .expect("close child");

    let child = store.get(&child_id).expect("get").expect("exists");
    assert_eq!(
        child.pointer("/metadata/prior_card_id"),
        Some(&json!(card_id))
    );
    assert_eq!(
        child.pointer("/metadata/prior_relation"),
        Some(&json!("sweep_variant"))
    );
    // The status cardbox has no flag for, surviving a real round trip.
    assert_eq!(
        child.pointer("/run/status"),
        Some(&json!("skipped")),
        "skipped must not read back as succeeded"
    );
    assert_eq!(child.pointer("/cardbox/state"), Some(&json!("closed_ok")));

    let walk = store
        .lineage(LineageQuery {
            card_id: child_id.clone(),
            depth: Some(3),
            ..LineageQuery::default()
        })
        .expect("lineage")
        .expect("root exists");
    assert_eq!(walk.root, child_id);
    let ids: Vec<&str> = walk.nodes.iter().map(|n| n.card_id.as_str()).collect();
    assert!(ids.contains(&child_id.as_str()) && ids.contains(&card_id.as_str()));
    let root_node = walk
        .nodes
        .iter()
        .find(|n| n.card_id == child_id)
        .expect("root node");
    assert_eq!(root_node.depth, 0);
    assert_eq!(root_node.pkg, "demo");
    let parent_node = walk
        .nodes
        .iter()
        .find(|n| n.card_id == card_id)
        .expect("parent node");
    assert_eq!(parent_node.depth, -1, "ancestors sit at negative depth");
    assert_eq!(walk.edges.len(), 1);
    assert_eq!(walk.edges[0].from, child_id);
    assert_eq!(walk.edges[0].to, card_id);
    assert_eq!(walk.edges[0].relation.as_deref(), Some("sweep_variant"));

    assert!(
        store
            .lineage(LineageQuery {
                card_id: "no-such-card".into(),
                ..LineageQuery::default()
            })
            .expect("lineage")
            .is_none(),
        "a missing root is Ok(None)"
    );
}

/// `review` and `caveats` map onto a cardbox verb. Nothing else does,
/// and the refusal says so rather than stringifying the section into a
/// tag where a predicate could no longer reach it.
#[test]
fn append_records_a_review_as_an_eval_and_refuses_anything_else() {
    let Some((store, _dir)) = store() else { return };

    let (card_id, _) = store
        .create(json!({
            "pkg": { "name": "demo" },
            "params": { "persona": { "moves_count": 3 } },
            "stats": { "pass_rate": 1.0 },
            "run": { "status": "succeeded" },
        }))
        .expect("create");

    let before = store.get(&card_id).expect("get").expect("exists");
    assert_eq!(before.pointer("/cardbox/evals"), Some(&json!(0)));

    let after = store
        .append(
            &card_id,
            json!({ "review": "reads well", "caveats": "n=1" }),
        )
        .expect("append review");
    assert_eq!(
        after.pointer("/cardbox/evals"),
        Some(&json!(1)),
        "the review was recorded as one human eval"
    );

    let err = store
        .append(&card_id, json!({ "persona": { "moves_count": 3 } }))
        .expect_err("a free-form section has no cardbox verb");
    assert!(err.contains("persona"), "names the section: {err}");
    assert!(err.contains("review"), "names what does map: {err}");
    assert!(
        err.contains("stop matching"),
        "names what stringifying it would cost: {err}"
    );

    assert!(store
        .append("no-such-card", json!({ "review": "x" }))
        .expect_err("missing card")
        .contains("not found"));
}

/// `create` writes a finished Card in one call, and the sections a
/// finished Card carries all land where a query can reach them.
#[test]
fn create_opens_and_closes_in_one_call() {
    let Some((store, _dir)) = store() else { return };

    let (card_id, _) = store
        .create(json!({
            "pkg": { "name": "nn" },
            "scenario": { "name": "bake" },
            "created_at": "2026-07-04T01:02:03Z",
            "params": { "lr": 0.001 },
            "metadata": { "nn": { "candle": { "bundle_ref": "nn/x" }, "architecture": "tiny" } },
            "stats": { "pass_rate": 0.5 },
            "cost": { "usd": 0.5 },
            "run": { "status": "succeeded", "flow": "nn_bake" },
        }))
        .expect("create");

    let card = store.get(&card_id).expect("get").expect("exists");
    assert_eq!(card["created_at"], json!("2026-07-04T01:02:03Z"));
    assert_eq!(card.pointer("/run/status"), Some(&json!("succeeded")));
    assert_eq!(card.pointer("/run/flow"), Some(&json!("nn_bake")));
    assert_eq!(card.pointer("/stats/pass_rate"), Some(&json!(0.5)));
    assert_eq!(card["cost"], json!({ "usd": 0.5 }));
    // `bridge/nn_card.rs` reads exactly this path.
    assert_eq!(
        card.pointer("/metadata/nn/candle/bundle_ref"),
        Some(&json!("nn/x"))
    );

    // Ordering on created_at is the tag column, which is what makes a
    // descending sort reproduce the file backend's order.
    let (older, _) = store
        .create(json!({
            "pkg": { "name": "nn" },
            "created_at": "2026-01-01T00:00:00Z",
            "run": { "status": "succeeded" },
        }))
        .expect("create older");
    let ordered = store
        .find(FindQuery {
            pkg: Some("nn".into()),
            order_by: algocline_engine::card::parse_order_by(&json!("-created_at"))
                .expect("parse_order_by"),
            ..FindQuery::default()
        })
        .expect("find ordered");
    assert_eq!(
        ordered
            .iter()
            .map(|s| s.card_id.as_str())
            .collect::<Vec<_>>(),
        vec![card_id.as_str(), older.as_str()],
        "newest first"
    );

    // A section with nowhere to go is refused, not silently dropped.
    let err = store
        .create(json!({
            "pkg": { "name": "nn" },
            "strategy_params": { "alpha": 0.7 },
        }))
        .expect_err("strategy_params has no slot");
    assert!(err.contains("strategy_params"), "{err}");
}
