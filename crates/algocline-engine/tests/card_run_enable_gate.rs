//! Phase 1-B integration tests for the `[setting.card].run` gate on
//! `alc.card.create` / `alc.card.append`.
//!
//! The gate is enforced inside `bridge::data::register_card`.  These tests
//! drive it end-to-end via a Lua VM configured with a tempdir-backed
//! `FileCardStore` so we can assert both the Lua return value and the
//! on-disk side effect (whether a Card TOML was written).  A missing card
//! file is used as the proxy for "no `CardEvent::Created` was published",
//! since `create_with_store` invokes `publish` only after a successful
//! `store.write_new_card` — a gate that short-circuits before the store
//! call cannot reach the event bus.
//!
//! Test matrix:
//!
//! | Case                                    | Enable | run field | Expected                    |
//! |-----------------------------------------|--------|-----------|-----------------------------|
//! | `alc_card_create_run_enabled_writes...` | on     | present   | Card written, `[run]` in TOML |
//! | `alc_card_create_run_disabled_returns...`| off   | present   | nil returned, 0 cards         |
//! | `run_enabled_omits_optional_fields`     | on     | status only | `[run]` has status, no reason/action |
//! | `run_enabled_rejects_invalid_status`    | on     | invalid   | Lua error naming all 3 statuses |

use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::{bridge, FileCardStore, JsonFileStore};
use mlua::prelude::*;
use tempfile::TempDir;

/// Build a Lua VM with the standard `alc.*` bridge installed, backed by
/// a fresh tempdir-rooted `FileCardStore`.  `card_run_enabled` toggles
/// the Phase 1-B gate under test.
///
/// Returns the `Lua`, the `card_store` (so tests can list cards from the
/// Rust side) and the owning `TempDir` (held to keep it alive for the
/// test duration).
fn make_lua_with_bridge(card_run_enabled: bool) -> (Lua, Arc<FileCardStore>, TempDir) {
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().to_path_buf();

    let card_store = Arc::new(FileCardStore::new(root.join("cards")));
    let state_store = Arc::new(JsonFileStore::new(root.join("state")));

    let config = bridge::BridgeConfig {
        llm_tx: None,
        ns: "card_run_gate_test".into(),
        custom_metrics: metrics.custom_metrics_handle(),
        stats: metrics.stats_handle(),
        budget: metrics.budget_handle(),
        progress: metrics.progress_handle(),
        lib_paths: vec![],
        variant_pkgs: vec![],
        state_store,
        card_store: Arc::clone(&card_store),
        card_run_enabled,
        scenarios_dir: root.join("scenarios"),
        nn_dir: root.join("nn"),
        log_sink: None,
    };

    let lua = Lua::new();
    let alc_table = lua.create_table().expect("create alc table");
    bridge::register(&lua, &alc_table, config).expect("bridge::register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    (lua, card_store, tmp)
}

/// Enable=ON + run field present: the card is written and the resulting
/// TOML contains a `[run]` section with `status = "failed"`.
#[test]
fn alc_card_create_run_enabled_writes_run_section() {
    let (lua, card_store, _tmp) = make_lua_with_bridge(true);

    let pkg = "phase1b_enabled_pkg";
    let script = format!(
        r#"
        local r = alc.card.create({{
            pkg = {{ name = "{pkg}" }},
            run = {{
                status = "failed",
                reason = "grader rating < 3",
                action = "write",
            }},
        }})
        return r.card_id
        "#
    );
    let card_id: String = lua
        .load(&script)
        .eval()
        .expect("enable-on create must succeed");

    // Card file must exist.
    let rows = card_store.list(Some(pkg)).expect("list");
    assert_eq!(rows.len(), 1, "exactly one card must be written");
    assert_eq!(rows[0].card_id, card_id);

    // TOML must contain a `[run]` section with the populated fields.
    let card = card_store
        .get(&card_id)
        .expect("read card")
        .expect("card present");
    let run = card
        .get("run")
        .expect("[run] section must be present when enable=on");
    assert_eq!(run.get("status").and_then(|v| v.as_str()), Some("failed"));
    assert_eq!(
        run.get("reason").and_then(|v| v.as_str()),
        Some("grader rating < 3")
    );
    assert_eq!(run.get("action").and_then(|v| v.as_str()), Some("write"));
}

/// Enable=OFF + run field present: the Lua closure returns `nil` and no
/// card file is written.  Since `CardEvent::Created` publication happens
/// only after `store.write_new_card`, a missing card file is a sufficient
/// proxy for "no event was published on the singleton event bus".
#[test]
fn alc_card_create_run_disabled_returns_nil_no_event() {
    let (lua, card_store, _tmp) = make_lua_with_bridge(false);

    let pkg = "phase1b_disabled_pkg";
    let script = format!(
        r#"
        local r = alc.card.create({{
            pkg = {{ name = "{pkg}" }},
            run = {{ status = "succeeded" }},
        }})
        return r
        "#
    );
    let value: LuaValue = lua
        .load(&script)
        .eval()
        .expect("enable-off create must not error");

    assert!(
        value.is_nil(),
        "enable=off create with run field must return nil, got: {value:?}"
    );

    // No card file must exist under the target pkg — the gate short-
    // circuits before `store.write_new_card` and therefore before
    // `publish(CardEvent::Created)`.
    let rows = card_store.list(Some(pkg)).expect("list");
    assert!(
        rows.is_empty(),
        "no card file must be written when gate is off, got {} rows",
        rows.len()
    );
}

/// Enable=ON, only `status` supplied: the resulting TOML must contain
/// `[run].status` but *not* `[run].reason` or `[run].action`.  This
/// verifies the `#[serde(skip_serializing_if = "Option::is_none")]` on
/// `RunSection` propagates through the Card write path.
#[test]
fn alc_card_create_run_enabled_omits_optional_fields_when_absent() {
    let (lua, card_store, _tmp) = make_lua_with_bridge(true);

    let pkg = "phase1b_only_status_pkg";
    let script = format!(
        r#"
        local r = alc.card.create({{
            pkg = {{ name = "{pkg}" }},
            run = {{ status = "skipped" }},
        }})
        return r.card_id
        "#
    );
    let card_id: String = lua.load(&script).eval().expect("create must succeed");

    let card = card_store
        .get(&card_id)
        .expect("read card")
        .expect("card present");
    let run = card.get("run").expect("[run] section must be present");
    assert_eq!(run.get("status").and_then(|v| v.as_str()), Some("skipped"));
    // Optional fields must be absent from the TOML entirely — the Rust
    // side round-trips TOML → JSON via `toml_to_json`, which yields an
    // Object without the missing keys, not a `null` value.
    assert!(
        run.get("reason").is_none(),
        "reason must be absent when Lua omits it, got: {:?}",
        run.get("reason")
    );
    assert!(
        run.get("action").is_none(),
        "action must be absent when Lua omits it, got: {:?}",
        run.get("action")
    );
}

/// Enable=ON, invalid status token: the closure must return a Lua error
/// whose message enumerates the accepted statuses (`succeeded, failed,
/// skipped`) so the caller can act on it without opening the docs.  The
/// gate order (validation before enable check) means invalid input is
/// rejected regardless of `card_run_enabled`.
#[test]
fn alc_card_create_run_enabled_rejects_invalid_status() {
    let (lua, _card_store, _tmp) = make_lua_with_bridge(true);

    let pkg = "phase1b_invalid_status_pkg";
    let script = format!(
        r#"
        local r = alc.card.create({{
            pkg = {{ name = "{pkg}" }},
            run = {{ status = "definitely_not_a_valid_status" }},
        }})
        return r
        "#
    );
    let err = lua
        .load(&script)
        .eval::<LuaValue>()
        .expect_err("invalid status must raise a Lua error");
    let msg = err.to_string();
    assert!(
        msg.contains("succeeded") && msg.contains("failed") && msg.contains("skipped"),
        "error message must enumerate accepted status tokens, got: {msg}"
    );
}
