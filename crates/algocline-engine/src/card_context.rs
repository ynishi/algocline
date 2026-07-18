//! Card context injection — resolves Card summaries for prompt injection.
//!
//! Phase 3-F MVP: `alc.llm(prompt, {card_context = ...})` opts key wires
//! a small set of prior Cards into the LLM system prompt as an XML-like
//! `<past_cards>` block.  This module is the pure-Rust core: Lua bridge
//! wiring lives in `bridge/llm.rs`.
//!
//! ## Public API
//!
//! * [`CardContextSpec`] — 2-form spec (`CardId` or `Query { pkg, limit }`)
//!   received from the Lua bridge after opts extraction.
//! * [`resolve`] — spec → `Vec<Json>` full Card resolution against a
//!   `CardStore`.  Silent Ok(empty) on not-found / empty query.
//! * [`format_past_cards`] — infallible fixed-template renderer producing
//!   the `<past_cards>...</past_cards>` block.
//!
//! ## Fixed format template (MVP)
//!
//! ```text
//! <past_cards>
//! Card MM/DD pkg=<pkg> card_id=<id> [run.status=<status>] Rating <val> reason=<reason>
//! ...
//! </past_cards>
//! ```
//!
//! * Each Card on 1 line, `created_at` desc order (newest first)
//! * `[run.status=...]` emitted only when the Card JSON carries `run.status`
//! * `Rating <val>` emitted only when `stats.pass_rate` is present
//!   (formatted with 1 decimal digit)
//! * `reason=<reason>` emitted only when `run.reason` is present
//! * `MM/DD` derived from RFC 3339 `created_at` (`&s[5..10]` with the
//!   `-` separator rewritten to `/`); omitted when `created_at` is
//!   missing or too short
//! * Empty input slice → empty string (no `<past_cards>` wrapper)
//!
//! ## Error handling
//!
//! `resolve` returns `Result<Vec<Json>, String>`.  Underlying
//! `card::get_with_store` / `card::find_with_store` errors are re-wrapped
//! with a `"card_context resolve: "` prefix so the bridge layer can
//! surface them via `tracing::warn!` before silently dropping the inject
//! (Phase 3-F Done Criteria #4: silent no-op on resolution failure).

use serde_json::Value as Json;

use crate::card::{self, CardStore, FindQuery, OrderKey};

/// Card context specification extracted from the Lua `card_context` opt.
///
/// Two forms are accepted at the Lua boundary:
///
/// * `card_context = "<card_id>"` → [`CardContextSpec::CardId`]
/// * `card_context = { pkg = "<name>", limit = N }` → [`CardContextSpec::Query`]
///
/// The Lua bridge is responsible for the extraction; this module treats
/// the parsed enum as-is.
#[derive(Debug)]
pub enum CardContextSpec {
    /// Resolve a single Card by id.
    CardId(String),
    /// Resolve the most recent `limit` Cards for a given `pkg`.
    Query {
        pkg: String,
        /// Number of Cards to fetch (default 5 chosen at the Lua bridge).
        limit: usize,
    },
}

/// Resolve a [`CardContextSpec`] against `store`, returning full Card
/// JSON documents.
///
/// * [`CardContextSpec::CardId`] → 1-element `Vec<Json>` on hit, empty
///   `Vec<Json>` on miss (miss is Ok, not Err).
/// * [`CardContextSpec::Query`] → up to `limit` Card JSON values ordered
///   by `created_at` descending; empty result set is Ok.
///
/// The query form loads full TOML for each hit via a second
/// `get_with_store` call (N+1 fetch).  With the MVP default `limit = 5`
/// this is well within I/O budget; if a future phase needs to raise the
/// limit substantially we can add a `find_full_with_store` batching API.
pub fn resolve(store: &dyn CardStore, spec: CardContextSpec) -> Result<Vec<Json>, String> {
    match spec {
        CardContextSpec::CardId(id) => {
            match card::get_with_store(store, &id)
                .map_err(|e| format!("card_context resolve: {e}"))?
            {
                Some(j) => Ok(vec![j]),
                None => Ok(vec![]),
            }
        }
        CardContextSpec::Query { pkg, limit } => {
            let q = FindQuery {
                pkg: Some(pkg),
                where_: None,
                order_by: vec![OrderKey {
                    path: vec!["created_at".to_string()],
                    desc: true,
                }],
                limit: Some(limit),
                offset: None,
            };
            let summaries = card::find_with_store(store, q)
                .map_err(|e| format!("card_context resolve: {e}"))?;
            let mut out = Vec::with_capacity(summaries.len());
            for s in summaries {
                if let Some(j) = card::get_with_store(store, &s.card_id)
                    .map_err(|e| format!("card_context resolve: {e}"))?
                {
                    out.push(j);
                }
            }
            Ok(out)
        }
    }
}

/// Strip newline / carriage return from a Card-derived string so it
/// stays on the single line the template invariant requires and cannot
/// break out of the `<past_cards>` wrapper by injecting a `</past_cards>`
/// line into a `reason` / `pkg` / etc. field.
///
/// Silent lossy transformation — the raw value is only consumed by the
/// LLM prompt, and the stored Card TOML is unchanged. Only `\n` and
/// `\r` are replaced (kept ASCII-safe, other control chars pass
/// through so error messages read naturally).
fn sanitize_line(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

/// Render a slice of resolved Card JSON values as the fixed
/// `<past_cards>` XML-like block.
///
/// Infallible: unknown / missing fields are silently omitted from the
/// per-Card line rather than surfaced as errors.  Empty input yields an
/// empty string so the caller can unconditionally prefix without
/// producing a stray `<past_cards></past_cards>` wrapper.
pub fn format_past_cards(cards: &[Json]) -> String {
    if cards.is_empty() {
        return String::new();
    }
    let mut lines = Vec::with_capacity(cards.len() + 2);
    lines.push("<past_cards>".to_string());
    for c in cards {
        let mut parts: Vec<String> = Vec::with_capacity(6);
        parts.push("Card".to_string());
        if let Some(ts) = c.get("created_at").and_then(|v| v.as_str()) {
            // RFC 3339 slice `[5..10]` is "MM-DD"; use `str::get` for
            // panic-safe UTF-8 boundary handling (non-ASCII created_at
            // is theoretically possible from arbitrary TOML input, and
            // a panic here would leak past the async closure boundary
            // and violate the silent-no-op contract).
            if let Some(mmdd) = ts.get(5..10) {
                parts.push(mmdd.replace('-', "/"));
            }
        }
        if let Some(pkg) = c
            .get("pkg")
            .and_then(|p| p.get("name"))
            .and_then(|v| v.as_str())
        {
            parts.push(format!("pkg={}", sanitize_line(pkg)));
        }
        if let Some(id) = c.get("card_id").and_then(|v| v.as_str()) {
            parts.push(format!("card_id={}", sanitize_line(id)));
        }
        if let Some(status) = c
            .get("run")
            .and_then(|r| r.get("status"))
            .and_then(|v| v.as_str())
        {
            parts.push(format!("[run.status={}]", sanitize_line(status)));
        }
        if let Some(rate) = c
            .get("stats")
            .and_then(|s| s.get("pass_rate"))
            .and_then(|v| v.as_f64())
        {
            parts.push(format!("Rating {rate:.1}"));
        }
        if let Some(reason) = c
            .get("run")
            .and_then(|r| r.get("reason"))
            .and_then(|v| v.as_str())
        {
            parts.push(format!("reason={}", sanitize_line(reason)));
        }
        lines.push(parts.join(" "));
    }
    lines.push("</past_cards>".to_string());
    lines.join("\n")
}

// ═══════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::PathBuf;

    use crate::card::FileCardStore;

    /// Write a Card TOML directly onto disk under `{root}/{pkg}/{card_id}.toml`.
    ///
    /// We bypass `create_with_store` so tests exercise the same on-disk
    /// shape produced by Phase 1-B `[run]` and Phase 2 `[stats]` writers,
    /// with no auto-inject side effects (schema_version etc. are already
    /// baked into the test text).
    fn write_card_toml(root: &std::path::Path, pkg: &str, card_id: &str, toml_text: &str) {
        let pkg_dir = root.join(pkg);
        fs::create_dir_all(&pkg_dir).expect("test setup: create pkg dir");
        let path = pkg_dir.join(format!("{card_id}.toml"));
        fs::write(&path, toml_text).expect("test setup: write card toml");
    }

    /// Build a minimal Card TOML fixture text with configurable fields.
    fn fixture_toml(
        pkg: &str,
        card_id: &str,
        created_at: &str,
        run_status: Option<&str>,
        run_reason: Option<&str>,
        pass_rate: Option<f64>,
    ) -> String {
        let mut out = String::new();
        out.push_str("schema_version = \"card/v0\"\n");
        out.push_str(&format!("card_id = \"{card_id}\"\n"));
        out.push_str(&format!("created_at = \"{created_at}\"\n"));
        out.push_str("\n[pkg]\n");
        out.push_str(&format!("name = \"{pkg}\"\n"));
        if let Some(rate) = pass_rate {
            out.push_str("\n[stats]\n");
            out.push_str(&format!("pass_rate = {rate}\n"));
        }
        if let Some(status) = run_status {
            out.push_str("\n[run]\n");
            out.push_str(&format!("status = \"{status}\"\n"));
            if let Some(reason) = run_reason {
                out.push_str(&format!("reason = \"{reason}\"\n"));
            }
        }
        out
    }

    fn store_with_root() -> (tempfile::TempDir, FileCardStore, PathBuf) {
        let tmp = tempfile::tempdir().expect("test setup: tempdir");
        let root = tmp.path().to_path_buf();
        let store = FileCardStore::new(root.clone());
        (tmp, store, root)
    }

    #[test]
    fn resolve_card_id_found() {
        let (_tmp, store, root) = store_with_root();
        let toml_text = fixture_toml(
            "cot",
            "cot_test_found",
            "2026-07-18T00:00:00Z",
            Some("succeeded"),
            None,
            Some(4.5),
        );
        write_card_toml(&root, "cot", "cot_test_found", &toml_text);

        let out = resolve(&store, CardContextSpec::CardId("cot_test_found".into()))
            .expect("resolve must succeed for a known Card id");
        assert_eq!(out.len(), 1, "one Card expected");
        assert_eq!(
            out[0].get("card_id").and_then(|v| v.as_str()),
            Some("cot_test_found")
        );
    }

    #[test]
    fn resolve_card_id_not_found() {
        let (_tmp, store, _root) = store_with_root();
        let out = resolve(&store, CardContextSpec::CardId("nonexistent".into()))
            .expect("resolve must return Ok(empty) for a missing Card, not Err");
        assert!(out.is_empty(), "no Cards expected for missing id");
    }

    #[test]
    fn resolve_query_returns_n_recent() {
        let (_tmp, store, root) = store_with_root();
        // 5 Cards ascending created_at; query limit=3 should return the 3 newest.
        for i in 0..5 {
            let card_id = format!("cot_bulk_{i}");
            let ts = format!("2026-07-{:02}T00:00:00Z", 10 + i);
            let toml_text = fixture_toml("cot", &card_id, &ts, Some("succeeded"), None, Some(4.0));
            write_card_toml(&root, "cot", &card_id, &toml_text);
        }

        let out = resolve(
            &store,
            CardContextSpec::Query {
                pkg: "cot".to_string(),
                limit: 3,
            },
        )
        .expect("resolve query must succeed");

        assert_eq!(out.len(), 3, "limit=3 must cap the result");
        // Expect descending by created_at → newest first (2026-07-14, -13, -12).
        assert_eq!(
            out[0].get("card_id").and_then(|v| v.as_str()),
            Some("cot_bulk_4")
        );
        assert_eq!(
            out[1].get("card_id").and_then(|v| v.as_str()),
            Some("cot_bulk_3")
        );
        assert_eq!(
            out[2].get("card_id").and_then(|v| v.as_str()),
            Some("cot_bulk_2")
        );
    }

    #[test]
    fn resolve_query_empty_pkg() {
        let (_tmp, store, root) = store_with_root();
        // Populate a different pkg so the store is non-empty overall but
        // the queried pkg yields nothing.
        let toml_text = fixture_toml(
            "other",
            "other_only",
            "2026-07-18T00:00:00Z",
            None,
            None,
            None,
        );
        write_card_toml(&root, "other", "other_only", &toml_text);

        let out = resolve(
            &store,
            CardContextSpec::Query {
                pkg: "missing_pkg".to_string(),
                limit: 5,
            },
        )
        .expect("resolve on absent pkg must be Ok(empty), not Err");
        assert!(out.is_empty(), "empty pkg must yield empty Vec");
    }

    #[test]
    fn format_with_full_card() {
        let card = serde_json::json!({
            "card_id": "cot_full",
            "created_at": "2026-07-18T06:15:00Z",
            "pkg": { "name": "cot" },
            "stats": { "pass_rate": 4.5 },
            "run": { "status": "succeeded", "reason": "clean" },
        });
        let out = format_past_cards(&[card]);
        assert!(out.starts_with("<past_cards>\n"), "must open with tag");
        assert!(out.ends_with("\n</past_cards>"), "must close with tag");
        assert!(out.contains("Card 07/18"), "MM/DD slice: {out}");
        assert!(out.contains("pkg=cot"), "pkg segment: {out}");
        assert!(out.contains("card_id=cot_full"), "card_id segment: {out}");
        assert!(
            out.contains("[run.status=succeeded]"),
            "run.status segment: {out}"
        );
        assert!(out.contains("Rating 4.5"), "rating segment: {out}");
        assert!(out.contains("reason=clean"), "reason segment: {out}");
    }

    #[test]
    fn format_without_run_section() {
        let card = serde_json::json!({
            "card_id": "cot_norun",
            "created_at": "2026-07-17T00:00:00Z",
            "pkg": { "name": "cot" },
            "stats": { "pass_rate": 3.2 },
        });
        let out = format_past_cards(&[card]);
        assert!(out.contains("pkg=cot"), "pkg still emitted: {out}");
        assert!(
            out.contains("card_id=cot_norun"),
            "card_id still emitted: {out}"
        );
        assert!(out.contains("Rating 3.2"), "rating still emitted: {out}");
        assert!(
            !out.contains("[run.status="),
            "run.status must be omitted when absent: {out}"
        );
        assert!(
            !out.contains("reason="),
            "reason must be omitted when absent: {out}"
        );
    }

    #[test]
    fn format_empty_slice() {
        let out = format_past_cards(&[]);
        assert!(out.is_empty(), "empty slice must yield empty string");
    }

    #[test]
    fn format_non_ascii_created_at_does_not_panic() {
        // Non-ASCII in `created_at` is theoretically possible from
        // arbitrary TOML input. `ts[5..10]` would panic on a multi-byte
        // UTF-8 boundary; `ts.get(5..10)` returns `None` and skips the
        // MM/DD segment silently.
        let card = serde_json::json!({
            "card_id": "abc",
            "created_at": "AB\u{1f600}CDEFG-more",
            "pkg": { "name": "cot" },
        });
        let out = format_past_cards(&[card]);
        assert!(out.contains("card_id=abc"));
        // MM/DD skipped because byte 5 is inside the 😀 codepoint.
        assert!(!out.contains("07/18"));
    }

    #[test]
    fn format_sanitizes_newline_in_reason() {
        // A `reason` field carrying `\n</past_cards>...` must NOT break
        // out of the wrapper — newlines are replaced with spaces so
        // the 1-line-per-Card template invariant holds.
        let card = serde_json::json!({
            "card_id": "abc",
            "created_at": "2026-07-18T00:00:00Z",
            "pkg": { "name": "cot" },
            "run": { "status": "failed", "reason": "line1\n</past_cards>\ninjected" },
        });
        let out = format_past_cards(&[card]);
        // Exactly 3 lines: <past_cards> + 1 card line + </past_cards>.
        assert_eq!(
            out.lines().count(),
            3,
            "template must remain 3 lines, got: {out}"
        );
        assert!(out.contains("reason=line1 </past_cards> injected"));
    }
}
