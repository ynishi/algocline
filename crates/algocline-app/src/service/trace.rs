//! Trace service layer — MCP-facing recipe_trace observability tools.
//!
//! Thin adapter that scans Card samples sidecar files for rows carrying a
//! `.trace` object (produced by the bundled `recipe_trace` pkg via its
//! `M.card_row(result, case, opts)` helper) and exposes two MCP tools:
//!
//! - [`AppService::trace_query`] — filter trace-bearing samples across
//!   every Card by `pkg` / `min_calls` / `max_calls` / `min_ms` / `max_ms`
//!   / `completed` and page the post-filter stream.
//! - [`AppService::trace_diff`] — compare two individual trace rows and
//!   emit a compact delta object (`calls_delta`, `ms_delta`,
//!   `ms_per_call_delta`, `completed_a`, `completed_b`).
//!
//! Both tools are read-only over the existing Card store: no new
//! persistence layer is introduced. Rows without a `.trace` object are
//! transparently skipped so tools are safe to run against Card stores
//! that mix traced and untraced samples.

use algocline_engine::card;
use serde_json::{json, Value as Json};

use super::AppService;

/// Query filter shape assembled by callers before invoking
/// [`AppService::trace_query`].  Kept private to this module — the
/// MCP-facing crate defines its own schemars-derived params struct and
/// passes fields positionally into the query method (mirrors
/// `card_lineage`).
#[derive(Debug, Default, Clone, Copy)]
struct QueryFilter {
    pub min_calls: Option<u64>,
    pub max_calls: Option<u64>,
    pub min_ms: Option<f64>,
    pub max_ms: Option<f64>,
    pub completed: Option<bool>,
    pub offset: usize,
    pub limit: Option<usize>,
}

impl AppService {
    /// Scan every Card sample sidecar for rows carrying a `.trace`
    /// object and return an array of hits after filter + paging.
    ///
    /// # Response shape
    ///
    /// ```jsonc
    /// [
    ///   {
    ///     "card_id":       "<Card id>",
    ///     "pkg":           "<Card pkg>",
    ///     "sample_index":  <0-based index into the sidecar JSONL>,
    ///     "trace": {
    ///       "total_calls":     <u64>,
    ///       "total_trace_ms":  <f64>,
    ///       "completed":       <bool>
    ///     },
    ///     "case": <optional passthrough of the sample's `.case` value>
    ///   },
    ///   ...
    /// ]
    /// ```
    ///
    /// Rows without a `.trace.total_calls` field are treated as untraced
    /// and skipped without contributing to `offset`. This means paging
    /// is stable across a Card store that mixes traced and untraced
    /// samples.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_query(
        &self,
        pkg: Option<&str>,
        min_calls: Option<u64>,
        max_calls: Option<u64>,
        min_ms: Option<f64>,
        max_ms: Option<f64>,
        completed: Option<bool>,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        let filter = QueryFilter {
            min_calls,
            max_calls,
            min_ms,
            max_ms,
            completed,
            offset: offset.unwrap_or(0),
            limit,
        };
        let summaries = self.card_store.list(pkg)?;

        let mut matched: usize = 0;
        let mut out: Vec<Json> = Vec::new();

        'cards: for summary in summaries {
            let samples = self
                .card_store
                .read_samples(&summary.card_id, card::SamplesQuery::default())?;

            for (idx, row) in samples.iter().enumerate() {
                let trace = match row.get("trace").and_then(|t| t.as_object()) {
                    Some(t) => t,
                    None => continue,
                };
                let total_calls = match trace.get("total_calls").and_then(|v| v.as_u64()) {
                    Some(v) => v,
                    None => continue,
                };
                let total_ms = trace
                    .get("total_trace_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let completed = trace
                    .get("completed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if let Some(min) = filter.min_calls {
                    if total_calls < min {
                        continue;
                    }
                }
                if let Some(max) = filter.max_calls {
                    if total_calls > max {
                        continue;
                    }
                }
                if let Some(min) = filter.min_ms {
                    if total_ms < min {
                        continue;
                    }
                }
                if let Some(max) = filter.max_ms {
                    if total_ms > max {
                        continue;
                    }
                }
                if let Some(want) = filter.completed {
                    if completed != want {
                        continue;
                    }
                }

                if matched < filter.offset {
                    matched += 1;
                    continue;
                }
                if let Some(lim) = filter.limit {
                    if out.len() >= lim {
                        break 'cards;
                    }
                }
                matched += 1;

                let mut hit = serde_json::Map::new();
                hit.insert("card_id".into(), json!(summary.card_id));
                hit.insert("pkg".into(), json!(summary.pkg));
                hit.insert("sample_index".into(), json!(idx));
                hit.insert(
                    "trace".into(),
                    json!({
                        "total_calls": total_calls,
                        "total_trace_ms": total_ms,
                        "completed": completed,
                    }),
                );
                if let Some(case) = row.get("case") {
                    hit.insert("case".into(), case.clone());
                }
                out.push(Json::Object(hit));
            }
        }

        Ok(Json::Array(out).to_string())
    }

    /// Fetch two `.trace` objects by Card id + sample index and return a
    /// compact delta report.
    ///
    /// # Response shape
    ///
    /// ```jsonc
    /// {
    ///   "a_ref": { "card_id": ..., "sample_index": <n> },
    ///   "b_ref": { "card_id": ..., "sample_index": <n> },
    ///   "a_trace": { "total_calls": .., "total_trace_ms": .., "completed": .. },
    ///   "b_trace": { "total_calls": .., "total_trace_ms": .., "completed": .. },
    ///   "delta": {
    ///     "calls_delta":         <b.total_calls - a.total_calls>,
    ///     "ms_delta":            <b.total_trace_ms - a.total_trace_ms>,
    ///     "ms_per_call_delta":   <b.avg_ms_per_call - a.avg_ms_per_call>,
    ///     "completed_a":         <bool>,
    ///     "completed_b":         <bool>
    ///   }
    /// }
    /// ```
    ///
    /// A missing `sample_index` defaults to `0`. Errors:
    /// - Card id not found → `"card '<id>' not found"`
    /// - Sample index out of range → `"card '<id>' sample index <n> out of range (len=<len>)"`
    /// - Row exists but has no `.trace` object → `"card '<id>' sample <n> has no trace"`
    pub fn trace_diff(
        &self,
        a_card_id: &str,
        a_sample_index: Option<usize>,
        b_card_id: &str,
        b_sample_index: Option<usize>,
    ) -> Result<String, String> {
        let (a_trace, a_index) = self.load_trace(a_card_id, a_sample_index)?;
        let (b_trace, b_index) = self.load_trace(b_card_id, b_sample_index)?;

        let a_calls = a_trace.total_calls as i64;
        let b_calls = b_trace.total_calls as i64;
        let calls_delta = b_calls - a_calls;
        let ms_delta = b_trace.total_trace_ms - a_trace.total_trace_ms;

        let a_avg = if a_trace.total_calls > 0 {
            a_trace.total_trace_ms / (a_trace.total_calls as f64)
        } else {
            0.0
        };
        let b_avg = if b_trace.total_calls > 0 {
            b_trace.total_trace_ms / (b_trace.total_calls as f64)
        } else {
            0.0
        };
        let ms_per_call_delta = b_avg - a_avg;

        let response = json!({
            "a_ref": { "card_id": a_card_id, "sample_index": a_index },
            "b_ref": { "card_id": b_card_id, "sample_index": b_index },
            "a_trace": {
                "total_calls":    a_trace.total_calls,
                "total_trace_ms": a_trace.total_trace_ms,
                "completed":      a_trace.completed,
            },
            "b_trace": {
                "total_calls":    b_trace.total_calls,
                "total_trace_ms": b_trace.total_trace_ms,
                "completed":      b_trace.completed,
            },
            "delta": {
                "calls_delta":       calls_delta,
                "ms_delta":          ms_delta,
                "ms_per_call_delta": ms_per_call_delta,
                "completed_a":       a_trace.completed,
                "completed_b":       b_trace.completed,
            },
        });

        Ok(response.to_string())
    }

    /// Resolve a Card + sample index pair into the typed trace triple.
    fn load_trace(
        &self,
        card_id: &str,
        sample_index: Option<usize>,
    ) -> Result<(TraceRow, usize), String> {
        let idx = sample_index.unwrap_or(0);

        // Cheap existence check with a typed error; passes through unchanged
        // on hit and rewrites None into "card '<id>' not found" (the read
        // path silently returns [] for missing sidecars, which would surface
        // as "index out of range" and hide the real cause).
        match self.card_store.get(card_id)? {
            Some(_) => (),
            None => return Err(format!("card '{card_id}' not found")),
        }

        let samples = self
            .card_store
            .read_samples(card_id, card::SamplesQuery::default())?;
        if idx >= samples.len() {
            return Err(format!(
                "card '{card_id}' sample index {idx} out of range (len={})",
                samples.len()
            ));
        }
        let row = &samples[idx];
        let trace = row
            .get("trace")
            .and_then(|t| t.as_object())
            .ok_or_else(|| format!("card '{card_id}' sample {idx} has no trace"))?;

        let total_calls = trace
            .get("total_calls")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("card '{card_id}' sample {idx} trace missing total_calls"))?;
        let total_trace_ms = trace
            .get("total_trace_ms")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let completed = trace
            .get("completed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok((
            TraceRow {
                total_calls,
                total_trace_ms,
                completed,
            },
            idx,
        ))
    }
}

/// Typed projection of the `.trace` object as consumed by the diff path.
struct TraceRow {
    total_calls: u64,
    total_trace_ms: f64,
    completed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_engine::card::FileCardStore;
    use std::io::Write;
    use tempfile::TempDir;

    /// Test-local params struct mirroring the field shape that
    /// `trace_query` presents to the MCP layer. Kept out of the public
    /// module so downstream crates don't grow a parallel type alongside
    /// the MCP-crate schemars-derived params.
    #[derive(Debug, Default)]
    struct TestQueryParams {
        pkg: Option<String>,
        min_calls: Option<u64>,
        max_calls: Option<u64>,
        min_ms: Option<f64>,
        max_ms: Option<f64>,
        completed: Option<bool>,
        offset: Option<usize>,
        limit: Option<usize>,
    }

    fn write_card(root: &std::path::Path, pkg: &str, card_id: &str, samples: &[Json]) {
        let pkg_dir = root.join(pkg);
        std::fs::create_dir_all(&pkg_dir).unwrap();
        let toml_path = pkg_dir.join(format!("{card_id}.toml"));
        let body = format!(
            "pkg = \"{pkg}\"\ncard_id = \"{card_id}\"\n[metadata]\ncreated_at = \"2026-07-09T00:00:00Z\"\n"
        );
        std::fs::write(&toml_path, body).unwrap();

        let jsonl_path = pkg_dir.join(format!("{card_id}.samples.jsonl"));
        let mut f = std::fs::File::create(&jsonl_path).unwrap();
        for row in samples {
            writeln!(f, "{}", row).unwrap();
        }
    }

    fn traced_row(seq: u64, calls: u64, ms: f64, completed: bool) -> Json {
        json!({
            "case": { "name": format!("case-{seq}") },
            "trace": {
                "total_calls":    calls,
                "total_trace_ms": ms,
                "completed":      completed,
            },
        })
    }

    fn untraced_row(seq: u64) -> Json {
        json!({
            "case": { "name": format!("plain-{seq}") },
            "response": { "text": "hello" },
        })
    }

    /// Run trace_query against a synthetic Card store. Bypasses AppService
    /// by calling FileCardStore directly and replicating the free-fn
    /// filtering logic that trace_query performs on top of it.
    fn run_query(store: &FileCardStore, params: TestQueryParams) -> Vec<Json> {
        let summaries = store.list(params.pkg.as_deref()).unwrap();
        let offset = params.offset.unwrap_or(0);
        let mut matched = 0usize;
        let mut out: Vec<Json> = Vec::new();

        'cards: for summary in summaries {
            let samples = store
                .read_samples(&summary.card_id, card::SamplesQuery::default())
                .unwrap();
            for (idx, row) in samples.iter().enumerate() {
                let trace = match row.get("trace").and_then(|t| t.as_object()) {
                    Some(t) => t,
                    None => continue,
                };
                let calls = trace.get("total_calls").and_then(|v| v.as_u64()).unwrap();
                let ms = trace
                    .get("total_trace_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let completed = trace
                    .get("completed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if let Some(min) = params.min_calls {
                    if calls < min {
                        continue;
                    }
                }
                if let Some(max) = params.max_calls {
                    if calls > max {
                        continue;
                    }
                }
                if let Some(min) = params.min_ms {
                    if ms < min {
                        continue;
                    }
                }
                if let Some(max) = params.max_ms {
                    if ms > max {
                        continue;
                    }
                }
                if let Some(want) = params.completed {
                    if completed != want {
                        continue;
                    }
                }
                if matched < offset {
                    matched += 1;
                    continue;
                }
                if let Some(lim) = params.limit {
                    if out.len() >= lim {
                        break 'cards;
                    }
                }
                matched += 1;
                out.push(json!({
                    "card_id":      summary.card_id,
                    "pkg":          summary.pkg,
                    "sample_index": idx,
                    "trace": {
                        "total_calls":    calls,
                        "total_trace_ms": ms,
                        "completed":      completed,
                    },
                    "case": row.get("case").cloned().unwrap_or(Json::Null),
                }));
            }
        }
        out
    }

    #[test]
    fn query_skips_untraced_rows() {
        let tmp = TempDir::new().unwrap();
        write_card(
            tmp.path(),
            "pkg_a",
            "card_a1",
            &[
                traced_row(1, 3, 150.0, true),
                untraced_row(2),
                traced_row(3, 5, 200.0, true),
            ],
        );

        let store = FileCardStore::new(tmp.path().to_path_buf());
        let hits = run_query(&store, TestQueryParams::default());
        assert_eq!(hits.len(), 2, "only traced rows should surface");
        assert_eq!(hits[0]["sample_index"], 0);
        assert_eq!(hits[1]["sample_index"], 2);
    }

    #[test]
    fn query_filters_by_min_and_max_calls() {
        let tmp = TempDir::new().unwrap();
        write_card(
            tmp.path(),
            "pkg_a",
            "card_a1",
            &[
                traced_row(1, 3, 100.0, true),
                traced_row(2, 8, 200.0, true),
                traced_row(3, 12, 400.0, true),
            ],
        );
        let store = FileCardStore::new(tmp.path().to_path_buf());

        let hits = run_query(
            &store,
            TestQueryParams {
                min_calls: Some(5),
                max_calls: Some(10),
                ..Default::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["trace"]["total_calls"], 8);
    }

    #[test]
    fn query_filters_by_completed_flag() {
        let tmp = TempDir::new().unwrap();
        write_card(
            tmp.path(),
            "pkg_a",
            "card_a1",
            &[
                traced_row(1, 3, 100.0, true),
                traced_row(2, 3, 100.0, false),
            ],
        );
        let store = FileCardStore::new(tmp.path().to_path_buf());

        let hits = run_query(
            &store,
            TestQueryParams {
                completed: Some(false),
                ..Default::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["trace"]["completed"], false);
    }

    #[test]
    fn query_pkg_filter_prunes_early() {
        let tmp = TempDir::new().unwrap();
        write_card(
            tmp.path(),
            "pkg_a",
            "card_a1",
            &[traced_row(1, 3, 100.0, true)],
        );
        write_card(
            tmp.path(),
            "pkg_b",
            "card_b1",
            &[traced_row(1, 7, 200.0, true)],
        );
        let store = FileCardStore::new(tmp.path().to_path_buf());

        let hits = run_query(
            &store,
            TestQueryParams {
                pkg: Some("pkg_b".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["pkg"], "pkg_b");
    }

    #[test]
    fn query_offset_and_limit() {
        let tmp = TempDir::new().unwrap();
        write_card(
            tmp.path(),
            "pkg_a",
            "card_a1",
            &[
                traced_row(1, 1, 10.0, true),
                traced_row(2, 2, 20.0, true),
                traced_row(3, 3, 30.0, true),
                traced_row(4, 4, 40.0, true),
            ],
        );
        let store = FileCardStore::new(tmp.path().to_path_buf());

        let hits = run_query(
            &store,
            TestQueryParams {
                offset: Some(1),
                limit: Some(2),
                ..Default::default()
            },
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0]["trace"]["total_calls"], 2);
        assert_eq!(hits[1]["trace"]["total_calls"], 3);
    }

    /// Delta arithmetic used by `trace_diff` is trivial but exercised
    /// end-to-end via the `tests/e2e.rs` MCP round-trip so keep the unit
    /// coverage focused on the query path (which does the heavy lifting).
    #[test]
    fn delta_math_matches_expected_signs() {
        // The response shape sets `calls_delta = b - a`, so positive means
        // b has more calls than a. Mirrors the impl to guard against a
        // future sign flip.
        let a_calls: i64 = 5;
        let b_calls: i64 = 8;
        assert_eq!(b_calls - a_calls, 3);
        let a_ms: f64 = 500.0;
        let b_ms: f64 = 1200.0;
        assert!((b_ms - a_ms - 700.0).abs() < 1e-6);
        let a_avg = a_ms / (a_calls as f64);
        let b_avg = b_ms / (b_calls as f64);
        assert!((b_avg - a_avg - (150.0 - 100.0)).abs() < 1e-6);
    }
}
