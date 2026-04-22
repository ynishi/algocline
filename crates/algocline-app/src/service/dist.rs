//! `AppService::hub_dist` — facade that chains `hub_reindex` and
//! `hub_gendoc` in a single call.
//!
//! This is the thin composition layer expected by downstream hub
//! repositories that want to regenerate `hub_index.json` and the
//! public documentation artifacts in one shot. It performs no
//! filesystem work of its own — it calls the two underlying
//! services in sequence and assembles their JSON responses into a
//! `{ "reindex": ..., "gendoc": ... }` envelope.
//!
//! Error propagation (per `CLAUDE.md §Service 層 Error 伝播規律`):
//!
//! - If `hub_reindex` fails, this function returns immediately with
//!   `Err("dist: reindex failed: {inner}")` and `hub_gendoc` is not
//!   invoked. No `warn!`, no silent drop.
//! - If `hub_gendoc` fails after a successful reindex, the returned
//!   `Err` text embeds the (already-succeeded) reindex JSON so the
//!   caller can observe both outcomes in a single response:
//!   `Err("dist: gendoc failed: {inner}\nreindex result (succeeded): {json}")`.
//!   The reindex-side side effect (the written `hub_index.json`) is
//!   not rolled back — callers must treat it as authoritative after
//!   a failed gendoc.
//! - Any JSON parse failure on the underlying responses is also
//!   propagated with a `dist:` prefix.

use super::AppService;

impl AppService {
    /// Run `hub_reindex` followed by `hub_gendoc` as a single call.
    ///
    /// See the module-level doc comment for error semantics. `source_dir`
    /// is forwarded to both steps; `output_path` is the reindex
    /// `hub_index.json` destination (callers typically point this at
    /// `{source_dir}/hub_index.json`); the remaining arguments are
    /// forwarded to `hub_gendoc` unchanged.
    pub fn hub_dist(
        &self,
        source_dir: &str,
        output_path: Option<&str>,
        out_dir: Option<&str>,
        projections: Option<&[String]>,
        config_path: Option<&str>,
        lint_strict: Option<bool>,
    ) -> Result<String, String> {
        // Step 1: reindex. Propagate failure immediately — gendoc is
        // not invoked when reindex cannot produce a fresh index.
        let reindex_json = self
            .hub_reindex(output_path, Some(source_dir))
            .map_err(|e| format!("dist: reindex failed: {e}"))?;

        // Step 2: gendoc. On failure, surface the reindex JSON so the
        // caller sees both the succeeded-half and the failed-half.
        let gendoc_json =
            match self.hub_gendoc(source_dir, out_dir, projections, config_path, lint_strict) {
                Ok(json) => json,
                Err(e) => {
                    return Err(format!(
                        "dist: gendoc failed: {e}\nreindex result (succeeded): {reindex_json}"
                    ));
                }
            };

        // Step 3: compose `{ reindex, gendoc }`.
        let reindex_val: serde_json::Value = serde_json::from_str(&reindex_json)
            .map_err(|e| format!("dist: reindex response parse: {e}"))?;
        let gendoc_val: serde_json::Value = serde_json::from_str(&gendoc_json)
            .map_err(|e| format!("dist: gendoc response parse: {e}"))?;

        let composed = serde_json::json!({
            "reindex": reindex_val,
            "gendoc": gendoc_val,
        });
        Ok(composed.to_string())
    }
}
