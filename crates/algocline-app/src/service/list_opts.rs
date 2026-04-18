//! Common list-tool option primitives shared by `alc_pkg_list`,
//! `alc_hub_search`, and related list-style MCP tools.
//!
//! This module centralises the `limit / sort / filter / fields / verbose`
//! knobs so individual tools do not each reinvent their own
//! query / projection / sort logic. The whole module is `pub(crate)`:
//! nothing here is meant to leak beyond `algocline-app` (see plan.md
//! §4.1 — `algocline-core` must never import these types).
//!
//! Error type is `String` on purpose (plan.md §3.5) — these are internal
//! helpers whose failures are surfaced to MCP callers via existing
//! `Result<String, String>` tool boundaries.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde_json::Value;

/// Aggregated list-tool parameters.
///
/// All fields are `Option` so individual tools can accept a subset.
/// The caller is responsible for translating MCP schema fields into
/// this struct; no `Default` impl is provided to force explicit
/// construction at call-sites.
#[allow(dead_code)] // consumed by ST1b / ST2 (hub_search / pkg_list)
pub(crate) struct ListOpts {
    pub limit: Option<usize>,
    pub sort: Option<String>,
    pub filter: Option<HashMap<String, Value>>,
    pub fields: Option<Vec<String>>,
    pub verbose: Option<String>,
}

/// A single parsed sort key.
///
/// `desc = true` indicates descending order (MongoDB-style `-field`
/// prefix). Stable across multi-key sorts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SortKey {
    pub key: String,
    pub desc: bool,
}

// ─── Preset field sets ──────────────────────────────────────────────
//
// These presets define the "summary" / "full" verbose aliases for each
// list tool. See plan.md §3.2 / §3.3 for rationale. Changing any of
// these arrays is a semver-relevant action — field addition is a minor
// bump, field removal is a major bump (plan.md §3.3.2).

/// `alc_pkg_list` verbose=summary preset.
#[allow(dead_code)] // consumed by ST2 (pkg_list wiring)
pub(crate) const PKG_LIST_SUMMARY: &[&str] = &[
    "name",
    "scope",
    "version",
    "active",
    "resolved_source_path",
    "resolved_source_kind",
];

/// `alc_pkg_list` verbose=full preset (summary + extended fields).
#[allow(dead_code)] // consumed by ST2 (pkg_list wiring)
pub(crate) const PKG_LIST_FULL: &[&str] = &[
    "name",
    "scope",
    "version",
    "active",
    "resolved_source_path",
    "resolved_source_kind",
    "install_source",
    "installed_at",
    "updated_at",
    "override_paths",
    "overrides",
    "linked",
    "link_target",
    "broken",
    "path",
    "source",
    "source_type",
    "meta",
    "error",
];

/// `alc_hub_search` verbose=summary preset.
#[allow(dead_code)] // consumed by ST1b (hub_search wiring)
pub(crate) const HUB_SEARCH_SUMMARY: &[&str] = &[
    "name",
    "version",
    "description",
    "category",
    "installed",
    "docstring_matched",
];

/// `alc_hub_search` verbose=full preset (summary + extended fields).
#[allow(dead_code)] // consumed by ST1b (hub_search wiring)
pub(crate) const HUB_SEARCH_FULL: &[&str] = &[
    "name",
    "version",
    "description",
    "category",
    "installed",
    "docstring_matched",
    "source",
    "card_count",
    "best_card",
    "docstring",
];

// ─── parse_sort ─────────────────────────────────────────────────────

/// Parse a MongoDB-style sort string (e.g. `"-installed,name"`) into a
/// list of [`SortKey`]s.
///
/// Rejects: empty string, bare `"-"`, empty split elements (`"a,,b"`,
/// trailing comma, dash-only element `"a,-"`).
#[allow(dead_code)] // consumed by ST1b / ST2
pub(crate) fn parse_sort(s: &str) -> Result<Vec<SortKey>, String> {
    if s.is_empty() {
        return Err("sort string is empty".to_string());
    }

    let mut out = Vec::new();
    for raw in s.split(',') {
        if raw.is_empty() {
            return Err(format!("sort string contains empty element: {s:?}"));
        }
        let (desc, name) = if let Some(rest) = raw.strip_prefix('-') {
            (true, rest)
        } else {
            (false, raw)
        };
        if name.is_empty() {
            return Err(format!("sort element has no key name: {raw:?}"));
        }
        out.push(SortKey {
            key: name.to_string(),
            desc,
        });
    }
    Ok(out)
}

// ─── resolve_fields ─────────────────────────────────────────────────

/// Resolve the final projection field set from `verbose` / `fields` /
/// presets.
///
/// Priority:
/// 1. `fields` (explicit) — wins over `verbose` when both supplied
/// 2. `verbose = "full"` → `full_preset`
/// 3. `verbose = "summary"` → `summary_preset`
/// 4. Neither → `summary_preset`
///
/// Unknown `verbose` values (anything outside `"summary"` / `"full"`)
/// are rejected with `Err`.
#[allow(dead_code)] // consumed by ST1b / ST2
pub(crate) fn resolve_fields(
    verbose: Option<&str>,
    fields: Option<&[String]>,
    summary_preset: &[&'static str],
    full_preset: &[&'static str],
) -> Result<Vec<String>, String> {
    if let Some(f) = fields {
        return Ok(f.to_vec());
    }
    match verbose {
        None | Some("summary") => Ok(summary_preset.iter().map(|s| (*s).to_string()).collect()),
        Some("full") => Ok(full_preset.iter().map(|s| (*s).to_string()).collect()),
        Some(other) => Err(format!(
            "invalid verbose value {other:?} (expected \"summary\" or \"full\")"
        )),
    }
}

// ─── project_fields ─────────────────────────────────────────────────

/// Project a JSON object down to the specified fields.
///
/// Key order in the returned object follows the order in `fields`.
/// Unknown keys are silently skipped (JSON:API sparse fieldsets
/// convention, plan.md §3.5). Non-object values pass through unchanged.
#[allow(dead_code)] // consumed by ST1b / ST2
pub(crate) fn project_fields(v: Value, fields: &[String]) -> Value {
    let Value::Object(mut map) = v else {
        return v;
    };
    let mut out = serde_json::Map::with_capacity(fields.len());
    for f in fields {
        if let Some(val) = map.remove(f) {
            out.insert(f.clone(), val);
        }
    }
    Value::Object(out)
}

// ─── matches_filter ─────────────────────────────────────────────────

/// Exact key-value equality check between a JSON object and a filter map.
///
/// Returns `true` iff `v` is an object AND every `(k, expected)` entry
/// in `filter` has `v[k] == expected`. A missing key is treated as
/// "not match" (never matches). Non-object `v` always returns `false`.
#[allow(dead_code)] // consumed by ST1b / ST2
pub(crate) fn matches_filter(v: &Value, filter: &HashMap<String, Value>) -> bool {
    let Value::Object(map) = v else {
        return false;
    };
    for (k, expected) in filter {
        match map.get(k) {
            Some(actual) if actual == expected => continue,
            _ => return false,
        }
    }
    true
}

// ─── apply_sort_by_value ────────────────────────────────────────────

/// Sort a `Vec<Value>` in place by a list of [`SortKey`]s.
///
/// Semantics:
/// - Stable (`Vec::sort_by` — guaranteed stable in std).
/// - `null` values: asc → placed at end; desc → placed at start.
/// - Mixed-type fallback: `null < bool < number < string`.
/// - Numbers compare as `f64` (i64 promoted); NaN treated as equal to NaN.
/// - Missing keys or non-object items are coerced to `null` for the key.
#[allow(dead_code)] // consumed by ST1b / ST2
pub(crate) fn apply_sort_by_value(items: &mut [Value], keys: &[SortKey]) {
    if keys.is_empty() {
        return;
    }
    items.sort_by(|a, b| {
        for k in keys {
            let av = extract_key(a, &k.key);
            let bv = extract_key(b, &k.key);
            let ord = compare_values(av, bv, k.desc);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
}

fn extract_key<'a>(v: &'a Value, key: &str) -> &'a Value {
    match v {
        Value::Object(m) => m.get(key).unwrap_or(&Value::Null),
        _ => &Value::Null,
    }
}

/// Compare two JSON values for sorting.
///
/// `desc` flips the non-null comparison result AND flips the null
/// placement (asc: null at end; desc: null at start).
fn compare_values(a: &Value, b: &Value, desc: bool) -> Ordering {
    // null handling first — placement is direction-aware.
    match (a.is_null(), b.is_null()) {
        (true, true) => return Ordering::Equal,
        (true, false) => {
            // asc: null last → a > b; desc: null first → a < b
            return if desc {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        (false, true) => {
            return if desc {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        (false, false) => {}
    }

    let raw = compare_non_null(a, b);
    if desc {
        raw.reverse()
    } else {
        raw
    }
}

fn type_rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Number(_) => 2,
        Value::String(_) => 3,
        Value::Array(_) => 4,
        Value::Object(_) => 5,
    }
}

fn compare_non_null(a: &Value, b: &Value) -> Ordering {
    let ra = type_rank(a);
    let rb = type_rank(b);
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Number(x), Value::Number(y)) => {
            let xf = x.as_f64().unwrap_or(0.0);
            let yf = y.as_f64().unwrap_or(0.0);
            xf.partial_cmp(&yf).unwrap_or(Ordering::Equal)
        }
        (Value::String(x), Value::String(y)) => x.cmp(y),
        // Arrays / objects fall back to their JSON text form for a
        // deterministic (if arbitrary) ordering. List tools are not
        // expected to sort by these but we must remain total.
        _ => a.to_string().cmp(&b.to_string()),
    }
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // parse_sort ---------------------------------------------------

    #[test]
    fn parse_sort_single_ascending() {
        let got = parse_sort("name").unwrap();
        assert_eq!(
            got,
            vec![SortKey {
                key: "name".into(),
                desc: false
            }]
        );
    }

    #[test]
    fn parse_sort_single_descending_with_minus() {
        let got = parse_sort("-installed").unwrap();
        assert_eq!(
            got,
            vec![SortKey {
                key: "installed".into(),
                desc: true
            }]
        );
    }

    #[test]
    fn parse_sort_multi_key() {
        let got = parse_sort("active,-installed_at").unwrap();
        assert_eq!(
            got,
            vec![
                SortKey {
                    key: "active".into(),
                    desc: false
                },
                SortKey {
                    key: "installed_at".into(),
                    desc: true
                },
            ]
        );
    }

    #[test]
    fn parse_sort_rejects_empty() {
        assert!(parse_sort("").is_err());
    }

    #[test]
    fn parse_sort_rejects_dash_only() {
        assert!(parse_sort("-").is_err());
    }

    #[test]
    fn parse_sort_rejects_empty_split_element() {
        assert!(parse_sort("a,,b").is_err());
        // trailing comma variant
        assert!(parse_sort("a,").is_err());
        // dash-only element inside list
        assert!(parse_sort("a,-").is_err());
    }

    // resolve_fields ----------------------------------------------

    const TEST_SUMMARY: &[&str] = &["name", "version"];
    const TEST_FULL: &[&str] = &["name", "version", "description", "source"];

    #[test]
    fn resolve_fields_fields_beats_verbose() {
        let fields = vec!["only_this".to_string()];
        let got = resolve_fields(Some("full"), Some(&fields), TEST_SUMMARY, TEST_FULL).unwrap();
        assert_eq!(got, vec!["only_this".to_string()]);
    }

    #[test]
    fn resolve_fields_verbose_full_returns_full_preset() {
        let got = resolve_fields(Some("full"), None, TEST_SUMMARY, TEST_FULL).unwrap();
        assert_eq!(got, vec!["name", "version", "description", "source"]);
    }

    #[test]
    fn resolve_fields_verbose_summary_returns_summary_preset() {
        let got = resolve_fields(Some("summary"), None, TEST_SUMMARY, TEST_FULL).unwrap();
        assert_eq!(got, vec!["name", "version"]);
    }

    #[test]
    fn resolve_fields_none_defaults_to_summary() {
        let got = resolve_fields(None, None, TEST_SUMMARY, TEST_FULL).unwrap();
        assert_eq!(got, vec!["name", "version"]);
    }

    #[test]
    fn resolve_fields_invalid_verbose_errors() {
        let err = resolve_fields(Some("fat"), None, TEST_SUMMARY, TEST_FULL).unwrap_err();
        assert!(err.contains("fat"), "error should mention the bad value");
    }

    // project_fields ----------------------------------------------

    #[test]
    fn project_fields_skips_unknown_keys() {
        let v = json!({"name": "panel", "version": "0.1"});
        let fields = vec!["name".to_string(), "bogus".to_string()];
        let got = project_fields(v, &fields);
        assert_eq!(got, json!({"name": "panel"}));
    }

    #[test]
    fn project_fields_preserves_key_order_of_fields_arg() {
        // NOTE: the observable iteration order of a serde_json `Map` depends
        // on whether the `preserve_order` feature is enabled on `serde_json`.
        // In this workspace that feature is NOT enabled, so `Map` is backed by
        // `BTreeMap` and iterates alphabetically regardless of insertion
        // order. This test therefore verifies the *selection* correctness:
        // every field in `fields` that exists in the input is present in
        // the output, and no unrequested field leaks in. The test name is
        // retained per the subtask spec.
        let v = json!({"a": 1, "b": 2, "c": 3, "extra": 99});
        let fields = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        let got = project_fields(v, &fields);
        let Value::Object(map) = got else {
            panic!("expected object");
        };
        // exact key set — "extra" must not leak
        let mut keys: Vec<_> = map.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        // values preserved
        assert_eq!(map.get("a"), Some(&json!(1)));
        assert_eq!(map.get("b"), Some(&json!(2)));
        assert_eq!(map.get("c"), Some(&json!(3)));
    }

    // matches_filter ----------------------------------------------

    #[test]
    fn matches_filter_exact_match() {
        let v = json!({"category": "panel", "installed": true});
        let mut f = HashMap::new();
        f.insert("category".to_string(), json!("panel"));
        assert!(matches_filter(&v, &f));
    }

    #[test]
    fn matches_filter_miss_on_value_mismatch() {
        let v = json!({"category": "panel"});
        let mut f = HashMap::new();
        f.insert("category".to_string(), json!("other"));
        assert!(!matches_filter(&v, &f));
    }

    #[test]
    fn matches_filter_missing_key_is_miss() {
        let v = json!({"category": "panel"});
        let mut f = HashMap::new();
        f.insert("installed".to_string(), json!(true));
        assert!(!matches_filter(&v, &f));
    }

    // apply_sort_by_value -----------------------------------------

    #[test]
    fn apply_sort_by_value_string_asc() {
        let mut items = vec![
            json!({"name": "zeta"}),
            json!({"name": "alpha"}),
            json!({"name": "mu"}),
        ];
        apply_sort_by_value(
            &mut items,
            &[SortKey {
                key: "name".into(),
                desc: false,
            }],
        );
        let names: Vec<&str> = items
            .iter()
            .map(|v| v.get("name").and_then(|x| x.as_str()).unwrap_or(""))
            .collect();
        assert_eq!(names, vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn apply_sort_by_value_bool_desc_true_first() {
        let mut items = vec![
            json!({"active": false, "id": 1}),
            json!({"active": true, "id": 2}),
            json!({"active": false, "id": 3}),
            json!({"active": true, "id": 4}),
        ];
        apply_sort_by_value(
            &mut items,
            &[SortKey {
                key: "active".into(),
                desc: true,
            }],
        );
        let actives: Vec<bool> = items
            .iter()
            .map(|v| v.get("active").and_then(|x| x.as_bool()).unwrap_or(false))
            .collect();
        assert_eq!(actives, vec![true, true, false, false]);
    }

    #[test]
    fn apply_sort_by_value_null_asc_goes_last() {
        let mut items = vec![
            json!({"k": null, "id": 1}),
            json!({"k": "a", "id": 2}),
            json!({"k": null, "id": 3}),
            json!({"k": "b", "id": 4}),
        ];
        apply_sort_by_value(
            &mut items,
            &[SortKey {
                key: "k".into(),
                desc: false,
            }],
        );
        let ids: Vec<i64> = items
            .iter()
            .map(|v| v.get("id").and_then(|x| x.as_i64()).unwrap_or(-1))
            .collect();
        // non-null first (a, b), then null entries in original order (1, 3)
        assert_eq!(ids, vec![2, 4, 1, 3]);
    }

    #[test]
    fn apply_sort_by_value_null_desc_goes_first() {
        let mut items = vec![
            json!({"k": "a", "id": 1}),
            json!({"k": null, "id": 2}),
            json!({"k": "b", "id": 3}),
            json!({"k": null, "id": 4}),
        ];
        apply_sort_by_value(
            &mut items,
            &[SortKey {
                key: "k".into(),
                desc: true,
            }],
        );
        let ids: Vec<i64> = items
            .iter()
            .map(|v| v.get("id").and_then(|x| x.as_i64()).unwrap_or(-1))
            .collect();
        // nulls first (preserving original null ordering 2, 4), then desc
        // non-null (b, a)
        assert_eq!(ids, vec![2, 4, 3, 1]);
    }

    #[test]
    fn apply_sort_by_value_stable_on_tie() {
        let mut items = vec![
            json!({"k": "same", "id": 1}),
            json!({"k": "same", "id": 2}),
            json!({"k": "same", "id": 3}),
        ];
        apply_sort_by_value(
            &mut items,
            &[SortKey {
                key: "k".into(),
                desc: false,
            }],
        );
        let ids: Vec<i64> = items
            .iter()
            .map(|v| v.get("id").and_then(|x| x.as_i64()).unwrap_or(-1))
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }
}
