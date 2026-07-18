//! `[setting.*]` generic config table — parse and field-level resolution.
//!
//! ## Resolution order (per-field, highest priority first)
//!
//! 1. Environment variable `ALC_SETTING_<TARGET>_<FIELD>` (screaming snake-case)
//! 2. Project-local `alc.local.toml` (worktree-scoped, git-ignored)
//! 3. Project-local `alc.toml` (git-tracked, shared)
//! 4. User-global `~/.algocline/config.toml`
//!
//! ## Schemaless design
//!
//! Target names and field names are arbitrary (`BTreeMap<String, _>`).
//! No per-target schema is enforced here; all schema validation is the
//! responsibility of the caller.
//!
//! ## Known targets
//!
//! Known targets are informational only. No target is whitelisted.
//! Callers may add new `[setting.<name>]` sections freely; the resolver
//! is target-agnostic (see `## Schemaless design` above).
//!
//! Targets currently used by algocline:
//!
//! - `card.run: bool` — Whether Card writes are enabled on the
//!   Run path. Consumer-side default: `false`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use algocline_core::AppDir;
use serde::{Deserialize, Serialize};

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum SettingError {
    #[error("setting: failed to read {path}: {source}")]
    IoRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("setting: failed to parse TOML at {path}: {source}")]
    TomlParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("setting: target '{target}' contains invalid characters (snake_case only)")]
    InvalidTarget { target: String },
}

// ─── Output types ────────────────────────────────────────────────────────────

/// Result of resolving `[setting.*]` configuration across all layers.
///
/// `resolved` maps `target -> field -> value`.
/// `sources` maps `target -> field -> layer` where layer ∈ {"env", "project", "global"}.
///
/// Both maps are always populated in a single call — callers never need to
/// perform additional TOML or env parsing to obtain the complete resolved config
/// (crux-card.md must_not_simplify: alc_setting_resolve single-call contract).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SettingResolved {
    /// Resolved field values per target.
    pub resolved: BTreeMap<String, BTreeMap<String, serde_json::Value>>,
    /// Winning layer per field per target.
    pub sources: BTreeMap<String, BTreeMap<String, String>>,
}

impl SettingResolved {
    /// Get a field value as a boolean.
    ///
    /// Returns `None` when the target is absent, the field is absent,
    /// or the value is not a boolean. Callers decide the default value
    /// (typically `.unwrap_or(false)` or `.unwrap_or(true)`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let resolved = resolve_setting(&app_dir, project_root, Some("card"))?;
    /// let run = resolved.get_bool("card", "run").unwrap_or(false);
    /// ```
    // The first production consumer is planned for Phase 1-B
    // (`bridge/data.rs::resolve_setting(...).get_bool("card", "run")`).
    // Until then, this method is exercised by unit tests only. The
    // `service::setting` module is `pub(crate)`, so the reachability analyser
    // cannot see the tests-only call sites as usages of the public API.
    #[allow(dead_code)]
    pub fn get_bool(&self, target: &str, field: &str) -> Option<bool> {
        self.resolved
            .get(target)
            .and_then(|fields| fields.get(field))
            .and_then(|v| v.as_bool())
    }
}

// ─── Internal TOML holder ────────────────────────────────────────────────────

/// Minimal TOML struct for `~/.algocline/config.toml` that captures only the
/// `[setting.*]` section. Other keys (e.g. `[hub]`) are ignored via
/// `#[serde(default)]`.
#[derive(Debug, Default, Deserialize)]
struct ConfigToml {
    #[serde(default)]
    setting: BTreeMap<String, BTreeMap<String, toml::Value>>,
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate a target name: must match `[a-z][a-z0-9_]*` (snake_case, lower).
///
/// Target names embedded in env-var keys are derived from the same rule.
pub fn validate_target(target: &str) -> Result<(), SettingError> {
    if target.is_empty() {
        return Err(SettingError::InvalidTarget {
            target: target.to_string(),
        });
    }

    let mut chars = target.chars();
    // Non-empty is guaranteed by the check above; map None to InvalidTarget to
    // avoid `.unwrap()` in library code (§1-2-5 panic-in-product).
    let first = chars.next().ok_or_else(|| SettingError::InvalidTarget {
        target: target.to_string(),
    })?;
    if !first.is_ascii_lowercase() {
        return Err(SettingError::InvalidTarget {
            target: target.to_string(),
        });
    }

    for c in chars {
        if !c.is_ascii_lowercase() && !c.is_ascii_digit() && c != '_' {
            return Err(SettingError::InvalidTarget {
                target: target.to_string(),
            });
        }
    }

    Ok(())
}

// ─── TOML value conversion ───────────────────────────────────────────────────

/// Convert `toml::Value` to `serde_json::Value`, preserving typed distinctions.
///
/// Mapping:
/// - `String`  → `String`
/// - `Boolean` → `Bool`
/// - `Integer` → `Number` (i64)
/// - `Float`   → `Number` (f64; falls back to `null` if non-finite)
/// - `Array`   → `Array` (recursive)
/// - `Table`   → `Object` (recursive)
/// - `Datetime`→ `String` (ISO-8601 representation)
fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Integer(i) => serde_json::json!(*i),
        toml::Value::Float(f) => {
            // serde_json requires finite floats; fall back to null for ±inf / NaN.
            serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        toml::Value::Array(arr) => serde_json::Value::Array(arr.iter().map(toml_to_json).collect()),
        toml::Value::Table(tbl) => {
            let map: serde_json::Map<String, serde_json::Value> = tbl
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    }
}

// ─── File loading ─────────────────────────────────────────────────────────────

/// Load the `[setting.*]` section from a TOML file at `path`.
///
/// Returns `Ok(BTreeMap::new())` when the file is absent (same "file absent
/// == empty config" convention as `hub.rs::collection_url_from_config`).
///
/// Any other I/O error is propagated as `SettingError::IoRead`.
/// A TOML parse error is propagated as `SettingError::TomlParse`.
fn load_setting_section(
    path: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, serde_json::Value>>, SettingError> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(source) => {
            return Err(SettingError::IoRead {
                path: path.to_path_buf(),
                source,
            })
        }
    };

    let doc: ConfigToml = toml::from_str(&content).map_err(|source| SettingError::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;

    let converted = doc
        .setting
        .into_iter()
        .map(|(target, fields)| {
            let json_fields: BTreeMap<String, serde_json::Value> = fields
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect();
            (target, json_fields)
        })
        .collect();

    Ok(converted)
}

// ─── Project layer ────────────────────────────────────────────────────────────

/// Merge `alc.toml` + `alc.local.toml` into a single project-layer map.
///
/// `alc.local.toml` wins over `alc.toml` at **field** granularity.
/// This re-applies the same field-level merge algorithm used for the
/// env / project / global layers, keeping the abstraction consistent.
fn load_project_layer(
    project_root: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, serde_json::Value>>, SettingError> {
    let alc_path = project_root.join("alc.toml");
    let local_path = project_root.join("alc.local.toml");

    let base = load_setting_section(&alc_path)?;
    let local = load_setting_section(&local_path)?;

    // Field-level merge: local wins over base.
    let mut merged: BTreeMap<String, BTreeMap<String, serde_json::Value>> = base;
    for (target, local_fields) in local {
        let entry = merged.entry(target).or_default();
        for (field, value) in local_fields {
            entry.insert(field, value);
        }
    }

    Ok(merged)
}

// ─── Env override scanning ────────────────────────────────────────────────────

/// Scan environment variables for `ALC_SETTING_<TARGET>_<FIELD>=<value>`.
///
/// Naming convention:
/// - Prefix: `ALC_SETTING_`
/// - Target: uppercase snake-case of the target name (e.g. `journal` → `JOURNAL`)
/// - Field:  uppercase snake-case of the field name  (e.g. `path`    → `PATH`)
///
/// Value parsing order (first match wins):
/// 1. `"true"` / `"false"` (case-insensitive) → `Bool`
/// 2. Decimal integer → `Number` (i64)
/// 3. Decimal float   → `Number` (f64)
/// 4. Anything else   → `String` (string fallback; caller decides interpretation)
///
/// The env-var values are always `String` at the OS level; this heuristic
/// matches figment's `Env` provider convention.
///
/// # Note on test isolation
///
/// env-var-touching tests must use `std::env::set_var` / `std::env::remove_var`
/// in a scoped, serialised fashion (see `#[serial]` annotation on those tests).
/// Neither `serial_test` nor `temp-env` is available in this workspace, so
/// isolation is manual. The `#[serial]` attribute is emulated by running the
/// relevant tests serially under a common module-level mutex — see test module
/// doc comment.
pub fn scan_env_overrides() -> BTreeMap<String, BTreeMap<String, serde_json::Value>> {
    const PREFIX: &str = "ALC_SETTING_";

    let mut result: BTreeMap<String, BTreeMap<String, serde_json::Value>> = BTreeMap::new();

    for (key, raw_value) in std::env::vars() {
        let Some(rest) = key.strip_prefix(PREFIX) else {
            continue;
        };

        // rest = "<TARGET>_<FIELD>" — split on the first `_`-separated segment
        // that separates the target from the field.
        // Both TARGET and FIELD may contain underscores themselves, so we need
        // a convention: the target portion ends at the first occurrence that,
        // when lowercased, forms a valid snake_case target.  In practice the
        // convention is that the target name is a single segment of
        // uppercase-only letters and digits separated by underscores, and the
        // field begins after the first underscore boundary.
        //
        // We apply a greedy approach: scan all possible split points and take
        // the longest valid target prefix (i.e. the last `_`-boundary at which
        // the left part contains at least one character).
        //
        // For the common case `JOURNAL_PATH`, there is only one split → target
        // = "journal", field = "path".
        //
        // For a two-word target like `MY_TARGET_PATH`, we need to distinguish
        // target="my_target" field="path" from target="my" field="target_path".
        // We take the split that produces the longest valid target name (greedy).
        //
        // The actual target name roundtrip: env key segment → lowercase.
        let segments: Vec<&str> = rest.split('_').collect();
        if segments.len() < 2 {
            // Need at least one target segment and one field segment.
            continue;
        }

        // Try splits from longest target prefix to shortest.
        // A valid env-derived target consists only of uppercase letters,
        // digits, and underscores; when lowercased it must also pass
        // `validate_target`.
        let mut found = false;
        for split_at in (1..segments.len()).rev() {
            let target_part = segments[..split_at].join("_");
            let field_part = segments[split_at..].join("_");

            if target_part.is_empty() || field_part.is_empty() {
                continue;
            }

            let target_lower = target_part.to_ascii_lowercase();
            let field_lower = field_part.to_ascii_lowercase();

            if validate_target(&target_lower).is_ok() {
                let value = parse_env_value(&raw_value);
                result
                    .entry(target_lower)
                    .or_default()
                    .insert(field_lower, value);
                found = true;
                break;
            }
        }
        let _ = found; // suppress unused warning
    }

    result
}

/// Parse a raw env-var string into a `serde_json::Value`.
///
/// Order: bool → i64 → f64 → String fallback.
fn parse_env_value(raw: &str) -> serde_json::Value {
    // Bool
    match raw.to_ascii_lowercase().as_str() {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        _ => {}
    }

    // i64
    if let Ok(i) = raw.parse::<i64>() {
        return serde_json::json!(i);
    }

    // f64
    if let Ok(f) = raw.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return serde_json::Value::Number(n);
        }
    }

    // String fallback
    serde_json::Value::String(raw.to_string())
}

// ─── Core resolution ──────────────────────────────────────────────────────────

/// Resolve `[setting.*]` configuration across all layers.
///
/// ## Resolution order (per field, highest priority wins)
///
/// 1. `ALC_SETTING_<TARGET>_<FIELD>` env vars
/// 2. `{project_root}/alc.local.toml` (local wins over base within project layer)
/// 3. `{project_root}/alc.toml`
/// 4. `~/.algocline/config.toml` (user-global)
///
/// ## Schemaless
///
/// Target names and field names are arbitrary.  No target is whitelisted or
/// validated by content — only the snake_case *format* is checked for
/// `target` when `target` is `Some(t)`.
///
/// ## Return value
///
/// Returns a `SettingResolved` containing both `resolved` (values) and
/// `sources` (winning layer per field) populated in a single call.
/// The caller never needs to perform additional parsing.
pub fn resolve_setting(
    app_dir: &AppDir,
    project_root: Option<&Path>,
    target: Option<&str>,
) -> Result<SettingResolved, SettingError> {
    // Validate target format if specified.
    if let Some(t) = target {
        validate_target(t)?;
    }

    // ── Gather each layer ──────────────────────────────────────────────────
    let env_layer = scan_env_overrides();

    let project_layer: BTreeMap<String, BTreeMap<String, serde_json::Value>> =
        if let Some(root) = project_root {
            load_project_layer(root)?
        } else {
            BTreeMap::new()
        };

    let global_layer = load_setting_section(&app_dir.config_toml())?;

    // ── Collect the union of all target names ──────────────────────────────
    let all_targets: std::collections::BTreeSet<String> = env_layer
        .keys()
        .chain(project_layer.keys())
        .chain(global_layer.keys())
        .cloned()
        .collect();

    let mut resolved: BTreeMap<String, BTreeMap<String, serde_json::Value>> = BTreeMap::new();
    let mut sources: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();

    for t in &all_targets {
        // Skip if a specific target was requested and this isn't it.
        if let Some(requested) = target {
            if t != requested {
                continue;
            }
        }

        let env_fields = env_layer.get(t);
        let project_fields = project_layer.get(t);
        let global_fields = global_layer.get(t);

        // Union of field names across all layers for this target.
        let all_fields: std::collections::BTreeSet<String> = env_fields
            .iter()
            .flat_map(|m| m.keys())
            .chain(project_fields.iter().flat_map(|m| m.keys()))
            .chain(global_fields.iter().flat_map(|m| m.keys()))
            .cloned()
            .collect();

        if all_fields.is_empty() {
            continue;
        }

        let mut t_resolved: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut t_sources: BTreeMap<String, String> = BTreeMap::new();

        for f in &all_fields {
            if let Some(v) = env_fields.and_then(|m| m.get(f)) {
                t_resolved.insert(f.clone(), v.clone());
                t_sources.insert(f.clone(), "env".to_string());
            } else if let Some(v) = project_fields.and_then(|m| m.get(f)) {
                t_resolved.insert(f.clone(), v.clone());
                t_sources.insert(f.clone(), "project".to_string());
            } else if let Some(v) = global_fields.and_then(|m| m.get(f)) {
                t_resolved.insert(f.clone(), v.clone());
                t_sources.insert(f.clone(), "global".to_string());
            }
            // else: no layer has the field — skip (field absent from output)
        }

        if !t_resolved.is_empty() {
            resolved.insert(t.clone(), t_resolved);
            sources.insert(t.clone(), t_sources);
        }
    }

    Ok(SettingResolved { resolved, sources })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Env-var test isolation ───────────────────────────────────────────────
    //
    // `serial_test` and `temp-env` are not available as workspace dev-deps.
    // We use a module-level `std::sync::Mutex` to serialise all tests that
    // touch env vars, preventing data races between test threads.

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard {
        key: String,
        old_value: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: &str) -> Self {
            let old_value = std::env::var(key).ok();
            // SAFETY: tests are serialised via ENV_MUTEX; no concurrent access.
            #[allow(deprecated)]
            unsafe {
                std::env::set_var(key, value);
            }
            Self {
                key: key.to_string(),
                old_value,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old_value {
                Some(v) => {
                    #[allow(deprecated)]
                    unsafe {
                        std::env::set_var(&self.key, v);
                    }
                }
                None => {
                    #[allow(deprecated)]
                    unsafe {
                        std::env::remove_var(&self.key);
                    }
                }
            }
        }
    }

    // ── validate_target tests (tests #1-3) ────────────────────────────────────

    #[test]
    fn validate_target_snake_case_ok() {
        assert!(validate_target("journal").is_ok());
        assert!(validate_target("my_advisor").is_ok());
        assert!(validate_target("a").is_ok());
        assert!(validate_target("a1").is_ok());
        assert!(validate_target("a_1_b").is_ok());
    }

    #[test]
    fn validate_target_rejects_uppercase() {
        assert!(matches!(
            validate_target("Journal"),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target("JOURNAL"),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target("my_Advisor"),
            Err(SettingError::InvalidTarget { .. })
        ));
    }

    #[test]
    fn validate_target_rejects_special() {
        assert!(matches!(
            validate_target("my-target"),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target("a.b"),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target(""),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target("1start"),
            Err(SettingError::InvalidTarget { .. })
        ));
        assert!(matches!(
            validate_target("_start"),
            Err(SettingError::InvalidTarget { .. })
        ));
    }

    // ── load_setting_section tests (tests #4-6) ───────────────────────────────

    #[test]
    fn load_setting_section_returns_empty_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.toml");
        let result = load_setting_section(&path).unwrap();
        assert!(result.is_empty(), "expected empty map for absent file");
    }

    #[test]
    fn load_setting_section_parses_string_bool_number() {
        let tmp = tempfile::tempdir().unwrap();
        let content = "[setting.j]\npath = \"a\"\npkg = true\nn = 42\n";
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, content).unwrap();

        let result = load_setting_section(&path).unwrap();
        let j = result.get("j").expect("expected target 'j'");

        assert_eq!(j.get("path"), Some(&serde_json::json!("a")));
        assert_eq!(j.get("pkg"), Some(&serde_json::json!(true)));
        assert_eq!(j.get("n"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn load_setting_section_propagates_parse_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.toml");
        std::fs::write(&path, "[[not valid toml\n").unwrap();

        let result = load_setting_section(&path);
        assert!(
            matches!(result, Err(SettingError::TomlParse { .. })),
            "expected TomlParse error, got: {result:?}"
        );
    }

    // ── scan_env_overrides tests (tests #7-9) ─────────────────────────────────

    #[test]
    fn scan_env_overrides_parses_screaming_snake() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_JOURNAL_PATH", "/x");

        let result = scan_env_overrides();
        let journal = result.get("journal").expect("expected 'journal' target");
        assert_eq!(
            journal.get("path"),
            Some(&serde_json::json!("/x")),
            "expected path = /x"
        );
    }

    #[test]
    fn scan_env_overrides_parses_bool() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_JOURNAL_PKG", "true");

        let result = scan_env_overrides();
        let journal = result.get("journal").expect("expected 'journal' target");
        assert_eq!(
            journal.get("pkg"),
            Some(&serde_json::Value::Bool(true)),
            "expected pkg = true (Bool)"
        );
    }

    #[test]
    fn scan_env_overrides_string_fallback() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_JOURNAL_PATH", "/some/path");

        let result = scan_env_overrides();
        let journal = result.get("journal").expect("expected 'journal' target");
        assert_eq!(
            journal.get("path"),
            Some(&serde_json::Value::String("/some/path".to_string())),
            "expected string fallback"
        );
    }

    // ── resolve_setting tests (tests #10-15) ─────────────────────────────────

    fn make_app_dir(root: &Path) -> AppDir {
        AppDir::new(root.to_path_buf())
    }

    fn write_config(dir: &Path, content: &str) {
        std::fs::write(dir.join("config.toml"), content).unwrap();
    }

    fn write_alc_toml(dir: &Path, content: &str) {
        std::fs::write(dir.join("alc.toml"), content).unwrap();
    }

    fn write_local_alc_toml(dir: &Path, content: &str) {
        std::fs::write(dir.join("alc.local.toml"), content).unwrap();
    }

    #[test]
    fn resolve_setting_field_level_env_wins() {
        // Crux-card #2 enforcement: env wins for `path`, global wins for `pkg`.
        // ENV_MUTEX serialises all resolve_setting tests to prevent ALC_SETTING_*
        // pollution across threads.
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_JOURNAL_PATH", "B");

        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(
            global_tmp.path(),
            "[setting.journal]\npath = \"A\"\npkg = true\n",
        );

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        let resolved_journal = result
            .resolved
            .get("journal")
            .expect("expected 'journal' in resolved");
        let sources_journal = result
            .sources
            .get("journal")
            .expect("expected 'journal' in sources");

        // env overrides path
        assert_eq!(
            resolved_journal.get("path"),
            Some(&serde_json::json!("B")),
            "env should win for 'path'"
        );
        // global wins for pkg (env has no override for pkg)
        assert_eq!(
            resolved_journal.get("pkg"),
            Some(&serde_json::json!(true)),
            "global should provide 'pkg'"
        );

        assert_eq!(sources_journal.get("path"), Some(&"env".to_string()));
        assert_eq!(sources_journal.get("pkg"), Some(&"global".to_string()));
    }

    #[test]
    fn resolve_setting_project_wins_over_global() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(global_tmp.path(), "[setting.journal]\npath = \"A\"\n");
        write_alc_toml(project_tmp.path(), "[setting.journal]\npath = \"B\"\n");

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        let resolved = result.resolved.get("journal").unwrap();
        let sources = result.sources.get("journal").unwrap();

        assert_eq!(resolved.get("path"), Some(&serde_json::json!("B")));
        assert_eq!(sources.get("path"), Some(&"project".to_string()));
    }

    #[test]
    fn resolve_setting_target_filter_all() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(
            global_tmp.path(),
            "[setting.journal]\npath = \"A\"\n[setting.advisor]\nmodel = \"gpt4\"\n",
        );

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        assert!(
            result.resolved.contains_key("journal"),
            "should include 'journal'"
        );
        assert!(
            result.resolved.contains_key("advisor"),
            "should include 'advisor'"
        );
    }

    #[test]
    fn resolve_setting_target_filter_specific() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(
            global_tmp.path(),
            "[setting.journal]\npath = \"A\"\n[setting.advisor]\nmodel = \"gpt4\"\n",
        );

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("journal")).unwrap();

        assert!(
            result.resolved.contains_key("journal"),
            "should include 'journal'"
        );
        assert!(
            !result.resolved.contains_key("advisor"),
            "should NOT include 'advisor' when filtering for 'journal'"
        );
    }

    #[test]
    fn resolve_setting_missing_field_absent() {
        // Fields with no source in any layer must be absent from output.
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        // Neither layer has 'port' for journal.
        write_config(global_tmp.path(), "[setting.journal]\npath = \"A\"\n");

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        let resolved = result.resolved.get("journal").unwrap();
        assert!(
            !resolved.contains_key("port"),
            "'port' should be absent when no layer defines it"
        );
        let sources = result.sources.get("journal").unwrap();
        assert!(
            !sources.contains_key("port"),
            "'port' should be absent from sources too"
        );
    }

    #[test]
    fn resolve_setting_returns_resolved_and_sources_both() {
        // Crux-card #3 enforcement: both `resolved` and `sources` must be
        // populated in a single call.
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(global_tmp.path(), "[setting.journal]\npath = \"A\"\n");

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        // Both maps must be populated.
        assert!(
            !result.resolved.is_empty(),
            "resolved must be populated in a single call"
        );
        assert!(
            !result.sources.is_empty(),
            "sources must be populated in the same call"
        );

        // The keys must be the same.
        assert_eq!(
            result.resolved.keys().collect::<Vec<_>>(),
            result.sources.keys().collect::<Vec<_>>(),
            "resolved and sources must have matching target keys"
        );
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn resolve_setting_invalid_target_returns_err() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        let app_dir = make_app_dir(global_tmp.path());

        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("Bad-Target"));
        assert!(
            matches!(result, Err(SettingError::InvalidTarget { .. })),
            "expected InvalidTarget error"
        );
    }

    #[test]
    fn resolve_setting_local_toml_wins_over_base() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();

        write_config(global_tmp.path(), "");
        write_alc_toml(project_tmp.path(), "[setting.journal]\npath = \"base\"\n");
        write_local_alc_toml(project_tmp.path(), "[setting.journal]\npath = \"local\"\n");

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), None).unwrap();

        let resolved = result.resolved.get("journal").unwrap();
        assert_eq!(
            resolved.get("path"),
            Some(&serde_json::json!("local")),
            "alc.local.toml should win over alc.toml"
        );
        let sources = result.sources.get("journal").unwrap();
        assert_eq!(sources.get("path"), Some(&"project".to_string()));
    }

    #[test]
    fn resolve_setting_no_project_root_uses_global_only() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();

        write_config(global_tmp.path(), "[setting.journal]\npath = \"G\"\n");

        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, None, None).unwrap();

        let resolved = result.resolved.get("journal").unwrap();
        assert_eq!(resolved.get("path"), Some(&serde_json::json!("G")));
        let sources = result.sources.get("journal").unwrap();
        assert_eq!(sources.get("path"), Some(&"global".to_string()));
    }

    // ── [setting.card].run cascade tests (Phase 1-A) ─────────────────

    #[test]
    fn card_run_env_wins() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_CARD_RUN", "true");
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "[setting.card]\nrun = false\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(result.get_bool("card", "run"), Some(true));
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"env".to_string())
        );
    }

    #[test]
    fn card_run_local_wins_over_alc() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "");
        write_alc_toml(project_tmp.path(), "[setting.card]\nrun = false\n");
        write_local_alc_toml(project_tmp.path(), "[setting.card]\nrun = true\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(result.get_bool("card", "run"), Some(true));
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"project".to_string())
        );
    }

    #[test]
    fn card_run_alc_wins_over_global() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "[setting.card]\nrun = false\n");
        write_alc_toml(project_tmp.path(), "[setting.card]\nrun = true\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(result.get_bool("card", "run"), Some(true));
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"project".to_string())
        );
    }

    #[test]
    fn card_run_global_only() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "[setting.card]\nrun = true\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(result.get_bool("card", "run"), Some(true));
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"global".to_string())
        );
    }

    #[test]
    fn card_run_absent_returns_none() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(result.get_bool("card", "run"), None);
        // Caller decides default:
        assert!(!result.get_bool("card", "run").unwrap_or(false));
    }

    #[test]
    fn card_run_non_bool_value_returns_none() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_config(global_tmp.path(), "[setting.card]\nrun = \"not_a_bool\"\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        // resolve_setting succeeds (schemaless), but get_bool downgrades non-bool to None.
        assert!(result.resolved.contains_key("card"));
        assert_eq!(result.get_bool("card", "run"), None);
    }

    /// Regression test carried from Phase 1-A code review pass (CG-1).
    ///
    /// Confirms that `ALC_SETTING_CARD_RUN=false` produces `Some(false)`
    /// (not `None` and not `Some(true)`) when no file layer contributes
    /// the field, so downstream callers can distinguish "explicitly off"
    /// from "not configured".
    #[test]
    fn card_run_env_false_produces_some_false() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_CARD_RUN", "false");
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        // No file layer at all — env is the sole source.
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(
            result.get_bool("card", "run"),
            Some(false),
            "env value 'false' must parse as Some(false), not None or Some(true)"
        );
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"env".to_string())
        );
        assert!(!result.get_bool("card", "run").unwrap_or(false));
    }

    /// Regression test carried from Phase 1-A code review pass (CG-2).
    ///
    /// Confirms env layer wins over `alc.local.toml` — a `true` env
    /// override defeats a `false` local file, and the winning layer is
    /// labelled `"env"` rather than `"project"`.
    #[test]
    fn card_run_env_wins_over_local() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("ALC_SETTING_CARD_RUN", "true");
        let global_tmp = tempfile::tempdir().unwrap();
        let project_tmp = tempfile::tempdir().unwrap();
        write_local_alc_toml(project_tmp.path(), "[setting.card]\nrun = false\n");
        let app_dir = make_app_dir(global_tmp.path());
        let result = resolve_setting(&app_dir, Some(project_tmp.path()), Some("card")).unwrap();
        assert_eq!(
            result.get_bool("card", "run"),
            Some(true),
            "env value 'true' must win over alc.local.toml 'false'"
        );
        assert_eq!(
            result.sources.get("card").unwrap().get("run"),
            Some(&"env".to_string()),
            "source layer for run must be 'env', not 'project'"
        );
    }
}
