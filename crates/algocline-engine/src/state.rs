//! Persistent key-value state backed by JSON files.
//!
//! ## Architecture
//!
//! All state operations go through the [`StateStore`] trait, which
//! abstracts the storage backend.  The default implementation,
//! [`JsonFileStore`], persists each namespace as a JSON file under
//! `~/.algocline/state/{namespace}.json` with atomic writes (tmp +
//! rename).
//!
//! ## Tier 1 — Current API
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `get` | Read a value (returns `None` if absent) |
//! | `set` | Write a value (upsert) |
//! | `delete` | Remove a key (returns whether it existed) |
//! | `keys` | List all keys in a namespace |
//! | `has` | Check existence (cost is backend-dependent) |
//! | `set_nx` | Set-if-not-exists (returns `false` if key already present) |
//! | `incr` | Counter increment — single-process atomic (read-modify-write) |
//!
//! ## Tier 2 — Future Extensions (design notes, not yet implemented)
//!
//! The following operations are planned but **not yet implemented**.
//! The trait is designed to accommodate them without breaking changes.
//! Review this list when adding a new backend.
//!
//! - **TTL**: `set(key, value, opts)` with `opts.ttl_secs`, plus
//!   `ttl(key) -> Option<Duration>` to query remaining time.  Useful
//!   for caching patterns (e.g. Hub index cache, LLM response cache).
//! - **Batch**: `mget(keys) -> Vec<Option<Value>>` and
//!   `mset(pairs) -> Result<()>`.  Reduces I/O round-trips for
//!   file/network backends.
//! - **clear**: Flush all keys in a namespace.  OpenResty's
//!   `flush_all` equivalent.
//!
//! ## Backend Swappability
//!
//! Because the engine interacts with state only through the
//! [`StateStore`] trait, backends can be swapped without changing Lua
//! code.  Planned backends:
//!
//! - `JsonFileStore` (current, default)
//! - In-memory `HashMap` (for tests and short-lived sessions)
//! - SQLite (for larger datasets with indexed queries)
//! - Redis (for distributed / multi-process scenarios)

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

// ═══════════════════════════════════════════════════════════════
// Trait
// ═══════════════════════════════════════════════════════════════

/// Backend-agnostic key-value state store.
///
/// All operations are namespace-scoped.  Implementations must be
/// `Send + Sync` so they can be shared across Lua VMs (e.g. fork).
pub trait StateStore: Send + Sync {
    /// Read a value.  Returns `None` if the key does not exist.
    fn get(&self, ns: &str, key: &str) -> Result<Option<Value>, String>;

    /// Write a value (upsert).
    fn set(&self, ns: &str, key: &str, value: Value) -> Result<(), String>;

    /// Remove a key.  Returns `true` if it existed.
    fn delete(&self, ns: &str, key: &str) -> Result<bool, String>;

    /// List all keys in a namespace.
    fn keys(&self, ns: &str) -> Result<Vec<String>, String>;

    /// Check whether a key exists.
    ///
    /// Whether this is cheaper than `get` + nil check depends on the
    /// backend.  `JsonFileStore` still loads the whole namespace; backends
    /// like Redis or SQLite can answer with an `EXISTS` command.
    fn has(&self, ns: &str, key: &str) -> Result<bool, String>;

    /// Set a value only if the key does **not** already exist.
    /// Returns `true` if the value was written, `false` if the key
    /// was already present.
    ///
    /// **Note:** `JsonFileStore` performs a non-locking load-check-save
    /// cycle.  This is safe within a single process but **not** across
    /// concurrent processes.  Backends with native CAS (Redis `SETNX`,
    /// SQLite transactions) will provide true atomicity.
    fn set_nx(&self, ns: &str, key: &str, value: Value) -> Result<bool, String>;

    /// Counter increment (single-process atomic).
    ///
    /// Adds `delta` to the current numeric value at `key`.  If the key
    /// is missing, initialises it to `default` before adding.  Returns
    /// the new value.
    ///
    /// **Note:** `JsonFileStore` performs a non-locking
    /// read-modify-write.  Safe within one process; use a backend with
    /// native `INCR` (Redis) or transactions (SQLite) for multi-process
    /// safety.
    ///
    /// Uses `f64` internally.  Integer-valued deltas are exact; fractional
    /// deltas may accumulate floating-point rounding errors over many calls.
    ///
    /// Errors if the existing value is not a JSON number.
    fn incr(&self, ns: &str, key: &str, delta: f64, default: f64) -> Result<f64, String>;
}

// ═══════════════════════════════════════════════════════════════
// JsonFileStore — default backend
// ═══════════════════════════════════════════════════════════════

/// JSON-file-backed state store.
///
/// Each namespace is a single JSON file at
/// `~/.algocline/state/{namespace}.json`.  Writes are atomic: the new
/// state is written to a `.tmp` sibling and then renamed.
pub struct JsonFileStore;

impl JsonFileStore {
    fn state_dir() -> Result<PathBuf, String> {
        let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
        let dir = home.join(".algocline").join("state");
        if !dir.exists() {
            fs::create_dir_all(&dir).map_err(|e| format!("Failed to create state dir: {e}"))?;
        }
        Ok(dir)
    }

    fn state_path(ns: &str) -> Result<PathBuf, String> {
        if ns.contains('/')
            || ns.contains('\\')
            || ns.contains("..")
            || ns.contains('\0')
            || ns.is_empty()
        {
            return Err(format!("Invalid namespace: '{ns}'"));
        }
        Ok(Self::state_dir()?.join(format!("{ns}.json")))
    }

    fn load(ns: &str) -> Result<HashMap<String, Value>, String> {
        let path = Self::state_path(ns)?;
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let content =
            fs::read_to_string(&path).map_err(|e| format!("Failed to read state '{ns}': {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse state '{ns}': {e}"))
    }

    fn save(ns: &str, data: &HashMap<String, Value>) -> Result<(), String> {
        let path = Self::state_path(ns)?;
        let tmp = path.with_extension("json.tmp");
        let content = serde_json::to_string_pretty(data)
            .map_err(|e| format!("Failed to serialize state: {e}"))?;
        fs::write(&tmp, &content).map_err(|e| format!("Failed to write state tmp: {e}"))?;
        fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename state file: {e}"))?;
        Ok(())
    }
}

impl StateStore for JsonFileStore {
    fn get(&self, ns: &str, key: &str) -> Result<Option<Value>, String> {
        let state = Self::load(ns)?;
        Ok(state.get(key).cloned())
    }

    fn set(&self, ns: &str, key: &str, value: Value) -> Result<(), String> {
        let mut state = Self::load(ns)?;
        state.insert(key.to_string(), value);
        Self::save(ns, &state)
    }

    fn delete(&self, ns: &str, key: &str) -> Result<bool, String> {
        let mut state = Self::load(ns)?;
        let existed = state.remove(key).is_some();
        if existed {
            Self::save(ns, &state)?;
        }
        Ok(existed)
    }

    fn keys(&self, ns: &str) -> Result<Vec<String>, String> {
        let state = Self::load(ns)?;
        Ok(state.keys().cloned().collect())
    }

    fn has(&self, ns: &str, key: &str) -> Result<bool, String> {
        let state = Self::load(ns)?;
        Ok(state.contains_key(key))
    }

    fn set_nx(&self, ns: &str, key: &str, value: Value) -> Result<bool, String> {
        let mut state = Self::load(ns)?;
        if state.contains_key(key) {
            return Ok(false);
        }
        state.insert(key.to_string(), value);
        Self::save(ns, &state)?;
        Ok(true)
    }

    fn incr(&self, ns: &str, key: &str, delta: f64, default: f64) -> Result<f64, String> {
        let mut state = Self::load(ns)?;
        let current = match state.get(key) {
            Some(v) => v
                .as_f64()
                .ok_or_else(|| format!("incr: value at '{key}' is not a number"))?,
            None => default,
        };
        let new_val = current + delta;
        state.insert(key.to_string(), serde_json::json!(new_val));
        Self::save(ns, &state)?;
        Ok(new_val)
    }
}

// ═══════════════════════════════════════════════════════════════
// Module-level functions — delegate to JsonFileStore singleton
// ═══════════════════════════════════════════════════════════════
//
// These preserve backwards compatibility with existing callers
// (bridge, tests) that use the free-function API.

static STORE: JsonFileStore = JsonFileStore;

pub fn get(ns: &str, key: &str) -> Result<Option<Value>, String> {
    STORE.get(ns, key)
}

pub fn set(ns: &str, key: &str, value: Value) -> Result<(), String> {
    STORE.set(ns, key, value)
}

pub fn delete(ns: &str, key: &str) -> Result<bool, String> {
    STORE.delete(ns, key)
}

pub fn keys(ns: &str) -> Result<Vec<String>, String> {
    STORE.keys(ns)
}

pub fn has(ns: &str, key: &str) -> Result<bool, String> {
    STORE.has(ns, key)
}

pub fn set_nx(ns: &str, key: &str, value: Value) -> Result<bool, String> {
    STORE.set_nx(ns, key, value)
}

pub fn incr(ns: &str, key: &str, delta: f64, default: f64) -> Result<f64, String> {
    STORE.incr(ns, key, delta, default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cleanup(ns: &str) {
        let _ = std::fs::remove_file(JsonFileStore::state_path(ns).unwrap());
    }

    /// RAII guard that removes the namespace file on drop, so cleanup
    /// runs even if the test panics mid-way.
    struct CleanupGuard(&'static str);
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(JsonFileStore::state_path(self.0).unwrap());
        }
    }

    #[test]
    fn roundtrip() {
        let ns = "_test_roundtrip";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        set(ns, "count", serde_json::json!(42)).unwrap();
        set(ns, "name", serde_json::json!("algocline")).unwrap();

        assert_eq!(get(ns, "count").unwrap(), Some(serde_json::json!(42)));
        assert_eq!(
            get(ns, "name").unwrap(),
            Some(serde_json::json!("algocline"))
        );
        assert_eq!(get(ns, "missing").unwrap(), None);

        let k = keys(ns).unwrap();
        assert!(k.contains(&"count".to_string()));
        assert!(k.contains(&"name".to_string()));

        assert!(delete(ns, "count").unwrap());
        assert!(!delete(ns, "count").unwrap());
        assert_eq!(get(ns, "count").unwrap(), None);
    }

    #[test]
    fn invalid_namespace() {
        assert!(JsonFileStore::state_path("../evil").is_err());
        assert!(JsonFileStore::state_path("foo/bar").is_err());
        assert!(JsonFileStore::state_path("foo\\bar").is_err());
        assert!(JsonFileStore::state_path("").is_err());
        assert!(JsonFileStore::state_path("foo\0bar").is_err());
    }

    #[test]
    fn get_nonexistent_namespace_returns_empty() {
        let result = get("_test_nonexistent_ns_12345", "any_key").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn keys_nonexistent_namespace_returns_empty() {
        let result = keys("_test_nonexistent_ns_12345").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn delete_nonexistent_key_returns_false() {
        let ns = "_test_delete_nonexistent";
        cleanup(ns);
        assert!(!delete(ns, "nope").unwrap());
    }

    #[test]
    fn set_overwrites_existing_value() {
        let ns = "_test_overwrite";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        set(ns, "k", serde_json::json!(1)).unwrap();
        set(ns, "k", serde_json::json!(2)).unwrap();
        assert_eq!(get(ns, "k").unwrap(), Some(serde_json::json!(2)));
    }

    #[test]
    fn state_path_valid_namespaces() {
        assert!(JsonFileStore::state_path("default").is_ok());
        assert!(JsonFileStore::state_path("my-app").is_ok());
        assert!(JsonFileStore::state_path("test_123").is_ok());
    }

    // ─── Tier 1: has / set_nx / incr ──────────────────────────

    #[test]
    fn has_returns_existence() {
        let ns = "_test_has";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        assert!(!has(ns, "x").unwrap());
        set(ns, "x", serde_json::json!(1)).unwrap();
        assert!(has(ns, "x").unwrap());
    }

    #[test]
    fn set_nx_only_sets_if_absent() {
        let ns = "_test_set_nx";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        assert!(set_nx(ns, "k", serde_json::json!("first")).unwrap());
        assert!(!set_nx(ns, "k", serde_json::json!("second")).unwrap());
        assert_eq!(
            get(ns, "k").unwrap(),
            Some(serde_json::json!("first")),
            "set_nx should not overwrite"
        );
    }

    #[test]
    fn incr_initialises_and_increments() {
        let ns = "_test_incr";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        // Missing key: initialise from default (0) + delta (1) = 1
        let v = incr(ns, "counter", 1.0, 0.0).unwrap();
        assert!((v - 1.0).abs() < f64::EPSILON);

        // Increment existing
        let v = incr(ns, "counter", 5.0, 0.0).unwrap();
        assert!((v - 6.0).abs() < f64::EPSILON);

        // Negative delta
        let v = incr(ns, "counter", -2.0, 0.0).unwrap();
        assert!((v - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn incr_rejects_non_numeric() {
        let ns = "_test_incr_err";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        set(ns, "s", serde_json::json!("hello")).unwrap();
        let err = incr(ns, "s", 1.0, 0.0).unwrap_err();
        assert!(err.contains("not a number"), "got: {err}");
    }

    #[test]
    fn incr_custom_default() {
        let ns = "_test_incr_default";
        cleanup(ns);
        let _g = CleanupGuard(ns);

        let v = incr(ns, "score", 10.0, 100.0).unwrap();
        assert!((v - 110.0).abs() < f64::EPSILON, "100 + 10 = 110");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Any valid namespace (alphanumeric + hyphen/underscore) round-trips through set/get.
        #[test]
        fn roundtrip_arbitrary_values(
            key in "[a-z]{1,20}",
            val in any::<i64>(),
        ) {
            let ns = "_proptest_rt";
            let json_val = serde_json::json!(val);
            set(ns, &key, json_val.clone()).unwrap();
            let got = get(ns, &key).unwrap();
            prop_assert_eq!(got, Some(json_val));
            let _ = delete(ns, &key);
        }

        /// Path traversal patterns are always rejected.
        #[test]
        fn traversal_always_rejected(
            prefix in "[a-z]{0,5}",
            suffix in "[a-z]{0,5}",
        ) {
            let evil = format!("{prefix}/../{suffix}");
            prop_assert!(JsonFileStore::state_path(&evil).is_err());
        }

        /// state_path rejects NUL bytes anywhere in the namespace.
        #[test]
        fn nul_byte_always_rejected(
            prefix in "[a-z]{0,10}",
            suffix in "[a-z]{0,10}",
        ) {
            let evil = format!("{prefix}\0{suffix}");
            prop_assert!(JsonFileStore::state_path(&evil).is_err());
        }
    }
}
