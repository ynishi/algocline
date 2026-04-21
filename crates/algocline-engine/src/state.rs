//! Persistent key-value state backed by JSON files.
//!
//! ## Architecture
//!
//! All state operations go through the [`StateStore`] trait, which
//! abstracts the storage backend.  The default implementation,
//! [`JsonFileStore`], persists each namespace as a JSON file under a
//! caller-provided root directory with atomic writes (tmp + rename).
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
use std::path::{Path, PathBuf};

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
/// `{root}/{namespace}.json`.  Writes are atomic: the new state is
/// written to a `.tmp` sibling and then renamed.
///
/// The root directory is provided at construction time; callers are
/// expected to resolve it from the service-layer `AppDir` abstraction
/// (typically `~/.algocline/state/`).
pub struct JsonFileStore {
    root: PathBuf,
}

impl JsonFileStore {
    /// Construct a store rooted at an explicit path.
    ///
    /// The directory is **not** created eagerly; it is created lazily
    /// on the first `set` / `set_nx` / `incr` call via [`Self::state_path`].
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Return the root directory this store writes under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Ensure the root directory exists, returning it.
    fn ensure_root(&self) -> Result<&Path, String> {
        if !self.root.exists() {
            fs::create_dir_all(&self.root)
                .map_err(|e| format!("Failed to create state dir: {e}"))?;
        }
        Ok(&self.root)
    }

    /// Resolve the JSON file path for a namespace, validating the name
    /// and creating the root directory on demand.
    pub fn state_path(&self, ns: &str) -> Result<PathBuf, String> {
        if ns.contains('/')
            || ns.contains('\\')
            || ns.contains("..")
            || ns.contains('\0')
            || ns.is_empty()
        {
            return Err(format!("Invalid namespace: '{ns}'"));
        }
        let dir = self.ensure_root()?;
        Ok(dir.join(format!("{ns}.json")))
    }

    fn load(&self, ns: &str) -> Result<HashMap<String, Value>, String> {
        let path = self.state_path(ns)?;
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let content =
            fs::read_to_string(&path).map_err(|e| format!("Failed to read state '{ns}': {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse state '{ns}': {e}"))
    }

    fn save(&self, ns: &str, data: &HashMap<String, Value>) -> Result<(), String> {
        let path = self.state_path(ns)?;
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
        let state = self.load(ns)?;
        Ok(state.get(key).cloned())
    }

    fn set(&self, ns: &str, key: &str, value: Value) -> Result<(), String> {
        let mut state = self.load(ns)?;
        state.insert(key.to_string(), value);
        self.save(ns, &state)
    }

    fn delete(&self, ns: &str, key: &str) -> Result<bool, String> {
        let mut state = self.load(ns)?;
        let existed = state.remove(key).is_some();
        if existed {
            self.save(ns, &state)?;
        }
        Ok(existed)
    }

    fn keys(&self, ns: &str) -> Result<Vec<String>, String> {
        let state = self.load(ns)?;
        Ok(state.keys().cloned().collect())
    }

    fn has(&self, ns: &str, key: &str) -> Result<bool, String> {
        let state = self.load(ns)?;
        Ok(state.contains_key(key))
    }

    fn set_nx(&self, ns: &str, key: &str, value: Value) -> Result<bool, String> {
        let mut state = self.load(ns)?;
        if state.contains_key(key) {
            return Ok(false);
        }
        state.insert(key.to_string(), value);
        self.save(ns, &state)?;
        Ok(true)
    }

    fn incr(&self, ns: &str, key: &str, delta: f64, default: f64) -> Result<f64, String> {
        let mut state = self.load(ns)?;
        let current = match state.get(key) {
            Some(v) => v
                .as_f64()
                .ok_or_else(|| format!("incr: value at '{key}' is not a number"))?,
            None => default,
        };
        let new_val = current + delta;
        state.insert(key.to_string(), serde_json::json!(new_val));
        self.save(ns, &state)?;
        Ok(new_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a JsonFileStore rooted in a fresh tempdir, returning both
    /// so the TempDir guard lives for the test duration.
    fn new_store() -> (JsonFileStore, TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = JsonFileStore::new(tmp.path().to_path_buf());
        (store, tmp)
    }

    #[test]
    fn roundtrip() {
        let (store, _tmp) = new_store();
        let ns = "rt";

        store.set(ns, "count", serde_json::json!(42)).unwrap();
        store
            .set(ns, "name", serde_json::json!("algocline"))
            .unwrap();

        assert_eq!(store.get(ns, "count").unwrap(), Some(serde_json::json!(42)));
        assert_eq!(
            store.get(ns, "name").unwrap(),
            Some(serde_json::json!("algocline"))
        );
        assert_eq!(store.get(ns, "missing").unwrap(), None);

        let k = store.keys(ns).unwrap();
        assert!(k.contains(&"count".to_string()));
        assert!(k.contains(&"name".to_string()));

        assert!(store.delete(ns, "count").unwrap());
        assert!(!store.delete(ns, "count").unwrap());
        assert_eq!(store.get(ns, "count").unwrap(), None);
    }

    #[test]
    fn invalid_namespace() {
        let (store, _tmp) = new_store();
        assert!(store.state_path("../evil").is_err());
        assert!(store.state_path("foo/bar").is_err());
        assert!(store.state_path("foo\\bar").is_err());
        assert!(store.state_path("").is_err());
        assert!(store.state_path("foo\0bar").is_err());
    }

    #[test]
    fn get_nonexistent_namespace_returns_empty() {
        let (store, _tmp) = new_store();
        let result = store.get("ghost_ns", "any_key").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn keys_nonexistent_namespace_returns_empty() {
        let (store, _tmp) = new_store();
        let result = store.keys("ghost_ns").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn delete_nonexistent_key_returns_false() {
        let (store, _tmp) = new_store();
        assert!(!store.delete("delns", "nope").unwrap());
    }

    #[test]
    fn set_overwrites_existing_value() {
        let (store, _tmp) = new_store();
        let ns = "ow";

        store.set(ns, "k", serde_json::json!(1)).unwrap();
        store.set(ns, "k", serde_json::json!(2)).unwrap();
        assert_eq!(store.get(ns, "k").unwrap(), Some(serde_json::json!(2)));
    }

    #[test]
    fn state_path_valid_namespaces() {
        let (store, _tmp) = new_store();
        assert!(store.state_path("default").is_ok());
        assert!(store.state_path("my-app").is_ok());
        assert!(store.state_path("test_123").is_ok());
    }

    // ─── Tier 1: has / set_nx / incr ──────────────────────────

    #[test]
    fn has_returns_existence() {
        let (store, _tmp) = new_store();
        let ns = "hasns";

        assert!(!store.has(ns, "x").unwrap());
        store.set(ns, "x", serde_json::json!(1)).unwrap();
        assert!(store.has(ns, "x").unwrap());
    }

    #[test]
    fn set_nx_only_sets_if_absent() {
        let (store, _tmp) = new_store();
        let ns = "snx";

        assert!(store.set_nx(ns, "k", serde_json::json!("first")).unwrap());
        assert!(!store.set_nx(ns, "k", serde_json::json!("second")).unwrap());
        assert_eq!(
            store.get(ns, "k").unwrap(),
            Some(serde_json::json!("first")),
            "set_nx should not overwrite"
        );
    }

    #[test]
    fn incr_initialises_and_increments() {
        let (store, _tmp) = new_store();
        let ns = "inc";

        // Missing key: initialise from default (0) + delta (1) = 1
        let v = store.incr(ns, "counter", 1.0, 0.0).unwrap();
        assert!((v - 1.0).abs() < f64::EPSILON);

        // Increment existing
        let v = store.incr(ns, "counter", 5.0, 0.0).unwrap();
        assert!((v - 6.0).abs() < f64::EPSILON);

        // Negative delta
        let v = store.incr(ns, "counter", -2.0, 0.0).unwrap();
        assert!((v - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn incr_rejects_non_numeric() {
        let (store, _tmp) = new_store();
        let ns = "incerr";

        store.set(ns, "s", serde_json::json!("hello")).unwrap();
        let err = store.incr(ns, "s", 1.0, 0.0).unwrap_err();
        assert!(err.contains("not a number"), "got: {err}");
    }

    #[test]
    fn incr_custom_default() {
        let (store, _tmp) = new_store();
        let ns = "incdef";

        let v = store.incr(ns, "score", 10.0, 100.0).unwrap();
        assert!((v - 110.0).abs() < f64::EPSILON, "100 + 10 = 110");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn new_store() -> (JsonFileStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = JsonFileStore::new(tmp.path().to_path_buf());
        (store, tmp)
    }

    proptest! {
        /// Any valid namespace (alphanumeric + hyphen/underscore) round-trips through set/get.
        #[test]
        fn roundtrip_arbitrary_values(
            key in "[a-z]{1,20}",
            val in any::<i64>(),
        ) {
            let (store, _tmp) = new_store();
            let ns = "rt";
            let json_val = serde_json::json!(val);
            store.set(ns, &key, json_val.clone()).unwrap();
            let got = store.get(ns, &key).unwrap();
            prop_assert_eq!(got, Some(json_val));
            let _ = store.delete(ns, &key);
        }

        /// Path traversal patterns are always rejected.
        #[test]
        fn traversal_always_rejected(
            prefix in "[a-z]{0,5}",
            suffix in "[a-z]{0,5}",
        ) {
            let (store, _tmp) = new_store();
            let evil = format!("{prefix}/../{suffix}");
            prop_assert!(store.state_path(&evil).is_err());
        }

        /// state_path rejects NUL bytes anywhere in the namespace.
        #[test]
        fn nul_byte_always_rejected(
            prefix in "[a-z]{0,10}",
            suffix in "[a-z]{0,10}",
        ) {
            let (store, _tmp) = new_store();
            let evil = format!("{prefix}\0{suffix}");
            prop_assert!(store.state_path(&evil).is_err());
        }
    }
}
