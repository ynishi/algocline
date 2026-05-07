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
use std::sync::{Arc, Mutex};

use serde_json::Value;

/// Whether a string is a safe path segment for `dispatch_path`.
///
/// Accepts ASCII alphanumerics, `_`, `-`, and `.` (single dots only —
/// path traversal `..` and reserved names `.` are rejected). Empty
/// strings and any other character (slash, backslash, NUL, control
/// chars, whitespace) cause dispatch to fall back to legacy single-file
/// storage.
fn is_safe_segment(s: &str) -> bool {
    if s.is_empty() || s == "." || s == ".." {
        return false;
    }
    if s.contains("..") {
        return false;
    }
    s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

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
    /// **Note:** `JsonFileStore` serialises this operation per namespace
    /// with an in-process `Mutex`, making it safe across concurrent tokio
    /// tasks within the same process.  Cross-process atomicity still
    /// requires a backend with native CAS (Redis `SETNX`, SQLite
    /// transactions).
    fn set_nx(&self, ns: &str, key: &str, value: Value) -> Result<bool, String>;

    /// Counter increment, serialised per namespace within the same process.
    ///
    /// Adds `delta` to the current numeric value at `key`.  If the key
    /// is missing, initialises it to `default` before adding.  Returns
    /// the new value.
    ///
    /// `JsonFileStore` acquires a per-namespace `Mutex` for the full
    /// read-modify-write cycle, preventing lost updates across concurrent
    /// tokio tasks.  For multi-process safety use a backend with native
    /// `INCR` (Redis) or transactions (SQLite).
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
///
/// ## Concurrency
///
/// Per-namespace locks (`std::sync::Mutex`) prevent lost updates under
/// concurrent `alc.state.*` calls within the same process.  The lock
/// is acquired for the full load → mutate → atomic-rename cycle, so
/// two tokio tasks operating on the **same** namespace are serialised.
///
/// Rationale for `std::sync::Mutex` over `tokio::sync::Mutex`:
/// all I/O inside the lock uses `std::fs` (synchronous, no `.await`),
/// so a standard mutex is sufficient and avoids holding a tokio mutex
/// across potential scheduler context switches.
///
/// **Multi-process safety is NOT provided.**  If multiple `alc`
/// processes share the same state directory (uncommon), use a backend
/// with native `INCR` (Redis) or transactions (SQLite).
pub struct JsonFileStore {
    root: PathBuf,
    /// Per-namespace locks.  Keyed by the resolved JSON file path so
    /// that namespace validation is already applied before lookup.
    locks: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

impl JsonFileStore {
    /// Construct a store rooted at an explicit path.
    ///
    /// The directory is **not** created eagerly; it is created lazily
    /// on the first `set` / `set_nx` / `incr` call via [`Self::state_path`].
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// Acquire (or create) the per-namespace lock and return a guard.
    ///
    /// The returned `std::sync::MutexGuard` keeps the namespace lock
    /// held for the duration of the caller's load → mutate → save cycle.
    /// The outer `locks` map is released immediately after the inner
    /// `Arc` is cloned, so contention on unrelated namespaces is zero.
    fn ns_lock(&self, path: &Path) -> Result<Arc<Mutex<()>>, String> {
        let mut map = self
            .locks
            .lock()
            .map_err(|_| "state: locks map poisoned".to_string())?;
        Ok(Arc::clone(
            map.entry(path.to_path_buf())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        ))
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

    /// Resolve the per-key dispatched file path for a `{prefix}:{id}`
    /// shaped key, returning `None` when the key does not match the
    /// dispatch contract (no `:`, multiple `:`, or unsafe characters
    /// in either segment).
    ///
    /// Dispatched layout (issue #1776868812):
    ///   `{root}/{prefix}/{id}.json` — file contents = the value as
    ///   raw JSON (no wrapper map). Each `flow.state_save(state)` call
    ///   becomes a single per-task file rather than another entry
    ///   crammed into `default.json`.
    ///
    /// Legacy layout: keys without `:` (or with unsafe characters)
    /// continue writing into `{ns}.json` so existing behaviour is
    /// preserved without migration.
    fn dispatch_path(&self, key: &str) -> Result<Option<PathBuf>, String> {
        let (prefix, id) = match key.split_once(':') {
            Some(pair) => pair,
            None => return Ok(None),
        };
        if !is_safe_segment(prefix) || !is_safe_segment(id) {
            return Ok(None);
        }
        let dir = self.ensure_root()?;
        Ok(Some(dir.join(prefix).join(format!("{id}.json"))))
    }

    /// Read a dispatched value file and deserialize it as JSON.
    /// Returns `Ok(None)` when the file does not exist.
    fn load_dispatched(&self, path: &Path) -> Result<Option<Value>, String> {
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read dispatched state '{}': {e}", path.display()))?;
        let v: Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse dispatched state '{}': {e}", path.display()))?;
        Ok(Some(v))
    }

    /// Atomically write a value to a dispatched file (tmp + rename).
    /// Creates the prefix subdirectory if missing.
    fn save_dispatched(&self, path: &Path, value: &Value) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create dispatched state dir '{}': {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let tmp = path.with_extension("json.tmp");
        let content = serde_json::to_string_pretty(value)
            .map_err(|e| format!("Failed to serialize dispatched state: {e}"))?;
        fs::write(&tmp, &content)
            .map_err(|e| format!("Failed to write dispatched state tmp: {e}"))?;
        fs::rename(&tmp, path)
            .map_err(|e| format!("Failed to rename dispatched state file: {e}"))?;
        Ok(())
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
        // Dispatched path takes precedence — when present, that file is
        // the canonical source. Falls back to the legacy `{ns}.json`
        // store so existing entries written before dispatch was enabled
        // remain readable without migration.
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            if let Some(v) = self.load_dispatched(&dpath)? {
                return Ok(Some(v));
            }
            // Fall through to legacy lookup so pre-dispatch values remain
            // visible until the next set() promotes them.
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let state = self.load(ns)?;
        Ok(state.get(key).cloned())
    }

    fn set(&self, ns: &str, key: &str, value: Value) -> Result<(), String> {
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            return self.save_dispatched(&dpath, &value);
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let mut state = self.load(ns)?;
        state.insert(key.to_string(), value);
        self.save(ns, &state)
    }

    fn delete(&self, ns: &str, key: &str) -> Result<bool, String> {
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            if dpath.exists() {
                fs::remove_file(&dpath).map_err(|e| {
                    format!(
                        "Failed to delete dispatched state '{}': {e}",
                        dpath.display()
                    )
                })?;
                return Ok(true);
            }
            // Fall through to legacy delete in case the entry only exists
            // in the legacy single-file store.
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let mut state = self.load(ns)?;
        let existed = state.remove(key).is_some();
        if existed {
            self.save(ns, &state)?;
        }
        Ok(existed)
    }

    fn keys(&self, ns: &str) -> Result<Vec<String>, String> {
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let state = self.load(ns)?;
        Ok(state.keys().cloned().collect())
    }

    fn has(&self, ns: &str, key: &str) -> Result<bool, String> {
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            if dpath.exists() {
                return Ok(true);
            }
            // Fall through to legacy check.
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let state = self.load(ns)?;
        Ok(state.contains_key(key))
    }

    fn set_nx(&self, ns: &str, key: &str, value: Value) -> Result<bool, String> {
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            if dpath.exists() {
                return Ok(false);
            }
            // Also honour any legacy entry to preserve set_nx semantics
            // across the migration boundary.
            let path = self.state_path(ns)?;
            if path.exists() {
                let state = self.load(ns)?;
                if state.contains_key(key) {
                    return Ok(false);
                }
            }
            self.save_dispatched(&dpath, &value)?;
            return Ok(true);
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
        let mut state = self.load(ns)?;
        if state.contains_key(key) {
            return Ok(false);
        }
        state.insert(key.to_string(), value);
        self.save(ns, &state)?;
        Ok(true)
    }

    fn incr(&self, ns: &str, key: &str, delta: f64, default: f64) -> Result<f64, String> {
        if let Some(dpath) = self.dispatch_path(key)? {
            let lock = self.ns_lock(&dpath)?;
            let _guard = lock
                .lock()
                .map_err(|_| format!("state: dispatch lock poisoned for key '{key}'"))?;
            let current = if let Some(v) = self.load_dispatched(&dpath)? {
                v.as_f64()
                    .ok_or_else(|| format!("incr: value at '{key}' is not a number"))?
            } else {
                // Fall back to any legacy value so incr stays monotonic
                // across the dispatch transition.
                let path = self.state_path(ns)?;
                if path.exists() {
                    let state = self.load(ns)?;
                    match state.get(key) {
                        Some(v) => v
                            .as_f64()
                            .ok_or_else(|| format!("incr: value at '{key}' is not a number"))?,
                        None => default,
                    }
                } else {
                    default
                }
            };
            let new_val = current + delta;
            self.save_dispatched(&dpath, &serde_json::json!(new_val))?;
            return Ok(new_val);
        }
        let path = self.state_path(ns)?;
        let lock = self.ns_lock(&path)?;
        let _guard = lock
            .lock()
            .map_err(|_| format!("state: lock poisoned for ns '{ns}'"))?;
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

    // ─── Per-key dispatch (issue #1776868812) ─────────────────────────

    /// Keys shaped `{prefix}:{id}` with safe segments are written to
    /// `{root}/{prefix}/{id}.json` rather than crammed into the legacy
    /// `{ns}.json` SSoT.
    #[test]
    fn dispatch_writes_to_per_key_file_for_prefix_id_keys() {
        let (store, tmp) = new_store();
        store
            .set(
                "default",
                "flow_orch:abc-123",
                serde_json::json!({"step": 1}),
            )
            .unwrap();
        let dispatched = tmp.path().join("flow_orch").join("abc-123.json");
        assert!(
            dispatched.exists(),
            "dispatched file must exist at {}",
            dispatched.display()
        );
        // Legacy file must NOT have been touched for this key.
        let legacy = tmp.path().join("default.json");
        assert!(
            !legacy.exists(),
            "legacy default.json must not be created for dispatched keys"
        );
    }

    /// Read path: dispatched file takes precedence; legacy `{ns}.json`
    /// is consulted only when the dispatched file is absent (so
    /// pre-dispatch entries remain readable without migration).
    #[test]
    fn dispatch_read_falls_back_to_legacy_for_unmigrated_entries() {
        let (store, tmp) = new_store();
        // Pre-populate the legacy default.json by writing a key without
        // a `:` (forces the legacy path) then manually inject the
        // dispatched-shaped key into the same file to simulate a state
        // produced before dispatch was enabled.
        store
            .set("default", "boot_marker", serde_json::json!(true))
            .unwrap();
        let legacy_path = tmp.path().join("default.json");
        let mut existing: HashMap<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(&legacy_path).unwrap()).unwrap();
        existing.insert(
            "flow_legacy:xyz".to_string(),
            serde_json::json!({"old": "value"}),
        );
        std::fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&existing).unwrap(),
        )
        .unwrap();

        // Read returns the legacy value because no dispatched file exists.
        assert_eq!(
            store.get("default", "flow_legacy:xyz").unwrap(),
            Some(serde_json::json!({"old": "value"})),
            "must fall back to legacy default.json when dispatched file absent"
        );

        // Once we set a new value, it lands in the dispatched file and
        // future reads see the new value (legacy entry is shadowed).
        store
            .set(
                "default",
                "flow_legacy:xyz",
                serde_json::json!({"new": "promoted"}),
            )
            .unwrap();
        assert!(
            tmp.path().join("flow_legacy").join("xyz.json").exists(),
            "set() must promote dispatched-shaped keys to per-key file"
        );
        assert_eq!(
            store.get("default", "flow_legacy:xyz").unwrap(),
            Some(serde_json::json!({"new": "promoted"})),
            "dispatched file must shadow legacy entry on subsequent reads"
        );
    }

    /// Keys without a `:` separator (or with unsafe characters in either
    /// segment) bypass dispatch and use the legacy single-file store.
    #[test]
    fn dispatch_skips_keys_without_colon_or_with_unsafe_segments() {
        let (store, tmp) = new_store();
        store
            .set("default", "no_colon", serde_json::json!(1))
            .unwrap();
        store
            .set("default", "bad/prefix:id", serde_json::json!(2))
            .unwrap();
        store
            .set("default", "prefix:bad/id", serde_json::json!(3))
            .unwrap();
        store
            .set("default", "prefix:..", serde_json::json!(4))
            .unwrap();
        // All four go to legacy default.json.
        let legacy = tmp.path().join("default.json");
        let raw: HashMap<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(&legacy).unwrap()).unwrap();
        assert_eq!(raw.get("no_colon"), Some(&serde_json::json!(1)));
        assert_eq!(raw.get("bad/prefix:id"), Some(&serde_json::json!(2)));
        assert_eq!(raw.get("prefix:bad/id"), Some(&serde_json::json!(3)));
        assert_eq!(raw.get("prefix:.."), Some(&serde_json::json!(4)));
        // No subdirectories were created.
        assert!(!tmp.path().join("bad").exists());
        assert!(!tmp.path().join("prefix").exists());
    }

    /// `delete` removes the dispatched file and returns `true`.
    #[test]
    fn dispatch_delete_removes_per_key_file() {
        let (store, tmp) = new_store();
        store.set("default", "p:q", serde_json::json!("v")).unwrap();
        let dispatched = tmp.path().join("p").join("q.json");
        assert!(
            dispatched.exists(),
            "dispatched file should exist before delete"
        );
        assert!(store.delete("default", "p:q").unwrap());
        assert!(
            !dispatched.exists(),
            "dispatched file should be removed after delete"
        );
        // Re-deleting returns false.
        assert!(!store.delete("default", "p:q").unwrap());
    }

    /// `has` reflects dispatched file existence.
    #[test]
    fn dispatch_has_reports_dispatched_file_existence() {
        let (store, _tmp) = new_store();
        assert!(!store.has("default", "p:q").unwrap());
        store.set("default", "p:q", serde_json::json!("v")).unwrap();
        assert!(store.has("default", "p:q").unwrap());
    }

    /// `set_nx` honours both the dispatched file and any legacy entry
    /// to keep set-if-not-exists semantics consistent across migration.
    #[test]
    fn dispatch_set_nx_blocks_when_legacy_or_dispatched_entry_exists() {
        let (store, tmp) = new_store();
        // Inject a legacy entry under the dispatched-shaped key.
        store
            .set("default", "boot", serde_json::json!(true))
            .unwrap();
        let legacy_path = tmp.path().join("default.json");
        let mut existing: HashMap<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(&legacy_path).unwrap()).unwrap();
        existing.insert("p:q".to_string(), serde_json::json!("legacy_only"));
        std::fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&existing).unwrap(),
        )
        .unwrap();
        // set_nx must refuse because the legacy entry exists.
        assert!(!store
            .set_nx("default", "p:q", serde_json::json!("new"))
            .unwrap());

        // For a fresh dispatched-shaped key with no legacy entry, set_nx
        // creates the dispatched file and returns true; second call
        // returns false because the dispatched file now exists.
        assert!(store
            .set_nx("default", "p:r", serde_json::json!("first"))
            .unwrap());
        assert!(tmp.path().join("p").join("r.json").exists());
        assert!(!store
            .set_nx("default", "p:r", serde_json::json!("second"))
            .unwrap());
    }

    /// `incr` operates on the dispatched file when the key matches the
    /// dispatch pattern; legacy values are migrated forward on the
    /// first call.
    #[test]
    fn dispatch_incr_promotes_legacy_value_on_first_call() {
        let (store, tmp) = new_store();
        // Pre-populate a legacy numeric value under a dispatched-shaped key.
        store.set("default", "seed", serde_json::json!(0)).unwrap();
        let legacy_path = tmp.path().join("default.json");
        let mut existing: HashMap<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(&legacy_path).unwrap()).unwrap();
        existing.insert("counter:cnt".to_string(), serde_json::json!(7));
        std::fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&existing).unwrap(),
        )
        .unwrap();

        // First incr: reads legacy value (7), writes new value (10) to
        // the dispatched file.
        let result = store.incr("default", "counter:cnt", 3.0, 0.0).unwrap();
        assert_eq!(result, 10.0);
        let dispatched = tmp.path().join("counter").join("cnt.json");
        assert!(dispatched.exists(), "dispatched file must be created");

        // Second incr: reads dispatched (10), writes 12.
        let result2 = store.incr("default", "counter:cnt", 2.0, 0.0).unwrap();
        assert_eq!(result2, 12.0);
    }

    /// `is_safe_segment` accepts alphanumerics + `_-.` and rejects
    /// path traversal sequences and reserved names.
    #[test]
    fn is_safe_segment_validates_path_safety() {
        assert!(is_safe_segment("flow_orch"));
        assert!(is_safe_segment("abc-123"));
        assert!(is_safe_segment("v1.2.3"));
        assert!(!is_safe_segment(""));
        assert!(!is_safe_segment("."));
        assert!(!is_safe_segment(".."));
        assert!(!is_safe_segment("a..b"));
        assert!(!is_safe_segment("a/b"));
        assert!(!is_safe_segment("a\\b"));
        assert!(!is_safe_segment("a b"));
        assert!(!is_safe_segment("a\0b"));
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
