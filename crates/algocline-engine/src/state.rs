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

// ═══════════════════════════════════════════════════════════════
// Typed error for dispatched-layout operations
// ═══════════════════════════════════════════════════════════════

/// Errors returned by the dispatched-layout helpers
/// (`list_dispatched`, `show_dispatched`, `reset_dispatched_with_backup`).
///
/// Unlike the legacy [`StateStore`] trait methods (which return `String`
/// errors), these helpers use a typed enum so callers can distinguish
/// missing-key from I/O failure at the type level without pattern-matching
/// on OS error codes.
#[derive(thiserror::Error, Debug)]
pub enum StateError {
    /// The requested key does not exist in the given namespace.
    ///
    /// # Arguments
    /// * `namespace` — the namespace that was searched
    /// * `key` — the key that was not found
    #[error("state: key '{key}' not found in namespace '{namespace}'")]
    KeyNotFound { namespace: String, key: String },

    /// A namespace or key segment failed the path-safety check.
    ///
    /// # Arguments
    /// * `which` — either `"namespace"` or `"key"`
    /// * `value` — the offending segment value
    #[error("state: unsafe {which} segment '{value}'")]
    UnsafeSegment { which: &'static str, value: String },

    /// A backup I/O operation (`fs::copy` to `.bak`) failed.
    ///
    /// Wraps the underlying [`std::io::Error`].  Kept separate from
    /// [`StateError::IoWrite`] so callers know that the live file was
    /// not yet touched when this error occurs.
    #[error("state: backup I/O failed: {0}")]
    IoBackup(#[source] std::io::Error),

    /// A read or directory-scan operation failed.
    ///
    /// Wraps the underlying [`std::io::Error`].  Covers `fs::read_to_string`,
    /// `fs::read_dir`, and `DirEntry` iteration.
    #[error("state: read failed: {0}")]
    IoRead(#[source] std::io::Error),

    /// A write or rename operation on the live file or its `.tmp`
    /// staging copy failed.
    ///
    /// Wraps the underlying [`std::io::Error`].  Covers `fs::write` and
    /// `fs::rename`.
    #[error("state: write failed: {0}")]
    IoWrite(#[source] std::io::Error),

    /// JSON serialization or deserialization failed.
    ///
    /// Uses `#[from]` so `?` auto-converts `serde_json::Error`.
    #[error("state: serialize/parse failed: {0}")]
    Serde(#[from] serde_json::Error),

    /// The stored JSON does not have the expected shape.
    ///
    /// # Arguments
    /// * `reason` — human-readable description of the shape violation
    ///   (e.g. `"missing 'data' top-level field"` or
    ///   `"data.completed_steps must be an array"`)
    #[error("state: shape invalid: {reason}")]
    ShapeInvalid { reason: String },
}

// ═══════════════════════════════════════════════════════════════
// ResetReport — return value of reset_dispatched_with_backup
// ═══════════════════════════════════════════════════════════════

/// Report returned by [`JsonFileStore::reset_dispatched_with_backup`]
/// describing what was modified.
#[derive(Debug, Clone)]
pub struct ResetReport {
    /// Path to the `.bak` snapshot created before the mutation.
    pub backup_path: PathBuf,
    /// Number of entries removed from `data.completed_steps`.
    pub steps_removed: usize,
    /// Number of keys deleted from the `data` top-level object.
    pub fields_removed: usize,
}

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

    // ─── Dispatched-layout helpers ─────────────────────────────────────────

    /// List all keys in the dispatched layout for a namespace.
    ///
    /// Enumerates `{root}/{namespace}/*.json` and returns the file names
    /// stripped of the `.json` extension, sorted lexicographically.
    /// Files with extensions other than `.json`, and `.bak` / `.tmp`
    /// siblings, are excluded.  If the namespace directory does not exist
    /// the result is an empty `Vec` (namespace-absent ≡ zero keys).
    ///
    /// # Arguments
    /// * `namespace` — the directory name under `root`; must pass
    ///   [`is_safe_segment`] validation
    ///
    /// # Returns
    /// A sorted list of key strings, or a [`StateError`] on I/O / validation
    /// failure.
    ///
    /// # Errors
    /// * [`StateError::UnsafeSegment`] if `namespace` fails path-safety check
    /// * [`StateError::IoRead`] if reading the directory fails
    pub fn list_dispatched(&self, namespace: &str) -> Result<Vec<String>, StateError> {
        if !is_safe_segment(namespace) {
            return Err(StateError::UnsafeSegment {
                which: "namespace",
                value: namespace.to_string(),
            });
        }
        let ns_dir = self.root.join(namespace);
        if !ns_dir.exists() {
            return Ok(Vec::new());
        }
        let mut keys = Vec::new();
        let entries = fs::read_dir(&ns_dir).map_err(StateError::IoRead)?;
        for entry in entries {
            let entry = entry.map_err(StateError::IoRead)?;
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            // Only include plain `.json` files; skip `.bak`, `.tmp`, and others.
            if !fname_str.ends_with(".json")
                || fname_str.ends_with(".json.bak")
                || fname_str.ends_with(".json.tmp")
            {
                continue;
            }
            // Strip the `.json` suffix to recover the key name.
            let key = fname_str
                .strip_suffix(".json")
                .unwrap_or(&fname_str)
                .to_string();
            keys.push(key);
        }
        keys.sort();
        Ok(keys)
    }

    /// Read the full JSON value for a dispatched-layout key.
    ///
    /// Reads `{root}/{namespace}/{key}.json` and deserializes it.
    ///
    /// # Arguments
    /// * `namespace` — the subdirectory name; must pass [`is_safe_segment`]
    /// * `key` — the file stem; must pass [`is_safe_segment`]
    ///
    /// # Returns
    /// The deserialized [`serde_json::Value`] on success.
    ///
    /// # Errors
    /// * [`StateError::UnsafeSegment`] if either segment fails path-safety check
    /// * [`StateError::KeyNotFound`] if the file does not exist
    /// * [`StateError::IoRead`] if the file cannot be read
    /// * [`StateError::Serde`] if the content is not valid JSON
    pub fn show_dispatched(
        &self,
        namespace: &str,
        key: &str,
    ) -> Result<serde_json::Value, StateError> {
        if !is_safe_segment(namespace) {
            return Err(StateError::UnsafeSegment {
                which: "namespace",
                value: namespace.to_string(),
            });
        }
        if !is_safe_segment(key) {
            return Err(StateError::UnsafeSegment {
                which: "key",
                value: key.to_string(),
            });
        }
        let target = self.root.join(namespace).join(format!("{key}.json"));
        if !target.exists() {
            return Err(StateError::KeyNotFound {
                namespace: namespace.to_string(),
                key: key.to_string(),
            });
        }
        let content = fs::read_to_string(&target).map_err(StateError::IoRead)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        Ok(value)
    }

    /// Atomically reset a dispatched-layout state file with a backup.
    ///
    /// Performs the following sequence in order (Crux atomicity contract):
    ///
    /// 1. Validate `namespace` and `key` with [`is_safe_segment`].
    /// 2. Compute target path: `{root}/{namespace}/{key}.json`.
    /// 3. Return [`StateError::KeyNotFound`] if the file does not exist.
    /// 4. Acquire the per-path mutex via [`Self::ns_lock`]; hold until rename.
    /// 5. Copy the live file to `{root}/{namespace}/{key}.json.bak` — the
    ///    live file is **not** touched before this point.
    /// 6. Load and parse the live file.
    /// 7. Apply in-memory mutations:
    ///    - Remove each element of `steps` from `data.completed_steps` (if
    ///      the array exists).
    ///    - Delete each element of `fields` from the `data` top-level object.
    ///    - If the top-level `data` field is absent or not an object, return
    ///      [`StateError::ShapeInvalid`].
    /// 8. Write the mutated value to `{target}.tmp`.
    /// 9. Rename `.tmp` → target (POSIX atomic on same filesystem).
    ///
    /// A crash between steps 5 and 9 leaves the `.bak` intact and the live
    /// file unmodified (or only partially written to `.tmp`), so the original
    /// state is always recoverable.
    ///
    /// # Arguments
    /// * `namespace` — subdirectory name; must pass [`is_safe_segment`]
    /// * `key` — file stem; must pass [`is_safe_segment`]
    /// * `steps` — step names to remove from `data.completed_steps`
    /// * `fields` — field names to delete from the `data` top-level object
    ///
    /// # Returns
    /// A [`ResetReport`] with the backup path and counts of removed items.
    ///
    /// # Errors
    /// * [`StateError::UnsafeSegment`] if either segment fails path-safety check
    /// * [`StateError::KeyNotFound`] if the file does not exist
    /// * [`StateError::ShapeInvalid`] if the lock is poisoned or the JSON
    ///   structure is not a `{data: {...}}` object
    /// * [`StateError::IoBackup`] if the `.bak` copy fails
    /// * [`StateError::IoRead`] if loading the live file fails
    /// * [`StateError::IoWrite`] if the `.tmp` write or rename fails
    /// * [`StateError::Serde`] if the file content is not valid JSON
    pub fn reset_dispatched_with_backup(
        &self,
        namespace: &str,
        key: &str,
        steps: &[String],
        fields: &[String],
    ) -> Result<ResetReport, StateError> {
        // (a) Validate path segments.
        if !is_safe_segment(namespace) {
            return Err(StateError::UnsafeSegment {
                which: "namespace",
                value: namespace.to_string(),
            });
        }
        if !is_safe_segment(key) {
            return Err(StateError::UnsafeSegment {
                which: "key",
                value: key.to_string(),
            });
        }

        // (b) Compute target path.
        let target = self.root.join(namespace).join(format!("{key}.json"));

        // (c) Return KeyNotFound if the file does not exist.
        if !target.exists() {
            return Err(StateError::KeyNotFound {
                namespace: namespace.to_string(),
                key: key.to_string(),
            });
        }

        // (c.5) Acquire the per-path mutex and hold it until after rename.
        let lock = self
            .ns_lock(&target)
            .map_err(|s| StateError::ShapeInvalid { reason: s })?;
        let _guard = lock.lock().map_err(|_| StateError::ShapeInvalid {
            reason: "lock poisoned".to_string(),
        })?;

        // (d) Create .bak backup — live file is not touched before this.
        let bak_path = target.with_extension("json.bak");
        fs::copy(&target, &bak_path).map_err(StateError::IoBackup)?;

        // (e) Load and parse the live file.
        let content = fs::read_to_string(&target).map_err(StateError::IoRead)?;
        let mut value: serde_json::Value = serde_json::from_str(&content)?;

        // (f) Apply in-memory mutations.
        let data = value
            .get_mut("data")
            .ok_or_else(|| StateError::ShapeInvalid {
                reason: "missing 'data' top-level field".to_string(),
            })?;
        let data_obj = data
            .as_object_mut()
            .ok_or_else(|| StateError::ShapeInvalid {
                reason: "'data' top-level field must be an object".to_string(),
            })?;

        // Remove matching entries from data.completed_steps.
        let mut steps_removed = 0usize;
        if !steps.is_empty() {
            if let Some(cs) = data_obj.get_mut("completed_steps") {
                if let Some(arr) = cs.as_array_mut() {
                    let before = arr.len();
                    arr.retain(|v| {
                        if let Some(s) = v.as_str() {
                            !steps.iter().any(|step| step == s)
                        } else {
                            true
                        }
                    });
                    steps_removed = before - arr.len();
                } else {
                    return Err(StateError::ShapeInvalid {
                        reason: "data.completed_steps must be an array".to_string(),
                    });
                }
            }
            // If completed_steps key is absent, nothing to remove — not an error.
        }

        // Delete requested fields from the data object.
        let mut fields_removed = 0usize;
        for field in fields {
            if data_obj.remove(field.as_str()).is_some() {
                fields_removed += 1;
            }
        }

        // (g) Write mutated value to .tmp staging file.
        let tmp = target.with_extension("json.tmp");
        let serialized = serde_json::to_string_pretty(&value)?;
        fs::write(&tmp, &serialized).map_err(StateError::IoWrite)?;

        // (h) Atomic rename: .tmp → live file.
        fs::rename(&tmp, &target).map_err(StateError::IoWrite)?;

        Ok(ResetReport {
            backup_path: bak_path,
            steps_removed,
            fields_removed,
        })
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

    // ─── Dispatched-layout helpers ─────────────────────────────────────────

    mod dispatched_layout {
        use super::*;

        /// Helper: write a JSON file directly into `{tmp}/{ns}/{key}.json`,
        /// creating the parent directory if needed.
        fn seed(tmp: &TempDir, ns: &str, key: &str, value: serde_json::Value) {
            let dir = tmp.path().join(ns);
            // Safe: test-only helper, directory creation cannot fail in practice
            fs::create_dir_all(&dir).expect("create ns dir");
            let path = dir.join(format!("{key}.json"));
            fs::write(
                path,
                serde_json::to_string_pretty(&value).expect("serialize"),
            )
            .expect("write seed file");
        }

        /// `list_dispatched` returns only `.json` files and strips the suffix.
        #[test]
        fn list_returns_json_keys_only() {
            let (store, tmp) = new_store();
            seed(&tmp, "myns", "alpha", serde_json::json!(1));
            seed(&tmp, "myns", "beta", serde_json::json!(2));
            // Place non-.json and sibling files that must be excluded.
            let ns_dir = tmp.path().join("myns");
            fs::write(ns_dir.join("alpha.json.bak"), b"backup").expect("write bak");
            fs::write(ns_dir.join("alpha.json.tmp"), b"tmp").expect("write tmp");
            fs::write(ns_dir.join("notes.txt"), b"text").expect("write txt");

            let keys = store.list_dispatched("myns").unwrap();
            assert_eq!(
                keys,
                vec!["alpha", "beta"],
                "must be sorted, .bak/.tmp/.txt excluded"
            );
        }

        /// `list_dispatched` returns an empty Vec when the namespace directory
        /// does not exist (no error).
        #[test]
        fn list_returns_empty_for_absent_namespace() {
            let (store, _tmp) = new_store();
            let keys = store.list_dispatched("ghost").unwrap();
            assert!(keys.is_empty(), "absent namespace should return empty Vec");
        }

        /// `list_dispatched` handles a namespace directory that exists but
        /// contains only non-`.json` files.
        #[test]
        fn list_returns_empty_when_only_non_json_files_present() {
            let (store, tmp) = new_store();
            let ns_dir = tmp.path().join("empty_ns");
            // Safe: test setup
            fs::create_dir_all(&ns_dir).expect("create dir");
            fs::write(ns_dir.join("readme.txt"), b"hi").expect("write");
            let keys = store.list_dispatched("empty_ns").unwrap();
            assert!(keys.is_empty());
        }

        /// `show_dispatched` returns `KeyNotFound` when the namespace
        /// directory itself does not exist.
        #[test]
        fn show_returns_key_not_found_for_absent_namespace() {
            let (store, _tmp) = new_store();
            let err = store.show_dispatched("nodir", "anykey").unwrap_err();
            assert!(
                matches!(err, StateError::KeyNotFound { .. }),
                "expected KeyNotFound, got: {err}"
            );
            // Confirm the message contains "not found" as specified by the error format.
            assert!(err.to_string().contains("not found"), "{err}");
        }

        /// `show_dispatched` returns `KeyNotFound` when the namespace
        /// directory exists but the key file is absent.
        #[test]
        fn show_returns_key_not_found_for_absent_key() {
            let (store, tmp) = new_store();
            // Create the namespace directory but not the key file.
            let ns_dir = tmp.path().join("myns2");
            // Safe: test setup
            fs::create_dir_all(&ns_dir).expect("create dir");

            let err = store.show_dispatched("myns2", "missing").unwrap_err();
            assert!(
                matches!(err, StateError::KeyNotFound { .. }),
                "expected KeyNotFound, got: {err}"
            );
        }

        /// `show_dispatched` returns the full JSON value when the key exists.
        #[test]
        fn show_returns_full_value_for_existing_key() {
            let (store, tmp) = new_store();
            let expected = serde_json::json!({"data": {"completed_steps": ["a", "b"], "x": 42}});
            seed(&tmp, "showns", "task1", expected.clone());

            let result = store.show_dispatched("showns", "task1").unwrap();
            assert_eq!(result, expected);
        }
    }

    mod reset_atomicity {
        use super::*;

        /// Helper: write a JSON file directly into `{tmp}/{ns}/{key}.json`.
        fn seed(tmp: &TempDir, ns: &str, key: &str, value: serde_json::Value) {
            let dir = tmp.path().join(ns);
            // Safe: test setup
            fs::create_dir_all(&dir).expect("create ns dir");
            let path = dir.join(format!("{key}.json"));
            fs::write(
                path,
                serde_json::to_string_pretty(&value).expect("serialize"),
            )
            .expect("write seed");
        }

        /// Reset removes specified steps and fields; backup file contains the
        /// original content; report reflects what was removed.
        #[test]
        fn reset_removes_steps_and_fields_and_creates_backup() {
            let (store, tmp) = new_store();
            let original = serde_json::json!({
                "data": {
                    "completed_steps": ["a", "b", "c"],
                    "x": 1,
                    "y": "hello"
                }
            });
            seed(&tmp, "testns", "task1", original.clone());

            let report = store
                .reset_dispatched_with_backup(
                    "testns",
                    "task1",
                    &["b".to_string()],
                    &["x".to_string()],
                )
                .unwrap();

            // Backup must exist and contain original content.
            let bak_path = tmp.path().join("testns").join("task1.json.bak");
            assert!(
                bak_path.exists(),
                ".bak file must exist at {}",
                bak_path.display()
            );
            assert_eq!(report.backup_path, bak_path);
            let bak_content: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&bak_path).expect("read bak"))
                    .expect("parse bak");
            assert_eq!(bak_content, original, ".bak must contain original content");

            // Live file must reflect mutations.
            let live_path = tmp.path().join("testns").join("task1.json");
            let live_content: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&live_path).expect("read live"))
                    .expect("parse live");
            let expected = serde_json::json!({
                "data": {
                    "completed_steps": ["a", "c"],
                    "y": "hello"
                }
            });
            assert_eq!(live_content, expected, "live file must be mutated");

            // Report counts.
            assert_eq!(report.steps_removed, 1, "one step removed");
            assert_eq!(report.fields_removed, 1, "one field removed");
        }

        /// Reset with both steps and fields removed (2-case variant).
        #[test]
        fn reset_removes_multiple_steps_and_fields() {
            let (store, tmp) = new_store();
            let original = serde_json::json!({
                "data": {
                    "completed_steps": ["s1", "s2", "s3", "s4"],
                    "repo_readiness": "NOT_READY",
                    "repo_readiness_report": "details here",
                    "plan_gate_retries": 2
                }
            });
            seed(&tmp, "orchns", "task-abc", original.clone());

            let report = store
                .reset_dispatched_with_backup(
                    "orchns",
                    "task-abc",
                    &["s2".to_string(), "s3".to_string()],
                    &[
                        "repo_readiness".to_string(),
                        "repo_readiness_report".to_string(),
                    ],
                )
                .unwrap();

            let live_path = tmp.path().join("orchns").join("task-abc.json");
            let live: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&live_path).expect("read"))
                    .expect("parse");
            assert_eq!(
                live["data"]["completed_steps"],
                serde_json::json!(["s1", "s4"])
            );
            assert!(live["data"].get("repo_readiness").is_none());
            assert!(live["data"].get("repo_readiness_report").is_none());
            assert_eq!(live["data"]["plan_gate_retries"], 2);

            assert_eq!(report.steps_removed, 2);
            assert_eq!(report.fields_removed, 2);
        }

        /// Reset on a missing key returns `KeyNotFound`.
        #[test]
        fn reset_returns_key_not_found_for_absent_file() {
            let (store, _tmp) = new_store();
            let err = store
                .reset_dispatched_with_backup("ns", "missing", &[], &[])
                .unwrap_err();
            assert!(
                matches!(err, StateError::KeyNotFound { .. }),
                "expected KeyNotFound, got: {err}"
            );
        }

        /// Reset returns `ShapeInvalid` when `data` top-level field is absent.
        #[test]
        fn reset_returns_shape_invalid_when_data_absent() {
            let (store, tmp) = new_store();
            // File has no "data" key.
            let bad = serde_json::json!({"identity": {"task_id": "t1"}});
            let dir = tmp.path().join("badns");
            // Safe: test setup
            fs::create_dir_all(&dir).expect("create dir");
            fs::write(
                dir.join("k.json"),
                serde_json::to_string_pretty(&bad).expect("ser"),
            )
            .expect("write");

            let err = store
                .reset_dispatched_with_backup("badns", "k", &["s".to_string()], &[])
                .unwrap_err();
            assert!(
                matches!(err, StateError::ShapeInvalid { .. }),
                "expected ShapeInvalid, got: {err}"
            );
            assert!(err.to_string().contains("data"), "{err}");
        }

        /// Reset returns `ShapeInvalid` when `data.completed_steps` is not
        /// an array.
        #[test]
        fn reset_returns_shape_invalid_when_completed_steps_not_array() {
            let (store, tmp) = new_store();
            // completed_steps is an object, not an array.
            let bad = serde_json::json!({"data": {"completed_steps": {"step": "a"}}});
            let dir = tmp.path().join("badns2");
            // Safe: test setup
            fs::create_dir_all(&dir).expect("create dir");
            fs::write(
                dir.join("k.json"),
                serde_json::to_string_pretty(&bad).expect("ser"),
            )
            .expect("write");

            let err = store
                .reset_dispatched_with_backup("badns2", "k", &["a".to_string()], &[])
                .unwrap_err();
            assert!(
                matches!(err, StateError::ShapeInvalid { .. }),
                "expected ShapeInvalid, got: {err}"
            );
            assert!(
                err.to_string().contains("completed_steps"),
                "message should mention completed_steps: {err}"
            );
        }
    }

    mod path_traversal {
        use super::*;

        /// `list_dispatched` rejects unsafe namespace segments.
        #[test]
        fn list_rejects_unsafe_namespace() {
            let (store, _tmp) = new_store();
            let err = store.list_dispatched("../evil").unwrap_err();
            assert!(
                matches!(
                    err,
                    StateError::UnsafeSegment {
                        which: "namespace",
                        ..
                    }
                ),
                "expected UnsafeSegment{{namespace}}, got: {err}"
            );
        }

        /// `show_dispatched` rejects an unsafe key segment.
        #[test]
        fn show_rejects_unsafe_key() {
            let (store, _tmp) = new_store();
            let err = store.show_dispatched("ns", "foo/bar").unwrap_err();
            assert!(
                matches!(err, StateError::UnsafeSegment { which: "key", .. }),
                "expected UnsafeSegment{{key}}, got: {err}"
            );
        }

        /// `reset_dispatched_with_backup` rejects an empty namespace segment.
        #[test]
        fn reset_rejects_empty_namespace() {
            let (store, _tmp) = new_store();
            let err = store
                .reset_dispatched_with_backup("", "key", &[], &[])
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    StateError::UnsafeSegment {
                        which: "namespace",
                        ..
                    }
                ),
                "expected UnsafeSegment{{namespace}}, got: {err}"
            );
        }

        /// `reset_dispatched_with_backup` rejects a `..` key segment.
        #[test]
        fn reset_rejects_dotdot_key() {
            let (store, _tmp) = new_store();
            let err = store
                .reset_dispatched_with_backup("ns", "..", &[], &[])
                .unwrap_err();
            assert!(
                matches!(err, StateError::UnsafeSegment { which: "key", .. }),
                "expected UnsafeSegment{{key}}, got: {err}"
            );
        }
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
