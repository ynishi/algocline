//! Persistent key-value state backed by JSON files.
//!
//! Storage: ~/.algocline/state/{namespace}.json
//! Each namespace is a flat JSON object { "key": value, ... }.
//! Writes are atomic (write to .tmp, rename).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Resolve the state directory, creating it if needed.
fn state_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let dir = home.join(".algocline").join("state");
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("Failed to create state dir: {e}"))?;
    }
    Ok(dir)
}

fn state_path(ns: &str) -> Result<PathBuf, String> {
    // Prevent path traversal and NUL byte injection
    if ns.contains('/')
        || ns.contains('\\')
        || ns.contains("..")
        || ns.contains('\0')
        || ns.is_empty()
    {
        return Err(format!("Invalid namespace: '{ns}'"));
    }
    Ok(state_dir()?.join(format!("{ns}.json")))
}

fn load_state(ns: &str) -> Result<HashMap<String, serde_json::Value>, String> {
    let path = state_path(ns)?;
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read state '{ns}': {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse state '{ns}': {e}"))
}

fn save_state(ns: &str, data: &HashMap<String, serde_json::Value>) -> Result<(), String> {
    let path = state_path(ns)?;
    let tmp = path.with_extension("json.tmp");
    let content = serde_json::to_string_pretty(data)
        .map_err(|e| format!("Failed to serialize state: {e}"))?;
    fs::write(&tmp, &content).map_err(|e| format!("Failed to write state tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename state file: {e}"))?;
    Ok(())
}

/// Get a value from persistent state.
pub fn get(ns: &str, key: &str) -> Result<Option<serde_json::Value>, String> {
    let state = load_state(ns)?;
    Ok(state.get(key).cloned())
}

/// Set a value in persistent state (atomic write).
pub fn set(ns: &str, key: &str, value: serde_json::Value) -> Result<(), String> {
    let mut state = load_state(ns)?;
    state.insert(key.to_string(), value);
    save_state(ns, &state)
}

/// List all keys in a namespace.
pub fn keys(ns: &str) -> Result<Vec<String>, String> {
    let state = load_state(ns)?;
    Ok(state.keys().cloned().collect())
}

/// Delete a key from persistent state.
pub fn delete(ns: &str, key: &str) -> Result<bool, String> {
    let mut state = load_state(ns)?;
    let existed = state.remove(key).is_some();
    if existed {
        save_state(ns, &state)?;
    }
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let ns = "_test_roundtrip";
        // Clean up first
        let _ = std::fs::remove_file(state_path(ns).unwrap());

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

        // Clean up
        let _ = std::fs::remove_file(state_path(ns).unwrap());
    }

    #[test]
    fn invalid_namespace() {
        assert!(state_path("../evil").is_err());
        assert!(state_path("foo/bar").is_err());
        assert!(state_path("foo\\bar").is_err());
        assert!(state_path("").is_err());
        assert!(state_path("foo\0bar").is_err());
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
        let _ = std::fs::remove_file(state_path(ns).unwrap());
        assert!(!delete(ns, "nope").unwrap());
    }

    #[test]
    fn set_overwrites_existing_value() {
        let ns = "_test_overwrite";
        let _ = std::fs::remove_file(state_path(ns).unwrap());

        set(ns, "k", serde_json::json!(1)).unwrap();
        set(ns, "k", serde_json::json!(2)).unwrap();
        assert_eq!(get(ns, "k").unwrap(), Some(serde_json::json!(2)));

        let _ = std::fs::remove_file(state_path(ns).unwrap());
    }

    #[test]
    fn state_path_valid_namespaces() {
        assert!(state_path("default").is_ok());
        assert!(state_path("my-app").is_ok());
        assert!(state_path("test_123").is_ok());
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
            prop_assert!(state_path(&evil).is_err());
        }

        /// state_path rejects NUL bytes anywhere in the namespace.
        #[test]
        fn nul_byte_always_rejected(
            prefix in "[a-z]{0,10}",
            suffix in "[a-z]{0,10}",
        ) {
            let evil = format!("{prefix}\0{suffix}");
            prop_assert!(state_path(&evil).is_err());
        }
    }
}
