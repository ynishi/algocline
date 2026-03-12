use std::collections::HashMap;

/// KV store written from Lua via alc.stats.record(key, value).
pub struct CustomMetrics {
    entries: HashMap<String, serde_json::Value>,
}

impl CustomMetrics {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn record(&mut self, key: String, value: serde_json::Value) {
        self.entries.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.entries.get(key)
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.entries).unwrap_or(serde_json::Value::Null)
    }
}

impl Default for CustomMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_and_get() {
        let mut cm = CustomMetrics::new();
        cm.record("key".into(), json!(42));
        assert_eq!(cm.get("key"), Some(&json!(42)));
    }

    #[test]
    fn get_missing_returns_none() {
        let cm = CustomMetrics::new();
        assert_eq!(cm.get("missing"), None);
    }

    #[test]
    fn record_overwrites() {
        let mut cm = CustomMetrics::new();
        cm.record("key".into(), json!(1));
        cm.record("key".into(), json!(2));
        assert_eq!(cm.get("key"), Some(&json!(2)));
    }

    #[test]
    fn to_json_includes_all_entries() {
        let mut cm = CustomMetrics::new();
        cm.record("a".into(), json!(1));
        cm.record("b".into(), json!("two"));
        let json = cm.to_json();
        assert_eq!(json.get("a").unwrap(), 1);
        assert_eq!(json.get("b").unwrap(), "two");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn record_then_get_consistent(key in "[a-zA-Z_]{1,30}", val in any::<i64>()) {
            let mut cm = CustomMetrics::new();
            let json_val = serde_json::json!(val);
            cm.record(key.clone(), json_val.clone());
            prop_assert_eq!(cm.get(&key), Some(&json_val));
        }

        #[test]
        fn last_write_wins(key in "[a-zA-Z_]{1,30}", v1 in any::<i64>(), v2 in any::<i64>()) {
            let mut cm = CustomMetrics::new();
            cm.record(key.clone(), serde_json::json!(v1));
            cm.record(key.clone(), serde_json::json!(v2));
            prop_assert_eq!(cm.get(&key), Some(&serde_json::json!(v2)));
        }

        #[test]
        fn to_json_contains_all_recorded(
            entries in proptest::collection::vec(
                ("[a-z]{1,10}", any::<i64>()),
                1..20,
            )
        ) {
            let mut cm = CustomMetrics::new();
            for (k, v) in &entries {
                cm.record(k.clone(), serde_json::json!(v));
            }
            let json = cm.to_json();
            // last-write-wins: check final value for each key
            let mut expected = std::collections::HashMap::new();
            for (k, v) in &entries {
                expected.insert(k.clone(), serde_json::json!(v));
            }
            for (k, v) in &expected {
                prop_assert_eq!(json.get(k), Some(v));
            }
        }
    }
}
