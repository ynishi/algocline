use serde::{Deserialize, Serialize};

/// All information needed to start an execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSpec {
    /// Resolved Lua source code.
    pub code: String,
    /// Context passed to Lua as the `ctx` global.
    pub ctx: serde_json::Value,
    /// Namespace for alc.state (from ctx._ns, defaults to "default").
    pub namespace: String,
}

impl ExecutionSpec {
    /// Construct from code and ctx. Extracts namespace from ctx._ns
    /// with "default" as fallback.
    pub fn new(code: String, ctx: serde_json::Value) -> Self {
        let namespace = ctx
            .as_object()
            .and_then(|o| o.get("_ns"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        Self {
            code,
            ctx,
            namespace,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn namespace_from_ctx() {
        let spec = ExecutionSpec::new("code".into(), json!({"_ns": "custom"}));
        assert_eq!(spec.namespace, "custom");
    }

    #[test]
    fn namespace_default_when_missing() {
        let spec = ExecutionSpec::new("code".into(), json!({"key": "value"}));
        assert_eq!(spec.namespace, "default");
    }

    #[test]
    fn namespace_default_when_null_ctx() {
        let spec = ExecutionSpec::new("code".into(), serde_json::Value::Null);
        assert_eq!(spec.namespace, "default");
    }

    #[test]
    fn namespace_default_when_ns_is_non_string() {
        let spec = ExecutionSpec::new("code".into(), json!({"_ns": 42}));
        assert_eq!(spec.namespace, "default");
    }

    #[test]
    fn roundtrip_json() {
        let spec = ExecutionSpec::new("return 1".into(), json!({"_ns": "test", "data": [1,2]}));
        let json = serde_json::to_value(&spec).unwrap();
        let restored: ExecutionSpec = serde_json::from_value(json).unwrap();
        assert_eq!(restored.code, spec.code);
        assert_eq!(restored.ctx, spec.ctx);
        assert_eq!(restored.namespace, spec.namespace);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_json_value() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            Just(serde_json::Value::Null),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(|n| serde_json::json!(n)),
            "\\PC{0,50}".prop_map(serde_json::Value::String),
            Just(serde_json::json!([])),
            Just(serde_json::json!({})),
            Just(serde_json::json!({"key": "value"})),
            "\\PC{1,30}".prop_map(|ns| serde_json::json!({"_ns": ns})),
        ]
    }

    proptest! {
        #[test]
        fn new_never_panics(code in "\\PC{0,200}", ctx in arb_json_value()) {
            let spec = ExecutionSpec::new(code, ctx);
            // namespace must always be a non-empty string
            prop_assert!(!spec.namespace.is_empty());
        }

        #[test]
        fn namespace_is_ctx_ns_or_default(ns in "\\PC{1,30}") {
            let ctx = serde_json::json!({"_ns": ns});
            let spec = ExecutionSpec::new("code".into(), ctx);
            prop_assert_eq!(&spec.namespace, &ns);
        }

        #[test]
        fn serde_roundtrip(code in "\\PC{0,100}", ctx in arb_json_value()) {
            let spec = ExecutionSpec::new(code, ctx);
            let json = serde_json::to_value(&spec).unwrap();
            let restored: ExecutionSpec = serde_json::from_value(json).unwrap();
            prop_assert_eq!(&restored.code, &spec.code);
            prop_assert_eq!(&restored.ctx, &spec.ctx);
            prop_assert_eq!(&restored.namespace, &spec.namespace);
        }
    }
}
