//! Card service layer — MCP-facing read/write operations.
//!
//! Thin adapter between MCP tool handlers and [`algocline_engine::card`].
//! All data flows through the engine; this layer handles JSON
//! serialization for the MCP transport.

use algocline_engine::card;

use super::AppService;

impl AppService {
    /// List Cards as JSON summaries, optionally filtered by package.
    pub fn card_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = card::list(pkg)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Fetch full Card body (Tier 1) by id.
    pub fn card_get(&self, card_id: &str) -> Result<String, String> {
        match card::get(card_id)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("card '{card_id}' not found")),
        }
    }

    /// Query Cards with sort, filter, and limit.
    #[allow(clippy::too_many_arguments)]
    pub fn card_find(
        &self,
        pkg: Option<String>,
        scenario: Option<String>,
        model: Option<String>,
        sort: Option<String>,
        limit: Option<usize>,
        min_pass_rate: Option<f64>,
    ) -> Result<String, String> {
        let q = card::FindQuery {
            pkg,
            scenario,
            model,
            sort,
            limit,
            min_pass_rate,
        };
        let rows = card::find(q)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Resolve alias then fetch the full Card.
    pub fn card_get_by_alias(&self, name: &str) -> Result<String, String> {
        match card::get_by_alias(name)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("alias '{name}' not found")),
        }
    }

    /// List aliases, optionally filtered by package.
    pub fn card_alias_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = card::alias_list(pkg)?;
        Ok(card::aliases_to_json(&rows).to_string())
    }

    /// Pin or rebind a mutable alias to a Card.
    pub fn card_alias_set(
        &self,
        name: &str,
        card_id: &str,
        pkg: Option<&str>,
        note: Option<&str>,
    ) -> Result<String, String> {
        let alias = card::alias_set(name, card_id, pkg, note)?;
        let arr = card::aliases_to_json(std::slice::from_ref(&alias));
        let single = arr
            .as_array()
            .and_then(|a| a.first().cloned())
            .unwrap_or(serde_json::Value::Null);
        Ok(single.to_string())
    }

    /// Additive-only annotation — new top-level keys only.
    pub fn card_append(
        &self,
        card_id: &str,
        fields: serde_json::Value,
    ) -> Result<String, String> {
        let merged = card::append(card_id, fields)?;
        Ok(merged.to_string())
    }

    /// Read per-case sidecar rows (Tier 2) with offset/limit paging.
    pub fn card_samples(
        &self,
        card_id: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<String, String> {
        let rows = card::read_samples(card_id, offset, limit)?;
        Ok(serde_json::Value::Array(rows).to_string())
    }
}
