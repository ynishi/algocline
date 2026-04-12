//! Hub — Remote Index search with local merge.
//!
//! Fetches a static JSON index from a remote URL, merges with locally
//! installed packages and cards, and returns search results with
//! `installed: true/false` for each entry.
//!
//! The remote index is cached at `~/.algocline/hub_cache.json` with a
//! configurable TTL (default 1 hour).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::manifest;
use super::AppService;

// ─── Constants ─────────────────────────────────────────────────

/// Default remote index URL. Points to the generated index in the
/// bundled-packages repository (GitHub Pages or raw).
const DEFAULT_INDEX_URL: &str =
    "https://raw.githubusercontent.com/ynishi/algocline-bundled-packages/main/hub_index.json";

/// Cache TTL in seconds (1 hour).
const CACHE_TTL_SECS: u64 = 3600;

// ─── Index schema ──────────────────────────────────────────────

/// Remote index — same shape as the local index so merge is trivial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HubIndex {
    pub schema_version: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub packages: Vec<IndexEntry>,
}

/// One package in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IndexEntry {
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub card_count: usize,
    #[serde(default)]
    pub best_card: Option<BestCard>,
}

/// Best card summary within a package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BestCard {
    pub card_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub pass_rate: f64,
    #[serde(default)]
    pub scenario: String,
}

/// Search result — index entry enriched with local install state.
#[derive(Debug, Clone, Serialize)]
struct SearchResult {
    name: String,
    version: String,
    description: String,
    category: String,
    source: String,
    installed: bool,
    card_count: usize,
    best_card: Option<BestCard>,
}

// ─── Remote Cache ─────────────────────────────────────────────
//
// Caches the remote index only. `hub_reindex` (local index generation)
// does NOT use this cache — it writes to a user-specified output path.

fn remote_cache_path() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("hub_remote_cache.json"))
}

/// Load cached remote index if fresh (within TTL).
fn load_remote_cache() -> Option<HubIndex> {
    let path = remote_cache_path().ok()?;
    if !path.exists() {
        return None;
    }
    let metadata = std::fs::metadata(&path).ok()?;
    let age = metadata.modified().ok()?.elapsed().ok()?;
    if age.as_secs() > CACHE_TTL_SECS {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Save remote index to cache file.
fn save_remote_cache(index: &HubIndex) {
    if let Ok(path) = remote_cache_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(index) {
            let _ = std::fs::write(&path, json);
        }
    }
}

// ─── Remote fetch ──────────────────────────────────────────────

/// HTTP request timeout (30 seconds).
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Fetch the remote index, using cache if fresh.
/// Falls back to an empty index on network failure (local-only mode).
fn fetch_remote_index(url: Option<&str>) -> (HubIndex, Option<String>) {
    // Try cache first
    if let Some(cached) = load_remote_cache() {
        return (cached, None);
    }

    let index_url = url.unwrap_or(DEFAULT_INDEX_URL);

    let result = (|| -> Result<HubIndex, String> {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(HTTP_TIMEOUT))
                .build(),
        );
        let body: String = agent
            .get(index_url)
            .call()
            .map_err(|e| format!("Failed to fetch remote index from {index_url}: {e}"))?
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("Failed to read response body: {e}"))?;

        let index: HubIndex = serde_json::from_str(&body)
            .map_err(|e| format!("Failed to parse remote index: {e}"))?;

        save_remote_cache(&index);
        Ok(index)
    })();

    match result {
        Ok(index) => (index, None),
        Err(e) => {
            // Fallback: empty remote index, search will still show local packages
            let empty = HubIndex {
                schema_version: "hub_index/v0".into(),
                updated_at: String::new(),
                packages: Vec::new(),
            };
            (empty, Some(format!("remote index unavailable ({e}), showing local packages only")))
        }
    }
}

// ─── Local state ───────────────────────────────────────────────

/// Build a set of locally installed package names from `installed.json`
/// and the `~/.algocline/packages/` directory.
fn installed_packages() -> HashMap<String, Option<String>> {
    let mut map = HashMap::new();

    // From manifest (has version info)
    if let Ok(m) = manifest::load_manifest() {
        for (name, entry) in &m.packages {
            map.insert(name.clone(), entry.version.clone());
        }
    }

    // Also scan packages/ dir in case manifest is stale
    if let Some(home) = dirs::home_dir() {
        let pkg_dir = home.join(".algocline").join("packages");
        if let Ok(entries) = std::fs::read_dir(&pkg_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        map.entry(name.to_string()).or_insert(None);
                    }
                }
            }
        }
    }

    map
}

/// Count local cards per package from `~/.algocline/cards/{pkg}/`.
fn local_card_counts() -> HashMap<String, usize> {
    let mut map = HashMap::new();
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return map,
    };
    let cards_dir = home.join(".algocline").join("cards");
    let entries = match std::fs::read_dir(&cards_dir) {
        Ok(e) => e,
        Err(_) => return map,
    };
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let pkg = match entry.file_name().to_str() {
            Some(n) => n.to_string(),
            None => continue,
        };
        let count = std::fs::read_dir(entry.path())
            .map(|es| {
                es.flatten()
                    .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
                    .count()
            })
            .unwrap_or(0);
        if count > 0 {
            map.insert(pkg, count);
        }
    }
    map
}

// ─── Merge ─────────────────────────────────────────────────────

/// Merge remote index with local install state.
fn merge(remote: &HubIndex) -> Vec<SearchResult> {
    let installed = installed_packages();
    let card_counts = local_card_counts();

    let mut results: Vec<SearchResult> = Vec::new();

    for entry in &remote.packages {
        let is_installed = installed.contains_key(&entry.name);
        let local_cards = card_counts.get(&entry.name).copied().unwrap_or(0);

        results.push(SearchResult {
            name: entry.name.clone(),
            version: entry.version.clone(),
            description: entry.description.clone(),
            category: entry.category.clone(),
            source: entry.source.clone(),
            installed: is_installed,
            card_count: if is_installed && local_cards > entry.card_count {
                local_cards
            } else {
                entry.card_count
            },
            best_card: entry.best_card.clone(),
        });
    }

    // Add local-only packages (not in remote index)
    for (name, version) in &installed {
        if results.iter().any(|r| r.name == *name) {
            continue;
        }
        results.push(SearchResult {
            name: name.clone(),
            version: version.clone().unwrap_or_default(),
            description: String::new(),
            category: String::new(),
            source: String::new(),
            installed: true,
            card_count: card_counts.get(name).copied().unwrap_or(0),
            best_card: None,
        });
    }

    results
}

// ─── Search (filtering) ───────────────────────────────────────

fn matches_query(result: &SearchResult, query: &str) -> bool {
    let q = query.to_lowercase();
    result.name.to_lowercase().contains(&q)
        || result.description.to_lowercase().contains(&q)
        || result.category.to_lowercase().contains(&q)
}

// ─── Index generation (reindex) ───────────────────────────────

/// Parse `M.meta = { ... }` from an `init.lua` file without Lua VM.
/// Returns (name, version, description, category) or None on failure.
fn parse_meta_from_init_lua(path: &std::path::Path) -> Option<(String, String, String, String)> {
    let content = std::fs::read_to_string(path).ok()?;
    // Limit search to first ~2KB (snap back to a char boundary)
    let mut limit = 2048.min(content.len());
    while limit > 0 && !content.is_char_boundary(limit) {
        limit -= 1;
    }
    let head = &content[..limit];

    // Find M.meta = { ... } block
    let meta_start = head.find("M.meta")?;
    let brace_start = head[meta_start..].find('{')? + meta_start;
    let brace_end = head[brace_start..].find('}')? + brace_start;
    let block = &head[brace_start + 1..brace_end];

    let extract = |field: &str| -> String {
        // Match: field = "value"
        block
            .find(field)
            .and_then(|pos| {
                let after = &block[pos + field.len()..];
                let q_start = after.find('"')? + 1;
                let q_end = after[q_start..].find('"')? + q_start;
                Some(after[q_start..q_end].to_string())
            })
            .unwrap_or_default()
    };

    let name = extract("name");
    if name.is_empty() {
        return None;
    }
    Some((name, extract("version"), extract("description"), extract("category")))
}

/// Build a hub index from locally installed packages.
fn build_local_index() -> HubIndex {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            return HubIndex {
                schema_version: "hub_index/v0".into(),
                updated_at: super::manifest::now_iso8601(),
                packages: Vec::new(),
            }
        }
    };

    let pkg_dir = home.join(".algocline").join("packages");
    let card_counts = local_card_counts();
    let manifest = manifest::load_manifest().unwrap_or_default();

    let mut entries = Vec::new();

    let dir_entries = match std::fs::read_dir(&pkg_dir) {
        Ok(e) => e,
        Err(_) => {
            return HubIndex {
                schema_version: "hub_index/v0".into(),
                updated_at: super::manifest::now_iso8601(),
                packages: Vec::new(),
            }
        }
    };

    for entry in dir_entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let dir_name = match entry.file_name().to_str() {
            Some(n) => n.to_string(),
            None => continue,
        };

        let init_lua = entry.path().join("init.lua");
        let (name, version, description, category) =
            parse_meta_from_init_lua(&init_lua).unwrap_or_else(|| {
                (dir_name.clone(), String::new(), String::new(), String::new())
            });

        // Use manifest source if available
        let source = manifest
            .packages
            .get(&dir_name)
            .map(|e| e.source.clone())
            .unwrap_or_default();

        entries.push(IndexEntry {
            name,
            version,
            description,
            category,
            source,
            card_count: card_counts.get(&dir_name).copied().unwrap_or(0),
            best_card: None,
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    HubIndex {
        schema_version: "hub_index/v0".into(),
        updated_at: super::manifest::now_iso8601(),
        packages: entries,
    }
}

// ─── Public API ────────────────────────────────────────────────

impl AppService {
    /// Generate a hub index from locally installed packages.
    ///
    /// Scans `~/.algocline/packages/` and parses `M.meta` from each
    /// `init.lua` without Lua VM. Writes the index to `output_path`
    /// (for CI / publishing). Does NOT touch the remote search cache.
    pub fn hub_reindex(&self, output_path: Option<&str>) -> Result<String, String> {
        let index = build_local_index();

        let written_path = if let Some(path) = output_path {
            let json = serde_json::to_string_pretty(&index)
                .map_err(|e| format!("Failed to serialize index: {e}"))?;
            std::fs::write(path, &json)
                .map_err(|e| format!("Failed to write index to {path}: {e}"))?;
            Some(path.to_string())
        } else {
            None
        };

        let response = serde_json::json!({
            "package_count": index.packages.len(),
            "updated_at": index.updated_at,
            "output_path": written_path,
        });
        Ok(response.to_string())
    }

    /// Search packages across remote index + local state.
    pub fn hub_search(
        &self,
        query: Option<&str>,
        category: Option<&str>,
        installed_only: Option<bool>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        let (remote, warning) = fetch_remote_index(None);
        let mut results = merge(&remote);

        // Filter by query
        if let Some(q) = query {
            if !q.is_empty() {
                results.retain(|r| matches_query(r, q));
            }
        }

        // Filter by category
        if let Some(cat) = category {
            let cat_lower = cat.to_lowercase();
            results.retain(|r| r.category.to_lowercase() == cat_lower);
        }

        // Filter by installed state
        if let Some(true) = installed_only {
            results.retain(|r| r.installed);
        }

        // Sort: installed first, then by name
        results.sort_by(|a, b| {
            b.installed
                .cmp(&a.installed)
                .then_with(|| a.name.cmp(&b.name))
        });

        // Limit
        let total = results.len();
        let limit = limit.unwrap_or(50);
        results.truncate(limit);

        let mut json = serde_json::json!({
            "results": results,
            "total": total,
            "index_url": DEFAULT_INDEX_URL,
        });
        if let Some(w) = warning {
            json["warning"] = serde_json::Value::String(w);
        }
        Ok(json.to_string())
    }
}
