//! Hub — package discovery, search, and index management.
//!
//! The Hub is algocline's package registry layer.  It aggregates remote
//! index data with local install state so that users (via AI) can
//! **discover** packages they haven't installed yet, and **inspect**
//! installed packages with full Card and eval statistics.
//!
//! ## Staged design
//!
//! | Stage | Scope | Status |
//! |-------|-------|--------|
//! | **1** | Card Collection install, Pkg-bundled cards | Done |
//! | **2** | Hub MCP tools (`hub_search`, `hub_info`, `hub_reindex`), local index | Done |
//! | **3** | Aggregated remote collection index, `hub_publish`, LP | Planned |
//!
//! ## MCP tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | `alc_hub_search` | Discover packages across remote + local indices |
//! | `alc_hub_info` | Detailed single-package view (meta + cards + aliases + stats) |
//! | `alc_hub_reindex` | Rebuild index from local packages or a repo checkout |
//!
//! ## Index schema (`hub_index/v0`)
//!
//! ```json
//! {
//!   "schema_version": "hub_index/v0",
//!   "updated_at": "2026-04-12T10:00:00Z",
//!   "packages": [{
//!     "name": "cot",
//!     "version": "0.1.0",
//!     "description": "Chain-of-Thought prompting",
//!     "category": "reasoning",
//!     "source": "https://github.com/...",
//!     "card_count": 3,
//!     "best_card": { "card_id": "...", "model": "...", "pass_rate": 0.82, "scenario": "..." }
//!   }]
//! }
//! ```
//!
//! Index generation uses `init.lua` M.meta parsing only — no Lua VM
//! required.  This keeps the index buildable in CI environments.
//!
//! ## Index URL discovery (4-tier)
//!
//! Sources are checked in priority order; URLs are deduplicated:
//!
//!   0. **Collection URL** — `[hub].collection_url` in `~/.algocline/config.toml`.
//!      Aggregated index containing all known packages (Stage 3).
//!   1. **Hub registries** — `~/.algocline/hub_registries.json`, auto-populated
//!      by `pkg_install` and `card_install`.
//!   2. **Installed manifest** — `~/.algocline/installed.json`, fallback for
//!      sources registered before registries existed.
//!   3. **Compiled-in seeds** — `AUTO_INSTALL_SOURCES` for first-run bootstrap.
//!
//! GitHub repo URLs are transformed to raw index URLs:
//!
//! ```text
//! https://github.com/{owner}/{repo}
//!   → https://raw.githubusercontent.com/{owner}/{repo}/main/hub_index.json
//! ```
//!
//! ## Caching
//!
//! Remote indices are cached per-source at
//! `~/.algocline/hub_cache/{hash}.json` where hash is FNV-1a of the
//! URL.  TTL is 1 hour.
//!
//! ## Registry persistence
//!
//! `~/.algocline/hub_registries.json` records source URLs from
//! `pkg_install` and `card_install`.  Written atomically (tempfile +
//! rename) to avoid corruption on interruption.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::list_opts::{
    apply_sort_by_value, matches_filter, parse_sort, project_fields, resolve_fields, ListOpts,
    HUB_SEARCH_FULL, HUB_SEARCH_SUMMARY,
};
use super::manifest;
use super::resolve::AUTO_INSTALL_SOURCES;
use super::AppService;

// ─── Constants ─────────────────────────────────────────────────

/// Cache TTL in seconds (1 hour).
const CACHE_TTL_SECS: u64 = 3600;

/// HTTP request timeout (30 seconds).
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

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
    /// Leading `---` docstring lines from init.lua (for full-text search).
    #[serde(default)]
    pub docstring: String,
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
///
/// `docstring` is `skip_serializing` so the default serde output never
/// exposes it (docstrings can be large and dominate payload size). The
/// `hub_search` projection path puts it back into the JSON object when
/// the resolved field set contains `"docstring"`, via
/// [`SearchResult::to_value_with_optional_docstring`].
///
/// `docstring_matched` is a query-time signal: it is `Some(true)` only
/// when the query hit docstring and none of {name, description, category}.
/// Otherwise (no query, or query hit any of the other fields) it is
/// `None` and omitted from the output.
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
    #[serde(skip_serializing)]
    docstring: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    docstring_matched: Option<bool>,
}

impl SearchResult {
    /// Serialize `self` to a JSON `Value`, optionally re-attaching
    /// `docstring` to the resulting object.
    ///
    /// `skip_serializing` removes `docstring` from every serde output
    /// path. When projection selects `docstring` as an output field, we
    /// need to put it back — this helper bridges that gap by inserting
    /// the field manually into the resulting `Value::Object`.
    ///
    /// Returns the original `Value` unchanged if serialization produced
    /// a non-object (should not happen for `SearchResult`, but we stay
    /// defensive because the downstream `project_fields` contract
    /// tolerates non-objects).
    fn to_value_with_optional_docstring(&self, include_docstring: bool) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if include_docstring {
            if let serde_json::Value::Object(ref mut map) = v {
                map.insert(
                    "docstring".to_string(),
                    serde_json::Value::String(self.docstring.clone()),
                );
            }
        }
        v
    }
}

// ─── Hub registries ───────────────────────────────────────────
//
// Persistent file (`~/.algocline/hub_registries.json`) that records
// source URLs from `pkg_install` and `card_install`.  This is the
// primary source for Hub index URL discovery — the manifest and
// `AUTO_INSTALL_SOURCES` serve as fallback seeds.

/// One entry in `hub_registries.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RegistryEntry {
    /// Original source URL (Git repo or local path).
    pub source: String,
    /// How it was registered: "pkg_install" or "card_install".
    pub origin: String,
    /// ISO 8601 timestamp of when the entry was added.
    pub added_at: String,
}

/// Top-level registries file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct HubRegistries {
    pub registries: Vec<RegistryEntry>,
}

fn registries_path() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("hub_registries.json"))
}

/// Load registries from disk.  Returns empty list if file is missing.
fn load_registries() -> HubRegistries {
    let path = match registries_path() {
        Ok(p) => p,
        Err(_) => return HubRegistries::default(),
    };
    if !path.exists() {
        return HubRegistries::default();
    }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default()
}

/// Register a source URL.  Deduplicates by normalized URL.
///
/// Uses atomic write (tempfile + rename) to avoid partial writes if
/// the process is interrupted.  Read-modify-write is not locked across
/// processes, but MCP servers are single-process so this is safe in
/// practice.
pub(crate) fn register_source(source: &str, origin: &str) {
    let normalized = source.trim_end_matches('/').to_string();
    if normalized.is_empty() {
        return;
    }
    // Skip local paths — they can't host a remote index
    if normalized.starts_with('/') || normalized.starts_with('.') {
        return;
    }

    let path = match registries_path() {
        Ok(p) => p,
        Err(_) => return,
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Re-read from disk right before write to minimize TOCTOU window
    let mut reg = load_registries();

    // Already registered?
    if reg
        .registries
        .iter()
        .any(|e| e.source.trim_end_matches('/') == normalized)
    {
        return;
    }

    reg.registries.push(RegistryEntry {
        source: normalized,
        origin: origin.to_string(),
        added_at: manifest::now_iso8601(),
    });

    // Atomic write: write to temp file, then rename
    match serde_json::to_string_pretty(&reg) {
        Ok(json) => {
            let tmp_path = path.with_extension("json.tmp");
            if let Err(e) = std::fs::write(&tmp_path, &json) {
                tracing::warn!("failed to write hub registries tmp: {e}");
                return;
            }
            if let Err(e) = std::fs::rename(&tmp_path, &path) {
                tracing::warn!("failed to rename hub registries: {e}");
                // Clean up tmp on failure
                let _ = std::fs::remove_file(&tmp_path);
            }
        }
        Err(e) => tracing::warn!("failed to serialize hub registries: {e}"),
    }
}

// ─── Hub config ──────────────────────────────────────────────
//
// Optional `[hub]` section in `~/.algocline/config.toml`:
//
//   [hub]
//   collection_url = "https://raw.githubusercontent.com/.../hub_index.json"
//
// When set, this is fetched as Tier 0 (the aggregated collection
// index containing all known packages, including uninstalled ones).

/// Read the `[hub].collection_url` from `~/.algocline/config.toml`.
fn collection_url_from_config() -> Option<String> {
    let home = dirs::home_dir()?;
    let path = home.join(".algocline").join("config.toml");
    let content = std::fs::read_to_string(&path).ok()?;
    let doc: toml_edit::DocumentMut = content.parse().ok()?;
    let url = doc
        .get("hub")?
        .get("collection_url")?
        .as_str()?
        .trim()
        .to_string();
    if url.is_empty() {
        None
    } else {
        Some(url)
    }
}

// ─── Index URL discovery ──────────────────────────────────────
//
// Derives remote index URLs from:
//   0. Hub Collection URL (from config.toml) — aggregated index
//   1. Hub registries (`hub_registries.json`) — primary source
//   2. Unique `source` fields in the installed-packages manifest
//   3. `AUTO_INSTALL_SOURCES` as fallback seeds (for first run)
//
// GitHub repos are transformed:
//   https://github.com/{owner}/{repo}  →
//   https://raw.githubusercontent.com/{owner}/{repo}/main/hub_index.json

/// Convert a GitHub repo URL to a raw `hub_index.json` URL.
/// Returns `None` for non-GitHub URLs (future: support other hosts).
fn repo_to_index_url(repo_url: &str) -> Option<String> {
    let trimmed = repo_url.trim_end_matches('/').trim_end_matches(".git");
    if let Some(path) = trimmed.strip_prefix("https://github.com/") {
        // path = "owner/repo"
        let parts: Vec<&str> = path.splitn(3, '/').collect();
        if parts.len() >= 2 {
            return Some(format!(
                "https://raw.githubusercontent.com/{}/{}/main/hub_index.json",
                parts[0], parts[1]
            ));
        }
    }
    // Non-GitHub URL: assume it's already a direct index URL
    if trimmed.ends_with(".json") {
        Some(trimmed.to_string())
    } else {
        None
    }
}

/// Collect unique index URLs from config + registries + manifest + bundled seeds.
fn discover_index_urls() -> Vec<String> {
    let mut index_urls: Vec<String> = Vec::new();

    // 0. From config.toml [hub].collection_url (Tier 0 — aggregated collection)
    if let Some(url) = collection_url_from_config() {
        index_urls.push(url);
    }

    let mut repo_urls: HashSet<String> = HashSet::new();

    // 1. From hub registries (primary)
    let reg = load_registries();
    for entry in &reg.registries {
        let normalized = entry.source.trim_end_matches('/').to_string();
        if !normalized.is_empty() {
            repo_urls.insert(normalized);
        }
    }

    // 2. From manifest (catch sources registered before hub_registries existed)
    if let Ok(m) = manifest::load_manifest() {
        for entry in m.packages.values() {
            let normalized = entry.source.trim_end_matches('/').to_string();
            if !normalized.is_empty() && !normalized.starts_with('/') {
                repo_urls.insert(normalized);
            }
        }
    }

    // 3. Fallback: bundled sources (ensures at least these are checked)
    for url in AUTO_INSTALL_SOURCES {
        repo_urls.insert(url.to_string());
    }

    // 4. Transform repo URLs → index URLs, dedup against Tier 0
    let existing: HashSet<String> = index_urls.iter().cloned().collect();
    let mut derived: Vec<String> = repo_urls
        .iter()
        .filter_map(|url| repo_to_index_url(url))
        .filter(|url| !existing.contains(url))
        .collect();
    derived.sort();
    derived.dedup();
    index_urls.extend(derived);

    index_urls
}

// ─── Per-source cache ─────────────────────────────────────────
//
// Each remote index is cached separately at
// `~/.algocline/hub_cache/{hash}.json` where hash is derived from
// the index URL. This avoids mixing data from different registries
// and allows per-source TTL validation.

fn cache_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("hub_cache"))
}

fn cache_key(url: &str) -> String {
    // Simple hash: use the URL bytes to produce a stable hex string.
    // Avoids pulling in a hash crate — good enough for cache file naming.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for b in url.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }
    format!("{h:016x}")
}

/// Load cached remote index for a specific URL if fresh (within TTL).
fn load_cached(url: &str) -> Option<HubIndex> {
    let dir = cache_dir().ok()?;
    let path = dir.join(format!("{}.json", cache_key(url)));
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

/// Save remote index to per-source cache file.
fn save_cached(url: &str, index: &HubIndex) {
    let dir = match cache_dir() {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("hub cache dir unavailable: {e}");
            return;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("failed to create hub cache dir: {e}");
        return;
    }
    let path = dir.join(format!("{}.json", cache_key(url)));
    match serde_json::to_string_pretty(index) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("failed to write hub cache {}: {e}", path.display());
            }
        }
        Err(e) => tracing::warn!("failed to serialize hub cache: {e}"),
    }
}

// ─── Remote fetch ──────────────────────────────────────────────

/// Fetch a single remote index by URL, using per-source cache.
fn fetch_one(url: &str) -> Result<HubIndex, String> {
    if let Some(cached) = load_cached(url) {
        return Ok(cached);
    }

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(HTTP_TIMEOUT))
            .build(),
    );
    let body: String = agent
        .get(url)
        .call()
        .map_err(|e| format!("Failed to fetch {url}: {e}"))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read response from {url}: {e}"))?;

    let index: HubIndex = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse index from {url}: {e}"))?;

    save_cached(url, &index);
    Ok(index)
}

/// Fetch all discovered remote indices and merge into one.
/// Falls back gracefully: failed sources are skipped with warnings.
fn fetch_remote_indices() -> (HubIndex, Vec<String>) {
    let urls = discover_index_urls();
    let mut all_packages: Vec<IndexEntry> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut warnings: Vec<String> = Vec::new();

    for url in &urls {
        match fetch_one(url) {
            Ok(index) => {
                for entry in index.packages {
                    if seen_names.insert(entry.name.clone()) {
                        all_packages.push(entry);
                    }
                    // If duplicate name across sources, first wins
                }
            }
            Err(e) => {
                warnings.push(e);
            }
        }
    }

    if all_packages.is_empty() && !warnings.is_empty() {
        warnings.insert(
            0,
            "all remote indices unavailable, showing local packages only".to_string(),
        );
    }

    let merged = HubIndex {
        schema_version: "hub_index/v0".into(),
        updated_at: String::new(),
        packages: all_packages,
    };
    (merged, warnings)
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

/// Count eval results for a specific package by scanning `~/.algocline/evals/`.
///
/// Reads only `.meta.json` files (lightweight) to check the strategy field.
/// Falls back to reading full eval JSON if meta is missing.
fn count_evals_for_pkg(pkg: &str) -> usize {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return 0,
    };
    let evals_dir = home.join(".algocline").join("evals");
    let entries = match std::fs::read_dir(&evals_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    // Collect all filenames first so ordering doesn't matter.
    // We track stems that have a .meta.json to avoid reading the full eval JSON.
    let mut meta_stems: HashSet<String> = HashSet::new();
    let mut meta_matches: usize = 0;
    let mut non_meta_paths: Vec<(PathBuf, String)> = Vec::new(); // (path, stem)

    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if name.ends_with(".meta.json") {
            let stem = name.trim_end_matches(".meta.json").to_string();
            meta_stems.insert(stem);
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if val.get("strategy").and_then(|s| s.as_str()) == Some(pkg) {
                        meta_matches += 1;
                    }
                }
            }
            continue;
        }

        // Skip non-json or comparison files
        if !name.ends_with(".json") || name.starts_with("compare_") {
            continue;
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        non_meta_paths.push((path, stem));
    }

    // Only read full eval JSON for entries without a .meta.json
    let fallback_matches = non_meta_paths
        .iter()
        .filter(|(_, stem)| !meta_stems.contains(stem))
        .filter(|(path, _)| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                .and_then(|v| v.get("strategy")?.as_str().map(|s| s == pkg))
                .unwrap_or(false)
        })
        .count();

    meta_matches + fallback_matches
}

// ─── Merge ─────────────────────────────────────────────────────

/// Merge remote index with local install state.
///
/// When a package is installed locally and the remote index lacks a
/// docstring (pre-v0.21 indices), the docstring is extracted from the
/// local `init.lua` so that full-text search works immediately.
fn merge(remote: &HubIndex) -> Vec<SearchResult> {
    let installed = installed_packages();
    let card_counts = local_card_counts();
    let pkg_dir = dirs::home_dir().map(|h| h.join(".algocline").join("packages"));

    let mut seen: HashSet<String> = HashSet::new();
    let mut results: Vec<SearchResult> = Vec::new();

    for entry in &remote.packages {
        let is_installed = installed.contains_key(&entry.name);
        let local_cards = card_counts.get(&entry.name).copied().unwrap_or(0);

        // Supplement empty docstring from local init.lua when installed
        let docstring = if entry.docstring.is_empty() && is_installed {
            pkg_dir
                .as_ref()
                .map(|d| extract_docstring(&d.join(&entry.name).join("init.lua")))
                .unwrap_or_default()
        } else {
            entry.docstring.clone()
        };

        seen.insert(entry.name.clone());
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
            docstring,
            docstring_matched: None,
        });
    }

    // Add local-only packages (not in remote index).
    for (name, version) in &installed {
        if seen.contains(name) {
            continue;
        }
        let docstring = pkg_dir
            .as_ref()
            .map(|d| extract_docstring(&d.join(name).join("init.lua")))
            .unwrap_or_default();
        results.push(SearchResult {
            name: name.clone(),
            version: version.clone().unwrap_or_default(),
            description: String::new(),
            category: String::new(),
            source: String::new(),
            installed: true,
            card_count: card_counts.get(name).copied().unwrap_or(0),
            best_card: None,
            docstring,
            docstring_matched: None,
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
        || result.docstring.to_lowercase().contains(&q)
}

// ─── Index generation (reindex) ───────────────────────────────

/// Extract leading `---` docstring lines from an `init.lua` file.
///
/// Collects consecutive lines starting with `---` (Lua doc-comment)
/// from the beginning of the file.  Stops at the first non-doc line.
/// Returns a single string with lines joined by newline, stripped of
/// the `---` prefix.  Used for full-text search in hub_search.
fn extract_docstring(path: &std::path::Path) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("---") {
            lines.push(rest.trim().to_string());
        } else if trimmed.is_empty() {
            // Allow blank lines within the docstring block
            continue;
        } else {
            break;
        }
    }
    lines.join("\n")
}

/// Parse `M.meta = { ... }` from an `init.lua` file without Lua VM.
///
/// Extracts (name, version, description, category) from the first
/// `M.meta = { ... }` block found anywhere in the file.
///
/// Supports string concatenation: `description = "foo " .. "bar"` is
/// collected as `"foo bar"`.
///
/// **Limitation**: Only supports flat key-value pairs inside `M.meta`.
/// Nested tables (e.g. `tags = { ... }`) are skipped via brace-depth
/// tracking. `M.meta` fields are expected to be simple (possibly
/// concatenated) string literals.
fn parse_meta_from_init_lua(path: &std::path::Path) -> Option<(String, String, String, String)> {
    let content = std::fs::read_to_string(path).ok()?;
    let head = content.as_str();

    // Find M.meta = { ... } block (with brace-depth tracking).
    // Skip occurrences inside Lua line comments (`-- ...`) so that
    // docstrings mentioning "M.meta" do not hijack the search.
    let mut search_from = 0;
    let meta_start = loop {
        let rel = head[search_from..].find("M.meta")?;
        let pos = search_from + rel;
        let line_start = head[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        if !head[line_start..pos].contains("--") {
            break pos;
        }
        search_from = pos + "M.meta".len();
    };
    let brace_start = head[meta_start..].find('{')? + meta_start;

    // Track brace depth to handle nested tables correctly
    let mut depth = 0;
    let mut brace_end = None;
    for (i, ch) in head[brace_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    brace_end = Some(brace_start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let brace_end = brace_end?;
    let block = &head[brace_start + 1..brace_end];

    let extract = |field: &str| -> String {
        // Match: field = "value" [.. "value" ...] with word-boundary check.
        // Walk through all occurrences of `field`, skipping matches inside
        // longer identifiers (e.g. "short_description"). On the first valid
        // occurrence, collect one or more `"..."` string literals joined by
        // `..` concatenation operators.
        let mut search_from = 0;
        while let Some(rel) = block[search_from..].find(field) {
            let pos = search_from + rel;
            let word_boundary = pos == 0 || {
                let prev = block.as_bytes()[pos - 1];
                !(prev.is_ascii_alphanumeric() || prev == b'_')
            };
            if word_boundary {
                let after = &block[pos + field.len()..];
                let mut collected = String::new();
                let mut cursor = 0usize;
                let mut found_any = false;
                loop {
                    let rest = &after[cursor..];
                    let Some(q_start_rel) = rest.find('"') else {
                        break;
                    };
                    if found_any {
                        // Between the prior closing quote and this opening
                        // quote, only whitespace and a single `..` operator
                        // are allowed. Anything else (comma, another field,
                        // etc.) ends the value.
                        let between = &rest[..q_start_rel];
                        if between.trim() != ".." {
                            break;
                        }
                    }
                    let lit_start = cursor + q_start_rel + 1;
                    let Some(q_end_rel) = after[lit_start..].find('"') else {
                        break;
                    };
                    collected.push_str(&after[lit_start..lit_start + q_end_rel]);
                    cursor = lit_start + q_end_rel + 1;
                    found_any = true;
                }
                if found_any {
                    return collected;
                }
            }
            search_from = pos + field.len();
        }
        String::new()
    };

    let name = extract("name");
    if name.is_empty() {
        return None;
    }
    Some((
        name,
        extract("version"),
        extract("description"),
        extract("category"),
    ))
}

/// Build a hub index by scanning a packages directory.
///
/// When `source_dir` is provided, scans that directory directly
/// (for generating an index from a repo checkout).  Metadata comes
/// only from `init.lua` — no manifest lookup, no card counts.
///
/// When `source_dir` is `None`, scans `~/.algocline/packages/` and
/// enriches entries with manifest source and local card counts.
fn build_index(source_dir: Option<&std::path::Path>) -> HubIndex {
    let empty = || HubIndex {
        schema_version: "hub_index/v0".into(),
        updated_at: super::manifest::now_iso8601(),
        packages: Vec::new(),
    };

    let pkg_dir = match source_dir {
        Some(d) => d.to_path_buf(),
        None => {
            let home = match dirs::home_dir() {
                Some(h) => h,
                None => return empty(),
            };
            home.join(".algocline").join("packages")
        }
    };

    let use_local_state = source_dir.is_none();
    let card_counts = if use_local_state {
        local_card_counts()
    } else {
        HashMap::new()
    };
    let manifest = if use_local_state {
        manifest::load_manifest().unwrap_or_default()
    } else {
        manifest::Manifest::default()
    };

    let mut entries = Vec::new();

    let dir_entries = match std::fs::read_dir(&pkg_dir) {
        Ok(e) => e,
        Err(_) => return empty(),
    };

    for entry in dir_entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let dir_name = match entry.file_name().to_str() {
            Some(n) if !n.starts_with('.') && !n.starts_with('_') => n.to_string(),
            _ => continue,
        };

        let init_lua = entry.path().join("init.lua");
        if !init_lua.exists() {
            continue;
        }

        let (name, version, description, category) = parse_meta_from_init_lua(&init_lua)
            .unwrap_or_else(|| {
                (
                    dir_name.clone(),
                    String::new(),
                    String::new(),
                    String::new(),
                )
            });

        let docstring = extract_docstring(&init_lua);

        // Use manifest source only for local-state mode
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
            docstring,
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
    /// Generate a hub index from a packages directory.
    ///
    /// When `source_dir` is provided, scans that directory (e.g. a
    /// repo checkout) — pure metadata extraction, no manifest or card
    /// data mixed in.  When omitted, scans `~/.algocline/packages/`.
    ///
    /// Writes the index to `output_path` (for CI / publishing).
    /// Does NOT touch the remote search cache.
    pub fn hub_reindex(
        &self,
        output_path: Option<&str>,
        source_dir: Option<&str>,
    ) -> Result<String, String> {
        let src = source_dir.map(std::path::Path::new);
        if let Some(d) = src {
            if !d.is_dir() {
                return Err(format!("source_dir '{}' is not a directory", d.display()));
            }
        }
        let index = build_index(src);

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
            "source_dir": source_dir,
        });
        Ok(response.to_string())
    }

    /// Show detailed information for a single package.
    ///
    /// Aggregates package metadata (from index or local `init.lua`),
    /// all Cards, aliases, and eval stats into one response.
    pub fn hub_info(&self, pkg: &str) -> Result<String, String> {
        use algocline_engine::card;

        // Guard against path traversal
        if pkg.contains("..") || pkg.contains('/') || pkg.contains('\\') {
            return Err(format!("Invalid package name: '{pkg}'"));
        }

        // Package metadata: try remote index first, fall back to local
        let installed = installed_packages();
        let is_installed = installed.contains_key(pkg);

        let (version, description, category, source) = {
            // Try to get from remote index
            let (remote, _) = fetch_remote_indices();
            if let Some(entry) = remote.packages.iter().find(|e| e.name == pkg) {
                (
                    entry.version.clone(),
                    entry.description.clone(),
                    entry.category.clone(),
                    entry.source.clone(),
                )
            } else if is_installed {
                // Fall back to local init.lua parse
                let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
                let init_lua = home
                    .join(".algocline")
                    .join("packages")
                    .join(pkg)
                    .join("init.lua");
                let meta = parse_meta_from_init_lua(&init_lua);
                let manifest_source = manifest::load_manifest()
                    .ok()
                    .and_then(|m| m.packages.get(pkg).map(|e| e.source.clone()))
                    .unwrap_or_default();
                match meta {
                    Some((_, v, d, c)) => (v, d, c, manifest_source),
                    None => (
                        installed.get(pkg).cloned().flatten().unwrap_or_default(),
                        String::new(),
                        String::new(),
                        manifest_source,
                    ),
                }
            } else {
                return Err(format!(
                    "Package '{pkg}' not found in remote indices or locally installed packages"
                ));
            }
        };

        // Cards for this package (single call, reused for stats)
        let card_rows = card::list(Some(pkg)).unwrap_or_default();
        let cards_json = card::summaries_to_json(&card_rows);

        // Aliases for this package
        let aliases_json = match card::alias_list(Some(pkg)) {
            Ok(rows) => card::aliases_to_json(&rows),
            Err(_) => serde_json::json!([]),
        };

        // Stats: card count, best pass_rate, eval count
        let card_count = card_rows.len();
        let best_pass_rate = card_rows
            .iter()
            .filter_map(|c| c.pass_rate)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_pass_rate = if best_pass_rate.is_finite() {
            Some(best_pass_rate)
        } else {
            None
        };

        // Eval count from evals directory
        let eval_count = count_evals_for_pkg(pkg);

        let response = serde_json::json!({
            "pkg": {
                "name": pkg,
                "version": version,
                "description": description,
                "category": category,
                "source": source,
                "installed": is_installed,
            },
            "cards": cards_json,
            "aliases": aliases_json,
            "stats": {
                "card_count": card_count,
                "eval_count": eval_count,
                "best_pass_rate": best_pass_rate,
            },
        });
        Ok(response.to_string())
    }

    /// Search packages across remote indices + local state.
    ///
    /// Index URLs are discovered from hub registries, manifest sources,
    /// and `AUTO_INSTALL_SOURCES`. Each source is cached independently.
    ///
    /// ## List-tool options (`opts`)
    ///
    /// The `opts` parameter carries the list-tool primitives
    /// (`limit / sort / filter / fields / verbose`) shared with other
    /// list-style MCP tools. Defaults:
    ///
    /// - `limit` — 50 when `None`. `Some(0)` means **no limit** (return
    ///   all matching entries — empty-means-all idiom).
    /// - `sort` — `"-installed,name"` when `None` (installed first, then
    ///   ascending by name).
    /// - `filter` — no additional filter. Legacy `category` /
    ///   `installed_only` parameters are merged into the filter map when
    ///   `filter` does not already contain those keys (explicit
    ///   `filter` wins on conflict).
    /// - `fields` / `verbose` — projection is applied to every entry in
    ///   the `results` array (see
    ///   [`super::list_opts::resolve_fields`]). Top-level keys
    ///   (`total`, `sources`, `warnings`) are never projected away.
    ///
    /// ## docstring handling
    ///
    /// [`SearchResult::docstring`] is `skip_serializing`, so it is
    /// absent from the default serialized view. When the resolved
    /// projection contains `"docstring"`, it is re-injected into the
    /// per-entry JSON via
    /// [`SearchResult::to_value_with_optional_docstring`].
    pub(crate) fn hub_search(
        &self,
        query: Option<&str>,
        category: Option<&str>,
        installed_only: Option<bool>,
        opts: ListOpts,
    ) -> Result<String, String> {
        let (remote, warnings) = fetch_remote_indices();
        let mut results = merge(&remote);

        // Filter by query (internal signal covers name/description/
        // category/docstring — `matches_query` unchanged).
        let query_lower = query.filter(|q| !q.is_empty()).map(|q| q.to_lowercase());
        if let Some(ref ql) = query_lower {
            results.retain(|r| matches_query(r, ql));
        }

        // Compute docstring_matched per remaining hit: Some(true) only
        // when the query matched docstring and none of {name,
        // description, category}; otherwise None.
        if let Some(ref ql) = query_lower {
            for r in &mut results {
                let other_hit = r.name.to_lowercase().contains(ql)
                    || r.description.to_lowercase().contains(ql)
                    || r.category.to_lowercase().contains(ql);
                let doc_hit = r.docstring.to_lowercase().contains(ql);
                r.docstring_matched = if !other_hit && doc_hit {
                    Some(true)
                } else {
                    None
                };
            }
        }

        // Build the effective filter map: start from explicit `opts.filter`,
        // then fold legacy `category` / `installed_only` in only if the
        // corresponding key is not already set (explicit filter wins).
        let mut filter_map: std::collections::HashMap<String, serde_json::Value> =
            opts.filter.unwrap_or_default();
        if let Some(cat) = category {
            filter_map
                .entry("category".to_string())
                .or_insert_with(|| serde_json::Value::String(cat.to_string()));
        }
        if let Some(only) = installed_only {
            // Preserve prior semantic: `installed_only=Some(false)` was a
            // no-op (it did not force `installed=false`). Only fold when
            // explicitly true.
            if only {
                filter_map
                    .entry("installed".to_string())
                    .or_insert(serde_json::Value::Bool(true));
            }
        }

        // Resolve sort keys up-front so an invalid sort string errors out
        // before we touch results.
        let sort_str = opts.sort.as_deref().unwrap_or("-installed,name");
        let sort_keys = parse_sort(sort_str)?;

        // Resolve projection fields; this also rejects unknown `verbose`
        // values before any heavy work.
        let fields = resolve_fields(
            opts.verbose.as_deref(),
            opts.fields.as_deref(),
            HUB_SEARCH_SUMMARY,
            HUB_SEARCH_FULL,
        )?;
        let include_docstring = fields.iter().any(|f| f == "docstring");

        // Serialize each result to a Value (docstring optionally attached)
        // so filter/sort/projection work uniformly on JSON values.
        let mut items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| r.to_value_with_optional_docstring(include_docstring))
            .collect();

        // Filter AFTER serialization so filter keys can reference
        // projection-level shape (e.g. `category`, `installed`).
        if !filter_map.is_empty() {
            items.retain(|v| matches_filter(v, &filter_map));
        }

        // Sort.
        apply_sort_by_value(&mut items, &sort_keys);

        // Limit. `limit = Some(0)` means "no limit" (return all results)
        // — mirrors the `empty=all & some=filter` idiom used across the
        // list-tool contract. `None` falls back to the default cap (50).
        let total = items.len();
        let limit = opts.limit.unwrap_or(50);
        if limit > 0 {
            items.truncate(limit);
        }

        // Projection (after truncation — unselected fields are stripped
        // from the kept entries only).
        let projected: Vec<serde_json::Value> = items
            .into_iter()
            .map(|v| project_fields(v, &fields))
            .collect();

        // Collect discovered sources for transparency.
        let sources = discover_index_urls();

        let mut json = serde_json::json!({
            "results": projected,
            "total": total,
            "sources": sources,
        });
        if !warnings.is_empty() {
            json["warnings"] = serde_json::json!(warnings);
        }
        Ok(json.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_to_index_url_github() {
        assert_eq!(
            repo_to_index_url("https://github.com/ynishi/algocline-bundled-packages"),
            Some(
                "https://raw.githubusercontent.com/ynishi/algocline-bundled-packages/main/hub_index.json"
                    .to_string()
            )
        );
    }

    #[test]
    fn repo_to_index_url_github_trailing_slash() {
        assert_eq!(
            repo_to_index_url("https://github.com/user/repo/"),
            Some("https://raw.githubusercontent.com/user/repo/main/hub_index.json".to_string())
        );
    }

    #[test]
    fn repo_to_index_url_github_dot_git() {
        assert_eq!(
            repo_to_index_url("https://github.com/user/repo.git"),
            Some("https://raw.githubusercontent.com/user/repo/main/hub_index.json".to_string())
        );
    }

    #[test]
    fn repo_to_index_url_direct_json() {
        assert_eq!(
            repo_to_index_url("https://example.com/my_index.json"),
            Some("https://example.com/my_index.json".to_string())
        );
    }

    #[test]
    fn repo_to_index_url_unknown_host_no_json() {
        assert_eq!(repo_to_index_url("https://example.com/some-repo"), None);
    }

    #[test]
    fn repo_to_index_url_local_path() {
        assert_eq!(repo_to_index_url("/home/user/my-pkg"), None);
    }

    #[test]
    fn cache_key_stable() {
        let k1 = cache_key("https://example.com/index.json");
        let k2 = cache_key("https://example.com/index.json");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 16); // 16 hex chars
    }

    #[test]
    fn cache_key_different_urls() {
        let k1 = cache_key("https://a.com/index.json");
        let k2 = cache_key("https://b.com/index.json");
        assert_ne!(k1, k2);
    }

    #[test]
    fn parse_meta_flat() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(
            &path,
            r#"
local M = {}
M.meta = {
    name = "my_pkg",
    version = "1.0.0",
    description = "A test package",
    category = "reasoning",
}
return M
"#,
        )
        .unwrap();

        let result = parse_meta_from_init_lua(&path).unwrap();
        assert_eq!(result.0, "my_pkg");
        assert_eq!(result.1, "1.0.0");
        assert_eq!(result.2, "A test package");
        assert_eq!(result.3, "reasoning");
    }

    #[test]
    fn parse_meta_nested_table() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(
            &path,
            r#"
local M = {}
M.meta = {
    name = "nested_pkg",
    tags = { "a", "b" },
    description = "After nested",
}
return M
"#,
        )
        .unwrap();

        let result = parse_meta_from_init_lua(&path).unwrap();
        assert_eq!(result.0, "nested_pkg");
        assert_eq!(result.2, "After nested");
    }

    /// End-to-end sanity check against a real bundled-packages checkout.
    /// Set `BUNDLED_PACKAGES_DIR` to the repo root and run with
    /// `cargo test -- --ignored parse_meta_real_bundled`.
    #[test]
    #[ignore]
    fn parse_meta_real_bundled_packages() {
        let Ok(root) = std::env::var("BUNDLED_PACKAGES_DIR") else {
            panic!("set BUNDLED_PACKAGES_DIR=/path/to/algocline-bundled-packages");
        };
        let root = std::path::Path::new(&root);
        let mut total = 0usize;
        let mut failed_parse: Vec<String> = Vec::new();
        let mut empty_desc: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(root).unwrap().flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name.starts_with('_') {
                continue;
            }
            let init_lua = entry.path().join("init.lua");
            if !init_lua.exists() {
                continue;
            }
            total += 1;
            match parse_meta_from_init_lua(&init_lua) {
                Some((_n, _v, desc, _c)) => {
                    if desc.is_empty() {
                        empty_desc.push(name);
                    }
                }
                None => failed_parse.push(name),
            }
        }
        assert!(total >= 100, "expected ≥100 pkgs, got {total}");
        assert!(
            failed_parse.is_empty(),
            "parse_meta returned None for {} pkgs: {:?}",
            failed_parse.len(),
            failed_parse
        );
        assert!(
            empty_desc.is_empty(),
            "empty description for {} pkgs: {:?}",
            empty_desc.len(),
            empty_desc
        );
    }

    #[test]
    fn parse_meta_concat_string_literals() {
        // description = "foo " .. "bar " .. "baz" should produce "foo bar baz"
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(
            &path,
            r#"
local M = {}
M.meta = {
    name = "concat_pkg",
    version = "0.1.0",
    description = "Adaptive Branching MCTS — Thompson Sampling with dynamic "
        .. "wider/deeper decisions. GEN node mechanism for principled branching. "
        .. "Consistently outperforms standard MCTS and repeated sampling.",
    category = "reasoning",
}
return M
"#,
        )
        .unwrap();

        let result = parse_meta_from_init_lua(&path).unwrap();
        assert_eq!(result.0, "concat_pkg");
        assert_eq!(result.1, "0.1.0");
        assert_eq!(
            result.2,
            "Adaptive Branching MCTS — Thompson Sampling with dynamic \
             wider/deeper decisions. GEN node mechanism for principled branching. \
             Consistently outperforms standard MCTS and repeated sampling."
        );
        assert_eq!(result.3, "reasoning");
    }

    #[test]
    fn parse_meta_large_leading_docstring() {
        // M.meta located beyond 2KB (long leading --- docstring) must still parse.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        let mut content = String::new();
        // Generate ~4KB of leading comments
        for i in 0..120 {
            content.push_str(&format!(
                "--- line {i}: this is a long documentation comment to push M.meta beyond the old 2KB scan window\n"
            ));
        }
        content.push_str(
            r#"
local M = {}
M.meta = {
    name = "late_meta_pkg",
    version = "0.2.0",
    description = "Located past 2KB",
    category = "test",
}
return M
"#,
        );
        std::fs::write(&path, &content).unwrap();
        assert!(content.len() > 2048, "fixture should exceed 2KB");

        let result = parse_meta_from_init_lua(&path).unwrap();
        assert_eq!(result.0, "late_meta_pkg");
        assert_eq!(result.1, "0.2.0");
        assert_eq!(result.2, "Located past 2KB");
        assert_eq!(result.3, "test");
    }

    #[test]
    fn parse_meta_word_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(
            &path,
            r#"
local M = {}
M.meta = {
    name = "wb_pkg",
    short_description = "should not match",
    description = "correct one",
}
return M
"#,
        )
        .unwrap();

        let result = parse_meta_from_init_lua(&path).unwrap();
        assert_eq!(result.0, "wb_pkg");
        assert_eq!(result.2, "correct one");
    }

    #[test]
    fn merge_dedup_uses_hashset() {
        // Verify that merge correctly handles local-only packages
        // without O(n*m) behavior (structural test).
        let remote = HubIndex {
            schema_version: "hub_index/v0".into(),
            updated_at: String::new(),
            packages: vec![IndexEntry {
                name: "remote_only".into(),
                version: "1.0".into(),
                description: "from remote".into(),
                category: "test".into(),
                source: String::new(),
                card_count: 0,
                best_card: None,
                docstring: String::new(),
            }],
        };

        let results = merge(&remote);
        // Should include remote_only + any locally installed packages
        assert!(results.iter().any(|r| r.name == "remote_only"));
    }

    #[test]
    fn extract_docstring_collects_leading_comments() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(
            &path,
            r#"--- cascade — Multi-level difficulty routing with confidence gating
--- Based on: "FrugalGPT" (Chen et al., 2023)
--- Uses Thompson Sampling for budget allocation.

local M = {}
M.meta = { name = "cascade" }
return M
"#,
        )
        .unwrap();

        let doc = extract_docstring(&path);
        assert!(doc.contains("FrugalGPT"), "should contain paper ref");
        assert!(
            doc.contains("Thompson Sampling"),
            "should contain technique"
        );
        assert!(!doc.contains("local M"), "should not contain code");
    }

    #[test]
    fn extract_docstring_empty_when_no_comments() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("init.lua");
        std::fs::write(&path, "local M = {}\nreturn M\n").unwrap();

        let doc = extract_docstring(&path);
        assert!(doc.is_empty());
    }

    #[test]
    fn matches_query_searches_docstring() {
        let result = SearchResult {
            name: "cascade".into(),
            version: "0.1.0".into(),
            description: "Multi-level routing".into(),
            category: "meta".into(),
            source: String::new(),
            installed: true,
            card_count: 0,
            best_card: None,
            docstring: "Based on FrugalGPT. Uses Thompson Sampling.".into(),
            docstring_matched: None,
        };

        assert!(matches_query(&result, "thompson"), "docstring match");
        assert!(matches_query(&result, "FrugalGPT"), "docstring match case");
        assert!(matches_query(&result, "routing"), "description match");
        assert!(!matches_query(&result, "bayesian"), "no match");
    }

    // ─── SearchResult::to_value_with_optional_docstring ────────────
    //
    // `docstring` is `skip_serializing` so the default JSON view must
    // omit it, and it is re-attached only when the projection path says
    // so. These tests pin the two branches of that helper — they are the
    // hinge that `verbose="full"` / `fields=["docstring"]` rely on.

    fn sample_search_result() -> SearchResult {
        SearchResult {
            name: "cascade".into(),
            version: "0.1.0".into(),
            description: "Multi-level routing".into(),
            category: "reasoning".into(),
            source: "https://example.com/cascade".into(),
            installed: true,
            card_count: 3,
            best_card: None,
            docstring: "Based on FrugalGPT. Uses Thompson Sampling.".into(),
            docstring_matched: None,
        }
    }

    #[test]
    fn to_value_default_omits_docstring() {
        let r = sample_search_result();
        let v = r.to_value_with_optional_docstring(false);
        let obj = v.as_object().expect("object");
        assert!(
            !obj.contains_key("docstring"),
            "default summary must not leak docstring"
        );
        assert_eq!(obj.get("name").and_then(|x| x.as_str()), Some("cascade"));
        // `docstring_matched` is Option<None> → `skip_serializing_if`
        // must omit it when the query did not mark a docstring-only hit.
        assert!(
            !obj.contains_key("docstring_matched"),
            "docstring_matched=None must be omitted"
        );
    }

    #[test]
    fn to_value_include_reattaches_docstring() {
        let r = sample_search_result();
        let v = r.to_value_with_optional_docstring(true);
        let obj = v.as_object().expect("object");
        assert_eq!(
            obj.get("docstring").and_then(|x| x.as_str()),
            Some("Based on FrugalGPT. Uses Thompson Sampling.")
        );
    }

    #[test]
    fn to_value_serializes_docstring_matched_when_set() {
        let mut r = sample_search_result();
        r.docstring_matched = Some(true);
        let v = r.to_value_with_optional_docstring(false);
        let obj = v.as_object().expect("object");
        assert_eq!(
            obj.get("docstring_matched").and_then(|x| x.as_bool()),
            Some(true)
        );
    }

    // ─── projection glue ──────────────────────────────────────────
    //
    // These tests exercise the projection path that `hub_search` uses to
    // shape output: `resolve_fields` + `project_fields` applied to a
    // `to_value_with_optional_docstring`-serialized entry. They pin the
    // wf-sim-verbose contract: `fields` wins over `verbose`, default
    // summary preset excludes docstring, `full` preset includes
    // docstring, unknown keys silently skipped.

    #[test]
    fn hub_search_default_summary_excludes_docstring() {
        let r = sample_search_result();
        let fields = resolve_fields(None, None, HUB_SEARCH_SUMMARY, HUB_SEARCH_FULL).unwrap();
        let include_docstring = fields.iter().any(|f| f == "docstring");
        let v = project_fields(
            r.to_value_with_optional_docstring(include_docstring),
            &fields,
        );
        let obj = v.as_object().expect("object");
        assert!(
            !obj.contains_key("docstring"),
            "summary preset must omit docstring"
        );
        // summary preset fields that are present on the sample entry
        for key in ["name", "version", "description", "category", "installed"] {
            assert!(obj.contains_key(key), "summary preset key {key} missing");
        }
    }

    #[test]
    fn hub_search_verbose_full_includes_docstring() {
        let r = sample_search_result();
        let fields =
            resolve_fields(Some("full"), None, HUB_SEARCH_SUMMARY, HUB_SEARCH_FULL).unwrap();
        let include_docstring = fields.iter().any(|f| f == "docstring");
        let v = project_fields(
            r.to_value_with_optional_docstring(include_docstring),
            &fields,
        );
        let obj = v.as_object().expect("object");
        assert_eq!(
            obj.get("docstring").and_then(|x| x.as_str()),
            Some("Based on FrugalGPT. Uses Thompson Sampling.")
        );
        // full preset superset keys
        for key in ["source", "card_count"] {
            assert!(obj.contains_key(key), "full preset key {key} missing");
        }
    }

    #[test]
    fn hub_search_fields_beats_verbose() {
        let r = sample_search_result();
        let explicit = vec!["name".to_string(), "docstring".to_string()];
        // verbose=summary normally excludes docstring, but explicit
        // fields must win.
        let fields = resolve_fields(
            Some("summary"),
            Some(&explicit),
            HUB_SEARCH_SUMMARY,
            HUB_SEARCH_FULL,
        )
        .unwrap();
        let include_docstring = fields.iter().any(|f| f == "docstring");
        let v = project_fields(
            r.to_value_with_optional_docstring(include_docstring),
            &fields,
        );
        let obj = v.as_object().expect("object");
        assert_eq!(obj.len(), 2, "only the two requested fields");
        assert!(obj.contains_key("name"));
        assert!(obj.contains_key("docstring"));
    }

    #[test]
    fn hub_search_fields_unknown_key_silently_skipped() {
        let r = sample_search_result();
        let explicit = vec!["name".to_string(), "bogus".to_string()];
        let fields =
            resolve_fields(None, Some(&explicit), HUB_SEARCH_SUMMARY, HUB_SEARCH_FULL).unwrap();
        let v = project_fields(r.to_value_with_optional_docstring(false), &fields);
        let obj = v.as_object().expect("object");
        assert_eq!(obj.len(), 1, "bogus must not appear");
        assert!(obj.contains_key("name"));
    }

    #[test]
    fn hub_search_invalid_verbose_errors() {
        let err =
            resolve_fields(Some("fat"), None, HUB_SEARCH_SUMMARY, HUB_SEARCH_FULL).unwrap_err();
        assert!(
            err.contains("fat"),
            "error must mention the offending value"
        );
    }

    // ─── docstring_matched classification ─────────────────────────
    //
    // The query-time classification rule: `docstring_matched = Some(true)`
    // only when the query hit docstring AND missed name/description/
    // category; otherwise `None` (and therefore omitted from output).
    // The logic lives inline in `hub_search`; we re-create it here over a
    // tiny local helper so the three cases stay pinned as a contract.

    fn classify(r: &SearchResult, query: &str) -> Option<bool> {
        let ql = query.to_lowercase();
        if query.is_empty() {
            return None;
        }
        let other_hit = r.name.to_lowercase().contains(&ql)
            || r.description.to_lowercase().contains(&ql)
            || r.category.to_lowercase().contains(&ql);
        let doc_hit = r.docstring.to_lowercase().contains(&ql);
        if !other_hit && doc_hit {
            Some(true)
        } else {
            None
        }
    }

    #[test]
    fn docstring_matched_true_when_only_docstring_hits() {
        let r = sample_search_result();
        // "Thompson" appears only in docstring of the sample entry.
        assert_eq!(classify(&r, "thompson"), Some(true));
    }

    #[test]
    fn docstring_matched_none_when_name_also_hits() {
        let r = sample_search_result();
        // "cascade" hits the name; docstring match is irrelevant now.
        assert_eq!(classify(&r, "cascade"), None);
    }

    #[test]
    fn docstring_matched_none_when_description_hits() {
        let r = sample_search_result();
        // "routing" hits description; should be None.
        assert_eq!(classify(&r, "routing"), None);
    }

    #[test]
    fn docstring_matched_none_when_query_empty() {
        let r = sample_search_result();
        assert_eq!(classify(&r, ""), None);
    }

    // ─── filter fold (legacy params → filter map) ─────────────────
    //
    // Behavioural rule: legacy `category` / `installed_only=true` fold
    // into the filter map only when the corresponding key is not
    // already set (explicit `filter` wins). `installed_only=false` is a
    // no-op (preserves prior semantics).

    fn build_filter_map(
        category: Option<&str>,
        installed_only: Option<bool>,
        explicit: Option<HashMap<String, serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut filter_map = explicit.unwrap_or_default();
        if let Some(cat) = category {
            filter_map
                .entry("category".to_string())
                .or_insert_with(|| serde_json::Value::String(cat.to_string()));
        }
        if let Some(only) = installed_only {
            if only {
                filter_map
                    .entry("installed".to_string())
                    .or_insert(serde_json::Value::Bool(true));
            }
        }
        filter_map
    }

    #[test]
    fn filter_by_category_via_legacy_param() {
        let m = build_filter_map(Some("reasoning"), None, None);
        assert_eq!(
            m.get("category"),
            Some(&serde_json::Value::String("reasoning".to_string()))
        );
    }

    #[test]
    fn filter_by_installed_only_via_legacy_param() {
        let m = build_filter_map(None, Some(true), None);
        assert_eq!(m.get("installed"), Some(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn filter_installed_only_false_is_noop() {
        let m = build_filter_map(None, Some(false), None);
        assert!(
            !m.contains_key("installed"),
            "installed_only=false should not fold in"
        );
    }

    #[test]
    fn filter_beats_legacy_param_on_conflict() {
        // Explicit filter says category=meta; legacy says reasoning.
        // Explicit must win.
        let mut explicit = HashMap::new();
        explicit.insert(
            "category".to_string(),
            serde_json::Value::String("meta".to_string()),
        );
        let m = build_filter_map(Some("reasoning"), None, Some(explicit));
        assert_eq!(
            m.get("category"),
            Some(&serde_json::Value::String("meta".to_string()))
        );
    }

    #[test]
    fn filter_merges_legacy_when_no_conflict() {
        // Explicit sets a different key; legacy category should still
        // be folded in.
        let mut explicit = HashMap::new();
        explicit.insert("installed".to_string(), serde_json::Value::Bool(true));
        let m = build_filter_map(Some("reasoning"), None, Some(explicit));
        assert_eq!(
            m.get("category"),
            Some(&serde_json::Value::String("reasoning".to_string()))
        );
        assert_eq!(m.get("installed"), Some(&serde_json::Value::Bool(true)));
    }

    // ─── default sort verification ────────────────────────────────

    #[test]
    fn default_sort_is_minus_installed_name() {
        let keys = parse_sort("-installed,name").unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].key, "installed");
        assert!(keys[0].desc, "installed must sort desc (true first)");
        assert_eq!(keys[1].key, "name");
        assert!(!keys[1].desc);

        // Apply it against a small vec and confirm the expected order.
        let mut items = vec![
            serde_json::json!({"installed": false, "name": "zeta"}),
            serde_json::json!({"installed": true, "name": "mu"}),
            serde_json::json!({"installed": false, "name": "alpha"}),
            serde_json::json!({"installed": true, "name": "beta"}),
        ];
        apply_sort_by_value(&mut items, &keys);
        let names: Vec<&str> = items
            .iter()
            .map(|v| v.get("name").and_then(|x| x.as_str()).unwrap_or(""))
            .collect();
        assert_eq!(names, vec!["beta", "mu", "alpha", "zeta"]);
    }
}
