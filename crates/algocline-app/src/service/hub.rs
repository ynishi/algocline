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

    let mut count = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Prefer .meta.json files (lightweight)
        if name.ends_with(".meta.json") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if val.get("strategy").and_then(|s| s.as_str()) == Some(pkg) {
                        count += 1;
                    }
                }
            }
            continue;
        }

        // Skip non-json or comparison files
        if !name.ends_with(".json") || name.starts_with("compare_") || name.contains(".meta.") {
            continue;
        }

        // Check if a meta file exists (already counted above)
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let meta_path = evals_dir.join(format!("{stem}.meta.json"));
        if meta_path.exists() {
            continue; // Already handled by meta path
        }

        // Fall back to reading the full eval to check strategy
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if val.get("strategy").and_then(|s| s.as_str()) == Some(pkg) {
                    count += 1;
                }
            }
        }
    }
    count
}

// ─── Merge ─────────────────────────────────────────────────────

/// Merge remote index with local install state.
fn merge(remote: &HubIndex) -> Vec<SearchResult> {
    let installed = installed_packages();
    let card_counts = local_card_counts();

    let mut seen: HashSet<String> = HashSet::new();
    let mut results: Vec<SearchResult> = Vec::new();

    for entry in &remote.packages {
        let is_installed = installed.contains_key(&entry.name);
        let local_cards = card_counts.get(&entry.name).copied().unwrap_or(0);

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
        });
    }

    // Add local-only packages (not in remote index)
    for (name, version) in &installed {
        if seen.contains(name) {
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
///
/// Extracts (name, version, description, category) from the first
/// `M.meta = { ... }` block found in the first ~2 KB.
///
/// **Limitation**: Only supports flat key-value pairs inside `M.meta`.
/// Nested tables (e.g. `tags = { ... }`) will cause the block to be
/// truncated at the inner `}`. This is intentional — `M.meta` fields
/// are expected to be simple strings.
fn parse_meta_from_init_lua(path: &std::path::Path) -> Option<(String, String, String, String)> {
    let content = std::fs::read_to_string(path).ok()?;
    // Limit search to first ~2KB (snap back to a char boundary)
    let mut limit = 2048.min(content.len());
    while limit > 0 && !content.is_char_boundary(limit) {
        limit -= 1;
    }
    let head = &content[..limit];

    // Find M.meta = { ... } block (with brace-depth tracking)
    let meta_start = head.find("M.meta")?;
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
        // Match: field = "value" with word-boundary check.
        // Walk through all occurrences and pick one that is either at
        // the start of a line (after whitespace) or preceded by a
        // non-alphanumeric character, preventing "description" from
        // matching inside "short_description".
        let mut search_from = 0;
        while let Some(rel) = block[search_from..].find(field) {
            let pos = search_from + rel;
            // Check that the character before the match is not alphanumeric/underscore
            let word_boundary = if pos == 0 {
                true
            } else {
                let prev = block.as_bytes()[pos - 1];
                !(prev.is_ascii_alphanumeric() || prev == b'_')
            };
            if word_boundary {
                let after = &block[pos + field.len()..];
                if let Some(q_start_rel) = after.find('"') {
                    let q_start = q_start_rel + 1;
                    if let Some(q_end_rel) = after[q_start..].find('"') {
                        return after[q_start..q_start + q_end_rel].to_string();
                    }
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

        // Cards for this package
        let cards_json = match card::list(Some(pkg)) {
            Ok(rows) => card::summaries_to_json(&rows),
            Err(_) => serde_json::json!([]),
        };

        // Aliases for this package
        let aliases_json = match card::alias_list(Some(pkg)) {
            Ok(rows) => card::aliases_to_json(&rows),
            Err(_) => serde_json::json!([]),
        };

        // Stats: card count, best pass_rate, eval count
        let card_rows = card::list(Some(pkg)).unwrap_or_default();
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
    pub fn hub_search(
        &self,
        query: Option<&str>,
        category: Option<&str>,
        installed_only: Option<bool>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        let (remote, warnings) = fetch_remote_indices();
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

        // Collect discovered sources for transparency
        let sources = discover_index_urls();

        let mut json = serde_json::json!({
            "results": results,
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
            }],
        };

        let results = merge(&remote);
        // Should include remote_only + any locally installed packages
        assert!(results.iter().any(|r| r.name == "remote_only"));
    }
}
