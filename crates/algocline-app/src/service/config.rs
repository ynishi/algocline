use std::path::{Path, PathBuf};

// ─── Application Config ─────────────────────────────────────────

/// How the log directory was resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogDirSource {
    /// `ALC_LOG_DIR` environment variable.
    EnvVar,
    /// `~/.algocline/logs` (home-based default).
    Home,
    /// `$XDG_STATE_HOME/algocline/logs` or `~/.local/state/algocline/logs`.
    StateDir,
    /// Current working directory fallback.
    CurrentDir,
    /// No writable directory found — file logging disabled.
    None,
}

impl std::fmt::Display for LogDirSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EnvVar => write!(f, "ALC_LOG_DIR"),
            Self::Home => write!(f, "~/.algocline/logs"),
            Self::StateDir => write!(f, "state_dir"),
            Self::CurrentDir => write!(f, "current_dir"),
            Self::None => write!(f, "none (stderr only)"),
        }
    }
}

/// Application-wide configuration resolved from environment variables.
///
/// Log directory resolution order:
/// 1. `ALC_LOG_DIR` env var (explicit override)
/// 2. `~/.algocline/logs` (home-based default)
/// 3. `$XDG_STATE_HOME/algocline/logs` or `~/.local/state/algocline/logs`
/// 4. Current working directory (sandbox fallback)
/// 5. `None` — stderr-only mode (no file logging)
///
/// - `ALC_LOG_LEVEL`: `full` (default) or `off`.
/// - `ALC_PROMPT_PREVIEW_CHARS`: char count for `alc_status(pending_filter="preview")`
///   prompt truncation. Falls back to
///   [`algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS`] when unset or
///   unparseable. Setting `0` yields empty previews — if you want no
///   prompt at all, use the `"meta"` preset instead.
#[derive(Clone, Debug)]
pub struct AppConfig {
    /// Resolved log directory, or `None` if no writable directory is available.
    pub log_dir: Option<PathBuf>,
    pub log_dir_source: LogDirSource,
    pub log_enabled: bool,
    /// Char count for `alc_status` prompt_preview truncation.
    pub prompt_preview_chars: usize,
}

impl AppConfig {
    /// Build from environment variables (single resolution point).
    pub fn from_env() -> Self {
        let (log_dir, log_dir_source) = Self::resolve_log_dir();

        let log_enabled = std::env::var("ALC_LOG_LEVEL")
            .map(|v| v.to_lowercase() != "off")
            .unwrap_or(true);

        let prompt_preview_chars = std::env::var("ALC_PROMPT_PREVIEW_CHARS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS);

        Self {
            log_dir,
            log_dir_source,
            log_enabled,
            prompt_preview_chars,
        }
    }

    /// Resolve log directory with fallback chain.
    ///
    /// Tries each candidate in order, creating the directory if needed via
    /// [`ensure_dir`](Self::ensure_dir). Returns `(Some(path), source)` on
    /// the first writable candidate, or `(None, LogDirSource::None)` if every
    /// candidate fails.
    ///
    /// ## Fallback order
    ///
    /// 1. `ALC_LOG_DIR` env var — explicit user/operator override.
    /// 2. `~/.algocline/logs` — home-based default (most common).
    /// 3. `$XDG_STATE_HOME/algocline/logs` (or `~/.local/state/…`).
    /// 4. `<cwd>/algocline-logs` — **sandbox fallback**.
    ///    In containerised / sandbox environments (Docker, CI runners,
    ///    restricted shells) the home directory and XDG paths may not
    ///    exist or may be read-only. The current working directory is
    ///    often the only writable location available, so we fall back
    ///    to it to preserve file logging in those environments.
    /// 5. `None` — no writable directory found; file logging is disabled
    ///    and the server operates in stderr-only tracing mode.
    fn resolve_log_dir() -> (Option<PathBuf>, LogDirSource) {
        // 1. ALC_LOG_DIR env (explicit override — highest priority)
        if let Ok(dir) = std::env::var("ALC_LOG_DIR") {
            let path = PathBuf::from(dir);
            if Self::ensure_dir(&path) {
                return (Some(path), LogDirSource::EnvVar);
            }
        }

        // 2. ~/.algocline/logs (home-based default)
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".algocline").join("logs");
            if Self::ensure_dir(&path) {
                return (Some(path), LogDirSource::Home);
            }
        }

        // 3. state_dir (XDG_STATE_HOME or ~/.local/state)
        if let Some(state) = dirs::state_dir() {
            let path = state.join("algocline").join("logs");
            if Self::ensure_dir(&path) {
                return (Some(path), LogDirSource::StateDir);
            }
        }

        // 4. Current working directory (sandbox fallback — see doc above)
        if let Ok(cwd) = std::env::current_dir() {
            let path = cwd.join("algocline-logs");
            if Self::ensure_dir(&path) {
                return (Some(path), LogDirSource::CurrentDir);
            }
        }

        // 5. No writable directory — stderr-only
        (None, LogDirSource::None)
    }

    /// Try to create the directory. Returns true if it exists and is writable.
    fn ensure_dir(path: &Path) -> bool {
        std::fs::create_dir_all(path).is_ok() && path.is_dir()
    }
}
