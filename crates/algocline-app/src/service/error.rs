use thiserror::Error;

/// Top-level service-layer error type.
#[derive(Debug, Error)]
pub(crate) enum ServiceError {
    #[error(transparent)]
    ProjectFiles(#[from] ProjectFilesError),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    #[error(transparent)]
    PkgList(#[from] PkgListError),
    #[error(transparent)]
    HubRegistries(#[from] HubRegistriesError),
    #[error(transparent)]
    CardPublish(#[from] CardPublishError),
    /// User-supplied parameter was invalid (sort key, field name, etc.).
    /// This is a client error (bad input), not a server-side failure.
    #[error("{0}")]
    InvalidInput(String),
}

impl From<ServiceError> for String {
    fn from(e: ServiceError) -> Self {
        e.to_string()
    }
}

#[derive(Debug, Error)]
pub enum ProjectFilesError {
    #[error("packages dir {path}: {source}")]
    PackagesDir {
        path: String,
        source: std::io::Error,
    },
    /// Advisory lock acquisition failed.
    #[error("project files lock: {0}")]
    Lock(#[from] crate::service::lock::LockError),
    #[error("alc.toml load: {0}")]
    AlcTomlLoad(String),
    #[error("alc.toml save: {0}")]
    AlcTomlSave(String),
    #[error("alc.lock load: {0}")]
    AlcLockLoad(String),
    #[error("alc.lock save: {0}")]
    AlcLockSave(String),
}

/// Errors arising from `pkg_list` filesystem reads.
///
/// Corruption (parse error / version mismatch) is distinguished from file-absent
/// (`load_lockfile` / `load_alc_toml` return `Ok(None)` for absent files) so
/// callers can surface the former as a `warnings` field in the MCP wire response.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Error)]
pub(crate) enum PkgListError {
    #[error("alc.lock parse: {0}")]
    LockfileParse(String),
    #[error("alc.toml parse: {0}")]
    AlcTomlParse(String),
    #[error("alc.local.toml parse: {0}")]
    AlcLocalTomlParse(String),
}

/// Errors arising from `hub_registries.json` reads.
///
/// File-absent is `Ok(HubRegistries::default())` — the file is created lazily.
/// Parse failures (corrupt JSON) are `Err` so callers can surface them in the
/// MCP wire `warnings` field instead of silently degrading hub discovery.
#[derive(Debug, Error)]
pub(crate) enum HubRegistriesError {
    #[error("hub_registries.json parse: {0}")]
    Parse(String),
}

#[derive(Debug, Error)]
pub enum TranscriptError {
    #[error("transcript log dir {path}: {source}")]
    LogDir {
        path: String,
        source: std::io::Error,
    },
    #[error("transcript path: {0}")]
    Path(String),
    #[error("transcript serialize: {0}")]
    Serialize(String),
    #[error("transcript write {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
}

/// Errors arising from `card_publish` operations.
///
/// Push failures due to missing credentials are distinguished from other
/// errors so callers can surface actionable setup guidance.  A successful
/// push is never rolled back when only reindex fails — that outcome is
/// represented as a separate field in `CardPublishOutcome`.
#[derive(Debug, Error)]
pub enum CardPublishError {
    /// Git push failed due to missing or invalid credentials.
    ///
    /// `guidance` contains actionable setup steps derived from
    /// `gh_credentials::diagnose()` + `build_guidance()`.
    #[error("missing credentials: {guidance}")]
    MissingCredentials { guidance: String },
    /// The requested card was not found in the card store.
    #[error("card '{0}' not found")]
    CardNotFound(String),
    /// A git subprocess (add/commit/push/rev-parse) failed for a reason
    /// other than credential authentication.
    #[error("git {cmd} failed: {stderr}")]
    GitCommand { cmd: String, stderr: String },
    /// Subprocess spawn failure or temporary directory creation error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// `target_repo` is not a valid URL (e.g. a bare pkg slug was supplied).
    #[error("invalid target_repo: {0}")]
    InvalidTarget(String),
}
