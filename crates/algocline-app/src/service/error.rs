// Note: future ServiceError variants (Manifest / Hub / Lockfile / EvalStore / Status) extend
// this enum as String->Result migration proceeds — see issue 1777125405-1441 Phase 2/3.

use thiserror::Error;

/// Top-level service-layer error type. Variants are added as `String`->`Result`
/// migration progresses (Phase 1 seeds `ProjectFiles` + `Transcript`; Phase 2/3
/// will add `Manifest`, `Hub`, `Lockfile`, `EvalStore`, `Status` sub-enums).
#[allow(dead_code)]
#[derive(Debug, Error)]
pub(crate) enum ServiceError {
    #[error(transparent)]
    ProjectFiles(#[from] ProjectFilesError),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
}

#[derive(Debug, Error)]
pub enum ProjectFilesError {
    #[error("packages dir {path}: {source}")]
    PackagesDir {
        path: String,
        source: std::io::Error,
    },
    #[error("project files lock: {0}")]
    Lock(String),
    #[error("alc.toml load: {0}")]
    AlcTomlLoad(String),
    #[error("alc.toml save: {0}")]
    AlcTomlSave(String),
    #[error("alc.lock load: {0}")]
    AlcLockLoad(String),
    #[error("alc.lock save: {0}")]
    AlcLockSave(String),
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
