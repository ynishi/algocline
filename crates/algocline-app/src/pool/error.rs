/// Errors produced by the process-pool IPC layer.
///
/// All variants are propagated to callers via `Result<T, PoolError>`.
/// At the `AppService` boundary (Subtask 5/6) these are converted to
/// `String` to satisfy the existing `EngineApi` wire contract.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// Unix-domain socket connection failed.
    #[error("failed to connect to UDS socket: {0}")]
    Connect(#[from] std::io::Error),

    /// Socket read or write failed after connection was established.
    #[error("failed to read/write UDS socket: {0}")]
    Io(String),

    /// A response line from the worker could not be parsed as valid JSON.
    #[error("failed to parse worker response: {0}")]
    ResponseParse(String),

    /// `registry.json` could not be parsed (corrupt or schema mismatch).
    ///
    /// Treating this as "empty registry" would silently kill live workers;
    /// callers must propagate this error.
    #[error("registry.json corrupted: {0}")]
    RegistryCorrupted(String),

    /// Spawning the worker subprocess failed.
    #[error("worker spawn failed: {0}")]
    Spawn(String),

    /// Version handshake with the worker was rejected or timed out.
    #[error("worker handshake failed: {0}")]
    Handshake(String),

    /// The requested session ID is not present in the pool.
    #[error("session not found in pool: {0}")]
    SessionNotFound(String),

    /// The worker was built with a different crate version.
    #[error("version mismatch: client={client}, server={server}")]
    VersionMismatch { client: String, server: String },
}
