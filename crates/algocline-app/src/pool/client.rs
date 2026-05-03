//! UDS client for connecting to a pool worker process.
//!
//! [`PoolClient`] is the MCP-side (AppService-side) handle that opens a Unix
//! domain socket connection to a worker subprocess and exchanges
//! [`PoolRequest`] / [`PoolResponse`] messages in JSON-line format.

use std::path::Path;

use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter},
    net::UnixStream,
    sync::Mutex,
};

use crate::pool::{
    error::PoolError,
    protocol::{PoolRequest, PoolResponse, PoolResponseData},
};

/// Version string embedded in every handshake to prevent client/server skew.
pub const POOL_PROTOCOL_VERSION: &str = env!("CARGO_PKG_VERSION");

// ─── Internal state ───────────────────────────────────────────────────────────

struct Inner {
    writer: BufWriter<tokio::net::unix::OwnedWriteHalf>,
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
}

// ─── PoolClient ───────────────────────────────────────────────────────────────

/// A thin UDS client for communicating with a pool worker process.
///
/// Each `PoolClient` instance owns a single Unix domain socket connection to
/// one worker. Messages are JSON lines (`\n`-terminated).
///
/// # Lifecycle
///
/// 1. Call [`PoolClient::connect`] — opens the socket and runs the version
///    handshake atomically.  If the handshake fails the connection is dropped
///    and an error is returned; no `PoolClient` instance is produced.
/// 2. Call [`PoolClient::send_request`] for each message.
/// 3. Drop the `PoolClient` when done; the underlying socket is closed.
pub struct PoolClient {
    inner: Mutex<Inner>,
}

impl PoolClient {
    /// Open a connection to a worker at `sock_path` and verify protocol version.
    ///
    /// The handshake (`PoolRequest::Handshake`) is performed inside this call.
    /// If the worker reports a different version, `Err(PoolError::VersionMismatch)`
    /// is returned and no `PoolClient` is constructed.
    ///
    /// # Errors
    ///
    /// - `PoolError::Connect` — socket connect failed (wraps `std::io::Error`).
    /// - `PoolError::Handshake` — response could not be parsed as valid JSON.
    /// - `PoolError::VersionMismatch` — worker version differs from client version.
    pub async fn connect(sock_path: &Path) -> Result<Self, PoolError> {
        let stream = UnixStream::connect(sock_path).await?;
        let (read_half, write_half) = stream.into_split();

        let mut inner = Inner {
            writer: BufWriter::new(write_half),
            reader: BufReader::new(read_half),
        };

        // Perform version handshake before returning.
        let handshake_req = PoolRequest::Handshake {
            version: POOL_PROTOCOL_VERSION.to_string(),
        };
        send_line(&mut inner, &handshake_req).await?;
        let resp = recv_line(&mut inner).await?;

        match &resp.data {
            Some(PoolResponseData::Handshake { version }) => {
                if version != POOL_PROTOCOL_VERSION {
                    return Err(PoolError::VersionMismatch {
                        client: POOL_PROTOCOL_VERSION.to_string(),
                        server: version.clone(),
                    });
                }
            }
            _ => {
                return Err(PoolError::Handshake(
                    "unexpected handshake response".to_string(),
                ));
            }
        }

        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Send a [`PoolRequest`] over the Unix domain socket and await the response.
    ///
    /// Serialises the request to a JSON line (`\n`-terminated), writes it via
    /// `tokio::io::AsyncWriteExt::write_all`, then reads the response with
    /// `tokio::io::AsyncBufReadExt::read_line`.
    ///
    /// # Concurrency
    ///
    /// A `tokio::sync::Mutex` guards the `BufWriter`/`BufReader` pair so that
    /// concurrent calls from multiple tasks are serialised. Only one request can
    /// be in-flight per `PoolClient` instance at a time.
    ///
    /// **Cancel safety of `read_line`**: `AsyncBufReadExt::read_line` is **not**
    /// cancel safe. If this future is dropped before `read_line` completes, the
    /// internal buffer may hold a partial line. After cancellation the connection
    /// is no longer usable; callers must drop this `PoolClient` and reconnect.
    ///
    /// **Send + Sync**: `PoolClient` is `Send` (all fields are `Send`). It is
    /// **not** `Sync` by default — callers should wrap in `Arc<Mutex<PoolClient>>`
    /// if shared across tasks.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::Connect` on I/O failure, `PoolError::Handshake` if the
    /// response cannot be parsed as valid JSON, `PoolError::VersionMismatch` if
    /// the server reports an incompatible version.
    ///
    /// # Panics
    ///
    /// Does not panic.
    pub async fn send_request(&mut self, req: PoolRequest) -> Result<PoolResponse, PoolError> {
        let inner = self.inner.get_mut();
        send_line(inner, &req).await?;
        recv_line(inner).await
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Write a JSON-line for `req` and flush.
async fn send_line(inner: &mut Inner, req: &PoolRequest) -> Result<(), PoolError> {
    let mut line = serde_json::to_string(req).map_err(|e| PoolError::Handshake(e.to_string()))?;
    line.push('\n');
    inner.writer.write_all(line.as_bytes()).await?;
    inner.writer.flush().await?;
    Ok(())
}

/// Read one JSON line and deserialise it as [`PoolResponse`].
async fn recv_line(inner: &mut Inner) -> Result<PoolResponse, PoolError> {
    let mut buf = String::new();
    inner.reader.read_line(&mut buf).await?;
    serde_json::from_str(buf.trim_end_matches('\n'))
        .map_err(|e| PoolError::Handshake(format!("response parse error: {e}")))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tokio::net::UnixListener;

    use super::*;
    use crate::pool::protocol::PoolResponseData;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Return a temporary directory and a socket path within it.
    fn temp_sock() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("worker.sock");
        (dir, sock)
    }

    /// Spawn a mock server that processes one connection.
    ///
    /// `handler` receives a mutable reference to an Inner and must process
    /// exactly the messages the test expects, then return.
    async fn spawn_server<F, Fut>(listener: UnixListener, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce(Inner) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let (r, w) = stream.into_split();
            let inner = Inner {
                writer: BufWriter::new(w),
                reader: BufReader::new(r),
            };
            handler(inner).await;
        })
    }

    /// Write a `PoolResponse` as a JSON line to `inner`.
    async fn server_send(inner: &mut Inner, resp: &PoolResponse) {
        let mut line = serde_json::to_string(resp).expect("serialize");
        line.push('\n');
        inner
            .writer
            .write_all(line.as_bytes())
            .await
            .expect("write");
        inner.writer.flush().await.expect("flush");
    }

    /// Read one request line from `inner`.
    async fn server_recv(inner: &mut Inner) -> PoolRequest {
        let mut buf = String::new();
        inner
            .reader
            .read_line(&mut buf)
            .await
            .expect("server read_line");
        serde_json::from_str(buf.trim_end_matches('\n')).expect("server deserialize")
    }

    // ── test: happy-path round-trip ───────────────────────────────────────────

    /// Full round-trip: handshake → run (paused) → continue → shutdown.
    ///
    /// The mock server matches the protocol version and returns canned responses
    /// for each op. Verifies that `PoolClient` correctly forwards every response
    /// back to the caller without re-interpreting the payload.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn round_trip_handshake_run_pause_continue_shutdown() {
        let (_dir, sock_path) = temp_sock();
        // Clean up any leftover socket from a previous run (EADDRINUSE guard).
        let _ = std::fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("bind");

        let server_handle = spawn_server(listener, |mut inner| async move {
            // 1. Handshake
            let req = server_recv(&mut inner).await;
            assert!(matches!(req, PoolRequest::Handshake { .. }));
            server_send(
                &mut inner,
                &PoolResponse::success(PoolResponseData::Handshake {
                    version: POOL_PROTOCOL_VERSION.to_string(),
                }),
            )
            .await;

            // 2. Run → Paused (FeedResult represented as raw JSON value)
            let req = server_recv(&mut inner).await;
            assert!(matches!(req, PoolRequest::Run { .. }));
            let feed_result = serde_json::json!({
                "type": "paused",
                "session_id": "test-sid",
                "prompt": "hi",
                "query_id": "q1"
            });
            server_send(
                &mut inner,
                &PoolResponse::success(PoolResponseData::Feed {
                    session_id: "test-sid".to_string(),
                    feed_result,
                }),
            )
            .await;

            // 3. Continue → Finished
            let req = server_recv(&mut inner).await;
            assert!(matches!(req, PoolRequest::Continue { .. }));
            let feed_result = serde_json::json!({"type": "finished", "output": "done"});
            server_send(
                &mut inner,
                &PoolResponse::success(PoolResponseData::Feed {
                    session_id: "test-sid".to_string(),
                    feed_result,
                }),
            )
            .await;

            // 4. Shutdown
            let req = server_recv(&mut inner).await;
            assert!(matches!(req, PoolRequest::Shutdown));
            server_send(
                &mut inner,
                &PoolResponse::success(PoolResponseData::Shutdown),
            )
            .await;
        })
        .await
        .expect("spawn_server");

        // --- Client side ---
        let mut client = PoolClient::connect(&sock_path).await.expect("connect");

        // Run
        let resp = client
            .send_request(PoolRequest::Run {
                code: "return alc.llm('hi')".to_string(),
                ctx: None,
                lib_paths: vec![],
            })
            .await
            .expect("run");
        assert!(resp.ok);
        assert!(matches!(resp.data, Some(PoolResponseData::Feed { .. })));

        // Continue
        let resp = client
            .send_request(PoolRequest::Continue {
                sid: "test-sid".to_string(),
                response: "ok".to_string(),
                query_id: Some("q1".to_string()),
                usage: None,
            })
            .await
            .expect("continue");
        assert!(resp.ok);
        assert!(matches!(resp.data, Some(PoolResponseData::Feed { .. })));

        // Shutdown
        let resp = client
            .send_request(PoolRequest::Shutdown)
            .await
            .expect("shutdown");
        assert!(resp.ok);
        assert!(matches!(resp.data, Some(PoolResponseData::Shutdown)));

        server_handle.await.expect("server task");
    }

    // ── test: version mismatch ────────────────────────────────────────────────

    /// When the worker responds with a different version, `PoolClient::connect`
    /// must return `Err(PoolError::VersionMismatch)` and no `PoolClient` is
    /// constructed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn version_mismatch_returns_pool_error() {
        let (_dir, sock_path) = temp_sock();
        let _ = std::fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("bind");

        let server_handle = spawn_server(listener, |mut inner| async move {
            // Consume the handshake request.
            let _ = server_recv(&mut inner).await;
            // Reply with a deliberately wrong version.
            server_send(
                &mut inner,
                &PoolResponse::success(PoolResponseData::Handshake {
                    version: "999.0.0".to_string(),
                }),
            )
            .await;
        })
        .await
        .expect("spawn_server");

        let err = PoolClient::connect(&sock_path)
            .await
            .expect_err("should fail with version mismatch");

        assert!(
            matches!(
                err,
                PoolError::VersionMismatch {
                    ref client,
                    ref server
                } if client == POOL_PROTOCOL_VERSION && server == "999.0.0"
            ),
            "unexpected error: {err:?}"
        );

        server_handle.await.expect("server task");
    }
}
