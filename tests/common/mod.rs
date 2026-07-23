//! Test harness for e2e tests.
//!
//! `TempAlcHome` spawns the `alc` binary with `ALC_HOME` (and
//! `ALC_PACKAGES_PATH`) pointing at a per-test tempdir, so that
//! `installed.json` / `config.toml` / `packages/` are isolated from the
//! developer's real `~/.algocline/`. Sibling issues `213cad4a` (pkg leak)
//! and `efd45582` (info snap contamination) are structurally resolved by
//! routing every affected test through this harness.
//!
//! ## Drop order
//!
//! Field declaration order is significant. Rust drops struct fields in
//! declaration order (Rust Reference §5.8.1). We declare:
//!   1. `client` — dropped first: cancels the MCP session and terminates
//!      the child `alc` process.
//!   2. `_tmp` — dropped next: removes the tempdir once the child no
//!      longer holds file handles under it.
//!   3. `home` — plain `PathBuf`; ordering is irrelevant.
//!
//! ## Concurrency
//!
//! `tokio::process::Command::env` sets env vars **only for the child
//! process**; the parent test binary's process-wide env is untouched.
//! Therefore no `serial_test` crate / `Mutex` guard is required and
//! parallel tests using this harness are race-free.

#![allow(dead_code)] // Not every test file consumes every helper.

use std::path::{Path, PathBuf};

use rmcp::{transport::TokioChildProcess, ServiceExt};
use tempfile::TempDir;

/// RAII harness that spawns an `alc` MCP server rooted at a fresh tempdir.
///
/// The tempdir survives for the lifetime of `TempAlcHome`. Drop order is
/// documented on the field list; do not reorder fields.
pub struct TempAlcHome {
    /// MCP client bound to the spawned `alc` child.
    ///
    /// Declared first so it drops first — `RunningService::drop` terminates
    /// the child process, releasing any file handles inside the tempdir
    /// before the tempdir itself is removed.
    pub client: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    /// TempDir handle; drop removes the directory tree at
    /// `{home}/**`. Must come after `client`.
    _tmp: TempDir,
    /// Absolute path to the tempdir root (equals `ALC_HOME`).
    home: PathBuf,
}

impl TempAlcHome {
    /// Spawn `alc` with `ALC_HOME` (and `ALC_PACKAGES_PATH`) pointing at a
    /// fresh tempdir, then initialize an MCP session over stdio.
    ///
    /// Pre-seeds `{home}/packages/` and `{home}/types/alc.d.lua` before spawn:
    ///
    /// - `resolve_lib_paths()` in `src/main.rs` filters `ALC_PACKAGES_PATH`
    ///   entries through `Path::is_dir()`. Without a pre-existing packages
    ///   directory the env var is silently dropped and `require()` falls back
    ///   to the developer's real `~/.algocline/packages/` — which defeats the
    ///   isolation guarantee and breaks tests that install then `require()`
    ///   (e.g. `test_alc_fork_roundtrip`).
    /// - `types_stub_path()` in `crates/algocline-app/src/service/resolve.rs`
    ///   returns `None` unless `{app_dir}/types/alc.d.lua` exists on disk. An
    ///   empty stub file is enough to make `alc_pkg_install` include the
    ///   `types_path` field in its response.
    ///
    /// Panics on tempdir / spawn / MCP-init failure — this is a test-only
    /// setup helper, so a clear stack trace is preferred over a `Result`.
    pub async fn connect() -> Self {
        let tmp = TempDir::new().expect("failed to create tempdir");
        let home = tmp.path().to_path_buf();
        let packages_path = home.join("packages");
        // Pre-create packages/ so resolve_lib_paths()'s is_dir() filter keeps
        // ALC_PACKAGES_PATH in the require search chain.
        std::fs::create_dir_all(&packages_path).expect("failed to create packages dir");
        // Pre-create an empty types stub so types_stub_path() returns Some(_).
        let types_dir = home.join("types");
        std::fs::create_dir_all(&types_dir).expect("failed to create types dir");
        std::fs::write(types_dir.join("alc.d.lua"), b"").expect("failed to seed alc.d.lua");
        let bin = std::env::var("CARGO_BIN_EXE_alc")
            .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")));
        let mut cmd = tokio::process::Command::new(bin);
        cmd.env("ALC_HOME", &home)
            .env("ALC_PACKAGES_PATH", &packages_path);
        let transport = TokioChildProcess::new(cmd).expect("failed to spawn alc server");
        let client = ().serve(transport).await.expect("failed to initialize MCP session");
        Self {
            client,
            _tmp: tmp,
            home,
        }
    }

    /// Absolute path to the tempdir root (`ALC_HOME`).
    pub fn home_path(&self) -> &Path {
        &self.home
    }

    /// `{home}/installed.json` — pkg manifest location under this harness.
    pub fn installed_json(&self) -> PathBuf {
        self.home.join("installed.json")
    }

    /// `{home}/packages` — installed pkg tree under this harness.
    pub fn packages_dir(&self) -> PathBuf {
        self.home.join("packages")
    }

    /// Explicit shutdown: cancel MCP session and consume `self` so the
    /// tempdir drop runs deterministically at the call site.
    pub async fn cancel(self) {
        self.client.cancel().await.expect("cancel failed");
    }
}
