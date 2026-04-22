//! Shared test helpers for `service` module tests.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use super::config::{AppConfig, LogDirSource};
use super::resolve::SearchPath;
use super::AppService;

/// Build a minimal `AppService` for tests (no search paths).
pub(super) async fn make_app_service() -> AppService {
    make_app_service_with_search_paths(vec![]).await
}

/// Build a minimal `AppService` with custom search paths.
///
/// Uses `AppConfig::from_env()` as the base so tests running under
/// `FakeHome` pick up the tempdir `$HOME` the guard installed — the Service
/// layer no longer reads `HOME` directly (Subtask 2b Inv-1), every path
/// flows from `AppConfig::app_dir()`. Subtask 2c replaces this indirection
/// with an explicit tempdir override so `FakeHome` / `HOME_MUTEX` can be
/// retired (軸 A).
pub(super) async fn make_app_service_with_search_paths(
    search_paths: Vec<SearchPath>,
) -> AppService {
    let executor = Arc::new(
        algocline_engine::Executor::new(vec![])
            .await
            .expect("executor"),
    );
    let mut log_config = AppConfig::from_env();
    log_config.log_dir = None;
    log_config.log_dir_source = LogDirSource::None;
    log_config.log_enabled = false;
    AppService::new(executor, log_config, search_paths)
}

// Serialize tests that manipulate HOME to prevent races.
static HOME_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard that sets `HOME` to a fresh tempdir for the test duration.
///
/// Acquires `HOME_MUTEX` to prevent parallel tests from racing on the
/// environment variable. Restores the previous value on drop.
///
/// Works with `#[tokio::test]` — unlike the closure-based `with_fake_home`,
/// this does not force `block_on` nesting.
pub(super) struct FakeHome {
    _tmp: tempfile::TempDir,
    _lock: MutexGuard<'static, ()>,
    prev: Option<String>,
    /// The temporary home directory path.
    pub home: PathBuf,
}

impl FakeHome {
    pub(super) fn new() -> Self {
        let lock = HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("HOME").ok();
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_path_buf();
        std::env::set_var("HOME", &home);
        Self {
            _tmp: tmp,
            _lock: lock,
            prev,
            home,
        }
    }
}

impl Drop for FakeHome {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }
}

/// Acquire the HOME mutex without changing HOME.
///
/// Use this in tests that read HOME (e.g. `is_package_installed`) to ensure
/// they do not run while a `FakeHome` is active.
pub(super) fn lock_home() -> MutexGuard<'static, ()> {
    HOME_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}
