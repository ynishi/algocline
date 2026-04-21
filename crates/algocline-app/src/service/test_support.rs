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
/// Goes through `AppService::new` so that the `state_store` / `card_store`
/// fields added in Subtask 2a are populated from `log_config.app_dir()`.
/// Subtask 2c formalises tempdir-backed `app_dir` for test isolation; until
/// then `AppConfig::default()` points at a relative `./.algocline/`, which is
/// enough for call-sites that never exercise `alc.state.*` / `alc.card.*`.
pub(super) async fn make_app_service_with_search_paths(
    search_paths: Vec<SearchPath>,
) -> AppService {
    let executor = Arc::new(
        algocline_engine::Executor::new(vec![])
            .await
            .expect("executor"),
    );
    let log_config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: false,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
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
