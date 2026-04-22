//! Shared test helpers for `service` module tests.
//!
//! Every helper here roots the resulting `AppService` at an explicit
//! tempdir via `AppConfig::with_app_dir`, so tests never read or write
//! the developer's real `$HOME`. The no-arg `make_app_service()`
//! variants leak a per-call tempdir guard with `mem::forget` — the OS
//! reclaims the directory when the test binary exits, and concurrent
//! tests get their own isolated roots without shared-state contention.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::AppDir;

use super::config::AppConfig;
use super::resolve::SearchPath;
use super::AppService;

/// Build a minimal `AppService` for tests (no search paths). Roots the
/// `AppDir` at a fresh leaked tempdir so the test never touches `$HOME`.
pub(super) async fn make_app_service() -> AppService {
    make_app_service_with_search_paths(vec![]).await
}

/// `make_app_service` with a custom package search path list.
pub(super) async fn make_app_service_with_search_paths(
    search_paths: Vec<SearchPath>,
) -> AppService {
    let tmp = tempfile::tempdir().expect("test tempdir");
    let root = tmp.path().to_path_buf();
    // Leak the guard so the dir survives for the test duration. The OS
    // reclaims it when the test binary exits — equivalent lifetime to a
    // `OnceLock`-backed shared dir, but per-call so concurrent tests
    // do not race on shared paths.
    std::mem::forget(tmp);
    make_app_service_at_with_search_paths(root, search_paths).await
}

/// Build an `AppService` rooted at the caller-provided directory. Use
/// when the test asserts on paths under the `AppDir` and therefore
/// needs to know where the root lives.
pub(super) async fn make_app_service_at(root: PathBuf) -> AppService {
    make_app_service_at_with_search_paths(root, vec![]).await
}

/// `make_app_service_at` with a custom package search path list.
pub(super) async fn make_app_service_at_with_search_paths(
    root: PathBuf,
    search_paths: Vec<SearchPath>,
) -> AppService {
    let executor = Arc::new(
        algocline_engine::Executor::new(vec![])
            .await
            .expect("executor"),
    );
    let log_config = AppConfig::default().with_app_dir(root).with_log_disabled();
    AppService::new(executor, log_config, search_paths)
}

/// Build an `AppDir` rooted at `root`. Tiny helper to keep tests from
/// importing `algocline_core::AppDir` directly when they only need it
/// to drive a free-fn that takes `&AppDir`.
pub(super) fn test_app_dir(root: &std::path::Path) -> AppDir {
    AppDir::new(root.to_path_buf())
}

// `AppConfig::with_log_disabled` lives in the production module
// (`config.rs`) — tests reuse it via the builder chain above so
// log-related fields stay private to that module.
