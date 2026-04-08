//! Integration-style tests for the `pkg_*` methods on `AppService`.

use crate::service::lockfile::{load_lockfile, LockFile, LockPackage};
use crate::service::source::PackageSource;
use crate::service::AppService;

fn make_lock_with_pkg(name: &str) -> LockFile {
    LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: name.to_string(),
            version: None,
            source: PackageSource::Path {
                path: format!("packages/{name}"),
            },
        }],
    }
}

async fn make_app_service() -> AppService {
    make_app_service_with_search_paths(vec![]).await
}

async fn make_app_service_with_search_paths(
    search_paths: Vec<crate::service::resolve::SearchPath>,
) -> AppService {
    use std::sync::Arc;

    let executor = Arc::new(
        algocline_engine::Executor::new(vec![])
            .await
            .expect("executor"),
    );
    AppService {
        executor,
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: crate::service::config::AppConfig {
            log_dir: None,
            log_dir_source: crate::service::config::LogDirSource::None,
            log_enabled: false,
        },
        search_paths,
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    }
}

// ── pkg_list tests ───────────────────────────────────────────

#[tokio::test]
async fn pkg_list_with_project() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Create a project-local package.
    let pkg_dir = project_root.join("my_local_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    // Write alc.lock.
    let lock = make_lock_with_pkg("my_local_pkg");
    // Adjust path to be relative.
    let lock = LockFile {
        packages: vec![LockPackage {
            name: "my_local_pkg".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "my_local_pkg".to_string(),
            },
        }],
        ..lock
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    // Should have the project-local package.
    let project_pkg = packages
        .iter()
        .find(|p| p["name"] == "my_local_pkg")
        .expect("my_local_pkg not found in pkg_list output");

    assert_eq!(project_pkg["scope"], "project");
    assert_eq!(project_pkg["source_type"], "path");
    assert_eq!(project_pkg["active"], true);

    // project_root and lockfile_path must be present.
    assert!(json["project_root"].is_string());
    assert!(json["lockfile_path"].is_string());
}

#[tokio::test]
async fn pkg_list_no_project_root() {
    let svc = make_app_service().await;

    // Should succeed even without project_root (no crash).
    let result = svc.pkg_list(None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert!(json["packages"].is_array());
}

// ── pkg_remove tests ─────────────────────────────────────────

#[tokio::test]
async fn pkg_remove_project_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Create the physical directory (should remain after removal).
    let pkg_dir = project_root.join("my_local_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    // Write alc.lock with the package.
    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "my_local_pkg".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "my_local_pkg".to_string(),
            },
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_remove(
            "my_local_pkg",
            Some(project_root.to_string_lossy().to_string()),
            None,
        )
        .await
        .unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(json["removed"], "my_local_pkg");
    assert_eq!(json["scope"], "project");

    // Physical directory must still exist.
    assert!(pkg_dir.exists(), "physical directory was deleted");

    // alc.lock must no longer contain the entry.
    let lock_after = load_lockfile(project_root).unwrap().unwrap();
    assert!(
        lock_after.packages.is_empty(),
        "alc.lock still contains the entry"
    );
}

#[tokio::test]
async fn pkg_remove_project_scope_not_found_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Write an alc.lock without the target package.
    let lock = make_lock_with_pkg("other_pkg");
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_remove(
            "nonexistent_pkg",
            Some(project_root.to_string_lossy().to_string()),
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found in alc.lock"));
}

/// A global package that exists on disk but is NOT registered in
/// `installed.json` must NOT emit a `source_type` field.
///
/// Previously the code wrote `source_type: "global"` (an invalid enum
/// value) as a placeholder. After the typed DTO rewrite, absent manifest
/// entries leave `source_type` out of the output entirely.
#[tokio::test]
async fn pkg_list_global_unregistered_has_no_source_type() {
    let tmp = tempfile::tempdir().unwrap();
    let search_dir = tmp.path().join("pkgs");
    std::fs::create_dir_all(&search_dir).unwrap();

    // Create a package directory with init.lua — but do NOT write
    // installed.json (simulating a hand-copied / ALC_PACKAGES_PATH package).
    let pkg_dir = search_dir.join("hand_copied_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(
        pkg_dir.join("init.lua"),
        "return { meta = { name = 'hand_copied_pkg' } }",
    )
    .unwrap();

    let search_path = crate::service::resolve::SearchPath {
        path: search_dir,
        source: crate::service::resolve::SearchPathSource::Env,
    };
    let svc = make_app_service_with_search_paths(vec![search_path]).await;
    let result = svc.pkg_list(None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "hand_copied_pkg")
        .expect("hand_copied_pkg not found in pkg_list output");

    // source_type must be absent (not "global" or any other invalid value).
    // serde_json::Value's Index impl returns Null for missing keys; we check
    // the underlying map directly to distinguish "absent" from "null".
    let pkg_map = pkg
        .as_object()
        .expect("package entry must be a JSON object");
    assert!(
        !pkg_map.contains_key("source_type"),
        "source_type should be absent for unregistered package, got: {:?}",
        pkg_map.get("source_type")
    );
    assert_eq!(pkg["scope"], "global");
    assert_eq!(pkg["active"], true);
}
