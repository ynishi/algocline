//! Integration-style tests for the `pkg_*` methods on `AppService`.

use crate::service::lockfile::{load_lockfile, LockFile, LockPackage};
use crate::service::source::PackageSource;
use crate::service::test_support::{make_app_service, make_app_service_with_search_paths};

fn make_lock_with_pkg(name: &str) -> LockFile {
    LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: name.to_string(),
            version: None,
            source: PackageSource::Installed,
        }],
    }
}

// ── pkg_list tests ───────────────────────────────────────────

#[tokio::test]
async fn pkg_list_with_project() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Create alc.toml declaring the project-local package.
    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nmy_local_pkg = \"*\"\n",
    )
    .unwrap();

    // Create a project-local package.
    let pkg_dir = project_root.join("my_local_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    // Write alc.lock with a Path entry for the package.
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

    // Create alc.toml declaring the package to remove.
    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nmy_local_pkg = \"*\"\n",
    )
    .unwrap();

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
            None, // version
        )
        .await
        .unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(json["removed"], "my_local_pkg");
    // New response has alc_toml and alc_lock fields (no scope field).
    assert!(json["alc_toml"].is_string());
    assert!(json["alc_lock"].is_string());

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

    // Create alc.toml with a different package (not the target).
    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nother_pkg = \"*\"\n",
    )
    .unwrap();

    // Write an alc.lock without the target package.
    let lock = make_lock_with_pkg("other_pkg");
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_remove(
            "nonexistent_pkg",
            Some(project_root.to_string_lossy().to_string()),
            None, // version
        )
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found in alc.lock"));
}

// ── resolved_source_path / resolved_source_kind / override_paths tests ────

/// Case 1: project `path` entry — `resolved_source_path` is the canonicalized
/// absolute path of the package directory; `resolved_source_kind = "local_path"`.
#[tokio::test]
async fn pkg_list_project_path_entry_has_resolved_source() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Create a vendor package directory inside the project.
    let pkg_dir = project_root.join("my_vendor_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    // alc.toml with path dependency.
    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nmy_vendor_pkg = { path = \"my_vendor_pkg\" }\n",
    )
    .unwrap();

    // alc.lock with Path source.
    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "my_vendor_pkg".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "my_vendor_pkg".to_string(),
            },
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "my_vendor_pkg")
        .expect("my_vendor_pkg not found");

    let expected_canonical = std::fs::canonicalize(&pkg_dir)
        .unwrap()
        .display()
        .to_string();

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should be canonicalized path"
    );
    assert_eq!(pkg["resolved_source_kind"], "local_path");
}

/// Case 2: project `path` entry where the vendor directory is itself a symlink —
/// `resolved_source_path` follows the symlink to the real target.
#[tokio::test]
async fn pkg_list_project_path_with_symlink_vendor_follows_target() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Create a real package directory somewhere else.
    let real_pkg = tmp.path().join("real_pkg_dir");
    std::fs::create_dir_all(&real_pkg).unwrap();
    std::fs::write(real_pkg.join("init.lua"), "return {}").unwrap();

    // Create a symlink inside the project pointing to the real dir.
    let symlink_in_project = project_root.join("sym_vendor_pkg");
    std::os::unix::fs::symlink(&real_pkg, &symlink_in_project).unwrap();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nsym_vendor_pkg = { path = \"sym_vendor_pkg\" }\n",
    )
    .unwrap();

    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "sym_vendor_pkg".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "sym_vendor_pkg".to_string(),
            },
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "sym_vendor_pkg")
        .expect("sym_vendor_pkg not found");

    // canonicalize follows the symlink to real_pkg.
    let expected_canonical = std::fs::canonicalize(&real_pkg)
        .unwrap()
        .display()
        .to_string();

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should resolve through symlink to real target"
    );
    assert_eq!(pkg["resolved_source_kind"], "local_path");
}

/// Case 3: project `installed` entry — `resolved_source_path` is
/// `{packages_dir()}/{name}` canonicalized; `resolved_source_kind = "installed"`.
#[tokio::test]
async fn pkg_list_project_installed_entry_has_resolved_source() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();
    let packages_dir = fake_home.home.join(".algocline").join("packages");
    let pkg_dir = packages_dir.join("installed_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\ninstalled_pkg = \"*\"\n",
    )
    .unwrap();

    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "installed_pkg".to_string(),
            version: None,
            source: PackageSource::Installed,
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let expected_canonical = std::fs::canonicalize(&pkg_dir)
        .unwrap()
        .display()
        .to_string();
    drop(fake_home);

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "installed_pkg")
        .expect("installed_pkg not found");

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should be packages_dir/<name> canonicalized"
    );
    assert_eq!(pkg["resolved_source_kind"], "installed");
}

/// Case 4 (light): project `installed` entry where `packages_dir/{name}` is itself
/// a symlink (linked package). The resolved path follows through to the real target.
#[tokio::test]
async fn pkg_list_project_installed_resolves_through_linked_pkg() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();
    let packages_dir = fake_home.home.join(".algocline").join("packages");
    std::fs::create_dir_all(&packages_dir).unwrap();

    // The "real" development directory (what the symlink points to).
    let real_dev_dir = fake_home.home.join("dev").join("linked_pkg_real");
    std::fs::create_dir_all(&real_dev_dir).unwrap();
    std::fs::write(real_dev_dir.join("init.lua"), "return {}").unwrap();

    // Symlink in packages_dir pointing to the dev dir.
    let symlink_path = packages_dir.join("linked_pkg");
    std::os::unix::fs::symlink(&real_dev_dir, &symlink_path).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nlinked_pkg = \"*\"\n",
    )
    .unwrap();

    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "linked_pkg".to_string(),
            version: None,
            source: PackageSource::Installed,
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    // canonicalize follows the symlink to the real dev dir.
    let expected_canonical = std::fs::canonicalize(&real_dev_dir)
        .unwrap()
        .display()
        .to_string();
    drop(fake_home);

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "linked_pkg")
        .expect("linked_pkg not found");

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should follow symlink in packages_dir to real target"
    );
    assert_eq!(pkg["resolved_source_kind"], "installed");
}

/// Case 5: global regular (non-symlink) package —
/// `resolved_source_path = canonicalize({search_path}/{name})`,
/// `resolved_source_kind = "installed"` (no manifest entry → "installed").
#[tokio::test]
async fn pkg_list_global_regular_pkg_has_resolved_source() {
    let tmp = tempfile::tempdir().unwrap();
    let search_dir = tmp.path().join("pkgs");
    std::fs::create_dir_all(&search_dir).unwrap();

    let pkg_dir = search_dir.join("regular_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    let search_path = crate::service::resolve::SearchPath {
        path: search_dir.clone(),
        source: crate::service::resolve::SearchPathSource::Env,
    };
    let svc = make_app_service_with_search_paths(vec![search_path]).await;
    let result = svc.pkg_list(None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let pkg = packages
        .iter()
        .find(|p| p["name"] == "regular_pkg")
        .expect("regular_pkg not found");

    let expected_canonical = std::fs::canonicalize(&pkg_dir)
        .unwrap()
        .display()
        .to_string();

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical
    );
    assert_eq!(pkg["resolved_source_kind"], "installed");
}

/// Case 6: global linked (symlink) package —
/// `resolved_source_path` is the canonicalized symlink target;
/// `resolved_source_kind = "linked"`.
#[tokio::test]
async fn pkg_list_global_linked_pkg_resolves_to_link_target() {
    let tmp = tempfile::tempdir().unwrap();
    let search_dir = tmp.path().join("pkgs");
    std::fs::create_dir_all(&search_dir).unwrap();

    // Real package directory (dev workspace).
    let real_dir = tmp.path().join("my_dev_pkg");
    std::fs::create_dir_all(&real_dir).unwrap();
    std::fs::write(real_dir.join("init.lua"), "return {}").unwrap();

    // Symlink in the search_dir pointing to real_dir.
    let link_path = search_dir.join("linked_global_pkg");
    std::os::unix::fs::symlink(&real_dir, &link_path).unwrap();

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
        .find(|p| p["name"] == "linked_global_pkg")
        .expect("linked_global_pkg not found");

    let expected_canonical = std::fs::canonicalize(&real_dir)
        .unwrap()
        .display()
        .to_string();

    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should point to real target"
    );
    assert_eq!(pkg["resolved_source_kind"], "linked");
    assert_eq!(pkg["linked"], true);
}

/// Case 7: global linked package with a dangling (broken) symlink —
/// `resolved_source_path` must be absent; `resolved_source_kind = "linked"`;
/// `broken = true`.
#[tokio::test]
async fn pkg_list_global_linked_broken_omits_resolved_source() {
    let tmp = tempfile::tempdir().unwrap();
    let search_dir = tmp.path().join("pkgs");
    std::fs::create_dir_all(&search_dir).unwrap();

    // Create a symlink pointing to a nonexistent path.
    let nonexistent_target = tmp.path().join("this_does_not_exist");
    let link_path = search_dir.join("broken_pkg");
    std::os::unix::fs::symlink(&nonexistent_target, &link_path).unwrap();

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
        .find(|p| p["name"] == "broken_pkg")
        .expect("broken_pkg not found");

    assert!(
        pkg.get("resolved_source_path").is_none() || pkg["resolved_source_path"].is_null(),
        "resolved_source_path must be absent for broken symlink"
    );
    assert_eq!(pkg["resolved_source_kind"], "linked");
    assert_eq!(pkg["broken"], true);
}

/// Case 8: two global search paths contain a package with the same name —
/// `override_paths` on the active entry lists the shadowed path(s) in
/// search-path order.
#[tokio::test]
async fn pkg_list_override_paths_global_shadow() {
    let tmp = tempfile::tempdir().unwrap();
    let search_dir1 = tmp.path().join("pkgs1");
    let search_dir2 = tmp.path().join("pkgs2");
    std::fs::create_dir_all(&search_dir1).unwrap();
    std::fs::create_dir_all(&search_dir2).unwrap();

    // Same package name in both search paths.
    for dir in [&search_dir1, &search_dir2] {
        let pkg_dir = dir.join("dup_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();
    }

    let svc = make_app_service_with_search_paths(vec![
        crate::service::resolve::SearchPath {
            path: search_dir1.clone(),
            source: crate::service::resolve::SearchPathSource::Env,
        },
        crate::service::resolve::SearchPath {
            path: search_dir2.clone(),
            source: crate::service::resolve::SearchPathSource::Env,
        },
    ])
    .await;

    let result = svc.pkg_list(None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    // Active entry is the one from search_dir1 (first wins).
    let active_pkg = packages
        .iter()
        .find(|p| p["name"] == "dup_pkg" && p["active"] == true)
        .expect("active dup_pkg not found");

    let override_paths = active_pkg["override_paths"]
        .as_array()
        .expect("override_paths should be an array on active entry");

    assert_eq!(
        override_paths.len(),
        1,
        "should have exactly one shadowed entry"
    );

    let expected_shadow = std::fs::canonicalize(search_dir2.join("dup_pkg"))
        .unwrap()
        .display()
        .to_string();

    assert_eq!(
        override_paths[0].as_str().unwrap(),
        expected_shadow,
        "override_paths[0] should be the canonicalized path in search_dir2"
    );
}

/// Case 9: project entry shadows a global entry —
/// the project entry's `override_paths` contains the global package path;
/// the inactive global entry has no `override_paths`.
#[tokio::test]
async fn pkg_list_override_paths_project_shadows_global() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();
    let search_dir = tmp.path().join("global_pkgs");
    std::fs::create_dir_all(&search_dir).unwrap();

    // Global package directory.
    let global_pkg_dir = search_dir.join("shared_pkg");
    std::fs::create_dir_all(&global_pkg_dir).unwrap();
    std::fs::write(global_pkg_dir.join("init.lua"), "return {}").unwrap();

    // Project vendor package directory.
    let local_pkg_dir = project_root.join("shared_pkg");
    std::fs::create_dir_all(&local_pkg_dir).unwrap();
    std::fs::write(local_pkg_dir.join("init.lua"), "return {}").unwrap();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nshared_pkg = { path = \"shared_pkg\" }\n",
    )
    .unwrap();

    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "shared_pkg".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "shared_pkg".to_string(),
            },
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    let svc = make_app_service_with_search_paths(vec![crate::service::resolve::SearchPath {
        path: search_dir.clone(),
        source: crate::service::resolve::SearchPathSource::Env,
    }])
    .await;

    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    // Project entry (active, scope = "project") must have override_paths with the global path.
    let project_entry = packages
        .iter()
        .find(|p| p["name"] == "shared_pkg" && p["scope"] == "project")
        .expect("project shared_pkg not found");

    let override_paths = project_entry["override_paths"]
        .as_array()
        .expect("project entry should have override_paths listing shadowed global");

    let expected_global_canonical = std::fs::canonicalize(&global_pkg_dir)
        .unwrap()
        .display()
        .to_string();

    assert!(
        override_paths
            .iter()
            .any(|p| p.as_str().unwrap() == expected_global_canonical),
        "project override_paths should include the global pkg canonical path"
    );

    // Inactive global entry must NOT have override_paths.
    let global_entry = packages
        .iter()
        .find(|p| p["name"] == "shared_pkg" && p["scope"] == "global")
        .expect("global shared_pkg not found");

    assert_eq!(
        global_entry["active"], false,
        "global entry should be inactive"
    );
    let global_map = global_entry
        .as_object()
        .expect("global entry must be object");
    assert!(
        !global_map.contains_key("override_paths"),
        "inactive global entry must not have override_paths, got: {:?}",
        global_map.get("override_paths")
    );
}

/// Regression: a project `installed` entry must not list its own backing
/// directory (`packages_dir/{name}`) in `override_paths` just because the
/// global search paths include `packages_dir`. The entry's own
/// `resolved_source_path` canonicalizes to the same location, so that
/// occurrence is not a genuine shadow and must be filtered out.
#[tokio::test]
async fn pkg_list_project_installed_does_not_self_shadow() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();
    let packages_dir = fake_home.home.join(".algocline").join("packages");
    let pkg_dir = packages_dir.join("self_shadow_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nself_shadow_pkg = \"*\"\n",
    )
    .unwrap();

    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "self_shadow_pkg".to_string(),
            version: None,
            source: PackageSource::Installed,
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    // Include packages_dir as a search path — this is the real production
    // topology (see `resolve_lib_paths` in src/main.rs).
    let svc = make_app_service_with_search_paths(vec![crate::service::resolve::SearchPath {
        path: packages_dir.clone(),
        source: crate::service::resolve::SearchPathSource::Default,
    }])
    .await;

    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    drop(fake_home);

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let project_entry = packages
        .iter()
        .find(|p| p["name"] == "self_shadow_pkg" && p["scope"] == "project")
        .expect("project self_shadow_pkg not found");

    let entry_map = project_entry
        .as_object()
        .expect("project entry must be object");

    assert!(
        !entry_map.contains_key("override_paths"),
        "project `installed` entry must not list its own backing dir as override_paths, got: {:?}",
        entry_map.get("override_paths")
    );
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

    // resolved_source_path must still be populated even without manifest
    // registration — filesystem access is independent of installed.json.
    let expected_canonical = std::fs::canonicalize(&pkg_dir)
        .unwrap()
        .display()
        .to_string();
    assert_eq!(
        pkg["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should be populated regardless of manifest state"
    );
    // Unregistered packages default to "installed" kind (not "bundled").
    assert_eq!(
        pkg["resolved_source_kind"], "installed",
        "unregistered global package should default to installed kind"
    );
}

// ── variant scope (alc.local.toml) ───────────────────────────

/// `alc.local.toml` declares a variant pkg → it appears in `pkg_list` with
/// `scope: "variant"`, `resolved_source_kind: "variant"`, `active: true`,
/// and `path` set to the absolute pkg dir.
#[tokio::test]
async fn pkg_list_variant_pkg_appears_with_variant_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Variant pkg lives outside the project root (typical worktree workflow).
    let pkg_dir = tmp.path().join("variant_src").join("my_variant_pkg");
    std::fs::create_dir_all(&pkg_dir).unwrap();
    std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

    std::fs::write(
        project_root.join("alc.local.toml"),
        format!(
            "[packages]\nmy_variant_pkg = {{ path = \"{}\" }}\n",
            pkg_dir.display()
        ),
    )
    .unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let entry = packages
        .iter()
        .find(|p| p["name"] == "my_variant_pkg")
        .expect("my_variant_pkg not found in pkg_list output");

    assert_eq!(entry["scope"], "variant");
    assert_eq!(entry["active"], true);
    assert_eq!(entry["source_type"], "path");
    assert_eq!(entry["resolved_source_kind"], "variant");

    let expected_canonical = std::fs::canonicalize(&pkg_dir)
        .unwrap()
        .display()
        .to_string();
    assert_eq!(
        entry["resolved_source_path"].as_str().unwrap(),
        expected_canonical,
        "resolved_source_path should canonicalize to the variant pkg dir"
    );
    assert_eq!(
        entry["path"].as_str().unwrap(),
        pkg_dir.display().to_string(),
        "path should be the absolute pkg_dir as declared in alc.local.toml"
    );
}

/// A variant pkg shadowing a same-name global package: the variant entry
/// is `active: true`, the global one is demoted to `active: false`.
#[tokio::test]
async fn pkg_list_variant_shadows_global() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();
    let global_dir = tmp.path().join("global_pkgs");
    std::fs::create_dir_all(&global_dir).unwrap();

    // Global pkg of the same name.
    let global_pkg = global_dir.join("shared");
    std::fs::create_dir_all(&global_pkg).unwrap();
    std::fs::write(global_pkg.join("init.lua"), "return {}").unwrap();

    // Variant pkg.
    let variant_pkg = tmp.path().join("variant_src").join("shared");
    std::fs::create_dir_all(&variant_pkg).unwrap();
    std::fs::write(variant_pkg.join("init.lua"), "return {}").unwrap();

    std::fs::write(
        project_root.join("alc.local.toml"),
        format!(
            "[packages]\nshared = {{ path = \"{}\" }}\n",
            variant_pkg.display()
        ),
    )
    .unwrap();

    let svc = make_app_service_with_search_paths(vec![crate::service::resolve::SearchPath {
        path: global_dir.clone(),
        source: crate::service::resolve::SearchPathSource::Env,
    }])
    .await;

    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let variant_entry = packages
        .iter()
        .find(|p| p["name"] == "shared" && p["scope"] == "variant")
        .expect("variant 'shared' entry not found");
    assert_eq!(variant_entry["active"], true);

    let global_entry = packages
        .iter()
        .find(|p| p["name"] == "shared" && p["scope"] == "global")
        .expect("global 'shared' entry not found");
    assert_eq!(
        global_entry["active"], false,
        "global entry must be demoted when shadowed by variant"
    );
}

/// A variant pkg with the same name as an `alc.toml`-declared project pkg:
/// the variant entry wins (`active: true`), the project entry is demoted.
#[tokio::test]
async fn pkg_list_variant_shadows_project() {
    let tmp = tempfile::tempdir().unwrap();
    let project_root = tmp.path();

    // Project pkg via alc.toml + alc.lock (path entry).
    let project_pkg = project_root.join("shared");
    std::fs::create_dir_all(&project_pkg).unwrap();
    std::fs::write(project_pkg.join("init.lua"), "return {}").unwrap();
    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nshared = { path = \"shared\" }\n",
    )
    .unwrap();
    let lock = LockFile {
        version: 1,
        packages: vec![LockPackage {
            name: "shared".to_string(),
            version: None,
            source: PackageSource::Path {
                path: "shared".to_string(),
            },
        }],
    };
    crate::service::lockfile::save_lockfile(project_root, &lock).unwrap();

    // Variant override.
    let variant_pkg = tmp.path().join("variant_src").join("shared");
    std::fs::create_dir_all(&variant_pkg).unwrap();
    std::fs::write(variant_pkg.join("init.lua"), "return {}").unwrap();
    std::fs::write(
        project_root.join("alc.local.toml"),
        format!(
            "[packages]\nshared = {{ path = \"{}\" }}\n",
            variant_pkg.display()
        ),
    )
    .unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_list(Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let packages = json["packages"].as_array().unwrap();

    let variant_entry = packages
        .iter()
        .find(|p| p["name"] == "shared" && p["scope"] == "variant")
        .expect("variant 'shared' entry not found");
    assert_eq!(variant_entry["active"], true);

    let project_entry = packages
        .iter()
        .find(|p| p["name"] == "shared" && p["scope"] == "project")
        .expect("project 'shared' entry not found");
    assert_eq!(
        project_entry["active"], false,
        "project entry must be demoted when shadowed by variant"
    );
}

// ── pkg_repair tests ────────────────────────────────────────

/// (B) installed dir missing: pkg_install populates manifest, then we delete
/// the dest dir, then pkg_repair must restore it via reinstall.
#[tokio::test]
async fn pkg_repair_reinstalls_missing_installed_dir() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();

    // Build a source pkg dir outside of HOME.
    let source = fake_home.home.join("src_repo").join("repair_pkg");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(
        source.join("init.lua"),
        "return { meta = { version = '0.1.0' } }",
    )
    .unwrap();

    let svc = make_app_service().await;

    // Initial install — populates installed.json and creates dest dir.
    svc.pkg_install(source.display().to_string(), None)
        .await
        .expect("initial install");

    let dest = fake_home
        .home
        .join(".algocline")
        .join("packages")
        .join("repair_pkg");
    assert!(dest.exists(), "dest must exist after install");

    // Simulate breakage: remove the dest dir.
    std::fs::remove_dir_all(&dest).unwrap();
    assert!(!dest.exists());

    // Repair — should re-run install from manifest source.
    let result = svc.pkg_repair(None, None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let repaired = json["repaired"].as_array().expect("repaired array");
    assert_eq!(repaired.len(), 1, "exactly one repair, got: {json}");
    assert_eq!(repaired[0]["name"], "repair_pkg");
    assert_eq!(repaired[0]["kind"], "installed_missing");
    assert_eq!(repaired[0]["action"], "reinstall");
    assert!(dest.exists(), "dest must be restored after repair");
}

/// Healthy package — manifest entry + dest exist → Skipped.
#[tokio::test]
async fn pkg_repair_skips_healthy_pkg() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();

    let source = fake_home.home.join("src_repo").join("healthy_pkg");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("init.lua"), "return {}").unwrap();

    let svc = make_app_service().await;
    svc.pkg_install(source.display().to_string(), None)
        .await
        .unwrap();

    let result = svc.pkg_repair(None, None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        json["repaired"].as_array().unwrap().is_empty(),
        "no repair expected"
    );
    let skipped = json["skipped"].as_array().unwrap();
    assert!(
        skipped.iter().any(|e| e["name"] == "healthy_pkg"),
        "healthy_pkg must be in skipped, got: {json}"
    );
}

/// (A) global symlink dangling — surfaced as unrepairable.
#[tokio::test]
async fn pkg_repair_reports_dangling_symlink_as_unrepairable() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();

    // Create the packages dir and a dangling symlink in it.
    let pkg_dir = fake_home.home.join(".algocline").join("packages");
    std::fs::create_dir_all(&pkg_dir).unwrap();

    let target = fake_home.home.join("does_not_exist");
    let link = pkg_dir.join("dangling_pkg");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let svc = make_app_service().await;
    let result = svc.pkg_repair(None, None).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let unrepairable = json["unrepairable"].as_array().expect("unrepairable array");
    let entry = unrepairable
        .iter()
        .find(|e| e["name"] == "dangling_pkg")
        .expect("dangling_pkg must surface as unrepairable");
    assert_eq!(entry["kind"], "symlink_dangling");
    assert!(
        entry["suggestion"]
            .as_str()
            .unwrap()
            .contains("alc_pkg_unlink"),
        "suggestion should mention alc_pkg_unlink"
    );
}

/// (C) project-scope `path = ...` declared in alc.toml but the path doesn't
/// exist on disk — surfaced as unrepairable with `scope: "project"`.
#[tokio::test]
async fn pkg_repair_reports_project_path_missing_as_unrepairable() {
    use crate::service::test_support::FakeHome;

    // FakeHome acquires HOME_MUTEX at struct-field level — matches the
    // other pkg_repair tests and avoids `await_holding_lock` on a local
    // `MutexGuard` binding.
    let fake_home = FakeHome::new();
    let project_root = fake_home.home.join("proj");
    std::fs::create_dir_all(&project_root).unwrap();

    std::fs::write(
        project_root.join("alc.toml"),
        "[packages]\nghost = { path = \"missing_dir\" }\n",
    )
    .unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_repair(None, Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let unrepairable = json["unrepairable"].as_array().unwrap();
    let entry = unrepairable
        .iter()
        .find(|e| e["name"] == "ghost" && e["scope"] == "project")
        .unwrap_or_else(|| panic!("ghost must surface as project path_missing, got: {json}"));
    assert_eq!(entry["kind"], "path_missing");
    assert!(entry["suggestion"].as_str().unwrap().contains("alc.toml"));
}

/// (D) variant-scope `path = ...` declared in alc.local.toml but the path
/// doesn't exist on disk — surfaced as unrepairable with `scope: "variant"`.
#[tokio::test]
async fn pkg_repair_reports_variant_path_missing_as_unrepairable() {
    use crate::service::test_support::FakeHome;

    let fake_home = FakeHome::new();
    let project_root = fake_home.home.join("proj");
    std::fs::create_dir_all(&project_root).unwrap();

    let absent = project_root.join("nope_pkg");
    std::fs::write(
        project_root.join("alc.local.toml"),
        format!(
            "[packages]\nnope_pkg = {{ path = \"{}\" }}\n",
            absent.display()
        ),
    )
    .unwrap();

    let svc = make_app_service().await;
    let result = svc
        .pkg_repair(None, Some(project_root.to_string_lossy().to_string()))
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let unrepairable = json["unrepairable"].as_array().unwrap();
    let entry = unrepairable
        .iter()
        .find(|e| e["name"] == "nope_pkg" && e["scope"] == "variant")
        .expect("nope_pkg must surface as variant path_missing");
    assert_eq!(entry["kind"], "path_missing");
    assert!(entry["suggestion"]
        .as_str()
        .unwrap()
        .contains("alc_pkg_unlink"));
}

/// `name` filter that matches nothing → Err with informative message.
#[tokio::test]
async fn pkg_repair_unknown_name_returns_error() {
    use crate::service::test_support::FakeHome;

    // FakeHome isolates HOME so this test doesn't depend on whether the
    // developer happens to have a package with the probe name installed.
    let _fake_home = FakeHome::new();

    let svc = make_app_service().await;
    let err = svc
        .pkg_repair(Some("nonexistent_pkg".to_string()), None)
        .await
        .unwrap_err();
    assert!(
        err.contains("nonexistent_pkg"),
        "error should mention the missing name, got: {err}"
    );
}
