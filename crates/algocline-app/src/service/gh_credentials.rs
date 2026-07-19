//! GitHub push-credential diagnostics.
//!
//! Inspects the host's push-credential readiness so `alc_info` can report
//! actionable setup guidance and `alc_card_publish` can include it in typed
//! `MissingCredentials` errors.
//!
//! All subprocess failures are absorbed into per-field `error: Option<String>`
//! fields so callers never short-circuit on a single missing tool.

use serde::Serialize;
use std::path::Path;
use std::process::Command;

// ─── Public Types ────────────────────────────────────────────────

/// Aggregate report of push-credential readiness on this host.
#[derive(Debug, Serialize)]
pub(crate) struct GhCredentialReport {
    pub(crate) gh_auth: GhAuthStatus,
    pub(crate) ssh_keys: SshKeyStatus,
    pub(crate) git_config: GitConfigStatus,
    pub(crate) origin_remote: OriginRemoteStatus,
}

/// Status of `gh auth status`.
#[derive(Debug, Serialize)]
pub(crate) struct GhAuthStatus {
    /// `false` when the `gh` binary is not found in PATH.
    pub(crate) available: bool,
    /// `true` when `gh auth status` exits 0 (i.e. logged in).
    pub(crate) logged_in: bool,
    /// Populated when the binary is missing or the command fails.
    pub(crate) error: Option<String>,
}

/// Status of SSH private keys in `~/.ssh/`.
#[derive(Debug, Serialize)]
pub(crate) struct SshKeyStatus {
    /// Paths (relative to `~`) of key files that were found.
    pub(crate) found: Vec<String>,
    /// `true` when at least one key is present.
    pub(crate) any_present: bool,
}

/// Status of `git config user.{name,email}`.
#[derive(Debug, Serialize)]
pub(crate) struct GitConfigStatus {
    pub(crate) user_name: Option<String>,
    pub(crate) user_email: Option<String>,
    /// `true` when both name and email are non-empty.
    pub(crate) complete: bool,
}

/// Status of `git remote get-url origin` for the given directory.
#[derive(Debug, Serialize)]
pub(crate) struct OriginRemoteStatus {
    pub(crate) url: Option<String>,
    pub(crate) present: bool,
    /// Populated when the command fails for a reason other than
    /// "no such remote" (e.g. the directory is not a git repo).
    pub(crate) error: Option<String>,
}

// ─── Public Functions ────────────────────────────────────────────

/// Diagnose host's GitHub push-credential readiness.
///
/// Inspects (a) gh auth status, (b) SSH key presence in `~/.ssh`,
/// (c) `git config user.{name,email}`, (d) `git remote get-url origin`
/// for `app_dir`.
///
/// All subprocess failures are absorbed into per-field `error: Some(...)`
/// fields.  The function never panics.
///
/// This is a **synchronous** function and uses `std::process::Command`.
/// Call it from `logging::info()` (sync) directly.  When calling from an
/// async context (e.g. `card_publish_inner`), wrap in
/// `tokio::task::spawn_blocking`.
pub(crate) fn diagnose(app_dir: &Path, home: Option<&Path>) -> GhCredentialReport {
    GhCredentialReport {
        gh_auth: check_gh_auth(),
        ssh_keys: check_ssh_keys(home),
        git_config: check_git_config(),
        origin_remote: check_origin_remote(app_dir),
    }
}

/// Build human-readable setup guidance from a `GhCredentialReport`.
///
/// Used by both `alc_info` display and `CardPublishError::MissingCredentials`
/// so guidance text has a single source of truth.
// allow(dead_code): consumed by card_publish (Subtask 2); defined here as
// shared source of truth per design.
#[allow(dead_code)]
pub(crate) fn build_guidance(report: &GhCredentialReport) -> String {
    let mut lines: Vec<String> = Vec::new();

    if !report.gh_auth.logged_in {
        if !report.gh_auth.available {
            lines.push(
                "- Install the GitHub CLI (https://cli.github.com) and run `gh auth login`."
                    .to_string(),
            );
        } else {
            lines.push("- Run `gh auth login` to authenticate with GitHub.".to_string());
        }
    }

    if !report.ssh_keys.any_present {
        lines.push(
            "- Generate an SSH key: `ssh-keygen -t ed25519` and add it via `gh ssh-key add`."
                .to_string(),
        );
    }

    if !report.git_config.complete {
        lines.push(
            "- Configure git identity: `git config --global user.name \"...\"` and `git config --global user.email \"...\"`."
                .to_string(),
        );
    }

    if !report.origin_remote.present {
        lines.push(
            "- Set the origin remote for your project: `git remote add origin <url>`.".to_string(),
        );
    }

    if lines.is_empty() {
        "All credential checks passed; the push failure may be caused by repository permissions or branch protection rules.".to_string()
    } else {
        format!("Detected setup gaps:\n{}", lines.join("\n"))
    }
}

// ─── Private Helpers ─────────────────────────────────────────────

fn check_gh_auth() -> GhAuthStatus {
    match Command::new("gh")
        .args(["auth", "status"])
        .env("LANG", "C")
        .output()
    {
        Ok(out) if out.status.success() => GhAuthStatus {
            available: true,
            logged_in: true,
            error: None,
        },
        Ok(out) => {
            // gh auth status exits non-zero when not logged in; stderr has the message.
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // Some versions of gh print the status to stdout even on failure.
            let msg = if !stderr.is_empty() { stderr } else { stdout };
            GhAuthStatus {
                available: true,
                logged_in: false,
                error: if msg.is_empty() { None } else { Some(msg) },
            }
        }
        Err(e) => GhAuthStatus {
            available: false,
            logged_in: false,
            error: Some(e.to_string()),
        },
    }
}

fn check_ssh_keys(home: Option<&Path>) -> SshKeyStatus {
    let candidates = [
        "id_ed25519",
        "id_rsa",
        "id_ecdsa",
        "id_dsa",
        "id_ed25519_sk",
        "id_ecdsa_sk",
    ];

    let found: Vec<String> = match home {
        None => Vec::new(),
        Some(h) => {
            let ssh_dir = h.join(".ssh");
            candidates
                .iter()
                .filter(|name| ssh_dir.join(name).exists())
                .map(|name| format!("~/.ssh/{name}"))
                .collect()
        }
    };

    let any_present = !found.is_empty();
    SshKeyStatus { found, any_present }
}

fn check_git_config() -> GitConfigStatus {
    let user_name = run_git_config("user.name");
    let user_email = run_git_config("user.email");

    let complete = user_name.as_deref().map(|s| !s.is_empty()).unwrap_or(false)
        && user_email
            .as_deref()
            .map(|s| !s.is_empty())
            .unwrap_or(false);

    GitConfigStatus {
        user_name,
        user_email,
        complete,
    }
}

/// Returns `Some(value)` when the config key is set, `None` otherwise.
fn run_git_config(key: &str) -> Option<String> {
    let out = match Command::new("git")
        .args(["config", "--get", key])
        .env("LANG", "C")
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!(error = %e, key, "git config spawn failed");
            return None;
        }
    };

    if out.status.success() {
        let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    } else {
        None
    }
}

fn check_origin_remote(dir: &Path) -> OriginRemoteStatus {
    match Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(dir)
        .env("LANG", "C")
        .output()
    {
        Ok(out) if out.status.success() => {
            let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let present = !url.is_empty();
            OriginRemoteStatus {
                url: if url.is_empty() { None } else { Some(url) },
                present,
                error: None,
            }
        }
        Ok(out) => {
            // Non-zero exit: either "no such remote" or not a git repo.
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            // "No such remote 'origin'" is an expected case; treat as not-present.
            OriginRemoteStatus {
                url: None,
                present: false,
                error: if stderr.is_empty() {
                    None
                } else {
                    Some(stderr)
                },
            }
        }
        Err(e) => OriginRemoteStatus {
            url: None,
            present: false,
            error: Some(e.to_string()),
        },
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Smoke test: `diagnose` must not panic even when `app_dir` does not
    /// exist (the subprocess is for git-remote-url only and handles errors
    /// gracefully).
    #[test]
    fn diagnose_does_not_panic_with_nonexistent_dir() {
        let dir = PathBuf::from("/nonexistent/path/that/does/not/exist");
        let report = diagnose(&dir, None);
        // origin_remote must absorb the error rather than panicking.
        assert!(!report.origin_remote.present);
    }

    /// Smoke test: `diagnose` returns a structurally valid report even when
    /// a fake executable path is used for the current directory (simulates
    /// gh / git not found scenarios indirectly by checking return shape).
    #[test]
    fn diagnose_returns_valid_schema() {
        // Use the current directory — git/gh may or may not be installed,
        // but the function must return a coherent report either way.
        let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let report = diagnose(&dir, None);

        // Structural invariant: any_present must match found.is_empty() negation.
        assert_eq!(
            report.ssh_keys.any_present,
            !report.ssh_keys.found.is_empty()
        );

        // Structural invariant: complete requires both fields non-None.
        if report.git_config.complete {
            assert!(report.git_config.user_name.is_some());
            assert!(report.git_config.user_email.is_some());
        }
    }

    /// build_guidance must not panic and returns a non-empty string when
    /// some checks fail.
    #[test]
    fn build_guidance_returns_nonempty_string_for_failing_report() {
        let report = GhCredentialReport {
            gh_auth: GhAuthStatus {
                available: false,
                logged_in: false,
                error: Some("binary not found".to_string()),
            },
            ssh_keys: SshKeyStatus {
                found: Vec::new(),
                any_present: false,
            },
            git_config: GitConfigStatus {
                user_name: None,
                user_email: None,
                complete: false,
            },
            origin_remote: OriginRemoteStatus {
                url: None,
                present: false,
                error: None,
            },
        };
        let guidance = build_guidance(&report);
        assert!(!guidance.is_empty());
        assert!(guidance.contains("setup gaps") || guidance.contains("gh auth"));
    }

    /// build_guidance returns the "all checks passed" message when everything is OK.
    #[test]
    fn build_guidance_all_ok() {
        let report = GhCredentialReport {
            gh_auth: GhAuthStatus {
                available: true,
                logged_in: true,
                error: None,
            },
            ssh_keys: SshKeyStatus {
                found: vec!["~/.ssh/id_ed25519".to_string()],
                any_present: true,
            },
            git_config: GitConfigStatus {
                user_name: Some("Alice".to_string()),
                user_email: Some("alice@example.com".to_string()),
                complete: true,
            },
            origin_remote: OriginRemoteStatus {
                url: Some("https://github.com/example/repo.git".to_string()),
                present: true,
                error: None,
            },
        };
        let guidance = build_guidance(&report);
        assert!(guidance.contains("All credential checks passed"));
    }
}
