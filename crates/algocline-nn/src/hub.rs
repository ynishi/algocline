//! Minimal HuggingFace Hub single-file fetch.
//!
//! Every hub call site in this crate wants the same thing: "GET one file
//! out of a repo and put it at `dest`". The artifact is then owned by the
//! caller's own cache directory (`<cache_dir>/<preset>.json`,
//! `<cache_dir>/base/<repo>.safetensors`), so nothing downstream depends
//! on a hub client's cache layout.
//!
//! # Why not `hf-hub`
//!
//! `hf-hub` 0.3.2 (and 0.4.x / 0.5.0) resolve a download by first issuing
//! a ranged request and then *manually* following the redirect: the raw
//! `Location` header value is handed straight to the HTTP client. The Hub
//! now answers `/resolve/<rev>/<file>` with `307` and a **relative**
//! `Location` (`/api/resolve-cache/models/...`), which is not a valid
//! absolute URL, so the download aborts with `RelativeUrlWithoutBase`
//! before any bytes are read.
//!
//! `ureq` resolves a relative `Location` against the request URI per
//! RFC 9110 §10.2.2, so letting the client follow its own redirects is
//! correct by construction. `hf-hub` 1.0.0 also fixes this, but it is a
//! full rewrite onto `reqwest` whose blocking API parks a `tokio` runtime
//! on a background thread — the bridge path this crate serves is
//! deliberately runtime-free.
//!
//! # Scope
//!
//! Public repos over anonymous HTTP, plus optional `HF_TOKEN` bearer auth
//! for gated ones. `HF_ENDPOINT` overrides the host (mirrors the
//! `huggingface_hub` Python convention). No repo listing, no revision
//! pinning beyond `main`, no shared cross-process cache.

use std::path::{Path, PathBuf};

/// Default Hub host, overridable with `HF_ENDPOINT`.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Revision every preset resolves against.
const REVISION: &str = "main";

/// Failure modes of [`download_to`].
///
/// Kept separate from the caller-facing error enums (`TokenizerError` /
/// `PretrainedError`) so each call site maps transport failures into its
/// own vocabulary rather than leaking this module's shape.
#[derive(Debug, thiserror::Error)]
pub(crate) enum HubError {
    /// Transport failure — DNS, TLS, connection, redirect, or a non-2xx
    /// status (`ureq` surfaces those as errors by default).
    #[error("GET {url}: {msg}")]
    Request {
        /// Fully-qualified URL that was requested.
        url: String,
        /// Underlying client error, rendered.
        msg: String,
    },
    /// Non-2xx status that reached us without being turned into a
    /// transport error.
    #[error("GET {url}: unexpected status {status}")]
    Status {
        /// Fully-qualified URL that was requested.
        url: String,
        /// Status code as received.
        status: u16,
    },
    /// Local filesystem failure while streaming the body to `dest`.
    #[error("write {path}: {msg}")]
    Io {
        /// Destination path (or its in-progress sibling).
        path: String,
        /// Underlying IO error, rendered.
        msg: String,
    },
}

/// Download `filename` from the `main` revision of `repo` and place it at
/// `dest`, creating `dest`'s parent directory if needed.
///
/// Resolves the Hub host from `HF_ENDPOINT`, falling back to
/// <https://huggingface.co>, then delegates to [`download_from`].
pub(crate) fn download_to(repo: &str, filename: &str, dest: &Path) -> Result<(), HubError> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
    download_from(&endpoint, repo, filename, dest)
}

/// [`download_to`] with the Hub host supplied explicitly.
///
/// The body is streamed to `<dest>.partial` and renamed into place only
/// after the last byte lands, so an interrupted fetch never leaves a
/// truncated file behind — every call site guards on `dest.exists()` to
/// decide whether a download is needed, and a half-written file would
/// otherwise be mistaken for a complete one.
fn download_from(endpoint: &str, repo: &str, filename: &str, dest: &Path) -> Result<(), HubError> {
    let url = format!(
        "{}/{repo}/resolve/{REVISION}/{filename}",
        endpoint.trim_end_matches('/')
    );

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .user_agent(concat!("algocline-nn/", env!("CARGO_PKG_VERSION")))
            .build(),
    );

    let mut request = agent.get(&url);
    // Gated repos need a bearer token; public ones ignore it. Blank is
    // treated as unset so an exported-but-empty `HF_TOKEN` does not turn
    // an anonymous fetch into a 401.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
    }

    let response = request.call().map_err(|e| HubError::Request {
        url: url.clone(),
        msg: e.to_string(),
    })?;

    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(HubError::Status { url, status });
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| HubError::Io {
            path: parent.display().to_string(),
            msg: e.to_string(),
        })?;
    }

    let partial = partial_path(dest);
    let mut file = std::fs::File::create(&partial).map_err(|e| HubError::Io {
        path: partial.display().to_string(),
        msg: e.to_string(),
    })?;
    let mut reader = response.into_body().into_reader();
    let copied = std::io::copy(&mut reader, &mut file).map_err(|e| HubError::Io {
        path: partial.display().to_string(),
        msg: e.to_string(),
    });
    drop(file);
    if let Err(e) = copied {
        // Best-effort cleanup so a retry starts from a clean slate; the
        // copy failure is what the caller needs to see either way.
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }

    std::fs::rename(&partial, dest).map_err(|e| HubError::Io {
        path: dest.display().to_string(),
        msg: e.to_string(),
    })
}

/// Sibling in-progress path for `dest` (`<dest>.partial`).
///
/// Appends rather than using `Path::with_extension`, which would replace
/// `.json` / `.safetensors` and collide across artifacts in one directory.
fn partial_path(dest: &Path) -> PathBuf {
    let mut raw = dest.as_os_str().to_owned();
    raw.push(".partial");
    PathBuf::from(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_path_appends_rather_than_replaces_extension() {
        assert_eq!(
            partial_path(Path::new("/tmp/nn/gpt2.json")),
            PathBuf::from("/tmp/nn/gpt2.json.partial")
        );
        assert_eq!(
            partial_path(Path::new("/tmp/nn/base/gpt2.safetensors")),
            PathBuf::from("/tmp/nn/base/gpt2.safetensors.partial")
        );
    }

    /// An unreachable host must surface as a transport error and leave no
    /// artifact behind — neither the destination (which every call site
    /// reads as "already downloaded") nor its `.partial` sibling.
    #[test]
    fn unreachable_endpoint_errors_without_leaving_artifacts() {
        let dir = std::env::temp_dir().join("alc-nn-hub-unreachable");
        let _ = std::fs::remove_dir_all(&dir);
        let dest = dir.join("tokenizer.json");

        // Port 1 on loopback refuses connections on every supported host.
        let err = download_from("http://127.0.0.1:1", "owner/repo", "tokenizer.json", &dest)
            .expect_err("connecting to a closed port must fail");
        assert!(
            matches!(err, HubError::Request { .. }),
            "unexpected error: {err}"
        );

        assert!(
            !dest.exists(),
            "a failed download must not leave the destination in place"
        );
        assert!(
            !partial_path(&dest).exists(),
            "a failed download must not leave a .partial file behind"
        );
    }
}
