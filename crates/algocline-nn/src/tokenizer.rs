//! HuggingFace `tokenizers` wrap with first-use download cache.
//!
//! Design §6.3 / §12 Q2 policy: the tokenizer artifact (`tokenizer.json`
//! in `tokenizers` format) is fetched from the HuggingFace hub on the
//! first call and cached to `<cache_dir>/<preset>.json`. Subsequent
//! calls read straight from disk with no network access (subtask
//! invariant #2).
//!
//! Preset → HF repo mapping:
//!
//! | preset  | repo                                    |
//! |---------|-----------------------------------------|
//! | `gpt2`  | `openai-community/gpt2`                 |
//! | `llama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`    |
//!
//! Error handling follows the crate's Service-layer error-propagation
//! discipline: every failure surfaces as [`TokenizerError`] rather
//! than silently returning an empty result.

use std::path::{Path, PathBuf};

/// Errors returned by [`HfTokenizer`] APIs.
///
/// Wrapping into named variants (rather than `String`) lets the Lua
/// bridge (`engine/src/bridge/nn_card.rs`) map the failure to an
/// actionable message without inspecting substrings.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// Preset name is not known to this crate.
    #[error("unknown tokenizer preset: {0}")]
    UnknownPreset(String),
    /// Tokenizer JSON download failure.
    #[error("hub download: {0}")]
    Download(String),
    /// Local cache IO failure (create-dir / copy / read).
    #[error("cache io: {0}")]
    CacheIo(String),
    /// tokenizers-crate parse or run failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

/// Loaded pre-trained tokenizer keyed by preset name.
pub struct HfTokenizer {
    preset: String,
    inner: tokenizers::Tokenizer,
    cache_path: PathBuf,
}

impl HfTokenizer {
    /// Load `preset` from `<cache_dir>/<preset>.json`, downloading it
    /// from HuggingFace on first use (subtask invariant #2).
    ///
    /// `cache_dir` is expected to be `<app_dir>/nn/tokenizers` — the
    /// caller (`bridge/nn_card.rs`) resolves it from
    /// [`algocline_core::AppDir::nn_dir`].
    pub fn load_cached(preset: &str, cache_dir: &Path) -> Result<Self, TokenizerError> {
        let repo = repo_for_preset(preset)
            .ok_or_else(|| TokenizerError::UnknownPreset(preset.to_string()))?;

        std::fs::create_dir_all(cache_dir)
            .map_err(|e| TokenizerError::CacheIo(format!("mkdir {:?}: {e}", cache_dir)))?;
        let cache_path = cache_dir.join(format!("{preset}.json"));

        if !cache_path.exists() {
            tracing::info!(
                target: "algocline_nn::tokenizer",
                preset,
                repo,
                cache = %cache_path.display(),
                "downloading tokenizer"
            );
            crate::hub::download_to(repo, "tokenizer.json", &cache_path)
                .map_err(|e| TokenizerError::Download(e.to_string()))?;
        }

        Self::load_from_file(preset, &cache_path)
    }

    /// Load a tokenizer from an existing file on disk.
    ///
    /// Bypasses the HF download path — useful for tests that ship a
    /// fixture and never touch the network.
    pub fn load_from_file(preset: &str, path: &Path) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))?;
        Ok(Self {
            preset: preset.to_string(),
            inner,
            cache_path: path.to_path_buf(),
        })
    }

    /// Preset name this tokenizer was constructed for.
    pub fn preset(&self) -> &str {
        &self.preset
    }

    /// Cache file path (absolute).
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Encode `text` to a plain `Vec<u32>` of BPE token ids (no special
    /// tokens added — training loops decide whether to prepend BOS).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a slice of token ids back to a string. `skip_special_tokens`
    /// is `true` (matches the HF default for GPT-2).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))
    }

    /// Vocabulary size reported by the underlying `tokenizers::Tokenizer`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// Map a preset name to its HuggingFace repo. Returns `None` when the
/// preset is not recognised.
fn repo_for_preset(preset: &str) -> Option<&'static str> {
    match preset {
        "gpt2" => Some("openai-community/gpt2"),
        "llama" | "tinyllama" => Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_preset_errors() {
        let dir = std::env::temp_dir().join("alc-nn-tok-unknown");
        let _ = std::fs::remove_dir_all(&dir);
        let err = match HfTokenizer::load_cached("nonsense-preset-xyz", &dir) {
            Err(e) => e,
            Ok(_) => panic!("expected an error"),
        };
        assert!(matches!(err, TokenizerError::UnknownPreset(_)));
    }

    #[test]
    fn repo_mapping_is_stable() {
        assert_eq!(repo_for_preset("gpt2"), Some("openai-community/gpt2"));
        assert_eq!(
            repo_for_preset("llama"),
            Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        );
        assert_eq!(repo_for_preset("tinyllama"), repo_for_preset("llama"));
        assert!(repo_for_preset("unknown").is_none());
    }
}
