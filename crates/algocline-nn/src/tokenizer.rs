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
//! Alongside the tokenizer itself, the first-use fetch also picks up
//! the repo's `tokenizer_config.json`, cached as
//! `<cache_dir>/<preset>-config.json`. That file is where a chat model
//! ships its **chat template** — the Jinja2 program that turns a list of
//! `{role, content}` turns into the exact prompt string the model was
//! instruction-tuned on. Rendering it is
//! [`HfTokenizer::apply_chat_template`]. A repo that ships no
//! `tokenizer_config.json` (base models, GPT-2) stays fully usable for
//! encode / decode; only chat rendering refuses.
//!
//! Error handling follows the crate's Service-layer error-propagation
//! discipline: every failure surfaces as [`TokenizerError`] rather
//! than silently returning an empty result.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
    /// Local cache IO failure (create-dir / copy / read / parse).
    #[error("cache io: {0}")]
    CacheIo(String),
    /// tokenizers-crate parse or run failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The preset carries no chat template, so a conversation cannot be
    /// rendered into a prompt.
    ///
    /// Distinct from [`Self::ChatTemplate`] because the two ask for
    /// different reactions: this one means "pick a chat model", the other
    /// means "the template the model shipped is broken".
    #[error("no chat_template on preset {0}: its tokenizer_config.json ships none")]
    NoChatTemplate(String),
    /// Chat template compile or render failure.
    #[error("chat template: {0}")]
    ChatTemplate(String),
}

/// One turn of a conversation handed to
/// [`HfTokenizer::apply_chat_template`].
///
/// `Serialize` is what puts the turn in reach of the template: a chat
/// template addresses `message['role']` / `message['content']`, so the
/// struct is handed to the Jinja context as-is rather than being
/// flattened into strings first.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// Speaker of this turn (`system` / `user` / `assistant` / `tool`).
    ///
    /// Not an enum: the accepted set is a property of each model's
    /// template, and the callers that need a fixed set (the Lua bridge)
    /// validate against their own allowlist before building a `Message`.
    pub role: String,
    /// Turn body.
    pub content: String,
}

/// Loaded pre-trained tokenizer keyed by preset name.
pub struct HfTokenizer {
    preset: String,
    inner: tokenizers::Tokenizer,
    cache_path: PathBuf,
    /// Parsed chat template, absent when the preset ships none.
    chat_template: Option<CompiledChatTemplate>,
    /// `bos_token` / `eos_token` from `tokenizer_config.json`. Chat
    /// templates reference them by name (a Llama template ends every
    /// turn with `{{ eos_token }}`), so an absent one has to render as
    /// the empty string rather than abort the render.
    bos_token: Option<String>,
    eos_token: Option<String>,
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

            // Fetched on this branch only, so a preset already in the
            // cache never reaches the network again (invariant #2). The
            // consequence is that a cache populated before chat-template
            // support has no config beside it; deleting the cached
            // `<preset>.json` re-runs both fetches.
            let fetched = crate::hub::download_optional(
                repo,
                "tokenizer_config.json",
                &config_path_for(&cache_path),
            )
            .map_err(|e| TokenizerError::Download(e.to_string()))?;
            if !fetched {
                tracing::info!(
                    target: "algocline_nn::tokenizer",
                    preset,
                    repo,
                    "repo ships no tokenizer_config.json; this preset has no chat template"
                );
            }
        }

        Self::load_from_file(preset, &cache_path)
    }

    /// Load a tokenizer from an existing file on disk.
    ///
    /// Bypasses the HF download path — useful for tests that ship a
    /// fixture and never touch the network.
    ///
    /// The chat template is read from `path`'s `-config.json` sibling
    /// (see [`config_path_for`]) when one exists, which is the same file
    /// [`Self::load_cached`] writes.
    pub fn load_from_file(preset: &str, path: &Path) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))?;
        let chat = ChatSetup::read(&config_path_for(path))?;
        Ok(Self {
            preset: preset.to_string(),
            inner,
            cache_path: path.to_path_buf(),
            chat_template: chat.template,
            bos_token: chat.bos_token,
            eos_token: chat.eos_token,
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

    /// Render `messages` through the preset's chat template.
    ///
    /// An instruction-tuned model only behaves as advertised when its
    /// turns are wrapped in the exact control tokens it was tuned on
    /// (`<|user|>` / `[INST]` / `<|start_header_id|>` / ...), and that
    /// wrapping is model-specific. The model author ships it as a Jinja2
    /// program in `tokenizer_config.json`, so the correct prompt is
    /// whatever that program renders — hand-assembling the string here
    /// would be a guess that silently drifts per model.
    ///
    /// `add_generation_prompt` asks the template to append the opening
    /// of the assistant's turn, which is what makes the result a prompt
    /// to continue rather than a transcript to score. Templates express
    /// it as `{% if add_generation_prompt %}`; one that ignores the flag
    /// renders the same string either way.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::NoChatTemplate`] when the preset ships no
    /// template (base models, GPT-2), and [`TokenizerError::ChatTemplate`]
    /// when rendering fails. Neither degrades to a hand-built fallback:
    /// a prompt in the wrong shape produces plausible-looking garbage
    /// instead of a visible failure.
    pub fn apply_chat_template(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let template = self
            .chat_template
            .as_ref()
            .ok_or_else(|| TokenizerError::NoChatTemplate(self.preset.clone()))?;
        template.render(
            messages,
            add_generation_prompt,
            self.bos_token.as_deref().unwrap_or(""),
            self.eos_token.as_deref().unwrap_or(""),
        )
    }

    /// Vocabulary size reported by the underlying `tokenizers::Tokenizer`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Decode every token id in `0..vocab_size()` to its surface string.
    ///
    /// Built for [`crate::sampling::RegexConstraint`], which needs to know
    /// the bytes each token contributes in order to walk a DFA one
    /// candidate token at a time. The returned vector is indexed by token
    /// id and always has exactly [`Self::vocab_size`] entries, so a caller
    /// can index it with a raw id without a bounds dance.
    ///
    /// # Empty entries
    ///
    /// An entry is the empty string when the id decodes to nothing:
    /// special tokens (dropped because [`Self::decode`] passes
    /// `skip_special_tokens = true`) and ids the model has no surface form
    /// for. `tokenizers` filters those ids out *before* the decoder runs
    /// and returns `Ok("")`, so an empty entry is the expected outcome for
    /// them rather than a swallowed failure. Consumers read an empty entry
    /// as "never emit this token" — a token contributing zero bytes cannot
    /// advance a state machine, so permitting it would let a constrained
    /// generation loop spin forever.
    ///
    /// A decode error therefore means something structurally wrong (a
    /// broken decoder configuration), which is a different condition
    /// entirely, and it propagates instead of collapsing into an empty
    /// entry.
    pub fn vocab_strings(&self) -> Result<Vec<String>, TokenizerError> {
        let vocab = self.vocab_size();
        let mut out = Vec::with_capacity(vocab);
        for raw_id in 0..vocab {
            let id = u32::try_from(raw_id).map_err(|_| {
                TokenizerError::Tokenizer(format!("token id {raw_id} does not fit in u32"))
            })?;
            out.push(self.decode(&[id])?);
        }
        Ok(out)
    }
}

/// Name the chat template is registered under inside its own
/// [`minijinja::Environment`]. Each template gets a private environment,
/// so the name only has to be stable, not unique across presets.
const CHAT_TEMPLATE_NAME: &str = "chat";

/// A chat template parsed once and ready to render.
///
/// Owns its `Environment` (registered through `add_template_owned`) so
/// the compiled program outlives the string read off disk, and so a
/// syntax error in a model's template is reported when the tokenizer is
/// loaded rather than on the first conversation.
struct CompiledChatTemplate {
    env: minijinja::Environment<'static>,
}

impl CompiledChatTemplate {
    /// Parse `source`, reporting a syntax error as
    /// [`TokenizerError::ChatTemplate`].
    fn compile(source: String) -> Result<Self, TokenizerError> {
        let mut env = minijinja::Environment::new();
        env.add_template_owned(CHAT_TEMPLATE_NAME, source)
            .map_err(|e| TokenizerError::ChatTemplate(format!("compile: {e}")))?;
        Ok(Self { env })
    }

    /// Render the conversation.
    ///
    /// The context mirrors what `transformers` exposes to a chat
    /// template — `messages`, `add_generation_prompt`, `bos_token`,
    /// `eos_token` — because templates are written against those names
    /// and reference them unconditionally.
    fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
    ) -> Result<String, TokenizerError> {
        let template = self
            .env
            .get_template(CHAT_TEMPLATE_NAME)
            .map_err(|e| TokenizerError::ChatTemplate(format!("lookup: {e}")))?;
        template
            .render(minijinja::context! {
                messages,
                add_generation_prompt,
                bos_token,
                eos_token,
            })
            .map_err(|e| TokenizerError::ChatTemplate(format!("render: {e}")))
    }
}

/// The chat-relevant half of a `tokenizer_config.json`.
#[derive(Default)]
struct ChatSetup {
    template: Option<CompiledChatTemplate>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatSetup {
    /// Read and parse `path`, or return an empty setup when no config
    /// file sits there.
    ///
    /// An absent file is the documented shape of a base-model repo, so
    /// it yields "no chat template". A file that exists but does not
    /// parse is a different thing entirely — a corrupted cache entry —
    /// and propagates.
    fn read(path: &Path) -> Result<Self, TokenizerError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .map_err(|e| TokenizerError::CacheIo(format!("read {}: {e}", path.display())))?;
        let config: TokenizerConfig = serde_json::from_str(&raw)
            .map_err(|e| TokenizerError::CacheIo(format!("parse {}: {e}", path.display())))?;

        let template = match config
            .chat_template
            .and_then(ChatTemplateField::into_source)
        {
            Some(source) => Some(CompiledChatTemplate::compile(source)?),
            None => None,
        };
        Ok(Self {
            template,
            bos_token: config.bos_token.map(TokenField::into_content),
            eos_token: config.eos_token.map(TokenField::into_content),
        })
    }
}

/// The fields of `tokenizer_config.json` this crate reads.
///
/// Every other key (tokenizer class, truncation defaults, per-model
/// flags) is ignored rather than rejected: the file is written by model
/// authors and grows keys with every release, none of which change how a
/// conversation renders.
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<ChatTemplateField>,
    #[serde(default)]
    bos_token: Option<TokenField>,
    #[serde(default)]
    eos_token: Option<TokenField>,
}

/// `chat_template` is either the template itself or, on repos shipping
/// several (tool-use / RAG variants), a named list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatTemplateField {
    Single(String),
    Named(Vec<NamedChatTemplate>),
}

/// One entry of the named-list form.
#[derive(Debug, Deserialize)]
struct NamedChatTemplate {
    name: String,
    template: String,
}

impl ChatTemplateField {
    /// The template source to compile.
    ///
    /// The named form resolves to the entry called `default`, matching
    /// which one `transformers` picks when the caller names no variant.
    /// A list without a `default` entry yields `None` — selecting a
    /// variant is not part of this surface, so the preset reads as
    /// having no template rather than silently rendering someone else's
    /// tool-call format.
    fn into_source(self) -> Option<String> {
        match self {
            Self::Single(source) => Some(source),
            Self::Named(entries) => entries
                .into_iter()
                .find(|entry| entry.name == "default")
                .map(|entry| entry.template),
        }
    }
}

/// `bos_token` / `eos_token` are either the literal string or an
/// `AddedToken` object carrying it plus strip / normalise flags that
/// only matter to the tokenizer, not to the template.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenField {
    Literal(String),
    Added { content: String },
}

impl TokenField {
    fn into_content(self) -> String {
        match self {
            Self::Literal(content) | Self::Added { content } => content,
        }
    }
}

/// Path of the `tokenizer_config.json` cached beside a tokenizer file:
/// `<dir>/gpt2.json` -> `<dir>/gpt2-config.json`.
///
/// Derived from the tokenizer's own name rather than being called
/// `tokenizer_config.json`, because one cache directory holds every
/// preset side by side and the upstream name would collide.
fn config_path_for(tokenizer_path: &Path) -> PathBuf {
    let mut raw = tokenizer_path.with_extension("").into_os_string();
    raw.push("-config.json");
    PathBuf::from(raw)
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

    /// Smallest tokenizer artifact that still round-trips through
    /// `tokenizers`: a WordLevel model with four ids and no decoder. Kept
    /// inline so the test never reaches the network.
    const WORD_LEVEL_FIXTURE: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "hello": 0, "world": 1, "!": 2, "[UNK]": 3 },
    "unk_token": "[UNK]"
  }
}"#;

    /// `vocab_strings` must cover the whole id space with no gaps —
    /// `RegexConstraint` indexes the result by raw token id, so a short
    /// vector would silently deny every id past its end.
    #[test]
    fn vocab_strings_covers_every_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, WORD_LEVEL_FIXTURE).unwrap();

        let tok = HfTokenizer::load_from_file("fixture", &path).unwrap();
        let vocab = tok.vocab_strings().unwrap();

        assert_eq!(vocab.len(), tok.vocab_size(), "one entry per token id");
        assert_eq!(vocab[0], "hello");
        assert_eq!(vocab[1], "world");
        assert_eq!(vocab[2], "!");
    }

    /// Chat-capable `tokenizer_config.json` in the shape a small chat
    /// model ships: a literal `bos_token`, an `AddedToken`-object
    /// `eos_token`, and a template that wraps each turn and honours
    /// `add_generation_prompt`.
    const CHAT_CONFIG: &str = r#"{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token": "<s>",
  "eos_token": { "content": "</s>", "lstrip": false, "rstrip": false },
  "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}{{ eos_token }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
}"#;

    /// Parses fine (filters resolve at render time) and then fails on
    /// the first render — the shape a template referencing a Jinja
    /// extension this engine does not carry would have.
    const BROKEN_RENDER_CONFIG: &str = r#"{
  "chat_template": "{{ messages | no_such_filter }}"
}"#;

    /// Tokenizer built from the inline fixture, with `config` written to
    /// the `-config.json` sibling `load_from_file` reads.
    fn fixture_tokenizer(dir: &Path, config: Option<&str>) -> Result<HfTokenizer, TokenizerError> {
        let path = dir.join("fixture.json");
        std::fs::write(&path, WORD_LEVEL_FIXTURE).unwrap();
        if let Some(config) = config {
            std::fs::write(config_path_for(&path), config).unwrap();
        }
        HfTokenizer::load_from_file("fixture", &path)
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message {
                role: "system".into(),
                content: "be brief".into(),
            },
            Message {
                role: "user".into(),
                content: "hello".into(),
            },
        ]
    }

    /// The rendered prompt is the template's output verbatim — turn
    /// markers, special tokens and the trailing assistant opening
    /// included. Asserted against the full string rather than a
    /// `contains`, because a prompt that is merely *nearly* right is
    /// exactly the failure this surface exists to prevent.
    #[test]
    fn chat_template_renders_the_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let tok = fixture_tokenizer(dir.path(), Some(CHAT_CONFIG)).unwrap();

        let rendered = tok.apply_chat_template(&conversation(), true).unwrap();
        assert_eq!(
            rendered,
            "<s><|system|>\nbe brief</s>\n<|user|>\nhello</s>\n<|assistant|>\n"
        );
    }

    /// `add_generation_prompt = false` drops the assistant opening: the
    /// flag has to reach the template, not be swallowed by the wrapper.
    #[test]
    fn add_generation_prompt_reaches_the_template() {
        let dir = tempfile::tempdir().unwrap();
        let tok = fixture_tokenizer(dir.path(), Some(CHAT_CONFIG)).unwrap();

        let with = tok.apply_chat_template(&conversation(), true).unwrap();
        let without = tok.apply_chat_template(&conversation(), false).unwrap();

        assert!(with.ends_with("<|assistant|>\n"), "rendered: {with}");
        assert!(!without.ends_with("<|assistant|>\n"), "rendered: {without}");
        assert_eq!(
            with,
            format!("{without}<|assistant|>\n"),
            "the flag must change nothing but the trailing opening"
        );
    }

    /// `bos_token` / `eos_token` come from `tokenizer_config.json` and
    /// are reachable by name, in both shapes the file uses (a literal
    /// string and an `AddedToken` object).
    #[test]
    fn special_tokens_are_visible_to_the_template() {
        let dir = tempfile::tempdir().unwrap();
        let tok = fixture_tokenizer(dir.path(), Some(CHAT_CONFIG)).unwrap();

        let rendered = tok.apply_chat_template(&conversation(), false).unwrap();
        assert!(rendered.starts_with("<s>"), "bos missing: {rendered}");
        assert_eq!(
            rendered.matches("</s>").count(),
            2,
            "one eos per turn: {rendered}"
        );
    }

    /// A preset without a template refuses by name instead of returning
    /// a plain concatenation — a caller handed a base model must find
    /// out here, not from a model answering nonsense.
    #[test]
    fn missing_chat_template_errors() {
        let dir = tempfile::tempdir().unwrap();
        let tok = fixture_tokenizer(dir.path(), None).unwrap();

        let err = tok
            .apply_chat_template(&conversation(), true)
            .expect_err("a preset with no chat template must refuse");
        assert!(
            matches!(err, TokenizerError::NoChatTemplate(ref p) if p == "fixture"),
            "unexpected error: {err}"
        );
        assert!(err.to_string().contains("no chat_template on preset"));
    }

    /// A template that parses but fails mid-render propagates the Jinja
    /// error rather than yielding a partial prompt.
    #[test]
    fn chat_template_render_failure_errors() {
        let dir = tempfile::tempdir().unwrap();
        let tok = fixture_tokenizer(dir.path(), Some(BROKEN_RENDER_CONFIG)).unwrap();

        let err = tok
            .apply_chat_template(&conversation(), true)
            .expect_err("an unresolvable filter must fail the render");
        assert!(
            matches!(err, TokenizerError::ChatTemplate(_)),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("render"),
            "the message must place the failure: {err}"
        );
    }

    #[test]
    fn config_path_sits_beside_the_tokenizer() {
        assert_eq!(
            config_path_for(Path::new("/tmp/nn/tokenizers/gpt2.json")),
            PathBuf::from("/tmp/nn/tokenizers/gpt2-config.json")
        );
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
