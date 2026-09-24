# algocline-nn::tokenizer

HuggingFace `tokenizers` wrap with first-use download cache.

Design §6.3 / §12 Q2 policy: the tokenizer artifact (`tokenizer.json`
in `tokenizers` format) is fetched from the HuggingFace hub on the
first call and cached to `<cache_dir>/<preset>.json`. Subsequent
calls read straight from disk with no network access (subtask
invariant #2).

Preset → HF repo mapping:

| preset  | repo                                    |
|---------|-----------------------------------------|
| `gpt2`  | `openai-community/gpt2`                 |
| `llama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`    |

Alongside the tokenizer itself, the first-use fetch also picks up
the repo's `tokenizer_config.json`, cached as
`<cache_dir>/<preset>-config.json`. That file is where a chat model
ships its **chat template** — the Jinja2 program that turns a list of
`{role, content}` turns into the exact prompt string the model was
instruction-tuned on. Rendering it is
[`HfTokenizer::apply_chat_template`]. A repo that ships no
`tokenizer_config.json` (base models, GPT-2) stays fully usable for
encode / decode; only chat rendering refuses.

Error handling follows the crate's Service-layer error-propagation
discipline: every failure surfaces as [`TokenizerError`] rather
than silently returning an empty result.

## Types

- `HfTokenizer` — Loaded pre-trained tokenizer keyed by preset name.
- `Message` — One turn of a conversation handed to
- `TokenizerError` — Errors returned by [`HfTokenizer`] APIs.

