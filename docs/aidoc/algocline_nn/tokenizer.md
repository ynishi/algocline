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

Error handling follows the crate's Service-layer error-propagation
discipline: every failure surfaces as [`TokenizerError`] rather
than silently returning an empty result.

## Types

- `HfTokenizer` — Loaded pre-trained tokenizer keyed by preset name.
- `TokenizerError` — Errors returned by [`HfTokenizer`] APIs.

