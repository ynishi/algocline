//! Integration tests for `HfTokenizer`.
//!
//! The default set does not touch the network — it exercises the error
//! paths (unknown preset, missing file) via the public crate surface.
//! The full HuggingFace round-trip (download `gpt2` tokenizer, encode +
//! decode "hello world") is gated behind `#[ignore]` so `cargo test`
//! stays offline by default; run with
//! `cargo test -p algocline-nn --test tokenizer_roundtrip -- --ignored`
//! when the network is available.

use algocline_nn::tokenizer::{HfTokenizer, TokenizerError};

#[test]
fn load_cached_unknown_preset_errors() {
    let cache_dir = std::env::temp_dir().join("alc-nn-tokroundtrip-unknown");
    let _ = std::fs::remove_dir_all(&cache_dir);
    // `HfTokenizer` does not implement `Debug`, so we cannot use
    // `.expect_err()`; deconstruct via `match` instead.
    let err = match HfTokenizer::load_cached("no-such-preset", &cache_dir) {
        Err(e) => e,
        Ok(_) => panic!("unknown preset must error"),
    };
    assert!(matches!(err, TokenizerError::UnknownPreset(_)));
}

#[test]
fn load_from_file_missing_path_errors() {
    let missing = std::env::temp_dir().join("alc-nn-tokroundtrip-missing.json");
    let _ = std::fs::remove_file(&missing);
    let err = match HfTokenizer::load_from_file("gpt2", &missing) {
        Err(e) => e,
        Ok(_) => panic!("missing tokenizer file must error"),
    };
    // tokenizers-crate surfaces a Tokenizer variant when it can't parse
    // (or open) the file. We assert the loud error path fires, not the
    // exact message.
    assert!(matches!(err, TokenizerError::Tokenizer(_)));
}

#[test]
#[ignore = "network: downloads openai-community/gpt2 tokenizer from HF hub"]
fn full_roundtrip_gpt2_encode_decode() {
    let cache_dir = std::env::temp_dir().join("alc-nn-tokroundtrip-gpt2");
    // Don't clean the dir — reuse the cache on repeat runs.
    let tok = HfTokenizer::load_cached("gpt2", &cache_dir)
        .expect("gpt2 tokenizer download / load");
    assert_eq!(tok.preset(), "gpt2");

    let text = "hello world";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty(), "encoded ids should not be empty");

    let decoded = tok.decode(&ids).expect("decode");
    assert!(
        decoded.contains("hello") && decoded.contains("world"),
        "roundtrip must preserve both words; got: {decoded:?}"
    );

    assert!(
        tok.vocab_size() >= 50_000,
        "gpt2 vocab is ~50257; got {}",
        tok.vocab_size()
    );

    assert!(tok.cache_path().exists(), "cache file must be materialized");
}
