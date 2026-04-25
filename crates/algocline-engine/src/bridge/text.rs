use mlua::prelude::*;
use mlua::LuaSerdeExt;

/// Register `alc.chunk(text, opts?)` — split text into chunks.
///
/// Lua usage:
///   local chunks = alc.chunk(text, { mode = "lines", size = 50 })
///   local chunks = alc.chunk(text, { mode = "lines", size = 50, overlap = 10 })
///   local chunks = alc.chunk(text, { mode = "chars", size = 2000 })
///
/// Returns: array of strings.
pub(super) fn register_chunk(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let chunk_fn = _lua.create_function(|lua, (text, opts): (String, Option<LuaTable>)| {
        let mode = opts
            .as_ref()
            .and_then(|o| o.get::<String>("mode").ok())
            .unwrap_or_else(|| "lines".into());
        let size = opts
            .as_ref()
            .and_then(|o| o.get::<usize>("size").ok())
            .unwrap_or(50);
        let overlap = opts
            .as_ref()
            .and_then(|o| o.get::<usize>("overlap").ok())
            .unwrap_or(0);

        let chunks: Vec<String> = match mode.as_str() {
            "chars" => chunk_by_chars(&text, size, overlap),
            _ => chunk_by_lines(&text, size, overlap),
        };

        lua.to_value(&chunks)
    })?;

    alc_table.set("chunk", chunk_fn)?;
    Ok(())
}

pub(super) fn chunk_by_lines(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || size == 0 {
        return vec![];
    }
    let step = if overlap < size { size - overlap } else { 1 };
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let end = (i + size).min(lines.len());
        chunks.push(lines[i..end].join("\n"));
        i += step;
        if end == lines.len() {
            break;
        }
    }
    chunks
}

pub(super) fn chunk_by_chars(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || size == 0 {
        return vec![];
    }
    let step = if overlap < size { size - overlap } else { 1 };
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + size).min(chars.len());
        chunks.push(chars[i..end].iter().collect());
        i += step;
        if end == chars.len() {
            break;
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::ExecutionMetrics;

    fn test_config() -> crate::bridge::BridgeConfig {
        use crate::card::FileCardStore;
        use crate::state::JsonFileStore;
        use std::sync::Arc;
        let metrics = ExecutionMetrics::new();
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        crate::bridge::BridgeConfig {
            llm_tx: None,
            ns: "default".into(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
            variant_pkgs: vec![],
            state_store: Arc::new(JsonFileStore::new(root.join("state"))),
            card_store: Arc::new(FileCardStore::new(root.join("cards"))),
            scenarios_dir: root.join("scenarios"),
            log_sink: None,
        }
    }

    // ─── chunk_by_lines tests ───

    #[test]
    fn chunk_lines_empty_text() {
        assert_eq!(chunk_by_lines("", 5, 0), Vec::<String>::new());
    }

    #[test]
    fn chunk_lines_single_line_exact_size() {
        let result = chunk_by_lines("hello", 1, 0);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn chunk_lines_single_line_size_larger() {
        let result = chunk_by_lines("hello", 10, 0);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn chunk_lines_exact_division() {
        let text = "a\nb\nc\nd";
        let result = chunk_by_lines(text, 2, 0);
        assert_eq!(result, vec!["a\nb", "c\nd"]);
    }

    #[test]
    fn chunk_lines_remainder() {
        let text = "a\nb\nc\nd\ne";
        let result = chunk_by_lines(text, 2, 0);
        assert_eq!(result, vec!["a\nb", "c\nd", "e"]);
    }

    #[test]
    fn chunk_lines_size_larger_than_total() {
        let text = "a\nb\nc";
        let result = chunk_by_lines(text, 100, 0);
        assert_eq!(result, vec!["a\nb\nc"]);
    }

    #[test]
    fn chunk_lines_with_overlap() {
        let text = "a\nb\nc\nd\ne";
        // size=3, overlap=1 → step=2
        let result = chunk_by_lines(text, 3, 1);
        assert_eq!(result, vec!["a\nb\nc", "c\nd\ne"]);
    }

    #[test]
    fn chunk_lines_overlap_equals_size_minus_one() {
        let text = "a\nb\nc\nd";
        // size=2, overlap=1 → step=1 (sliding window)
        let result = chunk_by_lines(text, 2, 1);
        assert_eq!(result, vec!["a\nb", "b\nc", "c\nd"]);
    }

    #[test]
    fn chunk_lines_overlap_ge_size_step_is_one() {
        let text = "a\nb\nc";
        // overlap >= size → step=1
        let result = chunk_by_lines(text, 2, 5);
        assert_eq!(result, vec!["a\nb", "b\nc"]);
    }

    #[test]
    fn chunk_lines_size_zero_returns_empty() {
        // size=0 should not produce infinite chunks
        let result = chunk_by_lines("a\nb\nc", 0, 0);
        assert_eq!(result, Vec::<String>::new());
    }

    // ─── chunk_by_chars tests ───

    #[test]
    fn chunk_chars_empty_text() {
        assert_eq!(chunk_by_chars("", 5, 0), Vec::<String>::new());
    }

    #[test]
    fn chunk_chars_exact_division() {
        let result = chunk_by_chars("abcdef", 3, 0);
        assert_eq!(result, vec!["abc", "def"]);
    }

    #[test]
    fn chunk_chars_remainder() {
        let result = chunk_by_chars("abcde", 3, 0);
        assert_eq!(result, vec!["abc", "de"]);
    }

    #[test]
    fn chunk_chars_size_larger_than_text() {
        let result = chunk_by_chars("abc", 100, 0);
        assert_eq!(result, vec!["abc"]);
    }

    #[test]
    fn chunk_chars_with_overlap() {
        // size=4, overlap=2 → step=2
        let result = chunk_by_chars("abcdef", 4, 2);
        assert_eq!(result, vec!["abcd", "cdef"]);
    }

    #[test]
    fn chunk_chars_overlap_ge_size_step_is_one() {
        // overlap >= size → step=1
        let result = chunk_by_chars("abc", 2, 3);
        assert_eq!(result, vec!["ab", "bc"]);
    }

    #[test]
    fn chunk_chars_multibyte() {
        // multibyte chars (3 bytes each in UTF-8, but split by char boundary)
        let result = chunk_by_chars("あいうえお", 2, 0);
        assert_eq!(result, vec!["あい", "うえ", "お"]);
    }

    #[test]
    fn chunk_chars_size_one() {
        let result = chunk_by_chars("abc", 1, 0);
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn chunk_chars_size_zero_returns_empty() {
        // size=0 should not produce infinite chunks
        let result = chunk_by_chars("abc", 0, 0);
        assert_eq!(result, Vec::<String>::new());
    }

    // ─── alc.chunk integration tests ───

    #[test]
    fn chunk_lua_lines_mode() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: Vec<String> = lua
            .load(r#"return alc.chunk("a\nb\nc\nd", { mode = "lines", size = 2 })"#)
            .eval()
            .unwrap();
        assert_eq!(result, vec!["a\nb", "c\nd"]);
    }

    #[test]
    fn chunk_lua_chars_mode() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: Vec<String> = lua
            .load(r#"return alc.chunk("abcdef", { mode = "chars", size = 3 })"#)
            .eval()
            .unwrap();
        assert_eq!(result, vec!["abc", "def"]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// chunk_by_lines never panics regardless of input.
        #[test]
        fn chunk_lines_never_panics(text in "\\PC{0,500}", size in 0usize..50, overlap in 0usize..50) {
            let _ = chunk_by_lines(&text, size, overlap);
        }

        /// chunk_by_chars never panics regardless of input.
        #[test]
        fn chunk_chars_never_panics(text in "\\PC{0,500}", size in 0usize..50, overlap in 0usize..50) {
            let _ = chunk_by_chars(&text, size, overlap);
        }

        /// All chars from the original text appear in at least one chunk (no data loss).
        #[test]
        fn chunk_chars_covers_all_input(text in "[a-z]{1,100}", size in 1usize..20) {
            let chunks = chunk_by_chars(&text, size, 0);
            let reconstructed: String = if chunks.len() <= 1 {
                chunks.into_iter().collect()
            } else {
                // Without overlap, concatenation should reproduce the original
                chunks.join("")
            };
            prop_assert_eq!(&reconstructed, &text);
        }

        /// All lines from the original text appear in at least one chunk (no data loss).
        #[test]
        fn chunk_lines_covers_all_input(
            lines in proptest::collection::vec("[a-z]{1,20}", 1..20),
            size in 1usize..10,
        ) {
            let text = lines.join("\n");
            let chunks = chunk_by_lines(&text, size, 0);
            let reconstructed = chunks.join("\n");
            prop_assert_eq!(&reconstructed, &text);
        }

        /// Each chunk has at most `size` characters.
        #[test]
        fn chunk_chars_respects_size(text in "[a-z]{1,200}", size in 1usize..50) {
            let chunks = chunk_by_chars(&text, size, 0);
            for chunk in &chunks {
                prop_assert!(chunk.chars().count() <= size,
                    "chunk length {} exceeds size {}", chunk.chars().count(), size);
            }
        }

        /// Each chunk has at most `size` lines.
        #[test]
        fn chunk_lines_respects_size(
            lines in proptest::collection::vec("[a-z]{1,10}", 1..30),
            size in 1usize..10,
        ) {
            let text = lines.join("\n");
            let chunks = chunk_by_lines(&text, size, 0);
            for chunk in &chunks {
                let line_count = chunk.lines().count();
                prop_assert!(line_count <= size,
                    "chunk has {} lines, exceeds size {}", line_count, size);
            }
        }

        /// With overlap, adjacent chunks share `overlap` characters.
        #[test]
        fn chunk_chars_overlap_shared(
            text in "[a-z]{10,100}",
            size in 3usize..15,
            overlap in 1usize..3,
        ) {
            prop_assume!(overlap < size);
            let chunks = chunk_by_chars(&text, size, overlap);
            if chunks.len() >= 2 {
                for i in 0..chunks.len() - 1 {
                    let suffix: String = chunks[i].chars().rev().take(overlap).collect::<Vec<_>>().into_iter().rev().collect();
                    let prefix: String = chunks[i + 1].chars().take(overlap).collect();
                    prop_assert_eq!(&suffix, &prefix,
                        "chunk[{}] suffix != chunk[{}] prefix", i, i + 1);
                }
            }
        }
    }
}
