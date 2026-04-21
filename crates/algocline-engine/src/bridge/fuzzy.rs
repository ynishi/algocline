use mlua::prelude::*;

/// Register `alc.match_enum(text, candidates, opts?)` — fuzzy enum matcher for LLM output.
///
/// Finds which candidate string appears in `text` (case-insensitive substring match).
/// If multiple candidates match, returns the one whose last occurrence is latest
/// (LLMs tend to state conclusions last).
/// If no substring match, falls back to fuzzy matching via `fuzzy_parser::distance`.
///
/// Lua usage:
///   local verdict = alc.match_enum(response, {"PASS", "BLOCKED"})
///   -- returns "PASS", "BLOCKED", or nil
///
/// opts (optional table):
///   threshold: minimum similarity for fuzzy fallback (default 0.7)
pub(super) fn register_fuzzy(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let match_enum = _lua.create_function(
        |_, (text, candidates, opts): (String, Vec<String>, Option<LuaTable>)| {
            let threshold = opts
                .as_ref()
                .and_then(|t| t.get::<f64>("threshold").ok())
                .unwrap_or(0.7);

            let text_lower = text.to_lowercase();

            // Phase 1: case-insensitive substring match.
            // If multiple candidates match, pick the one whose last occurrence is latest.
            let mut best: Option<(usize, &str)> = None; // (last_pos, candidate)
            for c in &candidates {
                let c_lower = c.to_lowercase();
                if let Some(pos) = text_lower.rfind(&c_lower) {
                    match best {
                        Some((prev_pos, _)) if pos > prev_pos => best = Some((pos, c)),
                        None => best = Some((pos, c)),
                        _ => {}
                    }
                }
            }
            if let Some((_, matched)) = best {
                return Ok(Some(matched.to_string()));
            }

            // Phase 2: fuzzy fallback — split text into words, compare each
            // word against candidates. Jaro-Winkler is designed for short strings,
            // so per-word comparison is more effective than whole-text comparison.
            let candidates_lower: Vec<String> =
                candidates.iter().map(|c| c.to_lowercase()).collect();
            let mut best_match: Option<(f64, usize)> = None; // (similarity, candidate_index)
            for token in text_lower.split_whitespace() {
                // Strip surrounding punctuation from the token for cleaner matching.
                let token = token.trim_matches(|c: char| !c.is_alphanumeric());
                if token.is_empty() {
                    continue;
                }
                for (i, cl) in candidates_lower.iter().enumerate() {
                    let sim = fuzzy_parser::distance::similarity(
                        token,
                        cl,
                        fuzzy_parser::distance::Algorithm::JaroWinkler,
                    );
                    if sim >= threshold {
                        match best_match {
                            Some((prev_sim, _)) if sim > prev_sim => {
                                best_match = Some((sim, i));
                            }
                            None => best_match = Some((sim, i)),
                            _ => {}
                        }
                    }
                }
            }
            if let Some((_, idx)) = best_match {
                return Ok(Some(candidates[idx].clone()));
            }

            Ok(None)
        },
    )?;

    alc_table.set("match_enum", match_enum)?;

    // alc.match_bool(text) -> true | false | nil
    //
    // Normalizes yes/no-style LLM responses.
    // Scans for affirmative/negative keywords (case-insensitive substring).
    // Returns the polarity of the last-occurring keyword, or nil if ambiguous/absent.
    //
    // Lua usage:
    //   local ok = alc.match_bool("Approved. The plan looks good.")  -- true
    //   local ok = alc.match_bool("rejected: missing tests")         -- false
    //   local ok = alc.match_bool("I need more information")         -- nil
    let match_bool = _lua.create_function(|_, text: String| {
        const TRUE_WORDS: &[&str] = &[
            "approved", "yes", "ok", "accept", "pass", "confirm", "agree", "true", "lgtm",
        ];
        const FALSE_WORDS: &[&str] = &[
            "rejected", "no", "deny", "block", "fail", "refuse", "disagree", "false",
        ];

        let text_lower = text.to_lowercase();
        let bytes = text_lower.as_bytes();

        // Check that the character at the given byte position is not alphanumeric (ASCII).
        // Returns true if pos is out of bounds or the character is a word boundary.
        let is_boundary =
            |pos: usize| -> bool { pos >= bytes.len() || !bytes[pos].is_ascii_alphanumeric() };

        // Find the last whole-word occurrence of any keyword from either group.
        let mut last_pos: Option<(usize, bool)> = None; // (pos, is_true)
        for word in TRUE_WORDS {
            // Scan all occurrences (rfind only gives the last, but we need boundary check)
            let w = word.as_bytes();
            let mut start = 0;
            while let Some(rel) = text_lower[start..].find(word) {
                let pos = start + rel;
                let before_ok = pos == 0 || is_boundary(pos - 1);
                let after_ok = is_boundary(pos + w.len());
                if before_ok && after_ok {
                    match last_pos {
                        Some((prev, _)) if pos > prev => last_pos = Some((pos, true)),
                        None => last_pos = Some((pos, true)),
                        _ => {}
                    }
                }
                start = pos + 1;
            }
        }
        for word in FALSE_WORDS {
            let w = word.as_bytes();
            let mut start = 0;
            while let Some(rel) = text_lower[start..].find(word) {
                let pos = start + rel;
                let before_ok = pos == 0 || is_boundary(pos - 1);
                let after_ok = is_boundary(pos + w.len());
                if before_ok && after_ok {
                    match last_pos {
                        Some((prev, _)) if pos > prev => last_pos = Some((pos, false)),
                        None => last_pos = Some((pos, false)),
                        _ => {}
                    }
                }
                start = pos + 1;
            }
        }

        Ok(last_pos.map(|(_, v)| v))
    })?;

    alc_table.set("match_bool", match_bool)?;
    Ok(())
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
        }
    }

    // ─── alc.match_enum tests ───

    #[test]
    fn match_enum_exact_substring() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.match_enum("Verdict: BLOCKED. Fix the issues.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_case_insensitive() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.match_enum("verdict: pass. all good.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "PASS");
    }

    #[test]
    fn match_enum_last_wins() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Both appear, but PASS is last → PASS wins
        let result: String = lua
            .load(r#"return alc.match_enum("Initially BLOCKED, but after review: PASS", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "PASS");
    }

    #[test]
    fn match_enum_no_match_returns_nil() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: LuaValue = lua
            .load(r#"return alc.match_enum("something unrelated", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_enum_fuzzy_typo_in_short_response() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "BLOKED" is a typo for "BLOCKED" — fuzzy should catch it
        let result: String = lua
            .load(r#"return alc.match_enum("BLOKED", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_fuzzy_works_in_long_text() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Long sentence with a typo "BLCKED" buried in it — per-word fuzzy should find it
        let result: String = lua
            .load(r#"return alc.match_enum("After careful review of all the evidence and considering multiple factors, the final verdict is BLCKED due to missing tests.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_fuzzy_nil_when_no_close_word() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // No word is close enough to any candidate
        let result: LuaValue = lua
            .load(r#"return alc.match_enum("The weather is nice today", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    // ─── alc.match_bool tests ───

    #[test]
    fn match_bool_approved() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: bool = lua
            .load(r#"return alc.match_bool("Approved. The plan looks good.")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_rejected() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: bool = lua
            .load(r#"return alc.match_bool("rejected: missing test coverage")"#)
            .eval()
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn match_bool_nil_on_ambiguous() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: LuaValue = lua
            .load(r#"return alc.match_bool("I need more information about the design")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_last_keyword_wins() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "no" appears, then "approved" later → true
        let result: bool = lua
            .load(r#"return alc.match_bool("No issues found. Approved.")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_rejects_partial_word_ok_in_bypass() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "ok" should NOT match inside "token" or "okay" without boundary
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("This is a broken token")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_rejects_pass_in_bypass() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "pass" should NOT match inside "bypass"
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("We need to bypass the filter")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_rejects_no_in_innovation() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "no" should NOT match inside "innovation"
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("Great innovation in technology")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_word_boundary_with_punctuation() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "yes" followed by punctuation should match
        let result: bool = lua
            .load(r#"return alc.match_bool("yes, that works")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_fail_in_failed_matches() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "fail" at word boundary within "failed" — "fail" + "ed" where 'e' is alphanumeric
        // should NOT match
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("The process failed gracefully")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }
}
