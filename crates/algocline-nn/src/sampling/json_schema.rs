//! JSON-schema-constrained decoding: a [`Constraint`] that admits only
//! token sequences spelling a JSON document valid against a schema.
//!
//! # Schema → regex → DFA
//!
//! [`JsonSchemaConstraint`] owns no automaton of its own. It compiles the
//! schema into a single regular expression once, at construction, and
//! delegates every per-token decision to [`RegexConstraint`] — the same
//! shape Outlines (dottxt-ai) uses. The reason is that a schema without
//! `$ref` describes a *finite* tree of alternations, concatenations and
//! repetitions, which is exactly the class a regular language covers; once
//! that translation is done, the hard part (walking a byte DFA over the
//! tokenizer's surface strings, rejecting a token the moment it can no
//! longer reach a full match) is already solved and does not want a second
//! implementation.
//!
//! # Failure is loud, and that is the whole point
//!
//! A keyword this version does not interpret is rejected at construction
//! rather than ignored. Ignoring one is not a harmless partial
//! implementation: every unhandled keyword *widens* the generated
//! language, so a caller who wrote `"pattern"` or `"maxLength"` would get
//! a constraint that silently permits the documents those keywords exist
//! to forbid. Since the sole reason to pay for constrained decoding is the
//! guarantee that the output is valid, a quietly weakened guarantee is
//! worse than no constraint at all — the caller would stop checking.
//!
//! The check is an allowlist (see [`reject_unsupported`]): each node
//! reports the keys it did *not* interpret, so a JSON Schema draft that
//! adds new keywords fails safe instead of being silently under-enforced.
//!
//! # KNOWN LIMITATION: every property must be required
//!
//! A schema whose `required` array does not cover every key of
//! `properties` is rejected. Optional properties turn a fixed
//! concatenation into a set of `2^n` comma placements — the commas sit
//! *between* members, so each present/absent combination changes the
//! separator layout rather than just deleting a member. Emitting that
//! expansion is possible but blows up the pattern and the DFA, and this
//! version does not pay that cost. Callers mark every property required,
//! or split the schema into the variants they actually intend to
//! generate.
//!
//! # KNOWN LIMITATION: compact output only
//!
//! The generated pattern permits no whitespace between structural tokens:
//! the output is always `{"a":1,"b":[2,3]}`, never `{ "a": 1 }`. JSON
//! treats the two as equivalent, but admitting optional whitespace at
//! every structural position multiplies the automaton for no gain in
//! expressiveness, and a deterministic surface form is easier to diff and
//! to assert on. Whitespace *inside* string values is unaffected — it is
//! ordinary string content.
//!
//! Property order is likewise fixed: members are emitted in sorted key
//! order (see [`object_to_regex`]).
//!
//! # Recursive schemas are out of reach, not just unimplemented
//!
//! `$ref` is rejected like any other unsupported keyword, but it deserves
//! a separate note: a self-referential schema (a tree node whose child is
//! the same node) describes a context-free language, and no regular
//! expression can match balanced nesting of unbounded depth. Supporting
//! recursion therefore cannot be done by extending this translation — it
//! needs a pushdown automaton, i.e. the grammar-constrained (GBNF) path
//! the module plan lists as a separate Layer 2 constraint. Non-recursive
//! `$ref` (plain reuse) could be handled here by inlining, and is simply
//! not implemented yet.

use std::collections::HashSet;

use candle_core::Result as CandleResult;
use serde_json::{Map, Value};

use super::constraint::{Constraint, RegexConstraint, TokenMask};

/// JSON string per RFC 8259: any character except `"`, `\` and the C0
/// controls, plus the two escape forms. Control characters are excluded
/// from the literal branch on purpose — JSON forbids them raw, and a
/// model that emitted one would produce a document `serde_json` refuses
/// to parse, which is precisely the outcome the constraint prevents.
const STRING_RE: &str = r#""(?:[^"\\\x00-\x1F]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*""#;

/// JSON integer. The `0|[1-9][0-9]*` alternation is what forbids leading
/// zeros (`01` is not JSON), and there is no leading `+` — JSON allows a
/// sign only on the negative side.
const INTEGER_RE: &str = r"-?(?:0|[1-9][0-9]*)";

/// JSON number: the integer part above, then an optional fraction and an
/// optional exponent. Both `.5` and `1.` are rejected by construction —
/// JSON requires a digit on each side of the point.
const NUMBER_RE: &str = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";

/// Grouped so the alternation cannot leak into a surrounding
/// concatenation (`{"a":true|false}` would otherwise parse as
/// `{"a":true` **or** `false}`).
const BOOLEAN_RE: &str = r"(?:true|false)";

const NULL_RE: &str = "null";

/// The `type` values this version can translate.
const SUPPORTED_TYPES: &[&str] = &[
    "object", "array", "string", "integer", "number", "boolean", "null",
];

/// Nesting limit for schema translation.
///
/// `schema_to_regex` recurses once per nesting level, so an adversarial
/// or generated schema could otherwise overflow the stack — a panic, in a
/// library, on caller-supplied data. Refusing past a generous depth keeps
/// the failure a `Result`.
const MAX_DEPTH: usize = 32;

/// Size limit on any single generated sub-pattern.
///
/// An array duplicates its item pattern (`item (,item)*`), so nested
/// arrays double the pattern text per level. Checking at every node caps
/// the peak allocation instead of discovering the problem as an
/// out-of-memory abort.
const MAX_REGEX_BYTES: usize = 1 << 20;

/// Restrict generation to token sequences spelling a JSON document that
/// validates against a schema.
///
/// The schema is translated to a regular expression at construction and
/// enforced by an inner [`RegexConstraint`]; see the module doc for the
/// rationale, the supported keyword set and the two known limitations
/// (all properties required, compact output only).
///
/// # Errors
///
/// [`JsonSchemaConstraint::new`] rejects — loudly, before a single token
/// is generated — a schema containing a keyword this version does not
/// interpret, an optional property, a node with neither `type` nor
/// `enum`, or nesting past [`MAX_DEPTH`].
#[derive(Debug, Clone)]
pub struct JsonSchemaConstraint {
    inner: RegexConstraint,
}

impl JsonSchemaConstraint {
    /// Compile `schema` into a regex and wrap it in a [`RegexConstraint`]
    /// over `vocab`.
    ///
    /// `vocab` is the surface string of every token id, indexed by id —
    /// the shape [`crate::tokenizer::HfTokenizer::vocab_strings`]
    /// produces.
    pub fn new(schema: &Value, vocab: Vec<String>) -> CandleResult<Self> {
        let pattern = schema_to_regex(schema, "#", 0)?;
        Ok(Self {
            inner: RegexConstraint::new(&pattern, vocab)?,
        })
    }
}

impl Constraint for JsonSchemaConstraint {
    fn mask(&self, prefix: &[u32]) -> TokenMask {
        self.inner.mask(prefix)
    }

    fn is_terminal(&self, prefix: &[u32]) -> bool {
        self.inner.is_terminal(prefix)
    }
}

// ─── schema translation ───────────────────────────────────────────────

/// Translate one schema node into a regex fragment.
///
/// `path` is a JSON-pointer-ish trail (`#/properties/user/items`) carried
/// purely so an error names the offending node; a schema of any size is
/// otherwise indistinguishable from its sub-schemas in a message.
fn schema_to_regex(schema: &Value, path: &str, depth: usize) -> CandleResult<String> {
    if depth > MAX_DEPTH {
        return Err(msg(format!(
            "JsonSchemaConstraint: schema nesting at {path} exceeds the {MAX_DEPTH} level limit"
        )));
    }
    let map = schema.as_object().ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: schema node at {path} must be a JSON object, got {}",
            json_type_name(schema)
        ))
    })?;

    let regex = if let Some(values) = map.get("enum") {
        enum_to_regex(map, values, path)?
    } else if let Some(ty) = map.get("type") {
        let ty = ty.as_str().ok_or_else(|| {
            msg(format!(
                "JsonSchemaConstraint: `type` at {path} must be a string, got {}",
                json_type_name(ty)
            ))
        })?;
        match ty {
            "object" => {
                reject_unsupported(map, &["type", "properties", "required"], path)?;
                object_to_regex(map, path, depth)?
            }
            "array" => {
                reject_unsupported(map, &["type", "items"], path)?;
                array_to_regex(map, path, depth)?
            }
            scalar @ ("string" | "integer" | "number" | "boolean" | "null") => {
                reject_unsupported(map, &["type"], path)?;
                match scalar {
                    "string" => STRING_RE,
                    "integer" => INTEGER_RE,
                    "number" => NUMBER_RE,
                    "boolean" => BOOLEAN_RE,
                    _ => NULL_RE,
                }
                .to_string()
            }
            other => return Err(unsupported_type(other, path)),
        }
    } else {
        // Neither keyword. Name whatever the node *does* carry first
        // (`$ref`, `anyOf`, `allOf`, ...) so the message points at the
        // construct that actually blocked translation.
        reject_unsupported(map, &[], path)?;
        return Err(msg(format!(
            "JsonSchemaConstraint: schema node at {path} declares neither `type` nor `enum`; \
             an unconstrained node would admit any JSON, defeating the constraint"
        )));
    };

    if regex.len() > MAX_REGEX_BYTES {
        return Err(msg(format!(
            "JsonSchemaConstraint: the pattern for {path} grew past {MAX_REGEX_BYTES} bytes; \
             an array duplicates its item pattern, so deeply nested arrays double it per level"
        )));
    }
    Ok(regex)
}

/// `{"k1":<v1>,"k2":<v2>}` with the members in **sorted key order**.
///
/// Declaration order is deliberately not used. `serde_json::Map` is a
/// `BTreeMap` unless the `preserve_order` feature is enabled, in which
/// case it becomes an `IndexMap` and iterates in declaration order —
/// and that feature is additive, so *any* crate in the dependency graph
/// can flip it. Keying the generated pattern on it would mean a schema
/// compiled today and a schema compiled after an unrelated `cargo add`
/// accept different documents. Sorting explicitly pins the order to a
/// property of the schema itself.
fn object_to_regex(map: &Map<String, Value>, path: &str, depth: usize) -> CandleResult<String> {
    let properties = map.get("properties").ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: object schema at {path} has no `properties`; \
             an object of unconstrained shape cannot be expressed as a regex"
        ))
    })?;
    let properties = properties.as_object().ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: `properties` at {path} must be a JSON object, got {}",
            json_type_name(properties)
        ))
    })?;
    let required = required_names(map, path)?;

    for name in &required {
        if !properties.contains_key(name.as_str()) {
            return Err(msg(format!(
                "JsonSchemaConstraint: `required` at {path} names {name:?}, \
                 which is not declared in `properties`"
            )));
        }
    }

    let mut entries: Vec<(&String, &Value)> = properties.iter().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut members: Vec<String> = Vec::with_capacity(entries.len());
    for (name, sub) in entries {
        if !required.contains(name.as_str()) {
            return Err(msg(format!(
                "JsonSchemaConstraint: property {name:?} at {path} is optional (absent from \
                 `required`). This version only supports schemas where every property is \
                 required; mark all properties required, or split the schema into the \
                 variants you intend to generate"
            )));
        }
        let key = regex_escape(&json_literal(&Value::String(name.to_string()))?);
        let value = schema_to_regex(sub, &format!("{path}/properties/{name}"), depth + 1)?;
        members.push(format!("{key}:{value}"));
    }

    let mut out = String::from(r"\{");
    out.push_str(&members.join(","));
    out.push_str(r"\}");
    Ok(out)
}

/// `[]` or `[item(,item)*]`.
///
/// The item pattern appears twice because a comma separates members
/// rather than terminating them; the alternative spellings that mention
/// it once (`(?:item)?(?:,item)*`) would accept a leading comma.
fn array_to_regex(map: &Map<String, Value>, path: &str, depth: usize) -> CandleResult<String> {
    let items = map.get("items").ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: array schema at {path} has no `items`; \
             an array of unconstrained element type cannot be expressed as a regex"
        ))
    })?;
    let item = schema_to_regex(items, &format!("{path}/items"), depth + 1)?;
    Ok(format!(r"\[(?:{item}(?:,{item})*)?\]"))
}

/// Alternation of the enum members, each serialised to its compact JSON
/// literal and then regex-escaped.
///
/// A sibling `type` is *interpreted*, not ignored: every member is
/// checked against it. That keeps the allowlist honest — the common
/// `{"type":"string","enum":[...]}` spelling stays legal without the
/// keyword becoming a no-op, and a schema whose members contradict its
/// own declared type is reported instead of silently generating one of
/// them anyway.
fn enum_to_regex(map: &Map<String, Value>, values: &Value, path: &str) -> CandleResult<String> {
    reject_unsupported(map, &["enum", "type"], path)?;
    let values = values.as_array().ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: `enum` at {path} must be an array, got {}",
            json_type_name(values)
        ))
    })?;
    if values.is_empty() {
        return Err(msg(format!(
            "JsonSchemaConstraint: `enum` at {path} is empty, so no document could be generated"
        )));
    }

    if let Some(ty) = map.get("type") {
        let ty = ty.as_str().ok_or_else(|| {
            msg(format!(
                "JsonSchemaConstraint: `type` at {path} must be a string, got {}",
                json_type_name(ty)
            ))
        })?;
        if !SUPPORTED_TYPES.contains(&ty) {
            return Err(unsupported_type(ty, path));
        }
        for (i, value) in values.iter().enumerate() {
            if !matches_type(value, ty) {
                return Err(msg(format!(
                    "JsonSchemaConstraint: `enum[{i}]` at {path} is a {} but `type` says {ty:?}; \
                     the schema contradicts itself",
                    json_type_name(value)
                )));
            }
        }
    }

    let mut alternatives: Vec<String> = Vec::with_capacity(values.len());
    for value in values {
        alternatives.push(regex_escape(&json_literal(value)?));
    }
    Ok(format!("(?:{})", alternatives.join("|")))
}

/// The `required` names as a set. Absent is an empty set, which is only
/// viable for an object with no properties — [`object_to_regex`] rejects
/// every uncovered property afterwards.
fn required_names(map: &Map<String, Value>, path: &str) -> CandleResult<HashSet<String>> {
    let Some(required) = map.get("required") else {
        return Ok(HashSet::new());
    };
    let list = required.as_array().ok_or_else(|| {
        msg(format!(
            "JsonSchemaConstraint: `required` at {path} must be an array, got {}",
            json_type_name(required)
        ))
    })?;

    let mut names = HashSet::with_capacity(list.len());
    for (i, entry) in list.iter().enumerate() {
        let name = entry.as_str().ok_or_else(|| {
            msg(format!(
                "JsonSchemaConstraint: `required[{i}]` at {path} must be a string, got {}",
                json_type_name(entry)
            ))
        })?;
        names.insert(name.to_string());
    }
    Ok(names)
}

/// Reject any key of `map` outside `interpreted`.
///
/// The allowlist direction is what makes this fail safe: a keyword nobody
/// thought to enumerate — a future draft's, a typo'd one — lands in the
/// error rather than being silently dropped and widening the language.
fn reject_unsupported(
    map: &Map<String, Value>,
    interpreted: &[&str],
    path: &str,
) -> CandleResult<()> {
    let mut unsupported: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|key| !interpreted.contains(key))
        .collect();
    if unsupported.is_empty() {
        return Ok(());
    }
    unsupported.sort_unstable();

    let interpreted = if interpreted.is_empty() {
        "nothing (this node declares neither `type` nor `enum`)".to_string()
    } else {
        interpreted.join(", ")
    };
    Err(msg(format!(
        "JsonSchemaConstraint: unsupported schema keyword(s) at {path}: {}. \
         Ignoring them would widen the accepted language, so they are refused rather than \
         dropped. This node interprets: {interpreted}",
        unsupported.join(", ")
    )))
}

// ─── helpers ──────────────────────────────────────────────────────────

fn msg(text: String) -> candle_core::Error {
    candle_core::Error::Msg(text)
}

fn unsupported_type(ty: &str, path: &str) -> candle_core::Error {
    msg(format!(
        "JsonSchemaConstraint: unsupported `type` {ty:?} at {path}; supported: {}",
        SUPPORTED_TYPES.join(", ")
    ))
}

/// The JSON Schema type name of a concrete value, used both for error
/// messages and for the `enum` / `type` consistency check.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Whether `value` satisfies the declared `type`.
///
/// `integer` is the one asymmetric case: JSON Schema treats it as a
/// refinement of `number`, so an integer satisfies `number` but not the
/// reverse.
fn matches_type(value: &Value, ty: &str) -> bool {
    match ty {
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        other => json_type_name(value) == other,
    }
}

/// Compact JSON serialisation of a value — the exact bytes the model is
/// then required to emit.
fn json_literal(value: &Value) -> CandleResult<String> {
    serde_json::to_string(value).map_err(|e| {
        msg(format!(
            "JsonSchemaConstraint: cannot serialise a schema literal to JSON: {e}"
        ))
    })
}

/// Escape every regex metacharacter in `literal`.
///
/// Hand-rolled rather than pulled from `regex-syntax`, which is a
/// transitive dependency of `regex-automata` and not a direct one; the
/// escaped set is the same one `regex_syntax::is_meta_character` reports.
/// The inputs are JSON literals, whose control characters are already
/// `\uXXXX`-escaped by `serde_json`, so only these ASCII punctuation
/// characters can ever need escaping.
fn regex_escape(literal: &str) -> String {
    const META: &[char] = &[
        '\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}', '^', '$', '#', '&', '-', '~',
    ];
    let mut out = String::with_capacity(literal.len());
    for ch in literal.chars() {
        if META.contains(&ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{ConstrainedSampler, GreedySampler, Sampler, TopKTopPSampler};
    use candle_core::{Device, Tensor};
    use serde_json::json;

    /// Single-character vocab covering every byte the schemas under test
    /// can produce. One character per token keeps the expected streams
    /// readable and lets a test address a position by its character.
    fn json_vocab() -> Vec<String> {
        const CHARS: &str = "{}[]\":,-+.E abcdefghijklmnopqrstuvwxyz0123456789";
        CHARS.chars().map(|c| c.to_string()).collect()
    }

    fn encode(text: &str) -> Vec<u32> {
        let vocab = json_vocab();
        text.chars()
            .map(|ch| {
                let piece = ch.to_string();
                vocab
                    .iter()
                    .position(|v| *v == piece)
                    .unwrap_or_else(|| panic!("test vocab lacks {ch:?}")) as u32
            })
            .collect()
    }

    fn decode(ids: &[u32]) -> String {
        let vocab = json_vocab();
        ids.iter().map(|id| vocab[*id as usize].clone()).collect()
    }

    fn constraint(schema: &Value) -> JsonSchemaConstraint {
        JsonSchemaConstraint::new(schema, json_vocab()).unwrap()
    }

    /// Whether the schema accepts `text` as a *complete* document.
    fn accepts(schema: &Value, text: &str) -> bool {
        constraint(schema).is_terminal(&encode(text))
    }

    /// The characters permitted right after `prefix`, sorted, so a test
    /// asserts on semantics instead of on which sparse [`TokenMask`]
    /// variant the heuristic happened to pick.
    fn allowed_after(c: &JsonSchemaConstraint, prefix: &str) -> Vec<String> {
        let vocab = json_vocab();
        let ids: Vec<u32> = match c.mask(&encode(prefix)) {
            TokenMask::AllowAll => (0..vocab.len() as u32).collect(),
            TokenMask::Allow(ids) => ids,
            TokenMask::Deny(denied) => (0..vocab.len() as u32)
                .filter(|id| !denied.contains(id))
                .collect(),
        };
        let mut out: Vec<String> = ids
            .into_iter()
            .map(|id| vocab[id as usize].clone())
            .collect();
        out.sort();
        out
    }

    /// Logits over `json_vocab()` whose argmax is `}` — a character that
    /// is illegal at every structural position of the object schema, so
    /// an unconstrained greedy sampler would produce nothing but garbage
    /// and the forced skeleton below is unambiguously the mask's doing.
    fn logits_favouring_close_brace() -> Tensor {
        let vocab = json_vocab();
        let mut values = vec![0.0f32; vocab.len()];
        values[encode("}")[0] as usize] = 10.0;
        Tensor::from_slice(&values, (vocab.len(),), &Device::Cpu).unwrap()
    }

    fn object_schema() -> Value {
        json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
        })
    }

    // ─── scalar fragments ─────────────────────────────────────────────

    /// Each scalar type accepts exactly the JSON grammar for it. The
    /// rejects are the cases a looser pattern would wave through: leading
    /// zeros, a bare `+` sign, a fraction missing a digit on either side,
    /// an unquoted string, a truncated keyword.
    #[test]
    fn scalar_types_accept_and_reject_the_json_grammar() {
        let cases: &[(Value, &[&str], &[&str])] = &[
            (
                json!({"type": "integer"}),
                &["0", "-0", "7", "-42", "1234567890"],
                &["", "01", "+1", "-", "1.5", "1e3", "a"],
            ),
            (
                json!({"type": "number"}),
                &["0", "-1.5", "1e10", "1.5E+3", "12.0", "-0.25e-2"],
                &["", ".5", "1.", "01", "+1", "1e", "e3"],
            ),
            (
                json!({"type": "boolean"}),
                &["true", "false"],
                &["", "tru", "truex", "1", "\"true\""],
            ),
            (
                json!({"type": "null"}),
                &["null"],
                &["", "nul", "nullx", "0"],
            ),
            (
                json!({"type": "string"}),
                &["\"\"", "\"ab\"", "\"a b\"", "\"{},:[]\""],
                &["", "ab", "\"", "\"a\"b\""],
            ),
            (
                json!({"enum": ["a", "b"]}),
                &["\"a\"", "\"b\""],
                &["", "a", "\"c\"", "\"ab\""],
            ),
            (
                json!({"type": "integer", "enum": [1, 22]}),
                &["1", "22"],
                &["", "2", "12", "3"],
            ),
        ];

        for (schema, ok, bad) in cases {
            for text in *ok {
                assert!(accepts(schema, text), "{schema} must accept {text:?}");
            }
            for text in *bad {
                assert!(!accepts(schema, text), "{schema} must reject {text:?}");
            }
        }
    }

    // ─── loud rejection of unsupported constructs ─────────────────────

    /// Every keyword this version does not interpret is refused, and the
    /// message names it. Silently ignoring one would widen the accepted
    /// language, which is the exact failure constrained decoding exists
    /// to prevent — a caller who wrote `pattern` would believe it bound.
    #[test]
    fn unsupported_keywords_are_rejected_by_name() {
        let cases: &[(Value, &str)] = &[
            (json!({"$ref": "#/$defs/node"}), "$ref"),
            (json!({"anyOf": [{"type": "string"}]}), "anyOf"),
            (json!({"oneOf": [{"type": "string"}]}), "oneOf"),
            (json!({"allOf": [{"type": "string"}]}), "allOf"),
            (json!({"not": {"type": "string"}}), "not"),
            (json!({"type": "string", "pattern": "^a+$"}), "pattern"),
            (json!({"type": "string", "minLength": 2}), "minLength"),
            (json!({"type": "string", "maxLength": 8}), "maxLength"),
            (json!({"type": "string", "format": "email"}), "format"),
            (json!({"type": "integer", "minimum": 0}), "minimum"),
            (json!({"type": "integer", "maximum": 9}), "maximum"),
            (
                json!({"type": "array", "items": {"type": "integer"}, "minItems": 1}),
                "minItems",
            ),
            (
                json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": true,
                }),
                "additionalProperties",
            ),
            (
                json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "patternProperties": {"^x": {"type": "integer"}},
                }),
                "patternProperties",
            ),
        ];

        for (schema, keyword) in cases {
            let err = JsonSchemaConstraint::new(schema, json_vocab())
                .expect_err(&format!("{schema} must be rejected"));
            let text = err.to_string();
            assert!(
                text.contains(keyword),
                "error for {schema} must name {keyword:?}, got: {text}"
            );
        }
    }

    /// An unsupported keyword nested inside an otherwise fine schema is
    /// caught too, and the message points at the offending node rather
    /// than at the root.
    #[test]
    fn nested_unsupported_keyword_reports_its_path() {
        let schema = json!({
            "type": "object",
            "properties": { "tags": { "type": "array", "items": { "type": "string", "pattern": "a" } } },
            "required": ["tags"],
        });
        let err = JsonSchemaConstraint::new(&schema, json_vocab())
            .expect_err("nested `pattern` must be rejected")
            .to_string();
        assert!(err.contains("pattern"), "must name the keyword: {err}");
        assert!(
            err.contains("#/properties/tags/items"),
            "must name the node: {err}"
        );
    }

    /// A node carrying no `type` and no `enum` constrains nothing, so
    /// accepting it would hand back a constraint that permits any JSON.
    #[test]
    fn a_node_without_type_or_enum_is_rejected() {
        let err = JsonSchemaConstraint::new(&json!({}), json_vocab())
            .expect_err("empty node must be rejected")
            .to_string();
        assert!(err.contains("neither `type` nor `enum`"), "got: {err}");
    }

    /// A schema whose `enum` members contradict its own `type` is a
    /// caller bug. Generating one of the members anyway would emit a
    /// document the caller's own validator rejects.
    #[test]
    fn enum_members_must_match_a_declared_type() {
        let err =
            JsonSchemaConstraint::new(&json!({"type": "integer", "enum": [1, "x"]}), json_vocab())
                .expect_err("mismatched enum member must be rejected")
                .to_string();
        assert!(err.contains("enum[1]"), "must name the member: {err}");
    }

    // ─── KNOWN LIMITATION: optional properties ────────────────────────

    /// A property missing from `required` is refused, with the property
    /// named and the workaround stated. See the module doc for why the
    /// `2^n` comma-placement expansion is not paid for here.
    #[test]
    fn optional_properties_are_rejected() {
        let partial = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        });
        let err = JsonSchemaConstraint::new(&partial, json_vocab())
            .expect_err("optional property must be rejected")
            .to_string();
        assert!(err.contains("\"b\""), "must name the property: {err}");
        assert!(err.contains("required"), "must point at `required`: {err}");

        let missing = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
        });
        assert!(
            JsonSchemaConstraint::new(&missing, json_vocab()).is_err(),
            "a missing `required` leaves every property optional"
        );

        // `required` naming a property that does not exist is the mirror
        // bug and must not be silently dropped either.
        let phantom = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost"],
        });
        let err = JsonSchemaConstraint::new(&phantom, json_vocab())
            .expect_err("unknown required name must be rejected")
            .to_string();
        assert!(err.contains("ghost"), "must name the phantom: {err}");

        // An object with no properties at all is legal and needs no
        // `required`; it only ever spells `{}`.
        assert!(accepts(&json!({"type": "object", "properties": {}}), "{}"));
    }

    // ─── object skeleton enforcement ──────────────────────────────────

    /// The structural skeleton is forced character by character: only `{`
    /// may open, only the declared key may follow, only `:` may end it,
    /// only `"` may open the value, and once the value's closing quote
    /// lands only `}` remains. The greedy sampler makes the effect
    /// unambiguous — every draw would be `}` without the mask.
    #[test]
    fn object_skeleton_is_forced() {
        let c = constraint(&object_schema());

        for (prefix, expected) in [
            ("", "{"),
            ("{", "\""),
            ("{\"", "n"),
            ("{\"n", "a"),
            ("{\"na", "m"),
            ("{\"nam", "e"),
            ("{\"name", "\""),
            ("{\"name\"", ":"),
            ("{\"name\":", "\""),
            ("{\"name\":\"ab\"", "}"),
        ] {
            assert_eq!(
                allowed_after(&c, prefix),
                vec![expected.to_string()],
                "after {prefix:?} only {expected:?} may follow"
            );
        }

        // Whitespace is structural nowhere (KNOWN LIMITATION: compact
        // output only) but is ordinary content inside a string value.
        assert!(!accepts(&object_schema(), "{ \"name\":\"ab\"}"));
        assert!(accepts(&object_schema(), "{\"name\":\"a b\"}"));

        let mut s = ConstrainedSampler::new(GreedySampler, constraint(&object_schema()));
        let drawn: Vec<u32> = (0..9)
            .map(|_| s.sample(&logits_favouring_close_brace()).unwrap())
            .collect();
        assert_eq!(decode(&drawn), "{\"name\":\"");
        assert!(!s.is_done(), "an unterminated document is not terminal");
    }

    /// Only a complete document is terminal. Without this the skeleton
    /// test above would still pass on an `is_terminal` that reported
    /// "the prefix has not died yet".
    #[test]
    fn only_a_complete_document_is_terminal() {
        let c = constraint(&object_schema());
        for partial in [
            "",
            "{",
            "{\"",
            "{\"name",
            "{\"name\"",
            "{\"name\":",
            "{\"name\":\"",
            "{\"name\":\"ab",
            "{\"name\":\"ab\"",
        ] {
            assert!(
                !c.is_terminal(&encode(partial)),
                "{partial:?} is only a partial document"
            );
        }
        assert!(c.is_terminal(&encode("{\"name\":\"ab\"}")));
        assert!(c.is_terminal(&encode("{\"name\":\"\"}")));
    }

    /// Members are emitted in sorted key order, not declaration order —
    /// the property that keeps the generated pattern independent of
    /// `serde_json`'s `preserve_order` feature. See
    /// [`object_to_regex`].
    #[test]
    fn object_members_follow_sorted_key_order() {
        let schema = json!({
            "type": "object",
            "properties": {
                "b": {"type": "integer"},
                "a": {"type": "integer"},
            },
            "required": ["b", "a"],
        });
        assert!(
            accepts(&schema, "{\"a\":1,\"b\":2}"),
            "sorted order accepted"
        );
        assert!(
            !accepts(&schema, "{\"b\":2,\"a\":1}"),
            "declaration order must not be accepted as well"
        );
    }

    /// Nested objects recurse. The inner object's own skeleton is forced
    /// the same way, so an empty inner object cannot satisfy a schema
    /// that declares a required member.
    #[test]
    fn nested_objects_compile() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["id", "name"],
                },
            },
            "required": ["user"],
        });

        assert!(accepts(&schema, "{\"user\":{\"id\":7,\"name\":\"ab\"}}"));
        assert!(!accepts(&schema, "{\"user\":{}}"));
        assert!(!accepts(&schema, "{\"user\":{\"id\":7}}"));
        assert!(!accepts(&schema, "{\"id\":7,\"name\":\"ab\"}"));

        let c = constraint(&schema);
        assert_eq!(
            allowed_after(&c, "{\"user\":"),
            vec!["{".to_string()],
            "a nested object may only open with a brace"
        );
    }

    // ─── arrays ───────────────────────────────────────────────────────

    /// Both the empty and the multi-element forms are accepted, and the
    /// separator cannot be stranded: a trailing or leading comma is
    /// unreachable rather than merely rejected after the fact.
    #[test]
    fn arrays_accept_empty_and_multi_element_forms() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});

        for text in ["[]", "[1]", "[1,2]", "[-3,0,42]"] {
            assert!(accepts(&schema, text), "{text:?} must be accepted");
        }
        for text in ["[", "]", "[1,]", "[,1]", "[1,,2]", "[1 2]", "[01]"] {
            assert!(!accepts(&schema, text), "{text:?} must be rejected");
        }

        let c = constraint(&schema);
        let after_open = allowed_after(&c, "[");
        assert!(after_open.contains(&"]".to_string()), "`[]` must stay open");
        assert!(after_open.contains(&"1".to_string()));
        assert!(
            !after_open.contains(&",".to_string()),
            "a leading comma must be unreachable"
        );

        let after_comma = allowed_after(&c, "[1,");
        assert!(
            !after_comma.contains(&"]".to_string()),
            "a trailing comma must be unreachable: {after_comma:?}"
        );
        assert!(after_comma.contains(&"2".to_string()));

        let mut expected: Vec<String> = ('0'..='9')
            .map(|d| d.to_string())
            .chain(["]".to_string(), ",".to_string()])
            .collect();
        expected.sort();
        assert_eq!(
            allowed_after(&c, "[1"),
            expected,
            "after one digit: more digits, a separator, or the close"
        );

        // Arrays of objects exercise recursion through the item path.
        let nested = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        });
        assert!(accepts(&nested, "[]"));
        assert!(accepts(&nested, "[{\"id\":1},{\"id\":2}]"));
        assert!(!accepts(&nested, "[{\"id\":1},]"));
    }

    // ─── composition ──────────────────────────────────────────────────

    /// Masking through a stochastic sampler keeps both guarantees at
    /// once: the seed still reproduces the stream, and every token the
    /// stream contains was permitted by the schema at the position it
    /// appeared — checked by replaying the stream through an independent
    /// constraint rather than by eyeballing the decoded text.
    #[test]
    fn composition_with_top_k_top_p_stays_reproducible_and_on_schema() {
        let vocab = json_vocab();
        let logits: Vec<f32> = (0..vocab.len()).map(|i| ((i % 7) as f32) * 0.5).collect();
        let logits = Tensor::from_slice(&logits, (vocab.len(),), &Device::Cpu).unwrap();

        let build = || {
            ConstrainedSampler::new(
                TopKTopPSampler::new(Some(8), Some(0.95), 1.0, 24601),
                constraint(&object_schema()),
            )
        };
        let mut a = build();
        let mut b = build();

        let seq_a: Vec<u32> = (0..12).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..12).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b, "constrained sampler diverged on shared seed");

        // Replay: every drawn token must have been in the permitted set
        // for the prefix that preceded it.
        let check = constraint(&object_schema());
        for (step, token) in seq_a.iter().enumerate() {
            let prefix = &seq_a[..step];
            let permitted = match check.mask(prefix) {
                TokenMask::AllowAll => (0..vocab.len() as u32).collect::<Vec<u32>>(),
                TokenMask::Allow(ids) => ids,
                TokenMask::Deny(denied) => (0..vocab.len() as u32)
                    .filter(|id| !denied.contains(id))
                    .collect(),
            };
            assert!(
                permitted.contains(token),
                "step {step}: token {:?} is off-schema after {:?}",
                vocab[*token as usize],
                decode(prefix)
            );
        }
        assert_eq!(
            decode(&seq_a[..9]),
            "{\"name\":\"",
            "the skeleton is forced regardless of the draw: {:?}",
            decode(&seq_a)
        );
    }

    // ─── translation guards ───────────────────────────────────────────

    /// Nesting past the limit is refused rather than recursed into — a
    /// stack overflow is a panic, and this is a library taking
    /// caller-supplied data.
    #[test]
    fn nesting_past_the_limit_is_rejected() {
        let mut schema = json!({"type": "integer"});
        for _ in 0..(MAX_DEPTH + 2) {
            schema = json!({"type": "array", "items": schema});
        }
        let err = JsonSchemaConstraint::new(&schema, json_vocab())
            .expect_err("over-deep schema must be rejected")
            .to_string();
        assert!(err.contains("nesting"), "got: {err}");
    }

    /// Regex metacharacters in a property name are escaped, so a key
    /// containing `.` or `+` matches itself rather than acting as a
    /// wildcard.
    #[test]
    fn property_names_are_regex_escaped() {
        let schema = json!({
            "type": "object",
            "properties": {"a.b": {"type": "integer"}},
            "required": ["a.b"],
        });
        assert!(accepts(&schema, "{\"a.b\":1}"));
        assert!(
            !accepts(&schema, "{\"axb\":1}"),
            "an unescaped `.` would match any character"
        );
    }

    /// A schema node that is not a JSON object at all (a bare `true`,
    /// which JSON Schema reads as "anything") is refused: it constrains
    /// nothing.
    #[test]
    fn non_object_schema_nodes_are_rejected() {
        for schema in [json!(true), json!([]), json!("string"), json!(null)] {
            assert!(
                JsonSchemaConstraint::new(&schema, json_vocab()).is_err(),
                "{schema} must be rejected as a schema node"
            );
        }
    }
}
