# algocline-nn::sampling::json_schema

JSON-schema-constrained decoding: a [`Constraint`] that admits only
token sequences spelling a JSON document valid against a schema.

# Schema → regex → DFA

[`JsonSchemaConstraint`] owns no automaton of its own. It compiles the
schema into a single regular expression once, at construction, and
delegates every per-token decision to [`RegexConstraint`] — the same
shape Outlines (dottxt-ai) uses. The reason is that a schema without
`$ref` describes a *finite* tree of alternations, concatenations and
repetitions, which is exactly the class a regular language covers; once
that translation is done, the hard part (walking a byte DFA over the
tokenizer's surface strings, rejecting a token the moment it can no
longer reach a full match) is already solved and does not want a second
implementation.

# Failure is loud, and that is the whole point

A keyword this version does not interpret is rejected at construction
rather than ignored. Ignoring one is not a harmless partial
implementation: every unhandled keyword *widens* the generated
language, so a caller who wrote `"pattern"` or `"maxLength"` would get
a constraint that silently permits the documents those keywords exist
to forbid. Since the sole reason to pay for constrained decoding is the
guarantee that the output is valid, a quietly weakened guarantee is
worse than no constraint at all — the caller would stop checking.

The check is an allowlist (see [`reject_unsupported`]): each node
reports the keys it did *not* interpret, so a JSON Schema draft that
adds new keywords fails safe instead of being silently under-enforced.

# KNOWN LIMITATION: every property must be required

A schema whose `required` array does not cover every key of
`properties` is rejected. Optional properties turn a fixed
concatenation into a set of `2^n` comma placements — the commas sit
*between* members, so each present/absent combination changes the
separator layout rather than just deleting a member. Emitting that
expansion is possible but blows up the pattern and the DFA, and this
version does not pay that cost. Callers mark every property required,
or split the schema into the variants they actually intend to
generate.

# KNOWN LIMITATION: compact output only

The generated pattern permits no whitespace between structural tokens:
the output is always `{"a":1,"b":[2,3]}`, never `{ "a": 1 }`. JSON
treats the two as equivalent, but admitting optional whitespace at
every structural position multiplies the automaton for no gain in
expressiveness, and a deterministic surface form is easier to diff and
to assert on. Whitespace *inside* string values is unaffected — it is
ordinary string content.

Property order is likewise fixed: members are emitted in sorted key
order (see [`object_to_regex`]).

# Recursive schemas are out of reach, not just unimplemented

`$ref` is rejected like any other unsupported keyword, but it deserves
a separate note: a self-referential schema (a tree node whose child is
the same node) describes a context-free language, and no regular
expression can match balanced nesting of unbounded depth. Supporting
recursion therefore cannot be done by extending this translation — it
needs a pushdown automaton, i.e. the grammar-constrained (GBNF) path
the module plan lists as a separate Layer 2 constraint. Non-recursive
`$ref` (plain reuse) could be handled here by inlining, and is simply
not implemented yet.

## Types

- `JsonSchemaConstraint` — Restrict generation to token sequences spelling a JSON document that

