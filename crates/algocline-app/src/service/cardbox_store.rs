//! [`CardboxStore`] — a [`CardBackend`] that keeps Cards in a
//! [cardbox](https://github.com/ynishi/cardbox) root by driving the
//! `cardbox` CLI.
//!
//! # Why the CLI and not the crate
//!
//! `runcard` (the crate behind the `cardbox` binary) pulls `htl`, which
//! pulls mlua 0.12 (`mlua-sys 0.12`, `links = "lua"`). This workspace is
//! on mlua 0.11 (`mlua-sys 0.10`, same `links` key), so cargo refuses any
//! graph containing both. The CLI is a process boundary that sidesteps
//! the link conflict entirely: JSON on stdout, roughly 30 ms per call.
//!
//! # The write mapping
//!
//! These were measured against a real store of 1104 Cards. Deviating from
//! them does not fail loudly — it makes queries return zero rows.
//!
//! | v0 Card | cardbox |
//! |---|---|
//! | `pkg.name` / `scenario.name` | `open --pkg` / `--scenario` (scenario falls back to the literal `none`) |
//! | `params` | `open --params`, verbatim |
//! | `model.id` | `open --model` |
//! | `created_at` | tag `created_at` — **not** `opened_ms`, which is the log's event time |
//! | `metadata.prior_card_id` | `open --parent` |
//! | `metadata.prior_relation` | tag `lineage.relation` |
//! | `metadata.<x>` (everything else) | `close --stats` under a `metadata` key |
//! | `run.flow` / `reason` / `action` | tags `run.flow` / `run.reason` / `run.action` |
//! | `run.status` | tag `run.status` (see *Representing `skipped`*) |
//! | `stats` / `cost` | `close --stats` / `--cost` |
//! | `review` / `caveats` (via `append`) | `eval --source human` |
//!
//! The `metadata` row is the one that was verified in both directions:
//! `cardbox compat find` translates the v0 path `metadata.nn.architecture`
//! to `stats.metadata.nn.architecture`, so metadata written into `params`
//! instead makes the v0 predicate `{metadata:{nn:{architecture:…}}}`
//! return **0 rows**.
//!
//! # Representing `skipped`
//!
//! cardbox has three states and only three: `open`, `closed_ok`,
//! `closed_failed`. There is no flag for a run that was skipped, and
//! `close` with no outcome flag at all silently defaults to `closed_ok`.
//!
//! So every `close` writes the v0 status to the tag `run.status`, for all
//! three statuses uniformly, and `skipped` additionally closes the Card
//! with `--ok` — because a skipped run did not fail, and leaving the Card
//! open would strand it, which is the failure mode the lifecycle exists to
//! prevent. `get` reads `run.status` from the tag in preference to
//! `state`, so `close(skipped)` → `get` round-trips faithfully; `state`
//! remains cardbox's own coarser view. A `where` on `run.status` is
//! translated to the tag, so the three statuses stay distinguishable in a
//! query. Nothing is silently mapped to `succeeded`.
//!
//! # Stated gaps
//!
//! Places where this backend is honestly not the file backend. Each is a
//! property of cardbox 0.1.1, not an omission here.
//!
//! * **`list` / `find` rows carry no `created_at` and no `run.flow`.**
//!   `cardbox compat find` projects `card_id` / `pkg` / `scenario` /
//!   `state` / `model` / `pass_rate` and no tags, and `created_at` lives
//!   in a tag. `opened_ms` is in the row but it is a different quantity,
//!   so it is not substituted. `get` has both fields; only the summary
//!   projection lacks them.
//! * **An `append`ed `review` / `caveats` cannot be read back.** It is
//!   recorded as an eval, and cardbox 0.1.1 has no CLI verb that returns
//!   eval payloads — `get` reports only a count, surfaced here as
//!   `cardbox.evals`.
//! * **`metadata` other than the two lineage fields is close-time data.**
//!   It lands in `close --stats`, so `open` refuses it rather than
//!   dropping it; carry it on `create`, or in `close`'s `stats.metadata`.
//! * **A v0 top-level section with no slot is refused, not dropped.**
//!   `strategy_params` is the one a real Card is likely to carry.
//! * **`order_by` takes one key.** `compat find` has a single
//!   `--order-by`, so a second key would be silently dropped; it errors
//!   instead.
//! * **`where` is AND-only.** `_or` / `_not` are refused for Cards (they
//!   are fine for sample rows, which cardbox evaluates with the full v0
//!   DSL).
//! * **No `CardEvent` is published.** Card sinks mirror the file backend
//!   only; `card_sink_backfill` keeps the trait default, which errors.
//! * **`card_id` is minted by cardbox** (`{pkg}_{scenario}_{ts}_{hex}`)
//!   unless the input names one, which is passed through as `--id`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use algocline_engine::card::{
    Alias, CardBackend, CloseOutcome, CmpOp, Comparison, FindQuery, LineageDirection, LineageEdge,
    LineageNode, LineageQuery, LineageResult, OrderKey, Predicate, RunStatus, SamplesQuery,
    Summary, SCHEMA_VERSION,
};
use serde_json::{json, Map, Value as Json};

/// Written as `--scenario` when a Card names none. `cardbox open`
/// requires the flag, and a sentinel that reads as a value beats an
/// empty string that reads as a bug.
const SCENARIO_NONE: &str = "none";

/// `--source` for every Card this backend opens. cardbox records who
/// produced a Card; for this backend that is always algocline.
const SOURCE_ALC: &str = "alc";

/// Tag holding the v0 `created_at`. See the module doc.
const TAG_CREATED_AT: &str = "created_at";

/// Tag holding `metadata.prior_relation`.
const TAG_PRIOR_RELATION: &str = "lineage.relation";

/// Top-level v0 Card keys `open` knows where to put.
const OPEN_KEYS: &[&str] = &[
    "pkg",
    "scenario",
    "params",
    "model",
    "metadata",
    "created_at",
    "created_by",
    "card_id",
    "run",
    // Accepted and deliberately not forwarded: cardbox keeps its own
    // schema version and computes its own params `fingerprint`.
    "schema_version",
    "param_fingerprint",
];

/// Additional top-level keys accepted by `create`, which takes a
/// finished Card and therefore also carries the close-side sections.
const CREATE_KEYS: &[&str] = &["stats", "cost"];

/// `metadata` sub-keys `open` can place. Everything else is close-time
/// data (`stats.metadata`) — see the module doc.
const OPEN_METADATA_KEYS: &[&str] = &["prior_card_id", "prior_relation"];

/// Which verb is building the `open` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    /// `CardBackend::open` — a run that is starting.
    Lifecycle,
    /// `CardBackend::create` — a finished Card, opened and closed back
    /// to back.
    Create,
}

/// Which `--where` dialect a predicate is being rendered for.
///
/// The two are genuinely different targets. `compat find` matches Card
/// columns, so paths need translating and the matcher is AND-only;
/// `rows` matches raw sample objects with the full v0 DSL, verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhereMode {
    Cards,
    Rows,
}

/// One `cardbox` invocation's arguments plus the tags to set after it.
///
/// Tags are a second call each — cardbox has no way to set one inline —
/// so they travel beside the args rather than inside them.
#[derive(Debug, Clone, PartialEq)]
struct WritePlan {
    args: Vec<String>,
    tags: Vec<(String, String)>,
}

// ═══════════════════════════════════════════════════════════════
// CardboxStore
// ═══════════════════════════════════════════════════════════════

/// A [`CardBackend`] backed by a cardbox root, driven through the
/// `cardbox` binary. See the module doc for the mapping and its gaps.
pub struct CardboxStore {
    root: PathBuf,
    bin: PathBuf,
}

impl CardboxStore {
    /// Construct a store over `root`, invoking `bin`.
    ///
    /// `bin` is resolved by the OS, so a bare `cardbox` is looked up on
    /// `PATH`; tests pass an absolute path.
    pub fn new(root: PathBuf, bin: PathBuf) -> Self {
        Self { root, bin }
    }

    /// Construct a store over `root` using `cardbox` from `PATH`.
    pub fn with_default_bin(root: PathBuf) -> Self {
        Self::new(root, PathBuf::from(DEFAULT_BIN))
    }

    /// The cardbox root this store writes into.
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ─── The one place a subprocess is spawned ─────────────────────
    //
    // Every verb goes through `raw`, so the `CARDBOX_ROOT` binding, the
    // exit-code-to-`Err` rule, and the JSON parse each exist once.

    /// Run `cardbox <args>` and return its stdout.
    ///
    /// A non-zero exit becomes `Err(stderr.trim())`: cardbox writes a
    /// refusal as `cardbox: <message>` on stderr, which is already the
    /// sentence a caller wants to read.
    fn raw(&self, args: &[String]) -> Result<String, String> {
        let out = Command::new(&self.bin)
            .args(args)
            .env("CARDBOX_ROOT", &self.root)
            .output()
            .map_err(|e| {
                format!(
                    "cardbox: failed to run '{}': {e} \
                     (the cardbox backend needs the cardbox binary on PATH)",
                    self.bin.display()
                )
            })?;
        if !out.status.success() {
            let msg = String::from_utf8_lossy(&out.stderr).trim().to_string();
            return Err(if msg.is_empty() {
                format!("cardbox: {} exited with {}", args.join(" "), out.status)
            } else {
                msg
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Run `cardbox <args>` and parse stdout as JSON.
    fn run(&self, args: &[String]) -> Result<Json, String> {
        let text = self.raw(args)?;
        serde_json::from_str(&text)
            .map_err(|e| format!("cardbox: '{}' returned invalid JSON: {e}", args.join(" ")))
    }

    /// Run `cardbox <args>` and parse stdout as a list.
    ///
    /// cardbox returns `{}` rather than `[]` for some empty results
    /// (`get`'s `parents` / `aliases` / `checkpoints`, `prune-log`). The
    /// quirk is absorbed here so no call site repeats it.
    fn run_list(&self, args: &[String]) -> Result<Vec<Json>, String> {
        Ok(as_list(Some(&self.run(args)?)))
    }

    /// Does a Card exist under `card_id`?
    ///
    /// `cardbox get` on a miss exits 1, which is indistinguishable from a
    /// real failure without matching on message text. `find` answers the
    /// same question with an exit code of 0 and an empty list, so the
    /// miss / failure split is structural rather than textual.
    fn exists(&self, card_id: &str) -> Result<bool, String> {
        let args = argv(&["find", "--where", &format!("id = {card_id}")]);
        Ok(!self.run_list(&args)?.is_empty())
    }

    /// `cardbox get`, as the raw cardbox object. `Ok(None)` on a miss.
    fn get_raw(&self, card_id: &str) -> Result<Option<Json>, String> {
        let args = argv(&["get", card_id]);
        match self.run(&args) {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                if self.exists(card_id)? {
                    Err(e)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Apply a [`WritePlan`]'s tags to `card_id`.
    fn apply_tags(&self, card_id: &str, tags: &[(String, String)]) -> Result<(), String> {
        for (k, v) in tags {
            self.run(&argv(&["tag", "set", card_id, k, v]))?;
        }
        Ok(())
    }

    /// `cardbox open` + the tags that carry the v0 fields with no flag.
    fn open_inner(&self, input: &Json, mode: OpenMode) -> Result<String, String> {
        let plan = open_plan(input, &now_rfc3339(), mode)?;
        let opened = self.run(&plan.args)?;
        let card_id = opened
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "cardbox: open returned no 'id'".to_string())?
            .to_string();
        self.apply_tags(&card_id, &plan.tags)?;
        Ok(card_id)
    }

    /// `cardbox close` + the `run.status` tag.
    fn close_inner(
        &self,
        card_id: &str,
        outcome: &CloseOutcome,
        extra_metadata: Option<&Json>,
    ) -> Result<(), String> {
        let plan = close_plan(card_id, outcome, extra_metadata)?;
        self.run(&plan.args)?;
        self.apply_tags(card_id, &plan.tags)
    }

    /// Shared body of `list` and `find` — both are `compat find`.
    fn compat_find(&self, q: &FindQuery) -> Result<Vec<Summary>, String> {
        let mut args = argv(&["compat", "find"]);
        if let Some(pkg) = &q.pkg {
            args.push("--pkg".into());
            args.push(pkg.clone());
        }
        if let Some(pred) = &q.where_ {
            args.push("--where".into());
            args.push(where_json(pred, WhereMode::Cards)?.to_string());
        }
        if let Some(col) = order_by_flag(&q.order_by)? {
            args.push(col);
        }
        if let Some(n) = q.limit {
            args.push("--limit".into());
            args.push(n.to_string());
        }
        if let Some(n) = q.offset {
            args.push("--offset".into());
            args.push(n.to_string());
        }
        Ok(self
            .run_list(&args)?
            .iter()
            .filter_map(summary_from_row)
            .collect())
    }
}

/// The binary `with_default_bin` reaches for.
const DEFAULT_BIN: &str = "cardbox";

// ═══════════════════════════════════════════════════════════════
// CardBackend
// ═══════════════════════════════════════════════════════════════

impl CardBackend for CardboxStore {
    /// A finished Card: `open` and `close` back to back, since v0's
    /// `create` takes one that is already complete.
    fn create(&self, input: Json) -> Result<(String, PathBuf), String> {
        let card_id = self.open_inner(&input, OpenMode::Create)?;
        let outcome = CloseOutcome {
            status: run_status_of(&input).unwrap_or(RunStatus::Succeeded),
            stats: input.get("stats").cloned(),
            cost: input.get("cost").cloned(),
            error: input
                .get("error")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        };
        let metadata = closing_metadata(input.get("metadata"));
        self.close_inner(&card_id, &outcome, metadata.as_ref())?;
        Ok((card_id, self.root.clone()))
    }

    fn get(&self, card_id: &str) -> Result<Option<Json>, String> {
        match self.get_raw(card_id)? {
            Some(cb) => Ok(Some(card_from_cardbox(&cb)?)),
            None => Ok(None),
        }
    }

    fn list(&self, pkg_filter: Option<&str>) -> Result<Vec<Summary>, String> {
        self.compat_find(&FindQuery {
            pkg: pkg_filter.map(str::to_string),
            ..FindQuery::default()
        })
    }

    fn find(&self, q: FindQuery) -> Result<Vec<Summary>, String> {
        self.compat_find(&q)
    }

    /// `review` / `caveats` become a human eval. Any other top-level key
    /// is refused rather than stringified into a tag.
    fn append(&self, card_id: &str, fields: Json) -> Result<Json, String> {
        let obj = fields
            .as_object()
            .ok_or_else(|| "alc.card.append: fields must be a table".to_string())?;
        if self.get_raw(card_id)?.is_none() {
            return Err(format!("alc.card.append: card '{card_id}' not found"));
        }
        let mut eval = Map::new();
        for (k, v) in obj {
            match k.as_str() {
                "review" | "caveats" => {
                    eval.insert(k.clone(), v.clone());
                }
                other => return Err(append_refusal(other)),
            }
        }
        if !eval.is_empty() {
            let file = tempfile::Builder::new()
                .prefix("alc-cardbox-eval-")
                .suffix(".json")
                .tempfile()
                .map_err(|e| format!("alc.card.append: failed to stage eval payload: {e}"))?;
            std::fs::write(file.path(), Json::Object(eval).to_string())
                .map_err(|e| format!("alc.card.append: failed to stage eval payload: {e}"))?;
            let path = file.path().to_string_lossy().into_owned();
            self.run(&argv(&[
                "eval", card_id, "--file", &path, "--source", "human",
            ]))?;
        }
        self.get(card_id)?
            .ok_or_else(|| format!("alc.card.append: card '{card_id}' not found"))
    }

    fn alias_set(
        &self,
        name: &str,
        card_id: &str,
        _pkg: Option<&str>,
        note: Option<&str>,
    ) -> Result<Alias, String> {
        // `pkg` is not forwarded: cardbox derives an alias's pkg from the
        // Card it binds, so a caller-supplied one could only disagree
        // with the store.
        let mut args = argv(&["alias", "set", name, card_id]);
        if let Some(n) = note {
            args.push("--note".into());
            args.push(n.to_string());
        }
        self.run(&args)?;
        self.alias_list(None)?
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| format!("alc.card.alias_set: alias '{name}' did not bind"))
    }

    fn alias_list(&self, pkg_filter: Option<&str>) -> Result<Vec<Alias>, String> {
        let mut args = argv(&["alias", "list"]);
        if let Some(p) = pkg_filter {
            args.push("--pkg".into());
            args.push(p.to_string());
        }
        Ok(self
            .run_list(&args)?
            .iter()
            .filter_map(alias_from_row)
            .collect())
    }

    /// Resolved through `alias list`, not `alias get`.
    ///
    /// `alias get` on an unbound name exits 1, and v0 wants `Ok(None)`
    /// for a miss — the same structural-versus-textual split as
    /// [`CardboxStore::exists`].
    fn get_by_alias(&self, name: &str) -> Result<Option<Json>, String> {
        let Some(alias) = self.alias_list(None)?.into_iter().find(|a| a.name == name) else {
            return Ok(None);
        };
        self.get(&alias.card_id)
    }

    /// Write-once, as on the file backend: cardbox's `samples` appends
    /// unconditionally, so the rule is enforced here.
    fn write_samples(&self, card_id: &str, samples: Vec<Json>) -> Result<PathBuf, String> {
        let cb = self
            .get_raw(card_id)?
            .ok_or_else(|| format!("alc.card.write_samples: card '{card_id}' not found"))?;
        let rows = cb
            .get("samples")
            .and_then(|s| s.get("rows"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if rows > 0 {
            return Err(format!(
                "alc.card.write_samples: samples already exist for card '{card_id}' (write-once)"
            ));
        }
        let mut body = String::new();
        for row in &samples {
            body.push_str(&row.to_string());
            body.push('\n');
        }
        let file = tempfile::Builder::new()
            .prefix("alc-cardbox-samples-")
            .suffix(".jsonl")
            .tempfile()
            .map_err(|e| format!("alc.card.write_samples: failed to stage rows: {e}"))?;
        std::fs::write(file.path(), body)
            .map_err(|e| format!("alc.card.write_samples: failed to stage rows: {e}"))?;
        let path = file.path().to_string_lossy().into_owned();
        self.run(&argv(&["samples", card_id, "--file", &path]))?;
        Ok(self.root.clone())
    }

    fn read_samples(&self, card_id: &str, q: SamplesQuery) -> Result<Vec<Json>, String> {
        let mut args = argv(&["rows", card_id]);
        if let Some(pred) = &q.where_ {
            args.push("--where".into());
            args.push(where_json(pred, WhereMode::Rows)?.to_string());
        }
        if let Some(n) = q.limit {
            args.push("--limit".into());
            args.push(n.to_string());
        }
        if q.offset > 0 {
            args.push("--offset".into());
            args.push(q.offset.to_string());
        }
        self.run_list(&args)
    }

    fn lineage(&self, q: LineageQuery) -> Result<Option<LineageResult>, String> {
        let depth = q.depth.unwrap_or(DEFAULT_LINEAGE_DEPTH);
        if self.get_raw(&q.card_id)?.is_none() {
            return Ok(None);
        }
        // Ask for one level past the cap so "there was more" is an
        // observation rather than a guess.
        let probe = depth.saturating_add(1);
        let walk = self.run(&argv(&[
            "lineage",
            &q.card_id,
            "--depth",
            &probe.to_string(),
        ]))?;

        let mut wanted: Vec<(String, i32)> = vec![(q.card_id.clone(), 0)];
        let mut truncated = false;
        for (key, sign) in [("ancestors", -1i32), ("descendants", 1i32)] {
            if !direction_wants(q.direction, sign) {
                continue;
            }
            for node in as_list(walk.get(key)) {
                let (Some(id), Some(d)) = (
                    node.get("id").and_then(|v| v.as_str()),
                    node.get("depth").and_then(|v| v.as_u64()),
                ) else {
                    continue;
                };
                if d as usize > depth {
                    truncated = true;
                    continue;
                }
                wanted.push((id.to_string(), sign * d as i32));
            }
        }

        // One `get` per node: cardbox's lineage rows carry an id and a
        // depth and nothing else, while a v0 node carries pkg and the
        // lineage fields. The walk is bounded by `depth`, so this stays
        // small.
        let mut nodes = Vec::with_capacity(wanted.len());
        let mut parent_of: BTreeMap<String, (String, Option<String>)> = BTreeMap::new();
        for (id, signed_depth) in &wanted {
            let Some(cb) = self.get_raw(id)? else {
                continue;
            };
            let card = card_from_cardbox(&cb)?;
            let prior_card_id = card
                .pointer("/metadata/prior_card_id")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let prior_relation = card
                .pointer("/metadata/prior_relation")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            if let Some(parent) = &prior_card_id {
                parent_of.insert(id.clone(), (parent.clone(), prior_relation.clone()));
            }
            nodes.push(LineageNode {
                card_id: id.clone(),
                pkg: card
                    .pointer("/pkg/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                prior_card_id,
                prior_relation,
                depth: *signed_depth,
                stats: if q.include_stats {
                    card.get("stats").cloned()
                } else {
                    None
                },
            });
        }

        // `relation_filter` cuts an edge, and cutting an edge cuts the
        // subtree behind it — cardbox expands the whole walk, so the
        // reachability is recomputed here rather than filtered row-wise,
        // which would keep nodes hanging off a severed edge.
        if let Some(filter) = &q.relation_filter {
            let keep = reachable_under_filter(&q.card_id, &nodes, &parent_of, filter);
            nodes.retain(|n| keep.contains(&n.card_id));
        }

        let present: std::collections::BTreeSet<String> =
            nodes.iter().map(|n| n.card_id.clone()).collect();
        let mut edges = Vec::new();
        for n in &nodes {
            if let Some(parent) = &n.prior_card_id {
                if present.contains(parent) {
                    edges.push(LineageEdge {
                        from: n.card_id.clone(),
                        to: parent.clone(),
                        relation: n.prior_relation.clone(),
                    });
                }
            }
        }

        Ok(Some(LineageResult {
            root: q.card_id,
            nodes,
            edges,
            truncated,
        }))
    }

    fn import_cards_from_dir(
        &self,
        source_dir: &Path,
        pkg: &str,
    ) -> Result<(Vec<String>, Vec<String>), String> {
        let entries = std::fs::read_dir(source_dir).map_err(|e| {
            format!(
                "alc.card.import: failed to read {}: {e}",
                source_dir.display()
            )
        })?;
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("toml"))
            .collect();
        files.sort();

        let (mut imported, mut skipped) = (Vec::new(), Vec::new());
        for path in files {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("alc.card.import: failed to read {}: {e}", path.display()))?;
            let parsed: toml::Value = toml::from_str(&text)
                .map_err(|e| format!("alc.card.import: failed to parse {}: {e}", path.display()))?;
            let mut card = toml_to_json(parsed);
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            let obj = card
                .as_object_mut()
                .ok_or_else(|| format!("alc.card.import: {} is not a table", path.display()))?;
            obj.entry("card_id".to_string())
                .or_insert_with(|| json!(stem));
            obj.insert("pkg".to_string(), json!({ "name": pkg }));
            let card_id = obj
                .get("card_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            // First writer wins, matching the file backend.
            if self.exists(&card_id)? {
                skipped.push(card_id);
                continue;
            }
            let (id, _) = self.create(card)?;

            let samples = path.with_file_name(format!("{stem}.samples.jsonl"));
            if samples.is_file() {
                let rows = std::fs::read_to_string(&samples).map_err(|e| {
                    format!("alc.card.import: failed to read {}: {e}", samples.display())
                })?;
                let parsed: Result<Vec<Json>, _> = rows
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(serde_json::from_str::<Json>)
                    .collect();
                let parsed = parsed.map_err(|e| {
                    format!(
                        "alc.card.import: failed to parse {}: {e}",
                        samples.display()
                    )
                })?;
                if !parsed.is_empty() {
                    self.write_samples(&id, parsed)?;
                }
            }
            imported.push(id);
        }
        Ok((imported, skipped))
    }

    /// `None` — a cardbox Card is a row in a store, not a file on disk.
    fn as_file_store(&self) -> Option<&algocline_engine::FileCardStore> {
        None
    }

    fn open(&self, input: Json) -> Result<(String, PathBuf), String> {
        let card_id = self.open_inner(&input, OpenMode::Lifecycle)?;
        Ok((card_id, self.root.clone()))
    }

    fn close(&self, card_id: &str, outcome: CloseOutcome) -> Result<Json, String> {
        self.close_inner(card_id, &outcome, None)?;
        self.get(card_id)?
            .ok_or_else(|| format!("alc.card.close: card '{card_id}' not found after close"))
    }
}

/// Same default as the file backend's lineage walk.
const DEFAULT_LINEAGE_DEPTH: usize = 10;

// ═══════════════════════════════════════════════════════════════
// Pure mapping — v0 Card → CLI arguments
// ═══════════════════════════════════════════════════════════════

/// Build an argv from string literals.
fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

/// The `cardbox open` invocation for `input`, plus the tags that carry
/// the v0 fields cardbox has no flag for.
///
/// `now` is injected so the `created_at` fallback is testable.
fn open_plan(input: &Json, now: &str, mode: OpenMode) -> Result<WritePlan, String> {
    let obj = input
        .as_object()
        .ok_or_else(|| "alc.card.open: input must be a table".to_string())?;

    for key in obj.keys() {
        let known = OPEN_KEYS.contains(&key.as_str())
            || (mode == OpenMode::Create && CREATE_KEYS.contains(&key.as_str()))
            || (mode == OpenMode::Create && key == "error");
        if !known {
            return Err(unmapped_section_refusal(key));
        }
    }

    let pkg = obj
        .get("pkg")
        .and_then(|p| p.get("name"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "alc.card.open: pkg.name is required".to_string())?;
    let scenario = obj
        .get("scenario")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(SCENARIO_NONE);
    let created_by = obj
        .get("created_by")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("alc@{}", env!("CARGO_PKG_VERSION")));

    let mut args = argv(&[
        "open",
        "--pkg",
        pkg,
        "--scenario",
        scenario,
        "--source",
        SOURCE_ALC,
        "--created-by",
        &created_by,
    ]);

    if let Some(params) = obj.get("params") {
        if !params.is_object() {
            return Err("alc.card.open: params must be a table".into());
        }
        args.push("--params".into());
        args.push(params.to_string());
    }
    if let Some(model) = obj
        .get("model")
        .and_then(|m| m.get("id"))
        .and_then(|v| v.as_str())
    {
        args.push("--model".into());
        args.push(model.to_string());
    }
    if let Some(id) = obj
        .get("card_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        args.push("--id".into());
        args.push(id.to_string());
    }

    let mut tags = vec![(
        TAG_CREATED_AT.to_string(),
        obj.get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or(now)
            .to_string(),
    )];

    if let Some(meta) = obj.get("metadata") {
        let meta = meta
            .as_object()
            .ok_or_else(|| "alc.card.open: metadata must be a table".to_string())?;
        for key in meta.keys() {
            if !OPEN_METADATA_KEYS.contains(&key.as_str()) && mode == OpenMode::Lifecycle {
                return Err(open_metadata_refusal(key));
            }
        }
        if let Some(parent) = meta.get("prior_card_id").and_then(|v| v.as_str()) {
            args.push("--parent".into());
            args.push(parent.to_string());
        }
        if let Some(rel) = meta.get("prior_relation").and_then(|v| v.as_str()) {
            tags.push((TAG_PRIOR_RELATION.to_string(), rel.to_string()));
        }
    }

    if let Some(run) = obj.get("run") {
        let run = run
            .as_object()
            .ok_or_else(|| "alc.card.open: run must be a table".to_string())?;
        for field in ["flow", "reason", "action"] {
            if let Some(v) = run.get(field).and_then(|v| v.as_str()) {
                tags.push((format!("run.{field}"), v.to_string()));
            }
        }
    }

    Ok(WritePlan { args, tags })
}

/// The `cardbox close` invocation for `outcome`, plus the `run.status`
/// tag that keeps the three v0 statuses apart (see the module doc).
///
/// `extra_metadata` is the v0 `metadata` minus its two lineage fields;
/// it rides into `--stats` under a `metadata` key, which is where
/// `compat find` looks for it.
fn close_plan(
    card_id: &str,
    outcome: &CloseOutcome,
    extra_metadata: Option<&Json>,
) -> Result<WritePlan, String> {
    let mut args = argv(&["close", card_id]);
    match outcome.status {
        // `skipped` closes ok because a skipped run did not fail, and the
        // tag below is what keeps it distinct from one that succeeded.
        RunStatus::Succeeded | RunStatus::Skipped => args.push("--ok".into()),
        RunStatus::Failed => {
            args.push("--failed".into());
            args.push("--error".into());
            // cardbox refuses `--failed` without a message, while v0
            // allows `error = nil`; a named absence beats a refusal.
            args.push(
                outcome
                    .error
                    .clone()
                    .unwrap_or_else(|| "run failed (no error message reported)".to_string()),
            );
        }
    }

    let mut stats = match outcome.stats.clone() {
        Some(Json::Object(m)) => m,
        Some(_) => return Err("alc.card.close: stats must be a table".into()),
        None => Map::new(),
    };
    if let Some(Json::Object(extra)) = extra_metadata {
        if !extra.is_empty() {
            let slot = stats
                .entry("metadata".to_string())
                .or_insert_with(|| Json::Object(Map::new()));
            match slot.as_object_mut() {
                Some(existing) => {
                    for (k, v) in extra {
                        existing.insert(k.clone(), v.clone());
                    }
                }
                None => return Err("alc.card.close: stats.metadata must be a table".into()),
            }
        }
    }
    if !stats.is_empty() {
        args.push("--stats".into());
        args.push(Json::Object(stats).to_string());
    }
    if let Some(cost) = &outcome.cost {
        args.push("--cost".into());
        args.push(cost.to_string());
    }

    Ok(WritePlan {
        args,
        tags: vec![(
            "run.status".to_string(),
            status_token(outcome.status).into(),
        )],
    })
}

/// The v0 `metadata` minus the two fields that have their own slots.
fn closing_metadata(metadata: Option<&Json>) -> Option<Json> {
    let obj = metadata?.as_object()?;
    let rest: Map<String, Json> = obj
        .iter()
        .filter(|(k, _)| !OPEN_METADATA_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    (!rest.is_empty()).then_some(Json::Object(rest))
}

/// `run.status` from a whole-Card input, for `create`.
fn run_status_of(input: &Json) -> Option<RunStatus> {
    serde_json::from_value(input.get("run")?.get("status")?.clone()).ok()
}

fn status_token(s: RunStatus) -> &'static str {
    match s {
        RunStatus::Succeeded => "succeeded",
        RunStatus::Failed => "failed",
        RunStatus::Skipped => "skipped",
    }
}

fn status_from_token(s: &str) -> Option<RunStatus> {
    match s {
        "succeeded" => Some(RunStatus::Succeeded),
        "failed" => Some(RunStatus::Failed),
        "skipped" => Some(RunStatus::Skipped),
        _ => None,
    }
}

/// The refusal for a top-level section this backend cannot place.
///
/// Names what the alternative would have cost, because the tempting fix
/// — stringify it into a tag — is the one that silently breaks queries.
fn append_refusal(key: &str) -> String {
    format!(
        "alc.card.append: the cardbox backend cannot append the section '{key}'. \
         Only 'review' and 'caveats' map onto a cardbox verb (they become a \
         human eval). A tag value is a string, so storing '{key}' as one would \
         turn it into JSON text and a predicate like \
         {{{key} = {{ … }}}} would stop matching it — the section is refused \
         rather than written somewhere it cannot be queried from."
    )
}

fn unmapped_section_refusal(key: &str) -> String {
    format!(
        "alc.card: the cardbox backend has no slot for the top-level section \
         '{key}'. Recognised sections are pkg, scenario, params, model, \
         metadata, run, stats, cost, created_at, created_by and card_id. \
         Move '{key}' under 'params' (input-side) or 'stats' (result-side) so \
         it stays queryable; it is refused rather than dropped."
    )
}

fn open_metadata_refusal(key: &str) -> String {
    format!(
        "alc.card.open: metadata.{key} cannot be recorded when a Card is \
         opened. On the cardbox backend metadata other than prior_card_id / \
         prior_relation lives in the close's stats.metadata, which is where \
         a metadata.{key} query reads it from. Pass it to alc.card.close as \
         stats.metadata.{key}, or write the whole Card with alc.card.create."
    )
}

// ═══════════════════════════════════════════════════════════════
// Pure mapping — cardbox JSON → v0 Card
// ═══════════════════════════════════════════════════════════════

/// Rebuild a v0 Card from a `cardbox get` object.
///
/// The inverse of the write mapping: `stats.metadata` comes back out to
/// `metadata`, the tags come back to `created_at` / `run.*` /
/// `metadata.prior_relation`, and `parents[0]` comes back to
/// `metadata.prior_card_id`. What cardbox knows and v0 has no field for
/// is kept under a `cardbox` sub-table rather than discarded.
fn card_from_cardbox(cb: &Json) -> Result<Json, String> {
    let obj = cb
        .as_object()
        .ok_or_else(|| "cardbox: get returned a non-object".to_string())?;
    let str_at = |k: &str| obj.get(k).and_then(|v| v.as_str()).map(str::to_string);

    let card_id = str_at("id").ok_or_else(|| "cardbox: get returned no 'id'".to_string())?;
    let tags: Map<String, Json> = obj
        .get("tags")
        .and_then(|t| t.as_object())
        .cloned()
        .unwrap_or_default();
    let tag = |k: &str| tags.get(k).and_then(|v| v.as_str()).map(str::to_string);

    let mut out = Map::new();
    out.insert("schema_version".into(), json!(SCHEMA_VERSION));
    out.insert("card_id".into(), json!(card_id));
    if let Some(pkg) = str_at("pkg") {
        out.insert("pkg".into(), json!({ "name": pkg }));
    }
    // The sentinel written when a Card named no scenario is read back as
    // "named no scenario". A Card whose scenario really is the string
    // "none" is indistinguishable from one that had none — the sentinel
    // is minted here, so this direction is the one that has to give.
    if let Some(scenario) = str_at("scenario").filter(|s| s != SCENARIO_NONE) {
        out.insert("scenario".into(), json!({ "name": scenario }));
    }
    if let Some(created_by) = str_at("created_by") {
        out.insert("created_by".into(), json!(created_by));
    }
    // `created_at` is run data and `opened_ms` is the log's event time;
    // the fallback is only for a Card this backend did not write, where
    // the event time is the only timestamp there is.
    let created_at = tag(TAG_CREATED_AT).or_else(|| {
        obj.get("opened_ms")
            .and_then(|v| v.as_i64())
            .map(rfc3339_from_epoch_ms)
    });
    if let Some(ts) = created_at {
        out.insert("created_at".into(), json!(ts));
    }
    if let Some(fp) = str_at("fingerprint") {
        out.insert("param_fingerprint".into(), json!(fp));
    }
    if let Some(params) = obj.get("params") {
        out.insert("params".into(), params.clone());
    }
    if let Some(model) = str_at("model") {
        out.insert("model".into(), json!({ "id": model }));
    }
    if let Some(err) = str_at("error") {
        out.insert("error".into(), json!(err));
    }
    if let Some(cost) = obj.get("cost") {
        out.insert("cost".into(), cost.clone());
    }

    // stats, minus the metadata that was folded into it on close.
    let stats: Map<String, Json> = obj
        .get("stats")
        .and_then(|s| s.as_object())
        .cloned()
        .unwrap_or_default();
    let mut metadata: Map<String, Json> = stats
        .get("metadata")
        .and_then(|m| m.as_object())
        .cloned()
        .unwrap_or_default();
    let rest: Map<String, Json> = stats
        .iter()
        .filter(|(k, _)| k.as_str() != "metadata")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if !rest.is_empty() {
        out.insert("stats".into(), Json::Object(rest));
    }

    if let Some(parent) = as_list(obj.get("parents")).first().and_then(|v| v.as_str()) {
        metadata.insert("prior_card_id".into(), json!(parent));
    }
    if let Some(rel) = tag(TAG_PRIOR_RELATION) {
        metadata.insert("prior_relation".into(), json!(rel));
    }
    if !metadata.is_empty() {
        out.insert("metadata".into(), Json::Object(metadata));
    }

    // `[run]`: status from the tag first, since that is the only place
    // `skipped` survives; cardbox's coarser state is the fallback for a
    // Card this backend did not close.
    let state = str_at("state").unwrap_or_default();
    let status =
        tag("run.status")
            .as_deref()
            .and_then(status_from_token)
            .or(match state.as_str() {
                "closed_ok" => Some(RunStatus::Succeeded),
                "closed_failed" => Some(RunStatus::Failed),
                _ => None,
            });
    // `flow` / `reason` / `action` are known when the run starts, so
    // they are present while the Card is still open and a status is
    // not. The section is built from whatever is known rather than
    // gated on the status, which would hide what an open Card does say
    // about itself.
    let mut run = Map::new();
    if let Some(status) = status {
        run.insert("status".into(), json!(status_token(status)));
    }
    for field in ["flow", "reason", "action"] {
        if let Some(v) = tag(&format!("run.{field}")) {
            run.insert(field.into(), json!(v));
        }
    }
    if !run.is_empty() {
        out.insert("run".into(), Json::Object(run));
    }

    // What cardbox knows and a v0 Card has no field for. Kept rather
    // than dropped, and kept out of the v0 namespace rather than
    // scattered across it.
    let mut native = Map::new();
    native.insert("state".into(), json!(state));
    for k in [
        "opened_ms",
        "closed_ms",
        "evals",
        "samples",
        "source",
        "note",
        "trace_id",
        "work_url",
    ] {
        if let Some(v) = obj.get(k) {
            native.insert(k.into(), v.clone());
        }
    }
    for k in ["aliases", "checkpoints"] {
        let list = as_list(obj.get(k));
        if !list.is_empty() {
            native.insert(k.into(), Json::Array(list));
        }
    }
    out.insert("cardbox".into(), Json::Object(native));

    Ok(Json::Object(out))
}

/// A `compat find` row as a v0 [`Summary`].
///
/// `created_at` and `flow` stay `None`: the row carries no tags, which
/// is where both live, and `opened_ms` — which the row does carry — is a
/// different quantity (see the module doc).
fn summary_from_row(row: &Json) -> Option<Summary> {
    let card_id = row.get("card_id").and_then(|v| v.as_str())?.to_string();
    Some(Summary {
        card_id,
        pkg: row
            .get("pkg")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        created_at: None,
        model: row
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        flow: None,
        scenario: row
            .get("scenario")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        pass_rate: row.get("pass_rate").and_then(|v| v.as_f64()),
    })
}

/// An `alias list` row as a v0 [`Alias`].
fn alias_from_row(row: &Json) -> Option<Alias> {
    Some(Alias {
        name: row.get("name").and_then(|v| v.as_str())?.to_string(),
        card_id: row.get("card_id").and_then(|v| v.as_str())?.to_string(),
        pkg: row.get("pkg").and_then(|v| v.as_str()).map(str::to_string),
        set_at: row
            .get("bound_ms")
            .and_then(|v| v.as_i64())
            .map(rfc3339_from_epoch_ms)
            .unwrap_or_default(),
        note: row.get("note").and_then(|v| v.as_str()).map(str::to_string),
    })
}

// ═══════════════════════════════════════════════════════════════
// Pure mapping — where / order_by
// ═══════════════════════════════════════════════════════════════

/// Render a parsed [`Predicate`] back into the JSON `--where` dialect.
///
/// [`FindQuery`] carries a parsed tree rather than the JSON it came
/// from, and cardbox owns the v0 translation (`compat find`), so the
/// shortest correct path is to hand the tree back as JSON rather than
/// reimplement the translation here.
fn where_json(pred: &Predicate, mode: WhereMode) -> Result<Json, String> {
    match pred {
        Predicate::And(subs) => {
            let mut acc = Json::Object(Map::new());
            for sub in subs {
                deep_merge(&mut acc, where_json(sub, mode)?);
            }
            Ok(acc)
        }
        Predicate::Or(subs) => match mode {
            WhereMode::Rows => {
                let mut arr = Vec::with_capacity(subs.len());
                for sub in subs {
                    arr.push(where_json(sub, mode)?);
                }
                Ok(json!({ "_or": arr }))
            }
            WhereMode::Cards => Err(AND_ONLY.to_string()),
        },
        Predicate::Not(inner) => match mode {
            WhereMode::Rows => Ok(json!({ "_not": where_json(inner, mode)? })),
            WhereMode::Cards => Err(AND_ONLY.to_string()),
        },
        Predicate::Cmp(cmp) => {
            let path = match mode {
                WhereMode::Cards => translate_card_path(&cmp.path)?,
                WhereMode::Rows => cmp.path.clone(),
            };
            Ok(nest(&path, leaf_value(cmp)))
        }
    }
}

const AND_ONLY: &str = "alc.card.find: the cardbox backend cannot evaluate '_or' / '_not' \
     over Cards — cardbox's compat find is AND-only, and it refuses a \
     where holding either rather than half-answering it. Split the query, \
     or filter the results. (Sample-row queries do support both.)";

/// The right-hand side of one comparison: a bare value for `eq`, an
/// operator object otherwise. This is exactly what the v0 parser reads.
fn leaf_value(cmp: &Comparison) -> Json {
    match cmp.op {
        CmpOp::Eq => cmp.value.clone(),
        ref op => {
            let mut m = Map::new();
            m.insert(op_key(op).to_string(), cmp.value.clone());
            Json::Object(m)
        }
    }
}

fn op_key(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Lte => "lte",
        CmpOp::Gt => "gt",
        CmpOp::Gte => "gte",
        CmpOp::In => "in",
        CmpOp::Nin => "nin",
        CmpOp::Exists => "exists",
        CmpOp::Contains => "contains",
        CmpOp::StartsWith => "starts_with",
    }
}

/// Wrap `leaf` in one object per path segment.
fn nest(path: &[String], leaf: Json) -> Json {
    let mut acc = leaf;
    for key in path.iter().rev() {
        let mut m = Map::new();
        m.insert(key.clone(), acc);
        acc = Json::Object(m);
    }
    acc
}

/// Merge `src` into `dst`, descending into objects present in both.
fn deep_merge(dst: &mut Json, src: Json) {
    match (dst.as_object_mut(), src) {
        (Some(d), Json::Object(s)) => {
            for (k, v) in s {
                match d.get_mut(&k) {
                    Some(existing) if existing.is_object() && v.is_object() => {
                        deep_merge(existing, v)
                    }
                    _ => {
                        d.insert(k, v);
                    }
                }
            }
        }
        (_, src) => *dst = src,
    }
}

/// Translate a v0 Card path into the column `compat find` matches on.
///
/// `metadata.*` is deliberately left alone: cardbox's own compat layer
/// already rewrites it to `stats.metadata.*`, and translating it here
/// too would produce `stats.stats.metadata.*`.
fn translate_card_path(path: &[String]) -> Result<Vec<String>, String> {
    let seg = |s: &str| vec![s.to_string()];
    let parts: Vec<&str> = path.iter().map(String::as_str).collect();
    Ok(match parts.as_slice() {
        ["created_at"] => vec!["tags".into(), TAG_CREATED_AT.into()],
        // `run` has no cardbox column; its fields are tags, and the tag
        // key is the dotted name literally (tags are a flat map).
        ["run", field @ ("status" | "flow" | "reason" | "action")] => {
            vec!["tags".into(), format!("run.{field}")]
        }
        ["run"] => {
            return Err(
                "alc.card.find: filter a field of 'run' (run.status / run.flow / \
                 run.reason / run.action), not the whole section — on the cardbox \
                 backend each is a separate tag."
                    .into(),
            )
        }
        ["metadata", "prior_relation"] => vec!["tags".into(), TAG_PRIOR_RELATION.into()],
        ["metadata", "prior_card_id"] => {
            return Err(
                "alc.card.find: metadata.prior_card_id is not a queryable column on \
                 the cardbox backend — a parent is an edge there, not a field. \
                 Use alc.card.lineage to ask about ancestry."
                    .into(),
            )
        }
        ["scenario", "name"] => seg("scenario"),
        ["model", "id"] => seg("model"),
        ["pkg", "name"] => seg("pkg"),
        ["card_id"] => seg("id"),
        ["param_fingerprint"] => seg("fingerprint"),
        _ => path.to_vec(),
    })
}

/// The `--order-by=…` flag for `order_by`, if any.
///
/// `compat find` takes a single sort key. A second one would be dropped
/// on the floor and silently change the order, so it errors instead.
fn order_by_flag(keys: &[OrderKey]) -> Result<Option<String>, String> {
    match keys {
        [] => Ok(None),
        [k] => {
            let col = translate_card_path(&k.path)?.join(".");
            let dash = if k.desc { "-" } else { "" };
            Ok(Some(format!("--order-by={dash}{col}")))
        }
        _ => Err(format!(
            "alc.card.find: the cardbox backend sorts on one key, not {}. \
             cardbox's compat find takes a single --order-by, so the remaining \
             keys would be dropped and the order would quietly differ.",
            keys.len()
        )),
    }
}

// ═══════════════════════════════════════════════════════════════
// Small shared helpers
// ═══════════════════════════════════════════════════════════════

/// Read a cardbox value as a list.
///
/// cardbox emits `{}` rather than `[]` for an empty `parents` /
/// `aliases` / `checkpoints`, so a list is "an array, or nothing".
fn as_list(v: Option<&Json>) -> Vec<Json> {
    match v {
        Some(Json::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

/// Does `direction` want edges walked with this sign (-1 up, +1 down)?
fn direction_wants(direction: LineageDirection, sign: i32) -> bool {
    match direction {
        LineageDirection::Both => true,
        LineageDirection::Up => sign < 0,
        LineageDirection::Down => sign > 0,
    }
}

/// The nodes still reachable from `root` when only edges whose relation
/// is in `filter` may be followed.
fn reachable_under_filter(
    root: &str,
    nodes: &[LineageNode],
    parent_of: &BTreeMap<String, (String, Option<String>)>,
    filter: &[String],
) -> std::collections::BTreeSet<String> {
    let passes = |relation: &Option<String>| {
        relation
            .as_deref()
            .is_some_and(|r| filter.iter().any(|f| f == r))
    };
    let mut keep = std::collections::BTreeSet::new();
    keep.insert(root.to_string());

    // Walk up from the root while each edge passes.
    let mut cursor = root.to_string();
    while let Some((parent, relation)) = parent_of.get(&cursor) {
        if !passes(relation) {
            break;
        }
        keep.insert(parent.clone());
        cursor = parent.clone();
    }

    // Walk down: a child is kept when its own edge passes and its parent
    // is already kept. Repeat until nothing new is added, so the order
    // `nodes` arrives in does not matter.
    loop {
        let before = keep.len();
        for n in nodes {
            if let Some(parent) = &n.prior_card_id {
                if keep.contains(parent) && passes(&n.prior_relation) {
                    keep.insert(n.card_id.clone());
                }
            }
        }
        if keep.len() == before {
            break;
        }
    }
    keep
}

/// RFC3339 UTC `YYYY-MM-DDTHH:MM:SSZ` for the current system time.
fn now_rfc3339() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    rfc3339_from_epoch_ms(ms)
}

/// RFC3339 UTC `YYYY-MM-DDTHH:MM:SSZ` from epoch milliseconds.
///
/// cardbox timestamps are epoch ms; v0 timestamps are RFC3339 strings,
/// and `card_context` slices `[5..10]` out of one to render `MM/DD`.
fn rfc3339_from_epoch_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (y, mo, d) = civil_from_days(days);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Howard Hinnant's `civil_from_days`: days since the Unix epoch to a
/// civil (year, month, day).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `toml::Value` → `serde_json::Value`, for `import_cards_from_dir`.
fn toml_to_json(v: toml::Value) -> Json {
    match v {
        toml::Value::String(s) => Json::String(s),
        toml::Value::Integer(i) => json!(i),
        toml::Value::Float(f) => json!(f),
        toml::Value::Boolean(b) => Json::Bool(b),
        toml::Value::Datetime(dt) => Json::String(dt.to_string()),
        toml::Value::Array(a) => Json::Array(a.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => {
            Json::Object(t.into_iter().map(|(k, v)| (k, toml_to_json(v))).collect())
        }
    }
}

#[cfg(test)]
mod tests;
