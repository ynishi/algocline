--- gameai_metrics.harvest_collection — collect one boss checkpoint per
--- band, write the manifest as JSON.
---
--- The staged / band judgment in `anymetric` emits `harvest` decisions
--- whose `meta` carries the hit band's `label` plus the observing step
--- and the raw view values. The trainer hook already appends a marker
--- record into the run log; this helper is the caller-side utility that
--- turns those markers plus the surrounding `records` (level / style
--- distance / trickiness observed on the same fire) into one entry per
--- label, and writes the whole thing out as a boss harvest manifest.
---
--- ## Why this lives in gameai (not anymetric)
---
--- `label` / `ckpt_path` / `card_id` / `alias` are all boss-harvest
--- vocabulary: they name a stage of the guardian's training arc and a
--- Card baked from a mid-run checkpoint. `anymetric` intentionally
--- speaks only in views / records / judgments and knows nothing about
--- Cards or ckpt paths, so keeping the collection helper here lets
--- `anymetric` stay eligible for later promotion into `bundled-packages`
--- without dragging gameai vocabulary along.
---
--- ## Layers
---
--- - **domain (anymetric)** — produces harvest Decision `{action, reason, meta}`
---   and appends the marker into the run log. Its `to_hook_action` also
---   answers the trainer with a keep, which pins the checkpoint out of
---   the rotation — that is what makes the `ckpt_path` recorded below
---   still resolve after the run. Before the keep existed, a manifest
---   written early in a long run named a file `ckpt_keep` rotations
---   later had already deleted.
--- - **caller (train script hook)** — decides on the harvest, bakes the
---   Card via `alc.nn.card.save_from_ckpt`, pins the alias, and hands
---   the (dec, info, records, extra) tuple to this collection helper.
--- - **collection (this module)** — extracts one entry per band label
---   (first-writer-wins by default), stores it in memory, and writes the
---   whole manifest as JSON on `:save()`.
---
--- The helper never reads back its own file and never queries the
--- registry — every value written into the manifest comes from the four
--- arguments to `:append(dec, info, records, extra)`. That keeps the
--- observation → collection direction one-way and makes the helper safe
--- to call from inside a hook (no filesystem read on the hot path).
---
--- ## Usage from a trainer `on_ckpt` hook
---
--- ```lua
--- local hc = require("gameai_metrics.harvest_collection")
--- local coll = hc.new({
---     path = "workspace/gameai-harvest/collection.json",
---     meta = { run_id = RUN_ID, style = "guardian", steps = 300,
---              ckpt_every = 60, gate_games = 50, seed = 0 },
---     bands = {
---         { lo = 0.10, hi = 0.30, label = "weak"   },
---         { lo = 0.55, hi = 0.85, label = "mid"    },
---         { lo = 0.85, hi = 0.98, label = "strong" },
---     },
---     -- policy = "first_writer_wins",   -- default
--- })
---
--- -- inside on_ckpt, after the staged judgment has returned a harvest:
--- if dec.action == "harvest" then
---     local card_id = alc.nn.card.save_from_ckpt(info.ckpt_path,
---         string.format("guardian_duel_npc_%s", dec.meta.label), meta)
---     local alias = string.format("guardian_duel_npc_%s", dec.meta.label)
---     alc.card.alias_set(alias, card_id, { note = "boss harvest" })
---     coll:append(dec, info, records, { card_id = card_id, alias = alias })
---     coll:save()   -- write-through, so a mid-run crash still leaves the entries collected so far.
--- end
--- ```
---
--- ## Field extraction rule
---
--- `:append` reads records by their `view_id`:
---
--- - `view_id = "level"` — `values.win_rate` -> `level_win_rate`, `values.ci_lower` -> `level_ci_lower`.
--- - `view_id = "sd_teacher"` — `values.value` -> `sd_teacher` (the
---   scalar Jensen-Shannon distance from `style_distance` lifted to
---   `{value = ...}` by anymetric's `lift_values`).
--- - `view_id = "trickiness"` — `values.value` -> `trickiness_norm` (the
---   normalised boss-seat entropy `trickiness` returns as `{value, raw_mean}`).
---
--- A record for one of these view ids being absent or carrying an
--- `error` field is fine — the corresponding field is left absent in
--- the entry rather than raising. The whole point of partial collection
--- is that a run can produce a manifest with one entry (mid only) or
--- two (weak + mid, strong never fired) and still be useful for a
--- Human observation note.

local fs = require("gameai_metrics._fs")

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics.harvest_collection",
    version = "0.1.0",
    description = "Collect one Card per band from staged harvest decisions and write a boss-harvest manifest.",
    category = "game",
}

local SCHEMA_VERSION = 1
local POLICIES = { first_writer_wins = true, last_writer_wins = true }

--- Copy an array of band tables, keeping the fields the manifest
--- records. The staged judgment already validated shape / disjointness
--- before it was handed to the caller, so this helper only refuses
--- obvious wiring mistakes (non-array / non-table entry / non-number
--- lo|hi / non-string label / non-table require).
---
--- A band's optional `require = {view_id, field, min}` is copied across
--- field by field. Without it the manifest would record a band as a bare
--- `{lo, hi, label}` interval and the second condition that actually
--- decided the harvest would be missing from the provenance — a reader
--- of the manifest could not tell a plain band hit from one that also
--- had to clear a floor on another view. The three fields are named
--- rather than shallow-copied so a caller's stray key cannot ride into
--- the manifest as if it were part of the contract.
local function normalise_bands(bands)
    if type(bands) ~= "table" then
        error(
            "harvest_collection.new: bands must be an array of {lo, hi, label}, got " .. type(bands),
            3
        )
    end
    local out = {}
    for index, band in ipairs(bands) do
        if type(band) ~= "table" then
            error(
                "harvest_collection.new: bands[" .. index .. "] must be a table, got " .. type(band),
                3
            )
        end
        if type(band.lo) ~= "number" then
            error(
                "harvest_collection.new: bands["
                    .. index
                    .. "].lo must be a number, got "
                    .. type(band.lo),
                3
            )
        end
        if type(band.hi) ~= "number" then
            error(
                "harvest_collection.new: bands["
                    .. index
                    .. "].hi must be a number, got "
                    .. type(band.hi),
                3
            )
        end
        if band.label ~= nil and type(band.label) ~= "string" then
            error(
                "harvest_collection.new: bands["
                    .. index
                    .. "].label must be a string or nil, got "
                    .. type(band.label),
                3
            )
        end
        if band.require ~= nil and type(band.require) ~= "table" then
            error(
                "harvest_collection.new: bands["
                    .. index
                    .. "].require must be a table or nil, got "
                    .. type(band.require),
                3
            )
        end
        out[index] = { lo = band.lo, hi = band.hi, label = band.label }
        if band.require ~= nil then
            out[index].require = {
                view_id = band.require.view_id,
                field = band.require.field,
                min = band.require.min,
            }
        end
    end
    return out
end

--- Shallow-copy the caller-supplied run metadata so a later mutation of
--- the original table cannot bleed into what `:save()` writes out.
local function copy_meta(meta)
    if meta == nil then
        return {}
    end
    if type(meta) ~= "table" then
        error("harvest_collection.new: meta must be a table or nil, got " .. type(meta), 3)
    end
    local out = {}
    for key, value in pairs(meta) do
        out[key] = value
    end
    return out
end

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "harvest_collection: alc.json_encode is required to write the manifest (host bridge missing)",
            0
        )
    end
    return alc.json_encode
end

--- Look up the record for `view_id`; return `nil` when it is absent or
--- carries an `error` field (the ErrorRecord shape observe uses). The
--- caller extracts a specific field off `.values` after this returns,
--- so an ErrorRecord read as "no record" is what keeps a broken metric
--- from writing garbage into the manifest.
local function find_record(records, view_id)
    if type(records) ~= "table" then
        return nil
    end
    for _, record in ipairs(records) do
        if type(record) == "table" and rawget(record, "view_id") == view_id then
            if record.error ~= nil then
                return nil
            end
            return record
        end
    end
    return nil
end

local function numeric_field(record, field)
    if record == nil then
        return nil
    end
    local values = record.values
    if type(values) ~= "table" then
        return nil
    end
    local raw = values[field]
    if type(raw) ~= "number" then
        return nil
    end
    return raw
end

--- Build the per-fire entry from the four caller arguments. Only fields
--- that are actually present are set; the rest stay absent so the JSON
--- output shows a partial harvest as an entry with missing keys rather
--- than as one packed with nulls that look like measurements.
local function build_entry(label, dec, info, records, extra)
    local entry = {
        label = label,
        step = info.step,
        ckpt_path = info.ckpt_path,
    }
    if extra ~= nil then
        if type(extra) ~= "table" then
            error("harvest_collection:append: extra must be a table or nil, got " .. type(extra), 3)
        end
        for key, value in pairs(extra) do
            if entry[key] == nil then
                entry[key] = value
            end
        end
    end
    -- level view
    local level_record = find_record(records, "level")
    local level_win = numeric_field(level_record, "win_rate")
    if level_win ~= nil then
        entry.level_win_rate = level_win
    end
    local level_ci = numeric_field(level_record, "ci_lower")
    if level_ci ~= nil then
        entry.level_ci_lower = level_ci
    end
    -- style_distance view, addressed by the caller-chosen id "sd_teacher"
    local sd_record = find_record(records, "sd_teacher")
    local sd_value = numeric_field(sd_record, "value")
    if sd_value ~= nil then
        entry.sd_teacher = sd_value
    end
    -- trickiness view; caller wires the boss-seat view as view_id="trickiness"
    local tr_record = find_record(records, "trickiness")
    local tr_value = numeric_field(tr_record, "value")
    if tr_value ~= nil then
        entry.trickiness_norm = tr_value
    end
    -- Keep the harvest reason around for debugging; downstream tooling
    -- can ignore it.
    if dec.reason ~= nil then
        entry.reason = dec.reason
    end
    return entry
end

local Collection = {}
Collection.__index = Collection

--- Append one harvest tuple. Returns `true` when the entry was stored,
--- `false` when it was skipped (first-writer-wins already has an entry
--- for the same label). Raises when the decision does not carry a
--- `harvest` action or is missing `meta.label` — that is a caller
--- wiring bug, not a runtime measurement gap.
---
---@param dec table Decision `{action = "harvest", reason, meta = {label, ...}}`
---@param info table trainer `on_ckpt` info `{step, ckpt_path, ...}`
---@param records table the same records the judgment read this fire
---@param extra table|nil optional extra fields for the entry (e.g. `{card_id, alias}`)
---@return boolean stored true when appended, false when skipped by policy
function Collection:append(dec, info, records, extra)
    if type(dec) ~= "table" then
        error("harvest_collection:append: dec must be a table, got " .. type(dec), 2)
    end
    if dec.action ~= "harvest" then
        error(
            "harvest_collection:append: dec.action must be 'harvest', got " .. tostring(dec.action),
            2
        )
    end
    if type(dec.meta) ~= "table" then
        error(
            "harvest_collection:append: dec.meta must be a table with a label, got "
                .. type(dec.meta),
            2
        )
    end
    local label = dec.meta.label
    if type(label) ~= "string" or label == "" then
        error("harvest_collection:append: dec.meta.label must be a non-empty string", 2)
    end
    if type(info) ~= "table" then
        error("harvest_collection:append: info must be a table, got " .. type(info), 2)
    end
    if type(info.step) ~= "number" then
        error("harvest_collection:append: info.step must be a number, got " .. type(info.step), 2)
    end
    if info.ckpt_path ~= nil and type(info.ckpt_path) ~= "string" then
        error(
            "harvest_collection:append: info.ckpt_path must be a string or nil, got "
                .. type(info.ckpt_path),
            2
        )
    end

    local existing_index = self._by_label[label]
    if existing_index ~= nil then
        if self._policy == "first_writer_wins" then
            return false
        end
        -- last_writer_wins: overwrite in place, preserving order.
        self._entries[existing_index] = build_entry(label, dec, info, records, extra)
        return true
    end
    local entry = build_entry(label, dec, info, records, extra)
    self._entries[#self._entries + 1] = entry
    self._by_label[label] = #self._entries
    return true
end

--- Return a shallow copy of the entries stored so far, in the order
--- they were first appended. The copy keeps a spec (or the training
--- script itself) from shortening the collection by mutating what it
--- reads.
---@return table entries
function Collection:entries()
    local out = {}
    for index, entry in ipairs(self._entries) do
        out[index] = entry
    end
    return out
end

--- Build the JSON payload without touching the filesystem. Exposed so a
--- spec can round-trip through the same shape `:save()` writes without
--- needing a temp file. The returned table is fresh on every call.
---@return table manifest
function Collection:_manifest()
    local manifest = { schema_version = SCHEMA_VERSION }
    for key, value in pairs(self._meta) do
        if manifest[key] == nil then
            manifest[key] = value
        end
    end
    manifest.bands = self._bands
    manifest.policy = self._policy
    manifest.entries = self:entries()
    return manifest
end

--- Encode the manifest and write it to the configured path. The file
--- is truncated on every call so the manifest reflects the current
--- state exactly; the collection is small enough (one entry per band)
--- that rewriting on every append is cheaper than any diff scheme.
--- A `write` failure raises loudly — a silent partial manifest would
--- be worse than an obvious crash, because a Human reading it later
--- would take the stale numbers at face value.
function Collection:save()
    local encode = require_json_encoder()
    local ok, encoded = pcall(encode, self:_manifest())
    if not ok then
        error("harvest_collection:save: failed to encode manifest: " .. tostring(encoded), 2)
    end
    -- Idempotent parent-dir mkdir: removes the previous iteration's
    -- Run 1 accident class (missing workspace/gameai-harvest/ crashed
    -- the trainer hook mid-run). See gameai_metrics._fs for the shell
    -- escape rationale.
    fs.ensure_parent_dir(self._path)
    local f, err = io.open(self._path, "w")
    if f == nil then
        error(
            string.format(
                "harvest_collection:save: cannot open %q for writing: %s",
                self._path,
                tostring(err)
            ),
            2
        )
    end
    local ok_write, write_err = pcall(function()
        f:write(encoded)
    end)
    f:close()
    if not ok_write then
        error(
            string.format(
                "harvest_collection:save: write to %q failed: %s",
                self._path,
                tostring(write_err)
            ),
            2
        )
    end
end

--- Configured output path (read-only accessor for specs / logs).
---@return string path
function Collection:path()
    return self._path
end

--- Configured policy (read-only accessor for specs / logs).
---@return string policy
function Collection:policy()
    return self._policy
end

--- Build a new collection. `opts` is a table:
---
--- - `path`   — string, output file path (required).
--- - `meta`   — table of caller-supplied run metadata (optional).
---   Every key is copied verbatim into the top level of the manifest,
---   e.g. `{run_id, style, steps, ckpt_every, gate_games, seed}`.
--- - `bands`  — array of `{lo, hi, label?, require?}` band tables
---   (required). Recorded into the manifest as-is, `require` included,
---   so the selection rule a harvest actually ran under stays readable
---   off the manifest alone; the staged judgment is the one that
---   enforces disjointness and validates `require` at read time.
--- - `policy` — string, `"first_writer_wins"` (default) or
---   `"last_writer_wins"`. `first_writer_wins` matches the
---   "pick the earliest fire that lands in the band" observation
---   pattern from the previous iteration; `last_writer_wins` lets a
---   caller take the latest hit instead.
---
---@param opts table
---@return table collection
function M.new(opts)
    if type(opts) ~= "table" then
        error("harvest_collection.new: opts must be a table, got " .. type(opts), 2)
    end
    if type(opts.path) ~= "string" or opts.path == "" then
        error("harvest_collection.new: opts.path must be a non-empty string", 2)
    end
    if opts.bands == nil then
        error("harvest_collection.new: opts.bands is required", 2)
    end
    local bands = normalise_bands(opts.bands)
    local meta = copy_meta(opts.meta)
    local policy = opts.policy or "first_writer_wins"
    if not POLICIES[policy] then
        error(
            "harvest_collection.new: unknown policy '"
                .. tostring(policy)
                .. "'; expected first_writer_wins or last_writer_wins",
            2
        )
    end
    return setmetatable({
        _path = opts.path,
        _meta = meta,
        _bands = bands,
        _policy = policy,
        _entries = {},
        _by_label = {},
    }, Collection)
end

M.SCHEMA_VERSION = SCHEMA_VERSION

return M
