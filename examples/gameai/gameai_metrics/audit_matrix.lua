--- gameai_metrics.audit_matrix — per-Card baseline plus pair-wise
--- style-distance matrix runner for a boss-harvest Card collection.
---
--- ## What this is (and is not)
---
--- `harvest_collection` writes one entry per band during a training run
--- and pins the `card_id` of each baked Card into the resulting
--- manifest. `audit_matrix` reads that manifest (or an explicit alias
--- list), reloads every Card by `card_id`, and produces two views for a
--- human observer:
---
--- 1. **per_card baseline** — every Card is re-measured on the same
---    three metrics that fired during training (`level` / `trickiness`
---    / `sd_teacher`). Re-measuring at a higher `n_games` than the
---    mid-run gate could afford tightens the Wilson interval without
---    re-training. The `sd_teacher` view is only emitted when a
---    `teacher_alias` is supplied — the teacher pin is optional.
---
--- 2. **pair-wise SD matrix** — `style_distance` is evaluated on every
---    unordered pair of Cards. Jensen-Shannon divergence is symmetric
---    by construction, so the runner computes each pair only once
---    (`(i, j)` with `i < j`) and mirrors the value to `sd_matrix[j][i]`;
---    the diagonal is filled with `0.0` so a caller can look up any
---    pair without knowing the iteration order.
---
--- Nothing here is a decision: the runner never compares its numbers
--- against a target, never picks a winner. What comes out is the raw
--- material a human reads to answer two questions the audit is set up
--- to inform:
---
--- - Does the strong / mid / weak separation observed during training
---   reproduce once the per-Card metrics are re-measured at higher `n`?
--- - Are the three Cards' policies actually far enough apart (large
---   pair-wise SD) to be worth pitting against each other in a
---   Card-vs-Card fight iteration?
---
--- ## Why it lives in `gameai_metrics` (not `anymetric`)
---
--- The runner speaks in boss-harvest vocabulary (`alias`, `card_id`,
--- `teacher_alias`, "boss state" as the prompt-set element type). That
--- is gameai-specific by design — `anymetric` deliberately stays free
--- of Card / seat / style vocabulary so it can be promoted into a
--- bundled package later. Keeping the audit runner one shelf up on the
--- gameai side is the same call `harvest_collection` made for the same
--- reason.
---
--- ## Contract
---
--- ```lua
--- local am = require("gameai_metrics.audit_matrix")
---
--- local audit = am.new({
---     collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
---     -- or, without a manifest:
---     -- aliases = { "guardian_duel_npc_weak", "_mid", "_strong" },
---     n_games = 200,           -- per-Card baseline sample size (default)
---     prompt_set_size = 16,    -- boss states for trickiness / SD (default)
---     seed = 20260731,
---     style = "guardian",      -- required, one of guardian_duel.STYLES
---     teacher_alias = "guardian_duel_npc",  -- optional
--- })
---
--- local report = audit:run()
--- audit:save("workspace/gameai-harvest/audit_run2.json")
--- ```
---
--- The report shape is
---
--- ```
--- {
---     per_card = {
---         [alias] = {
---             win_rate      = 0.42,
---             ci_lower      = 0.30,
---             ci_upper      = 0.55,
---             trickiness_norm = 0.41,   -- boss-seat normalised entropy
---             sd_teacher    = 0.089,    -- absent when teacher_alias is nil
---             step          = 180,      -- inherited from the harvest entry
---         },
---         ...
---     },
---     sd_matrix = {
---         [alias_i] = { [alias_j] = <number>, ... },   -- symmetric, diag 0
---         ...
---     },
---     meta = { n_games, prompt_set_size, seed, style, teacher_alias, ... },
--- }
--- ```
---
--- `save()` writes the same object as JSON via `alc.json_encode`, after
--- calling `gameai_metrics._fs.ensure_parent_dir(path)` so a fresh
--- `workspace/` sub-tree is created on the first run rather than
--- surfacing as an obscure `io.open` `No such file or directory` (the
--- same failure mode `harvest_collection` prevented at ST-0).

local am = require("anymetric")
local duel = require("guardian_duel")
local boss_seat = require("gameai_metrics.boss_seat")
local fs = require("gameai_metrics._fs")

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics.audit_matrix",
    version = "0.1.0",
    description = "Per-Card baseline + pair-wise SD matrix runner for a boss-harvest Card collection.",
    category = "game",
}

local DEFAULT_N_GAMES = 200
local DEFAULT_PROMPT_SET_SIZE = 16
local DEFAULT_SEED = 0

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "audit_matrix: alc.json_encode is required to write the manifest (host bridge missing)",
            0
        )
    end
    return alc.json_encode
end

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "audit_matrix: alc.json_decode is required to read the collection manifest "
                .. "(host bridge missing)",
            0
        )
    end
    return alc.json_decode
end

local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("audit_matrix: alc.nn.card is unavailable; build algocline with --features nn")
    end
    if type(alc.nn.card.load_handle) ~= "function" then
        error("audit_matrix: alc.nn.card.load_handle is unavailable")
    end
end

local function require_registry()
    local reg = alc and alc.nn and alc.nn.metric and alc.nn.metric.registry
    if type(reg) ~= "table" or type(reg.evaluate) ~= "function" then
        error(
            "audit_matrix: alc.nn.metric.registry is not available on this VM "
                .. "(build without the nn feature? require gameai_metrics first)",
            0
        )
    end
    return reg
end

local function require_card_alias_bridge()
    if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
        error(
            "audit_matrix: alc.card.get_by_alias is unavailable "
                .. "(needed to resolve an alias to a card_id when opts.aliases is set)"
        )
    end
end

-- ─── Option validation ─────────────────────────────────────────────

local function decode_int(raw, default, field, must_be_positive)
    if raw == nil then
        return default
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge then
        error(
            string.format("audit_matrix: %s must be a finite integer, got %s", field, tostring(raw)),
            3
        )
    end
    local i = math.floor(raw)
    if must_be_positive and i <= 0 then
        error(
            string.format(
                "audit_matrix: %s must be a positive integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    if not must_be_positive and i < 0 then
        error(
            string.format(
                "audit_matrix: %s must be a non-negative integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    return i
end

--- Build a prompt set of boss states by seeding fresh games. The
--- caller-supplied `opts.prompt_set` overrides this (so an audit can
--- reuse the exact `check_states` a training run measured against),
--- but when it is absent the runner produces `prompt_set_size` states
--- deterministically from `seed`. Each seed lands a mode-0 opening
--- state; that is the same shape `boss_seat.require_state` accepts.
local function build_prompt_set(seed, size)
    local out = {}
    for i = 1, size do
        out[i] = duel.new_game(seed + i).boss
    end
    return out
end

--- Normalise `opts.aliases` into a `{alias, card_id}` pair list.
--- Each `card_id` is resolved lazily through `alc.card.get_by_alias`;
--- an alias without a bound Card is a loud error (same treatment the
--- collection-path branch gives to a missing `card_id`).
local function resolve_alias_list(aliases)
    if type(aliases) ~= "table" or #aliases == 0 then
        error("audit_matrix: opts.aliases must be a non-empty array of alias strings", 3)
    end
    require_card_alias_bridge()
    local seen = {}
    local out = {}
    for i, alias in ipairs(aliases) do
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format(
                    "audit_matrix: opts.aliases[%d] must be a non-empty string, got %s",
                    i,
                    type(alias)
                ),
                3
            )
        end
        if seen[alias] then
            error(string.format("audit_matrix: opts.aliases lists %q twice", alias), 3)
        end
        seen[alias] = true
        local entry = alc.card.get_by_alias(alias)
        if type(entry) ~= "table" or type(entry.card_id) ~= "string" or entry.card_id == "" then
            error(
                string.format(
                    "audit_matrix: alias %q is not bound to a Card with a card_id "
                        .. "(alc.card.get_by_alias returned no card_id)",
                    alias
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = entry.card_id, step = nil }
    end
    return out
end

--- Read a harvest manifest and turn its `entries` array into the same
--- `{alias, card_id, step}` shape the aliases path produces. Missing
--- `card_id` on any entry is a loud error (a partial audit that
--- silently skipped Cards would produce a matrix whose axes did not
--- match the caller's expectation).
local function resolve_from_collection(path)
    if type(path) ~= "string" or path == "" then
        error("audit_matrix: opts.collection_path must be a non-empty string", 3)
    end
    local decode = require_json_decoder()
    local f, open_err = io.open(path, "r")
    if f == nil then
        error(
            string.format(
                "audit_matrix: cannot open collection_path %q: %s",
                path,
                tostring(open_err)
            ),
            3
        )
    end
    local body = f:read("a")
    f:close()
    local ok, parsed = pcall(decode, body)
    if not ok then
        error(
            string.format(
                "audit_matrix: failed to decode collection_path %q: %s",
                path,
                tostring(parsed)
            ),
            3
        )
    end
    if type(parsed) ~= "table" or type(parsed.entries) ~= "table" then
        error(
            string.format(
                "audit_matrix: collection_path %q has no entries array (schema drift?)",
                path
            ),
            3
        )
    end
    if #parsed.entries == 0 then
        error(
            string.format(
                "audit_matrix: collection_path %q has zero entries; nothing to audit",
                path
            ),
            3
        )
    end
    local seen = {}
    local out = {}
    for i, entry in ipairs(parsed.entries) do
        if type(entry) ~= "table" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] must be a table, got %s",
                    i,
                    type(entry)
                ),
                3
            )
        end
        local alias = entry.alias
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format("audit_matrix: collection_path entries[%d] has no alias string", i),
                3
            )
        end
        if seen[alias] then
            error(
                string.format("audit_matrix: collection_path entries list alias %q twice", alias),
                3
            )
        end
        seen[alias] = true
        local card_id = entry.card_id
        if type(card_id) ~= "string" or card_id == "" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] (alias %q) has no card_id; "
                        .. "the audit cannot reload a Card without its stored card_id",
                    i,
                    alias
                ),
                3
            )
        end
        local step = entry.step
        if step ~= nil and type(step) ~= "number" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] (alias %q) has non-numeric step %s",
                    i,
                    alias,
                    tostring(step)
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = card_id, step = step }
    end
    return out
end

-- ─── Views ─────────────────────────────────────────────────────────

--- Assemble the per-Card views. `n_games` / `seed` are pushed into
--- the `level` view config so the level registry entry
--- (`alc.nn.metric.registry.register("level", ...)`) picks them up
--- per-fire; the trickiness / style_distance views need neither.
--- The `sd_teacher` view is only added when a `teacher_alias` was
--- supplied; without one the view would have no reference Card to
--- measure against.
local function build_views(style, prompt_set, teacher_alias, n_games, seed)
    local views = {
        am.view("level", "level", {
            seat = "boss",
            opponents = { "random" },
            style = style,
            n_games = n_games,
            seed = seed,
        }),
        am.view("trickiness", "trickiness", {
            seat = "boss",
            style = style,
            prompt_set = prompt_set,
        }),
    }
    if teacher_alias ~= nil then
        views[#views + 1] = am.view("sd_teacher", "style_distance", {
            seat = "boss",
            style = style,
            prompt_set = prompt_set,
            card_b = teacher_alias,
        })
    end
    return views
end

--- Read a numeric field off a Record.values table. Returns `nil` when
--- the record is missing, errored, or the field is not a number — the
--- runner writes each field as absent on the report entry in that
--- case rather than emitting a spurious zero.
local function numeric_field(record, field)
    if type(record) ~= "table" or record.error ~= nil then
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

local function find_record(records, view_id)
    for _, record in ipairs(records) do
        if type(record) == "table" and rawget(record, "view_id") == view_id then
            return record
        end
    end
    return nil
end

--- Extract the baseline fields the audit report cares about, in the
--- order the caller expects to read them. Any missing / errored field
--- stays absent on the entry — a measurement gap is not a labelled
--- outcome.
local function extract_baseline(records, teacher_alias)
    local level_rec = find_record(records, "level")
    local tricky_rec = find_record(records, "trickiness")
    local sd_rec = find_record(records, "sd_teacher")

    local entry = {}
    local win_rate = numeric_field(level_rec, "win_rate")
    if win_rate ~= nil then
        entry.win_rate = win_rate
    end
    local ci_lower = numeric_field(level_rec, "ci_lower")
    if ci_lower ~= nil then
        entry.ci_lower = ci_lower
    end
    local ci_upper = numeric_field(level_rec, "ci_upper")
    if ci_upper ~= nil then
        entry.ci_upper = ci_upper
    end
    local tricky_val = numeric_field(tricky_rec, "value")
    if tricky_val ~= nil then
        entry.trickiness_norm = tricky_val
    end
    -- Only attach sd_teacher if the caller pinned a teacher_alias;
    -- otherwise the view was never built and the record is absent by
    -- design (rather than a measurement failure).
    if teacher_alias ~= nil then
        local sd_val = numeric_field(sd_rec, "value")
        if sd_val ~= nil then
            entry.sd_teacher = sd_val
        end
    end
    return entry
end

-- ─── SD matrix ─────────────────────────────────────────────────────

--- Evaluate `style_distance` for the ordered pair `(a, b)`. Any raise
--- from the registry (a mask mismatch, a bad prompt state) propagates
--- to the caller so the audit fails loudly rather than silently
--- publishing a matrix with a zero where a broken metric hid a real
--- distance.
local function evaluate_pair(registry, handle_a, handle_b, style, prompt_set)
    local raw = registry.evaluate("style_distance", {
        card_a = handle_a,
        card_b = handle_b,
        prompt_set = prompt_set,
        seat = "boss",
        style = style,
    })
    -- `style_distance` returns a scalar; the compose loop in
    -- gameai_metrics/init.lua does not lift it. Accept both shapes so
    -- a spec that mimics the anymetric lift form still works.
    if type(raw) == "number" then
        return raw
    end
    if type(raw) == "table" and type(raw.value) == "number" then
        return raw.value
    end
    error(
        "audit_matrix: style_distance returned "
            .. type(raw)
            .. "; expected a number or a { value = <n> } table"
    )
end

--- Fill in `sd_matrix[alias_i][alias_j]` for every unordered pair
--- exactly once (`i < j`), then mirror the value onto `[alias_j][alias_i]`.
--- The diagonal is filled with `0.0` so a downstream lookup does not
--- have to special-case the identity pair.
local function build_sd_matrix(cards, style, prompt_set)
    local registry = require_registry()
    local matrix = {}
    for _, card in ipairs(cards) do
        matrix[card.alias] = { [card.alias] = 0.0 }
    end
    for i = 1, #cards - 1 do
        for j = i + 1, #cards do
            local a, b = cards[i], cards[j]
            local sd = evaluate_pair(registry, a.handle, b.handle, style, prompt_set)
            matrix[a.alias][b.alias] = sd
            matrix[b.alias][a.alias] = sd
        end
    end
    return matrix
end

-- ─── Runner ────────────────────────────────────────────────────────

local Audit = {}
Audit.__index = Audit

--- Execute the audit. Loads every Card handle by `card_id`, fires the
--- per-Card views once per Card, then evaluates the pair-wise SD
--- matrix. Returns the report table and caches it on the Audit so
--- `:save()` and `:report()` can read the same object without a
--- second run.
---@return table report
function Audit:run()
    require_nn_card()

    -- Reload every Card first. Doing this up-front means a missing
    -- card_id fails before any metric runs, so an audit either
    -- completes for every Card or fails cleanly with the offending
    -- alias named.
    local cards = {}
    for _, entry in ipairs(self._cards) do
        local ok, handle = pcall(alc.nn.card.load_handle, entry.card_id)
        if not ok then
            error(
                string.format(
                    "audit_matrix: alc.nn.card.load_handle(%q) failed for alias %q: %s",
                    entry.card_id,
                    entry.alias,
                    tostring(handle)
                )
            )
        end
        if handle == nil then
            error(
                string.format(
                    "audit_matrix: alc.nn.card.load_handle(%q) returned nil for alias %q "
                        .. "(card_id likely removed from the Card store)",
                    entry.card_id,
                    entry.alias
                )
            )
        end
        cards[#cards + 1] = {
            alias = entry.alias,
            card_id = entry.card_id,
            step = entry.step,
            handle = handle,
        }
    end

    local views =
        build_views(self._style, self._prompt_set, self._teacher_alias, self._n_games, self._seed)

    local per_card = {}
    for _, card in ipairs(cards) do
        -- observe requires a numeric step; when the collection carried
        -- one we pass it through (the "this Card was baked at step N"
        -- label) and default to 0 otherwise so the aliases-only path
        -- still produces observable records.
        local shared_step = card.step
        if type(shared_step) ~= "number" then
            shared_step = 0
        end
        local records = am.observe(views, { card = card.handle, step = shared_step })
        local baseline = extract_baseline(records, self._teacher_alias)
        if card.step ~= nil then
            baseline.step = card.step
        end
        per_card[card.alias] = baseline
    end

    local sd_matrix = build_sd_matrix(cards, self._style, self._prompt_set)

    local meta = {
        n_games = self._n_games,
        prompt_set_size = #self._prompt_set,
        seed = self._seed,
        style = self._style,
    }
    if self._teacher_alias ~= nil then
        meta.teacher_alias = self._teacher_alias
    end
    if self._collection_path ~= nil then
        meta.collection_path = self._collection_path
    end

    local report = {
        per_card = per_card,
        sd_matrix = sd_matrix,
        meta = meta,
    }
    self._report = report
    return report
end

--- Read-only accessor for the last `:run()` report. Raises when
--- called before `:run()` — a save() without a report would write an
--- empty manifest that looked like a successful audit.
---@return table report
function Audit:report()
    if self._report == nil then
        error("audit_matrix:report: no report yet; call :run() first", 2)
    end
    return self._report
end

--- Encode the report and write it to `path`. Ensures the parent
--- directory exists first (via `gameai_metrics._fs`), so a first save
--- into a fresh workspace sub-tree does not surface as an obscure
--- `io.open` error.
---@param path string
function Audit:save(path)
    if type(path) ~= "string" or path == "" then
        error("audit_matrix:save: path must be a non-empty string", 2)
    end
    if self._report == nil then
        error("audit_matrix:save: no report yet; call :run() before :save()", 2)
    end
    fs.ensure_parent_dir(path)
    local encode = require_json_encoder()
    local ok, encoded = pcall(encode, self._report)
    if not ok then
        error("audit_matrix:save: failed to encode report: " .. tostring(encoded), 2)
    end
    local f, err = io.open(path, "w")
    if f == nil then
        error(
            string.format("audit_matrix:save: cannot open %q for writing: %s", path, tostring(err)),
            2
        )
    end
    local ok_write, write_err = pcall(function()
        f:write(encoded)
    end)
    f:close()
    if not ok_write then
        error(
            string.format("audit_matrix:save: write to %q failed: %s", path, tostring(write_err)),
            2
        )
    end
end

--- Build a new audit runner. `opts` is a table:
---
--- - `collection_path` — string, harvest manifest to read (produced
---   by `gameai_metrics.harvest_collection:save()`). One of
---   `collection_path` / `aliases` is required; passing both is a
---   loud error.
--- - `aliases` — array of alias strings. Each alias is resolved
---   through `alc.card.get_by_alias`, so an alias without a bound
---   Card raises immediately.
--- - `n_games` — integer, sample size for the `level` view (default
---   `200`). Recorded on `meta.n_games` for the report; the actual
---   `level` metric reads it out of its own registry ctx, which is
---   supplied per-fire below via the level view config.
--- - `prompt_set_size` — integer, size of the boss-state prompt set
---   used by the `trickiness` and `style_distance` views (default
---   `16`). Ignored when `opts.prompt_set` is supplied.
--- - `prompt_set` — optional array of boss states. When absent the
---   runner builds one deterministically from `seed` so an audit
---   remains reproducible without a manifest-side prompt set.
--- - `seed` — integer, seeds both the built-in prompt set and the
---   pair-wise SD evaluation (default `0`).
--- - `style` — required, one of `guardian_duel.STYLES`. Passed
---   verbatim into every boss-seat view.
--- - `teacher_alias` — optional string; when set, the runner adds an
---   `sd_teacher` view and reports `sd_teacher` per Card.
---
---@param opts table
---@return table audit
function M.new(opts)
    if type(opts) ~= "table" then
        error("audit_matrix.new: opts must be a table, got " .. type(opts), 2)
    end

    if opts.collection_path ~= nil and opts.aliases ~= nil then
        error("audit_matrix.new: pass either opts.collection_path or opts.aliases, not both", 2)
    end
    if opts.collection_path == nil and opts.aliases == nil then
        error("audit_matrix.new: opts.collection_path or opts.aliases is required", 2)
    end

    local style = boss_seat.require_style(opts.style, "audit_matrix.new")

    if opts.teacher_alias ~= nil then
        if type(opts.teacher_alias) ~= "string" or opts.teacher_alias == "" then
            error(
                "audit_matrix.new: opts.teacher_alias must be a non-empty string or nil, got "
                    .. type(opts.teacher_alias),
                2
            )
        end
    end

    local n_games = decode_int(opts.n_games, DEFAULT_N_GAMES, "n_games", true)
    local prompt_set_size =
        decode_int(opts.prompt_set_size, DEFAULT_PROMPT_SET_SIZE, "prompt_set_size", true)
    local seed = decode_int(opts.seed, DEFAULT_SEED, "seed", false)

    local prompt_set
    if opts.prompt_set ~= nil then
        if type(opts.prompt_set) ~= "table" or #opts.prompt_set == 0 then
            error("audit_matrix.new: opts.prompt_set must be a non-empty array of boss states", 2)
        end
        prompt_set = opts.prompt_set
    else
        prompt_set = build_prompt_set(seed, prompt_set_size)
    end

    local cards
    if opts.collection_path ~= nil then
        cards = resolve_from_collection(opts.collection_path)
    else
        cards = resolve_alias_list(opts.aliases)
    end
    if #cards < 2 then
        error(
            string.format(
                "audit_matrix.new: at least two Cards are needed for a pair-wise SD matrix, got %d",
                #cards
            ),
            2
        )
    end

    return setmetatable({
        _cards = cards,
        _style = style,
        _teacher_alias = opts.teacher_alias,
        _n_games = n_games,
        _prompt_set_size = #prompt_set,
        _seed = seed,
        _prompt_set = prompt_set,
        _collection_path = opts.collection_path,
        _report = nil,
    }, Audit)
end

--- Read-only accessors, mostly for specs / driver logs.
M.DEFAULT_N_GAMES = DEFAULT_N_GAMES
M.DEFAULT_PROMPT_SET_SIZE = DEFAULT_PROMPT_SET_SIZE
M.DEFAULT_SEED = DEFAULT_SEED

return M
