--- anymetric — observe one training run through several independent
--- metric views, then decide separately.
---
--- The module keeps two layers apart on purpose:
---
--- - **measurement**: a `View` binds a metric function to a fixed
---   ctx (`config`). `observe(views, shared_ctx)` fires every view once
---   per checkpoint and returns one uniform `Record` per view. Views are
---   evaluated under `pcall`, so a metric that blows up leaves an
---   `ErrorRecord` behind instead of killing the training run.
---
--- - **judgment**: a `Judgment` is `fn(records) -> Decision` where
---   `Decision = { action = "break"|"continue"|"harvest", reason = <string>,
---   meta = <table|nil> }`. Judgments read the records of the views they
---   were told to read and nothing else, so a strength gate never
---   accidentally couples itself to a personality metric that happens to
---   be observed alongside it. The one exception is opt-in and written
---   down band by band: a `staged` band may carry an explicit
---   `require = { view_id, field, min }`, and then that band — only that
---   band, only that named view, only that named field — is allowed to
---   read a second view before it harvests. A coupling the caller spelled
---   out is not the accidental one this rule exists to prevent; a view no
---   band named still goes unread. `meta` is an optional caller-defined table
---   that band / staged judgments use to carry the hit `label` plus the
---   observing step and raw values; simpler judgments (threshold,
---   never_break) leave it absent so downstream callers see the exact
---   same shape they always did.
---
--- Only `to_hook_action` knows about the trainer hook ABI, and the
--- domain's third action (`harvest`) now has somewhere to land in it:
--- the bridge takes `{ action = "continue", keep = "<reason>" }`, and
--- the trainer holds that checkpoint out of its rotation. Until it did,
--- a harvest was flattened to `"continue"` and the file the manifest
--- named could be deleted before the run ended. When a harvest decision
--- carries `meta`, the marker record appended to the run log carries
--- the same table so a collection helper can extract the label without
--- a second judgment call.
---
--- Nothing in this module is game-specific: the metric functions, ctx
--- keys and the fields a judgment reads are all supplied by the caller.
---
--- ## Usage from a trainer `on_ckpt` hook
---
--- ```lua
--- local am = require("anymetric")
--- local run_log = am.run_log.new()
--- -- `level` / `style_distance` are `fn(ctx) -> table | scalar` supplied
--- -- by the caller; this module never learns what they measure.
--- local views = {
---     am.view("level", level, { opponents = { "random" }, n_games = 50,
---                               required = { "opponents" } }),
---     am.view("sd_teacher", style_distance, { card_b = TEACHER, prompt_set = STATES }),
--- }
--- local judgment = am.judgment.threshold({
---     view_id = "level", field = "ci_lower", op = ">=", value = 0.55,
--- })
---
--- on_ckpt = function(info)
---     local records = am.observe(views, { card = info.handle, step = info.step })
---     run_log:append(records)
---     alc.log.info(am.log_line(records))
---     return am.to_hook_action(judgment(records), run_log)
--- end
--- ```

local M = {}

---@type AlcMeta
M.meta = {
    name = "anymetric",
    version = "0.1.0",
    description = "AnyMetric domain: metric views, observation records, run log, judgments, hook adapter",
    category = "eval",
}

--- Keys `observe` supplies per fire. They always come from `shared_ctx`,
--- never from a view `config`, because their value changes on every
--- checkpoint.
local RESERVED_KEYS = { card = true, step = true }

local function reserved_key_list()
    local names = {}
    for key in pairs(RESERVED_KEYS) do
        names[#names + 1] = key
    end
    table.sort(names)
    return table.concat(names, ", ")
end

-- ---------------------------------------------------------------------
-- View
-- ---------------------------------------------------------------------

--- Bind a metric to a fixed ctx.
---
--- `metric` is the function that does the measuring, `fn(ctx) -> table |
--- scalar`. It is held directly rather than looked up by name: a view is
--- built in the same Lua chunk that owns the metric, so a name would be
--- turned into a string and back into the same function with no boundary
--- in between.
---
--- `view_id` is a first-class field rather than a derived label: the same
--- metric is routinely observed through several views that differ only in
--- their `config` (two opponent pools, two teachers), and every record,
--- log line and judgment addresses a view by that id. It is also what
--- names the view in an error, which is why the metric itself needs no
--- name of its own.
---
--- `config` is the per-view half of the ctx the metric receives. The
--- reserved keys (`card`, `step`) must not appear in it — `observe`
--- supplies those per fire.
---
--- `config.required` is an optional array of ctx key names the view
--- declares it cannot run without. A metric is a bare function and has no
--- required-key mechanism of its own, so the check lives here: a missing
--- key raises immediately at view-construction time, naming both the view
--- and the key, instead of surfacing as an opaque metric error 60 steps
--- into a training run. `required` itself is stripped from the stored
--- config and never reaches the metric.
---
---@param view_id string caller label, unique within one `observe` call
---@param metric function `fn(ctx) -> table | scalar`
---@param config table|nil fixed ctx for this view (`required` optional)
---@return table view `{ view_id, metric, config }`
function M.view(view_id, metric, config)
    if type(view_id) ~= "string" or view_id == "" then
        error("anymetric.view: view_id must be a non-empty string", 2)
    end
    if type(metric) ~= "function" then
        error(
            "anymetric.view: metric must be a function (view '"
                .. view_id
                .. "'), got "
                .. type(metric),
            2
        )
    end
    if config ~= nil and type(config) ~= "table" then
        error(
            "anymetric.view: config must be a table or nil (view '"
                .. view_id
                .. "'), got "
                .. type(config),
            2
        )
    end
    config = config or {}

    local required = config.required
    if required ~= nil and type(required) ~= "table" then
        error(
            "anymetric.view: config.required must be an array of key names (view '"
                .. view_id
                .. "'), got "
                .. type(required),
            2
        )
    end

    local bound = {}
    for key, value in pairs(config) do
        if key ~= "required" then
            if RESERVED_KEYS[key] then
                error(
                    "anymetric.view: config must not set the reserved key '"
                        .. key
                        .. "' (view '"
                        .. view_id
                        .. "'); reserved keys ("
                        .. reserved_key_list()
                        .. ") are supplied per fire by observe",
                    2
                )
            end
            bound[key] = value
        end
    end

    for _, key in ipairs(required or {}) do
        if type(key) ~= "string" or key == "" then
            error(
                "anymetric.view: config.required entries must be non-empty strings (view '"
                    .. view_id
                    .. "')",
                2
            )
        end
        if RESERVED_KEYS[key] then
            error(
                "anymetric.view: reserved key '"
                    .. key
                    .. "' must not be declared required (view '"
                    .. view_id
                    .. "'); observe always supplies it",
                2
            )
        end
        if bound[key] == nil then
            error(
                "anymetric.view: view '"
                    .. view_id
                    .. "' requires config key '"
                    .. key
                    .. "' but it is missing",
                2
            )
        end
    end

    return { view_id = view_id, metric = metric, config = bound }
end

-- ---------------------------------------------------------------------
-- observe / Record
-- ---------------------------------------------------------------------

--- Lift whatever the metric returned into the uniform `values` table.
--- A table passes through as-is (numeric fields assumed); a scalar is
--- wrapped as `{ value = <x> }` so every record has the same shape.
---
--- The view id, not a metric name, is what the message blames: a metric
--- is an anonymous function here, and the id is the only handle the
--- caller can act on.
local function lift_values(view_id, raw)
    local kind = type(raw)
    if kind == "table" then
        return raw
    end
    if kind == "number" or kind == "string" or kind == "boolean" then
        return { value = raw }
    end
    error("view '" .. view_id .. "' returned " .. kind .. "; expected a table or a scalar", 0)
end

local function evaluate_view(view, ctx)
    return lift_values(view.view_id, view.metric(ctx))
end

--- Merge a view config with the per-fire shared ctx. Shared entries win
--- on collision, so the reserved keys always carry the current
--- checkpoint's value; every other key can only come from the config.
local function merge_ctx(config, shared_ctx)
    local ctx = {}
    for key, value in pairs(config) do
        ctx[key] = value
    end
    for key, value in pairs(shared_ctx) do
        ctx[key] = value
    end
    return ctx
end

--- Fire every view once and return one record per view, in view order.
---
--- Records are uniform: `{ step, view_id, values }` on success,
--- `{ step, view_id, error = <message> }` when the metric raised. Each
--- view is evaluated under its own `pcall`, so one broken metric neither
--- hides the other views nor propagates into the hook — a hook error
--- reaches the trainer as `TrainError::Hook`, which skips the final save
--- and throws the terminal checkpoint away. Measurement failure must not
--- cost the training result.
---
--- Malformed input (duplicate view ids, a non-table view, a missing
--- `step`) is a wiring bug rather than a measurement failure and is
--- raised loudly before any metric runs.
---
---@param views table array of views (see `M.view`)
---@param shared_ctx table per-fire ctx, `{ card = <handle>, step = <number> }`
---@return table records array of Record / ErrorRecord
function M.observe(views, shared_ctx)
    if type(views) ~= "table" then
        error("anymetric.observe: views must be an array of views, got " .. type(views), 2)
    end
    if shared_ctx ~= nil and type(shared_ctx) ~= "table" then
        error("anymetric.observe: shared_ctx must be a table or nil, got " .. type(shared_ctx), 2)
    end
    shared_ctx = shared_ctx or {}
    if type(shared_ctx.step) ~= "number" then
        error(
            "anymetric.observe: shared_ctx.step must be a number (the per-fire checkpoint step)",
            2
        )
    end

    local seen = {}
    for index, view in ipairs(views) do
        if type(view) ~= "table" then
            error(
                "anymetric.observe: views[" .. index .. "] must be a table, got " .. type(view),
                2
            )
        end
        if type(view.view_id) ~= "string" or view.view_id == "" then
            error("anymetric.observe: views[" .. index .. "] has no view_id", 2)
        end
        if type(view.metric) ~= "function" then
            error(
                "anymetric.observe: view '"
                    .. view.view_id
                    .. "' has no metric function, got "
                    .. type(view.metric),
                2
            )
        end
        if seen[view.view_id] then
            error(
                "anymetric.observe: duplicate view_id '"
                    .. view.view_id
                    .. "' (views["
                    .. seen[view.view_id]
                    .. "] and views["
                    .. index
                    .. "]); view ids address records and must be unique",
                2
            )
        end
        seen[view.view_id] = index
    end

    local step = shared_ctx.step
    local records = {}
    for _, view in ipairs(views) do
        local ctx = merge_ctx(view.config or {}, shared_ctx)
        local ok, result = pcall(evaluate_view, view, ctx)
        if ok then
            records[#records + 1] = {
                step = step,
                view_id = view.view_id,
                values = result,
            }
        else
            records[#records + 1] = {
                step = step,
                view_id = view.view_id,
                error = tostring(result),
            }
        end
    end
    return records
end

-- ---------------------------------------------------------------------
-- run_log
-- ---------------------------------------------------------------------

local RunLog = {}
RunLog.__index = RunLog

--- Append a batch of records. Append-only: the log never rewrites or
--- drops what it already holds, which is what makes it usable as the
--- source of a hand-written observation note.
---@param records table array of records
---@return table self
function RunLog:append(records)
    if type(records) ~= "table" then
        error(
            "anymetric.run_log:append: records must be an array of records, got " .. type(records),
            2
        )
    end
    for index, record in ipairs(records) do
        if type(record) ~= "table" then
            error("anymetric.run_log:append: records[" .. index .. "] must be a table", 2)
        end
        self._records[#self._records + 1] = record
    end
    return self
end

--- Every record appended so far, in append order. Returns a fresh array
--- so a caller cannot shorten the log by mutating what it reads.
---@return table records
function RunLog:all()
    local out = {}
    for index, record in ipairs(self._records) do
        out[index] = record
    end
    return out
end

M.run_log = {}

--- Create an empty append-only run log.
---@return table run_log
function M.run_log.new()
    return setmetatable({ _records = {} }, RunLog)
end

-- ---------------------------------------------------------------------
-- Judgment / Decision
-- ---------------------------------------------------------------------

local COMPARATORS = {
    [">="] = function(a, b)
        return a >= b
    end,
    ["<="] = function(a, b)
        return a <= b
    end,
    [">"] = function(a, b)
        return a > b
    end,
    ["<"] = function(a, b)
        return a < b
    end,
}

local function comparator_list()
    local ops = {}
    for op in pairs(COMPARATORS) do
        ops[#ops + 1] = op
    end
    table.sort(ops)
    return table.concat(ops, " ")
end

local function decision(action, reason, meta)
    return { action = action, reason = reason, meta = meta }
end

M.judgment = {}

--- A judgment that breaks the run once one numeric field of one view
--- crosses a threshold.
---
--- The field name is caller-supplied on purpose: the domain has no
--- opinion about what "good" means, so no metric vocabulary leaks in
--- here. A gate on win rate is
--- `threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.55 })`.
---
--- The judgment reads the records of `view_id` only. Records of other
--- views observed in the same fire are never touched, which is what keeps
--- a strength gate from silently reacting to a personality metric.
--- When the target view produced no record, produced an `ErrorRecord`, or
--- has no numeric field of that name, the judgment continues and says so
--- in `reason` — a measurement gap is not evidence that the threshold was
--- met.
---
---@param opts table `{ view_id, field, op, value }`, op ∈ `>= <= > <`
---@return function judgment `fn(records) -> Decision`
function M.judgment.threshold(opts)
    if type(opts) ~= "table" then
        error("anymetric.judgment.threshold: opts must be a table, got " .. type(opts), 2)
    end
    local view_id, field, op, value = opts.view_id, opts.field, opts.op, opts.value
    if type(view_id) ~= "string" or view_id == "" then
        error("anymetric.judgment.threshold: view_id must be a non-empty string", 2)
    end
    if type(field) ~= "string" or field == "" then
        error("anymetric.judgment.threshold: field must be a non-empty string", 2)
    end
    if type(value) ~= "number" then
        error("anymetric.judgment.threshold: value must be a number, got " .. type(value), 2)
    end
    local compare = type(op) == "string" and COMPARATORS[op] or nil
    if compare == nil then
        error(
            "anymetric.judgment.threshold: unknown op '"
                .. tostring(op)
                .. "'; expected one of "
                .. comparator_list(),
            2
        )
    end

    return function(records)
        if type(records) ~= "table" then
            error(
                "anymetric.judgment.threshold: records must be an array, got " .. type(records),
                2
            )
        end
        local target
        for _, record in ipairs(records) do
            if type(record) == "table" and rawget(record, "view_id") == view_id then
                target = record
            end
        end
        if target == nil then
            return decision("continue", string.format("view '%s' produced no record", view_id))
        end
        if target.error ~= nil then
            return decision(
                "continue",
                string.format("view '%s' errored: %s", view_id, tostring(target.error))
            )
        end
        local values = target.values
        local actual = type(values) == "table" and values[field] or nil
        if type(actual) ~= "number" then
            return decision(
                "continue",
                string.format("view '%s' has no numeric field '%s'", view_id, field)
            )
        end
        if compare(actual, value) then
            return decision(
                "break",
                string.format("%s.%s = %.4f %s %.4f", view_id, field, actual, op, value)
            )
        end
        return decision(
            "continue",
            string.format("%s.%s = %.4f, want %s %.4f", view_id, field, actual, op, value)
        )
    end
end

--- Find the record for `view_id` in `records` and, if the field is a
--- usable number, return `(record, value, nil)`. Otherwise return
--- `(nil, nil, continue_decision)` where the continue decision names the
--- reason (no record / errored / missing field / non-numeric field), so
--- the caller can return it verbatim. Meta is left absent on the miss
--- path — a measurement gap is not a labelled outcome.
local function read_numeric_field(records, view_id, field)
    if type(records) ~= "table" then
        error("anymetric.judgment: records must be an array, got " .. type(records), 3)
    end
    local target
    for _, record in ipairs(records) do
        if type(record) == "table" and rawget(record, "view_id") == view_id then
            target = record
        end
    end
    if target == nil then
        return nil,
            nil,
            decision("continue", string.format("view '%s' produced no record", view_id))
    end
    if target.error ~= nil then
        return nil,
            nil,
            decision(
                "continue",
                string.format("view '%s' errored: %s", view_id, tostring(target.error))
            )
    end
    local values = target.values
    local actual = type(values) == "table" and values[field] or nil
    if type(actual) ~= "number" then
        return nil,
            nil,
            decision(
                "continue",
                string.format("view '%s' has no numeric field '%s'", view_id, field)
            )
    end
    return target, actual, nil
end

--- Validate a band's optional `require` clause. Returns
--- `(require_table|nil, error_message|nil)` rather than raising, so the
--- caller keeps one error level for every band problem.
---
--- `allow_require` is false on the single-view paths. Refusing the
--- clause there is deliberate: a judgment that reads one view cannot
--- honour a second-view condition, and silently dropping it would read
--- as "the requirement is in force" to anyone looking at the caller.
local function check_require(raw, allow_require)
    if raw == nil then
        return nil, nil
    end
    if not allow_require then
        return nil,
            "band.require is not supported here; this judgment reads a single view, so a "
                .. "second-view condition would be silently ignored — use judgment.staged with a "
                .. "require band instead"
    end
    if type(raw) ~= "table" then
        return nil, "band.require must be a table { view_id, field, min }, got " .. type(raw)
    end
    local view_id, field, min = raw.view_id, raw.field, raw.min
    if type(view_id) ~= "string" or view_id == "" then
        return nil, "band.require.view_id must be a non-empty string"
    end
    if type(field) ~= "string" or field == "" then
        return nil,
            "band.require.field must be a non-empty string (require view '" .. view_id .. "')"
    end
    -- Infinities and NaN are refused alongside non-numbers: a NaN floor
    -- compares false against every measurement, which reads as "the band
    -- never qualifies" instead of as the wiring bug it is.
    if type(min) ~= "number" or min ~= min or min == math.huge or min == -math.huge then
        return nil, "band.require.min must be a finite number, got " .. tostring(min)
    end
    return { view_id = view_id, field = field, min = min }, nil
end

local function normalise_band(band, where, allow_require)
    if type(band) ~= "table" then
        error(where .. ": band must be a table, got " .. type(band), 3)
    end
    local lo, hi, label = band.lo, band.hi, band.label
    if type(lo) ~= "number" then
        error(where .. ": band.lo must be a number, got " .. type(lo), 3)
    end
    if type(hi) ~= "number" then
        error(where .. ": band.hi must be a number, got " .. type(hi), 3)
    end
    if lo > hi then
        error(string.format("%s: band.lo (%.4f) must be <= band.hi (%.4f)", where, lo, hi), 3)
    end
    if label ~= nil and type(label) ~= "string" then
        error(where .. ": band.label must be a string or nil, got " .. type(label), 3)
    end
    local required, require_error = check_require(band.require, allow_require)
    if require_error ~= nil then
        error(where .. ": " .. require_error, 3)
    end
    return { lo = lo, hi = hi, label = label, require = required }
end

--- Name a band hit for a `reason` string, before it is known whether the
--- hit actually harvests.
local function band_hit_prefix(view_id, field, actual, band)
    if band.label ~= nil then
        return string.format("band hit (%s) at %s.%s = %.4f", band.label, view_id, field, actual)
    end
    return string.format(
        "band hit [%.4f, %.4f] at %s.%s = %.4f",
        band.lo,
        band.hi,
        view_id,
        field,
        actual
    )
end

--- Build a harvest decision for a hit band, carrying the label plus the
--- observing step and raw values in `meta`.
---
--- `require_hit` is the `{view_id, field, value, min}` of a satisfied
--- `require` clause, or nil for a band that carried none. When present
--- it is copied into `meta.require` as `{view_id, field, value}`, so the
--- provenance of the second condition survives into the marker record
--- and from there into a harvest manifest.
local function harvest_hit(view_id, field, record, actual, band, require_hit)
    local reason
    if band.label ~= nil then
        reason = string.format(
            "%s.%s = %.4f in [%.4f, %.4f] (%s)",
            view_id,
            field,
            actual,
            band.lo,
            band.hi,
            band.label
        )
    else
        reason =
            string.format("%s.%s = %.4f in [%.4f, %.4f]", view_id, field, actual, band.lo, band.hi)
    end
    local meta = {
        label = band.label,
        step = record.step,
        values = record.values,
    }
    if require_hit ~= nil then
        reason = string.format(
            "%s; required %s.%s = %.4f >= min=%.4f",
            reason,
            require_hit.view_id,
            require_hit.field,
            require_hit.value,
            require_hit.min
        )
        meta.require = {
            view_id = require_hit.view_id,
            field = require_hit.field,
            value = require_hit.value,
        }
    end
    return decision("harvest", reason, meta)
end

--- A judgment that harvests once one numeric field of one view falls
--- inside a single closed interval `[lo, hi]`, continues while the
--- value is below `lo`, and breaks once it rises above `hi`.
---
--- Both ends are inclusive so the boundary picks the band above rather
--- than falling into a silent gap. `label` is optional; when set it is
--- copied into the harvest decision's `meta.label` (and its `reason`),
--- so a collection helper can address the hit without a second judgment
--- call.
---
--- Like `threshold`, the judgment reads the records of `view_id` only.
--- A missing record, an ErrorRecord, or a non-numeric field all resolve
--- to `continue` and name the reason.
---
--- A `require` clause (the second-view condition `staged` bands accept)
--- is refused here rather than ignored: this judgment reads one view, so
--- honouring it is impossible and dropping it silently would leave a
--- caller believing a condition is in force that nothing checks.
---
---@param opts table `{ view_id, field, lo, hi, label? }`
---@return function judgment `fn(records) -> Decision`
function M.judgment.band(opts)
    if type(opts) ~= "table" then
        error("anymetric.judgment.band: opts must be a table, got " .. type(opts), 2)
    end
    local view_id, field = opts.view_id, opts.field
    if type(view_id) ~= "string" or view_id == "" then
        error("anymetric.judgment.band: view_id must be a non-empty string", 2)
    end
    if type(field) ~= "string" or field == "" then
        error("anymetric.judgment.band: field must be a non-empty string", 2)
    end
    local band = normalise_band(
        { lo = opts.lo, hi = opts.hi, label = opts.label, require = opts.require },
        "anymetric.judgment.band",
        false
    )

    return function(records)
        local record, actual, miss = read_numeric_field(records, view_id, field)
        if miss ~= nil then
            return miss
        end
        if actual < band.lo then
            local suffix = band.label and (" (" .. band.label .. ")") or ""
            return decision(
                "continue",
                string.format(
                    "%s.%s = %.4f below lo=%.4f%s",
                    view_id,
                    field,
                    actual,
                    band.lo,
                    suffix
                )
            )
        end
        if actual > band.hi then
            local suffix = band.label and (" (" .. band.label .. ")") or ""
            return decision(
                "break",
                string.format(
                    "%s.%s = %.4f above hi=%.4f%s",
                    view_id,
                    field,
                    actual,
                    band.hi,
                    suffix
                )
            )
        end
        return harvest_hit(view_id, field, record, actual, band)
    end
end

--- A staged judgment: several disjoint bands in ascending `lo` order,
--- each with an optional `label`. When the observed field falls into
--- one of the bands the judgment harvests and copies that band's label
--- into `meta.label`; below the lowest band it continues; above the
--- highest band's `hi` it breaks. Between two bands (no band contains
--- the value) it also continues.
---
--- The judgment is stateless — it holds no history of which labels
--- have been harvested. Enforcing "one bake per label" is the caller's
--- job (the collection helper handles first-writer-wins). The one
--- structural rule enforced here is that bands are disjoint
--- (`bands[i].hi < bands[i+1].lo`); overlapping bands would make the
--- hit label non-deterministic, which is a caller wiring bug and is
--- raised loudly.
---
--- A one-band staged judgment is allowed and is equivalent to a single
--- `band(...)`; an empty bands list is a wiring bug and raises.
---
--- ## Per-band `require` (opt-in second view)
---
--- A band may carry `require = { view_id, field, min }`. It is the one
--- documented exception to "a judgment reads the view it was bound to
--- and nothing else": when that band is hit, and only then, the judgment
--- also reads `require.view_id`'s record and harvests only if its
--- numeric `field` is `>= min`. Below the floor it continues and names
--- both numbers. A second view that produced no record, produced an
--- ErrorRecord, or holds no numeric field of that name also continues —
--- a measurement gap is not evidence that the requirement was met, and
--- treating it as a pass is how a floor quietly stops being a floor.
---
--- The clause is per band: bands without it behave exactly as before,
--- and a view no band named is never read. On a satisfied requirement
--- the harvest carries `meta.require = { view_id, field, value }` so the
--- second condition is visible in the marker record rather than only in
--- the caller's configuration.
---
---@param opts table `{ view_id, field, bands = { { lo, hi, label?, require? }, ... } }`
---@return function judgment `fn(records) -> Decision`
function M.judgment.staged(opts)
    if type(opts) ~= "table" then
        error("anymetric.judgment.staged: opts must be a table, got " .. type(opts), 2)
    end
    local view_id, field = opts.view_id, opts.field
    if type(view_id) ~= "string" or view_id == "" then
        error("anymetric.judgment.staged: view_id must be a non-empty string", 2)
    end
    if type(field) ~= "string" or field == "" then
        error("anymetric.judgment.staged: field must be a non-empty string", 2)
    end
    if type(opts.bands) ~= "table" then
        error("anymetric.judgment.staged: bands must be an array, got " .. type(opts.bands), 2)
    end
    if #opts.bands == 0 then
        error("anymetric.judgment.staged: bands must not be empty", 2)
    end

    local bands = {}
    for index, raw in ipairs(opts.bands) do
        bands[index] =
            normalise_band(raw, "anymetric.judgment.staged: bands[" .. index .. "]", true)
    end
    for index = 2, #bands do
        local prev, curr = bands[index - 1], bands[index]
        if not (prev.hi < curr.lo) then
            error(
                string.format(
                    "anymetric.judgment.staged: bands must be in ascending order and disjoint "
                        .. "(bands[%d].hi=%.4f must be < bands[%d].lo=%.4f)",
                    index - 1,
                    prev.hi,
                    index,
                    curr.lo
                ),
                2
            )
        end
    end

    local top = bands[#bands]

    return function(records)
        local record, actual, miss = read_numeric_field(records, view_id, field)
        if miss ~= nil then
            return miss
        end
        if actual > top.hi then
            local suffix = top.label and (" (" .. top.label .. ")") or ""
            return decision(
                "break",
                string.format(
                    "%s.%s = %.4f above top hi=%.4f%s",
                    view_id,
                    field,
                    actual,
                    top.hi,
                    suffix
                )
            )
        end
        for _, band in ipairs(bands) do
            if actual >= band.lo and actual <= band.hi then
                local required = band.require
                if required == nil then
                    return harvest_hit(view_id, field, record, actual, band)
                end
                -- The band named a second view, so this hit — and no
                -- other band's — reads it.
                local _, secondary, secondary_miss =
                    read_numeric_field(records, required.view_id, required.field)
                if secondary_miss ~= nil then
                    return decision(
                        "continue",
                        string.format(
                            "%s but the required %s.%s could not be read: %s",
                            band_hit_prefix(view_id, field, actual, band),
                            required.view_id,
                            required.field,
                            secondary_miss.reason
                        )
                    )
                end
                if secondary < required.min then
                    return decision(
                        "continue",
                        string.format(
                            "%s but %s.%s = %.4f below required min=%.4f",
                            band_hit_prefix(view_id, field, actual, band),
                            required.view_id,
                            required.field,
                            secondary,
                            required.min
                        )
                    )
                end
                return harvest_hit(view_id, field, record, actual, band, {
                    view_id = required.view_id,
                    field = required.field,
                    value = secondary,
                    min = required.min,
                })
            end
        end
        return decision(
            "continue",
            string.format("%s.%s = %.4f between bands (no hit)", view_id, field, actual)
        )
    end
end

--- A judgment that never stops the run. Used when the run is observed
--- for its record trail only (gate disabled).
---@return function judgment `fn(records) -> Decision`
function M.judgment.never_break()
    return function()
        return decision("continue", "gate disabled (never_break)")
    end
end

-- ---------------------------------------------------------------------
-- Hook adapter
-- ---------------------------------------------------------------------

--- Project a `Decision` onto the trainer hook ABI.
---
--- `break` and `continue` are the ABI's own words and pass through as
--- strings. `harvest` returns the table form
--- `{ action = "continue", keep = "<label or reason>" }`: the run goes
--- on and the trainer holds that checkpoint out of its rotation.
---
--- The keep is the point. A harvested checkpoint is the one the
--- collection manifest names by `ckpt_path`, and before the ABI could
--- say "hold this one" that file was still in the rotating window —
--- `ckpt_keep` newer checkpoints later it was gone, and the manifest
--- pointed at nothing. Holding it costs the window nothing (pinned
--- files are not counted against `ckpt_keep`).
---
--- `keep` carries `meta.label` when the decision has one, because that
--- is the name the manifest files the entry under; a decision built by
--- hand falls back to its `reason`. Either way the trainer treats it as
--- an opaque note and hands it back on the run's candidate list.
---
--- When the harvest decision carries a `meta` table, that table is
--- copied onto the marker record so a collection helper can extract the
--- band label / step / values without a second judgment call. A harvest
--- decision without `meta` (e.g. a caller who built one by hand)
--- appends the same `{ harvest, reason }` shape the previous iteration
--- used, so downstream code that never read `meta` stays byte-identical.
---
---@param dec table Decision `{ action, reason, meta? }`
---@param run_log table|nil run log, required for `harvest`
---@return string|table action `"break"` / `"continue"`, or the keep table
function M.to_hook_action(dec, run_log)
    if type(dec) ~= "table" then
        error("anymetric.to_hook_action: decision must be a table, got " .. type(dec), 2)
    end
    local action = dec.action
    if action == "break" or action == "continue" then
        return action
    end
    if action == "harvest" then
        if type(run_log) ~= "table" or type(run_log.append) ~= "function" then
            error(
                "anymetric.to_hook_action: a harvest decision needs a run_log to append the marker to",
                2
            )
        end
        local marker = { harvest = true, reason = dec.reason }
        if dec.meta ~= nil then
            if type(dec.meta) ~= "table" then
                error(
                    "anymetric.to_hook_action: decision.meta must be a table or nil, got "
                        .. type(dec.meta),
                    2
                )
            end
            for key, value in pairs(dec.meta) do
                if marker[key] == nil then
                    marker[key] = value
                end
            end
        end
        run_log:append({ marker })
        -- The label is what the manifest files this entry under, so it
        -- is the note worth carrying onto the trainer's candidate list;
        -- a hand-built decision without one falls back to its reason.
        local label = dec.meta and dec.meta.label
        local keep = (type(label) == "string" and label ~= "") and label or dec.reason
        if type(keep) ~= "string" or keep == "" then
            keep = true
        end
        return { action = "continue", keep = keep }
    end
    error(
        "anymetric.to_hook_action: unknown decision action '"
            .. tostring(action)
            .. "'; expected break, continue or harvest",
        2
    )
end

-- ---------------------------------------------------------------------
-- Logging
-- ---------------------------------------------------------------------

local function format_scalar(value)
    if type(value) == "number" then
        return string.format("%.4f", value)
    end
    return tostring(value)
end

local function format_values(values)
    if type(values) ~= "table" then
        return format_scalar(values)
    end
    local keys = {}
    for key in pairs(values) do
        keys[#keys + 1] = key
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    if #keys == 0 then
        return "(empty)"
    end
    local parts = {}
    for _, key in ipairs(keys) do
        parts[#parts + 1] = tostring(key) .. "=" .. format_scalar(values[key])
    end
    return table.concat(parts, " ")
end

--- Render one fire's records as a single line, with the view fields in a
--- stable (sorted) order so two fires are diffable.
---
--- The line is returned, not printed: which sink it goes to (`alc.log`,
--- a file, an observation note) is the caller's decision.
---
---@param records table array of records
---@return string line
function M.log_line(records)
    if type(records) ~= "table" then
        error("anymetric.log_line: records must be an array, got " .. type(records), 2)
    end
    if #records == 0 then
        return "[anymetric] (no records)"
    end
    local parts = {}
    for _, record in ipairs(records) do
        if record.error ~= nil then
            parts[#parts + 1] =
                string.format("%s: ERROR %s", tostring(record.view_id), tostring(record.error))
        else
            parts[#parts + 1] =
                string.format("%s: %s", tostring(record.view_id), format_values(record.values))
        end
    end
    return string.format(
        "[anymetric] step=%s | %s",
        tostring(records[1].step),
        table.concat(parts, " | ")
    )
end

return M
