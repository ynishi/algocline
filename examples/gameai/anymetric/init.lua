--- anymetric — observe one training run through several independent
--- metric views, then decide separately.
---
--- The module keeps two layers apart on purpose:
---
--- - **measurement**: a `View` binds a registered metric name to a fixed
---   ctx (`config`). `observe(views, shared_ctx)` fires every view once
---   per checkpoint and returns one uniform `Record` per view. Views are
---   evaluated under `pcall`, so a metric that blows up leaves an
---   `ErrorRecord` behind instead of killing the training run.
---
--- - **judgment**: a `Judgment` is `fn(records) -> Decision` where
---   `Decision = { action = "break"|"continue"|"harvest", reason = <string> }`.
---   Judgments read the records of the views they were told to read and
---   nothing else, so a strength gate never accidentally couples itself
---   to a personality metric that happens to be observed alongside it.
---
--- Only `to_hook_action` knows about the trainer hook ABI (the bridge
--- accepts `"break"` / `"continue"` / `nil` and nothing else), so the
--- domain is free to carry a third action (`harvest`) that the ABI has
--- no room for.
---
--- Nothing in this module is game-specific: metric names, ctx keys and
--- the fields a judgment reads are all supplied by the caller.
---
--- ## Usage from a trainer `on_ckpt` hook
---
--- ```lua
--- local am = require("anymetric")
--- local run_log = am.run_log.new()
--- local views = {
---     am.view("level", "level", { opponents = { "random" }, n_games = 50,
---                                 required = { "opponents" } }),
---     am.view("sd_teacher", "style_distance", { card_b = TEACHER, prompt_set = STATES }),
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

--- Bind a registered metric to a fixed ctx.
---
--- `view_id` is a first-class field rather than a derived label: the same
--- metric is routinely observed through several views that differ only in
--- their `config` (two opponent pools, two teachers), and every record,
--- log line and judgment addresses a view by that id.
---
--- `config` is the per-view half of the ctx the metric receives. The
--- reserved keys (`card`, `step`) must not appear in it — `observe`
--- supplies those per fire.
---
--- `config.required` is an optional array of ctx key names the view
--- declares it cannot run without. The metric registry has no
--- required-key mechanism of its own, so the check lives here: a missing
--- key raises immediately at view-construction time, naming both the view
--- and the key, instead of surfacing as an opaque metric error 60 steps
--- into a training run. `required` itself is stripped from the stored
--- config and never reaches the metric.
---
---@param view_id string caller label, unique within one `observe` call
---@param metric_name string name registered in `alc.nn.metric.registry`
---@param config table|nil fixed ctx for this view (`required` optional)
---@return table view `{ view_id, metric, config }`
function M.view(view_id, metric_name, config)
    if type(view_id) ~= "string" or view_id == "" then
        error("anymetric.view: view_id must be a non-empty string", 2)
    end
    if type(metric_name) ~= "string" or metric_name == "" then
        error(
            "anymetric.view: metric_name must be a non-empty string (view '" .. view_id .. "')",
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

    return { view_id = view_id, metric = metric_name, config = bound }
end

-- ---------------------------------------------------------------------
-- observe / Record
-- ---------------------------------------------------------------------

local function registry()
    local reg = alc and alc.nn and alc.nn.metric and alc.nn.metric.registry
    if type(reg) ~= "table" or type(reg.evaluate) ~= "function" then
        error(
            "alc.nn.metric.registry is not available on this VM "
                .. "(build without the nn feature? require the metric pkg first)",
            0
        )
    end
    return reg
end

--- Lift whatever the metric returned into the uniform `values` table.
--- A table passes through as-is (numeric fields assumed); a scalar is
--- wrapped as `{ value = <x> }` so every record has the same shape.
local function lift_values(metric_name, raw)
    local kind = type(raw)
    if kind == "table" then
        return raw
    end
    if kind == "number" or kind == "string" or kind == "boolean" then
        return { value = raw }
    end
    error("metric '" .. metric_name .. "' returned " .. kind .. "; expected a table or a scalar", 0)
end

local function evaluate_view(view, ctx)
    return lift_values(view.metric, registry().evaluate(view.metric, ctx))
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
--- Records are uniform: `{ step, view_id, metric, values }` on success,
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
        if type(view.metric) ~= "string" or view.metric == "" then
            error("anymetric.observe: view '" .. view.view_id .. "' has no metric name", 2)
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
                metric = view.metric,
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

local function decision(action, reason)
    return { action = action, reason = reason }
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
--- The bridge accepts `"break"` / `"continue"` / `nil` and refuses
--- anything else, so the third domain action lands here: `harvest` keeps
--- the run going and writes a marker record into the run log, which is
--- where the next iteration's tiered-checkpoint collection will read it
--- from. Keeping the projection in one function is what lets the domain
--- grow actions the ABI cannot express.
---
---@param dec table Decision `{ action, reason }`
---@param run_log table|nil run log, required for `harvest`
---@return string action `"break"` or `"continue"`
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
        run_log:append({ { harvest = true, reason = dec.reason } })
        return "continue"
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
