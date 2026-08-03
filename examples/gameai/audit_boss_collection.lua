-- Audit a boss-harvest Card collection through `gameai_metrics.audit_matrix`.
--
-- Self-contained script for `alc_run` (`code_file` form). This is a
-- thin driver over `audit_matrix.new(opts):run() → :save(output)`; the
-- runner does all of the work, the driver only decodes the caller ctx,
-- fills the defaults from the module's own `DEFAULT_*` constants, calls
-- the runner and prints a one-line human summary before returning the
-- report table. No `alc.llm` call happens anywhere on the path, so the
-- run never pauses for a host response.
--
--   alc_run(
--     code_file = "<repo>/examples/gameai/audit_boss_collection.lua",
--     ctx = {
--       -- required
--       collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
--       output          = "workspace/gameai-harvest/audit_run2.json",
--       -- optional (all fall back to audit_matrix.DEFAULT_*)
--       n_games         = 200,
--       prompt_set_size = 16,
--       seed            = 20260731,
--       style           = "guardian",
--       teacher_alias   = "guardian_duel_npc",
--       temperature     = 1.0,
--     },
--   )
--
-- ## Required ctx fields (loud on absence)
--
-- - `collection_path` — harvest manifest written by
--   `gameai_metrics.harvest_collection:save()`; the driver refuses to
--   guess a path because a bad guess would either silently pick up an
--   unrelated file or trip an obscure `io.open` error further down.
-- - `output` — where the audit JSON is written. The parent directory
--   is created for free via `gameai_metrics._fs.ensure_parent_dir`, so
--   a fresh `workspace/` sub-tree does not need pre-mkdir.
--
-- ## Optional ctx fields (default from audit_matrix)
--
-- - `n_games` (default `audit_matrix.DEFAULT_N_GAMES` = 200) — sample
--   size for the per-Card `level` view; the higher the number the
--   tighter the Wilson interval each Card lands on.
-- - `prompt_set_size` (default `audit_matrix.DEFAULT_PROMPT_SET_SIZE`
--   = 16) — how many boss states the `trickiness` and pair-wise
--   `style_distance` views walk. A hand-supplied `prompt_set` is not
--   exposed on the driver; the runner builds one deterministically
--   from `seed` and that is the reproducible path.
-- - `seed` (default `audit_matrix.DEFAULT_SEED`) — seeds both the
--   built-in prompt set and the pair-wise SD evaluation.
-- - `style` (default `"guardian"`) — passed verbatim into every
--   boss-seat view. Must be one of `guardian_duel.STYLES`.
-- - `teacher_alias` (default `nil`) — when set the runner adds an
--   `sd_teacher` view and reports `sd_teacher` per Card. Pin one to
--   see how far each baked Card sits from the teacher policy.
-- - `temperature` (default `nil` = greedy) — when set, the per-Card
--   `level` view decodes at that temperature instead of greedily, and
--   the value lands on `meta.temperature`. Pass `1.0` to read a Card
--   on the same decode scale a `fight_matrix` run uses, which is what
--   makes an audit and a fight differ in one variable. Only the
--   number-ness is checked on the driver; the runner owns the
--   positive / finite rule and the "absent means greedy" rule.
--
-- ## Return / log
--
-- The driver returns the runner's report table verbatim (per_card /
-- sd_matrix / meta), so a Rust smoke harness can assert on it. It also
-- emits one `alc.log("info", ...)` header line plus one line per Card
-- plus one line per unordered SD pair, all under the `[gameai-audit]`
-- tag. That is the human summary an operator scans before opening the
-- saved JSON.
--
-- ## Why the driver, not `alc_run(code = "...")` inline
--
-- The runner boots two host bridges (`alc.nn.card.load_handle` and
-- `alc.nn.metric.registry.evaluate`) whose failure mode is easier to
-- read once with a named driver on the top of the stack than every
-- time a fresh ad-hoc snippet is pasted. The driver is the same shape
-- `train_guardian_npc.lua` / `eval_guardian_player_generalization.lua`
-- ship in this directory.

local am = require("gameai_metrics.audit_matrix")

-- `ctx` is a global injected by alc_run. When the caller passes no ctx
-- the global is a non-table userdata that cannot be indexed, so every
-- read goes through pcall.
local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

--- Decode a required non-empty string field or raise with the field
--- name literal so the caller does not have to guess what is missing.
local function require_string(field)
    local v = ctx_field(field)
    if type(v) ~= "string" or v == "" then
        error(
            string.format(
                "audit_boss_collection: ctx.%s is required and must be a non-empty string, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Decode an optional integer field, falling back to `default`. A
--- caller may pass a JSON number or omit the field entirely; anything
--- else (string, table, boolean) is refused loudly rather than
--- silently coerced.
local function optional_int(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "number" then
        error(
            string.format(
                "audit_boss_collection: ctx.%s must be a number or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return math.floor(v)
end

--- Decode an optional number field, falling back to `default`. Unlike
--- `optional_int` the value keeps its fractional part — a temperature
--- of `0.7` is a legitimate ask and flooring it would silently turn it
--- into greedy-ish nonsense. The positive / finite check stays in the
--- runner (`audit_matrix`'s temperature decode), which owns the
--- "absent means greedy" rule.
local function optional_number(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "number" then
        error(
            string.format(
                "audit_boss_collection: ctx.%s must be a number or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Decode an optional string field, falling back to `default` (which
--- may be `nil` for a field that has no default at all, e.g.
--- `teacher_alias`).
local function optional_string(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "string" or v == "" then
        error(
            string.format(
                "audit_boss_collection: ctx.%s must be a non-empty string or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

local COLLECTION_PATH = require_string("collection_path")
local OUTPUT = require_string("output")
local N_GAMES = optional_int("n_games", am.DEFAULT_N_GAMES)
local PROMPT_SET_SIZE = optional_int("prompt_set_size", am.DEFAULT_PROMPT_SET_SIZE)
local SEED = optional_int("seed", am.DEFAULT_SEED)
local STYLE = optional_string("style", "guardian")
local TEACHER_ALIAS = optional_string("teacher_alias", nil)
local TEMPERATURE = optional_number("temperature", nil)

local function log(msg)
    alc.log("info", "[gameai-audit] " .. msg)
end

--- Format one numeric field for the summary line. Missing / errored
--- fields land as `-` rather than a spurious zero, mirroring the
--- runner's per_card omission rule (a measurement gap is not a
--- labelled outcome).
local function fmt_num(value)
    if type(value) ~= "number" then
        return "-"
    end
    return string.format("%.3f", value)
end

--- Aliases in the report land as `sd_matrix` keys; the caller cares
--- about a stable summary order, so the driver walks them sorted
--- rather than in the runner's internal ordering.
local function sorted_aliases(sd_matrix)
    local out = {}
    for alias in pairs(sd_matrix) do
        out[#out + 1] = alias
    end
    table.sort(out)
    return out
end

local audit = am.new({
    collection_path = COLLECTION_PATH,
    n_games = N_GAMES,
    prompt_set_size = PROMPT_SET_SIZE,
    seed = SEED,
    style = STYLE,
    teacher_alias = TEACHER_ALIAS,
    temperature = TEMPERATURE,
})

local report = audit:run()
audit:save(OUTPUT)

local aliases = sorted_aliases(report.sd_matrix)
log(string.format(
    "audit: style=%s n_games=%d prompt_set=%d temperature=%s aliases=%d -> %s",
    STYLE,
    N_GAMES,
    report.meta.prompt_set_size,
    -- `greedy` rather than `-`: the field distinguishes the two
    -- decode modes at a glance when several audit runs scroll by.
    TEMPERATURE ~= nil and string.format("%g", TEMPERATURE) or "greedy",
    #aliases,
    OUTPUT
))

-- Per-Card summary: one line per alias, tab-separated
-- `alias / win_rate / sd_teacher / trickiness`. `sd_teacher` is `-`
-- when `teacher_alias` was not pinned; the driver treats the two
-- absences (no view / measurement failed) identically for the log
-- summary because both mean "no number to compare".
for _, alias in ipairs(aliases) do
    local entry = report.per_card[alias] or {}
    log(
        string.format(
            "per_card\t%s\twin_rate=%s\tsd_teacher=%s\ttrickiness=%s",
            alias,
            fmt_num(entry.win_rate),
            fmt_num(entry.sd_teacher),
            fmt_num(entry.trickiness_norm)
        )
    )
end

-- Pair-wise SD summary: one line per unordered pair (`i < j`), so a
-- 3-alias collection prints 3 lines. The diagonal is skipped (always
-- 0.0 by construction).
for i = 1, #aliases - 1 do
    for j = i + 1, #aliases do
        local a, b = aliases[i], aliases[j]
        log(string.format("SD(%s,%s)=%s", a, b, fmt_num(report.sd_matrix[a][b])))
    end
end

return report
