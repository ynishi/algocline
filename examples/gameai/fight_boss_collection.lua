-- Fight a boss-harvest Card collection against a player Card pool
-- through `gameai_metrics.fight_matrix`.
--
-- Self-contained script for `alc_run` (`code_file` form). This is a
-- thin driver over `fight_matrix.new(opts):run() → :save(output)`; the
-- runner does all of the work, the driver only decodes the caller ctx,
-- fills the defaults from the module's own `DEFAULT_*` constants, calls
-- the runner and prints a per-cell human summary before returning the
-- report table. No `alc.llm` call happens anywhere on the path, so the
-- run never pauses for a host response.
--
--   alc_run(
--     code_file = "<repo>/examples/gameai/fight_boss_collection.lua",
--     ctx = {
--       -- required
--       collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
--       players         = { "guardian_player_npc_sentinel" },
--       output          = "workspace/gameai-harvest/fight_run1.json",
--       -- optional (all fall back to fight_matrix.DEFAULT_* / "guardian")
--       n_games         = 200,
--       seed            = 20260731,
--       style           = "guardian",
--       temperature     = 1.0,
--     },
--   )
--
-- ## Required ctx fields (loud on absence)
--
-- - `collection_path` — harvest manifest written by
--   `gameai_metrics.harvest_collection:save()`; it names the boss axis.
--   The driver refuses to guess a path because a bad guess would either
--   silently pick up an unrelated file or trip an obscure `io.open`
--   error further down. The manifest-free spelling (`bosses = {...}`)
--   is a runner option, not a driver one: this driver exists to fight a
--   harvested collection.
-- - `players` — non-empty array of player Card aliases (the player
--   axis). The driver only checks that the field *is* a non-empty
--   array, so the message names this script when the ctx shape is
--   wrong; per-element checks (non-empty strings, no duplicates) stay
--   in `fight_matrix.decode_players`, which already raises with the
--   offending index.
-- - `output` — where the fight JSON is written. The parent directory
--   is created for free via `gameai_metrics._fs.ensure_parent_dir`, so
--   a fresh `workspace/` sub-tree does not need pre-mkdir.
--
-- ## Optional ctx fields (default from fight_matrix)
--
-- - `n_games` (default `fight_matrix.DEFAULT_N_GAMES` = 200) — fights
--   **per cell**, so a 3x1 matrix at the default plays 600 games. The
--   higher the number the tighter the Wilson interval each cell lands
--   on.
-- - `seed` (default `fight_matrix.DEFAULT_SEED`) — seeds every
--   temperature draw of the run; one seed replays the whole matrix.
-- - `style` (default `"guardian"`) — the basis the boss prompt encodes
--   distances against *and* the basis the player view is built under:
--   one fight, one board. Must be one of `guardian_duel.STYLES`.
-- - `temperature` (default `fight_matrix.DEFAULT_TEMPERATURE` = 1.0) —
--   the decode temperature both seats sample at. Omitting the key
--   means `1.0`, never greedy: `guardian_duel` carries no RNG and
--   `new_game` opens the same position for every seed, so a greedy
--   Card-vs-Card fight is one game replayed N times. "Greedy" has no
--   spelling on this path by design; the runner refuses a non-numeric
--   or non-positive value, and so does this driver.
--
-- ## Boss perspective
--
-- Every reported number is from the **boss** seat: `win_rate` is the
-- boss Card's rate with draws counted as half a win, and
-- `final_hp_margin_mean` is `boss.hp - player.hp`. The player-side rate
-- is exactly `1 - win_rate`, so neither the report nor this summary
-- prints it — one number per cell, one convention, no sign to track.
--
-- ## Return / log
--
-- The driver returns the runner's report table verbatim (matrix /
-- meta), so a Rust smoke harness can assert on it. It also emits one
-- `alc.log("info", ...)` header line plus one line per matrix cell, all
-- under the `[gameai-fight]` tag. That is the human summary an operator
-- scans before opening the saved JSON. Cells are walked in
-- `meta.bosses` x `meta.players` order — the axes keep the order they
-- were given in, which the alias-keyed `matrix` tables do not carry.
--
-- ## Why the driver, not `alc_run(code = "...")` inline
--
-- The runner boots a host bridge per boss Card (`alc.nn.card
-- .load_handle`) and samples through `alc.nn.sampler`, whose failure
-- modes are easier to read once with a named driver on the top of the
-- stack than every time a fresh ad-hoc snippet is pasted. The driver is
-- the same shape `audit_boss_collection.lua` /
-- `eval_guardian_player_generalization.lua` ship in this directory.

local fm = require("gameai_metrics.fight_matrix")

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
                "fight_boss_collection: ctx.%s is required and must be a non-empty string, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Decode a required non-empty array field. Only the array-ness is
--- checked here: element validation belongs to the runner, which names
--- the offending index. The driver's job is to make a ctx shaped like a
--- JSON object / string / number fail with *this* script's name rather
--- than with a `fight_matrix:` prefix the caller never called.
local function require_array(field)
    local v = ctx_field(field)
    if type(v) ~= "table" or #v == 0 then
        error(
            string.format(
                "fight_boss_collection: ctx.%s is required and must be a non-empty array, got %s",
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
                "fight_boss_collection: ctx.%s must be a number or nil, got %s",
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
--- runner (`fight_matrix.decode_temperature`), which owns the "no
--- greedy spelling" rule.
local function optional_number(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "number" then
        error(
            string.format(
                "fight_boss_collection: ctx.%s must be a number or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Decode an optional string field, falling back to `default`.
local function optional_string(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "string" or v == "" then
        error(
            string.format(
                "fight_boss_collection: ctx.%s must be a non-empty string or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

local COLLECTION_PATH = require_string("collection_path")
local PLAYERS = require_array("players")
local OUTPUT = require_string("output")
local N_GAMES = optional_int("n_games", fm.DEFAULT_N_GAMES)
local SEED = optional_int("seed", fm.DEFAULT_SEED)
local STYLE = optional_string("style", "guardian")
local TEMPERATURE = optional_number("temperature", fm.DEFAULT_TEMPERATURE)

local function log(msg)
    alc.log("info", "[gameai-fight] " .. msg)
end

--- Format one numeric field for the summary line. Missing / errored
--- fields land as `-` rather than a spurious zero: a measurement gap is
--- not a labelled outcome, and a `0.000` win rate reads as a Card that
--- lost every game.
local function fmt_num(value)
    if type(value) ~= "number" then
        return "-"
    end
    return string.format("%.3f", value)
end

local fight = fm.new({
    collection_path = COLLECTION_PATH,
    players = PLAYERS,
    n_games = N_GAMES,
    seed = SEED,
    style = STYLE,
    temperature = TEMPERATURE,
})

local report = fight:run()
fight:save(OUTPUT)

local bosses = report.meta.bosses
local players = report.meta.players
log(
    string.format(
        "fight: style=%s n_games=%d temperature=%s bosses=%d players=%d cells=%d -> %s",
        report.meta.style,
        report.meta.n_games,
        fmt_num(report.meta.temperature),
        #bosses,
        #players,
        #bosses * #players,
        OUTPUT
    )
)

-- Per-cell summary: one line per (boss, player) pair, tab-separated
-- `boss / player / win_rate / Wilson CI / game length / hp margin`. The
-- axes are walked in `meta` order (the order the caller gave them in),
-- not sorted, so the summary reads in the same order as the collection
-- manifest and the `players` ctx array.
for _, boss in ipairs(bosses) do
    local row = report.matrix[boss] or {}
    for _, player in ipairs(players) do
        local cell = row[player] or {}
        log(
            string.format(
                "fight\t%s\tvs\t%s\twin_rate=%s\tci=[%s,%s]\tlen=%s\tmargin=%s",
                boss,
                player,
                fmt_num(cell.win_rate),
                fmt_num(cell.ci_lower),
                fmt_num(cell.ci_upper),
                fmt_num(cell.game_length_mean),
                fmt_num(cell.final_hp_margin_mean)
            )
        )
    end
end

return report
