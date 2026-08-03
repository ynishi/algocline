-- Sweep a boss-harvest Card collection against a player Card pool over
-- a grid of decode temperatures, through `gameai_metrics.fight_matrix`.
--
-- Self-contained script for `alc_run` (`code_file` form). Where
-- `fight_boss_collection.lua` measures one matrix at one temperature,
-- this driver runs that same matrix once per grid point and collects the
-- results into a single report. Everything else — the boss axis, the
-- player axis, the game loop, the Wilson interval — still belongs to the
-- runner; this file only decodes the caller ctx, loops, checks that
-- nothing but the temperature moved between runs, writes the JSON and
-- prints a human summary. No `alc.llm` call happens anywhere on the
-- path, so the run never pauses for a host response.
--
--   alc_run(
--     code_file = "<repo>/examples/gameai/fight_boss_sweep.lua",
--     ctx = {
--       -- required
--       collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
--       players         = { "guardian_player_npc_sentinel" },
--       output          = "workspace/gameai-harvest/fight_sweep1.json",
--       temperatures    = { 0.25, 0.5, 1.0, 1.5, 2.0 },
--       -- optional (all fall back to fight_matrix.DEFAULT_* / "guardian")
--       n_games         = 200,
--       seed            = 20260731,
--       style           = "guardian",
--       per_game        = false,
--     },
--   )
--
-- ## Required ctx fields (loud on absence)
--
-- - `collection_path` — harvest manifest written by
--   `gameai_metrics.harvest_collection:save()`; it names the boss axis.
--   The driver refuses to guess a path because a bad guess would either
--   silently pick up an unrelated file or trip an obscure `io.open`
--   error further down.
-- - `players` — non-empty array of player Card aliases (the player
--   axis). The driver only checks that the field *is* a non-empty
--   array, so the message names this script when the ctx shape is
--   wrong; per-element checks (non-empty strings, no duplicates) stay
--   in `fight_matrix.decode_players`, which already raises with the
--   offending index.
-- - `output` — where the sweep JSON is written. The parent directory is
--   created for free via `gameai_metrics._fs.ensure_parent_dir`, so a
--   fresh `workspace/` sub-tree does not need pre-mkdir.
-- - `temperatures` — non-empty array of finite positive numbers, the
--   grid. Unlike the single-temperature driver this one owns the
--   element checks rather than leaving them to
--   `fight_matrix.decode_temperature`: a grid point that the runner
--   would refuse has to be caught *before* the first fight runs, or the
--   sweep spends minutes measuring and then dies on the last point. A
--   repeated value is refused for the same reason a repeated alias is —
--   it would run one measurement twice under one label and invite a
--   reader to treat the pair as a replication. The order given is the
--   order kept: the grid is not sorted, so a caller who wants an
--   ascending curve passes an ascending array.
--
-- ## Optional ctx fields (default from fight_matrix)
--
-- - `n_games` (default `fight_matrix.DEFAULT_N_GAMES` = 200) — fights
--   **per cell**, so a 3x1 matrix over a 5-point grid at the default
--   plays 3000 games.
-- - `seed` (default `fight_matrix.DEFAULT_SEED`) — see Reproducibility
--   below; every grid point is run under this one seed.
-- - `style` (default `"guardian"`) — the basis the boss prompt encodes
--   distances against *and* the basis the player view is built under:
--   one fight, one board. Must be one of `guardian_duel.STYLES`.
-- - `per_game` (default `false`) — when true every cell of every grid
--   point also carries a `games` array: one `{outcome, game_length,
--   final_hp_margin}` record per fight of that cell, in played order.
--   That multiplies the output size by `n_games` records per cell *per
--   grid point*, so it is off unless a caller is after the
--   distributions. Only a boolean or nil is accepted — a truthiness
--   read would turn the string `"false"` into the opposite of what it
--   says.
--
-- There is deliberately no `temperature` (singular) field: this driver's
-- whole subject is the grid, and accepting both spellings would leave a
-- reader guessing which one won. A ctx that carries one anyway — the
-- easy slip when a `fight_boss_collection` ctx is copied and
-- `temperatures` bolted on — is refused loudly rather than silently
-- outvoted by the grid, for the same reason `level` refuses an
-- `opponent_style` no opponent reads: an ignored opt reads as though it
-- applied.
--
-- ## A single-variable sweep, not a paired comparison
--
-- Every grid point runs under the same `seed`, the same axes and the
-- same `n_games`, so the temperature is the only input that moves. That
-- is a reproducibility claim and nothing more: the seed replays a run,
-- it does **not** pair the games of two grid points. Two runs at
-- different temperatures diverge at the first draw that resolves
-- differently, and from there the draw sequences no longer correspond
-- game for game — there is nothing to difference out and no variance
-- reduction to collect. (`level.lua` withdraws the same claim for the
-- same reason, and `new_game` opens the same position for every seed, so
-- there is not even a per-game opening for two runs to share.) The
-- difference between two cells of this report is a difference of two
-- independent estimates; read it against the Wilson intervals, not as a
-- paired delta.
--
-- To make "only the temperature moved" checkable rather than merely
-- claimed, the driver compares every grid point's `meta` against the
-- first one's and raises when `bosses`, `players`, `n_games`, `seed` or
-- `style` differ. Those cannot drift under the current runner, which is
-- exactly why the check is cheap: it costs one comparison per grid point
-- and it fails loudly the day something upstream starts deriving one of
-- them per call.
--
-- ## Boss perspective
--
-- Every reported number is from the **boss** seat: `win_rate` is the
-- boss Card's rate with draws counted as half a win, and
-- `final_hp_margin_mean` is `boss.hp - player.hp`. The player-side rate
-- is exactly `1 - win_rate`, so neither the report nor this summary
-- prints it — one number per cell, one convention, no sign to track.
--
-- "Greedy" has no spelling on this path, inherited from `fight_matrix`:
-- `guardian_duel` carries no RNG and `new_game` opens the same position
-- for every seed, so a greedy Card-vs-Card fight is one game replayed N
-- times. Every grid point is therefore a positive temperature, and `0`
-- is refused rather than read as "greedy".
--
-- ## Output shape
--
-- ```
-- {
--     sweep = {
--         { temperature = 0.25, matrix = <fight_matrix report.matrix> },
--         ...  -- one entry per grid point, in ctx.temperatures order
--     },
--     meta = { n_games, seed, style, per_game, temperatures,
--              bosses, players, collection_path },
-- }
-- ```
--
-- `sweep` is an array rather than a table keyed by temperature: JSON
-- object keys are strings, so a number-keyed map would round-trip as
-- `"0.25"` / `"1"` and lose the caller's order on the way back in.
-- `meta.bosses` / `meta.players` are read off the first grid point's
-- report (the runner is the authority on what it measured) and the
-- remaining points are checked against them, as described above.
--
-- ## Failure and partial output
--
-- A failure at any grid point aborts the whole sweep: the error
-- propagates, `output` is never opened, and the summary below is never
-- printed. The save happens once, after the last grid point landed, so
-- there is no half-written JSON for a reader to mistake for a completed
-- sweep. Re-running is the recovery path — the sweep is deterministic
-- under its seed.
--
-- ## Return / log
--
-- The driver returns the report table above, so a Rust smoke harness can
-- assert on it. It also emits, under the `[gameai-sweep]` tag, one
-- header line for the sweep, then per grid point one header line plus
-- one line per matrix cell. Cells are walked in `meta.bosses` x
-- `meta.players` order — the axes keep the order they were given in,
-- which the alias-keyed `matrix` tables do not carry.

local fm = require("gameai_metrics.fight_matrix")
local fs = require("gameai_metrics._fs")

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
                "fight_boss_sweep: ctx.%s is required and must be a non-empty string, got %s",
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
                "fight_boss_sweep: ctx.%s is required and must be a non-empty array, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Decode the temperature grid: a non-empty array of finite positive
--- numbers with no repeats, in the order the caller gave them.
---
--- The element checks duplicate `fight_matrix.decode_temperature` on
--- purpose. That one fires when its grid point is reached, which on a
--- sweep means after every earlier point has already been measured; a
--- grid is worth validating whole before the first fight starts. The
--- returned array is a copy, so a later mutation of the ctx table cannot
--- desynchronise the loop from the recorded `meta.temperatures`.
local function require_temperatures()
    local raw = ctx_field("temperatures")
    if type(raw) ~= "table" or #raw == 0 then
        error(
            string.format(
                "fight_boss_sweep: ctx.temperatures is required and must be a non-empty array "
                    .. "of finite positive numbers, got %s",
                type(raw)
            )
        )
    end
    local seen = {}
    local out = {}
    for i = 1, #raw do
        local value = raw[i]
        if type(value) ~= "number" then
            error(
                string.format(
                    "fight_boss_sweep: ctx.temperatures[%d] must be a number, got %s",
                    i,
                    type(value)
                )
            )
        end
        -- `value ~= value` is the NaN test; `math.huge` and any
        -- non-positive value are refused for the reason the header
        -- gives (a fight has no greedy mode).
        if value ~= value or value == math.huge or value <= 0 then
            error(
                string.format(
                    "fight_boss_sweep: ctx.temperatures[%d] must be a finite positive number "
                        .. "(a fight has no greedy mode), got %s",
                    i,
                    tostring(value)
                )
            )
        end
        if seen[value] ~= nil then
            error(
                string.format(
                    "fight_boss_sweep: ctx.temperatures lists %s twice (index %d and %d); "
                        .. "one grid point measured twice is not a replication",
                    tostring(value),
                    seen[value],
                    i
                )
            )
        end
        seen[value] = i
        out[i] = value
    end
    return out
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
                "fight_boss_sweep: ctx.%s must be a number or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return math.floor(v)
end

--- Decode an optional boolean field, falling back to `default`. A
--- non-boolean is refused rather than read for truthiness: a JSON ctx
--- carrying the string `"false"` would otherwise switch the flag *on*,
--- which is the one misreading a boolean opt can make silently.
local function optional_boolean(field, default)
    local v = ctx_field(field)
    if v == nil then
        return default
    end
    if type(v) ~= "boolean" then
        error(
            string.format(
                "fight_boss_sweep: ctx.%s must be a boolean or nil, got %s",
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
                "fight_boss_sweep: ctx.%s must be a non-empty string or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

-- Refused rather than ignored: the grid is the only temperature input,
-- and a leftover singular field (a copied fight_boss_collection ctx)
-- would otherwise read as though it applied. See the header.
if ctx_field("temperature") ~= nil then
    error(
        "fight_boss_sweep: ctx.temperature (singular) is not read by this driver; "
            .. "put every grid point in ctx.temperatures instead"
    )
end

local COLLECTION_PATH = require_string("collection_path")
local PLAYERS = require_array("players")
local OUTPUT = require_string("output")
local TEMPERATURES = require_temperatures()
local N_GAMES = optional_int("n_games", fm.DEFAULT_N_GAMES)
local SEED = optional_int("seed", fm.DEFAULT_SEED)
local STYLE = optional_string("style", "guardian")
local PER_GAME = optional_boolean("per_game", false)

local function log(msg)
    alc.log("info", "[gameai-sweep] " .. msg)
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

--- Format a grid point for the log. `%g` keeps `0.25` and `1` readable
--- instead of padding every temperature to three decimals like the
--- measured quantities above — the grid is an input, not a measurement.
local function fmt_temp(value)
    return string.format("%g", value)
end

-- ─── Cross-point invariant checks ───────────────────────────────────
--
-- "Only the temperature moved" is the premise of the whole report, so
-- it is checked rather than assumed. The first grid point's meta is the
-- reference; every later one is compared against it and a mismatch
-- names the field, the offending grid point and both values.

local function invariant_error(field, index, temperature, expected, got)
    error(
        string.format(
            "fight_boss_sweep: grid point %d (temperature %s) reports %s=%s but the first point "
                .. "reported %s; a temperature sweep must hold every other input fixed",
            index,
            fmt_temp(temperature),
            field,
            tostring(got),
            tostring(expected)
        )
    )
end

local function require_scalar_match(field, expected, got, index, temperature)
    if got ~= expected then
        invariant_error(field, index, temperature, expected, got)
    end
end

local function require_axis_match(field, expected, got, index, temperature)
    if type(got) ~= "table" then
        invariant_error(field, index, temperature, "an array", type(got))
    end
    if #got ~= #expected then
        invariant_error(
            field,
            index,
            temperature,
            string.format("%d entries", #expected),
            string.format("%d entries", #got)
        )
    end
    for i = 1, #expected do
        if got[i] ~= expected[i] then
            invariant_error(
                string.format("%s[%d]", field, i),
                index,
                temperature,
                expected[i],
                got[i]
            )
        end
    end
end

--- Read the axes off the reference report, refusing a meta that cannot
--- describe a matrix. Without this the summary loop below would print
--- nothing and the caller would read an empty sweep as a measured one.
local function require_axes(meta)
    if type(meta.bosses) ~= "table" or #meta.bosses == 0 then
        error("fight_boss_sweep: fight_matrix reported no bosses axis for the first grid point")
    end
    if type(meta.players) ~= "table" or #meta.players == 0 then
        error("fight_boss_sweep: fight_matrix reported no players axis for the first grid point")
    end
end

-- ─── The sweep ──────────────────────────────────────────────────────

local sweep = {}
local reference = nil

for index, temperature in ipairs(TEMPERATURES) do
    local fight = fm.new({
        collection_path = COLLECTION_PATH,
        players = PLAYERS,
        n_games = N_GAMES,
        seed = SEED,
        style = STYLE,
        temperature = temperature,
        per_game = PER_GAME,
    })
    local report = fight:run()
    if
        type(report) ~= "table"
        or type(report.matrix) ~= "table"
        or type(report.meta) ~= "table"
    then
        error(
            string.format(
                "fight_boss_sweep: fight_matrix returned no matrix / meta table for "
                    .. "temperature %s (got %s)",
                fmt_temp(temperature),
                type(report)
            )
        )
    end
    if reference == nil then
        require_axes(report.meta)
        reference = report.meta
    else
        require_axis_match("bosses", reference.bosses, report.meta.bosses, index, temperature)
        require_axis_match("players", reference.players, report.meta.players, index, temperature)
        require_scalar_match("n_games", reference.n_games, report.meta.n_games, index, temperature)
        require_scalar_match("seed", reference.seed, report.meta.seed, index, temperature)
        require_scalar_match("style", reference.style, report.meta.style, index, temperature)
    end
    sweep[index] = { temperature = temperature, matrix = report.matrix }
end

local bosses = reference.bosses
local players = reference.players

local report = {
    sweep = sweep,
    meta = {
        -- The measured quantities are read back off the runner rather
        -- than echoed from the ctx: what the report describes is what
        -- was actually run, and the invariant checks above already
        -- proved every grid point agrees on `n_games` / `seed` /
        -- `style` / the two axes. `per_game` rides along from the same
        -- reference point; it is a flag this driver passes unchanged to
        -- every point, so there is no second value for it to disagree
        -- with.
        n_games = reference.n_games,
        seed = reference.seed,
        style = reference.style,
        per_game = reference.per_game,
        temperatures = TEMPERATURES,
        bosses = bosses,
        players = players,
        collection_path = COLLECTION_PATH,
    },
}

-- Written once, after the last grid point landed: a sweep that died
-- half-way leaves no file rather than a JSON that looks complete.
fs.ensure_parent_dir(OUTPUT)
if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
    error("fight_boss_sweep: alc.json_encode is required to write the report (host bridge missing)")
end
local encoded_ok, encoded = pcall(alc.json_encode, report)
if not encoded_ok then
    error("fight_boss_sweep: failed to encode the sweep report: " .. tostring(encoded))
end
local out_file, open_err = io.open(OUTPUT, "w")
if out_file == nil then
    error(
        string.format(
            "fight_boss_sweep: cannot open %q for writing: %s",
            OUTPUT,
            tostring(open_err)
        )
    )
end
local write_ok, write_err = pcall(function()
    out_file:write(encoded)
end)
out_file:close()
if not write_ok then
    error(string.format("fight_boss_sweep: write to %q failed: %s", OUTPUT, tostring(write_err)))
end

-- ─── Summary ────────────────────────────────────────────────────────
--
-- Emitted after the save for the same reason the save comes last: a
-- sweep that did not finish prints no summary, so a scrollback with a
-- header line in it is a sweep that completed.

log(
    string.format(
        "sweep: style=%s n_games=%d per_game=%s temperatures=%d bosses=%d players=%d cells=%d -> %s",
        tostring(report.meta.style),
        report.meta.n_games,
        tostring(report.meta.per_game),
        #TEMPERATURES,
        #bosses,
        #players,
        #TEMPERATURES * #bosses * #players,
        OUTPUT
    )
)

for index, entry in ipairs(sweep) do
    log(
        string.format(
            "sweep\tT=%s\tpoint %d/%d\tcells=%d",
            fmt_temp(entry.temperature),
            index,
            #TEMPERATURES,
            #bosses * #players
        )
    )
    for _, boss in ipairs(bosses) do
        local row = entry.matrix[boss] or {}
        for _, player in ipairs(players) do
            local cell = row[player] or {}
            log(
                string.format(
                    "sweep\tT=%s\t%s\tvs\t%s\twin_rate=%s\tci=[%s,%s]\tlen=%s\tmargin=%s",
                    fmt_temp(entry.temperature),
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
end

return report
