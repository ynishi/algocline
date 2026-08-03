--- gameai_metrics.fight_matrix — Card-vs-Card fight matrix runner: every
--- boss Card of a collection against every player Card of a pool.
---
--- ## What this is (and is not)
---
--- `audit_matrix` re-measures each boss Card on its own axes (win rate
--- against a scripted opponent, trickiness, distance from a teacher).
--- Those numbers describe a Card in isolation. This runner answers the
--- other question: what happens when two *baked* policies meet. Each
--- boss Card holds the boss seat and each player Card holds the player
--- seat — both decode their own vocabulary through their own seat, so
--- the fight stays the fixed-role game `guardian_duel` implements. No
--- composite wiring, no seat swapping.
---
--- Nothing here is a decision. The runner never compares a cell against
--- a target and never picks a winner: what comes out is the raw material
--- a human reads when asking whether a strong Card is worth adopting, or
--- whether two Cards are actually close enough to make a fight
--- interesting. Reading it belongs to the judgment layer (measurement
--- and judgment are separate layers by design).
---
--- ## Boss perspective
---
--- Every reported number is from the **boss** seat: `win_rate` is the
--- boss Card's rate, draws counted as half a win. The player-side rate
--- is exactly `1 - win_rate`, so it is not reported separately — one
--- number per cell, one convention, no sign to keep track of.
--- `final_hp_margin_mean` is likewise `boss.hp - player.hp`.
---
--- ## Why a temperature is mandatory here
---
--- `guardian_duel` carries no RNG and `new_game` opens the same position
--- for every seed, so a fight between two greedy Cards is one game
--- replayed N times: the win rate over that batch is a rate over a
--- single sample and its Wilson interval is a point. The only source of
--- variance in a Card-vs-Card fight is the decode itself, so this runner
--- always passes a temperature down to `level` (default `1.0`) and
--- refuses a non-numeric one. `1.0` is the same scale `trickiness`
--- measures its entropy on, which is what lets an audit's
--- `trickiness_norm` and the spread of a fight refer to one distribution.
---
--- `opts.temperature` is *required* in the design sense: an omitted key
--- falls back to `1.0` rather than to greedy. Lua cannot tell an omitted
--- key from an explicit `nil`, so "greedy" simply has no spelling on
--- this runner — a caller who wants a greedy measurement calls `level`
--- directly, where a greedy fight against a scripted opponent is still
--- a legitimate thing to ask for.
---
--- ## Reproducibility
---
--- One `seed` replays a whole run: `level` derives every temperature
--- draw from `seed + k` for a run-local counter. That is the only claim
--- made — it is **not** a paired comparison. `new_game` opens the same
--- position for every seed, so there is no per-game opening for two
--- opponents to share and no variance reduction to gain from sharing it.
---
--- ## Contract
---
--- ```lua
--- local fm = require("gameai_metrics.fight_matrix")
---
--- local fight = fm.new({
---     collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
---     -- or, without a manifest:
---     -- bosses = { "guardian_duel_npc_weak", "guardian_duel_npc_mid" },
---     players = { "guardian_player_npc_sentinel" },  -- required
---     n_games = 200,       -- fights per cell (default)
---     seed = 20260731,
---     style = "guardian",  -- required, one of guardian_duel.STYLES
---     temperature = 1.0,   -- default
---     per_game = false,    -- default; true adds a `games` array per cell
--- })
---
--- local report = fight:run()
--- fight:save("workspace/gameai-harvest/fight_run1.json")
--- ```
---
--- The report shape is
---
--- ```
--- {
---     matrix = {
---         [boss_alias] = {
---             [player_alias] = {
---                 win_rate = 0.62,     -- boss seat, draws = 0.5
---                 ci_lower = 0.55,     -- Wilson 95% over this cell
---                 ci_upper = 0.68,
---                 n_games  = 200,      -- fights in this cell
---                 game_length_mean     = 6.4,   -- turns played
---                 final_hp_margin_mean = 3.1,   -- boss hp - player hp
---             },
---             ...
---         },
---         ...
---     },
---     meta = { n_games, seed, style, temperature, bosses, players,
---              collection_path? },
--- }
--- ```
---
--- `meta.bosses` and `meta.players` keep the order the axes were given
--- in, which the `matrix` tables (keyed by alias) do not carry.
---
--- ## One `level` call per row
---
--- A boss row is a single `level(boss_handle, nil, n_games, seed, {seat
--- = "boss", opponents = players, ...})` call whose `per_opponent` table
--- *is* the row. The game loop, the Wilson interval and the pool
--- contract therefore live in exactly one place. Two consequences worth
--- knowing:
---
--- - The same player alias is resolved and loaded once per boss row (a
---   3-boss run loads each player Card 3 times). Cells are independent
---   measurements, so this costs load time and nothing else.
--- - A cell is `level`'s `per_opponent` entry verbatim. A field added
---   there shows up here without a change to this file. That is how
---   `opts.per_game = true` puts a `games` array (one record per fight
---   of the cell) inside every cell: this runner only forwards the flag.
---
--- `save()` writes the report as JSON via `alc.json_encode`, after
--- `gameai_metrics._fs.ensure_parent_dir(path)`, so the first write into
--- a fresh `workspace/` sub-tree creates its directories rather than
--- surfacing as an obscure `io.open` "No such file or directory".

local boss_seat = require("gameai_metrics.boss_seat")
local fs = require("gameai_metrics._fs")

-- `level` is required directly, not through the parent package. This
-- runner composes the metric as a plain function and never reads
-- alc.nn.metric.registry, so the `require("gameai_metrics")` that
-- audit_matrix needs (its views fire through the registry) would only
-- add a dependency here. If a future view of this runner goes through
-- the registry, that parent require has to come back with it.
local level = require("gameai_metrics.level")

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics.fight_matrix",
    version = "0.1.0",
    description = "Card-vs-Card fight matrix runner: boss Cards against player Cards, per-cell win rate + Wilson CI.",
    category = "game",
}

local DEFAULT_N_GAMES = 200
local DEFAULT_SEED = 0
local DEFAULT_TEMPERATURE = 1.0

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "fight_matrix: alc.json_encode is required to write the report (host bridge missing)",
            0
        )
    end
    return alc.json_encode
end

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "fight_matrix: alc.json_decode is required to read the collection manifest "
                .. "(host bridge missing)",
            0
        )
    end
    return alc.json_decode
end

local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("fight_matrix: alc.nn.card is unavailable; build algocline with --features nn")
    end
    if type(alc.nn.card.load_handle) ~= "function" then
        error("fight_matrix: alc.nn.card.load_handle is unavailable")
    end
end

local function require_card_alias_bridge()
    if
        type(alc) ~= "table"
        or type(alc.card) ~= "table"
        or type(alc.card.get_by_alias) ~= "function"
    then
        error(
            "fight_matrix: alc.card.get_by_alias is unavailable "
                .. "(needed to resolve an alias to a card_id when opts.bosses is set)"
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
            string.format("fight_matrix: %s must be a finite integer, got %s", field, tostring(raw)),
            3
        )
    end
    local i = math.floor(raw)
    if must_be_positive and i <= 0 then
        error(
            string.format(
                "fight_matrix: %s must be a positive integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    if not must_be_positive and i < 0 then
        error(
            string.format(
                "fight_matrix: %s must be a non-negative integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    return i
end

--- Decode the fight temperature.
---
--- An absent key means the default `1.0`, never greedy: a greedy
--- Card-vs-Card fight measures one game N times (see the header), so
--- this runner has no spelling for it. Anything that is not a finite
--- positive number is refused here rather than inside `level`, so the
--- message names the layer the caller called.
local function decode_temperature(raw)
    if raw == nil then
        return DEFAULT_TEMPERATURE
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error(
            string.format(
                "fight_matrix: opts.temperature must be a finite positive number "
                    .. "(a fight has no greedy mode; omit the key for the default %s), got %s",
                tostring(DEFAULT_TEMPERATURE),
                tostring(raw)
            ),
            3
        )
    end
    return raw
end

--- Decode the per-game record flag.
---
--- Forwarded to `level` as-is; the records themselves ride into every
--- cell through the verbatim `per_opponent` transcription. Refused here
--- when it is not a boolean so the message names the layer the caller
--- called, and because `per_game = "false"` is truthy in Lua — reading
--- it for truthiness would turn a typo into 200 records per cell.
local function decode_per_game(raw)
    if raw == nil then
        return false
    end
    if type(raw) ~= "boolean" then
        error(
            string.format(
                "fight_matrix: opts.per_game must be true, false or nil, got %s",
                tostring(raw)
            ),
            3
        )
    end
    return raw
end

--- Normalise `opts.players` into a list of alias strings.
---
--- Checked here rather than left to `level.decode_pool`, which would
--- raise the same class of error with a `level:` prefix and leave the
--- caller looking for a pool they never named.
local function decode_players(players)
    if type(players) ~= "table" or #players == 0 then
        error(
            "fight_matrix: opts.players is required and must be a non-empty array of "
                .. "player Card aliases",
            3
        )
    end
    local seen = {}
    local out = {}
    for i = 1, #players do
        local alias = players[i]
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format(
                    "fight_matrix: opts.players[%d] must be a non-empty string, got %s",
                    i,
                    type(alias)
                ),
                3
            )
        end
        if seen[alias] then
            error(string.format("fight_matrix: opts.players lists %q twice", alias), 3)
        end
        seen[alias] = true
        out[i] = alias
    end
    return out
end

--- Normalise `opts.bosses` into a `{alias, card_id}` pair list.
---
--- Each `card_id` is resolved through `alc.card.get_by_alias`; an alias
--- without a bound Card is a loud error, the same treatment the
--- collection-path branch gives a missing `card_id`.
local function resolve_boss_aliases(aliases)
    if type(aliases) ~= "table" or #aliases == 0 then
        error("fight_matrix: opts.bosses must be a non-empty array of alias strings", 3)
    end
    require_card_alias_bridge()
    local seen = {}
    local out = {}
    for i = 1, #aliases do
        local alias = aliases[i]
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format(
                    "fight_matrix: opts.bosses[%d] must be a non-empty string, got %s",
                    i,
                    type(alias)
                ),
                3
            )
        end
        if seen[alias] then
            error(string.format("fight_matrix: opts.bosses lists %q twice", alias), 3)
        end
        seen[alias] = true
        local entry = alc.card.get_by_alias(alias)
        if type(entry) ~= "table" or type(entry.card_id) ~= "string" or entry.card_id == "" then
            error(
                string.format(
                    "fight_matrix: boss alias %q is not bound to a Card with a card_id "
                        .. "(alc.card.get_by_alias returned no card_id)",
                    alias
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = entry.card_id }
    end
    return out
end

--- Read a harvest manifest and turn its `entries` array into the same
--- `{alias, card_id}` shape the aliases path produces. A missing
--- `card_id` on any entry is a loud error: a partial fight would publish
--- a matrix whose boss axis did not match the collection the caller
--- pointed at.
local function resolve_bosses_from_collection(path)
    if type(path) ~= "string" or path == "" then
        error("fight_matrix: opts.collection_path must be a non-empty string", 3)
    end
    local decode = require_json_decoder()
    local f, open_err = io.open(path, "r")
    if f == nil then
        error(
            string.format(
                "fight_matrix: cannot open collection_path %q: %s",
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
                "fight_matrix: failed to decode collection_path %q: %s",
                path,
                tostring(parsed)
            ),
            3
        )
    end
    if type(parsed) ~= "table" or type(parsed.entries) ~= "table" then
        error(
            string.format(
                "fight_matrix: collection_path %q has no entries array (schema drift?)",
                path
            ),
            3
        )
    end
    if #parsed.entries == 0 then
        error(
            string.format(
                "fight_matrix: collection_path %q has zero entries; nothing to fight",
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
                    "fight_matrix: collection_path entries[%d] must be a table, got %s",
                    i,
                    type(entry)
                ),
                3
            )
        end
        local alias = entry.alias
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format("fight_matrix: collection_path entries[%d] has no alias string", i),
                3
            )
        end
        if seen[alias] then
            error(
                string.format("fight_matrix: collection_path entries list alias %q twice", alias),
                3
            )
        end
        seen[alias] = true
        local card_id = entry.card_id
        if type(card_id) ~= "string" or card_id == "" then
            error(
                string.format(
                    "fight_matrix: collection_path entries[%d] (alias %q) has no card_id; "
                        .. "the fight cannot reload a Card without its stored card_id",
                    i,
                    alias
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = card_id }
    end
    return out
end

-- ─── Runner ────────────────────────────────────────────────────────

local Fight = {}
Fight.__index = Fight

--- Load one boss Card by `card_id`, naming its alias on either failure
--- mode (a raise from the loader, or a `nil` handle from a card_id that
--- is no longer in the Card store).
local function load_boss_handle(entry)
    local ok, handle = pcall(alc.nn.card.load_handle, entry.card_id)
    if not ok then
        error(
            string.format(
                "fight_matrix: alc.nn.card.load_handle(%q) failed for boss alias %q: %s",
                entry.card_id,
                entry.alias,
                tostring(handle)
            )
        )
    end
    if handle == nil then
        error(
            string.format(
                "fight_matrix: alc.nn.card.load_handle(%q) returned nil for boss alias %q "
                    .. "(card_id likely removed from the Card store)",
                entry.card_id,
                entry.alias
            )
        )
    end
    return handle
end

--- Pick the cells of one boss row out of a `level` result, in the order
--- the player axis was given. A player named on the axis but absent from
--- `per_opponent` is a contract break between this runner and `level`,
--- so it raises rather than leaving a hole in the matrix.
local function row_from_level(result, players, boss_alias)
    if type(result) ~= "table" or type(result.per_opponent) ~= "table" then
        error(
            string.format(
                "fight_matrix: level returned no per_opponent table for boss %q (got %s)",
                boss_alias,
                type(result)
            )
        )
    end
    local row = {}
    for _, player in ipairs(players) do
        local cell = result.per_opponent[player]
        if type(cell) ~= "table" then
            error(
                string.format(
                    "fight_matrix: level reported no cell for boss %q vs player %q",
                    boss_alias,
                    player
                )
            )
        end
        row[player] = cell
    end
    return row
end

--- Run the whole matrix: one `level` call per boss row, every player of
--- the pool measured inside it. Boss handles are loaded up front so a
--- card_id that no longer resolves fails before any fight runs — a
--- matrix either covers every cell or fails with the offending alias
--- named. The report is cached so `:save()` / `:report()` read the same
--- object without a second run.
---@return table report
function Fight:run()
    require_nn_card()

    local handles = {}
    for i, entry in ipairs(self._bosses) do
        handles[i] = load_boss_handle(entry)
    end

    local matrix = {}
    for i, entry in ipairs(self._bosses) do
        local result = level(handles[i], nil, self._n_games, self._seed, {
            seat = "boss",
            style = self._style,
            opponents = self._players,
            temperature = self._temperature,
            per_game = self._per_game,
        })
        matrix[entry.alias] = row_from_level(result, self._players, entry.alias)
    end

    local bosses = {}
    for i, entry in ipairs(self._bosses) do
        bosses[i] = entry.alias
    end
    local players = {}
    for i, alias in ipairs(self._players) do
        players[i] = alias
    end

    local meta = {
        n_games = self._n_games,
        seed = self._seed,
        style = self._style,
        temperature = self._temperature,
        -- Recorded so a saved report is self-describing: a cell without
        -- `games` could otherwise mean either "the flag was off" or "a
        -- field went missing", and the JSON alone could not say which.
        per_game = self._per_game,
        bosses = bosses,
        players = players,
    }
    if self._collection_path ~= nil then
        meta.collection_path = self._collection_path
    end

    local report = { matrix = matrix, meta = meta }
    self._report = report
    return report
end

--- Read-only accessor for the last `:run()` report. Raises when called
--- before `:run()` — a report that did not exist would otherwise read as
--- an empty matrix.
---@return table report
function Fight:report()
    if self._report == nil then
        error("fight_matrix:report: no report yet; call :run() first", 2)
    end
    return self._report
end

--- Encode the report and write it to `path`, creating the parent
--- directory first (via `gameai_metrics._fs`) so a first save into a
--- fresh workspace sub-tree does not surface as an obscure `io.open`
--- error.
---@param path string
function Fight:save(path)
    if type(path) ~= "string" or path == "" then
        error("fight_matrix:save: path must be a non-empty string", 2)
    end
    if self._report == nil then
        error("fight_matrix:save: no report yet; call :run() before :save()", 2)
    end
    fs.ensure_parent_dir(path)
    local encode = require_json_encoder()
    local ok, encoded = pcall(encode, self._report)
    if not ok then
        error("fight_matrix:save: failed to encode report: " .. tostring(encoded), 2)
    end
    local f, err = io.open(path, "w")
    if f == nil then
        error(
            string.format("fight_matrix:save: cannot open %q for writing: %s", path, tostring(err)),
            2
        )
    end
    local ok_write, write_err = pcall(function()
        f:write(encoded)
    end)
    f:close()
    if not ok_write then
        error(
            string.format("fight_matrix:save: write to %q failed: %s", path, tostring(write_err)),
            2
        )
    end
end

--- Build a new fight runner. `opts` is a table:
---
--- - `collection_path` — string, harvest manifest naming the boss axis
---   (produced by `gameai_metrics.harvest_collection:save()`). One of
---   `collection_path` / `bosses` is required; passing both is a loud
---   error.
--- - `bosses` — array of boss Card alias strings, the manifest-free
---   spelling of the same axis. Each alias is resolved through
---   `alc.card.get_by_alias`, so an unbound alias raises immediately.
--- - `players` — required, non-empty array of player Card aliases. This
---   axis has no manifest form: player Cards are baked one at a time
---   (`bake_guardian_player_from_log.lua`), not harvested in bands.
--- - `n_games` — integer, fights **per cell** (default `200`). A 3x2
---   matrix at the default therefore plays 1200 games.
--- - `seed` — integer, seeds every temperature draw of the run
---   (default `0`). See Reproducibility in the header.
--- - `style` — required, one of `guardian_duel.STYLES`. It is the basis
---   the boss prompt encodes distances against *and* the basis the
---   player view is built under: one fight, one board.
--- - `temperature` — positive finite number, default `1.0`. There is no
---   greedy mode; see the header for why.
--- - `per_game` — boolean, default `false`. Forwarded to `level`, which
---   then puts a `games` array (one record per fight, in played order)
---   inside every cell. Off by default: the flag multiplies the report
---   size by the cell count and a caller reading only rates has no use
---   for it.
---
---@param opts table
---@return table fight
function M.new(opts)
    if type(opts) ~= "table" then
        error("fight_matrix.new: opts must be a table, got " .. type(opts), 2)
    end

    if opts.collection_path ~= nil and opts.bosses ~= nil then
        error("fight_matrix.new: pass either opts.collection_path or opts.bosses, not both", 2)
    end
    if opts.collection_path == nil and opts.bosses == nil then
        error("fight_matrix.new: opts.collection_path or opts.bosses is required", 2)
    end

    local style = boss_seat.require_style(opts.style, "fight_matrix.new")
    local players = decode_players(opts.players)
    local n_games = decode_int(opts.n_games, DEFAULT_N_GAMES, "n_games", true)
    local seed = decode_int(opts.seed, DEFAULT_SEED, "seed", false)
    local temperature = decode_temperature(opts.temperature)
    local per_game = decode_per_game(opts.per_game)

    local bosses
    if opts.collection_path ~= nil then
        bosses = resolve_bosses_from_collection(opts.collection_path)
    else
        bosses = resolve_boss_aliases(opts.bosses)
    end

    return setmetatable({
        _bosses = bosses,
        _players = players,
        _style = style,
        _n_games = n_games,
        _seed = seed,
        _temperature = temperature,
        _per_game = per_game,
        _collection_path = opts.collection_path,
        _report = nil,
    }, Fight)
end

--- Read-only accessors, mostly for specs / driver logs.
M.DEFAULT_N_GAMES = DEFAULT_N_GAMES
M.DEFAULT_SEED = DEFAULT_SEED
M.DEFAULT_TEMPERATURE = DEFAULT_TEMPERATURE

return M
