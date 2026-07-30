--- card_duel_tournament — round robin between the card duel NPC styles
---
--- Plays every unordered pair of trained styles against each other and
--- folds the games into a win rate matrix. Both seats are driven by the
--- `card_duel_npc` package, so the match-up is model against model
--- rather than model against a reference policy: the only thing that
--- differs between the two sides is the Card alias behind each style.
---
--- ## Usage
---
--- ```lua
--- local tournament = require("card_duel_tournament")
--- local out = tournament.run({
---     styles = { "timid", "bold", "aggressive" },
---     games_per_pair = 10,
---     seed = 7,
--- })
--- print(out.result)
--- print(out.matrix.timid.bold.winrate)
--- ```
---
--- ## Algorithm
---
--- 1. Validate the requested styles against `card_duel.STYLES`, so a
---    typo fails here rather than as a missing Card alias halfway
---    through the run.
--- 2. For every unordered pair `(i < j)` play `games_per_pair` games.
---    The game seed is derived as
---    `seed + pair_index * 1000 + game_index`, which makes the whole
---    tournament reproducible from the one seed while keeping the deals
---    of different pairs independent.
--- 3. Each round asks `card_duel_npc` for one move per seat with
---    `mode = "decide"` and the alias `alias_prefix .. style`. The
---    answer carries the chosen rank plus the `gated` telemetry flag,
---    which is accumulated per style.
--- 4. The loop exits on `card_duel.is_over` and only then asks for
---    `card_duel.winner`. An unfinished game has no winner, and folding
---    that `nil` into `"draw"` would hide a loop that stopped early.
---
--- ## Entry contract
---
--- `run(ctx)` takes `{ styles?, games_per_pair?, seed?, alias_prefix? }`
--- and returns
---
--- - `matrix[a][b]` — `{ wins, losses, draws, winrate }` from the point
---   of view of `a`; `winrate` is `wins / games_per_pair`, so a pair
---   with many draws leaves both directions below `0.5`
--- - `summary[style]` — `{ total_winrate, gated_rate, avg_point_margin }`
---   over every game the style played
--- - `styles` / `games_per_pair` / `seed` — the effective settings
--- - `result` — one line an eval grader can read
---
--- The same fields may arrive JSON-encoded in `ctx.task`: the evalframe
--- provider passes a case input that way, and decoding it here lets one
--- entry point serve both `alc_run` and `alc_eval`.
---
--- ## Caveats
---
--- The average game length is not reported because it is constant: the
--- rules always run five rounds. `avg_point_margin` (own points minus
--- the opponent's, averaged over the games a style played) is the
--- comparable per-style number.
---
--- Within a pair the first style always takes the `p1` seat. The rules
--- are symmetric — both seats reveal simultaneously and score by the
--- same comparison — so the seat carries no advantage, but a style that
--- reads `opp_played` still sees a different history depending on the
--- seat it sits in.
---
--- Every decision is a model decode, so the cost is
--- `pairs * games_per_pair * 10` forward passes. The defaults (four
--- styles, twenty games) are already 1,200 decodes.

local duel = require("card_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "card_duel_tournament",
    version = "0.1.0",
    description = "Round robin between trained card duel NPC styles, reported as a win rate matrix",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `card_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            task = T.string:is_optional():describe(
                "JSON object carrying the same fields (used by the evalframe provider)"
            ),
            styles = T.array_of(T.string):is_optional():describe(
                "Styles to enter, a subset of card_duel.STYLES (default: four styles)"
            ),
            games_per_pair = T.number
                :is_optional()
                :describe("Games played per unordered pair (default: 20)"),
            seed = T.number:is_optional():describe("Base seed the game seeds derive from"),
            alias_prefix = T.string
                :is_optional()
                :describe("Card alias prefix per style (default: card_duel_npc_)"),
        }),
        result = T.string:describe(
            "One-line summary: tournament styles=.. games=.. top=.. winrate=.."
        ),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

--- Styles entered when the caller names none.
local DEFAULT_STYLES = { "timid", "bold", "aggressive", "defensive" }

local DEFAULT_GAMES_PER_PAIR = 20
local DEFAULT_SEED = 20260731
local DEFAULT_ALIAS_PREFIX = "card_duel_npc_"

--- Seed distance between two pairs. `games_per_pair` is rejected at or
--- above this value, so no two pairs can ever draw the same deal.
local PAIR_SEED_STRIDE = 1000

-- ─── Host surface guards ────────────────────────────────────────────

local function require_json()
    if
        type(alc) ~= "table"
        or type(alc.json_encode) ~= "function"
        or type(alc.json_decode) ~= "function"
    then
        error("card_duel_tournament: alc.json_encode / alc.json_decode are required")
    end
end

-- ─── Request parsing ────────────────────────────────────────────────

--- Read an optional numeric setting.
---
--- A field the caller omitted falls back to `default`, but a field that
--- is present and not a number fails loudly: silently substituting the
--- default would run a different tournament from the one that was asked
--- for and report it under the caller's own parameters.
---@param name string Field name, for the message
---@param raw any
---@param default number
---@return number
local function optional_number(name, raw, default)
    if raw == nil then
        return default
    end
    local value = tonumber(raw)
    if value == nil then
        error(
            string.format(
                "card_duel_tournament: %s must be a number, got '%s'",
                name,
                tostring(raw)
            )
        )
    end
    return value
end

--- Merge the JSON task payload over the plain ctx fields.
---
--- `alc_run` passes the settings directly while the evalframe provider
--- wraps the case input in `ctx.task`; the task wins so a scenario case
--- can override a strategy-wide option.
local function effective_ctx(ctx)
    local out = {}
    for k, v in pairs(ctx) do
        out[k] = v
    end
    local task = ctx.task
    if type(task) == "string" and #task > 0 then
        local decoded = alc.json_decode(task)
        if type(decoded) ~= "table" then
            error("card_duel_tournament: ctx.task must decode to a JSON object")
        end
        for k, v in pairs(decoded) do
            out[k] = v
        end
    end
    return out
end

--- Validate the requested styles against the canonical list.
---
--- A name outside `card_duel.STYLES` has no trained alias, and a
--- duplicate would enter a style against itself, so both fail here with
--- the valid names spelled out.
---@param raw any
---@return string[] styles
local function resolve_styles(raw)
    if raw == nil then
        raw = DEFAULT_STYLES
    end
    if type(raw) ~= "table" then
        error("card_duel_tournament: styles must be an array of names, got " .. type(raw))
    end
    local known = {}
    for _, name in ipairs(duel.STYLES) do
        known[name] = true
    end
    local seen, out = {}, {}
    for _, name in ipairs(raw) do
        if type(name) ~= "string" or not known[name] then
            error(
                string.format(
                    "card_duel_tournament: unknown style %s (valid: %s)",
                    tostring(name),
                    table.concat(duel.STYLES, ", ")
                )
            )
        end
        if seen[name] then
            error(string.format("card_duel_tournament: style %q is listed twice", name))
        end
        seen[name] = true
        out[#out + 1] = name
    end
    if #out < 2 then
        error(string.format("card_duel_tournament: at least two styles are required, got %d", #out))
    end
    return out
end

--- Validate the games played per pair.
---
--- The upper bound is the seed stride: game seeds are derived as
--- `seed + pair_index * PAIR_SEED_STRIDE + game_index`, so a count at or
--- above the stride would make the tail of one pair replay the deals of
--- the next one and quietly correlate two rows of the matrix.
local function resolve_games_per_pair(raw)
    local games = math.floor(optional_number("games_per_pair", raw, DEFAULT_GAMES_PER_PAIR))
    if games <= 0 then
        error("card_duel_tournament: games_per_pair must be a positive integer, got " .. games)
    end
    if games >= PAIR_SEED_STRIDE then
        error(
            string.format(
                "card_duel_tournament: games_per_pair must stay below the pair seed stride of %d "
                    .. "so pairs cannot share a deal, got %d",
                PAIR_SEED_STRIDE,
                games
            )
        )
    end
    return games
end

-- ─── Aggregation ────────────────────────────────────────────────────

--- Fold one game into the per-style decode telemetry.
local function note_decision(telemetry, style, gated)
    local row = telemetry[style]
    if row == nil then
        row = { decisions = 0, gated = 0 }
        telemetry[style] = row
    end
    row.decisions = row.decisions + 1
    if gated then
        row.gated = row.gated + 1
    end
end

--- Fold the game records into the matrix, the per-style summary and the
--- one-line result.
---
--- Exposed as `M._aggregate` as a test seam: the package spec drives it
--- with a hand-written record list, which checks the arithmetic without
--- loading a model.
---
--- A record is `{ a, b, winner, margin_a }` where `winner` is the
--- `card_duel.winner` verdict for that game (`"p1"` is `a`) and
--- `margin_a` is `a`'s points minus `b`'s.
---@param styles string[]
---@param games_per_pair integer Denominator of the per-cell win rate
---@param records table[] One entry per game played
---@param telemetry table|nil `{ [style] = { decisions, gated } }`
---@return table folded `{ matrix, summary, result }`
local function aggregate(styles, games_per_pair, records, telemetry)
    if type(styles) ~= "table" or #styles < 2 then
        error("card_duel_tournament: aggregate needs at least two styles")
    end
    if type(records) ~= "table" then
        error("card_duel_tournament: aggregate needs a record array, got " .. type(records))
    end
    telemetry = telemetry or {}

    local matrix, totals = {}, {}
    for _, a in ipairs(styles) do
        matrix[a] = {}
        totals[a] = { games = 0, wins = 0, margin = 0 }
        for _, b in ipairs(styles) do
            if a ~= b then
                matrix[a][b] = { wins = 0, losses = 0, draws = 0, winrate = 0.0 }
            end
        end
    end

    for index, rec in ipairs(records) do
        local cell_a = matrix[rec.a] and matrix[rec.a][rec.b]
        local cell_b = matrix[rec.b] and matrix[rec.b][rec.a]
        if cell_a == nil or cell_b == nil then
            error(
                string.format(
                    "card_duel_tournament: record %d names a pair outside the style list (%s vs %s)",
                    index,
                    tostring(rec.a),
                    tostring(rec.b)
                )
            )
        end
        local margin = tonumber(rec.margin_a)
        if margin == nil then
            error(string.format("card_duel_tournament: record %d has no numeric margin_a", index))
        end

        if rec.winner == "p1" then
            cell_a.wins = cell_a.wins + 1
            cell_b.losses = cell_b.losses + 1
            totals[rec.a].wins = totals[rec.a].wins + 1
        elseif rec.winner == "p2" then
            cell_a.losses = cell_a.losses + 1
            cell_b.wins = cell_b.wins + 1
            totals[rec.b].wins = totals[rec.b].wins + 1
        elseif rec.winner == "draw" then
            cell_a.draws = cell_a.draws + 1
            cell_b.draws = cell_b.draws + 1
        else
            error(
                string.format(
                    "card_duel_tournament: record %d has winner %s (expected p1 / p2 / draw)",
                    index,
                    tostring(rec.winner)
                )
            )
        end

        totals[rec.a].games = totals[rec.a].games + 1
        totals[rec.a].margin = totals[rec.a].margin + margin
        totals[rec.b].games = totals[rec.b].games + 1
        totals[rec.b].margin = totals[rec.b].margin - margin
    end

    for _, a in ipairs(styles) do
        for _, b in ipairs(styles) do
            if a ~= b then
                matrix[a][b].winrate = matrix[a][b].wins / games_per_pair
            end
        end
    end

    local summary = {}
    for _, style in ipairs(styles) do
        local t = totals[style]
        local tel = telemetry[style] or {}
        local decisions = tonumber(tel.decisions) or 0
        local gated = tonumber(tel.gated) or 0
        summary[style] = {
            total_winrate = t.games > 0 and t.wins / t.games or 0.0,
            gated_rate = decisions > 0 and gated / decisions or 0.0,
            avg_point_margin = t.games > 0 and t.margin / t.games or 0.0,
        }
    end

    -- Ties keep the first style in the requested order, so the line is
    -- reproducible for a reproducible tournament.
    local top = styles[1]
    for _, style in ipairs(styles) do
        if summary[style].total_winrate > summary[top].total_winrate then
            top = style
        end
    end

    return {
        matrix = matrix,
        summary = summary,
        result = string.format(
            "tournament styles=%d games=%d top=%s winrate=%.2f",
            #styles,
            #records,
            top,
            summary[top].total_winrate
        ),
    }
end

M._aggregate = aggregate

-- ─── Match play ─────────────────────────────────────────────────────

--- Ask the NPC package for one move.
---
--- The answer is the flat `action=.. legal=.. raw_legal=.. gated=..`
--- line the NPC returns. Both fields this reads are required: a missing
--- action would otherwise become a fallback move that scores a policy
--- nobody trained, and a missing gate flag would be folded into `false`
--- — the best possible value — which would report a healthy
--- `gated_rate` for an NPC whose telemetry stopped arriving.
---@param npc table `card_duel_npc` module
---@param alias string Card alias to decode with
---@param state table Per-player state
---@return integer rank
---@return boolean gated
local function decide(npc, alias, state)
    local out = npc.run({
        task = alc.json_encode({ mode = "decide", state = state }),
        card_alias = alias,
    })
    local text = type(out) == "table" and out.result or out
    if type(text) ~= "string" then
        error("card_duel_tournament: NPC answer must be a string, got " .. type(text))
    end
    local rank = tonumber(text:match("action=(%d+)"))
    if rank == nil then
        error(string.format("card_duel_tournament: NPC answer %q carries no action", text))
    end
    local gated = text:match("gated=(%a+)")
    if gated == nil then
        error(string.format("card_duel_tournament: NPC answer %q carries no gated flag", text))
    end
    return rank, gated == "true"
end

--- Play one game between two styles and return its record.
local function play_game(npc, alias_prefix, style_a, style_b, seed, telemetry)
    local g = duel.new_game(seed)
    while not duel.is_over(g) do
        local rank_a, gated_a = decide(npc, alias_prefix .. style_a, g.p1)
        local rank_b, gated_b = decide(npc, alias_prefix .. style_b, g.p2)
        note_decision(telemetry, style_a, gated_a)
        note_decision(telemetry, style_b, gated_b)
        g = duel.apply(g, rank_a, rank_b)
    end
    local winner = duel.winner(g)
    if winner == nil then
        -- Unreachable while `is_over` guards the loop. Kept loud so a
        -- future rules change surfaces here instead of being folded
        -- into the draw column.
        error("card_duel_tournament: the game loop left an unfinished game")
    end
    return {
        a = style_a,
        b = style_b,
        winner = winner,
        margin_a = g.p1.my_points - g.p2.my_points,
    }
end

-- ─── Strategy entry ─────────────────────────────────────────────────

---@param ctx table `{ styles?, games_per_pair?, seed?, alias_prefix?, task? }`
---@return table result `{ matrix, summary, styles, games_per_pair, seed, result }`
function M.run(ctx)
    require_json()
    local req = effective_ctx(ctx or {})

    local styles = resolve_styles(req.styles)
    local games_per_pair = resolve_games_per_pair(req.games_per_pair)
    local seed = math.floor(optional_number("seed", req.seed, DEFAULT_SEED))
    local alias_prefix = req.alias_prefix or DEFAULT_ALIAS_PREFIX
    if type(alias_prefix) ~= "string" then
        error("card_duel_tournament: alias_prefix must be a string, got " .. type(alias_prefix))
    end

    local npc = require("card_duel_npc")

    local records, telemetry = {}, {}
    local pair_index = 0
    for i = 1, #styles - 1 do
        for j = i + 1, #styles do
            pair_index = pair_index + 1
            for game_index = 1, games_per_pair do
                local game_seed = seed + pair_index * PAIR_SEED_STRIDE + game_index
                records[#records + 1] =
                    play_game(npc, alias_prefix, styles[i], styles[j], game_seed, telemetry)
            end
        end
    end

    local folded = aggregate(styles, games_per_pair, records, telemetry)
    return {
        matrix = folded.matrix,
        summary = folded.summary,
        styles = styles,
        games_per_pair = games_per_pair,
        seed = seed,
        result = folded.result,
    }
end

return M
