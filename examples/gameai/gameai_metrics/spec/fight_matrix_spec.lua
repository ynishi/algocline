-- gameai_metrics/spec/fight_matrix_spec.lua
--
-- Package-level spec for `fight_matrix`. The runner composes one
-- `level` call per boss row, so the spec replaces `level` with a fake
-- before requiring the module under test and asserts on (a) what the
-- runner asked `level` for and (b) how it folded the answers into a
-- matrix. No Card, no model and no game loop are involved.
--
-- ## Ordering constraint
--
-- `fight_matrix` requires `gameai_metrics.level` at load time, so the
-- fake has to sit in `package.loaded` *before* the
-- `require("gameai_metrics.fight_matrix")` below — the same constraint
-- level_spec observes when it seeds `alc.*` ahead of its own require.
-- Every spec file runs in a fresh VM (see `pkg/test_run.rs`), so the
-- substitution cannot leak into level_spec.
--
-- `alc.card` / `alc.nn.card.load_handle` / `alc.json_encode` /
-- `alc.json_decode` are stubbed for the same reason audit_matrix_spec
-- stubs them: the boss axis is resolved by alias or by manifest, and
-- `save()` round-trips the report through JSON.

local describe, it, expect = lust.describe, lust.it, lust.expect

alc = alc or {}

-- ─── Minimal JSON encoder / decoder ─────────────────────────────────
--
-- Same subset as audit_matrix_spec.lua: numbers, strings (with `\"` /
-- `\\` escapes), booleans, arrays (dense 1..n) and objects with keys
-- sorted for determinism. Enough to write a collection fixture and to
-- read a saved report back.

local function encode(value)
    local kind = type(value)
    if value == nil then
        return "null"
    end
    if kind == "boolean" or kind == "number" then
        return tostring(value)
    end
    if kind == "string" then
        return string.format("%q", value)
    end
    if kind ~= "table" then
        error("spec json_encode: unsupported type " .. kind, 0)
    end
    local n = #value
    if n > 0 then
        local total = 0
        for _ in pairs(value) do
            total = total + 1
        end
        if total == n then
            local items = {}
            for index = 1, n do
                items[index] = encode(value[index])
            end
            return "[" .. table.concat(items, ",") .. "]"
        end
    end
    local keys = {}
    for key in pairs(value) do
        keys[#keys + 1] = key
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    local fields = {}
    for _, key in ipairs(keys) do
        fields[#fields + 1] = string.format("%q", tostring(key)) .. ":" .. encode(value[key])
    end
    return "{" .. table.concat(fields, ",") .. "}"
end

local function json_decode(text)
    local pos = 1
    local function skip_ws()
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == " " or ch == "\n" or ch == "\r" or ch == "\t" then
                pos = pos + 1
            else
                return
            end
        end
    end
    local parse_value
    local function parse_string()
        assert(text:sub(pos, pos) == '"', "spec json_decode: expected string at " .. pos)
        pos = pos + 1
        local start = pos
        local parts = {}
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == '"' then
                parts[#parts + 1] = text:sub(start, pos - 1)
                pos = pos + 1
                return table.concat(parts)
            elseif ch == "\\" then
                parts[#parts + 1] = text:sub(start, pos - 1)
                local esc = text:sub(pos + 1, pos + 1)
                if esc == '"' or esc == "\\" or esc == "/" then
                    parts[#parts + 1] = esc
                elseif esc == "n" then
                    parts[#parts + 1] = "\n"
                elseif esc == "t" then
                    parts[#parts + 1] = "\t"
                elseif esc == "r" then
                    parts[#parts + 1] = "\r"
                else
                    error("spec json_decode: unsupported escape \\" .. esc, 0)
                end
                pos = pos + 2
                start = pos
            else
                pos = pos + 1
            end
        end
        error("spec json_decode: unterminated string", 0)
    end
    local function parse_number()
        local start = pos
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch:match("[%-%+%d%.eE]") then
                pos = pos + 1
            else
                break
            end
        end
        return tonumber(text:sub(start, pos - 1))
    end
    local function parse_array()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "]" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            out[#out + 1] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "]" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or ] at " .. pos, 0)
            end
        end
    end
    local function parse_object()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "}" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            local key = parse_string()
            skip_ws()
            assert(text:sub(pos, pos) == ":", "spec json_decode: expected : at " .. pos)
            pos = pos + 1
            skip_ws()
            out[key] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "}" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or } at " .. pos, 0)
            end
        end
    end
    parse_value = function()
        skip_ws()
        local ch = text:sub(pos, pos)
        if ch == "{" then
            return parse_object()
        end
        if ch == "[" then
            return parse_array()
        end
        if ch == '"' then
            return parse_string()
        end
        if ch == "t" and text:sub(pos, pos + 3) == "true" then
            pos = pos + 4
            return true
        end
        if ch == "f" and text:sub(pos, pos + 4) == "false" then
            pos = pos + 5
            return false
        end
        if ch == "n" and text:sub(pos, pos + 3) == "null" then
            pos = pos + 4
            return nil
        end
        return parse_number()
    end
    skip_ws()
    return parse_value()
end

alc.json_encode = encode
alc.json_decode = json_decode

-- ─── Card store stubs ───────────────────────────────────────────────
--
-- `make_boss(alias)` returns a marker table so the fake `level` can name
-- the row it is being called for. `load_handle` is keyed on `card_id`
-- (`"card-" .. alias`), which is what the collection-path branch feeds
-- it straight from the manifest.

local HANDLES_BY_ALIAS = {}
local HANDLES_BY_CARD_ID = {}
local LOAD_CALLS = {}

local function make_boss(alias)
    local handle = { alias = alias }
    HANDLES_BY_ALIAS[alias] = handle
    HANDLES_BY_CARD_ID["card-" .. alias] = handle
    return handle
end

local function install_card_stubs()
    alc.card = {
        get_by_alias = function(alias)
            if HANDLES_BY_ALIAS[alias] == nil then
                return nil
            end
            return { card_id = "card-" .. alias }
        end,
    }
    alc.nn = alc.nn or {}
    alc.nn.card = {
        load_handle = function(card_id)
            LOAD_CALLS[#LOAD_CALLS + 1] = card_id
            return HANDLES_BY_CARD_ID[card_id]
        end,
    }
end

install_card_stubs()

-- ─── Fake `level` ───────────────────────────────────────────────────
--
-- Records every call verbatim (so the spec can assert seat / style /
-- temperature / opponents forwarding) and answers with one canned
-- `per_opponent` row per name on `opts.opponents`. `OMIT_PLAYER` drops a
-- name from the answer, which is how the "level broke the contract"
-- branch is reached without a real metric.

local LEVEL_CALLS = {}
local CELL_WIN_RATE = {}
local OMIT_PLAYER = nil

local function cell_key(boss_alias, player)
    return tostring(boss_alias) .. "|" .. tostring(player)
end

local function set_win_rate(boss_alias, player, win_rate)
    CELL_WIN_RATE[cell_key(boss_alias, player)] = win_rate
end

--- One canned cell. Every field `level.per_opponent` carries is present
--- so the runner's verbatim transcription is observable field by field.
---
--- `per_game` mirrors what the real `level` does with the flag: it adds
--- a `games` array to the cell and changes nothing else. Two records are
--- enough — the runner never reads inside them, so the assertion this
--- fake supports is "the array rode along", not its length.
---
--- `per_move` nests one level deeper, again mirroring `level`: a `moves`
--- array inside each per-game record. Two moves per record, for the same
--- reason.
local function make_cell(boss_alias, player, n_games, per_game, per_move)
    local win_rate = CELL_WIN_RATE[cell_key(boss_alias, player)] or 0.5
    local cell = {
        win_rate = win_rate,
        ci_lower = win_rate - 0.05,
        ci_upper = win_rate + 0.05,
        n_games = n_games,
        game_length_mean = 6.0 + win_rate,
        final_hp_margin_mean = 10.0 * win_rate - 5.0,
    }
    if per_game then
        cell.games = {
            { outcome = 1.0, game_length = 7, final_hp_margin = 3 },
            { outcome = 0.0, game_length = 9, final_hp_margin = -2 },
        }
        if per_move then
            cell.games[1].moves = {
                {
                    turn = 1,
                    mode = 0,
                    intent = "-",
                    boss_action = "c",
                    player_action = "p",
                    boss_hp = 45,
                    player_hp = 43,
                },
                {
                    turn = 2,
                    mode = 0,
                    intent = "f",
                    boss_action = "f",
                    player_action = "b",
                    boss_hp = 41,
                    player_hp = 40,
                },
            }
        end
    end
    return cell
end

local function fake_level(card, opponent, n_games, seed, opts)
    opts = opts or {}
    LEVEL_CALLS[#LEVEL_CALLS + 1] = {
        card = card,
        opponent = opponent,
        n_games = n_games,
        seed = seed,
        opts = opts,
    }
    local boss_alias = type(card) == "table" and card.alias or nil
    local per_opponent = {}
    local wins, total = 0.0, 0
    for _, player in ipairs(opts.opponents or {}) do
        if player ~= OMIT_PLAYER then
            local cell = make_cell(boss_alias, player, n_games, opts.per_game, opts.per_move)
            per_opponent[player] = cell
            wins = wins + cell.win_rate * n_games
            total = total + n_games
        end
    end
    return {
        win_rate = total > 0 and wins / total or 0.0,
        ci_lower = 0.0,
        ci_upper = 1.0,
        wins = wins,
        n_games = total,
        win_rate_min = 0.0,
        game_length_mean = 6.0,
        final_hp_margin_mean = 0.0,
        per_opponent = per_opponent,
    }
end

package.loaded["gameai_metrics.level"] = fake_level

local fight_matrix = require("gameai_metrics.fight_matrix")

local BASE_STYLE = "guardian"

local function reset_all()
    HANDLES_BY_ALIAS = {}
    HANDLES_BY_CARD_ID = {}
    LOAD_CALLS = {}
    LEVEL_CALLS = {}
    CELL_WIN_RATE = {}
    OMIT_PLAYER = nil
    -- Re-install so the closures see the fresh tables.
    install_card_stubs()
end

local _tmp_counter = 0
local function tmp_path()
    _tmp_counter = _tmp_counter + 1
    return os.tmpname() .. "-fight-" .. tostring(_tmp_counter)
end

local function write_collection(path, entries)
    local body = alc.json_encode({
        schema_version = 1,
        style = BASE_STYLE,
        policy = "first_writer_wins",
        entries = entries,
    })
    local f = io.open(path, "w")
    f:write(body)
    f:close()
end

-- ─── Specs: option validation ───────────────────────────────────────

describe("gameai_metrics.fight_matrix.new — options", function()
    it("refuses opts that are not a table", function()
        local ok, err = pcall(fight_matrix.new, "nope")
        expect(ok).to.equal(false)
        expect(err:find("opts must be a table") ~= nil).to.equal(true)
    end)

    it("refuses both collection_path and bosses at once", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            collection_path = "/tmp/x.json",
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("not both") ~= nil).to.equal(true)
    end)

    it("refuses neither collection_path nor bosses", function()
        reset_all()
        local ok, err = pcall(fight_matrix.new, {
            players = { "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("required") ~= nil).to.equal(true)
    end)

    it("refuses a missing style", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1" },
        })
        expect(ok).to.equal(false)
        expect(err:find("style") ~= nil).to.equal(true)
    end)

    it("refuses missing players", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("players") ~= nil).to.equal(true)
    end)

    it("refuses an empty players array", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = {},
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("non%-empty array") ~= nil).to.equal(true)
    end)

    it("refuses a duplicated player alias with a fight_matrix message", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1", "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("twice") ~= nil).to.equal(true)
        -- The check has to fire here rather than inside level, whose
        -- message would name a pool the caller never spelled.
        expect(err:find("fight_matrix") ~= nil).to.equal(true)
        expect(err:find("level:") == nil).to.equal(true)
    end)

    it("refuses a non-string player entry", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1", 42 },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("players%[2%]") ~= nil).to.equal(true)
    end)

    it("refuses a non-number temperature", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            temperature = false,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
        expect(err:find("fight_matrix") ~= nil).to.equal(true)
    end)

    it("refuses a zero or negative temperature", function()
        reset_all()
        make_boss("b1")
        for _, bad in ipairs({ 0, -1.5 }) do
            local ok, err = pcall(fight_matrix.new, {
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                temperature = bad,
            })
            expect(ok).to.equal(false)
            expect(err:find("temperature") ~= nil).to.equal(true)
        end
    end)

    it("refuses a non-finite temperature", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            temperature = math.huge,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)

    it("refuses a non-boolean per_game", function()
        reset_all()
        make_boss("b1")
        for _, bad in ipairs({ "true", "false", 1, 0 }) do
            local ok, err = pcall(fight_matrix.new, {
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                per_game = bad,
            })
            expect(ok).to.equal(false)
            expect(err:find("per_game") ~= nil).to.equal(true)
            -- The check belongs to this layer, not to level: a caller
            -- who mistyped the flag named *this* runner.
            expect(err:find("fight_matrix") ~= nil).to.equal(true)
        end
    end)

    it("refuses a non-boolean per_move", function()
        reset_all()
        make_boss("b1")
        for _, bad in ipairs({ "true", "false", 1, 0 }) do
            local ok, err = pcall(fight_matrix.new, {
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                per_game = true,
                per_move = bad,
            })
            expect(ok).to.equal(false)
            expect(err:find("per_move") ~= nil).to.equal(true)
            expect(err:find("fight_matrix") ~= nil).to.equal(true)
        end
    end)

    it("refuses per_move without per_game", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            per_move = true,
        })
        expect(ok).to.equal(false)
        -- Named here rather than left to `level`: the caller called this
        -- runner, and the refusal has to land before the first row of
        -- fights rather than after it.
        expect(err:find("per_move") ~= nil).to.equal(true)
        expect(err:find("per_game") ~= nil).to.equal(true)
        expect(err:find("fight_matrix") ~= nil).to.equal(true)
        local off = pcall(fight_matrix.new, {
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            per_game = false,
            per_move = true,
        })
        expect(off).to.equal(false)
    end)

    it("refuses an unbound boss alias", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1", "ghost" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("ghost") ~= nil).to.equal(true)
    end)

    it("refuses a duplicated boss alias", function()
        reset_all()
        make_boss("b1")
        local ok, err = pcall(fight_matrix.new, {
            bosses = { "b1", "b1" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("twice") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: run() matrix shape ──────────────────────────────────────

describe("gameai_metrics.fight_matrix:run — matrix shape", function()
    local BOSSES = { "boss_weak", "boss_mid", "boss_strong" }
    local PLAYERS = { "player_sentinel", "player_rusher" }

    local function three_by_two()
        reset_all()
        for _, alias in ipairs(BOSSES) do
            make_boss(alias)
        end
        set_win_rate("boss_weak", "player_sentinel", 0.21)
        set_win_rate("boss_weak", "player_rusher", 0.33)
        set_win_rate("boss_mid", "player_sentinel", 0.62)
        set_win_rate("boss_mid", "player_rusher", 0.58)
        set_win_rate("boss_strong", "player_sentinel", 0.88)
        set_win_rate("boss_strong", "player_rusher", 0.79)
        return fight_matrix.new({
            bosses = BOSSES,
            players = PLAYERS,
            style = BASE_STYLE,
            n_games = 50,
            seed = 7,
            temperature = 0.8,
        })
    end

    it("fills every cell of the N x M matrix", function()
        local report = three_by_two():run()
        for _, boss in ipairs(BOSSES) do
            local row = report.matrix[boss]
            expect(type(row)).to.equal("table")
            for _, player in ipairs(PLAYERS) do
                local cell = row[player]
                expect(type(cell)).to.equal("table")
                expect(type(cell.win_rate)).to.equal("number")
                expect(type(cell.ci_lower)).to.equal("number")
                expect(type(cell.ci_upper)).to.equal("number")
                expect(cell.n_games).to.equal(50)
                expect(type(cell.game_length_mean)).to.equal("number")
                expect(type(cell.final_hp_margin_mean)).to.equal("number")
            end
        end
    end)

    it("transcribes each per_opponent cell to its own boss row", function()
        local report = three_by_two():run()
        expect(report.matrix.boss_weak.player_sentinel.win_rate).to.equal(0.21)
        expect(report.matrix.boss_weak.player_rusher.win_rate).to.equal(0.33)
        expect(report.matrix.boss_mid.player_sentinel.win_rate).to.equal(0.62)
        expect(report.matrix.boss_strong.player_rusher.win_rate).to.equal(0.79)
        -- Derived fields ride along verbatim.
        expect(report.matrix.boss_strong.player_sentinel.ci_upper).to.equal(0.88 + 0.05)
        expect(report.matrix.boss_strong.player_sentinel.final_hp_margin_mean).to.equal(
            10.0 * 0.88 - 5.0
        )
    end)

    it("carries a per_game cell's games array through verbatim", function()
        reset_all()
        make_boss("b1")
        set_win_rate("b1", "p1", 0.42)
        local report = fight_matrix
            .new({
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                n_games = 2,
                per_game = true,
            })
            :run()
        local cell = report.matrix.b1.p1
        -- The runner transcribes the cell whole, so a field `level`
        -- added under the flag needs no code here to arrive.
        expect(type(cell.games)).to.equal("table")
        expect(#cell.games).to.equal(2)
        expect(cell.games[1].outcome).to.equal(1.0)
        expect(cell.games[2].game_length).to.equal(9)
        expect(cell.games[2].final_hp_margin).to.equal(-2)
        expect(cell.win_rate).to.equal(0.42)
    end)

    it("carries a per_move game's moves array through verbatim", function()
        reset_all()
        make_boss("b1")
        set_win_rate("b1", "p1", 0.42)
        local report = fight_matrix
            .new({
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                n_games = 2,
                per_game = true,
                per_move = true,
            })
            :run()
        local cell = report.matrix.b1.p1
        -- Two levels of nesting arrive through the same verbatim cell
        -- transcription: this runner reads neither array.
        local moves = cell.games[1].moves
        expect(type(moves)).to.equal("table")
        expect(#moves).to.equal(2)
        expect(moves[1].intent).to.equal("-")
        expect(moves[2].boss_action).to.equal("f")
        expect(moves[2].player_hp).to.equal(40)
        expect(cell.games[2].moves).to.equal(nil)
    end)

    it("leaves the games records without a moves array by default", function()
        reset_all()
        make_boss("b1")
        local report = fight_matrix
            .new({
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                per_game = true,
            })
            :run()
        expect(report.matrix.b1.p1.games[1].moves).to.equal(nil)
    end)

    it("leaves the cells without a games array by default", function()
        reset_all()
        make_boss("b1")
        local report = fight_matrix
            .new({
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
            })
            :run()
        expect(report.matrix.b1.p1.games).to.equal(nil)
    end)

    it("reports meta with the axes in the order they were given", function()
        local report = three_by_two():run()
        expect(report.meta.n_games).to.equal(50)
        expect(report.meta.seed).to.equal(7)
        expect(report.meta.style).to.equal(BASE_STYLE)
        expect(report.meta.temperature).to.equal(0.8)
        expect(table.concat(report.meta.bosses, ",")).to.equal(table.concat(BOSSES, ","))
        expect(table.concat(report.meta.players, ",")).to.equal(table.concat(PLAYERS, ","))
        expect(report.meta.collection_path).to.equal(nil)
    end)

    it("records both record flags in meta, so a saved report is self-describing", function()
        reset_all()
        make_boss("b1")
        local off =
            fight_matrix.new({ bosses = { "b1" }, players = { "p1" }, style = BASE_STYLE }):run()
        expect(off.meta.per_game).to.equal(false)
        expect(off.meta.per_move).to.equal(false)
        local on = fight_matrix
            .new({
                bosses = { "b1" },
                players = { "p1" },
                style = BASE_STYLE,
                per_game = true,
                per_move = true,
            })
            :run()
        expect(on.meta.per_game).to.equal(true)
        -- Without this a reader of the JSON cannot tell a run made
        -- without transcripts from one whose transcripts went missing.
        expect(on.meta.per_move).to.equal(true)
    end)

    it("defaults n_games=200 / seed=0 / temperature=1.0", function()
        reset_all()
        make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        local report = fight:run()
        expect(report.meta.n_games).to.equal(200)
        expect(report.meta.seed).to.equal(0)
        expect(report.meta.temperature).to.equal(1.0)
        expect(fight_matrix.DEFAULT_N_GAMES).to.equal(200)
        expect(fight_matrix.DEFAULT_SEED).to.equal(0)
        expect(fight_matrix.DEFAULT_TEMPERATURE).to.equal(1.0)
    end)

    it("returns the same report from :report() as from :run()", function()
        local fight = three_by_two()
        local report = fight:run()
        expect(fight:report()).to.equal(report)
    end)

    it("raises when :report() is called before :run()", function()
        local fight = three_by_two()
        local ok, err = pcall(fight.report, fight)
        expect(ok).to.equal(false)
        expect(err:find("no report yet") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: what the runner asks `level` for ────────────────────────

describe("gameai_metrics.fight_matrix:run — level invocation", function()
    it("makes exactly one level call per boss row", function()
        reset_all()
        make_boss("b1")
        make_boss("b2")
        local fight = fight_matrix.new({
            bosses = { "b1", "b2" },
            players = { "p1", "p2", "p3" },
            style = BASE_STYLE,
        })
        fight:run()
        expect(#LEVEL_CALLS).to.equal(2)
    end)

    it("passes the loaded boss handle, seat=boss, style and temperature", function()
        reset_all()
        local handle = make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            n_games = 25,
            seed = 11,
            temperature = 1.5,
        })
        fight:run()
        local call = LEVEL_CALLS[1]
        expect(call.card).to.equal(handle)
        expect(call.opponent).to.equal(nil)
        expect(call.n_games).to.equal(25)
        expect(call.seed).to.equal(11)
        expect(call.opts.seat).to.equal("boss")
        expect(call.opts.style).to.equal(BASE_STYLE)
        expect(call.opts.temperature).to.equal(1.5)
    end)

    it("forwards the whole player pool as opts.opponents", function()
        reset_all()
        make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1", "p2", "p3" },
            style = BASE_STYLE,
        })
        fight:run()
        expect(table.concat(LEVEL_CALLS[1].opts.opponents, ",")).to.equal("p1,p2,p3")
    end)

    it("forwards opts.per_game to every level call", function()
        reset_all()
        make_boss("b1")
        make_boss("b2")
        local fight = fight_matrix.new({
            bosses = { "b1", "b2" },
            players = { "p1" },
            style = BASE_STYLE,
            per_game = true,
        })
        fight:run()
        expect(#LEVEL_CALLS).to.equal(2)
        expect(LEVEL_CALLS[1].opts.per_game).to.equal(true)
        expect(LEVEL_CALLS[2].opts.per_game).to.equal(true)
    end)

    it("forwards per_game = false when the caller names none", function()
        reset_all()
        make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        fight:run()
        expect(LEVEL_CALLS[1].opts.per_game).to.equal(false)
    end)

    it("forwards opts.per_move to every level call", function()
        reset_all()
        make_boss("b1")
        make_boss("b2")
        local fight = fight_matrix.new({
            bosses = { "b1", "b2" },
            players = { "p1" },
            style = BASE_STYLE,
            per_game = true,
            per_move = true,
        })
        fight:run()
        expect(#LEVEL_CALLS).to.equal(2)
        -- A flag decoded here but dropped on the way down would leave
        -- the report shape unchanged and raise nowhere.
        expect(LEVEL_CALLS[1].opts.per_move).to.equal(true)
        expect(LEVEL_CALLS[2].opts.per_move).to.equal(true)
    end)

    it("forwards per_move = false when the caller names none", function()
        reset_all()
        make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            per_game = true,
        })
        fight:run()
        expect(LEVEL_CALLS[1].opts.per_move).to.equal(false)
    end)

    it("raises when level answers without a cell for a named player", function()
        reset_all()
        make_boss("b1")
        local fight = fight_matrix.new({
            bosses = { "b1" },
            players = { "p1", "p2" },
            style = BASE_STYLE,
        })
        OMIT_PLAYER = "p2"
        local ok, err = pcall(fight.run, fight)
        expect(ok).to.equal(false)
        expect(err:find("p2") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: collection_path branch ──────────────────────────────────

describe("gameai_metrics.fight_matrix — collection_path", function()
    it("loads each entry by its stored card_id and reports the path", function()
        reset_all()
        make_boss("guardian_duel_npc_weak")
        make_boss("guardian_duel_npc_mid")
        set_win_rate("guardian_duel_npc_mid", "sentinel", 0.71)
        local path = tmp_path()
        write_collection(path, {
            {
                label = "weak",
                step = 60,
                alias = "guardian_duel_npc_weak",
                card_id = "card-guardian_duel_npc_weak",
            },
            {
                label = "mid",
                step = 180,
                alias = "guardian_duel_npc_mid",
                card_id = "card-guardian_duel_npc_mid",
            },
        })
        local fight = fight_matrix.new({
            collection_path = path,
            players = { "sentinel" },
            style = BASE_STYLE,
            n_games = 10,
        })
        local report = fight:run()
        os.remove(path)
        -- The card_id goes to load_handle directly; the alias branch of
        -- alc.card.get_by_alias is never consulted on this path.
        expect(table.concat(LOAD_CALLS, ",")).to.equal(
            "card-guardian_duel_npc_weak,card-guardian_duel_npc_mid"
        )
        expect(report.matrix.guardian_duel_npc_mid.sentinel.win_rate).to.equal(0.71)
        expect(report.meta.collection_path).to.equal(path)
        expect(table.concat(report.meta.bosses, ",")).to.equal(
            "guardian_duel_npc_weak,guardian_duel_npc_mid"
        )
    end)

    it("raises loudly when an entry has no card_id", function()
        reset_all()
        make_boss("weak")
        make_boss("mid")
        local path = tmp_path()
        write_collection(path, {
            { label = "weak", alias = "weak", card_id = "card-weak" },
            { label = "mid", alias = "mid" },
        })
        local ok, err = pcall(fight_matrix.new, {
            collection_path = path,
            players = { "p1" },
            style = BASE_STYLE,
        })
        os.remove(path)
        expect(ok).to.equal(false)
        expect(err:find("card_id") ~= nil).to.equal(true)
    end)

    it("raises loudly when a card_id vanished between new() and run()", function()
        reset_all()
        make_boss("weak")
        make_boss("mid")
        local path = tmp_path()
        write_collection(path, {
            { label = "weak", alias = "weak", card_id = "card-weak" },
            { label = "mid", alias = "mid", card_id = "card-mid" },
        })
        local fight = fight_matrix.new({
            collection_path = path,
            players = { "p1" },
            style = BASE_STYLE,
        })
        os.remove(path)
        HANDLES_BY_CARD_ID["card-mid"] = nil
        local ok, err = pcall(fight.run, fight)
        expect(ok).to.equal(false)
        expect(err:find('alias "mid"') ~= nil).to.equal(true)
        -- No partial matrix: the row that did resolve is not published.
        local reported, report_err = pcall(fight.report, fight)
        expect(reported).to.equal(false)
        expect(report_err:find("no report yet") ~= nil).to.equal(true)
    end)

    it("propagates a load_handle raise", function()
        reset_all()
        make_boss("weak")
        local fight = fight_matrix.new({
            bosses = { "weak" },
            players = { "p1" },
            style = BASE_STYLE,
        })
        alc.nn.card.load_handle = function(_)
            error("engine: candle backend fell over", 0)
        end
        local ok, err = pcall(fight.run, fight)
        expect(ok).to.equal(false)
        expect(err:find("load_handle") ~= nil).to.equal(true)
    end)

    it("raises loudly when the manifest has zero entries", function()
        reset_all()
        local path = tmp_path()
        write_collection(path, {})
        local ok, err = pcall(fight_matrix.new, {
            collection_path = path,
            players = { "p1" },
            style = BASE_STYLE,
        })
        os.remove(path)
        expect(ok).to.equal(false)
        expect(err:find("zero entries") ~= nil).to.equal(true)
    end)

    it("raises loudly when the file cannot be opened", function()
        reset_all()
        local ok, err = pcall(fight_matrix.new, {
            collection_path = "/no/such/dir/fight-missing.json",
            players = { "p1" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("cannot open") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: save() and _fs.ensure_parent_dir integration ────────────

describe("gameai_metrics.fight_matrix:save", function()
    local function tiny_fight()
        reset_all()
        make_boss("b1")
        set_win_rate("b1", "p1", 0.64)
        return fight_matrix.new({
            bosses = { "b1" },
            players = { "p1" },
            style = BASE_STYLE,
            n_games = 10,
            seed = 3,
        })
    end

    it("refuses save() before run()", function()
        local fight = tiny_fight()
        local ok, err = pcall(fight.save, fight, "/tmp/never.json")
        expect(ok).to.equal(false)
        expect(err:find("no report") ~= nil).to.equal(true)
    end)

    it("refuses a path that is not a string", function()
        local fight = tiny_fight()
        fight:run()
        local ok, err = pcall(fight.save, fight, 42)
        expect(ok).to.equal(false)
        expect(err:find("non%-empty string") ~= nil).to.equal(true)
    end)

    it("writes JSON that round-trips through json_decode", function()
        local fight = tiny_fight()
        fight:run()
        local path = tmp_path()
        fight:save(path)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(parsed.meta.style).to.equal(BASE_STYLE)
        expect(parsed.meta.n_games).to.equal(10)
        expect(parsed.meta.seed).to.equal(3)
        expect(parsed.meta.temperature).to.equal(1.0)
        expect(parsed.matrix.b1.p1.win_rate).to.equal(0.64)
        expect(parsed.matrix.b1.p1.n_games).to.equal(10)
        expect(type(parsed.matrix.b1.p1.game_length_mean)).to.equal("number")
    end)

    it("creates missing parent directories via _fs.ensure_parent_dir", function()
        local fight = tiny_fight()
        fight:run()
        -- Reserve a unique tmp path, remove the file, and use its
        -- basename as the root of a nested directory that does not
        -- exist yet — mirrors the audit_matrix_spec pattern.
        local base = os.tmpname()
        os.remove(base)
        local dir = base .. "-fight-parent"
        local path = dir .. "/nested/fight.json"
        fight:save(path)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        os.remove(dir .. "/nested")
        os.remove(dir)
        local parsed = json_decode(body)
        expect(parsed.matrix.b1.p1.win_rate).to.equal(0.64)
    end)
end)
