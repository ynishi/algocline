-- card_duel_tournament/spec/card_duel_tournament_spec.lua
--
-- Package-level spec for the tournament runner. Run with
-- `alc_pkg_test pkg="card_duel_tournament"` after `alc_pkg_link` has
-- registered `card_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- The suite never loads a model: `card_duel_npc` is replaced through
-- `package.preload` by a stub that answers from the rules alone, so the
-- aggregation and the match loop are covered on the default feature set
-- while the decode path stays the NPC package's own concern.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── NPC stub ───────────────────────────────────────────────────────
--
-- Answers with the lowest legal rank, or the highest one when the alias
-- names the bold style, which makes the outcome of a timid-vs-bold pair
-- a fact about the rules rather than about a trained model. The gate
-- flag is reported as set for the timid alias only, so `gated_rate` has
-- two distinct values to check.

local npc_calls = 0

package.preload["card_duel_npc"] = function()
    local duel = require("card_duel")
    return {
        run = function(ctx)
            npc_calls = npc_calls + 1
            local req = alc.json_decode(ctx.task)
            local legal = duel.legal_actions(req.state)
            local alias = ctx.card_alias or ""
            local bold = alias:find("bold", 1, true) ~= nil
            local rank = bold and legal[#legal] or legal[1]
            return {
                result = string.format(
                    "action=%d legal=true raw_legal=true gated=%s",
                    rank,
                    tostring(not bold)
                ),
            }
        end,
    }
end

local tournament = require("card_duel_tournament")

-- ─── Hand-written records for the aggregation seam ──────────────────

local STYLES = { "timid", "bold" }
local GAMES_PER_PAIR = 4

--- Three losses and one draw for `timid` against `bold`.
local function sample_records()
    local records = {}
    for _ = 1, 3 do
        records[#records + 1] = { a = "timid", b = "bold", winner = "p2", margin_a = -2 }
    end
    records[#records + 1] = { a = "timid", b = "bold", winner = "draw", margin_a = 0 }
    return records
end

local function sample_telemetry()
    return {
        timid = { decisions = 20, gated = 5 },
        bold = { decisions = 20, gated = 0 },
    }
end

local function folded()
    return tournament._aggregate(STYLES, GAMES_PER_PAIR, sample_records(), sample_telemetry())
end

describe("card_duel_tournament._aggregate matrix", function()
    it("counts the games from the winner's point of view", function()
        local out = folded()
        expect(out.matrix.timid.bold.wins).to.equal(0)
        expect(out.matrix.timid.bold.losses).to.equal(3)
        expect(out.matrix.timid.bold.draws).to.equal(1)
        expect(out.matrix.bold.timid.wins).to.equal(3)
        expect(out.matrix.bold.timid.losses).to.equal(0)
        expect(out.matrix.bold.timid.draws).to.equal(1)
    end)

    it("divides the win rate by games_per_pair", function()
        local out = folded()
        expect(out.matrix.bold.timid.winrate).to.equal(0.75)
        expect(out.matrix.timid.bold.winrate).to.equal(0.0)
    end)

    it("leaves no self-pair cell", function()
        local out = folded()
        expect(out.matrix.timid.timid).to.equal(nil)
        expect(out.matrix.bold.bold).to.equal(nil)
    end)
end)

describe("card_duel_tournament._aggregate summary", function()
    it("averages the win rate over every game a style played", function()
        local out = folded()
        expect(out.summary.bold.total_winrate).to.equal(0.75)
        expect(out.summary.timid.total_winrate).to.equal(0.0)
    end)

    it("averages the point margin with the sign of each side", function()
        local out = folded()
        expect(out.summary.timid.avg_point_margin).to.equal(-1.5)
        expect(out.summary.bold.avg_point_margin).to.equal(1.5)
    end)

    it("reports the gate rate per style", function()
        local out = folded()
        expect(out.summary.timid.gated_rate).to.equal(0.25)
        expect(out.summary.bold.gated_rate).to.equal(0.0)
    end)

    it("reports a zero gate rate when no telemetry was collected", function()
        local out = tournament._aggregate(STYLES, GAMES_PER_PAIR, sample_records(), nil)
        expect(out.summary.timid.gated_rate).to.equal(0.0)
    end)
end)

describe("card_duel_tournament._aggregate result line", function()
    it("names the leader and its win rate", function()
        local out = folded()
        expect(out.result).to.equal("tournament styles=2 games=4 top=bold winrate=0.75")
    end)

    it("matches the shape the eval grader reads", function()
        local out = folded()
        expect(out.result:match("winrate=(%d+%.%d%d)")).to.equal("0.75")
        expect(out.result:match("^tournament styles=%d+ games=%d+ top=%a") ~= nil).to.equal(true)
    end)

    it("rejects a record naming a style outside the list", function()
        expect(function()
            tournament._aggregate(STYLES, GAMES_PER_PAIR, {
                { a = "timid", b = "mimic", winner = "p1", margin_a = 1 },
            }, nil)
        end).to.fail()
    end)

    it("rejects a record with an unknown winner", function()
        expect(function()
            tournament._aggregate(STYLES, GAMES_PER_PAIR, {
                { a = "timid", b = "bold", winner = "nobody", margin_a = 1 },
            }, nil)
        end).to.fail()
    end)
end)

describe("card_duel_tournament.run validation", function()
    it("rejects a style outside card_duel.STYLES", function()
        expect(function()
            tournament.run({ styles = { "timid", "reckless" }, games_per_pair = 1 })
        end).to.fail()
    end)

    it("rejects the same style twice", function()
        expect(function()
            tournament.run({ styles = { "timid", "timid" }, games_per_pair = 1 })
        end).to.fail()
    end)

    it("rejects a single-entry style list", function()
        expect(function()
            tournament.run({ styles = { "timid" }, games_per_pair = 1 })
        end).to.fail()
    end)

    it("rejects a non-positive games_per_pair", function()
        expect(function()
            tournament.run({ styles = { "timid", "bold" }, games_per_pair = 0 })
        end).to.fail()
    end)

    it("rejects a games_per_pair that is not a number", function()
        expect(function()
            tournament.run({ styles = { "timid", "bold" }, games_per_pair = "many" })
        end).to.fail()
    end)

    it("rejects a seed that is not a number", function()
        expect(function()
            tournament.run({ styles = { "timid", "bold" }, games_per_pair = 1, seed = "later" })
        end).to.fail()
    end)

    it("rejects a games_per_pair at the pair seed stride", function()
        -- 1000 games per pair would make the tail of one pair replay the
        -- deals of the next, so the bound is refused rather than derived
        -- into a silent overlap.
        expect(function()
            tournament.run({ styles = { "timid", "bold" }, games_per_pair = 1000 })
        end).to.fail()
    end)

    it("rejects an NPC answer without the gate flag", function()
        local npc = require("card_duel_npc")
        local original = npc.run
        npc.run = function()
            return { result = "action=1 legal=true raw_legal=true" }
        end
        local ok = pcall(tournament.run, {
            styles = { "timid", "bold" },
            games_per_pair = 1,
            seed = 5,
        })
        npc.run = original
        expect(ok).to.equal(false)
    end)
end)

describe("card_duel_tournament.run against the stub NPC", function()
    it("plays games_per_pair games for the one pair", function()
        local out = tournament.run({
            styles = { "timid", "bold" },
            games_per_pair = 2,
            seed = 5,
        })
        local cell = out.matrix.timid.bold
        expect(cell.wins + cell.losses + cell.draws).to.equal(2)
        expect(out.games_per_pair).to.equal(2)
        expect(out.seed).to.equal(5)
    end)

    it("carries the stub gate flag into the summary", function()
        local out = tournament.run({
            styles = { "timid", "bold" },
            games_per_pair = 2,
            seed = 5,
        })
        expect(out.summary.timid.gated_rate).to.equal(1.0)
        expect(out.summary.bold.gated_rate).to.equal(0.0)
    end)

    it("is reproducible from the seed", function()
        local args = { styles = { "timid", "bold" }, games_per_pair = 3, seed = 11 }
        expect(tournament.run(args).result).to.equal(tournament.run(args).result)
    end)

    it("reads the settings from a JSON task payload", function()
        local out = tournament.run({
            task = '{"styles":["timid","bold"],"games_per_pair":2,"seed":5}',
        })
        expect(out.games_per_pair).to.equal(2)
        expect(out.seed).to.equal(5)
        expect(out.result:match("games=(%d+)")).to.equal("2")
    end)

    it("asks the NPC once per seat per round", function()
        local before = npc_calls
        tournament.run({ styles = { "timid", "bold" }, games_per_pair = 1, seed = 5 })
        expect(npc_calls - before).to.equal(2 * 5)
    end)
end)
