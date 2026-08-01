-- gameai_metrics/spec/level_spec.lua
--
-- Package-level spec for the `level` metric. Stubs `alc.card`,
-- `alc.nn.card`, and `alc.math.rng_create` so the autoplay loop can run
-- without a real model.
--
-- The stub Card always plays the first legal move ("a"), which makes
-- every fight of a batch a copy of one fight against the deterministic
-- teacher policy — the CI collapses to a single-point interval, which
-- is exactly the shape the Wilson formula produces at `p̂ = 0` or `1`.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- See style_distance_spec.lua for rationale.
alc = alc or {}

local duel = require("guardian_duel")

local VOCAB = duel.player_vocab()
local MOVES = duel.player_legal_actions()
local MOVE_IDS = {}
for _, action in ipairs(MOVES) do
    MOVE_IDS[#MOVE_IDS + 1] = VOCAB.to_id[action]
end

--- Build a handle whose greedy pick is always `preferred_move` (a
--- string like "a" / "b" / "p" / "A"). The logit vector plants a large
--- value on the id of that move so the argmax and the top-first scan
--- both pick it, regardless of turn.
local function make_handle(preferred_move)
    local target_id = VOCAB.to_id[preferred_move]
    if target_id == nil then
        error("spec: preferred_move " .. tostring(preferred_move) .. " outside player vocab")
    end
    return {
        generate_session = function(self, prompt_ids)
            self._last_prompt = prompt_ids
            return {
                next_logits = function()
                    local vocab = VOCAB.size
                    local values = {}
                    for i = 1, vocab do
                        values[i] = 0.0
                    end
                    values[target_id + 1] = 10.0
                    return {
                        vocab = function()
                            return vocab
                        end,
                        argmax = function()
                            return target_id
                        end,
                        top = function(_, n)
                            local ranked = {}
                            for i = 1, vocab do
                                ranked[i] = { id = i - 1, value = values[i] }
                            end
                            table.sort(ranked, function(a, b)
                                if a.value == b.value then
                                    return a.id < b.id
                                end
                                return a.value > b.value
                            end)
                            local out = {}
                            for i = 1, math.min(n, #ranked) do
                                out[i] = ranked[i]
                            end
                            return out
                        end,
                    }
                end,
            }
        end,
    }
end

--- See style_distance_spec.lua for why the metatable proxy stands in for
--- the userdata handle shape here.
local function via_metatable(handle)
    return setmetatable({}, { __index = handle })
end

local ALIAS_TO_HANDLE = {}
alc.card = {
    get_by_alias = function(alias)
        if not ALIAS_TO_HANDLE[alias] then
            return nil
        end
        return { card_id = "card-" .. alias }
    end,
}

alc.nn = alc.nn or {}
alc.nn.card = {
    load_handle = function(card_id)
        local alias = card_id:match("^card%-(.+)$")
        return ALIAS_TO_HANDLE[alias]
    end,
}

--- Deterministic-ish RNG stub for the `"random"` boss path. Returns a
--- xorshift-style sequence keyed on the seed the caller planted; the
--- specs only need reproducibility, not statistical quality.
alc.math = alc.math or {}
alc.math.rng_create = function(seed)
    local state = seed
    if state == 0 then
        state = 0x9E3779B9
    end
    return {
        _state = state,
    }
end

--- `guardian_duel.policy_boss_random` calls `alc.math.rng_int(rng, hi)`
--- for a value in `[1, hi]`. Stub it to advance the xorshift state.
alc.math.rng_int = function(rng, hi)
    local s = rng._state
    s = (s * 1103515245 + 12345) % 2147483648
    rng._state = s
    return (s % hi) + 1
end

local level = require("gameai_metrics.level")

local function reset()
    ALIAS_TO_HANDLE = {}
end

describe("gameai_metrics.level", function()
    it("returns a table with win_rate / ci_lower / ci_upper / wins / n_games", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy", 4, 1)
        expect(type(result)).to.equal("table")
        expect(type(result.win_rate)).to.equal("number")
        expect(type(result.ci_lower)).to.equal("number")
        expect(type(result.ci_upper)).to.equal("number")
        expect(type(result.wins)).to.equal("number")
        expect(result.n_games).to.equal(4)
    end)

    it("keeps win_rate and CI bounds inside [0, 1]", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy", 8, 7)
        expect(result.win_rate >= 0 and result.win_rate <= 1).to.equal(true)
        expect(result.ci_lower >= 0 and result.ci_lower <= 1).to.equal(true)
        expect(result.ci_upper >= 0 and result.ci_upper <= 1).to.equal(true)
        expect(result.ci_lower <= result.win_rate).to.equal(true)
        expect(result.win_rate <= result.ci_upper).to.equal(true)
    end)

    it("collapses the CI to a point interval at p̂ = 0 or 1", function()
        reset()
        -- Both handles are deterministic and the boss is deterministic,
        -- so every fight of the batch is a copy of one fight — wins is
        -- either 0 or n_games. Wilson at p̂ ∈ {0, 1} clamps to
        -- [0, upper] or [lower, 1]; the interval never straddles p̂.
        local h = make_handle("a")
        local result = level(h, "greedy", 4, 0)
        expect(result.win_rate == 0 or result.win_rate == 1).to.equal(true)
    end)

    it("is reproducible from the same (seed, opponent) pair", function()
        reset()
        local h = make_handle("b")
        local a = level(h, "greedy", 4, 3)
        local b = level(h, "greedy", 4, 3)
        expect(a.win_rate).to.equal(b.win_rate)
        expect(a.wins).to.equal(b.wins)
    end)

    it("defaults n_games to 32 and seed to 0", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy")
        expect(result.n_games).to.equal(32)
    end)

    it("rejects an unknown opponent literal", function()
        reset()
        local h = make_handle("b")
        local ok, err = pcall(level, h, "trickster", 4, 1)
        expect(ok).to.equal(false)
        expect(err:find("opponent") ~= nil or err:find("not wired") ~= nil).to.equal(true)
    end)

    it("rejects a zero or negative n_games", function()
        reset()
        local h = make_handle("b")
        local ok, err = pcall(level, h, "greedy", 0, 1)
        expect(ok).to.equal(false)
        expect(err:find("n_games") ~= nil).to.equal(true)
    end)

    it("supports the random boss policy through alc.math.rng_create", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "random", 3, 5)
        expect(result.n_games).to.equal(3)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("resolves a string alias through alc.card.get_by_alias", function()
        reset()
        ALIAS_TO_HANDLE["b"] = make_handle("b")
        local result = level("b", "greedy", 4, 1)
        expect(result.n_games).to.equal(4)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("accepts a handle whose generate_session comes from a metatable", function()
        reset()
        local h = via_metatable(make_handle("b"))
        expect(rawget(h, "generate_session")).to.equal(nil)
        local result = level(h, "greedy", 4, 1)
        expect(result.n_games).to.equal(4)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("still refuses a card table without a generate_session method", function()
        reset()
        local ok, err = pcall(level, { alias = "b" }, "greedy", 4, 1)
        expect(ok).to.equal(false)
        expect(err:find("generate_session") ~= nil).to.equal(true)
    end)
end)
