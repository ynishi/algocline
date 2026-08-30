-- gameai_metrics/spec/trickiness_spec.lua
--
-- Package-level spec for the `trickiness` metric. Same host-stub shape
-- as `style_distance_spec.lua`; the only interesting difference is that
-- the entropy primitive is what the compose call is asserting against.

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

local function make_handle(alias)
    local handle = { alias = alias, _default_logits = nil }
    function handle:generate_session(prompt_ids)
        self._last_prompt = prompt_ids
        local h = self
        return {
            next_logits = function()
                local vocab = VOCAB.size
                local values = {}
                for i = 1, vocab do
                    values[i] = 0.0
                end
                local row = h._default_logits or {}
                for i, id in ipairs(MOVE_IDS) do
                    values[id + 1] = row[i] or 0.0
                end
                return {
                    vocab = function()
                        return vocab
                    end,
                    argmax = function()
                        local best_id, best_v = 0, values[1]
                        for i = 2, vocab do
                            if values[i] > best_v then
                                best_v, best_id = values[i], i - 1
                            end
                        end
                        return best_id
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
    end
    return handle
end

--- See style_distance_spec.lua for why the metatable proxy stands in for
--- the userdata handle shape here.
local function via_metatable(handle)
    return setmetatable({}, { __index = handle })
end

-- ─── Boss-seat stub ─────────────────────────────────────────────────
--
-- The boss seat reads the *boss* alphabet and a state-dependent legal
-- set, so it needs a handle of its own.

local BOSS_VOCAB = duel.vocab()

--- Handle whose next-token logits are planted per boss move character;
--- every other token reads zero (so the default row is uniform over the
--- state's legal set, whichever size that is).
local function make_boss_handle(logits)
    local handle = { _logits = logits or {} }
    function handle:generate_session(prompt_ids)
        self._last_prompt = prompt_ids
        local h = self
        return {
            next_logits = function()
                local vocab = BOSS_VOCAB.size
                local values = {}
                for i = 1, vocab do
                    values[i] = 0.0
                end
                for ch, value in pairs(h._logits) do
                    local id = BOSS_VOCAB.to_id[ch]
                    if id == nil then
                        error("spec: char " .. tostring(ch) .. " is outside the boss vocab")
                    end
                    values[id + 1] = value
                end
                return {
                    vocab = function()
                        return vocab
                    end,
                    argmax = function()
                        local best_id, best_v = 0, values[1]
                        for i = 2, vocab do
                            if values[i] > best_v then
                                best_v, best_id = values[i], i - 1
                            end
                        end
                        return best_id
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
    end
    return handle
end

--- Mode-0 opening state (five legal boss moves).
local function boss_state(fields)
    local state = duel.new_game(1).boss
    for k, v in pairs(fields or {}) do
        state[k] = v
    end
    return state
end

--- Mid-shift state (all six boss moves legal).
local function boss_state_mode1()
    return boss_state({ mode = 1, cycle = 1, shifts = 1 })
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

--- Every entropy call the metric made since the last reset.
local ENTROPY_CALLS = {}
alc.math = alc.math or {}
alc.math.entropy = function(p)
    ENTROPY_CALLS[#ENTROPY_CALLS + 1] = { p = p }
    -- Real entropy for a length-4 distribution (natural log), so the
    -- specs can assert 0.0 on a peaked row and log(4) on a uniform one
    -- without depending on the compiled primitive being reachable in
    -- the pkg_test VM.
    local h = 0.0
    for _, x in ipairs(p) do
        if x > 0 then
            h = h - x * math.log(x)
        end
    end
    return h
end

local trickiness = require("gameai_metrics.trickiness")

local function reset()
    ENTROPY_CALLS = {}
    ALIAS_TO_HANDLE = {}
end

local function view(fields)
    local v = {
        turn = 1,
        mode = 0,
        boss_hp = duel.BOSS_MAX_HP,
        shift_distance = duel.threshold_damage("guardian", 0),
        hp = duel.PLAYER_MAX_HP,
        weakened = false,
        exposed = false,
        spikes = false,
        intent = duel.NO_INTENT,
    }
    for k, val in pairs(fields or {}) do
        v[k] = val
    end
    return v
end

describe("gameai_metrics.trickiness", function()
    it("reports log(4) for a Card with uniform logits (max entropy)", function()
        reset()
        local h = make_handle("A")
        -- default_logits nil → all zeros → uniform softmax → entropy log 4
        local t = trickiness(h, { view(), view({ turn = 2 }) })
        expect(math.abs(t - math.log(4)) < 1e-6).to.equal(true)
        expect(#ENTROPY_CALLS).to.equal(2)
    end)

    it("reports ~0 for a Card with a single dominant move", function()
        reset()
        local h = make_handle("A")
        h._default_logits = { 30.0, 0.0, 0.0, 0.0 }
        local t = trickiness(h, { view(), view({ turn = 2 }) })
        expect(t < 1e-6).to.equal(true)
        expect(#ENTROPY_CALLS).to.equal(2)
    end)

    it("temperature widens the distribution (higher entropy)", function()
        reset()
        local h = make_handle("A")
        h._default_logits = { 2.0, 1.0, 0.5, 0.0 }
        local low = trickiness(h, { view() }, 0.5)
        local mid = trickiness(h, { view() }, 1.0)
        local high = trickiness(h, { view() }, 2.0)
        -- Higher temperature flattens the softmax, so the entropy of a
        -- temperature-swept distribution is monotone in `t`.
        expect(low < mid).to.equal(true)
        expect(mid < high).to.equal(true)
    end)

    it("defaults temperature to 1.0 when omitted", function()
        reset()
        local h = make_handle("A")
        h._default_logits = { 2.0, 1.0, 0.5, 0.0 }
        local a = trickiness(h, { view() })
        local b = trickiness(h, { view() }, 1.0)
        expect(math.abs(a - b) < 1e-9).to.equal(true)
    end)

    it("rejects a non-positive temperature", function()
        reset()
        local h = make_handle("A")
        local ok, err = pcall(trickiness, h, { view() }, 0)
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)

    it("rejects an empty prompt_set", function()
        reset()
        local h = make_handle("A")
        local ok, err = pcall(trickiness, h, {})
        expect(ok).to.equal(false)
        expect(err:find("empty") ~= nil).to.equal(true)
    end)

    it("resolves a string alias through alc.card.get_by_alias", function()
        reset()
        local h = make_handle("A")
        ALIAS_TO_HANDLE["A"] = h
        local t = trickiness("A", { view() })
        expect(math.abs(t - math.log(4)) < 1e-6).to.equal(true)
        expect(#ENTROPY_CALLS).to.equal(1)
    end)

    it("accepts a handle whose generate_session comes from a metatable", function()
        reset()
        local h = via_metatable(make_handle("A"))
        expect(rawget(h, "generate_session")).to.equal(nil)
        local t = trickiness(h, { view() })
        expect(math.abs(t - math.log(4)) < 1e-6).to.equal(true)
        expect(#ENTROPY_CALLS).to.equal(1)
    end)

    it("still refuses a table without a generate_session method", function()
        reset()
        local ok, err = pcall(trickiness, { alias = "A" }, { view() })
        expect(ok).to.equal(false)
        expect(err:find("generate_session") ~= nil).to.equal(true)
    end)

    describe("seat defaults", function()
        it("keeps the scalar return with opts omitted, empty, or seat=player", function()
            reset()
            local h = make_handle("A")
            h._default_logits = { 2.0, 1.0, 0.5, 0.0 }
            local bare = trickiness(h, { view() })
            local empty = trickiness(h, { view() }, nil, {})
            local explicit = trickiness(h, { view() }, nil, { seat = "player" })
            expect(type(bare)).to.equal("number")
            expect(empty).to.equal(bare)
            expect(explicit).to.equal(bare)
        end)

        it("refuses a non-table opts", function()
            reset()
            local h = make_handle("A")
            local ok, err = pcall(trickiness, h, { view() }, nil, "boss")
            expect(ok).to.equal(false)
            expect(err:find("opts") ~= nil).to.equal(true)
        end)
    end)

    describe('seat = "boss"', function()
        it("returns a normalised value alongside the raw mean", function()
            reset()
            local h = make_boss_handle()
            local t = trickiness(h, { boss_state() }, nil, { seat = "boss", style = "guardian" })
            expect(type(t)).to.equal("table")
            expect(type(t.value)).to.equal("number")
            expect(type(t.raw_mean)).to.equal("number")
            expect(#ENTROPY_CALLS).to.equal(1)
        end)

        it("puts the same ceiling on a 5-move and a 6-move state", function()
            reset()
            local h = make_boss_handle()
            -- Uniform logits: the raw entropy is log 5 in mode 0 and
            -- log 6 in mode 1, so only the normalised value is
            -- comparable across the two.
            local mode0 = trickiness(
                h,
                { boss_state() },
                nil,
                { seat = "boss", style = "guardian" }
            )
            local mode1 = trickiness(
                h,
                { boss_state_mode1() },
                nil,
                { seat = "boss", style = "guardian" }
            )
            expect(math.abs(mode0.raw_mean - math.log(5)) < 1e-9).to.equal(true)
            expect(math.abs(mode1.raw_mean - math.log(6)) < 1e-9).to.equal(true)
            expect(math.abs(mode0.value - 1.0) < 1e-9).to.equal(true)
            expect(math.abs(mode1.value - 1.0) < 1e-9).to.equal(true)
        end)

        it("averages a mixed prompt set on the normalised scale", function()
            reset()
            local h = make_boss_handle()
            local t = trickiness(
                h,
                { boss_state(), boss_state_mode1() },
                nil,
                { seat = "boss", style = "guardian" }
            )
            expect(math.abs(t.value - 1.0) < 1e-9).to.equal(true)
            expect(math.abs(t.raw_mean - (math.log(5) + math.log(6)) / 2) < 1e-9).to.equal(true)
            expect(#ENTROPY_CALLS).to.equal(2)
        end)

        it("reports ~0 for a boss Card committed to one move", function()
            reset()
            local h = make_boss_handle({ f = 40.0 })
            local t = trickiness(
                h,
                { boss_state(), boss_state_mode1() },
                nil,
                { seat = "boss", style = "guardian" }
            )
            expect(t.value < 1e-6).to.equal(true)
            expect(t.raw_mean < 1e-6).to.equal(true)
        end)

        it("keeps the normalised value inside [0, 1] for a middling row", function()
            reset()
            local h = make_boss_handle({ f = 2.0, c = 1.0, v = 0.5 })
            local prompt_set = { boss_state(), boss_state_mode1() }
            for _, temperature in ipairs({ 0.5, 1.0, 2.0 }) do
                local t =
                    trickiness(h, prompt_set, temperature, { seat = "boss", style = "guardian" })
                expect(t.value >= 0 and t.value <= 1).to.equal(true)
                expect(t.value > 0).to.equal(true)
            end
        end)

        it("requires a style", function()
            reset()
            local h = make_boss_handle()
            local ok, err = pcall(trickiness, h, { boss_state() }, nil, { seat = "boss" })
            expect(ok).to.equal(false)
            expect(err:find("style") ~= nil).to.equal(true)
        end)

        it("names a player view in the prompt set as a seat mismatch", function()
            reset()
            local h = make_boss_handle()
            local ok, err = pcall(
                trickiness,
                h,
                { view() },
                nil,
                { seat = "boss", style = "guardian" }
            )
            expect(ok).to.equal(false)
            expect(err:find("player view") ~= nil).to.equal(true)
            expect(err:find("prompt_set%[1%]") ~= nil).to.equal(true)
        end)
    end)
end)
