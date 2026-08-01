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
alc.nn.metric = alc.nn.metric or {}
alc.nn.metric.entropy = function(p)
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
end)
