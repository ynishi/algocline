-- gameai_metrics/spec/style_distance_spec.lua
--
-- Package-level spec for the `style_distance` metric. Run with
-- `alc_pkg_test pkg="gameai_metrics"` after `alc_pkg_link` has
-- registered `guardian_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No real Card and no real model are touched: `alc.card`,
-- `alc.nn.card`, and `alc.nn.metric.js` are replaced by stubs so the
-- specs assert (a) the compose loop calls `js` once per prompt, (b)
-- argument validation refuses the shapes it says it does, (c) the
-- softmax feeds `js` a length-4 row that sums to 1.0.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- Bootstrap `alc` when the spec is executed outside `alc_pkg_test` (e.g.
-- via lua-debugger for a local smoke). Under `alc_pkg_test` the bridge
-- has already installed the real `alc` table and this is a no-op.
alc = alc or {}

local duel = require("guardian_duel")

-- ─── Host stubs ─────────────────────────────────────────────────────

local VOCAB = duel.player_vocab()
local MOVES = duel.player_legal_actions()
local MOVE_IDS = {}
for _, action in ipairs(MOVES) do
    MOVE_IDS[#MOVE_IDS + 1] = VOCAB.to_id[action]
end

-- Handle table records which alias / card_id it was born from and lets
-- the spec plant the logit vector every decision reads.
local function make_handle(alias, logits_by_view)
    local handle = {
        alias = alias,
        _logits_by_view = logits_by_view or {},
        _default_logits = nil,
    }
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
                -- Plant per-move logits at the LEGAL IDS. The default is
                -- zero which softmaxes to uniform 0.25 for every move —
                -- what the empty argument path exercises.
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

--- Stub `alc.card.get_by_alias` → `alc.nn.card.load_handle` chain.
local ALIAS_TO_HANDLE = {}
alc.card = {
    get_by_alias = function(alias)
        if not ALIAS_TO_HANDLE[alias] then
            return nil
        end
        return { card_id = "card-" .. alias }
    end,
}

-- Real `alc.nn` may not exist in this pkg_test VM (the `nn` feature
-- gates it). Stub the two entries the metric touches.
alc.nn = alc.nn or {}
alc.nn.card = {
    load_handle = function(card_id)
        local alias = card_id:match("^card%-(.+)$")
        return ALIAS_TO_HANDLE[alias]
    end,
}

--- Record every `js` call so specs can assert compose behaviour.
local JS_CALLS = {}
alc.nn.metric = alc.nn.metric or {}
alc.nn.metric.js = function(p, q)
    JS_CALLS[#JS_CALLS + 1] = { p = p, q = q }
    -- Return 0 when p == q element-wise, 1 otherwise — a deterministic
    -- stand-in that lets the mean assert on a lossless invariant
    -- without pulling in the real primitive.
    if #p ~= #q then
        error("js stub: length mismatch")
    end
    for i = 1, #p do
        if math.abs(p[i] - q[i]) > 1e-9 then
            return 1.0
        end
    end
    return 0.0
end

local style_distance = require("gameai_metrics.style_distance")

--- Reset every spec-visible piece of stub state.
local function reset()
    JS_CALLS = {}
    ALIAS_TO_HANDLE = {}
end

--- Build a fresh view with every player_view field present.
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

describe("gameai_metrics.style_distance", function()
    it("returns 0.0 when two Cards agree on every view", function()
        reset()
        local ha = make_handle("A")
        local hb = make_handle("B")
        -- Both handles carry the same default (all zero) → uniform
        -- softmax → identical distributions.
        local prompt_set = { view(), view({ turn = 2 }), view({ turn = 3 }) }
        local d = style_distance(ha, hb, prompt_set)
        expect(d).to.equal(0.0)
        expect(#JS_CALLS).to.equal(3)
    end)

    it("calls js once per prompt with length-4 distributions summing to 1", function()
        reset()
        local ha = make_handle("A")
        ha._default_logits = { 10.0, 0.0, 0.0, 0.0 } -- peak on move 1
        local hb = make_handle("B")
        hb._default_logits = { 0.0, 0.0, 0.0, 10.0 } -- peak on move 4
        local prompt_set = { view(), view({ turn = 2 }) }
        local d = style_distance(ha, hb, prompt_set)
        expect(d).to.equal(1.0) -- stub js returns 1 when p ≠ q
        expect(#JS_CALLS).to.equal(2)
        for _, call in ipairs(JS_CALLS) do
            expect(#call.p).to.equal(4)
            expect(#call.q).to.equal(4)
            local sp = 0.0
            for _, x in ipairs(call.p) do
                sp = sp + x
            end
            expect(math.abs(sp - 1.0) < 1e-6).to.equal(true)
            local sq = 0.0
            for _, x in ipairs(call.q) do
                sq = sq + x
            end
            expect(math.abs(sq - 1.0) < 1e-6).to.equal(true)
        end
    end)

    it("rejects a non-table prompt_set", function()
        reset()
        local ha = make_handle("A")
        local hb = make_handle("B")
        local ok, err = pcall(style_distance, ha, hb, "not-a-table")
        expect(ok).to.equal(false)
        expect(err:find("prompt_set") ~= nil).to.equal(true)
    end)

    it("rejects an empty prompt_set", function()
        reset()
        local ha = make_handle("A")
        local hb = make_handle("B")
        local ok, err = pcall(style_distance, ha, hb, {})
        expect(ok).to.equal(false)
        expect(err:find("empty") ~= nil).to.equal(true)
    end)

    it("rejects a card_a that is neither a string nor a handle table", function()
        reset()
        local hb = make_handle("B")
        local ok, err = pcall(style_distance, 42, hb, { view() })
        expect(ok).to.equal(false)
        expect(err:find("card_a") ~= nil).to.equal(true)
    end)

    it("resolves a string alias through alc.card.get_by_alias", function()
        reset()
        local ha = make_handle("A")
        local hb = make_handle("B")
        ALIAS_TO_HANDLE["A"] = ha
        ALIAS_TO_HANDLE["B"] = hb
        local d = style_distance("A", "B", { view() })
        expect(d).to.equal(0.0) -- both uniform → stub js returns 0
        expect(#JS_CALLS).to.equal(1)
    end)
end)
