-- gameai_metrics/spec/boss_seat_spec.lua
--
-- Package-level spec for the shared boss-seat helpers. Run with
-- `alc_pkg_test pkg="gameai_metrics"` after `alc_pkg_link` has
-- registered `guardian_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No real Card is touched: a stub handle plants the next-token logits
-- every decision reads, so the specs assert (a) the prompt is the one
-- `guardian_duel_npc.decide` builds, (b) the mask follows the state's
-- legal set (five moves in mode 0, six in mode 1), (c) the greedy pick
-- is gated to that set, and (d) a player view handed to the boss seat is
-- named as a seat mismatch rather than as a malformed state.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- See style_distance_spec.lua for rationale.
alc = alc or {}

--- `alc.math.softmax(logits)` — the host surface the metric reads since
--- the hand-rolled loop moved to mathlib. Same method: subtract the max
--- before exponentiating, then normalise.
alc.math = alc.math or {}
alc.math.softmax = function(logits)
    local max_l = logits[1]
    for i = 2, #logits do
        if logits[i] > max_l then
            max_l = logits[i]
        end
    end
    local out, sum = {}, 0.0
    for i, l in ipairs(logits) do
        local w = math.exp(l - max_l)
        out[i] = w
        sum = sum + w
    end
    for i, w in ipairs(out) do
        out[i] = w / sum
    end
    return out
end

local duel = require("guardian_duel")
local boss_seat = require("gameai_metrics.boss_seat")

local VOCAB = duel.vocab()

--- Mode-0 opening state: five legal moves (`t` needs mode 1).
local function mode0_state(fields)
    local state = duel.new_game(1).boss
    for k, v in pairs(fields or {}) do
        state[k] = v
    end
    return state
end

--- Mid-shift state: all six boss moves legal.
local function mode1_state(fields)
    local merged = { mode = 1, cycle = 1, shifts = 1 }
    for k, v in pairs(fields or {}) do
        merged[k] = v
    end
    return mode0_state(merged)
end

--- Handle whose next-token logits are planted per boss move character.
---
--- Characters outside `logits` read zero, so a caller can plant a single
--- large value and know every other token is flat.
local function make_handle(logits)
    local handle = { _logits = logits or {} }
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
                for ch, value in pairs(h._logits) do
                    local id = VOCAB.to_id[ch]
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

describe("gameai_metrics.boss_seat", function()
    describe("require_seat", function()
        it("defaults to the player seat when the opt is absent", function()
            expect(boss_seat.require_seat(nil, "spec")).to.equal("player")
        end)

        it("accepts both spellings", function()
            expect(boss_seat.require_seat("player", "spec")).to.equal("player")
            expect(boss_seat.require_seat("boss", "spec")).to.equal("boss")
        end)

        it("refuses an unknown seat instead of falling back", function()
            local ok, err = pcall(boss_seat.require_seat, "referee", "spec")
            expect(ok).to.equal(false)
            expect(err:find("seat") ~= nil).to.equal(true)
        end)
    end)

    describe("require_style", function()
        it("accepts a canonical style", function()
            expect(boss_seat.require_style("guardian", "spec")).to.equal("guardian")
        end)

        it("refuses a missing style loudly", function()
            local ok, err = pcall(boss_seat.require_style, nil, "spec")
            expect(ok).to.equal(false)
            expect(err:find("style") ~= nil).to.equal(true)
        end)

        it("refuses an unknown style and lists the valid names", function()
            local ok, err = pcall(boss_seat.require_style, "trickster", "spec")
            expect(ok).to.equal(false)
            expect(err:find("guardian") ~= nil).to.equal(true)
        end)
    end)

    describe("require_state", function()
        it("accepts a boss state", function()
            local state = mode0_state()
            expect(boss_seat.require_state(state, "spec: state")).to.equal(state)
        end)

        it("names a player view as a seat mismatch", function()
            local view = {
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
            local ok, err = pcall(boss_seat.require_state, view, "spec: prompt_set[1]")
            expect(ok).to.equal(false)
            expect(err:find("player view") ~= nil).to.equal(true)
        end)

        it("names the missing boss-state field", function()
            local ok, err = pcall(boss_seat.require_state, { mode = 0 }, "spec: state")
            expect(ok).to.equal(false)
            expect(err:find("cycle") ~= nil).to.equal(true)
        end)

        it("refuses a non-table", function()
            local ok, err = pcall(boss_seat.require_state, 42, "spec: state")
            expect(ok).to.equal(false)
            expect(err:find("boss state") ~= nil).to.equal(true)
        end)
    end)

    describe("legal", function()
        it("offers five moves in mode 0 and six in mode 1", function()
            expect(#boss_seat.legal(mode0_state()).ids).to.equal(5)
            expect(#boss_seat.legal(mode1_state()).ids).to.equal(6)
        end)

        it("keeps the twin slam out of mode 0", function()
            local mode0 = boss_seat.legal(mode0_state())
            local mode1 = boss_seat.legal(mode1_state())
            local function has(legal, action)
                for _, ch in ipairs(legal.actions) do
                    if ch == action then
                        return true
                    end
                end
                return false
            end
            expect(has(mode0, "t")).to.equal(false)
            expect(has(mode1, "t")).to.equal(true)
        end)
    end)

    describe("encode", function()
        it("is the guardian_duel encoding plus the separator", function()
            local state = mode0_state()
            expect(boss_seat.encode(state, "guardian")).to.equal(
                duel.encode(state, "guardian") .. ">"
            )
        end)

        it("refuses an unknown style through guardian_duel", function()
            local ok = pcall(boss_seat.encode, mode0_state(), "trickster")
            expect(ok).to.equal(false)
        end)
    end)

    describe("probs", function()
        it("returns one probability per legal move, summing to 1", function()
            local probs, legal = boss_seat.probs(make_handle(), mode0_state(), "guardian")
            expect(#probs).to.equal(#legal.ids)
            expect(#probs).to.equal(5)
            local sum = 0.0
            for _, x in ipairs(probs) do
                sum = sum + x
            end
            expect(math.abs(sum - 1.0) < 1e-9).to.equal(true)
        end)

        it("is uniform for flat logits", function()
            local probs = boss_seat.probs(make_handle(), mode1_state(), "guardian")
            for _, x in ipairs(probs) do
                expect(math.abs(x - 1.0 / 6.0) < 1e-9).to.equal(true)
            end
        end)

        it("ignores logits on tokens outside the legal set", function()
            -- `>` is in the vocabulary but is not a move; planting a huge
            -- value on it must not change the row.
            local flat = boss_seat.probs(make_handle(), mode0_state(), "guardian")
            local loud = boss_seat.probs(make_handle({ [">"] = 50.0 }), mode0_state(), "guardian")
            for i = 1, #flat do
                expect(math.abs(flat[i] - loud[i]) < 1e-9).to.equal(true)
            end
        end)

        it("flattens as the temperature rises", function()
            local handle = make_handle({ f = 2.0, c = 1.0 })
            local cold = boss_seat.probs(handle, mode0_state(), "guardian", 0.5)
            local warm = boss_seat.probs(handle, mode0_state(), "guardian", 2.0)
            local function peak(row)
                local best = row[1]
                for _, x in ipairs(row) do
                    if x > best then
                        best = x
                    end
                end
                return best
            end
            expect(peak(warm) < peak(cold)).to.equal(true)
        end)

        it("refuses a non-positive temperature", function()
            local ok, err = pcall(boss_seat.probs, make_handle(), mode0_state(), "guardian", 0)
            expect(ok).to.equal(false)
            expect(err:find("temperature") ~= nil).to.equal(true)
        end)
    end)

    describe("decide", function()
        it("returns the highest-ranked legal move", function()
            local action = boss_seat.decide(make_handle({ v = 9.0 }), mode0_state(), "guardian")
            expect(action).to.equal("v")
        end)

        it("skips a higher-ranked token that is not a move", function()
            local handle = make_handle({ [">"] = 50.0, w = 9.0 })
            expect(boss_seat.decide(handle, mode0_state(), "guardian")).to.equal("w")
        end)

        it("skips the twin slam while it is illegal", function()
            local handle = make_handle({ t = 50.0, d = 9.0 })
            expect(boss_seat.decide(handle, mode0_state(), "guardian")).to.equal("d")
            expect(boss_seat.decide(handle, mode1_state(), "guardian")).to.equal("t")
        end)
    end)
end)
