-- gameai_metrics/spec/othello_seat_spec.lua
--
-- Package-level spec for the shared Othello-seat helpers. Run with
-- `alc_pkg_test pkg="gameai_metrics"` after `alc_pkg_link` has
-- registered `othello6` and this package, or through the `mlua-probe`
-- runner with `examples/gameai` on the search path. The `lust` globals
-- are pre-loaded by both runners.
--
-- No real Card is touched: a stub handle plants the next-token logits
-- every decision reads, so the cases assert (a) the prompt is the one
-- `othello6_npc.decide` builds, (b) the row is masked to the position's
-- legal set and normalised over it, (c) temperature flattens that row
-- rather than reordering it, (d) the greedy pick is gated to the legal
-- set even when the argmax is a board character, and (e) a position
-- whose only move is the pass reads as a one-move distribution rather
-- than as an error.

local describe, it, expect = lust.describe, lust.it, lust.expect

local othello = require("othello6")
local seat = require("gameai_metrics.othello_seat")

local VOCAB = othello.vocab()

--- The opening, where black has four placements and the corner is not
--- one of them.
local function opening()
    return othello.new_game(1)
end

--- The opening after one legal move, so a case can read a prompt that
--- carries a move rather than the bare marker.
local function after_one_move()
    local state = opening()
    return othello.apply(state, othello.legal_actions(state)[1])
end

--- A position whose side to move holds no disc at all, so it has no
--- placement and `legal_actions` answers the pass alone.
local function pass_only()
    return othello.state_from_rows({
        "BBBBBB",
        "BBBBBB",
        "BBBBBB",
        "BBBBBB",
        "BBBBBB",
        "BBBBB.",
    }, "white")
end

--- Handle whose next-token logits are planted per character.
---
--- Characters outside `logits` read zero, so a caller can plant a single
--- large value and know every other token is flat. The prompt of the
--- last call is kept so a case can read what the seat encoded.
---@param logits table Character-keyed logit values
---@return table handle
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
                        error("spec: char " .. tostring(ch) .. " is outside the move vocab")
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

--- Plant a distinct logit on every legal move of `state`, descending in
--- `legal_actions` order, plus whatever else the case asks for.
---@param state table Position
---@param extra table|nil Character-keyed logits merged on top
---@return table handle, string[] legal
local function handle_over_legal(state, extra)
    local legal = othello.legal_actions(state)
    local planted = {}
    for index, action in ipairs(legal) do
        planted[action] = 4.0 - index
    end
    for ch, value in pairs(extra or {}) do
        planted[ch] = value
    end
    return make_handle(planted), legal
end

describe("gameai_metrics.othello_seat.legal", function()
    it("answers the rules module's legal set, in its order", function()
        local state = opening()
        local expected = othello.legal_actions(state)
        local actual = seat.legal(state)
        expect(#actual).to.equal(#expected)
        for index, action in ipairs(expected) do
            expect(actual[index]).to.equal(action)
        end
    end)

    it("answers the pass alone on a position with no placement", function()
        local actual = seat.legal(pass_only())
        expect(#actual).to.equal(1)
        expect(actual[1]).to.equal(othello.PASS)
    end)
end)

describe("gameai_metrics.othello_seat.encode", function()
    it("writes the prompt the rules module writes", function()
        expect(seat.encode(opening())).to.equal(othello.encode(opening()))
        local played = after_one_move()
        expect(seat.encode(played)).to.equal(othello.encode(played))
    end)

    it("appends no separator, unlike the guardian boss seat", function()
        -- An Othello line is the move sequence and the model answers the
        -- next character of it directly; a separator would be a token
        -- the corpus never carried.
        expect(seat.encode(opening())).to.equal(othello.BOS)
    end)

    it("is the prompt the decode actually opens a session over", function()
        local state = after_one_move()
        local handle = handle_over_legal(state)
        seat.decide(handle, state)
        local expected = othello.to_ids(othello.encode(state))
        expect(#handle._last_prompt).to.equal(#expected)
        for index, id in ipairs(expected) do
            expect(handle._last_prompt[index]).to.equal(id)
        end
    end)
end)

describe("gameai_metrics.othello_seat.probs", function()
    it("puts all of its mass on the legal moves", function()
        local state = opening()
        local handle, legal = handle_over_legal(state)
        local probs, returned = seat.probs(handle, state)
        expect(#probs).to.equal(#legal)
        expect(#returned).to.equal(#legal)
        local sum = 0.0
        for index, p in ipairs(probs) do
            expect(returned[index]).to.equal(legal[index])
            expect(p > 0).to.equal(true)
            sum = sum + p
        end
        expect(math.abs(sum - 1.0) < 1e-12).to.equal(true)
    end)

    it("gives an illegal move no mass however large its logit is", function()
        -- The corner is empty at the opening but bracketing nothing, so
        -- it is not a legal placement; a huge logit on it must not
        -- reach the row at all.
        local state = opening()
        local corner = othello.action_of_index(0)
        for _, action in ipairs(othello.legal_actions(state)) do
            expect(action ~= corner).to.equal(true)
        end

        local baseline = seat.probs(handle_over_legal(state), state)
        local flood_handle = handle_over_legal(state, { [corner] = 50.0 })
        local flooded, returned = seat.probs(flood_handle, state)
        for _, action in ipairs(returned) do
            expect(action ~= corner).to.equal(true)
        end
        for index, p in ipairs(baseline) do
            expect(math.abs(flooded[index] - p) < 1e-12).to.equal(true)
        end
    end)

    it("flattens the row as the temperature rises", function()
        local state = opening()
        local handle = handle_over_legal(state)
        local function peak(t)
            local probs = seat.probs(handle, state, t)
            local best = probs[1]
            for _, p in ipairs(probs) do
                if p > best then
                    best = p
                end
            end
            return best
        end
        local cold, warm, hot = peak(0.5), peak(1.0), peak(4.0)
        expect(cold > warm).to.equal(true)
        expect(warm > hot).to.equal(true)
        -- The flat limit is the uniform row over the legal set, which is
        -- the floor the peak walks down towards.
        expect(hot > 1.0 / #othello.legal_actions(state)).to.equal(true)
    end)

    it("answers a certainty on a position whose only move is the pass", function()
        local state = pass_only()
        local probs, legal = seat.probs(make_handle({}), state)
        expect(#legal).to.equal(1)
        expect(legal[1]).to.equal(othello.PASS)
        expect(#probs).to.equal(1)
        expect(math.abs(probs[1] - 1.0) < 1e-12).to.equal(true)
    end)

    it("refuses a temperature that is not a finite positive number", function()
        local state = opening()
        local handle = handle_over_legal(state)
        for _, t in ipairs({ 0, -1, math.huge }) do
            expect(function()
                seat.probs(handle, state, t)
            end).to.fail()
        end
        expect(function()
            seat.probs(handle, state, "1.0")
        end).to.fail()
    end)
end)

describe("gameai_metrics.othello_seat.decide", function()
    it("answers the highest-ranked legal move", function()
        local state = opening()
        local legal = othello.legal_actions(state)
        -- The last legal move in rules order is planted above the rest,
        -- so a decode that answered by position rather than by logit
        -- would answer the first one instead.
        local handle = handle_over_legal(state, { [legal[#legal]] = 9.0 })
        expect(seat.decide(handle, state)).to.equal(legal[#legal])
    end)

    it("steps past an argmax that is not a legal move", function()
        -- A board character and an empty corner both outrank every legal
        -- move here; the gate has to walk down to the first legal entry
        -- rather than answer the top one.
        local state = opening()
        local legal = othello.legal_actions(state)
        local handle = handle_over_legal(state, {
            B = 99.0,
            [othello.action_of_index(0)] = 50.0,
            [legal[2]] = 9.0,
        })
        expect(seat.decide(handle, state)).to.equal(legal[2])
    end)

    it("agrees with the argmax of its own probability row", function()
        local state = after_one_move()
        local handle, legal = handle_over_legal(state, { [othello.action_of_index(0)] = 30.0 })
        local probs = seat.probs(handle, state)
        local best_index = 1
        for index, p in ipairs(probs) do
            if p > probs[best_index] then
                best_index = index
            end
        end
        expect(seat.decide(handle, state)).to.equal(legal[best_index])
    end)

    it("answers the pass on a position whose only move is the pass", function()
        local state = pass_only()
        expect(seat.decide(make_handle({ B = 99.0 }), state)).to.equal(othello.PASS)
    end)
end)
