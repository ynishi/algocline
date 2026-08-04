-- othello6_teacher/spec/othello6_teacher_spec.lua
--
-- Package-level spec for the 6x6 Othello teachers. Run with
-- `alc_pkg_test pkg="othello6_teacher"` after `alc_pkg_link` has
-- registered the collection, or through the `mlua-probe` runner with
-- `examples/gameai` on the search path. The `lust` globals are
-- pre-loaded by both runners.
--
-- The teacher is what every later measurement is taken against: a Card
-- is scored by how often it answers what the teacher answered. A
-- teacher that quietly played a slightly different move would still
-- produce a corpus, still train, and still report an agreement rate,
-- and the rate would be measuring the bug. So the pruning is checked
-- against an unpruned search rather than trusted, the two dials are
-- checked to actually move the answer, and the answer is checked to be
-- the same one every time it is asked for.
--
-- Moves are written as `square(row, col)` rather than as their letters:
-- the alphabet belongs to `othello6.action_of_index`, and a spec that
-- spelled the letters out would have to be rewritten if the alphabet
-- ever moved.

local describe, it, expect = lust.describe, lust.it, lust.expect

local othello = require("othello6")
local teacher = require("othello6_teacher")

--- Move that plays a square, by board coordinates.
local function square(row, col)
    return othello.action_of_index(row * othello.BOARD_SIZE + col)
end

local function position(rows, turn)
    return othello.state_from_rows(rows, turn)
end

local function opening()
    return othello.new_game(1)
end

--- Three black discs over three white ones, black to move.
---
--- Five placements, all of them flipping something, so the styles and
--- the depths have room to disagree.
local function three_rows()
    return position({
        "......",
        "......",
        ".BBB..",
        ".WWW..",
        "......",
        "......",
    }, "black")
end

--- A mid-game position taken from played moves.
local function midgame()
    return position({
        "...B..",
        "...B..",
        ".BWB..",
        "BBBW..",
        "...BW.",
        "......",
    }, "white")
end

--- A crowded position, where most of the board is already committed.
local function crowded()
    return position({
        "..WB..",
        "..BW.B",
        "WBBBB.",
        "BBBBB.",
        "...BW.",
        "......",
    }, "white")
end

--- White to move with no disc of its own anywhere.
---
--- A placement has to bracket against an own disc, so neither side has
--- one here: white passes, black passes, and the game is over two plies
--- later however deep the search was asked to go.
local function blocked()
    return position({
        "BBB...",
        "BBB...",
        "......",
        "......",
        "......",
        "......",
    }, "white")
end

--- A finished game: full board, twenty black against sixteen white.
local function finished(turn)
    return position({
        "BBBBBB",
        "BBBBBB",
        "BBBBBB",
        "BBWWWW",
        "WWWWWW",
        "WWWWWW",
    }, turn)
end

local CROSS_CHECK = {
    { name = "opening", make = opening },
    { name = "three rows", make = three_rows },
    { name = "midgame", make = midgame },
    { name = "crowded", make = crowded },
    { name = "blocked", make = blocked },
}

describe("othello6_teacher", function()
    describe("declared dials", function()
        it("names the styles othello6 declares, in the same order", function()
            expect(table.concat(teacher.STYLES, ",")).to.equal(table.concat(othello.STYLES, ","))
        end)

        it("sweeps the depths of the design", function()
            expect(table.concat(teacher.DEPTHS, ",")).to.equal("1,2,4,6")
        end)

        it("refuses a style it has no evaluator for", function()
            expect(function()
                teacher.evaluate(opening(), "positional")
            end).to.fail()
            expect(function()
                teacher.search(opening(), 2, nil)
            end).to.fail()
        end)

        it("refuses a depth that is not a positive integer", function()
            expect(function()
                teacher.search(opening(), 0, "corner")
            end).to.fail()
            expect(function()
                teacher.search(opening(), 1.5, "corner")
            end).to.fail()
            expect(function()
                teacher.policy(-1, "corner")
            end).to.fail()
        end)

        it("refuses a position that is not a table", function()
            expect(function()
                teacher.search("...", 2, "corner")
            end).to.fail()
        end)
    end)

    -- The load-bearing test of the package. Alpha-beta may visit fewer
    -- nodes than the plain negamax but it may not reach a different
    -- answer, and "fewer nodes" is exactly the condition under which a
    -- wrong answer is hard to notice by looking at one game.
    describe("alpha-beta answers what the unpruned search answers", function()
        for _, case in ipairs(CROSS_CHECK) do
            for _, style in ipairs(othello.STYLES) do
                for depth = 1, 4 do
                    it(string.format("%s / %s / depth %d", case.name, style, depth), function()
                        local state = case.make()
                        local pruned, pruned_value = teacher.search(state, depth, style)
                        local plain, plain_value = teacher.search_naive(state, depth, style)
                        expect(pruned).to.equal(plain)
                        expect(pruned_value).to.equal(plain_value)
                    end)
                end
            end
        end
    end)

    describe("the answer is deterministic", function()
        for _, style in ipairs(othello.STYLES) do
            it("repeats itself on " .. style, function()
                for _, depth in ipairs({ 1, 2, 4 }) do
                    local state = midgame()
                    local first = teacher.search(state, depth, style)
                    for _ = 1, 10 do
                        local again, value = teacher.search(midgame(), depth, style)
                        expect(again).to.equal(first)
                        expect(value).to.equal(select(2, teacher.search(state, depth, style)))
                    end
                end
            end)
        end

        it("breaks ties towards the lowest square index", function()
            -- Every placement here flips one disc, so greedy scores them
            -- all alike at depth 1 and the tie is what decides.
            local state = position({
                "......",
                "......",
                "..BW..",
                "..WB..",
                "......",
                "......",
            }, "black")
            local legal = othello.legal_actions(state)
            local values = {}
            for _, action in ipairs(legal) do
                values[#values + 1] = teacher.evaluate(othello.apply(state, action), "greedy")
            end
            for i = 2, #values do
                expect(values[i]).to.equal(values[1])
            end
            expect(teacher.search(state, 1, "greedy")).to.equal(legal[1])
        end)
    end)

    -- If the depth did not change the answer there would be nothing for
    -- V2 (the depth ladder) to measure, so a position where it does is
    -- pinned rather than assumed.
    describe("the depth changes the answer", function()
        it("greedy looks past the biggest immediate flip", function()
            local state = three_rows()
            expect(teacher.search(state, 1, "greedy")).to.equal(square(4, 1))
            expect(teacher.search(state, 2, "greedy")).to.equal(square(4, 0))
        end)

        it("mobility picks a different square one ply deeper", function()
            local state = position({
                "......",
                "......",
                ".BBB..",
                "..BW..",
                "......",
                "......",
            }, "white")
            expect(teacher.search(state, 1, "mobility")).to.equal(square(1, 1))
            expect(teacher.search(state, 2, "mobility")).to.equal(square(1, 3))
        end)
    end)

    -- Likewise for V3 (style separation): the three evaluation functions
    -- have to be able to disagree at all.
    describe("the style changes the answer", function()
        it("splits three ways on one position", function()
            local state = three_rows()
            local corner = teacher.search(state, 2, "corner")
            local mobility = teacher.search(state, 2, "mobility")
            local greedy = teacher.search(state, 2, "greedy")
            expect(corner).to.equal(square(4, 3))
            expect(mobility).to.equal(square(4, 2))
            expect(greedy).to.equal(square(4, 0))
            expect(corner == mobility).to.equal(false)
            expect(corner == greedy).to.equal(false)
            expect(mobility == greedy).to.equal(false)
        end)
    end)

    describe("evaluate", function()
        it("answers zero on the symmetric opening under every style", function()
            for _, style in ipairs(othello.STYLES) do
                expect(teacher.evaluate(opening(), style)).to.equal(0)
            end
        end)

        it("is negated when the other side is to move", function()
            local boards = {
                { "...B..", "...B..", ".BWB..", "BBBW..", "...BW.", "......" },
                { "..WB..", "..BW.B", "WBBBB.", "BBBBB.", "...BW.", "......" },
                { "BBBBBB", "BBBBBB", "BBBBBB", "BBWWWW", "WWWWWW", "WWWWWW" },
            }
            for _, style in ipairs(othello.STYLES) do
                for _, rows in ipairs(boards) do
                    local black = teacher.evaluate(position(rows, "black"), style)
                    local white = teacher.evaluate(position(rows, "white"), style)
                    expect(black).to.equal(-white)
                end
            end
        end)

        describe("corner", function()
            local function single(row, col)
                local rows = { "......", "......", "......", "......", "......", "......" }
                local line = rows[row + 1]
                rows[row + 1] = line:sub(1, col) .. "B" .. line:sub(col + 2)
                return position(rows, "black")
            end

            it("pays for a corner and charges for the squares beside it", function()
                expect(teacher.evaluate(single(0, 0), "corner")).to.equal(30)
                expect(teacher.evaluate(single(1, 1), "corner")).to.equal(-12)
                expect(teacher.evaluate(single(0, 1), "corner")).to.equal(-12)
                expect(teacher.evaluate(single(1, 0), "corner")).to.equal(-12)
                expect(teacher.evaluate(single(0, 2), "corner")).to.equal(5)
                expect(teacher.evaluate(single(2, 2), "corner")).to.equal(1)
            end)

            it("charges the same weights to the opponent", function()
                local rows = { "W.....", "......", "......", "......", "......", "......" }
                expect(teacher.evaluate(position(rows, "black"), "corner")).to.equal(-30)
            end)

            it("does not count discs", function()
                local one_corner = single(0, 0)
                local four_middle = position({
                    "......",
                    "......",
                    "..BB..",
                    "..BB..",
                    "......",
                    "......",
                }, "black")
                expect(teacher.evaluate(one_corner, "corner")).to.equal(30)
                expect(teacher.evaluate(four_middle, "corner")).to.equal(4)
                -- Fewer discs, higher corner score: the disc count is not
                -- part of this style.
                expect(teacher.evaluate(one_corner, "greedy")).to.equal(1)
                expect(teacher.evaluate(four_middle, "greedy")).to.equal(4)
            end)
        end)

        describe("mobility", function()
            it("scores the extra placement when the disc counts are level", function()
                -- Three discs each, black has five placements to white's
                -- four: 1 * 10 for the move, 0 for the discs.
                local state = position({
                    "......",
                    "......",
                    "..WWW.",
                    "..BB..",
                    "...B..",
                    "......",
                }, "black")
                expect(teacher.evaluate(state, "greedy")).to.equal(0)
                expect(teacher.evaluate(state, "mobility")).to.equal(10)
            end)

            it("does not read a forced pass as a move", function()
                -- White has no placement at all here, so the difference is
                -- the whole of black's count rather than one less.
                local state = position({
                    "BBB...",
                    "BBB...",
                    "......",
                    "......",
                    "......",
                    "......",
                }, "black")
                expect(#othello.legal_actions(state)).to.equal(1)
                expect(othello.legal_actions(state)[1]).to.equal(othello.PASS)
                expect(teacher.evaluate(state, "mobility")).to.equal(6)
            end)
        end)

        describe("greedy", function()
            it("answers the disc difference and nothing else", function()
                local rows = { "BBB...", "W.....", "......", "......", "......", "......" }
                expect(teacher.evaluate(position(rows, "black"), "greedy")).to.equal(2)
                expect(teacher.evaluate(position(rows, "white"), "greedy")).to.equal(-2)
            end)
        end)

        describe("a finished game", function()
            it("is scored by the result plus the margin, whatever the style", function()
                for _, style in ipairs(othello.STYLES) do
                    expect(teacher.evaluate(finished("black"), style)).to.equal(
                        teacher.WIN_SCORE + 4
                    )
                    expect(teacher.evaluate(finished("white"), style)).to.equal(
                        -teacher.WIN_SCORE - 4
                    )
                end
            end)

            it("outranks every unfinished position", function()
                for _, style in ipairs(othello.STYLES) do
                    local won = teacher.evaluate(finished("black"), style)
                    local lost = teacher.evaluate(finished("white"), style)
                    expect(won > lost).to.equal(true)
                    expect(won > teacher.evaluate(midgame(), style)).to.equal(true)
                    expect(lost < teacher.evaluate(midgame(), style)).to.equal(true)
                end
            end)

            it("calls a full board with equal discs a draw", function()
                local drawn = position({
                    "BBBBBB",
                    "BBBBBB",
                    "BBBBBB",
                    "WWWWWW",
                    "WWWWWW",
                    "WWWWWW",
                }, "black")
                expect(othello.is_over(drawn)).to.equal(true)
                expect(teacher.evaluate(drawn, "greedy")).to.equal(0)
            end)
        end)
    end)

    describe("search stops at a finished game", function()
        for _, style in ipairs(othello.STYLES) do
            it("has no move to answer with on " .. style, function()
                for _, depth in ipairs(teacher.DEPTHS) do
                    local action, value = teacher.search(finished("black"), depth, style)
                    expect(action).to.equal(nil)
                    expect(value).to.equal(teacher.evaluate(finished("black"), style))
                end
            end)
        end

        it("does not search past the result", function()
            -- One ply from the end of the board: the reply fills it and
            -- the deeper search can only report the same finish.
            local nearly = position({
                "BBBBBB",
                "BBBBBB",
                "BBBBBB",
                "BBWWWW",
                "WWWWWW",
                "WWWWW.",
            }, "black")
            expect(othello.is_over(nearly)).to.equal(false)
            local shallow_action, shallow_value = teacher.search(nearly, 2, "greedy")
            local deep_action, deep_value = teacher.search(nearly, 6, "greedy")
            expect(deep_action).to.equal(shallow_action)
            expect(deep_value).to.equal(shallow_value)
        end)
    end)

    describe("passes", function()
        it("answers the pass when there is no placement", function()
            for _, style in ipairs(othello.STYLES) do
                for _, depth in ipairs(teacher.DEPTHS) do
                    expect(teacher.search(blocked(), depth, style)).to.equal(othello.PASS)
                end
            end
        end)

        it("ends the game on the second pass instead of recursing", function()
            -- White passes, black passes, the game is over with black six
            -- discs up. Every depth from two on has to see the same
            -- finish rather than walk a chain of passes.
            for _, style in ipairs(othello.STYLES) do
                for _, depth in ipairs({ 2, 4, 6 }) do
                    local action, value = teacher.search(blocked(), depth, style)
                    expect(action).to.equal(othello.PASS)
                    expect(value).to.equal(-teacher.WIN_SCORE - 6)
                end
            end
        end)

        it("spends a ply on the pass like any other move", function()
            -- Depth one cannot reach the finish that depth two reaches,
            -- so the pass is not a free move.
            local shallow = select(2, teacher.search(blocked(), 1, "greedy"))
            local deeper = select(2, teacher.search(blocked(), 2, "greedy"))
            expect(shallow > deeper).to.equal(true)
        end)
    end)

    describe("policy", function()
        it("answers what search answers", function()
            for _, style in ipairs(othello.STYLES) do
                for _, depth in ipairs({ 1, 2, 4 }) do
                    local policy = teacher.policy(depth, style)
                    for _, case in ipairs(CROSS_CHECK) do
                        expect(policy(case.make())).to.equal(
                            (teacher.search(case.make(), depth, style))
                        )
                    end
                end
            end
        end)

        it("holds no state between calls", function()
            local policy = teacher.policy(2, "corner")
            local state = midgame()
            local first = policy(state)
            for _ = 1, 10 do
                expect(policy(state)).to.equal(first)
            end
        end)

        it("leaves the position it was handed untouched", function()
            local state = midgame()
            local before = othello.encode(state)
            teacher.policy(4, "mobility")(state)
            expect(othello.encode(state)).to.equal(before)
        end)

        it("refuses to label a finished game", function()
            expect(function()
                teacher.policy(2, "greedy")(finished("black"))
            end).to.fail()
        end)
    end)

    describe("every answer is a legal move", function()
        for _, case in ipairs(CROSS_CHECK) do
            it("on " .. case.name, function()
                for _, style in ipairs(othello.STYLES) do
                    for _, depth in ipairs(teacher.DEPTHS) do
                        local state = case.make()
                        local action = teacher.search(state, depth, style)
                        local legal = othello.legal_actions(state)
                        local found = false
                        for _, candidate in ipairs(legal) do
                            if candidate == action then
                                found = true
                            end
                        end
                        expect(found).to.equal(true)
                        -- And the rules module accepts it.
                        expect(function()
                            othello.apply(state, action)
                        end).to_not.fail()
                    end
                end
            end)
        end

        it("plays a whole game against itself without an illegal move", function()
            local state = opening()
            local black = teacher.policy(2, "corner")
            local white = teacher.policy(2, "greedy")
            local plies = 0
            while not othello.is_over(state) do
                local action = othello.side_to_move(state) == "black" and black(state)
                    or white(state)
                state = othello.apply(state, action)
                plies = plies + 1
                expect(plies <= othello.CELLS).to.equal(true)
            end
            expect(othello.winner(state)).to_not.equal(nil)
        end)
    end)
end)
