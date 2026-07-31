-- guardian_duel/spec/guardian_duel_spec.lua
--
-- Package-level spec for the guardian duel rules. Run with
-- `alc_pkg_test pkg="guardian_duel"` after `alc_pkg_link` has
-- registered the package. The `lust` globals are pre-loaded by the
-- runner.
--
-- The numbers pinned here are the demo's own balance: every damage,
-- block and threshold value the rules use is asserted somewhere below,
-- so a tweak to one of them shows up as a failing expectation rather
-- than as a boss that quietly stops staggering.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("guardian_duel")

local CTX_LEN = 16

--- A boss state with every field present, overridden field by field.
local function boss_state(fields)
    local state = {
        cycle = 0,
        mode = 0,
        hp = duel.BOSS_MAX_HP,
        damage_since_shift = 0,
        last_player = 0,
        turn = 1,
        shifts = 0,
        block = 0,
        thorns = 0,
    }
    for key, value in pairs(fields or {}) do
        state[key] = value
    end
    return state
end

--- Play `turns` turns of a style against a blocking player, collecting
--- the boss answers. Blocking deals no damage, so the accumulated
--- damage stays where the caller put it.
local function walk(g, policy, turns)
    local actions = {}
    for _ = 1, turns do
        local action = policy(g.boss)
        actions[#actions + 1] = action
        g = duel.apply(g, "b", action)
    end
    return table.concat(actions), g
end

--- Decode a padded token row back to the training line it carries.
local function row_text(row)
    local v = duel.vocab()
    local chars = {}
    for _, id in ipairs(row) do
        if id == v.pad_id then
            break
        end
        chars[#chars + 1] = v.to_char[id]
    end
    return table.concat(chars)
end

describe("guardian_duel.new_game", function()
    it("opens with both sides at full health", function()
        local g = duel.new_game(1)
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP)
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP)
    end)

    it("opens at the head of the cycle in mode zero", function()
        local g = duel.new_game(1)
        expect(g.boss.cycle).to.equal(0)
        expect(g.boss.mode).to.equal(0)
        expect(g.boss.shifts).to.equal(0)
        expect(g.boss.turn).to.equal(1)
        expect(g.boss.last_player).to.equal(0)
    end)

    it("rejects a seed that is not a number", function()
        expect(function()
            duel.new_game("1")
        end).to.fail()
    end)
end)

-- ─── Encoding ───────────────────────────────────────────────────────

describe("guardian_duel.encode", function()
    it("writes the six fields in layout order", function()
        -- Twelve damage taken of the fifteen the teacher tolerates
        -- leaves three, which is one whole bucket of distance.
        local state = boss_state({
            cycle = 2,
            mode = 1,
            hp = 20,
            damage_since_shift = 12,
            last_player = 3,
            turn = 4,
        })
        expect(duel.encode(state, "guardian")).to.equal("C2M1H4D1L3T4")
    end)

    it("encodes the opening state", function()
        expect(duel.encode(duel.new_game(1).boss, "guardian")).to.equal("C0M0H9D3L0T1")
    end)

    it("measures the distance against the style", function()
        -- The same untouched boss stands three buckets from a teacher
        -- stagger, two from an impatient one and five from a stubborn
        -- one.
        local state = boss_state()
        expect(duel.encode(state, "guardian")).to.equal("C0M0H9D3L0T1")
        expect(duel.encode(state, "turtle")).to.equal("C0M0H9D2L0T1")
        expect(duel.encode(state, "rusher")).to.equal("C0M0H9D5L0T1")
    end)

    it("is exactly twelve chars on every state of a fight", function()
        local g = duel.new_game(21)
        local rng = alc.math.rng_create(5)
        while not duel.is_over(g) do
            local encoded = duel.encode(g.boss, "guardian")
            expect(#encoded).to.equal(duel.ENCODED_LEN)
            expect(#encoded).to.equal(12)
            g = duel.apply(g, duel.policy_player_random(rng), duel.policy_guardian(g.boss))
        end
    end)

    it("keeps prompt plus action inside the tiny preset context", function()
        local g = duel.new_game(22)
        local rng = alc.math.rng_create(9)
        while not duel.is_over(g) do
            local prompt = duel.encode(g.boss, "turtle") .. ">"
            expect(#duel.to_ids(prompt) + 1 <= duel.CTX_BUDGET).to.equal(true)
            g = duel.apply(g, duel.policy_player_random(rng), duel.policy_turtle(g.boss))
        end
    end)

    it("rejects a turn outside the fight", function()
        -- Turn ten would need two chars and blow the twelve-char layout.
        expect(function()
            duel.encode(boss_state({ turn = duel.TURN_LIMIT + 1 }), "guardian")
        end).to.fail()
    end)

    it("rejects a player move index the rules cannot produce", function()
        expect(function()
            duel.encode(boss_state({ last_player = 5 }), "guardian")
        end).to.fail()
    end)

    it("rejects health above the maximum and damage below zero", function()
        expect(function()
            duel.encode(boss_state({ hp = duel.BOSS_MAX_HP + 1 }), "guardian")
        end).to.fail()
        expect(function()
            duel.encode(boss_state({ damage_since_shift = -1 }), "guardian")
        end).to.fail()
    end)

    it("rejects a cycle index outside the cycle", function()
        expect(function()
            duel.encode(boss_state({ cycle = duel.CYCLE_LEN }), "guardian")
        end).to.fail()
    end)

    it("rejects a state that is not a table", function()
        expect(function()
            duel.encode("C0M0H9D3L0T1", "guardian")
        end).to.fail()
    end)

    it("rejects a missing or unknown style", function()
        -- Guessing a threshold would hand the model a distance its
        -- labels do not follow.
        expect(function()
            duel.encode(boss_state())
        end).to.fail()
        expect(function()
            duel.encode(boss_state(), "berserker")
        end).to.fail()
    end)
end)

describe("guardian_duel buckets", function()
    it("keeps bucket zero for an exhausted side", function()
        expect(duel.hp_bucket(0)).to.equal(0)
        expect(duel.hp_bucket(1)).to.equal(1)
        expect(duel.hp_bucket(duel.BOSS_MAX_HP)).to.equal(9)
    end)

    it("keeps bucket zero for a shift that is due", function()
        -- Any remaining sliver rounds up, so `D0` never means "nearly".
        expect(duel.distance_bucket(0)).to.equal(0)
        expect(duel.distance_bucket(1)).to.equal(1)
        expect(duel.distance_bucket(duel.DAMAGE_BUCKET_SIZE)).to.equal(1)
        expect(duel.distance_bucket(duel.DAMAGE_BUCKET_SIZE + 1)).to.equal(2)
        expect(duel.distance_bucket(9 * duel.DAMAGE_BUCKET_SIZE)).to.equal(9)
    end)

    it("reads a style threshold back in damage", function()
        expect(duel.threshold_damage("guardian", 0)).to.equal(15)
        expect(duel.threshold_damage("guardian", 1)).to.equal(20)
        expect(duel.threshold_damage("turtle", 0)).to.equal(10)
        expect(duel.threshold_damage("rusher", 0)).to.equal(25)
    end)

    it("reports the distance left to a shift", function()
        local state = boss_state({ damage_since_shift = 12 })
        expect(duel.shift_distance("guardian", state)).to.equal(3)
        expect(duel.shift_distance("rusher", state)).to.equal(13)
        -- Past the threshold the distance is zero rather than negative.
        expect(duel.shift_distance("turtle", state)).to.equal(0)
    end)

    it("rejects health the rules cannot reach", function()
        expect(function()
            duel.hp_bucket(duel.BOSS_MAX_HP + 1)
        end).to.fail()
        expect(function()
            duel.hp_bucket(-1)
        end).to.fail()
    end)

    it("rejects a distance the rules cannot reach", function()
        expect(function()
            duel.distance_bucket(-1)
        end).to.fail()
    end)
end)

describe("guardian_duel.legal_actions", function()
    it("withholds the twin slam until the boss has spikes up", function()
        local legal = duel.legal_actions(boss_state({ mode = 0 }))
        expect(#legal).to.equal(5)
        expect(table.concat(legal)).to.equal("cfvwd")
    end)

    it("allows the twin slam in mode one", function()
        local legal = duel.legal_actions(boss_state({ mode = 1 }))
        expect(#legal).to.equal(6)
        expect(table.concat(legal)).to.equal("cfvwdt")
    end)

    it("gives the player all four moves", function()
        expect(table.concat(duel.player_legal_actions())).to.equal("aAbp")
    end)

    it("rejects a state without a mode", function()
        expect(function()
            duel.legal_actions({})
        end).to.fail()
    end)
end)

-- ─── Cycle and mode shift ───────────────────────────────────────────

describe("guardian_duel.policy_guardian cycle", function()
    it("walks charge, fierce, vent, whirlwind and wraps", function()
        local actions = walk(duel.new_game(1), duel.policy_guardian, 5)
        expect(actions).to.equal("cfvwc")
    end)

    it("advances the cycle index one step per turn", function()
        local g = duel.new_game(1)
        expect(g.boss.cycle).to.equal(0)
        g = duel.apply(g, "b", "c")
        expect(g.boss.cycle).to.equal(1)
        g = duel.apply(g, "b", "f")
        expect(g.boss.cycle).to.equal(2)
        g = duel.apply(g, "b", "v")
        expect(g.boss.cycle).to.equal(3)
        g = duel.apply(g, "b", "w")
        expect(g.boss.cycle).to.equal(0)
    end)
end)

describe("guardian_duel mode shift threshold", function()
    --- The `D` character of a state as the model sees it.
    local function distance_char(state, style)
        return duel.encode(state, style):sub(8, 8)
    end

    -- The teacher tolerates fifteen damage before its first stagger, so
    -- one bucket of distance is the last rung it walks its cycle on.
    it("keeps walking the cycle one bucket below the threshold", function()
        local state = boss_state({ damage_since_shift = 10 })
        expect(distance_char(state, "guardian")).to.equal("1")
        expect(duel.policy_guardian(state)).to.equal("c")
    end)

    it("shifts on the exact threshold", function()
        local state = boss_state({ damage_since_shift = 15 })
        expect(distance_char(state, "guardian")).to.equal("0")
        expect(duel.policy_guardian(state)).to.equal("d")
    end)

    it("shifts above the threshold", function()
        local state = boss_state({ damage_since_shift = 22 })
        expect(distance_char(state, "guardian")).to.equal("0")
        expect(duel.policy_guardian(state)).to.equal("d")
    end)

    it("interrupts the cycle wherever it stands", function()
        local state = boss_state({ cycle = 2, damage_since_shift = 22 })
        expect(duel.policy_guardian(state)).to.equal("d")
    end)

    it("raises the threshold by one for every completed shift", function()
        expect(duel.shift_threshold("guardian", 0)).to.equal(3)
        expect(duel.shift_threshold("guardian", 1)).to.equal(4)
        expect(duel.shift_threshold("guardian", 2)).to.equal(5)
    end)

    it("no longer shifts on the old threshold after one shift", function()
        -- The same fifteen damage that was due before the first shift
        -- is a bucket short of the raised threshold, and the encoding
        -- says so: `D0` becomes `D1`.
        local state = boss_state({ shifts = 1, damage_since_shift = 15 })
        expect(distance_char(state, "guardian")).to.equal("1")
        expect(duel.policy_guardian(state)).to.equal("c")
        state.damage_since_shift = 20
        expect(distance_char(state, "guardian")).to.equal("0")
        expect(duel.policy_guardian(state)).to.equal("d")
    end)

    it("moves the distance up when the threshold rises", function()
        local before = boss_state({ shifts = 0, damage_since_shift = 15 })
        local after = boss_state({ shifts = 1, damage_since_shift = 15 })
        expect(distance_char(before, "guardian")).to.equal("0")
        expect(distance_char(after, "guardian")).to.equal("1")
        -- Same accumulated damage, opposite answers — which is exactly
        -- why the field cannot carry the accumulation itself.
        expect(duel.policy_guardian(before)).to.equal("d")
        expect(duel.policy_guardian(after)).to.equal("c")
    end)

    it("rejects a shift count the fight cannot reach", function()
        expect(function()
            duel.shift_threshold("guardian", -1)
        end).to.fail()
    end)

    it("rejects an unknown style", function()
        expect(function()
            duel.shift_threshold("berserker", 0)
        end).to.fail()
    end)
end)

describe("guardian_duel encoding as a sufficient statistic", function()
    --- A state at `distance` damage from the next shift of `style`.
    local function at_distance(style, shifts, distance)
        return boss_state({
            hp = 30,
            shifts = shifts,
            damage_since_shift = duel.threshold_damage(style, shifts) - distance,
        })
    end

    it("gives one answer per encoded state below the threshold", function()
        -- Four different shift counts, one bucket of distance each: the
        -- twelve characters are identical, so the label must be too.
        local encoded, answer
        for shifts = 0, 3 do
            local state = at_distance("guardian", shifts, duel.DAMAGE_BUCKET_SIZE)
            local text = duel.encode(state, "guardian")
            local action = duel.policy_guardian(state)
            expect(text).to.equal(encoded or text)
            expect(action).to.equal(answer or action)
            encoded, answer = text, action
        end
        expect(encoded).to.equal("C0M0H6D1L0T1")
        expect(answer).to.equal("c")
    end)

    it("gives one answer per encoded state on the threshold", function()
        local encoded, answer
        for shifts = 0, 3 do
            local state = at_distance("guardian", shifts, 0)
            local text = duel.encode(state, "guardian")
            local action = duel.policy_guardian(state)
            expect(text).to.equal(encoded or text)
            expect(action).to.equal(answer or action)
            encoded, answer = text, action
        end
        expect(encoded).to.equal("C0M0H6D0L0T1")
        expect(answer).to.equal("d")
    end)

    it("holds for every style", function()
        for _, style in ipairs(duel.STYLES) do
            local policy = duel["policy_" .. style]
            local due = at_distance(style, 2, 0)
            local not_due = at_distance(style, 2, duel.DAMAGE_BUCKET_SIZE)
            expect(duel.encode(due, style)).to.equal("C0M0H6D0L0T1")
            expect(duel.encode(not_due, style)).to.equal("C0M0H6D1L0T1")
            expect(policy(due)).to.equal("d")
            expect(policy(not_due)).to.equal(duel.style_cycle(style)[1])
        end
    end)
end)

describe("guardian_duel defensive sub-sequence", function()
    it("answers defensive, defensive, twin slam", function()
        expect(duel.policy_guardian(boss_state({ mode = 1, cycle = 0 }))).to.equal("d")
        expect(duel.policy_guardian(boss_state({ mode = 1, cycle = 1 }))).to.equal("d")
        expect(duel.policy_guardian(boss_state({ mode = 1, cycle = 2 }))).to.equal("t")
    end)

    it("still slams past the end of the sequence", function()
        expect(duel.policy_guardian(boss_state({ mode = 1, cycle = 3 }))).to.equal("t")
    end)

    it("ignores the damage counter while it is rolled up", function()
        local state = boss_state({ mode = 1, cycle = 2, damage_since_shift = 40 })
        expect(duel.policy_guardian(state)).to.equal("t")
    end)

    it("runs the whole shift and returns to the cycle", function()
        local g = duel.new_game(2)
        g.boss.damage_since_shift = 15
        local actions
        actions, g = walk(g, duel.policy_guardian, 3)
        expect(actions).to.equal("ddt")
        expect(g.boss.mode).to.equal(0)
        expect(g.boss.cycle).to.equal(0)
        expect(g.boss.shifts).to.equal(1)
        expect(g.boss.damage_since_shift).to.equal(0)
        expect(g.boss.thorns).to.equal(0)
        -- The slam is the only boss move that reaches through a block.
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 4)
    end)

    it("resumes the cycle from its head after the slam", function()
        local g = duel.new_game(2)
        g.boss.damage_since_shift = 15
        local _
        _, g = walk(g, duel.policy_guardian, 3)
        local actions = walk(g, duel.policy_guardian, 2)
        expect(actions).to.equal("cf")
    end)

    it("enters the sequence at its second step", function()
        local g = duel.apply(duel.new_game(2), "b", "d")
        expect(g.boss.mode).to.equal(1)
        expect(g.boss.cycle).to.equal(1)
        expect(g.boss.thorns).to.equal(3)
    end)
end)

-- ─── Style zoo ──────────────────────────────────────────────────────

describe("guardian_duel.STYLES", function()
    it("exports a policy for every canonical style", function()
        expect(#duel.STYLES).to.equal(3)
        for _, style in ipairs(duel.STYLES) do
            expect(type(duel["policy_" .. style])).to.equal("function")
        end
    end)

    it("gives every style its own cycle", function()
        expect(table.concat(duel.style_cycle("guardian"))).to.equal("cfvw")
        expect(table.concat(duel.style_cycle("rusher"))).to.equal("fwfv")
        expect(table.concat(duel.style_cycle("turtle"))).to.equal("cvcw")
    end)

    it("hands back a copy of the cycle", function()
        local cycle = duel.style_cycle("guardian")
        cycle[1] = "t"
        expect(duel.style_cycle("guardian")[1]).to.equal("c")
    end)

    it("walks the offensive cycle as rusher", function()
        local actions = walk(duel.new_game(3), duel.policy_rusher, 4)
        expect(actions).to.equal("fwfv")
    end)

    it("walks the defensive cycle as turtle", function()
        local actions = walk(duel.new_game(3), duel.policy_turtle, 4)
        expect(actions).to.equal("cvcw")
    end)

    it("staggers earliest as turtle and latest as rusher", function()
        -- Two damage buckets: only the turtle has seen enough.
        local early = boss_state({ damage_since_shift = 10 })
        expect(duel.policy_turtle(early)).to.equal("d")
        expect(duel.policy_guardian(early)).to.equal("c")
        expect(duel.policy_rusher(early)).to.equal("f")

        -- Three buckets: the teacher joins in, the rusher pushes on.
        local mid = boss_state({ damage_since_shift = 15 })
        expect(duel.policy_turtle(mid)).to.equal("d")
        expect(duel.policy_guardian(mid)).to.equal("d")
        expect(duel.policy_rusher(mid)).to.equal("f")

        -- Five buckets: even the rusher rolls up.
        local late = boss_state({ damage_since_shift = 25 })
        expect(duel.policy_rusher(late)).to.equal("d")
    end)

    it("orders the style thresholds turtle, guardian, rusher", function()
        expect(duel.shift_threshold("turtle", 0)).to.equal(2)
        expect(duel.shift_threshold("guardian", 0)).to.equal(3)
        expect(duel.shift_threshold("rusher", 0)).to.equal(5)
    end)

    it("is deterministic for every style", function()
        local state = boss_state({ cycle = 1, hp = 30, damage_since_shift = 8, turn = 4 })
        for _, style in ipairs(duel.STYLES) do
            local policy = duel["policy_" .. style]
            expect(policy(state)).to.equal(policy(state))
        end
    end)

    it("rejects a state that is missing a field", function()
        local state = boss_state()
        state.shifts = nil
        expect(function()
            duel.policy_guardian(state)
        end).to.fail()
    end)

    it("rejects a mode the rules do not have", function()
        expect(function()
            duel.policy_guardian(boss_state({ mode = 2 }))
        end).to.fail()
    end)
end)

-- ─── Turn resolution ────────────────────────────────────────────────

describe("guardian_duel.apply", function()
    it("charges through the player's next attack", function()
        local g = duel.apply(duel.new_game(4), "b", "c")
        expect(g.boss.block).to.equal(5)
        -- A light attack of four is fully absorbed by the charge.
        g = duel.apply(g, "a", "f")
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP)
        expect(g.boss.damage_since_shift).to.equal(0)
    end)

    it("punishes the turn after a heavy attack", function()
        local g = duel.apply(duel.new_game(3), "A", "c")
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP - 9)
        expect(g.boss.damage_since_shift).to.equal(9)
        -- Fierce is nine, plus three because the player swung heavy.
        g = duel.apply(g, "a", "f")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 12)
    end)

    it("does not turn a zero-damage move into a punish", function()
        local g = duel.apply(duel.new_game(3), "A", "f")
        g = duel.apply(g, "b", "c")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 9)
    end)

    it("blocks six of the incoming damage", function()
        local g = duel.apply(duel.new_game(6), "b", "f")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 3)
    end)

    it("reflects the spikes on an attack but not on a block", function()
        local g = duel.apply(duel.new_game(4), "b", "d")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP)
        g = duel.apply(g, "a", "d")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 3)
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP - 4)
        g = duel.apply(g, "b", "t")
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP - 3 - 4)
    end)

    it("weakens the player's next attack after a vent", function()
        local g = duel.apply(duel.new_game(5), "b", "v")
        expect(g.player.weak).to.equal(true)
        -- A light attack of four lands as two while weakened.
        g = duel.apply(g, "a", "c")
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP - 2)
        expect(g.player.weak).to.equal(false)
    end)

    it("records the player's last move as its enum index", function()
        local g = duel.apply(duel.new_game(7), "p", "c")
        expect(g.boss.last_player).to.equal(4)
        expect(g.boss.hp).to.equal(duel.BOSS_MAX_HP - 2)
    end)

    it("marks the turn after a poke as revealed", function()
        local g = duel.apply(duel.new_game(7), "p", "c")
        expect(g.revealed).to.equal(true)
        g = duel.apply(g, "a", "f")
        expect(g.revealed).to.equal(false)
    end)

    it("lets a felled boss skip its answer", function()
        local g = duel.new_game(8)
        g.boss.hp = 4
        g = duel.apply(g, "a", "f")
        expect(g.boss.hp).to.equal(0)
        expect(g.player.hp).to.equal(duel.PLAYER_MAX_HP)
        expect(duel.winner(g)).to.equal("player")
    end)

    it("rejects a player move outside the four", function()
        expect(function()
            duel.apply(duel.new_game(9), "x", "c")
        end).to.fail()
    end)

    it("rejects a twin slam without spikes", function()
        expect(function()
            duel.apply(duel.new_game(9), "b", "t")
        end).to.fail()
    end)

    it("rejects a boss move outside the six", function()
        expect(function()
            duel.apply(duel.new_game(9), "b", "z")
        end).to.fail()
    end)

    it("rejects a turn played after the fight ended", function()
        local g = duel.new_game(10)
        for _ = 1, duel.TURN_LIMIT do
            g = duel.apply(g, "b", "c")
        end
        expect(duel.is_over(g)).to.equal(true)
        expect(function()
            duel.apply(g, "b", "c")
        end).to.fail()
    end)
end)

describe("guardian_duel.winner", function()
    it("has no winner before the fight ends", function()
        expect(duel.winner(duel.new_game(11))).to.equal(nil)
    end)

    it("names the boss when the player goes down", function()
        local g = duel.new_game(12)
        g.player.hp = 6
        g = duel.apply(g, "a", "f")
        expect(g.player.hp).to.equal(0)
        expect(duel.winner(g)).to.equal("boss")
    end)

    it("compares health buckets at the turn limit", function()
        local g = duel.new_game(13)
        for _ = 1, duel.TURN_LIMIT do
            g = duel.apply(g, "b", "c")
        end
        expect(duel.winner(g)).to.equal("draw")
    end)

    it("names the healthier side at the turn limit", function()
        local g = duel.new_game(14)
        for _ = 1, duel.TURN_LIMIT do
            g = duel.apply(g, "b", "f")
        end
        -- The blocked fierce still costs three a turn, so the boss ends
        -- the fight in the higher bucket.
        expect(duel.winner(g)).to.equal("boss")
    end)
end)

describe("guardian_duel game loop", function()
    it("terminates and names a winner", function()
        local g = duel.new_game(15)
        local rng = alc.math.rng_create(16)
        while not duel.is_over(g) do
            g = duel.apply(g, duel.policy_player_random(rng), duel.policy_guardian(g.boss))
        end
        local w = duel.winner(g)
        expect(w == "player" or w == "boss" or w == "draw").to.equal(true)
    end)

    it("only ever answers with a legal boss move", function()
        local g = duel.new_game(17)
        local rng = alc.math.rng_create(18)
        while not duel.is_over(g) do
            local action = duel.policy_rusher(g.boss)
            local legal = duel.legal_actions(g.boss)
            local found = false
            for _, ch in ipairs(legal) do
                found = found or ch == action
            end
            expect(found).to.equal(true)
            g = duel.apply(g, duel.policy_player_random(rng), action)
        end
    end)
end)

-- ─── Corpus ─────────────────────────────────────────────────────────

describe("guardian_duel.build_corpus", function()
    it("emits at most one row per turn of every playout", function()
        local rows = duel.build_corpus(duel.policy_guardian, {
            ctx_len = CTX_LEN,
            games = 3,
            style = "guardian",
            seed = 11,
        })
        expect(#rows >= 3).to.equal(true)
        expect(#rows <= 3 * duel.MAX_ROWS_PER_GAME).to.equal(true)
    end)

    it("writes state, separator and boss move on every line", function()
        local rows = duel.build_corpus(duel.policy_turtle, {
            ctx_len = CTX_LEN,
            games = 2,
            style = "turtle",
            seed = 13,
        })
        for _, row in ipairs(rows) do
            local text = row_text(row)
            expect(text:match("^C%dM%dH%dD%dL%dT%d>[cfvwdt]\n$") ~= nil).to.equal(true)
        end
    end)

    it("pads every row to the context window", function()
        local rows = duel.build_corpus(duel.policy_rusher, {
            ctx_len = CTX_LEN,
            games = 2,
            style = "rusher",
            seed = 17,
        })
        local pad_id = duel.vocab().pad_id
        for _, row in ipairs(rows) do
            expect(#row).to.equal(CTX_LEN)
            expect(row[#row]).to.equal(pad_id)
        end
    end)

    it("honours a pad id override", function()
        local rows = duel.build_corpus(duel.policy_guardian, {
            ctx_len = CTX_LEN,
            games = 1,
            style = "guardian",
            seed = 19,
            pad_id = 2,
        })
        expect(rows[1][#rows[1]]).to.equal(2)
    end)

    it("is reproducible from the seed", function()
        local opts = { ctx_len = CTX_LEN, games = 2, style = "guardian", seed = 5 }
        local a = duel.build_corpus(duel.policy_guardian, opts)
        local b = duel.build_corpus(duel.policy_guardian, opts)
        expect(#a).to.equal(#b)
        expect(row_text(a[1])).to.equal(row_text(b[1]))
        expect(row_text(a[#a])).to.equal(row_text(b[#b]))
    end)

    it("opens every playout on the opening state", function()
        local rows = duel.build_corpus(duel.policy_guardian, {
            ctx_len = CTX_LEN,
            games = 1,
            style = "guardian",
            seed = 23,
        })
        expect(row_text(rows[1])).to.equal("C0M0H9D3L0T1>c\n")
    end)

    it("measures the distance against the labelling style", function()
        -- The impatient boss opens two buckets from its stagger, so its
        -- corpus opens on `D2` where the teacher's opens on `D3`.
        local rows = duel.build_corpus(duel.policy_turtle, {
            ctx_len = CTX_LEN,
            games = 1,
            style = "turtle",
            seed = 23,
        })
        expect(row_text(rows[1])).to.equal("C0M0H9D2L0T1>c\n")
    end)

    it("refuses a context window the line does not fit in", function()
        -- Truncating instead would teach the model a state it can never
        -- be asked about at decode time.
        expect(function()
            duel.build_corpus(duel.policy_guardian, {
                ctx_len = 4,
                games = 1,
                style = "guardian",
                seed = 23,
            })
        end).to.fail()
    end)

    it("rejects a policy that is not a function", function()
        expect(function()
            duel.build_corpus("guardian", {
                ctx_len = CTX_LEN,
                games = 1,
                style = "guardian",
                seed = 23,
            })
        end).to.fail()
    end)

    it("rejects a missing context window or game count", function()
        expect(function()
            duel.build_corpus(duel.policy_guardian, { games = 1, style = "guardian" })
        end).to.fail()
        expect(function()
            duel.build_corpus(duel.policy_guardian, {
                ctx_len = CTX_LEN,
                games = 0,
                style = "guardian",
            })
        end).to.fail()
    end)

    it("rejects a missing or unknown distance basis", function()
        -- Defaulting the style would encode a distance the labels do
        -- not follow.
        expect(function()
            duel.build_corpus(duel.policy_guardian, { ctx_len = CTX_LEN, games = 1 })
        end).to.fail()
        expect(function()
            duel.build_corpus(duel.policy_guardian, {
                ctx_len = CTX_LEN,
                games = 1,
                style = "berserker",
            })
        end).to.fail()
    end)

    it("rejects a pad id that is not a number", function()
        expect(function()
            duel.build_corpus(duel.policy_guardian, {
                ctx_len = CTX_LEN,
                games = 1,
                style = "guardian",
                pad_id = "0",
            })
        end).to.fail()
    end)
end)

describe("guardian_duel.sample_states", function()
    it("collects one state per turn played", function()
        local states = duel.sample_states({ games = 4, seed = 3 })
        expect(#states >= 4).to.equal(true)
        expect(#states <= 4 * duel.MAX_ROWS_PER_GAME).to.equal(true)
    end)

    it("collects states the encoder accepts", function()
        for _, state in ipairs(duel.sample_states({ games = 4, seed = 4 })) do
            expect(#duel.encode(state, "guardian")).to.equal(duel.ENCODED_LEN)
        end
    end)

    it("reaches the rolled-up mode", function()
        -- A random boss walks into mode one on its own; without that the
        -- validation batch would never test the defensive branch. The
        -- sweep over several seeds keeps the check off a single lucky
        -- stream.
        local seen = false
        for seed = 1, 5 do
            for _, state in ipairs(duel.sample_states({ games = 4, seed = seed })) do
                seen = seen or state.mode == 1
            end
        end
        expect(seen).to.equal(true)
    end)

    it("is reproducible from the seed", function()
        local a = duel.sample_states({ games = 2, seed = 29 })
        local b = duel.sample_states({ games = 2, seed = 29 })
        expect(#a).to.equal(#b)
        for i, state in ipairs(a) do
            expect(duel.encode(state, "guardian")).to.equal(duel.encode(b[i], "guardian"))
        end
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            duel.sample_states({ games = 0 })
        end).to.fail()
    end)
end)

-- ─── Synthesised policies ───────────────────────────────────────────

describe("guardian_duel.compile_policy", function()
    local states = { boss_state(), boss_state({ mode = 1, cycle = 2, hp = 20 }) }

    it("accepts a chunk that answers legally and deterministically", function()
        local policy = duel.compile_policy("return function(state) return 'c' end", {
            games = 2,
            seed = 31,
        })
        expect(policy(states[1])).to.equal("c")
    end)

    it("accepts a chunk that reads the state", function()
        local source = [[
return function(state)
    if state.mode == 1 then
        return "t"
    end
    return "w"
end
]]
        local policy = duel.compile_policy(source, { states = states })
        expect(policy(states[1])).to.equal("w")
        expect(policy(states[2])).to.equal("t")
    end)

    it("rejects a chunk that does not compile", function()
        expect(function()
            duel.compile_policy("return function(state", { states = states })
        end).to.fail()
    end)

    it("rejects a chunk that does not return a function", function()
        expect(function()
            duel.compile_policy("return 7", { states = states })
        end).to.fail()
    end)

    it("rejects an answer outside the boss moves", function()
        expect(function()
            duel.compile_policy("return function(state) return 'x' end", { states = states })
        end).to.fail()
    end)

    it("rejects a twin slam the state does not allow", function()
        expect(function()
            duel.compile_policy("return function(state) return 't' end", {
                states = { boss_state() },
            })
        end).to.fail()
    end)

    it("rejects an answer that is not a move char", function()
        expect(function()
            duel.compile_policy("return function(state) return nil end", { states = states })
        end).to.fail()
    end)

    it("rejects a policy that answers the same state twice differently", function()
        local flip = [[
local n = 0
return function(state)
    n = n + 1
    if n % 2 == 0 then
        return "c"
    end
    return "f"
end
]]
        expect(function()
            duel.compile_policy(flip, { states = states })
        end).to.fail()
    end)

    it("rejects a chunk reaching for a global outside the sandbox", function()
        -- `os` / `io` / `load` / `setmetatable` are all absent, so the
        -- call raises instead of touching the host.
        expect(function()
            duel.compile_policy("return function(state) return os.time() end", {
                states = states,
            })
        end).to.fail()
        expect(function()
            duel.compile_policy("return function(state) return load('return 1')() end", {
                states = states,
            })
        end).to.fail()
    end)

    it("rejects a source that is not a string", function()
        expect(function()
            duel.compile_policy(nil, { states = states })
        end).to.fail()
    end)

    it("rejects an empty validation batch", function()
        expect(function()
            duel.compile_policy("return function(state) return 'c' end", { states = {} })
        end).to.fail()
    end)

    it("keeps a mutating chunk away from the live state", function()
        local mutating = [[
return function(state)
    state.mode = 1
    state.damage_since_shift = 0
    return "c"
end
]]
        local state = boss_state({ mode = 0, damage_since_shift = 20 })
        local policy = duel.compile_policy(mutating, { states = { state } })
        expect(state.mode).to.equal(0)
        expect(state.damage_since_shift).to.equal(20)
        expect(policy(state)).to.equal("c")
        expect(state.mode).to.equal(0)
    end)

    it("rejects a state that is not a boss state", function()
        local policy = duel.compile_policy("return function(state) return 'c' end", {
            states = states,
        })
        expect(function()
            policy({ mode = 0 })
        end).to.fail()
    end)

    it("labels a corpus like any other policy", function()
        local policy = duel.compile_policy("return function(state) return 'w' end", {
            games = 2,
            seed = 37,
        })
        local rows = duel.build_corpus(policy, {
            ctx_len = CTX_LEN,
            games = 2,
            style = "guardian",
            seed = 37,
        })
        expect(#rows >= 2).to.equal(true)
        for _, row in ipairs(rows) do
            expect(row_text(row):match("^C%dM%dH%dD%dL%dT%d>w\n$") ~= nil).to.equal(true)
        end
    end)
end)

-- ─── Vocabulary ─────────────────────────────────────────────────────

describe("guardian_duel.vocab", function()
    it("fits the tiny preset vocabulary", function()
        expect(duel.vocab().size <= 64).to.equal(true)
    end)

    it("round-trips every alphabet char", function()
        local v = duel.vocab()
        local text = "C0M1H9D8L4T7>cfvwdt"
        local ids = duel.to_ids(text)
        local back = {}
        for _, id in ipairs(ids) do
            back[#back + 1] = v.to_char[id]
        end
        expect(table.concat(back)).to.equal(text)
    end)

    it("hands back a copy of the maps", function()
        local v = duel.vocab()
        local id = v.to_id["C"]
        v.to_id["C"] = 99
        expect(duel.vocab().to_id["C"]).to.equal(id)
    end)

    it("refuses a player move letter", function()
        -- The player's moves reach the model as the `L` digit, so their
        -- letters are outside the alphabet on purpose.
        expect(function()
            duel.to_ids("a")
        end).to.fail()
        expect(function()
            duel.to_ids("A")
        end).to.fail()
    end)

    it("refuses text that is not a string", function()
        expect(function()
            duel.to_ids(12)
        end).to.fail()
    end)
end)

describe("guardian_duel.run", function()
    it("returns the encoded opening state", function()
        expect(duel.run({ seed = 1 }).result).to.equal("C0M0H9D3L0T1")
    end)

    it("defaults the seed and the teacher style", function()
        expect(duel.run({}).result).to.equal("C0M0H9D3L0T1")
    end)

    it("honours a style", function()
        expect(duel.run({ style = "turtle" }).result).to.equal("C0M0H9D2L0T1")
    end)

    it("rejects an unknown style", function()
        expect(function()
            duel.run({ style = "berserker" })
        end).to.fail()
    end)
end)
