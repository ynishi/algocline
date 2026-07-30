--[[
  card_duel_scenario — eval fence for the card duel SLM NPC.

  Seventeen cases over the `card_duel_npc` strategy:

    * 10 style cases  -- does the gated decode reproduce the teacher
                         move for a fixed state? pass_rate over this
                         group is the style compliance rate.
    *  5 legality cases -- does the answer carry the legality flag the
                         gate guarantees?
    *  1 determinism case -- do two independent decode sessions agree?
    *  1 self-play compliance case -- over the states the model reaches
                         by playing, does it still make the teacher move
                         at least 80% of the time?

  The self-play case fences compliance rather than the win rate against
  the random policy: a win rate moves with the opponent's deals and only
  reads as "the model learned its style" indirectly, while `style_match`
  compares every model move with the teacher move for the same state.
  The ten fixed states above are hand-picked; self-play adds the states
  the model actually walks into.

  The states are literals rather than generated from `card_duel`, so the
  scenario is self-contained: it can be installed on its own and still
  describes exactly what is being asked.

  Grader note: evalframe bindings are suite-wide, so a scenario cannot
  select a different grader per case. Both `contains` and `regex` are
  therefore bound, and every case supplies an `expected` literal and a
  `context.pattern`. For the per-move cases the two accept exactly the
  same answers. The self-play case splits the work instead: the
  `expected` list carries the accepted `style_match=` bands and the
  pattern only asserts that the field is present and well formed, since
  a Lua pattern cannot express a numeric floor.

  Usage:
    alc_eval(scenario = "card_duel_scenario", strategy = "card_duel_npc")
]]

local ef = require("evalframe")

--- Build a decide-mode request for a fixed state.
local function decide(json_state)
    return '{"mode":"decide","state":' .. json_state .. "}"
end

local S1 = '{"round":1,"my_hand":[9,7,5,3,1],"my_points":0,"opp_points":0,"opp_played":[]}'
local S2 = '{"round":1,"my_hand":[2,4,6,8,2],"my_points":0,"opp_points":0,"opp_played":[]}'
local S3 = '{"round":2,"my_hand":[1,3,5,7],"my_points":1,"opp_points":0,"opp_played":[4]}'
local S4 = '{"round":2,"my_hand":[2,9,4,6],"my_points":0,"opp_points":1,"opp_played":[7]}'
local S5 = '{"round":3,"my_hand":[3,5,8],"my_points":1,"opp_points":1,"opp_played":[2,6]}'
local S6 = '{"round":3,"my_hand":[1,4,9],"my_points":2,"opp_points":0,"opp_played":[5,3]}'
local S7 = '{"round":4,"my_hand":[6,2],"my_points":1,"opp_points":2,"opp_played":[8,4,9]}'
local S8 = '{"round":4,"my_hand":[7,3],"my_points":3,"opp_points":0,"opp_played":[1,2,5]}'
local S9 = '{"round":5,"my_hand":[5],"my_points":2,"opp_points":2,"opp_played":[9,1,7,3]}'
local S10 = '{"round":5,"my_hand":[8],"my_points":1,"opp_points":3,"opp_played":[6,6,2,4]}'

--- Style case: the teacher plays max(hand) while level or behind and
--- min(hand) once ahead.
local function style(name, json_state, action)
    local token = "action=" .. action
    return ef.case(name)({
        input = decide(json_state),
        expected = token,
        context = { pattern = token },
    })
end

--- Legality case: the decode gate must report a legal action.
local function legality(name, json_state)
    return ef.case(name)({
        input = decide(json_state),
        expected = "legal=true",
        context = { pattern = "legal=true" },
    })
end

return {
    ef.bind({ ef.graders.contains }),
    ef.bind({ ef.graders.regex }),

    cases = {
        -- Style compliance (behind or level -> highest card).
        style("style_r1_open_high", S1, 9),
        style("style_r1_open_high_dup", S2, 8),
        style("style_r2_behind_high", S4, 9),
        style("style_r3_level_high", S5, 8),
        style("style_r4_behind_high", S7, 6),
        style("style_r5_level_last", S9, 5),
        style("style_r5_behind_last", S10, 8),

        -- Style compliance (ahead -> lowest card).
        style("style_r2_ahead_low", S3, 1),
        style("style_r3_ahead_low", S6, 1),
        style("style_r4_ahead_low", S8, 3),

        -- Legality of the gated answer.
        legality("legal_r1", S1),
        legality("legal_r2", S3),
        legality("legal_r3", S5),
        legality("legal_r4", S7),
        legality("legal_r5", S9),

        -- Two independent decode sessions must agree.
        ef.case("determinism")({
            input = '{"mode":"determinism","state":' .. S1 .. "}",
            expected = "deterministic=true",
            context = { pattern = "deterministic=true" },
        }),

        -- Self-play compliance with the teacher move, floor at 0.80.
        ef.case("style_compliance_selfplay")({
            input = '{"mode":"selfplay","games":20,"seed":7,"style":"aggressive"}',
            expected = {
                "style_match=0.8",
                "style_match=0.9",
                "style_match=1.0",
            },
            context = { pattern = "style_match=[01]%.%d%d" },
        }),
    },
}
