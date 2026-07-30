--[[
  card_duel_scenario_timid — eval fence for the timid card duel NPC.

  Sixteen cases over the `card_duel_npc` strategy:

    * 10 style cases  -- does the gated decode reproduce the timid move
                         (the lowest distinct rank in hand) for a fixed
                         state? pass_rate over this group is the style
                         compliance rate.
    *  5 legality cases -- does the answer carry the legality flag the
                         gate guarantees?
    *  1 determinism case -- do two independent decode sessions agree?

  The Card trained on the timid style is pinned to its own alias, so the
  scenario has to be driven against that alias rather than the default
  one:

    alc_eval(
      scenario = "card_duel_scenario_timid",
      strategy = "card_duel_npc",
      strategy_opts = { card_alias = "card_duel_npc_timid" }
    )

  There is no win rate case here: self-play inside the NPC package pits
  the Card against the random policy, and a style that always plays its
  lowest card is expected to lose that match. Measuring it would fence
  the style's strength rather than the model's compliance with it.

  The states are the same literals as `card_duel_scenario.lua` and are
  written out rather than generated from `card_duel`, so the scenario is
  self-contained: it can be installed on its own and still describes
  exactly what is being asked.

  Grader note: evalframe bindings are suite-wide, so a scenario cannot
  select a different grader per case. Both `contains` and `regex` are
  therefore bound, and every case supplies an `expected` literal and a
  `context.pattern` that accept exactly the same answers, which keeps
  `pass_rate` readable as the compliance rate.
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

--- Style case: the timid style plays min(hand) whatever the score gap
--- and the round say.
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
        -- Style compliance: always the lowest distinct rank in hand.
        style("style_r1_open_low", S1, 1),
        style("style_r1_open_low_dup", S2, 2),
        style("style_r2_ahead_low", S3, 1),
        style("style_r2_behind_low", S4, 2),
        style("style_r3_level_low", S5, 3),
        style("style_r3_ahead_low", S6, 1),
        style("style_r4_behind_low", S7, 2),
        style("style_r4_ahead_low", S8, 3),
        style("style_r5_level_last", S9, 5),
        style("style_r5_behind_last", S10, 8),

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
    },
}
