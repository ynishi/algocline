--[[
  card_duel_tournament_scenario — eval fence for the style round robin.

  One case over the `card_duel_tournament` strategy: run a three-style
  tournament and require the leading style to reach a win rate floor.

    alc_eval(
      scenario = "card_duel_tournament_scenario",
      strategy = "card_duel_tournament"
    )

  Every style listed in the case input needs a trained Card pinned to
  `card_duel_npc_<style>`, which `train_card_duel_npc.lua` writes when it
  is run with `ctx.style = "all"`.

  Grader note: evalframe bindings are suite-wide, so this file cannot
  share a suite with the per-move scenarios -- they bind `contains` and
  `regex`, which cannot express a numeric floor. It binds one custom
  grader instead, `winrate_at_least`, which reads the `winrate=` field of
  the result line and compares it with `case.context.min`. That is the
  numeric fence the string graders could only approximate by listing the
  accepted prefixes.

  The floor is deliberately low. A round robin has no reference opponent:
  every seat is a trained Card, so the leading style is only as strong as
  the field it plays in, and draws are counted for neither side. What the
  case fences is that the tournament ran, produced a readable summary and
  found a style that wins a reasonable share of its games -- not that a
  particular style is the strongest one.
]]

local ef = require("evalframe")
local grader = require("evalframe.model.grader")

--- Numeric floor on the leading style's win rate.
---
--- Reads `winrate=<x.xx>` out of the one-line tournament summary and
--- compares it with `case.context.min`. A missing or unparsable field
--- fails rather than defaulting to zero, so a broken summary is not
--- reported as a merely weak result.
local winrate_at_least = grader("winrate_at_least")({
    check = function(resp, case)
        local wr = tonumber((resp.text or ""):match("winrate=([%d%.]+)"))
        local min = (case.context and case.context.min) or 0.4
        return wr ~= nil and wr >= min
    end,
})

return {
    ef.bind({ winrate_at_least }),

    cases = {
        ef.case("tournament_three_styles")({
            input = '{"styles":["timid","bold","aggressive"],"games_per_pair":10,"seed":7}',
            context = { min = 0.4 },
        }),
    },
}
