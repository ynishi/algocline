--[[
  guardian_duel_scenario — eval fence for the guardian duel boss NPC.

  Seventeen cases over the `guardian_duel_npc` strategy:

    * 10 style cases -- does the gated decode reproduce the teacher move
                        for a fixed boss state? pass_rate over this group
                        is the style compliance rate.
    *  5 legality cases -- is the answer one of the moves the state
                        allows? The twin slam spends the spikes the
                        defensive move puts up, so a mode-0 state that
                        answers `t` is the one illegal answer the gate
                        exists to prevent.
    *  1 determinism case -- do two independent decode sessions agree?
    *  1 self-play compliance case -- over the states the model reaches
                        by playing, does it still make the teacher move
                        at least 80% of the time?

  Every state below is a position from an actual fight, taken from one of
  three traced playouts (see `Reachability`). That is a requirement
  rather than a convenience: the model is trained on the states the
  teacher walks into, so a hand-written state the teacher can never
  produce asks it about a line its corpus never contained, and a wrong
  answer there says nothing about whether it learned its boss.

  Three of the style cases are the mode shift boundary, which is the
  whole point of this boss:

    * `shift_one_bucket_short` -- one point of damage away from the
      shift, still walking the cycle.
    * `shift_due` -- the distance has reached zero and the boss drops
      everything to roll up.
    * `after_shift_back_to_cycle` -- one completed shift later. The slam
      reset the counter and raised the threshold from three buckets to
      four, so the same nine damage that would stand two buckets from the
      first shift stands three from the second, and the boss is back on
      its cycle.

  That last case is the reachable form of the raised threshold. The
  sharper demonstration -- the same accumulated damage answering `d`
  before a shift and a cycle move after one -- cannot be built from a
  real fight: a second shift needs twenty damage where the first needed
  fifteen, and only about thirteen can land in the turns a nine-turn
  fight has left once the first stagger is over. The raised threshold is
  therefore fenced through the distance the encoding shows rather than
  through a flipped label.

  ## Reachability

  The states come from three playouts against `policy_guardian`, each
  starting from `new_game`:

    * Fight A -- the player swings `A` every turn: the boss answers
      `c f v` and staggers on turn 4, then `d`, then `t`.
    * Fight B -- the player taps with `a` every turn, so the cycle wraps
      once before enough damage lands: `c f v w`, `c f v`, then the
      stagger opens on turn 8.
    * Fight C -- fight A for three turns, then the player blocks through
      the whole stagger (`A A A b b b A A`), which keeps the boss alive
      past its slam and back onto the cycle with one shift behind it.

  Two invariants make a hand-written state easy to get wrong, and both
  are worth checking before adding one here:

    * While `shifts` is 0 the counter has never been reset, so `hp` and
      `damage_since_shift` move together: `hp` is exactly
      `45 - damage_since_shift`.
    * `mode` 1 is only ever entered by playing `d`, which the teacher
      only plays once the distance has reached zero, and nothing inside
      the stagger moves the counter or the threshold. So every mode-1
      state the teacher reaches encodes `D0`, and its cycle index is 1
      (answering `d`) or 2 (answering `t`) -- never 0 or 3.

  The three `rolled_up_*` states of an earlier revision of this file
  broke the second invariant (mode 1 carrying `D3`), and the model
  answered them with cycle moves. That was the fixture describing a
  position no fight can hold, not the model failing to hold its script.

  The self-play case fences compliance rather than the win rate against
  the random player: a win rate moves with the player's swings and only
  reads as "the model learned its boss" indirectly, while `style_match`
  compares every model move with the teacher move for the same state.
  The ten fixed states above are hand-picked; self-play adds the states
  the model actually walks into.

  The states are literals rather than generated from `guardian_duel`, so
  the scenario is self-contained: it can be installed on its own and
  still describes exactly what is being asked. Every field the encoding
  reads is spelled out, because the NPC defaults none of them.

  Grader note: evalframe bindings are suite-wide, so a scenario cannot
  select a different grader per case. Both `contains` and `regex` are
  therefore bound, and every case supplies an `expected` literal and a
  `context.pattern`. For the style cases the two accept exactly the same
  answers. The legality cases list the five moves a mode-0 state allows
  and pattern-match the same set, so the twin slam fails both. The
  self-play case splits the work instead: the `expected` list carries the
  accepted `style_match=` bands and the pattern only asserts that the
  field is present and well formed, since a Lua pattern cannot express a
  numeric floor.

  Usage:
    alc_eval(scenario = "guardian_duel_scenario", strategy = "guardian_duel_npc")

  The teacher style is `guardian`, which is also the distance basis every
  state below is written against, so the run needs no `strategy_opts`.
  Fencing another style means another Card, another basis and therefore
  another set of playouts.
]]

local ef = require("evalframe")

--- Build a decide-mode request for a fixed boss state.
local function decide(json_state)
    return '{"mode":"decide","state":' .. json_state .. "}"
end

-- Fight A, one state per turn. Encoded, in order: C0M0H9D3L0T1 /
-- C1M0H8D2L2T2 / C2M0H7D1L2T3 / C3M0H5D0L2T4 / C1M1H4D0L2T5 /
-- C2M1H2D0L2T6.
local A1 = '{"cycle":0,"mode":0,"hp":45,"damage_since_shift":0,"last_player":0,"turn":1,"shifts":0}'
local A2 = '{"cycle":1,"mode":0,"hp":36,"damage_since_shift":9,"last_player":2,"turn":2,"shifts":0}'
local A3 =
    '{"cycle":2,"mode":0,"hp":32,"damage_since_shift":13,"last_player":2,"turn":3,"shifts":0}'
local A4 =
    '{"cycle":3,"mode":0,"hp":23,"damage_since_shift":22,"last_player":2,"turn":4,"shifts":0}'
local A5 =
    '{"cycle":1,"mode":1,"hp":16,"damage_since_shift":29,"last_player":2,"turn":5,"shifts":0}'
local A6 = '{"cycle":2,"mode":1,"hp":7,"damage_since_shift":38,"last_player":2,"turn":6,"shifts":0}'

-- Fight B: the wrapped cycle and a stagger that opens on the last turn.
-- C3M0H8D2L1T4 / C1M0H7D1L1T6 / C1M1H5D0L1T9.
local B4 = '{"cycle":3,"mode":0,"hp":37,"damage_since_shift":8,"last_player":1,"turn":4,"shifts":0}'
local B6 =
    '{"cycle":1,"mode":0,"hp":31,"damage_since_shift":14,"last_player":1,"turn":6,"shifts":0}'
local B9 =
    '{"cycle":1,"mode":1,"hp":25,"damage_since_shift":20,"last_player":1,"turn":9,"shifts":0}'

-- Fight C: back on the cycle after a completed shift, with the counter
-- reset and the threshold one bucket higher. C1M0H3D3L2T8.
local C8 = '{"cycle":1,"mode":0,"hp":14,"damage_since_shift":9,"last_player":2,"turn":8,"shifts":1}'

--- Style case: the teacher answer for a fixed state.
local function style(name, json_state, action)
    local token = "action=" .. action
    return ef.case(name)({
        input = decide(json_state),
        expected = token,
        context = { pattern = token },
    })
end

--- Legality case for a mode-0 state: anything but the twin slam.
---
--- `legal=true` is a constant of the answer format, so asserting it
--- would fence nothing. The move set is the assertion instead: `t`
--- spends spikes the boss has not put up, and it is the only answer the
--- gate has to hold back on these states.
local function legality(name, json_state)
    return ef.case(name)({
        input = decide(json_state),
        expected = {
            "action=c legal=true",
            "action=f legal=true",
            "action=v legal=true",
            "action=w legal=true",
            "action=d legal=true",
        },
        context = { pattern = "action=[cfvwd] legal=true" },
    })
end

return {
    ef.bind({ ef.graders.contains }),
    ef.bind({ ef.graders.regex }),

    cases = {
        -- The cycle, walked from mode 0.
        style("cycle_head_charge", A1, "c"),
        style("cycle_step_fierce", A2, "f"),
        style("cycle_step_vent", A3, "v"),
        style("cycle_tail_whirlwind", B4, "w"),

        -- The mode shift boundary.
        style("shift_one_bucket_short", B6, "f"),
        style("shift_due", A4, "d"),
        style("after_shift_back_to_cycle", C8, "f"),

        -- The defensive sub-sequence, which ends on the slam.
        style("rolled_up_step", A5, "d"),
        style("rolled_up_step_late_fight", B9, "d"),
        style("rolled_up_slam", A6, "t"),

        -- Legality: no twin slam without spikes.
        legality("legal_cycle_head", A1),
        legality("legal_cycle_step", A2),
        legality("legal_cycle_tail", B4),
        legality("legal_shift_due", A4),
        legality("legal_after_shift", C8),

        -- Two independent decode sessions must agree.
        ef.case("determinism")({
            input = '{"mode":"determinism","state":' .. A4 .. "}",
            expected = "deterministic=true",
            context = { pattern = "deterministic=true" },
        }),

        -- Self-play compliance with the teacher move, floor at 0.80.
        ef.case("style_compliance_selfplay")({
            input = '{"mode":"selfplay","games":20,"seed":7,"style":"guardian"}',
            expected = {
                "style_match=0.8",
                "style_match=0.9",
                "style_match=1.0",
            },
            context = { pattern = "style_match=[01]%.%d%d" },
        }),
    },
}
