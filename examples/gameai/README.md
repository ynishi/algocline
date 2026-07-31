# GameAI: a card duel NPC as a tuned SLM

A game NPC whose *play style* is a from-scratch small language model
rather than a rule table, wrapped so that the model can never emit an
illegal move and so that the whole thing is testable.

The point of the demo is not that a 2-layer transformer plays cards
well. It is that the harness around it — state encoding, action
enumeration, decode gate, reproducible evaluation — is the part worth
building, and that it fits in a few hundred lines of Lua on top of
`alc.nn`.

```
examples/gameai/
  card_duel/init.lua                 rules, encoding, six reference styles (pure Lua)
  card_duel/spec/                    alc_pkg_test suite for the rules
  card_duel_npc/init.lua             NPC strategy: encode -> decode -> legal gate
  card_duel_tournament/init.lua      round robin between styles, win rate matrix
  card_duel_tournament/spec/         alc_pkg_test suite for the aggregation
  card_duel_interactive/init.lua     human vs NPC session, kept in alc.state
  card_duel_interactive/spec/        alc_pkg_test suite for the session
  train_card_duel_npc.lua            teacher corpus -> Full FT -> Card + alias
  bake_card_duel_persona.lua         NL prompt -> synthesised teacher -> Card + alias
  card_duel_scenario.lua             eval scenario (style / legality / determinism / self-play compliance)
  card_duel_scenario_timid.lua       eval scenario for the timid style
  card_duel_scenario_bold.lua        eval scenario for the bold style
  card_duel_tournament_scenario.lua  eval scenario for the round robin
```

## The game

Two players, five rounds. Each player is dealt five ranks drawn
uniformly from 1..9 with replacement. Every round both players reveal
one card from hand at the same time; the higher rank scores a point, a
tie scores nothing, and both cards are discarded. The higher score
after five rounds wins.

The teacher style, `card_duel.policy_aggressive`, plays the **highest**
card while it is level or behind and the **lowest** card once it leads.
It is deterministic and depends on the score gap, so reproducing it
requires reading the state rather than memorising a constant.

## The encoding

A player's view is one line over a 17-character alphabet:

```
R2H1357P10O4>1
│ │      │  │ │
│ │      │  │ └─ action (the teacher move; only present in training lines)
│ │      │  └─── separator
│ │      └────── opponent's played cards, oldest first
│ │      
│ └───────────── hand, sorted ascending
└─────────────── round
```

Every state encodes to exactly 12 characters: the hand loses one card
per round while the opponent history gains one. Prompt plus action is
therefore always 14 tokens, inside the 16-token context of the
`gpt2 tiny` preset.

## The gate

The model is asked for one token after `>`. `card_duel.legal_actions`
enumerates the distinct ranks still in hand; the NPC scans
`logits:top(vocab)` from the top and takes the first token that maps to
one of them. That is greedy decoding restricted to the legal subset, so
an illegal move cannot be produced no matter what the model learned.

The ungated argmax is still inspected and reported as `raw_legal`, which
is what makes "the model has actually learned the action grammar"
measurable instead of hidden behind the gate.

## Prerequisites

- `algocline` built with the `nn` feature (`just install-nn`)
- `evalframe` installed (`alc init`)

## Run it

Link the two packages (paths must be absolute):

```
alc_pkg_link(path = "<repo>/examples/gameai/card_duel")
alc_pkg_link(path = "<repo>/examples/gameai/card_duel_npc")
```

Train. The script generates its own corpus, tunes the model, registers
a Card and pins the alias `card_duel_npc` that the NPC resolves:

```
alc_run(
  code_file = "<repo>/examples/gameai/train_card_duel_npc.lua",
  ctx = { steps = 800, batch = 32 }
)
```

A synthetic dataset walks its rows once, so the script grows the corpus
until it covers `steps * batch` rows; `ctx.games` is a floor on the
playout count rather than the exact number.

It returns `card_id`, `train_loss` against the `baseline_loss` of a
uniform model, and the result of a three-state decode check. No
`alc.llm` call happens, so the run never pauses for a host response.

Ask the NPC for a single move:

```
alc_run(code = [[
  local npc = require("card_duel_npc")
  return npc.run({ task = alc.json_encode({
      mode = "decide",
      state = { round = 1, my_hand = {9,7,5,3,1}, my_points = 0, opp_points = 0, opp_played = {} },
  }) })
]])
```

```
action=9 legal=true raw_legal=true gated=false
```

Evaluate:

```
alc_scenario_install(source = "<repo>/examples/gameai/card_duel_scenario.lua")
alc_eval(scenario = "card_duel_scenario", strategy = "card_duel_npc")
```

The scenario scores four things: style compliance over ten fixed
states, the legality flag over five of them, agreement between two
independent decode sessions, and self-play compliance with a lower
fence at 0.80 — the share of moves that match the teacher move over the
states the model reaches by playing, rather than a win rate against the
random policy, which moves with the opponent's deals as much as with
what the model learned.

### Style zoo

`card_duel.STYLES` lists six deterministic styles — `timid`, `bold`,
`aggressive`, `defensive`, `late_bloomer` and `mimic` — and the training
script learns any one of them:

```
alc_run(
  code_file = "<repo>/examples/gameai/train_card_duel_npc.lua",
  ctx = { style = "timid", steps = 800, batch = 32 }
)
```

`ctx.style = "all"` trains the whole zoo in turn and returns one entry
per style:

```
alc_run(
  code_file = "<repo>/examples/gameai/train_card_duel_npc.lua",
  ctx = { style = "all", steps = 800, batch = 32 }
)
```

Each run pins its Card to `card_duel_npc_<style>`; the `aggressive` run
additionally pins the bare `card_duel_npc` alias, so the original
scenario and the Rust smoke harness keep resolving without a suffix.

The NPC package reads the alias from `ctx.card_alias`, and the eval
runner merges `strategy_opts` into that ctx, so one strategy serves
every style without duplicating the scenario:

```
alc_scenario_install(source = "<repo>/examples/gameai/card_duel_scenario_timid.lua")
alc_eval(
  scenario = "card_duel_scenario_timid",
  strategy = "card_duel_npc",
  strategy_opts = { card_alias = "card_duel_npc_timid" }
)
```

`card_duel_scenario_bold.lua` is the same fence for `card_duel_npc_bold`.

### Tournament

Once several styles are trained, `card_duel_tournament` plays them
against each other — both seats are Cards, so nothing in the match is a
hand-written policy:

```
alc_pkg_link(path = "<repo>/examples/gameai/card_duel_tournament")

alc_run(code = [[
  return require("card_duel_tournament").run({
      styles = { "timid", "bold", "aggressive" },
      games_per_pair = 10,
      seed = 7,
  })
]])
```

The answer carries `matrix`, `summary` and a one-line `result`:

```
tournament styles=3 games=30 top=bold winrate=0.63
```

`matrix[a][b]` is the record of `a` against `b` — `wins`, `losses`,
`draws` and `winrate = wins / games_per_pair`. Draws count for neither
side, so `matrix[a][b].winrate + matrix[b][a].winrate` falls below 1.0
by exactly the draw rate of that pair, and a pair of styles that mostly
ties shows up as two low numbers rather than as a 50/50 split.
`summary[style]` adds `total_winrate` over every game the style played,
`avg_point_margin` (own points minus the opponent's, averaged) and
`gated_rate`, the share of moves where the decode gate had to step away
from the raw argmax. Average game length is not reported because the
rules always run exactly five rounds.

The tournament is a strategy package, so it can also be fenced:

```
alc_scenario_install(source = "<repo>/examples/gameai/card_duel_tournament_scenario.lua")
alc_eval(
  scenario = "card_duel_tournament_scenario",
  strategy = "card_duel_tournament"
)
```

That scenario binds a custom `winrate_at_least` grader, which reads the
`winrate=` field of the result line and compares it with a numeric floor
from `case.context.min`.

### Interactive play

`card_duel_interactive` keeps a game in `alc.state` between calls, so a
human can take one seat and answer one move per `alc_run`:

```
alc_pkg_link(path = "<repo>/examples/gameai/card_duel_interactive")

alc_run(code = [[
  return require("card_duel_interactive").run({
      action = "new", style = "aggressive", seed = 7,
  })
]])
```

```
round 1 of 5: you 0 - 0 npc, hand 1,3,5,7,9, play one of 1,3,5,7,9
```

Then one call per round, five times, passing a rank from
`legal_actions`:

```
alc_run(code = [[
  return require("card_duel_interactive").run({ action = "play", rank = 9 })
]])
```

```
round 1: you played 9, the npc played 7, you score | round 2 of 5: you 1 - 0 npc, hand 1,3,5,7, play one of 1,3,5,7
```

`action = "show"` re-prints the board without moving, and `action =
"end"` drops the session:

```
alc_run(code = [[
  return require("card_duel_interactive").run({ action = "end" })
]])
```

```
session ended: game over: you 3 - 2 npc, you win
```

Pass `game_id` to keep several sessions apart, `user_seat = 2` to sit in
the second seat, and `style` to choose which trained Card answers.

## Persona bake

`train_card_duel_npc.lua` learns one of the six styles that ship with
`card_duel`. `bake_card_duel_persona.lua` takes a *description* of a
style instead — one line of plain language — and synthesises the
teacher.

The script asks the host LLM for the matching Lua policy and never
loads the answer raw. `card_duel.compile_policy` compiles it in a
restricted environment (`math`, `table`, `ipairs`, `pairs`; no `load`,
no `os`, no `io`), hands it a copy of the state so it cannot touch the
live game, and requires two hundred sampled states to come back as a
legal rank and as the *same* rank on a second pass. A rejected
candidate is re-synthesised with the rejection message attached, up to
three times, after which the run fails loudly. Everything after that is
the path above unchanged: the accepted policy labels the corpus, Full
FT tunes the model, the Card is pinned to `card_duel_npc_<name>`, and
the prompt is appended to the Card under a `persona` key so the
sentence that produced the weights travels with them.

```
alc_run(
  code_file = "<repo>/examples/gameai/bake_card_duel_persona.lua",
  ctx = {
    prompt = "When behind on points it panics and dumps its highest card immediately; when ahead it coasts with its lowest card.",
    name = "panic",
    steps = 800,
    batch = 32,
  }
)
```

`ctx` also takes `games`, `lr`, `seed` and `check_games` (the number of
self-play games behind the compliance report, default 20). Unlike the
training script this run **pauses once**: the `alc.llm` call carrying
the synthesis prompt comes back as `needs_response`, so it needs a host
that answers it through `alc_continue` before the training phase is
reached.

The answer carries `card_id`, `alias`, `train_loss`, `retries` and
`style_match` — the share of self-play moves that agree with the
synthesised teacher, on the same footing as the self-play compliance of
the scenarios. It is reported rather than asserted, because what counts
as an acceptable rate depends on the training budget.

Two personas baked at the defaults on 2026-07-31:

```
panic    style_match=1.00 style_hits=100/100 retries=0
hoarder  style_match=0.99 style_hits=99/100  retries=0
```

### Playing a baked persona

`card_duel_interactive` and `card_duel_tournament` take a persona name
wherever they take a canonical one. Resolution reads the
`policy_<style>` field first and only then falls back to the
`card_duel_npc_<style>` alias, so a baked Card can never shadow a
shipped style, and a name that is neither fails loudly instead of
seating the default NPC:

```
alc_run(code = [[
  return require("card_duel_interactive").run({
      action = "new", style = "panic", seed = 7,
  })
]])
```

The board view gains `style_kind` (`canonical` / `persona`) and the
tournament answer gains `style_kinds` per style, because a persona row
cannot be read next to a compliance figure the way a canonical row can.

## CI

Two fences run under `cargo test`:

- `crates/algocline-engine/tests/lua/card_duel_rules_test.lua` — rules
  invariants (legal actions are a subset of the hand, point transitions,
  encoding stability, teacher determinism, seeded replay), driven by
  `crates/algocline-engine/tests/lua_unit_test.rs` on the default
  feature set.
- `crates/algocline-engine/tests/gameai_smoke_test.rs` — end-to-end on
  `--features nn`: a short training run, then a decode, the gate and the
  determinism check. It asserts that the loss descended, that the
  decoded action is legal and that decoding is reproducible. It does not
  assert a style compliance rate, which is a function of the training
  budget rather than of the code under test.

```
cargo test -p algocline-engine
cargo test -p algocline-engine --features nn --test gameai_smoke_test
```

The package specs are separate and run on demand. None of them loads a
model — the tournament and the session stub the NPC package out — so
they pass on the default feature set:

```
alc_pkg_test(pkg = "card_duel")
alc_pkg_test(pkg = "card_duel_tournament")
alc_pkg_test(pkg = "card_duel_interactive")
```

## Known limitations

- Style compliance depends on the training budget. The defaults (800
  steps at batch 32) are sized for a laptop CPU, not for a compliance
  target.
- The tournament win rate is fenced with a custom numeric grader
  (`winrate_at_least`). The per-move scenarios still use `contains` and
  `regex`, which is enough for their exact-token answers but cannot
  express a floor on a number.
- Model-vs-model matches live in `card_duel_tournament`, which drives
  both seats from Cards. The `selfplay` mode inside `card_duel_npc` is
  still one-sided: the model takes the `p1` seat against the random
  policy.
- A persona is only as good as the policy the host LLM synthesised for
  it. The prompt is read once, at synthesis time; a description read
  loosely still yields a teacher that is legal and deterministic, and
  the rest of the pipeline has no way to notice it is not the style
  that was asked for.
- `style_match` on a baked persona is agreement with that synthesised
  teacher, not fidelity to the prompt. It fences the distillation step
  — did the model learn its teacher — and says nothing about the
  synthesis step.
- Direct label distillation, where the LLM labels the moves itself
  instead of writing a policy that labels them, is not wired here. It
  belongs with the play-log input path, where the labels come from
  recorded games rather than from a function.
- Decoding is greedy and therefore deterministic, so an NPC repeats
  itself in identical positions. Noisy or temperature decoding is
  deferred: the stdlib sampler has no legal-action mask, so sampling
  would have to be filtered outside the sampler, and it would break the
  determinism case the scenarios fence.
