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
  card_duel/init.lua          rules, encoding, reference policies (pure Lua)
  card_duel/spec/             alc_pkg_test suite for the rules
  card_duel_npc/init.lua      NPC strategy: encode -> decode -> legal gate
  train_card_duel_npc.lua     teacher corpus -> Full FT -> Card + alias
  card_duel_scenario.lua      eval scenario (style / legality / determinism / win rate)
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
independent decode sessions, and a self-play win rate against the
random policy with a lower fence at 0.50.

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

The package spec is separate and runs on demand:

```
alc_pkg_test(pkg = "card_duel")
```

## Known limitations

- Style compliance depends on the training budget. The defaults (800
  steps at batch 32) are sized for a laptop CPU, not for a compliance
  target.
- The win rate is fenced with a regex threshold rather than a numeric
  grader; a custom grader would express it directly.
- Only the `p1` seat is driven by the model in self-play. A
  model-vs-model match would need a second alias.
