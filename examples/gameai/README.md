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
  bake_card_duel_from_log.lua        interactive move_log -> Card + alias (play-log bake)
  card_duel_scenario.lua             eval scenario (style / legality / determinism / self-play compliance)
  card_duel_scenario_timid.lua       eval scenario for the timid style
  card_duel_scenario_bold.lua        eval scenario for the bold style
  card_duel_tournament_scenario.lua  eval scenario for the round robin

  guardian_duel/init.lua             boss rules: cycle, mode shift, three boss styles (pure Lua)
  guardian_duel/spec/                alc_pkg_test suite for the boss rules
  guardian_duel_npc/init.lua         boss NPC strategy: encode -> decode -> legal gate
  guardian_duel_npc/spec/            alc_pkg_test suite for the boss NPC
  guardian_duel_interactive/init.lua human vs boss session, kept in alc.state
  guardian_duel_interactive/spec/    alc_pkg_test suite for the session
  guardian_player_npc/init.lua       player NPC strategy: encode -> decode -> legal gate
  guardian_player_npc/spec/          alc_pkg_test suite for the player NPC
  train_guardian_npc.lua             teacher corpus -> Full FT -> Card + alias
  bake_guardian_persona.lua          NL prompt -> synthesised boss -> Card + alias
  bake_guardian_player_from_log.lua  player half of a move_log -> Card + alias
  guardian_duel_scenario.lua         eval scenario (style / mode shift / legality / self-play)
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

### Baking from a play log

The third bake input is a game someone actually played. The `end`
action returns the session's `move_log` — the position the human was
looking at before each move (round, hand, both scores, opponent
history) and the rank they chose — and `bake_card_duel_from_log.lua`
trains on exactly those rows:

```
alc_run(
  code_file = "<repo>/examples/gameai/bake_card_duel_from_log.lua",
  ctx = { moves = <move_log>, name = "mine", steps = 800, batch = 32 }
)
```

`ctx` takes `moves` and `name` (a `^[a-z0-9_]+$` slug, pinned as the
alias `card_duel_npc_<name>`) plus `steps`, `batch`, `lr` and `seed`.
The logs of several sessions can be concatenated into one `moves`
array; nothing in a row is session-scoped. No `alc.llm` call happens
anywhere on this path, so unlike the persona bake the run never pauses.

There is no teacher function here, so there is nothing to score the
model against beyond the log itself. `log_match` is the replay of the
logged positions — the states the model was trained on — and is
therefore a training fit rather than a generalisation rate.
`replicate_factor` is the other half of the same caveat: it states how
many times the log had to be repeated to answer `steps * batch`
samples, and replication buys steps, not coverage.

Three games played strictly lowest-card, baked at the defaults on
2026-07-31:

```
moves=15 log_match=1.00 train_loss=0.187 replicate_factor=1707
```

The fifteen logged positions come back exactly. Probing positions the
log never contained, four of which left a free choice, two of the four
agreed with the same rule — coverage is the number of games in the log,
so a log meant to carry a style wants tens of games rather than three.

## Guardian duel

The second game in this directory is a boss fight, and it is here to ask
a different question: can the same pipeline learn a boss *script* — a
behaviour the player is meant to read and plan around — rather than a
move preference?

Its shape is borrowed from Slay the Spire's
[The Guardian](https://slaythespire.wiki.gg/wiki/The_Guardian): the boss
walks a fixed four-move cycle, a defensive sub-sequence interrupts that
cycle once the boss has taken enough damage since its last interruption,
and the damage it tolerates grows every time the interruption completes
("Gains 30 Mode Shift, increased by 10 for each time Mode Shift
triggered this combat"). Only that observable structure is borrowed.
Every number below is this demo's own, sized for a nine-turn duel, and
pinned by `guardian_duel/spec`.

### The fight

Nine turns, one player against one boss, both starting at 45 health.
Each turn the player moves first and the boss answers:

| player | effect |
|---|---|
| `a` | light attack |
| `A` | heavy attack, and the boss's next answer hits harder |
| `b` | block, which absorbs the answer of the same turn |
| `p` | poke, which deals almost nothing and reveals the next answer |

| boss | effect |
|---|---|
| `c` | charge, blocking the player's next attack |
| `f` | fierce blow |
| `v` | vent, weakening the player's next attack |
| `w` | whirlwind |
| `d` | defensive, putting up spikes that retaliate on every attack |
| `t` | twin slam, which spends the spikes and ends the stagger |

The fight ends when either side reaches zero or after nine turns, in
which case the higher health bucket wins.

`guardian_duel.policy_guardian` is the teacher: it walks `c f v w` in
mode 0, drops everything and plays `d` the moment the damage taken since
its last shift reaches its threshold, walks `d d t` while rolled up, and
returns to the cycle with a threshold one bucket higher than before. Two
variants ride the same machinery — `rusher` soaks five buckets before it
bothers to roll up and cycles `f w f v`, `turtle` rolls up after two and
cycles `c v c w` — so one rule set trains and compares three bosses.

The twin slam is the only conditional move: it spends spikes, so it is
legal in mode 1 alone. That single condition is what the decode gate has
to hold back.

### The encoding

A boss state is one line over a 25-character alphabet:

```
C2M0H5D0L1T5>d
 │ │ │ │ │ │ └─ boss move (only present in training lines)
 │ │ │ │ │ └─── turn
 │ │ │ │ └───── the player's last move, 0 until the first one
 │ │ │ └─────── damage buckets left before the next mode shift
 │ │ └───────── boss health bucket
 │ └─────────── mode: 0 walking the cycle, 1 rolled up
 └───────────── index into the cycle
```

Every state encodes to exactly 12 characters whatever the turn, so
prompt plus action is 14 tokens and a training line 15, inside the
16-token context of the `gpt2 tiny` preset. No context change was needed
for this game.

`D` is the distance left to the next mode shift rather than the damage
already taken, and that is what makes twelve characters enough to answer
from. The threshold rises with every shift, so the same accumulated
damage means "roll up now" early in a fight and "keep swinging" after a
stagger, and a model reading the damage would have to recover the shift
count from the rest of the line before it could tell the two apart. The
distance never asks it to: `D0` and "the shift is due" are the same
statement for every style and every shift count, so the twelve
characters are a sufficient statistic for the teacher. The number of
completed shifts has no field of its own; it is folded into this one.

The price is that the projection has to know whose threshold it is
measuring, which is why `guardian_duel.encode` takes a style, why the
NPC is told the style its Card was trained as, and why a persona Card
records the basis it borrowed.

### Run it

```
alc_pkg_link(path = "<repo>/examples/gameai/guardian_duel")
alc_pkg_link(path = "<repo>/examples/gameai/guardian_duel_npc")
alc_pkg_link(path = "<repo>/examples/gameai/guardian_duel_interactive")
```

Train the teacher. As with the card duel the script builds its own
corpus, tunes the model, registers a Card and pins the alias:

```
alc_run(
  code_file = "<repo>/examples/gameai/train_guardian_npc.lua",
  ctx = { style = "guardian", steps = 800, batch = 32 }
)
```

`ctx.style = "all"` trains `guardian`, `rusher` and `turtle` in turn.
Each run pins `guardian_duel_npc_<style>`; the `guardian` run also pins
the bare `guardian_duel_npc` alias, which is the one the NPC falls back
to.

Ask the boss for one move:

```
alc_run(code = [[
  local npc = require("guardian_duel_npc")
  return npc.run({ task = alc.json_encode({
      mode = "decide",
      state = { cycle = 2, mode = 0, hp = 25, damage_since_shift = 15,
                last_player = 1, turn = 5, shifts = 0 },
  }) })
]])
```

```
action=d legal=true raw_legal=true gated=false
```

Nothing in a boss state is optional. A missing `shifts` would move the
distance the model reads and a missing `mode` would offer it the twin
slam, so the rules module names the field it did not get instead of
filling one in.

`bake_guardian_persona.lua` is the prompt path, and it works exactly
like the card duel bake — synthesise a Lua policy, compile it in the
restricted environment, make it answer sampled states legally and
deterministically, then distil it — with one extra field:

```
alc_run(
  code_file = "<repo>/examples/gameai/bake_guardian_persona.lua",
  ctx = {
    prompt = "When it has taken enough damage since its last shift it drops everything and turtles; otherwise it walks its four-move cycle.",
    name = "warden",
    basis_style = "guardian",
    steps = 800,
    batch = 32,
  }
)
```

A persona has no mode-shift threshold in the rules, so it borrows one.
`basis_style` names it, and the bake writes it onto the Card under
`persona.basis_style` so every later decode reads the same basis the
corpus was labelled against.

Evaluate:

```
alc_scenario_install(source = "<repo>/examples/gameai/guardian_duel_scenario.lua")
alc_eval(scenario = "guardian_duel_scenario", strategy = "guardian_duel_npc")
```

Seventeen cases: ten fixed states against the teacher move, five states
where the answer has to stay out of the twin slam, agreement between two
independent decode sessions, and self-play compliance with a floor at
0.80. Three of the ten are the mode shift boundary — one point of damage
short of the shift, exactly on it, and back on the cycle after a
completed shift with the threshold one bucket higher.

Every fixed state is a position from a traced playout rather than a
hand-written one, and the scenario names the three fights it took them
from. A state the teacher cannot reach — a boss rolled up before it has
taken the damage that rolls it up, say — is off-distribution for a model
trained on the teacher's own play, so a wrong answer there measures the
fixture rather than the model. An earlier revision of this file learned
that the expensive way.

### Fighting it

`guardian_duel_interactive` keeps a fight in `alc.state` between calls,
so a human takes the player seat and answers one move per `alc_run`:

```
alc_run(code = [[
  return require("guardian_duel_interactive").run({
      action = "new", style = "guardian", seed = 7,
  })
]])
```

```
turn 1 of 9: you 45 hp - boss 45 hp, 15 damage to its next shift, play one of a, A, b, p
```

```
alc_run(code = [[
  return require("guardian_duel_interactive").run({ action = "play", move = "A" })
]])
```

```
turn 1: you played A, the boss answered c | turn 2 of 9: you 45 hp - boss 36 hp, 6 damage to its next shift, play one of a, A, b, p
```

The remaining distance is on the board on purpose: a boss whose
staggering is legible is the point of the game, and the number shown is
the raw form of the `D` field the model reads. `action = "show"`
re-prints the board without moving and `action = "end"` drops the
session and hands back the transcript:

```
session ended: fight over: you 12 hp - boss 0 hp, you win
```

Pass `game_id` to keep several fights apart, `style` to choose which
trained Card answers, and `basis_style` to override the basis a persona
borrowed. The `move_log` returned by `end` has one entry per turn: the
boss state the answer was decoded from, the answer, the player move and
whether the player had already been shown it.

### The poke, and what `intent` means here

Slay the Spire shows the next enemy action every turn — the
[Intent](https://slaythespire.wiki.gg/wiki/Intent) display. Here that
readout is scaled down to something the player has to buy: `p` deals the
least damage of the four moves — half a light attack, less than a
quarter of a heavy one — in exchange for the answer of the next turn.

```
alc_run(code = [[
  return require("guardian_duel_interactive").run({ action = "play", move = "p" })
]])
```

```
turn 2: you played p, the boss answered f | turn 3 of 9: you 33 hp - boss 36 hp, 6 damage to its next shift, play one of a, A, b, p | it will answer v
```

The reveal is a look-ahead of the *next decode*, not a second prediction
that could disagree with it. The session decodes the answer for the
position that follows the poke, stores it, and the next `play` replays
that stored move instead of asking the model again — so the board cannot
promise `v` and then swing something else. This is exact rather than
lucky because the boss answers from the position at the head of the
turn, which the player's own move never touches. A stored reveal that no
longer belongs to the turn about to be played is a loud error instead of
a fresh decode: it means the session was rewritten between calls, and
the player paid a turn for a look at a position that no longer exists.

### Baking the player

The `move_log` a finished fight hands back has two halves per turn. The
boss half is the state its answer was decoded from; the player half is
the *player view* — what the board showed the human before they chose,
and the move they chose from it. `bake_guardian_player_from_log.lua`
reads the second half and distils that habit into a Card:

```
alc_pkg_link(path = "<repo>/examples/gameai/guardian_player_npc")

alc_run(
  code_file = "<repo>/examples/gameai/bake_guardian_player_from_log.lua",
  ctx = { moves = <move_log>, name = "ytk", steps = 800, batch = 32 }
)
```

The two seats are not symmetric — a boss answers from a script it
carries, a player answers from what the board shows — but the pipeline
is neutral about which one it is holding. A role brings its own view,
its own alphabet and its own Card; corpus building, Full FT, the alias
and the decode gate underneath are the same code.

The asymmetry lives in the encoding, and what it leaves out is the
point. A player view carries only what the board shows: the boss mode,
both health buckets, the distance to the next shift, the turn and the
three status bits. The boss cycle index is not among them, and putting
it there would end the poke — a player who reads the next answer for
free has no reason to buy it.

`guardian_player_npc` seats the baked Card and plays it:

```
alc_run(code = [[
  return require("guardian_player_npc").run({
      task = alc.json_encode({ mode = "autoplay", games = 1, boss_style = "guardian" }),
      card_alias = "guardian_player_npc_ytk",
  })
]])
```

`boss_style` names both the boss the fights are played against and the
distance basis every generated view is measured against — the `D` a
model was logged under has to be the `D` it is autoplayed under.
`boss_card_alias` seats a boss Card in place of the teacher policy, so a
baked player can be put in front of a baked boss. `games` defaults to 1
because both seats decode greedily from a fixed opening: ten games are
ten copies of one fight rather than ten samples.

A nine-move human game replayed with logging, baked at the defaults on
2026-07-31 (`log_match=1.00`, `train_loss=0.151`), and the baked player
replayed the original fight move for move — 9 of 9, the same 15-10 win,
`raw_legal` 9 of 9.

That is a fit, not a generalisation. Nine positions from one fight are
what the model saw and what it gave back; nothing there says how it
answers a position the log never contained. The scaling caveat is the
card duel one — see "Baking from a play log" — coverage is the number
of games in the log, so a log meant to carry a style wants tens of them.

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

Both fences are card duel ones: the guardian duel packages have no entry
in the Rust harness yet and are covered by their package specs alone.

The package specs are separate and run on demand. None of them loads a
model — the tournament, the sessions and the boss NPC stub the model
layer out — so they pass on the default feature set:

```
alc_pkg_test(pkg = "card_duel")
alc_pkg_test(pkg = "card_duel_tournament")
alc_pkg_test(pkg = "card_duel_interactive")
alc_pkg_test(pkg = "guardian_duel")
alc_pkg_test(pkg = "guardian_duel_npc")
alc_pkg_test(pkg = "guardian_duel_interactive")
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
- An empty `opp_played` in a `move_log` entry is an empty Lua table,
  which JSON renders as `{}` rather than `[]`.
  `card_duel.rows_from_moves` reads either shape, so a log handed
  straight back to the bake script is unaffected, but a strict JSON
  consumer that carries a log across a transport has to expect the
  object form on the first round of every game.
- A guardian duel persona borrows the mode-shift threshold of a
  canonical style, because the `D` field of the encoding is a distance
  to *some* threshold and twelve characters have no room for another
  one. A persona that staggers on a rule of its own is therefore a rules
  change rather than a prompt.
- A poke buys the boss's next answer, but the player view has no field
  for it. The transcript records only whether the answer had already
  been revealed, not which one it was, so a Card baked from a log cannot
  condition on a reveal the human paid for — every poked turn trains as
  though the board had said nothing. Learning to play off a reveal needs
  an `intent` field in the view and in the player encoding, which is
  future work.
- `guardian_duel` duplicates the corpus, sampling and sandbox halves of
  `card_duel` instead of sharing them. Two rule sets are not enough to
  tell which parts are common; a third is the point at which to extract
  them.
- Decoding is greedy and therefore deterministic, so an NPC repeats
  itself in identical positions. Noisy or temperature decoding is
  deferred: the stdlib sampler has no legal-action mask, so sampling
  would have to be filtered outside the sampler, and it would break the
  determinism case the scenarios fence.
