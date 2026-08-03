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
boss state the answer was decoded from, the answer, the player move,
whether the player had already been shown it and — inside the player
view of that turn — which move they were shown.

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
both health buckets, the distance to the next shift, the turn, the
three status bits and the answer a poke revealed. The boss cycle index
is not among them, and putting it there would end the poke — a player
who reads the next answer for free has no reason to buy it.

The reveal is the one boss-side value a player view is allowed to
carry, because it is the one the board really showed. It is the last
character of the line, `-` on a turn nobody bought a look at and the
boss move itself on a turn somebody did:

```
M0H9D3Y9T1S0->a     turn 1, nothing revealed, light attack
M1H4D0Y6T5S4t>b     turn 5, the slam was poked for, block
```

Without it every poked turn would train as though the board had said
nothing, and "I blocked because I had been shown the slam" would reach
the model as "I blocked". The field has no letter of its own because
the budget has no room for one: thirteen characters, the separator, the
move and the newline are exactly the sixteen tokens of the `gpt2 tiny`
context, so the player line now fills the window rather than padding
it. A `move_log` recorded before the field existed is rejected by
`bake_guardian_player_from_log.lua` by entry number instead of being
padded with the placeholder — nothing outside the session that played
the fight knows what the board showed, and a guess there teaches the
model that reveals say nothing. Old logs are re-collected, not
repaired.

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
`raw_legal` 9 of 9. That run predates the `intent` field, so its log is
one of the ones the bake script now refuses: the numbers stand for the
twelve-character layout they were measured on, and reproducing them
means playing the fight again rather than re-baking the old
transcript.

That is a fit, not a generalisation. Nine positions from one fight are
what the model saw and what it gave back; nothing there says how it
answers a position the log never contained. The scaling caveat is the
card duel one — see "Baking from a play log" — coverage is the number
of games in the log, so a log meant to carry a style wants tens of them.

### Drawing instead of scanning

`decide_noisy` answers the same position by drawing from the model
rather than by scanning its ranking, and it stays legal by construction:
a temperature sampler is wrapped in `alc.nn.constraint.allow_list` over
the four move ids, so every other logit is `-inf` before the draw
happens. An illegal move is not rejected and redrawn, it is not
representable.

```
alc_run(code = [[
  return require("guardian_player_npc").run({
      task = alc.json_encode({
          mode = "decide_noisy", view = <player view>, temperature = 0.8, seed = 4,
      }),
      card_alias = "guardian_player_npc_ytk",
  })
]])
```

```
action=A legal=true raw_legal=true noisy=true temperature=0.8 seed=4
```

`seed` is required rather than defaulted, and the chain is rebuilt for
every decision. Both follow from the bridge: composing a sampler with a
constraint *consumes* both handles, because a sampler owns its RNG and
two handles onto one RNG would interleave draws from generations that
each believe they are reproducible from their seed. So the caller
derives the seed — from a turn number, from a run seed plus an index —
and a replay that derives it the same way draws the same move.
`raw_legal` is still the argmax, and still says whether the model was
answering the question at all.

`autoplay` takes the same `temperature`, and that is what turns a batch
into a sample: with it present `games` fights are `games` fights rather
than one counted N times, and the summary carries `noisy=true
temperature=<t>` so a sweep is three calls that each say what they were.
Every decision draws under `seed + game * (TURN_LIMIT + 1) + turn`, a
stride one wider than the longest fight, so no two decisions of a run
share a draw and a single turn of a single game can be rebuilt without
replaying the fights in front of it.

```
alc_run(code = [[
  return require("guardian_player_npc").run({
      task = alc.json_encode({
          mode = "autoplay", games = 20, seed = 5,
          boss_style = "guardian", temperature = 0.8,
      }),
      card_alias = "guardian_player_npc_ytk",
  })
]])
```

```
winrate=0.42 raw_legal=1.00 moves=176 a=61 A=48 b=44 p=23 noisy=true temperature=0.8
```

(Shape of the answer, not a measured run.) `decide` and `determinism`
are untouched by any of this: they never reach the sampler, so the
determinism the scenarios fence is still a property of the greedy
path.

### Proving the bake generalises

The claim "tens of games are enough for the habits to carry" is
reproducible, not anecdotal. The repo ships a sample play log at the
scale the caveat asks for, and a script that bakes it and scores the
result on games the corpus never contained:

```
data/guardian_sample_playlog_train.json     36 games, 309 logged turns
data/guardian_sample_playlog_holdout.json   18 games, 153 logged turns
```

```
alc_run(
  code_file = "<repo>/examples/gameai/eval_guardian_player_generalization.lua",
  ctx = {
    train_file   = "<repo>/examples/gameai/data/guardian_sample_playlog_train.json",
    holdout_file = "<repo>/examples/gameai/data/guardian_sample_playlog_holdout.json",
  }
)
```

The script bakes the training log with the exact parameters of
`bake_guardian_player_from_log.lua`, decodes every held-out position
through the fresh Card, and returns `pass` against explicit fences —
style match on held-out rule positions ≥ 0.90, raw decode legality
≥ 0.95, loss below the uniform baseline. Rates, not golden values:
training shuffles, so reruns move the digits but not the verdict.

The sample was played by a fixed conditional style ("sentinel", six
rules documented in `gen_guardian_sample_playlog.lua`, which replays
the deterministic collection and can regenerate or verify the data
files at any time). That is what makes the measurement sharp: rules
R1-R5 are functions of the player view alone, so every held-out
position where one applies has a computable ground truth, and each
position is classified seen/unseen by its 13-character encoding
against the training set. Whose hands played the log is irrelevant to
what is being proved — the proof needs a log with consistent habits
in it, not a particular author.

Measured on 2026-08-01 (two runs): rule positions 58/58 on held-out
games both times, including every unseen one (19/19); `raw_legal`
1.00, `gated_rate` 0.00. Overall held-out match sat at 0.61-0.73
because the free slots — positions where the style genuinely leaves
the move open — land at their noise floor, as they should: the view
underdetermines them, so no amount of data closes that gap. The
`log_match` analogue on the training set itself came in around 0.76,
which is the memorisation of the small-log regime giving way to rule
learning; a model that can no longer replay its corpus verbatim but
answers every rule position it never saw is the shape of the claim
this section set out to prove.

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
alc_pkg_test(pkg = "guardian_player_npc")
```

## Observing a bake — anymetric + gameai_metrics

Two packages turn a training run into an observable object:

- **`anymetric/`** — a generic observation domain with no gameai
  vocabulary. A *view* binds a registered metric name to a fixed config
  under a caller-chosen `view_id`; `observe(views, {card, step})`
  evaluates every view per checkpoint behind a per-view `pcall` (a
  broken metric yields an `ErrorRecord` instead of killing the run and
  its terminal checkpoint) and returns uniform records; *judgment*
  (`threshold` / `never_break`) consumes records and returns a
  `Decision {action, reason}` which a thin adapter projects onto the
  trainer hook ABI. Measurement and judgment stay separate layers:
  a metric never gates, a judgment never measures.
- **`gameai_metrics/`** — the game-specific instances (`level` =
  win rate + Wilson CI against an opponent pool, `style_distance` =
  adherence to a same-basis teacher, `trickiness` = sampler entropy,
  normalised on the boss seat where the legal set is state-dependent).
  All three accept `seat = "boss"` to measure a boss-side bake through
  the shared `boss_seat.lua` encode path.

`train_guardian_npc.lua` wires them together: pass `ckpt_every > 0`
(plus `gate_games` / `teacher_alias` / `enable_gate` /
`target_win_rate_lo`) and every checkpoint is measured on all three
axes independently. Only the strength axis may stop the run
(`ci_lower >= target_win_rate_lo`); the personality axes are logged
for a human to judge — how much of the character survives the win-rate
climb is deliberately not a machine's call. `ckpt_every = 0` (the
default) restores the plain training run untouched.

### Harvesting staged bosses in a single run

`enable_stages = true` swaps the single-threshold gate for a *staged*
judgment that carves the strength axis into named bands (default
`weak [0.10, 0.30]` / `mid [0.55, 0.85]` / `strong [0.851, 0.98]` on
`level.ci_lower`). Every checkpoint whose `ci_lower` lands in a band
is immediately baked into a durable Card via
`alc.nn.card.save_from_ckpt`, pinned to
`<stage_alias_prefix>_<label>` (`guardian_duel_npc_weak` /
`_mid` / `_strong` by default), and recorded in a JSON manifest at
`collection_path`. First-writer-wins per label: the earliest fire to
enter a band harvests it, because the earlier fire keeps more of the
teacher-distance and sampler-entropy that the mid- to strong-band
policy loses on its way to the win-rate ceiling. A checkpoint above
the topmost band `hi` stops the run (a break); everything below the
lowest band `lo` keeps training. `enable_stages` and `enable_gate`
compose: the gate can still break early on the strength axis while
staged is still filling weaker labels. `stage_bands` in `ctx`
overrides the defaults if the measured-in-the-wild bands live
elsewhere.

`alc.nn.card.save_from_ckpt(path, name, meta)` is the additive
sibling of `alc.nn.card.save`: instead of taking an in-memory vars
table it copies a raw safetensors file (a `run_full_ft` checkpoint,
in this pipeline) into the Card store's canonical layout and mints
a Card. It lets the harvest hook promote a mid-run checkpoint to a
persistent artefact without paying the round-trip through vars.

### Auditing a harvested collection

`gameai_metrics.audit_matrix` reads back a manifest that
`harvest_collection` wrote, restores each Card from its recorded
`card_id`, and produces two tables side by side: a per-Card
baseline (`level.win_rate` + Wilson CI vs the random player policy,
`trickiness` on the boss seat, `style_distance` vs the teacher) at
higher game counts than the harvest hook could afford, and a
pair-wise `style_distance` matrix between the Cards themselves.
The pair-wise matrix answers the question the harvest hook cannot:
"how different is each stage from the others, not just from the
teacher?" — the human-judgment input for whether a strong Card is
a distinct policy or the mid Card carried further along the same
teacher-collapse trajectory. `examples/gameai/audit_boss_collection.lua`
is a thin `alc_run` driver: point it at a harvest manifest with
`ctx.collection_path` and a `ctx.output` path, and it writes the
audit as JSON plus a summary line per Card and per Card-pair.

The auto-mkdir helper that both the harvest manifest writer and
the audit writer use lives in `gameai_metrics._fs` — the manifest
writer used to require the parent directory to exist before the
first `on_ckpt` fire, which meant a missing `workspace/…` prefix
crashed the trainer mid-run; `_fs.ensure_parent_dir` runs `mkdir
-p` under a POSIX single-quote escape so any path the caller can
name is safe to hand to `:save()`.

### Fighting a collection against player Cards

`gameai_metrics.fight_matrix` is the step after the audit: instead
of describing each boss Card in isolation, it plays every boss Card
of a collection against every player Card of a pool and reports one
cell per pairing — boss-seat `win_rate` with a Wilson CI, plus the
mean game length and the mean final hp margin (`boss.hp -
player.hp`), the two numbers that tell a close brawl from a rout
when the rate alone cannot. Both seats decode their own vocabulary
on their own seat, so the fight stays the fixed-role game the rules
implement.

The matrix always runs under a decode temperature (default `1.0`).
`guardian_duel` carries no RNG and its openings do not vary, so two
greedy Cards replay one identical game N times; the decode draw is
the only source of variance a Card-vs-Card fight has, and `1.0` is
the same scale `trickiness` measures entropy on — an audit's
`trickiness_norm` and the spread of a fight refer to one
distribution. The player axis is fed by
`bake_guardian_player_from_log.lua`, which now also accepts
`ctx.moves_path` (a JSON file holding the logged turns, e.g. the
shipped `data/guardian_sample_playlog_train.json`) as the exclusive
alternative to an inline `ctx.moves` array.
`examples/gameai/fight_boss_collection.lua` is the thin `alc_run`
driver: a harvest manifest for the boss axis, a `players` alias
array, an `output` path, and it writes the matrix as JSON plus one
summary line per cell.

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
- The player line now fills the tiny preset context exactly (thirteen
  characters plus the separator, the move and the newline). There is no
  room left for another field, so the next thing worth showing the
  player — a second turn of look-ahead, the boss's shift count — is a
  context-window change rather than a layout one.
- A play log recorded before the `intent` field cannot be baked. The
  bake script rejects it by entry number and says so; the fix is to
  play the fight again with the current session, because what the board
  showed on a poked turn is not recoverable from the log.
- `guardian_duel` duplicates the corpus, sampling and sandbox halves of
  `card_duel` instead of sharing them. Two rule sets are not enough to
  tell which parts are common; a third is the point at which to extract
  them.
- Noisy decoding exists on the player NPC alone
  (`guardian_player_npc`'s `decide_noisy`, and `temperature` on its
  autoplay). The card duel NPC, the boss NPC and the tournament still
  decode greedily and therefore repeat themselves in identical
  positions. Wiring the same allow-list draw into them is a port of a
  dozen lines rather than a design question, but their eval scenarios
  fence determinism, so it is a mode next to the greedy one there too —
  not a replacement for it.
