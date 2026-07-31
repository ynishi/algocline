--- guardian_duel — nine-turn boss duel used as the GameAI SLM boss playground
---
--- Pure-Lua rules for a one-boss fight small enough that a from-scratch
--- tiny SLM can learn the boss script from it, yet structured like a
--- real boss rather than a lookup table: the boss walks a fixed
--- four-move cycle, a defensive sub-sequence interrupts that cycle once
--- the boss has taken enough damage since its last interruption, and
--- the damage it tolerates grows every time the interruption completes.
---
--- That shape is the one Slay the Spire's The Guardian shows to the
--- player (cycle, mode shift on accumulated damage, threshold that
--- rises with every shift), scaled down to a nine-turn duel. Only the
--- observable structure is borrowed; every number below is this demo's
--- own and is pinned by the package spec.
---
--- ## Usage
---
--- ```lua
--- local duel = require("guardian_duel")
--- local g = duel.new_game(1) -- prompt: duel.encode(g.boss, "guardian")
--- local rng = alc.math.rng_create(7)
--- while not duel.is_over(g) do
---     local boss = duel.policy_guardian(g.boss)
---     g = duel.apply(g, duel.policy_player_random(rng), boss)
--- end
--- print(duel.winner(g))
--- ```
---
--- ## Algorithm
---
--- 1. `new_game(seed)` opens the fight with both sides at full health,
---    the boss at cycle index 0 in mode 0.
--- 2. Each turn the player acts first (`a` light / `A` heavy / `b`
---    block / `p` poke) and the boss answers (`c` charge / `f` fierce /
---    `v` vent / `w` whirlwind / `d` defensive / `t` twin slam).
--- 3. The fight ends when either side drops to zero health or after
---    `TURN_LIMIT` turns, in which case the higher health bucket wins.
---
--- The teacher style (`policy_guardian`) answers from the boss state
--- alone: in mode 1 it walks `SHIFT_SEQUENCE`, in mode 0 it plays `d`
--- once the damage taken since the last shift has reached
--- `threshold_damage`, and otherwise the current entry of its cycle.
--- `M.STYLES` carries two more styles over the same machinery, a
--- pushier one that lets far more damage through before it rolls up and
--- a defensive one that rolls up almost immediately, so one set of rules
--- trains and compares several boss NPCs.
---
--- The `D` field of the encoding is the distance left to that
--- threshold rather than the damage already taken, so `D0` and "the
--- boss shifts now" are the same statement for every style and every
--- shift count (see `encode`).
---
--- ## Entry contract
---
--- - `new_game` / `apply` / `is_over` / `winner` — fight progression
--- - `legal_actions` / `player_legal_actions` / `encode` / `vocab` /
---   `to_ids` — NPC-facing view
--- - `policy_<style>` for every name in `STYLES` — deterministic styles
--- - `policy_player_random` / `policy_boss_random` — random opponents
---   used by self-play
--- - `build_corpus` — supervised training lines for any boss policy
--- - `sample_states` / `compile_policy` — sandbox and validation for a
---   synthesised (LLM-written) boss policy chunk
--- - `player_view` / `player_encode` / `player_vocab` /
---   `player_to_ids` / `rows_from_player_moves` — the same fight from
---   the player's chair, for baking a player NPC out of a play log
--- - `hp_bucket` / `distance_bucket` / `shift_threshold` /
---   `threshold_damage` / `style_cycle` — the derived quantities the
---   NPC, the interactive session and the eval scenario read
--- - `run` — Strategy entry; returns the encoded opening state
---
--- ## Caveats
---
--- The encoding is sized against the `gpt2 tiny` preset context window
--- (16 tokens). Every state encodes to exactly `ENCODED_LEN` (12)
--- characters, so `encode(state, style) .. ">" .. action .. "\n"` is
--- always 15 tokens whatever the turn; the module refuses to load if
--- that ever stops fitting.
---
--- `shifts` has no field of its own in the 12-character budget, so an
--- encoding that showed the damage already taken would leave the model
--- to work out the threshold that damage is measured against: the same
--- counter reads as "roll up now" before the first stagger and as "keep
--- swinging" after one has raised the bar. Folding the threshold into
--- `D` as the remaining distance removes that inference at the source.
--- The twelve characters become a sufficient statistic for the
--- decision — `D0` *is* the shift condition — so the answer is a
--- function of what the model can see, whatever the shift count behind
--- it.
---
--- That is a structural property rather than a repair of an observed
--- collision. Inside a nine-turn fight no reachable pair of states
--- encodes alike and carries opposite labels even under the damage
--- form, because health is not independent of damage: every point the
--- boss loses is a point on its counter, and a completed shift costs it
--- a whole threshold on top, so a post-shift state always sits lower in
--- `H` than a same-counter pre-shift one. The distance field is worth
--- having because the model never has to make the inference at all, not
--- because the inference would have gone wrong.
---
--- This is why `encode` takes the style whose threshold it is measuring
--- against. A synthesised persona is therefore trained under a declared
--- style's distance basis; a persona that staggers on a rule of its own
--- needs its own basis rather than a borrowed one.
---
--- The player-side projection is a second, independent encoding of the
--- same fight, with its own layout, its own alphabet and its own corpus
--- builder (see `player_encode`). It is not the boss encoding with the
--- seats swapped: the two chairs see different things, and the boss
--- script position in particular is hidden from the player by design.
---
--- `build_corpus`, `sample_states` and `compile_policy` mirror the
--- `card_duel` functions of the same name rather than sharing them: two
--- rule sets are not yet enough to tell which parts are common. A third
--- rules package is the point at which the corpus / sandbox half should
--- be extracted into a shared module.
---
--- `alc.math.rng_create` / `alc.math.rng_int` are the only host calls
--- this module makes. It never calls `alc.llm`, so it can run inside a
--- plain Lua VM that stubs `alc.math`.

-- `alc_shapes` is optional: the rules module has to stay loadable in
-- the bare mlua-lspec VM used by `crates/algocline-engine/tests/lua`,
-- which has no package registry. When the shapes package is present
-- (normal MCP session) the full typed spec is declared; otherwise the
-- entry is declared without shapes rather than failing the load.
local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "guardian_duel",
    version = "0.1.0",
    description = "Nine-turn boss duel rules, encoding and reference boss styles for SLM NPC demos",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, so the module still loads in a VM
-- without a package registry.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            seed = T.number:is_optional():describe("Fight seed for the opening state (default: 1)"),
            style = T.string:is_optional():describe(
                "Boss style the distance field is measured against (default: guardian)"
            ),
        }),
        result = T.string:describe("Encoded opening state of the boss"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

-- ─── Constants ──────────────────────────────────────────────────────

--- Turns a fight may last before the health buckets decide it.
local TURN_LIMIT = 9

--- Health both sides open with. The two maxima are equal so the
--- end-of-fight comparison can put the buckets side by side.
local BOSS_MAX_HP = 45
local PLAYER_MAX_HP = 45

--- Health per encoded bucket: `BOSS_MAX_HP / HP_BUCKET_SIZE` is the top
--- bucket, and only an exhausted side lands on bucket 0.
local HP_BUCKET_SIZE = 5

--- Damage per encoded bucket of the distance to the next mode shift.
local DAMAGE_BUCKET_SIZE = 5

--- Highest value any encoded bucket may take, so a bucket is one char.
local MAX_BUCKET = 9

--- Entries in a style cycle, which is also the range of the encoded
--- cycle index.
local CYCLE_LEN = 4

--- Characters `encode` emits, checked on every call.
local ENCODED_LEN = 12

--- Context window of the `gpt2 tiny` preset the corpus targets.
local CTX_BUDGET = 16

--- Tokens one training line costs: the state, `>`, the action, `\n`.
local ROW_LEN = ENCODED_LEN + 3

-- A line that does not fit the preset cannot be trained on at all, so
-- the budget is checked once at load time rather than per row.
if ROW_LEN > CTX_BUDGET then
    error(
        string.format(
            "guardian_duel: a training line needs %d tokens but the tiny preset context is %d",
            ROW_LEN,
            CTX_BUDGET
        )
    )
end

--- Player moves, in the order their enum index is encoded (1-based; 0
--- is reserved for "the fight has not seen a player move yet").
local PLAYER_ACTIONS = { "a", "A", "b", "p" }

--- Damage a player move deals to the boss before block.
local PLAYER_DAMAGE = { a = 4, A = 9, b = 0, p = 2 }

--- Player moves the boss spikes retaliate against.
local PLAYER_ATTACKS = { a = true, A = true, p = true }

--- Damage `b` absorbs from the boss answer of the same turn.
local PLAYER_BLOCK = 6

--- Extra damage the boss deals on the turn after the player used `A`.
local HEAVY_EXPOSURE = 3

--- Damage `v` shaves off the player's next attack.
local WEAK_PENALTY = 2

--- Boss moves, in the order `legal_actions` reports them.
local BOSS_ACTIONS = { "c", "f", "v", "w", "d", "t" }

--- Damage a boss move deals to the player before block.
local BOSS_DAMAGE = { c = 0, f = 9, v = 5, w = 6, d = 0, t = 10 }

--- Damage `c` absorbs from the player's next attack.
local CHARGE_BLOCK = 5

--- Damage the spikes of `d` reflect on every player attack.
local THORNS = 3

--- Defensive sub-sequence a mode shift walks, indexed by the cycle
--- field. The sequence ends on the slam, and a state parked past its
--- end still slams rather than falling back to the cycle.
local SHIFT_SEQUENCE = { "d", "d", "t" }

--- Boss styles: the cycle they walk in mode 0 and the damage bucket
--- they tolerate before the first shift.
local STYLE_SPECS = {
    guardian = { cycle = { "c", "f", "v", "w" }, base_threshold = 3 },
    rusher = { cycle = { "f", "w", "f", "v" }, base_threshold = 5 },
    turtle = { cycle = { "c", "v", "c", "w" }, base_threshold = 2 },
}

--- Widest distance to a mode shift the rules can produce: the most
--- patient style, at the highest shift count a fight has turns for.
--- Anything beyond it is a corrupt state rather than a far-off shift.
local MAX_BASE_THRESHOLD = 0
for _, spec in pairs(STYLE_SPECS) do
    if spec.base_threshold > MAX_BASE_THRESHOLD then
        MAX_BASE_THRESHOLD = spec.base_threshold
    end
end
local MAX_DISTANCE = (MAX_BASE_THRESHOLD + TURN_LIMIT) * DAMAGE_BUCKET_SIZE

--- Stride between the fight seed and the opponent RNG seed of a
--- playout, so two playouts of the same batch never share a stream.
local RNG_STRIDE = 7919

--- Enum index of every player move, and of the heavy attack whose
--- exposure the boss punishes on the following turn.
local PLAYER_INDEX = {}
for index, ch in ipairs(PLAYER_ACTIONS) do
    PLAYER_INDEX[ch] = index
end
local HEAVY_INDEX = PLAYER_INDEX["A"]

M.TURN_LIMIT = TURN_LIMIT
M.BOSS_MAX_HP = BOSS_MAX_HP
M.PLAYER_MAX_HP = PLAYER_MAX_HP
M.HP_BUCKET_SIZE = HP_BUCKET_SIZE
M.DAMAGE_BUCKET_SIZE = DAMAGE_BUCKET_SIZE
M.CYCLE_LEN = CYCLE_LEN
M.ENCODED_LEN = ENCODED_LEN
M.CTX_BUDGET = CTX_BUDGET

--- Training lines one playout contributes at most: the boss speaks once
--- per turn, and a fight that ends early simply contributes fewer.
M.MAX_ROWS_PER_GAME = TURN_LIMIT

--- Char alphabet, indexed by model token id.
---
--- Index 1 holds token id 0 (the padding token), so `id = index - 1`.
--- Twenty-five entries keep the whole alphabet inside the `gpt2 tiny`
--- vocabulary of 64. The player move letters are deliberately absent:
--- the state carries the player's last move as an enum digit, so a
--- player letter in a training line is a bug rather than a state.
local CHARS = {
    "\0",
    "\n",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "C",
    "M",
    "H",
    "D",
    "L",
    "T",
    "c",
    "f",
    "v",
    "w",
    "d",
    "t",
    ">",
}

local TO_ID = {}
local TO_CHAR = {}
for index, ch in ipairs(CHARS) do
    local id = index - 1
    TO_ID[ch] = id
    TO_CHAR[id] = ch
end

--- Char-to-token-id map shared by the trainer and the NPC.
---
--- Returned tables are fresh copies so a caller cannot corrupt the
--- module-level maps that every other entry point reads.
---@return table vocab `{ size, pad_id, to_id, to_char }`
function M.vocab()
    local to_id, to_char = {}, {}
    for ch, id in pairs(TO_ID) do
        to_id[ch] = id
    end
    for id, ch in pairs(TO_CHAR) do
        to_char[id] = ch
    end
    return {
        size = #CHARS,
        pad_id = TO_ID["\0"],
        to_id = to_id,
        to_char = to_char,
    }
end

--- Map a string over the module alphabet to token ids.
---
--- Errors on an unknown character instead of substituting a filler: a
--- silently replaced char would train the model on a state it can never
--- be asked about at decode time.
---@param text string
---@return integer[] ids
function M.to_ids(text)
    if type(text) ~= "string" then
        error("guardian_duel.to_ids: text must be a string, got " .. type(text))
    end
    local ids = {}
    for i = 1, #text do
        local ch = text:sub(i, i)
        local id = TO_ID[ch]
        if id == nil then
            error(
                string.format(
                    "guardian_duel.to_ids: char %q at %d is outside the vocabulary",
                    ch,
                    i
                )
            )
        end
        ids[#ids + 1] = id
    end
    return ids
end

-- ─── Internal helpers ───────────────────────────────────────────────

local function copy_list(list)
    local out = {}
    for i, v in ipairs(list) do
        out[i] = v
    end
    return out
end

local function require_rng()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("guardian_duel: alc.math is required (alc.math.rng_create / alc.math.rng_int)")
    end
    return alc.math
end

--- Read an integer field of a caller-supplied table, or fail naming it.
local function require_int(fn, field, value, min, max)
    if type(value) ~= "number" or value ~= math.floor(value) or value < min or value > max then
        error(
            string.format(
                "guardian_duel.%s: %s must be an integer in %d..%d, got %s",
                fn,
                field,
                min,
                max,
                tostring(value)
            )
        )
    end
    return value
end

--- Validate every field the encoder and the reference styles read.
---
--- The check runs before any of them touches a field, so a state built
--- by hand (a logged fight, a scenario case) fails on the field it got
--- wrong rather than on the character it would have produced.
local function require_boss_state(fn, state)
    if type(state) ~= "table" then
        error(string.format("guardian_duel.%s: state must be a table, got %s", fn, type(state)))
    end
    require_int(fn, "state.cycle", state.cycle, 0, CYCLE_LEN - 1)
    require_int(fn, "state.mode", state.mode, 0, 1)
    require_int(fn, "state.hp", state.hp, 0, BOSS_MAX_HP)
    require_int(fn, "state.damage_since_shift", state.damage_since_shift, 0, BOSS_MAX_HP)
    require_int(fn, "state.last_player", state.last_player, 0, #PLAYER_ACTIONS)
    require_int(fn, "state.turn", state.turn, 1, TURN_LIMIT)
    require_int(fn, "state.shifts", state.shifts, 0, TURN_LIMIT)
    return state
end

--- Copy of a boss state, so a caller-supplied policy cannot mutate the
--- live fight.
---
--- `block` and `thorns` are engine bookkeeping rather than encoded
--- fields: a hand-built state that carries neither starts them at zero,
--- which is what a fresh fight holds.
local function copy_boss(state)
    return {
        cycle = state.cycle,
        mode = state.mode,
        hp = state.hp,
        damage_since_shift = state.damage_since_shift,
        last_player = state.last_player,
        turn = state.turn,
        shifts = state.shifts,
        block = state.block or 0,
        thorns = state.thorns or 0,
    }
end

--- Style-free rendering of a state, for error messages.
---
--- `encode` needs a style to measure its distance field against, and a
--- diagnostic must not have to invent one, so a rejection names the raw
--- fields instead of a projection that would depend on the answer the
--- caller got wrong.
local function state_summary(state)
    if type(state) ~= "table" then
        return type(state)
    end
    return string.format(
        "cycle=%s mode=%s hp=%s damage=%s last=%s turn=%s shifts=%s",
        tostring(state.cycle),
        tostring(state.mode),
        tostring(state.hp),
        tostring(state.damage_since_shift),
        tostring(state.last_player),
        tostring(state.turn),
        tostring(state.shifts)
    )
end

--- Style spec for a name, or a loud error listing the known styles.
local function require_style(fn, style)
    local spec = STYLE_SPECS[style]
    if spec == nil then
        local names = copy_list(M.STYLES)
        table.sort(names)
        error(
            string.format(
                "guardian_duel.%s: unknown style %s, expected one of %s",
                fn,
                tostring(style),
                table.concat(names, ", ")
            )
        )
    end
    return spec
end

-- ─── Derived quantities ─────────────────────────────────────────────

--- Encoded health bucket of a raw health value.
---
--- Bucket 0 is reserved for an exhausted side: any surviving sliver of
--- health rounds up to bucket 1, so `H0` never means "still standing".
---@param hp integer
---@return integer bucket `0..MAX_BUCKET`
function M.hp_bucket(hp)
    require_int("hp_bucket", "hp", hp, 0, BOSS_MAX_HP)
    if hp == 0 then
        return 0
    end
    return math.ceil(hp / HP_BUCKET_SIZE)
end

--- Encoded bucket of the damage still owed before the next mode shift.
---
--- Bucket 0 is reserved for a distance of zero, which is exactly the
--- condition under which a style shifts: any remaining sliver rounds up
--- to bucket 1, so `D0` never means "nearly there". A distance wider
--- than the top bucket reads as the top bucket, which a fight of
--- `TURN_LIMIT` turns cannot reach.
---@param distance integer Damage still owed
---@return integer bucket `0..MAX_BUCKET`
function M.distance_bucket(distance)
    require_int("distance_bucket", "distance", distance, 0, MAX_DISTANCE)
    if distance == 0 then
        return 0
    end
    local bucket = math.ceil(distance / DAMAGE_BUCKET_SIZE)
    if bucket > MAX_BUCKET then
        return MAX_BUCKET
    end
    return bucket
end

--- Damage bucket a style tolerates before its next mode shift.
---
--- The threshold starts at the style's base and rises by one for every
--- completed shift, so each stagger costs the player more than the last
--- one did.
---@param style string One of `STYLES`
---@param shifts integer Completed mode shifts
---@return integer threshold
function M.shift_threshold(style, shifts)
    local spec = require_style("shift_threshold", style)
    require_int("shift_threshold", "shifts", shifts, 0, TURN_LIMIT)
    return spec.base_threshold + shifts
end

--- Raw damage a style tolerates before its next mode shift.
---
--- The bucket threshold read back in damage, which is the form both the
--- decision (`damage_since_shift >= threshold_damage`) and the encoding
--- (`distance = threshold_damage - damage_since_shift`) work in, so the
--- two can never disagree about where the boundary sits.
local function threshold_damage_of(spec, shifts)
    return (spec.base_threshold + shifts) * DAMAGE_BUCKET_SIZE
end

---@param style string One of `STYLES`
---@param shifts integer Completed mode shifts
---@return integer damage
function M.threshold_damage(style, shifts)
    local spec = require_style("threshold_damage", style)
    require_int("threshold_damage", "shifts", shifts, 0, TURN_LIMIT)
    return threshold_damage_of(spec, shifts)
end

--- Damage still owed before `style` shifts out of its cycle.
---@param style string One of `STYLES`
---@param state table Boss state
---@return integer distance `0` once the shift is due
function M.shift_distance(style, state)
    local spec = require_style("shift_distance", style)
    require_boss_state("shift_distance", state)
    return math.max(0, threshold_damage_of(spec, state.shifts) - state.damage_since_shift)
end

--- Mode 0 cycle of a style, as a fresh copy.
---@param style string One of `STYLES`
---@return string[] cycle
function M.style_cycle(style)
    return copy_list(require_style("style_cycle", style).cycle)
end

--- Defensive sub-sequence every style walks during a mode shift.
---@return string[] sequence
function M.shift_sequence()
    return copy_list(SHIFT_SEQUENCE)
end

-- ─── Fight progression ──────────────────────────────────────────────

--- Open a fight.
---
--- The opening is the same for every seed — both sides at full health,
--- the boss at the head of its cycle. The seed is carried so a corpus
--- row can be traced back to the playout that produced it and so the
--- signature matches the other rules packages; the variety in a corpus
--- comes from the player stream, not from the deal.
---@param seed integer
---@return table game `{ turn, seed, revealed, player, boss }`
function M.new_game(seed)
    if type(seed) ~= "number" then
        error("guardian_duel.new_game: seed must be a number, got " .. type(seed))
    end
    return {
        turn = 1,
        seed = seed,
        revealed = false,
        player = { hp = PLAYER_MAX_HP, weak = false },
        boss = {
            cycle = 0,
            mode = 0,
            hp = BOSS_MAX_HP,
            damage_since_shift = 0,
            last_player = 0,
            turn = 1,
            shifts = 0,
            block = 0,
            thorns = 0,
        },
    }
end

--- Boss moves that are legal from a state.
---
--- Only the twin slam is conditional: it spends the spikes the
--- defensive move put up, so it needs the boss to be in mode 1. The
--- other five are always available, which leaves a synthesised style
--- room to differ from the teacher without leaving the rules.
---@param state table Boss state
---@return string[] actions
function M.legal_actions(state)
    if type(state) ~= "table" then
        error("guardian_duel.legal_actions: state must be a table, got " .. type(state))
    end
    require_int("legal_actions", "state.mode", state.mode, 0, 1)
    local out = {}
    for _, ch in ipairs(BOSS_ACTIONS) do
        if ch ~= "t" or state.mode == 1 then
            out[#out + 1] = ch
        end
    end
    return out
end

--- Player moves, which are legal on every turn of every fight.
---@return string[] actions
function M.player_legal_actions()
    return copy_list(PLAYER_ACTIONS)
end

--- Cycle index the boss holds after playing `action`.
---
--- In mode 0 the cycle wraps; a shift enters the defensive sequence at
--- its second step (the `d` that opened it was the first), the slam
--- returns to the head of the cycle, and any other move inside mode 1
--- advances the sequence without leaving it.
local function next_cycle(mode, cycle, action)
    if action == "t" then
        return 0
    end
    if action == "d" and mode == 0 then
        return 1
    end
    if mode == 1 then
        return math.min(cycle + 1, CYCLE_LEN - 1)
    end
    return (cycle + 1) % CYCLE_LEN
end

--- Play one turn.
---
--- The player acts first, so `b` protects against the boss answer of
--- the same turn while `c` protects the boss against the player attack
--- of the next one. A boss that is exhausted by the player attack does
--- not answer at all.
---
--- Both moves are validated before any state is built, so an illegal
--- move is a loud error rather than a half-applied turn.
---@param g table Game
---@param player_action string One of `player_legal_actions`
---@param boss_action string One of `legal_actions(g.boss)`
---@return table game Next game state
function M.apply(g, player_action, boss_action)
    if M.is_over(g) then
        error("guardian_duel.apply: fight is already over")
    end
    local player_index = PLAYER_INDEX[player_action]
    if player_index == nil then
        error(
            string.format(
                "guardian_duel.apply: player action must be one of %s, got %s",
                table.concat(PLAYER_ACTIONS, ", "),
                tostring(player_action)
            )
        )
    end
    local boss, player = g.boss, g.player
    require_boss_state("apply", boss)
    require_int("apply", "boss.block", boss.block, 0, CHARGE_BLOCK)
    require_int("apply", "boss.thorns", boss.thorns, 0, THORNS)
    require_int("apply", "player.hp", player.hp, 0, PLAYER_MAX_HP)

    local legal = M.legal_actions(boss)
    local allowed = false
    for _, ch in ipairs(legal) do
        if ch == boss_action then
            allowed = true
            break
        end
    end
    if not allowed then
        error(
            string.format(
                "guardian_duel.apply: boss action %s is not one of the legal moves %s on state (%s)",
                tostring(boss_action),
                table.concat(legal, ", "),
                state_summary(boss)
            )
        )
    end

    -- Player phase.
    local outgoing = PLAYER_DAMAGE[player_action]
    if player.weak and outgoing > 0 then
        outgoing = math.max(0, outgoing - WEAK_PENALTY)
    end
    local dealt = math.max(0, outgoing - boss.block)
    local boss_hp = math.max(0, boss.hp - dealt)
    -- An overkill hit can push the counter past the health it measures;
    -- it is held at the top of its encodable range, which only ever
    -- happens on the final state of a fight the boss just lost.
    local damage_since_shift = math.min(BOSS_MAX_HP, boss.damage_since_shift + dealt)
    local player_hp = player.hp
    if boss.thorns > 0 and PLAYER_ATTACKS[player_action] then
        player_hp = math.max(0, player_hp - boss.thorns)
    end

    -- Boss phase. A boss that just went down does not answer, and a
    -- player the spikes just felled takes nothing further.
    local mode, cycle, shifts = boss.mode, boss.cycle, boss.shifts
    local block, thorns, weak = 0, boss.thorns, false
    if boss_hp > 0 and player_hp > 0 then
        local incoming = BOSS_DAMAGE[boss_action]
        if incoming > 0 and boss.last_player == HEAVY_INDEX then
            incoming = incoming + HEAVY_EXPOSURE
        end
        if player_action == "b" then
            incoming = math.max(0, incoming - PLAYER_BLOCK)
        end
        player_hp = math.max(0, player_hp - incoming)

        if boss_action == "c" then
            block = CHARGE_BLOCK
        elseif boss_action == "v" then
            weak = true
        elseif boss_action == "d" then
            thorns = THORNS
        elseif boss_action == "t" then
            thorns = 0
            shifts = shifts + 1
            damage_since_shift = 0
        end
        cycle = next_cycle(mode, cycle, boss_action)
        if boss_action == "d" then
            mode = 1
        elseif boss_action == "t" then
            mode = 0
        end
    end

    local turn = g.turn + 1
    return {
        turn = turn,
        seed = g.seed,
        -- The poke buys a look at the answer that comes next; showing it
        -- is the interactive session's job, marking it is this one's.
        revealed = player_action == "p",
        player = { hp = player_hp, weak = weak },
        boss = {
            cycle = cycle,
            mode = mode,
            hp = boss_hp,
            damage_since_shift = damage_since_shift,
            last_player = player_index,
            turn = turn,
            shifts = shifts,
            block = block,
            thorns = thorns,
        },
    }
end

---@param g table Game
---@return boolean over
function M.is_over(g)
    if type(g) ~= "table" or type(g.turn) ~= "number" then
        error("guardian_duel.is_over: game.turn must be a number")
    end
    if type(g.boss) ~= "table" or type(g.player) ~= "table" then
        error("guardian_duel.is_over: game must carry a boss and a player")
    end
    if type(g.boss.hp) ~= "number" or type(g.player.hp) ~= "number" then
        error("guardian_duel.is_over: both sides must carry a numeric hp")
    end
    return g.turn > TURN_LIMIT or g.boss.hp <= 0 or g.player.hp <= 0
end

--- Winner of a finished fight.
---
--- Returns `nil` while the fight is still running: the caller asked a
--- question that has no answer yet, and inventing `"draw"` would hide a
--- loop that stopped one turn early.
---@param g table Game
---@return string|nil winner `"player"` / `"boss"` / `"draw"`, or nil when unfinished
function M.winner(g)
    if not M.is_over(g) then
        return nil
    end
    local boss_down = g.boss.hp <= 0
    local player_down = g.player.hp <= 0
    if boss_down and player_down then
        return "draw"
    elseif boss_down then
        return "player"
    elseif player_down then
        return "boss"
    end
    local boss_bucket = M.hp_bucket(g.boss.hp)
    local player_bucket = M.hp_bucket(g.player.hp)
    if player_bucket > boss_bucket then
        return "player"
    elseif boss_bucket > player_bucket then
        return "boss"
    end
    return "draw"
end

-- ─── Encoding ───────────────────────────────────────────────────────

--- Encode a boss state as one line over the module alphabet.
---
--- Layout: `C<cycle>M<mode>H<health>D<distance to the next mode
--- shift>L<last player move>T<turn>`, every field one character. `L0`
--- means the fight has not seen a player move yet; the four moves
--- themselves are `1..4` in the order of `player_legal_actions`.
---
--- `D` is the distance left to the shift rather than the damage already
--- taken, which is what makes these twelve characters a sufficient
--- statistic for the answer: the threshold rises with every shift, so
--- the same accumulated damage means "roll up now" early in a fight and
--- "keep swinging" later, while the same distance always means the same
--- thing. `D0` and the mode-0 shift condition are one and the same
--- (see `decide`), so no two states that encode alike can carry
--- different labels.
---
--- The distance is measured against `style`'s threshold, which is why
--- the projection takes one: a corpus belongs to the style it was
--- labelled with.
---@param state table Boss state
---@param style string One of `STYLES`
---@return string encoded Exactly `ENCODED_LEN` characters
function M.encode(state, style)
    require_boss_state("encode", state)
    local spec = require_style("encode", style)
    local distance = math.max(0, threshold_damage_of(spec, state.shifts) - state.damage_since_shift)
    local text = table.concat({
        "C",
        tostring(state.cycle),
        "M",
        tostring(state.mode),
        "H",
        tostring(M.hp_bucket(state.hp)),
        "D",
        tostring(M.distance_bucket(distance)),
        "L",
        tostring(state.last_player),
        "T",
        tostring(state.turn),
    })
    if #text ~= ENCODED_LEN then
        error(
            string.format(
                "guardian_duel.encode: state encoded to %d chars but the layout is %d (%s)",
                #text,
                ENCODED_LEN,
                text
            )
        )
    end
    return text
end

-- ─── Policies ───────────────────────────────────────────────────────

--- Answer of a style from a boss state.
---
--- Mode 1 walks the defensive sequence and ignores everything else: the
--- boss has already committed to the stagger. Mode 0 checks the
--- accumulated damage first, so the shift interrupts the cycle at
--- whatever index the cycle happens to hold, and only then plays the
--- cycle entry.
---
--- The mode-0 condition is stated on the raw state, but it is the same
--- condition the encoding shows: `damage_since_shift >=
--- threshold_damage` holds exactly when the distance is zero, which is
--- exactly when `encode` writes `D0`. The encoded state is therefore a
--- sufficient statistic for this function — a model reading `D0`
--- answers `d` without having to guess how many shifts came before.
local function decide(fn, spec, state)
    require_boss_state(fn, state)
    if state.mode == 1 then
        local step = state.cycle + 1
        if step > #SHIFT_SEQUENCE then
            step = #SHIFT_SEQUENCE
        end
        return SHIFT_SEQUENCE[step]
    end
    if state.damage_since_shift >= threshold_damage_of(spec, state.shifts) then
        return "d"
    end
    return spec.cycle[state.cycle + 1]
end

--- Teacher style: charge, fierce, vent, whirlwind, and roll up once the
--- fight has cost it three damage buckets since its last stagger.
---@param state table Boss state
---@return string action
function M.policy_guardian(state)
    return decide("policy_guardian", STYLE_SPECS.guardian, state)
end

--- Pushy variant: an all-offence cycle that soaks five damage buckets
--- before it bothers to roll up, so it staggers late and rarely.
---@param state table Boss state
---@return string action
function M.policy_rusher(state)
    return decide("policy_rusher", STYLE_SPECS.rusher, state)
end

--- Defensive variant: a charge-heavy cycle that rolls up after two
--- damage buckets, so it staggers early and often.
---@param state table Boss state
---@return string action
function M.policy_turtle(state)
    return decide("policy_turtle", STYLE_SPECS.turtle, state)
end

--- Canonical style names, in the order the trainer and the eval
--- scenario iterate them.
---
--- Every entry `s` has a matching `M["policy_" .. s]`; callers that take
--- a style name validate against this list instead of hard-coding their
--- own copy.
M.STYLES = { "guardian", "rusher", "turtle" }

--- Uniform choice over the player moves, driven by a caller-owned RNG.
---
--- The move needs no state to filter on — all four are legal on every
--- turn — so the RNG is the only argument, and it is a parameter rather
--- than module state so a self-play loop stays reproducible from its
--- own seed.
---@param rng userdata `alc.math.rng_create` handle
---@return string action
function M.policy_player_random(rng)
    local math_ns = require_rng()
    return PLAYER_ACTIONS[math_ns.rng_int(rng, 1, #PLAYER_ACTIONS)]
end

--- Uniform choice over the legal boss moves.
---
--- Used to sample states rather than to fight: a random boss reaches
--- mode / cycle combinations no reference style would walk into, which
--- is exactly the coverage a validation batch wants.
---@param state table Boss state
---@param rng userdata `alc.math.rng_create` handle
---@return string action
function M.policy_boss_random(state, rng)
    local legal = M.legal_actions(state)
    local math_ns = require_rng()
    return legal[math_ns.rng_int(rng, 1, #legal)]
end

-- ─── Corpus ─────────────────────────────────────────────────────────

--- Encode one training line and pad it to the model context window.
---
--- The line is `<encoded state>><action>\n`. A line that does not fit
--- is a loud error rather than a truncation: a truncated line teaches
--- the model a state it can never be asked about at decode time.
local function make_row(state, style, action, ctx_len, pad_id)
    local ids = M.to_ids(M.encode(state, style) .. ">" .. tostring(action) .. "\n")
    if #ids > ctx_len then
        error(
            string.format(
                "guardian_duel.build_corpus: encoded line needs %d tokens but the context is %d",
                #ids,
                ctx_len
            )
        )
    end
    for _ = #ids + 1, ctx_len do
        ids[#ids + 1] = pad_id
    end
    return ids
end

--- Build the supervised corpus that teaches `policy` to a model.
---
--- The boss speaks once per turn and the player answers at random, so
--- one playout contributes one row per turn — at most
--- `MAX_ROWS_PER_GAME`, fewer when a side goes down before the turn
--- limit. The labelling policy also drives the boss, so the states in
--- the corpus are the ones that policy actually reaches.
---
--- `alc.nn.data.synthetic` walks its rows once, so a trainer asking for
--- `steps * batch` rows needs roughly `games >= steps * batch /
--- MAX_ROWS_PER_GAME`; computing that floor is left to the caller,
--- which is the only side that knows the training budget.
---
--- `opts.style` is the threshold the distance field is measured
--- against, and it has no default: for a reference style it is the
--- style the policy belongs to, and for a synthesised persona it is the
--- basis the persona was written for. Guessing it would hand the model
--- a distance the labels do not follow.
---@param policy fun(state: table): string Labelling boss policy
---@param opts table `{ ctx_len, games, style, seed?, pad_id? }`
---@return integer[][] rows Token id rows, each `ctx_len` long
function M.build_corpus(policy, opts)
    if type(policy) ~= "function" then
        error("guardian_duel.build_corpus: policy must be a function, got " .. type(policy))
    end
    if type(opts) ~= "table" then
        error("guardian_duel.build_corpus: opts must be a table, got " .. type(opts))
    end
    local ctx_len = tonumber(opts.ctx_len)
    if ctx_len == nil or ctx_len < 1 then
        error("guardian_duel.build_corpus: opts.ctx_len must be a positive number")
    end
    ctx_len = math.floor(ctx_len)
    local games = tonumber(opts.games)
    if games == nil or games < 1 then
        error("guardian_duel.build_corpus: opts.games must be a positive number")
    end
    games = math.floor(games)
    require_style("build_corpus", opts.style)
    local seed = math.floor(tonumber(opts.seed) or 1)
    local pad_id = opts.pad_id
    if pad_id == nil then
        pad_id = TO_ID["\0"]
    end
    if type(pad_id) ~= "number" then
        error("guardian_duel.build_corpus: opts.pad_id must be a number, got " .. type(pad_id))
    end

    local math_ns = require_rng()
    local rows = {}
    for i = 1, games do
        local g = M.new_game(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not M.is_over(g) do
            local boss_action = policy(g.boss)
            rows[#rows + 1] = make_row(g.boss, opts.style, boss_action, ctx_len, pad_id)
            g = M.apply(g, M.policy_player_random(rng), boss_action)
        end
    end
    return rows
end

-- ─── Player-side view ───────────────────────────────────────────────
--
-- Everything above answers the boss seat. What follows answers the
-- other chair: the same fight as the player sees it, encoded so a tiny
-- SLM can be baked from a log of the moves a human actually played
-- (`examples/gameai/bake_guardian_player_from_log.lua`).
--
-- The two sides share no layout, no alphabet and no corpus builder. A
-- boss state and a player view are different questions, and the two
-- alphabets are different id spaces: a Card baked on one and decoded
-- through the other would answer legally and mean nothing. The boss
-- alphabet already refuses the player letters (see `CHARS`); the player
-- alphabet returns the favour by carrying none of the boss moves.

--- Fields of the player encoding, in layout order. Each is one letter
--- and one digit:
---
--- - `M` boss mode — 0 while it walks its cycle, 1 while it is rolled
---   up and the slam is coming
--- - `H` boss health bucket
--- - `D` bucket of the damage the boss still tolerates before its next
---   mode shift, the same distance the boss encoding carries
--- - `Y` your own health bucket
--- - `T` turn
--- - `S` status flags, folded into one digit (see `PLAYER_FLAGS`)
local PLAYER_FIELDS = { "M", "H", "D", "Y", "T", "S" }

--- Characters `player_encode` emits, checked on every call.
local PLAYER_ENCODED_LEN = 2 * #PLAYER_FIELDS

--- Tokens one player training line costs: the view, `>`, the move and
--- the newline. The budget is checked once at load, like the boss row:
--- a line that does not fit the preset cannot be trained on at all.
local PLAYER_ROW_LEN = PLAYER_ENCODED_LEN + 3

if PLAYER_ROW_LEN > CTX_BUDGET then
    error(
        string.format(
            "guardian_duel: a player training line needs %d tokens but the tiny preset context is %d",
            PLAYER_ROW_LEN,
            CTX_BUDGET
        )
    )
end

-- Both health buckets are read with `hp_bucket`, which is written
-- against the boss maximum, so the two maxima have to stay equal. They
-- are (the end-of-fight comparison needs it), and saying so at load
-- keeps a later tweak from producing a player bucket the encoder
-- rejects halfway through a fight.
if PLAYER_MAX_HP ~= BOSS_MAX_HP then
    error("guardian_duel: the two health maxima must be equal for the shared bucket function")
end

--- Weight of every status flag inside the `S` digit.
---
--- The three are the modifiers that move the damage arithmetic without
--- showing up in any other field: `weakened` shaves the player's next
--- attack, `exposed` adds to the boss answer that follows a heavy
--- attack, and `spikes` bites back on every attack while the boss is
--- rolled up. Folding them into one digit is what buys the sixth field
--- its place inside twelve characters.
local PLAYER_FLAGS = { weakened = 1, exposed = 2, spikes = 4 }

--- Char alphabet of the player side, indexed by model token id.
---
--- Index 1 holds token id 0 (the padding token), so `id = index - 1`.
--- Twenty-three entries keep it inside the `gpt2 tiny` vocabulary of
--- 64. The boss move letters are deliberately absent: a boss letter in
--- a player training line is a bug rather than a move.
local PLAYER_CHARS = {
    "\0",
    "\n",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "D",
    "H",
    "M",
    "S",
    "T",
    "Y",
    ">",
    "a",
    "A",
    "b",
    "p",
}

--- Vocabulary size of the `gpt2 tiny` preset both corpora target.
local VOCAB_BUDGET = 64

if #CHARS > VOCAB_BUDGET or #PLAYER_CHARS > VOCAB_BUDGET then
    error(
        string.format(
            "guardian_duel: alphabets of %d and %d chars do not both fit the preset vocabulary of %d",
            #CHARS,
            #PLAYER_CHARS,
            VOCAB_BUDGET
        )
    )
end

local PLAYER_TO_ID = {}
local PLAYER_TO_CHAR = {}
for index, ch in ipairs(PLAYER_CHARS) do
    local id = index - 1
    PLAYER_TO_ID[ch] = id
    PLAYER_TO_CHAR[id] = ch
end

M.PLAYER_ENCODED_LEN = PLAYER_ENCODED_LEN

--- Char-to-token-id map of the player alphabet.
---
--- A separate table from `vocab` rather than a superset of it: the two
--- corpora are trained into separate Cards, and an id that meant `f` in
--- one and `a` in the other would only be found by reading gibberish
--- out of a model that looked healthy.
---@return table vocab `{ size, pad_id, to_id, to_char }`
function M.player_vocab()
    local to_id, to_char = {}, {}
    for ch, id in pairs(PLAYER_TO_ID) do
        to_id[ch] = id
    end
    for id, ch in pairs(PLAYER_TO_CHAR) do
        to_char[id] = ch
    end
    return {
        size = #PLAYER_CHARS,
        pad_id = PLAYER_TO_ID["\0"],
        to_id = to_id,
        to_char = to_char,
    }
end

--- Map a string over the player alphabet to token ids.
---
--- Errors on an unknown character exactly as `to_ids` does, which is
--- also what stops a boss line from being trained into a player Card:
--- every boss move letter is outside this alphabet.
---@param text string
---@return integer[] ids
function M.player_to_ids(text)
    if type(text) ~= "string" then
        error("guardian_duel.player_to_ids: text must be a string, got " .. type(text))
    end
    local ids = {}
    for i = 1, #text do
        local ch = text:sub(i, i)
        local id = PLAYER_TO_ID[ch]
        if id == nil then
            error(
                string.format(
                    "guardian_duel.player_to_ids: char %q at %d is outside the player vocabulary",
                    ch,
                    i
                )
            )
        end
        ids[#ids + 1] = id
    end
    return ids
end

--- Read a boolean field of a player view, or fail naming it.
---
--- Only a boolean is accepted: `weakened = "no"` would read as true and
--- flip a flag the log says was down.
local function require_flag(fn, field, value)
    if type(value) ~= "boolean" then
        error(
            string.format(
                "guardian_duel.%s: %s must be a boolean, got %s",
                fn,
                field,
                tostring(value)
            )
        )
    end
    return value
end

--- Validate every field the player encoding reads.
---
--- Like `require_boss_state`, the check runs before the encoder touches
--- a field, so a view built by hand — a logged fight, a scenario case —
--- fails on the field it got wrong rather than on the character it
--- would have produced.
local function require_player_view(fn, view)
    if type(view) ~= "table" then
        error(string.format("guardian_duel.%s: view must be a table, got %s", fn, type(view)))
    end
    require_int(fn, "view.mode", view.mode, 0, 1)
    require_int(fn, "view.boss_hp", view.boss_hp, 0, BOSS_MAX_HP)
    require_int(fn, "view.shift_distance", view.shift_distance, 0, MAX_DISTANCE)
    require_int(fn, "view.hp", view.hp, 0, PLAYER_MAX_HP)
    require_int(fn, "view.turn", view.turn, 1, TURN_LIMIT)
    require_flag(fn, "view.weakened", view.weakened)
    require_flag(fn, "view.exposed", view.exposed)
    require_flag(fn, "view.spikes", view.spikes)
    return view
end

--- Copy of a view, so a caller replaying a log cannot write into the
--- table the corpus was built from.
local function copy_player_view(view)
    return {
        turn = view.turn,
        mode = view.mode,
        boss_hp = view.boss_hp,
        shift_distance = view.shift_distance,
        hp = view.hp,
        weakened = view.weakened,
        exposed = view.exposed,
        spikes = view.spikes,
    }
end

--- What the player can see at the head of a turn.
---
--- One constructor for both callers — the interactive transcript and
--- the player NPC's own playouts — because the two have to agree: a
--- view built one way at log time and another way at play time would
--- train a model on a question nobody asks it afterwards.
---
--- `style` is the distance basis the `D` field is measured against, the
--- same argument `encode` takes and for the same reason. The board a
--- human plays from shows that distance, so a logged view records the
--- number that was on the screen.
---
--- `exposed` and `spikes` are not printed on the board as such, but
--- both are knowable to the player who is looking at it: the exposure
--- is the heavy attack they themselves played last turn, and the spikes
--- go up with the mode shift the board reports.
---@param g table Game
---@param style string One of `STYLES`, the distance basis
---@return table view `{ turn, mode, boss_hp, shift_distance, hp, weakened, exposed, spikes }`
function M.player_view(g, style)
    require_style("player_view", style)
    if type(g) ~= "table" or type(g.boss) ~= "table" or type(g.player) ~= "table" then
        error("guardian_duel.player_view: game must carry a boss and a player")
    end
    local boss = require_boss_state("player_view", g.boss)
    require_int("player_view", "player.hp", g.player.hp, 0, PLAYER_MAX_HP)
    local weak = g.player.weak
    if weak ~= nil and type(weak) ~= "boolean" then
        error("guardian_duel.player_view: player.weak must be a boolean, got " .. type(weak))
    end
    local thorns = boss.thorns or 0
    require_int("player_view", "boss.thorns", thorns, 0, THORNS)
    return {
        turn = boss.turn,
        mode = boss.mode,
        boss_hp = boss.hp,
        shift_distance = M.shift_distance(style, boss),
        hp = g.player.hp,
        weakened = weak == true,
        exposed = boss.last_player == HEAVY_INDEX,
        spikes = thorns > 0,
    }
end

--- Encode a player view as one line over the player alphabet.
---
--- Layout: `M<boss mode>H<boss health>D<distance to the next mode
--- shift>Y<your health>T<turn>S<status flags>`, every field one
--- character.
---
--- ## The field that is not here
---
--- Six fields fit; seven were wanted. The one left out is the boss
--- cycle index — the position it holds in its four-move script — and it
--- is left out because the player never has it. The board a human plays
--- from shows the mode, both healths, the distance, the turn and the
--- weakened flag; it does not show which entry of the cycle comes next,
--- and the poke exists precisely so that look can be bought for a turn.
--- Encoding it would hand the model a field no human decision was ever
--- a function of, and would also seat an NPC that reads the boss script
--- through a wall.
---
--- What the six that are here buy, in the order a player uses them:
---
--- - blocking well needs to know what is coming. `M` and `D` are the
---   two halves of that: `D0` says the cycle is about to be
---   interrupted, `M1` says the slam is already on its way, and the
---   `S` bit for the spikes says an attack costs health this turn
--- - pressing an advantage needs `H`, and `S` says whether this turn's
---   attack lands weakened
--- - the endgame arithmetic needs `Y`, `H` and `T` together: the fight
---   is decided on health buckets at the turn limit, so how many turns
---   are left decides whether to race or to stall
---@param view table Player view, as `player_view` builds one
---@return string encoded Exactly `PLAYER_ENCODED_LEN` characters
function M.player_encode(view)
    require_player_view("player_encode", view)
    local flags = 0
    for field, weight in pairs(PLAYER_FLAGS) do
        if view[field] then
            flags = flags + weight
        end
    end
    local text = table.concat({
        "M",
        tostring(view.mode),
        "H",
        tostring(M.hp_bucket(view.boss_hp)),
        "D",
        tostring(M.distance_bucket(view.shift_distance)),
        "Y",
        tostring(M.hp_bucket(view.hp)),
        "T",
        tostring(view.turn),
        "S",
        tostring(flags),
    })
    if #text ~= PLAYER_ENCODED_LEN then
        error(
            string.format(
                "guardian_duel.player_encode: view encoded to %d chars but the layout is %d (%s)",
                #text,
                PLAYER_ENCODED_LEN,
                text
            )
        )
    end
    return text
end

--- Encode one player training line and pad it to the context window.
local function make_player_row(view, action, ctx_len, pad_id)
    local ids = M.player_to_ids(M.player_encode(view) .. ">" .. action .. "\n")
    if #ids > ctx_len then
        error(
            string.format(
                "guardian_duel.rows_from_player_moves: encoded line needs %d tokens "
                    .. "but the context is %d",
                #ids,
                ctx_len
            )
        )
    end
    for _ = #ids + 1, ctx_len do
        ids[#ids + 1] = pad_id
    end
    return ids
end

--- Build the supervised corpus that teaches a play log to a model.
---
--- Where `build_corpus` labels sampled boss states with a policy
--- function, this labels the player views of a log with the moves that
--- were actually played — the `move_log` an interactive session hands
--- back when it ends, passed in unchanged. There is no teacher function
--- behind such a corpus, so nothing downstream can score the model on a
--- position the log does not contain.
---
--- Every entry is validated and every failure names the entry: a log is
--- data the caller collected elsewhere, and a view that is off by a
--- field still encodes to characters inside the alphabet.
---
--- Returns the padded rows and the `{ view, action }` pairs behind
--- them, so a caller that also replays the log against the trained
--- model does not have to validate it a second time.
---@param moves table[] Log entries `{ player = <view>, player_action = <move> }`
---@param opts table `{ ctx_len, pad_id? }`
---@return integer[][] rows Token id rows, each `ctx_len` long
---@return table[] plays `{ { view = <player view>, action = <move> } }`
function M.rows_from_player_moves(moves, opts)
    if type(moves) ~= "table" then
        error("guardian_duel.rows_from_player_moves: moves must be a table, got " .. type(moves))
    end
    if #moves == 0 then
        error("guardian_duel.rows_from_player_moves: moves must hold at least one logged move")
    end
    if type(opts) ~= "table" then
        error("guardian_duel.rows_from_player_moves: opts must be a table, got " .. type(opts))
    end
    local ctx_len = tonumber(opts.ctx_len)
    if ctx_len == nil or ctx_len < 1 then
        error("guardian_duel.rows_from_player_moves: opts.ctx_len must be a positive number")
    end
    ctx_len = math.floor(ctx_len)
    local pad_id = opts.pad_id
    if pad_id == nil then
        pad_id = PLAYER_TO_ID["\0"]
    end
    if type(pad_id) ~= "number" then
        error(
            "guardian_duel.rows_from_player_moves: opts.pad_id must be a number, got "
                .. type(pad_id)
        )
    end

    local rows, plays = {}, {}
    for i = 1, #moves do
        local move = moves[i]
        if type(move) ~= "table" then
            error(
                string.format(
                    "guardian_duel.rows_from_player_moves: move %d must be a table, got %s",
                    i,
                    type(move)
                )
            )
        end
        -- The entry is a transcript row, so the view sits under `player`
        -- next to the boss half of the same turn. An entry without one
        -- is a log from a session that did not record the player's side
        -- rather than a turn the player sat out.
        local where = string.format("rows_from_player_moves: move %d", i)
        local view = require_player_view(where, move.player)
        local action = move.player_action
        if PLAYER_INDEX[action] == nil then
            error(
                string.format(
                    "guardian_duel.rows_from_player_moves: move %d played %s, "
                        .. "which is not one of the moves %s",
                    i,
                    tostring(action),
                    table.concat(PLAYER_ACTIONS, ", ")
                )
            )
        end
        rows[#rows + 1] = make_player_row(view, action, ctx_len, pad_id)
        plays[#plays + 1] = { view = copy_player_view(view), action = action }
    end
    return rows, plays
end

-- ─── Synthesised policies ───────────────────────────────────────────
--
-- A persona boss starts as a Lua chunk written by an LLM
-- (`examples/gameai/bake_guardian_persona.lua`). Such a chunk is never
-- loaded raw: it is compiled into a restricted environment and then has
-- to answer a batch of sampled states legally and deterministically
-- before it is allowed to label a single training line.

--- Collect boss states from random self-play.
---
--- Both sides move at random, so the walk reaches mode / cycle / damage
--- combinations a scripted fight never produces, and it is fully
--- determined by `seed`, so a validation verdict is reproducible.
---@param opts table|nil `{ games?, seed? }`
---@return table[] states One state per turn played
function M.sample_states(opts)
    opts = opts or {}
    local games = math.floor(tonumber(opts.games) or 20)
    if games < 1 then
        error("guardian_duel.sample_states: opts.games must be a positive number")
    end
    local seed = math.floor(tonumber(opts.seed) or 1)
    local math_ns = require_rng()
    local states = {}
    for i = 1, games do
        local g = M.new_game(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not M.is_over(g) do
            states[#states + 1] = g.boss
            g = M.apply(g, M.policy_player_random(rng), M.policy_boss_random(g.boss, rng))
        end
    end
    return states
end

--- Environment a synthesised chunk is compiled into.
---
--- Only the pure parts of the standard library are reachable, and
--- `math` / `table` are shallow copies so a chunk cannot reach the host
--- tables through them. There is no `load`, no `setmetatable`, no
--- `os` / `io` / `require`, so a chunk can compute but cannot observe
--- or change anything outside its own argument. `math.random` is
--- present but useless: the determinism check below rejects any policy
--- that answers the same state twice with two different moves.
local function sandbox_env()
    local math_copy, table_copy = {}, {}
    for k, v in pairs(math) do
        math_copy[k] = v
    end
    for k, v in pairs(table) do
        table_copy[k] = v
    end
    return {
        math = math_copy,
        table = table_copy,
        ipairs = ipairs,
        pairs = pairs,
    }
end

--- Whether `action` is one of the moves `state` allows.
local function is_legal_action(state, action)
    if type(action) ~= "string" then
        return false
    end
    for _, ch in ipairs(M.legal_actions(state)) do
        if ch == action then
            return true
        end
    end
    return false
end

--- Compile and validate a synthesised boss policy chunk.
---
--- `source` is expected to be `return function(state) ... end`. It is
--- loaded in text mode only (never bytecode) into `sandbox_env`, and
--- the returned function is then asked about every state in
--- `opts.states` (sampled from random self-play when the caller passes
--- none). A candidate is accepted only when, for every state, the call
--- succeeds, the answer is a legal move, and a second call with the
--- same state returns the same move.
---
--- Every rejection is a loud error naming the state and the reason, so
--- a caller driving an LLM can feed the message straight back into the
--- next synthesis attempt.
---
--- The accepted policy is wrapped so it receives a copy of the state: a
--- chunk that rewrites `mode` or `damage_since_shift` would otherwise
--- corrupt the fight it is labelling, which is a silent data fault
--- rather than a visible one.
---@param source string Lua chunk returning a policy function
---@param opts table|nil `{ states?, games?, seed?, chunk_name? }`
---@return fun(state: table): string policy
function M.compile_policy(source, opts)
    if type(source) ~= "string" then
        error("guardian_duel.compile_policy: source must be a string, got " .. type(source))
    end
    opts = opts or {}
    local chunk_name = opts.chunk_name or "synthesised_policy"
    local chunk, load_err = load(source, "=" .. chunk_name, "t", sandbox_env())
    if chunk == nil then
        error("guardian_duel.compile_policy: source does not compile: " .. tostring(load_err))
    end
    local ran, policy = pcall(chunk)
    if not ran then
        error(
            "guardian_duel.compile_policy: chunk raised while returning the policy: "
                .. tostring(policy)
        )
    end
    if type(policy) ~= "function" then
        error("guardian_duel.compile_policy: chunk must return a function, got " .. type(policy))
    end

    local states = opts.states
    if states == nil then
        states = M.sample_states({ games = opts.games, seed = opts.seed })
    end
    if type(states) ~= "table" or #states == 0 then
        error("guardian_duel.compile_policy: opts.states must hold at least one state")
    end

    local guarded = function(state)
        return policy(copy_boss(require_boss_state("compile_policy", state)))
    end

    for i, state in ipairs(states) do
        local ok, action = pcall(guarded, state)
        if not ok then
            error(
                string.format(
                    "guardian_duel.compile_policy: policy raised on state %d (%s): %s",
                    i,
                    state_summary(state),
                    tostring(action)
                )
            )
        end
        if not is_legal_action(state, action) then
            error(
                string.format(
                    "guardian_duel.compile_policy: policy answered %s on state %d (%s), "
                        .. "which is not one of the legal moves %s",
                    tostring(action),
                    i,
                    state_summary(state),
                    table.concat(M.legal_actions(state), ", ")
                )
            )
        end
        local ok_again, again = pcall(guarded, state)
        if not ok_again or again ~= action then
            error(
                string.format(
                    "guardian_duel.compile_policy: policy is not deterministic on state %d (%s): "
                        .. "%s then %s",
                    i,
                    state_summary(state),
                    tostring(action),
                    tostring(again)
                )
            )
        end
    end

    return guarded
end

-- ─── Strategy entry ─────────────────────────────────────────────────

--- Return the encoded opening state for a seed.
---
--- The style decides what the distance field measures, so the demo
--- entry names the teacher when the caller does not: an opening state
--- reads `D3` for the teacher and `D2` for the impatient variant, and
--- both are the same fight.
---@param ctx table `{ seed?, style? }`
---@return table result `{ result = <encoded state> }`
function M.run(ctx)
    ctx = ctx or {}
    local seed = tonumber(ctx.seed) or 1
    local style = ctx.style or M.STYLES[1]
    local g = M.new_game(seed)
    return { result = M.encode(g.boss, style) }
end

return M
