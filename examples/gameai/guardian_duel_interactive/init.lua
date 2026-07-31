--- guardian_duel_interactive — fight the SLM boss one turn per call
---
--- Turns the one-shot `guardian_duel_npc` decision into a session a
--- human can sit in: one `alc_run` call per turn, with the fight kept in
--- `alc.state` between calls. The package owns no rules of its own — it
--- validates the human move against
--- `guardian_duel.player_legal_actions`, asks the NPC for the boss seat
--- and hands the turn back to `guardian_duel.apply`.
---
--- ## Usage
---
--- ```lua
--- local fight = require("guardian_duel_interactive")
--- fight.run({ action = "new", style = "guardian", seed = 7 })
--- fight.run({ action = "play", move = "A" })
--- fight.run({ action = "play", move = "p" }) -- reveals the next answer
--- fight.run({ action = "show" })
--- fight.run({ action = "end" })
--- ```
---
--- ## Algorithm
---
--- 1. `new` opens a fight with `guardian_duel.new_game` and stores
---    `{ g, style, style_kind, basis_style }` under
---    `"gameai_guardian:" .. game_id`. `alc.state` is file backed, so the
---    session survives between `alc_run` calls; several fights can
---    coexist under different `game_id` values.
--- 2. `play` validates the requested move against the four player moves,
---    asks `guardian_duel_npc` for the boss answer with `mode = "decide"`
---    and the alias `guardian_duel_npc_<style>`, and applies the pair.
---    The boss answers from the state at the head of the turn, which is
---    why the human move never has to be sent to the model.
--- 3. The winner is only read once `guardian_duel.is_over` is true, and
---    it is translated to the human's point of view (`you` / `boss` /
---    `draw`). An unfinished fight has no winner; reporting it as a draw
---    would hide a session that stopped early.
--- 4. Every turn is appended to a `move_log` kept next to the fight: the
---    boss state the answer came from, the player view the human chose
---    from, the boss answer and the human move. `end` returns that log.
--- 5. `end` deletes the session. A finished fight is kept until then, so
---    the final board can still be shown.
---
--- ## Entry contract
---
--- `run(ctx)` dispatches on `ctx.action`:
---
--- - `new` — `{ style?, basis_style?, seed?, game_id? }`
--- - `play` — `{ move, game_id? }`
--- - `show` — `{ game_id? }`
--- - `end` — `{ game_id? }`
---
--- Every action returns the same board view — `turn`, `your_hp`,
--- `boss_hp`, `boss_mode`, `boss_shifts`, `shift_distance`,
--- `legal_actions`, `move_help`, `status` and a one-line `text` — plus
--- `style`, `style_kind` (`canonical` / `persona`) and `basis_style`,
--- `intent` while the boss answer is revealed, `last_turn` after a move,
--- `winner` once the fight is over, and `ended` plus `move_log` after
--- `end`. `result` mirrors `text`, which is the field an algocline
--- caller reads by convention.
---
--- `shift_distance` is the damage the boss still tolerates before it
--- rolls up, measured against `basis_style`. It is shown rather than
--- hidden because it is the same number the model reads as the `D` field
--- of its encoded state: the fight is about a boss whose staggering is
--- legible, not about guessing it. It is omitted once the fight is over,
--- where the position is past the range the rules encode.
---
--- ## The poke, and what `intent` means here
---
--- `p` deals the least damage of the four player moves in exchange for
--- knowing the boss answer of the following turn. That is the
--- Slay the Spire *Intent* display scaled down to one move: there, the
--- next enemy action is shown every turn; here it has to be bought.
---
--- The reveal is a look-ahead of the *next decode*, not a second
--- prediction that could disagree with it. Right after a poke lands, the
--- session decodes the answer for the position that follows and stores
--- it; the next `play` plays that stored move instead of decoding again.
--- The board therefore cannot promise one move and play another. A
--- stored reveal that does not belong to the turn about to be played is
--- a loud error rather than a fresh decode, because it means the session
--- was rewritten between calls and the player paid for a look at a
--- position that no longer exists.
---
--- ## Styles
---
--- A *canonical* style is one `guardian_duel` ships a `policy_<style>`
--- for, and it is its own distance basis. A *persona* style is baked
--- from a prompt by `examples/gameai/bake_guardian_persona.lua`: it
--- leaves a Card pinned to `guardian_duel_npc_<style>` behind and no
--- teacher policy at all, so the name is accepted once that alias
--- resolves. A persona has no mode-shift threshold of its own, so it
--- borrows one at bake time and is decoded under that borrowed basis for
--- the rest of its life; the session reads it from `persona.basis_style`
--- on the Card and refuses to guess when it is absent. A name that is
--- neither canonical nor pinned fails loudly rather than falling back to
--- the default style, which would seat a boss the caller never asked
--- for.
---
--- ## Caveats
---
--- The boss side needs a build with the `nn` feature and a Card pinned
--- to `guardian_duel_npc_<style>`; without one the `play` action fails
--- at Card load rather than falling back to `policy_guardian`, which
--- would quietly stop measuring the model.
---
--- The session is stored under a caller-chosen `game_id` with no expiry.
--- Sessions that are never ended stay in the state file.
---
--- `move_log` is a transcript of both seats. Each entry carries the boss
--- state an answer came from and the answer itself — the pair
--- `guardian_duel.build_corpus` labels — next to the player view the
--- human chose from and the move they played, which is the pair
--- `guardian_duel.rows_from_player_moves` bakes. The second half is
--- what `examples/gameai/bake_guardian_player_from_log.lua` reads; the
--- boss half has no bake script of its own, because a boss learned from
--- a transcript would be a copy of the model that produced it.
---
--- The player view is the position as the board showed it, recorded
--- before the turn was applied. It is measured against `basis_style`,
--- which is also what the board's `shift_distance` is measured against,
--- so a Card baked from the log answers under the basis it was logged
--- under and no other.

local duel = require("guardian_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "guardian_duel_interactive",
    version = "0.1.0",
    description = "Interactive guardian duel against the SLM boss NPC, persisted through alc.state",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `guardian_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            action = T.string:describe("One of: new / play / show / end"),
            move = T.string:is_optional():describe(
                "Player move, required by the play action: a light / A heavy / "
                    .. "b block / p poke"
            ),
            style = T.string:is_optional():describe(
                "Boss style for a new fight: a guardian_duel style or a persona style "
                    .. "pinned to guardian_duel_npc_<style> (default: guardian)"
            ),
            basis_style = T.string:is_optional():describe(
                "Distance basis of a persona style, overriding the one on its Card"
            ),
            seed = T.number:is_optional():describe("Fight seed for a new fight"),
            game_id = T.string:is_optional():describe("Session name (default: default)"),
        }),
        result = T.string:describe("One-line board summary; the full view is returned alongside"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

--- State key prefix. The suffix is the caller's `game_id`.
local KEY_PREFIX = "gameai_guardian:"

--- Card alias prefix written by the training and bake scripts.
local ALIAS_PREFIX = "guardian_duel_npc_"

local DEFAULT_STYLE = "guardian"
local DEFAULT_SEED = 20260731
local DEFAULT_GAME_ID = "default"

--- What the four player moves do, handed out with the board so a caller
--- reading `legal_actions` does not have to keep the rules open.
local MOVE_HELP = {
    a = "light attack",
    A = "heavy attack, and the boss hits harder next turn",
    b = "block the answer of this turn",
    p = "poke, and see the boss answer of the next turn",
}

-- ─── Host surface guards ────────────────────────────────────────────

--- Resolve `alc.state` at call time so a caller may stub it.
local function state_ns()
    if type(alc) ~= "table" or type(alc.state) ~= "table" then
        error("guardian_duel_interactive: alc.state is required to persist the session")
    end
    return alc.state
end

local function require_json()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error("guardian_duel_interactive: alc.json_encode is required")
    end
end

--- Fetch the Card pinned to `alias`.
---
--- A persona style has no `policy_<style>` in `guardian_duel` — the bake
--- script leaves only a Card behind — so the Card is both the evidence
--- that the style exists and the place its distance basis is recorded. A
--- host that cannot answer is reported through `reason` rather than as
--- "no such alias": the two cases need different fixes, and folding the
--- first into the second would tell the caller to check a spelling that
--- is already right.
---@param alias string
---@return table|nil card
---@return string|nil reason Why there is no Card to return
local function alias_card(alias)
    if
        type(alc) ~= "table"
        or type(alc.card) ~= "table"
        or type(alc.card.get_by_alias) ~= "function"
    then
        return nil, "alc.card.get_by_alias is unavailable"
    end
    local ok, card = pcall(alc.card.get_by_alias, alias)
    if not ok then
        return nil, tostring(card)
    end
    if card == nil then
        return nil, "no Card is pinned to it"
    end
    return card, nil
end

-- ─── Request parsing ────────────────────────────────────────────────

--- Read an optional numeric field.
---
--- A field the caller omitted falls back to `default`, but a field that
--- is present and not a number fails loudly: `seed = "later"` silently
--- becoming the default seed would open a fight nobody asked for.
---@param name string Field name, for the message
---@param raw any
---@param default number
---@return number
local function optional_number(name, raw, default)
    if raw == nil then
        return default
    end
    local value = tonumber(raw)
    if value == nil then
        error(
            string.format(
                "guardian_duel_interactive: %s must be a number, got '%s'",
                name,
                tostring(raw)
            )
        )
    end
    return value
end

-- ─── Session helpers ────────────────────────────────────────────────

local function session_key(ctx)
    local game_id = ctx.game_id
    if game_id ~= nil and type(game_id) ~= "string" then
        error("guardian_duel_interactive: game_id must be a string, got " .. type(game_id))
    end
    return KEY_PREFIX .. (game_id or DEFAULT_GAME_ID)
end

--- Load the session or fail loudly.
---
--- A missing session means the caller skipped `new`, which is worth an
--- error rather than an implicit fresh fight: the fight would use a seed
--- and a style nobody chose.
local function load_session(key)
    local session = state_ns().get(key)
    if type(session) ~= "table" or type(session.g) ~= "table" then
        error('guardian_duel_interactive: no active fight (call action="new" first)')
    end
    return session
end

local function save_session(key, session)
    state_ns().set(key, session)
end

-- ─── Shared helpers ─────────────────────────────────────────────────

local function copy_list(list)
    local out = {}
    for i, v in ipairs(list or {}) do
        out[i] = v
    end
    return out
end

--- The seven fields of a boss state the encoding and the rules read.
---
--- The live state also carries the block and spike counters, which are
--- engine bookkeeping: they never reach the model, so they are left out
--- of the request and out of the transcript rather than shipped across
--- the JSON boundary as though they mattered to the answer.
local function boss_payload(boss)
    return {
        cycle = boss.cycle,
        mode = boss.mode,
        hp = boss.hp,
        damage_since_shift = boss.damage_since_shift,
        last_player = boss.last_player,
        turn = boss.turn,
        shifts = boss.shifts,
    }
end

local function copy_move_help()
    local out = {}
    for move, text in pairs(MOVE_HELP) do
        out[move] = text
    end
    return out
end

local function canonical_names()
    return table.concat(duel.STYLES, ", ")
end

-- ─── Styles ─────────────────────────────────────────────────────────

--- Check a distance basis against `guardian_duel.STYLES`.
---@param raw any
---@param source string Where the name came from, for the message
---@return string basis
local function require_basis(raw, source)
    if type(raw) ~= "string" then
        error(
            string.format(
                "guardian_duel_interactive: %s must be a string, got %s",
                source,
                type(raw)
            )
        )
    end
    for _, name in ipairs(duel.STYLES) do
        if name == raw then
            return raw
        end
    end
    error(
        string.format(
            "guardian_duel_interactive: %s %q is not one of %s",
            source,
            raw,
            canonical_names()
        )
    )
end

--- Resolve the boss style of a new fight, its kind and its basis.
---
--- A canonical style is decided by the `policy_<style>` field and is its
--- own basis, so the alias lookup is never reached for one and a persona
--- Card cannot shadow a shipped style. A persona style borrows a basis,
--- which is read from the Card the bake script wrote it onto; a caller
--- may override it, but neither the session nor the NPC will guess one,
--- because a wrong basis feeds the model a `D` field its corpus never
--- carried and the answers stay legal while quietly meaning something
--- else.
---@param raw any Requested style name, or nil for the default
---@param basis_raw any Requested distance basis, or nil
---@return string style
---@return string kind `"canonical"` or `"persona"`
---@return string basis
local function resolve_style(raw, basis_raw)
    local style = raw or DEFAULT_STYLE
    if type(style) ~= "string" then
        error(
            string.format("guardian_duel_interactive: style must be a string, got %s", type(style))
        )
    end
    if type(duel["policy_" .. style]) == "function" then
        if basis_raw ~= nil then
            error(
                string.format(
                    "guardian_duel_interactive: style %q is canonical and is its own distance "
                        .. "basis, so basis_style %s cannot apply to it",
                    style,
                    tostring(basis_raw)
                )
            )
        end
        return style, "canonical", style
    end

    local alias = ALIAS_PREFIX .. style
    local card, reason = alias_card(alias)
    if card == nil then
        error(
            string.format(
                "guardian_duel_interactive: unknown style %s (canonical: %s; persona alias %s: %s)",
                tostring(raw),
                canonical_names(),
                alias,
                reason or "no Card is pinned to it"
            )
        )
    end
    if basis_raw ~= nil then
        return style, "persona", require_basis(basis_raw, "basis_style")
    end
    local persona = card.persona
    local basis = type(persona) == "table" and persona.basis_style or nil
    if basis == nil then
        error(
            string.format(
                "guardian_duel_interactive: persona style %q carries no persona.basis_style on "
                    .. "alias %s; pass basis_style to name the threshold its states are measured "
                    .. "against",
                style,
                alias
            )
        )
    end
    return style, "persona", require_basis(basis, "persona.basis_style of " .. alias)
end

-- ─── Boss seat ──────────────────────────────────────────────────────

--- Ask the NPC package for the boss answer to the current position.
---
--- The answer is checked against the legal moves even though the decode
--- gate already restricts it: this session also replays a stored answer
--- a turn after it was decoded, and a failure that only surfaced inside
--- `guardian_duel.apply` would point at the replay rather than at the
--- decode that produced it.
---@param session table
---@return string action
local function decode_boss(session)
    require_json()
    local npc = require("guardian_duel_npc")
    local out = npc.run({
        task = alc.json_encode({ mode = "decide", state = boss_payload(session.g.boss) }),
        card_alias = ALIAS_PREFIX .. session.style,
        style = session.basis_style,
    })
    local text = type(out) == "table" and out.result or out
    if type(text) ~= "string" then
        error("guardian_duel_interactive: NPC answer must be a string, got " .. type(text))
    end
    local action = text:match("action=(%a)")
    if action == nil then
        error(string.format("guardian_duel_interactive: NPC answer %q carries no action", text))
    end
    local legal = duel.legal_actions(session.g.boss)
    for _, candidate in ipairs(legal) do
        if candidate == action then
            return action
        end
    end
    error(
        string.format(
            "guardian_duel_interactive: NPC answered %s, which is not one of the legal moves %s",
            action,
            table.concat(legal, ", ")
        )
    )
end

--- The revealed boss answer for the turn about to be played, if any.
---
--- A stored reveal always belongs to the current turn: it is written
--- right after the poke that bought it and consumed by the next move. A
--- mismatch means the session was rewritten between calls, which is
--- reported rather than shown as "nothing was revealed" — the board
--- would otherwise silently drop a look the player paid a turn for.
---@param session table
---@return string|nil action
local function current_intent(session)
    local intent = session.intent
    if intent == nil then
        return nil
    end
    if type(intent) ~= "table" or type(intent.action) ~= "string" then
        error("guardian_duel_interactive: the stored reveal is not a move")
    end
    if intent.turn ~= session.g.turn then
        error(
            string.format(
                "guardian_duel_interactive: the revealed answer belongs to turn %s but the fight "
                    .. "is on turn %s",
                tostring(intent.turn),
                tostring(session.g.turn)
            )
        )
    end
    return intent.action
end

-- ─── Play log ───────────────────────────────────────────────────────

--- Copy of a player view, so a caller cannot write back into the
--- transcript through the log it was handed.
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

--- Append the turn that was just played.
---
--- The game is the one both seats were asked about, before `apply`: the
--- boss state is the position the model answered from, and the player
--- view is what the human was looking at when they chose their move.
--- After `apply` both belong to a question nobody put to either side.
---
--- The two halves are kept apart because they are two different
--- projections of the same turn, and each is baked through its own
--- encoding: `guardian_duel.encode` reads the boss half,
--- `guardian_duel.player_encode` reads the player half.
local function log_turn(session, g, player_action, boss_action, revealed)
    local log = session.move_log
    if log == nil then
        log = {}
        session.move_log = log
    end
    log[#log + 1] = {
        turn = g.boss.turn,
        boss = boss_payload(g.boss),
        player = duel.player_view(g, session.basis_style),
        boss_action = boss_action,
        player_action = player_action,
        revealed = revealed,
    }
end

--- The log as the caller receives it: a copy, entries included.
---
--- A fight that took no turn returns an empty array rather than nothing,
--- so a caller can tell an unplayed session from a version of this
--- package that did not keep a transcript yet.
---
--- An entry written by a session that predates the player view carries
--- none, and is handed back without one rather than with an invented
--- snapshot: `guardian_duel.rows_from_player_moves` then rejects that
--- entry by name at bake time, which is where the gap belongs.
local function copy_move_log(session)
    local out = {}
    for i, entry in ipairs(session.move_log or {}) do
        out[i] = {
            turn = entry.turn,
            boss = boss_payload(entry.boss),
            player = entry.player and copy_player_view(entry.player) or nil,
            boss_action = entry.boss_action,
            player_action = entry.player_action,
            revealed = entry.revealed,
        }
    end
    return out
end

-- ─── Board view ─────────────────────────────────────────────────────

--- Translate the rules verdict to the human's point of view.
---
--- Only called once `is_over` is true; `guardian_duel.winner` returns
--- nil for a running fight and folding that into `"draw"` would report a
--- result for a fight that has none.
local function human_winner(session)
    local verdict = duel.winner(session.g)
    if verdict == nil then
        error("guardian_duel_interactive: winner asked for a fight that is not over")
    end
    if verdict == "player" then
        return "you"
    end
    return verdict
end

local function render_text(view)
    if view.status == "finished" then
        local verdict = "a draw"
        if view.winner == "you" then
            verdict = "you win"
        elseif view.winner == "boss" then
            verdict = "the boss wins"
        end
        return string.format(
            "fight over: you %d hp - boss %d hp, %s",
            view.your_hp,
            view.boss_hp,
            verdict
        )
    end
    local stance
    if view.boss_mode == 1 then
        stance = "it is rolled up"
    else
        stance = string.format("%d damage to its next shift", view.shift_distance)
    end
    local text = string.format(
        "turn %d of %d: you %d hp - boss %d hp, %s, play one of %s",
        view.turn,
        duel.TURN_LIMIT,
        view.your_hp,
        view.boss_hp,
        stance,
        table.concat(view.legal_actions, ", ")
    )
    if view.intent ~= nil then
        text = text .. " | it will answer " .. view.intent
    end
    return text
end

--- Build the board the caller sees after every action.
local function board_view(session)
    local g = session.g
    local over = duel.is_over(g)
    local view = {
        turn = g.turn,
        style = session.style,
        style_kind = session.style_kind,
        basis_style = session.basis_style,
        your_hp = g.player.hp,
        weakened = g.player.weak == true,
        boss_hp = g.boss.hp,
        boss_mode = g.boss.mode,
        boss_shifts = g.boss.shifts,
        legal_actions = over and {} or duel.player_legal_actions(),
        move_help = copy_move_help(),
        status = over and "finished" or "your_turn",
    }
    if over then
        view.winner = human_winner(session)
    else
        -- Past the turn limit the boss state leaves the range the rules
        -- encode, so the distance is only read while a turn is playable.
        view.shift_distance = duel.shift_distance(session.basis_style, g.boss)
        view.intent = current_intent(session)
    end
    view.text = render_text(view)
    return view
end

-- ─── Actions ────────────────────────────────────────────────────────

local function action_new(ctx, key)
    local style, kind, basis = resolve_style(ctx.style, ctx.basis_style)
    local session = {
        g = duel.new_game(math.floor(optional_number("seed", ctx.seed, DEFAULT_SEED))),
        style = style,
        style_kind = kind,
        basis_style = basis,
    }
    save_session(key, session)
    return board_view(session)
end

--- Validate the human move against the four the rules allow.
local function check_move(raw)
    if type(raw) ~= "string" then
        error(string.format("guardian_duel_interactive: move must be a string, got %s", type(raw)))
    end
    local legal = duel.player_legal_actions()
    for _, candidate in ipairs(legal) do
        if candidate == raw then
            return raw
        end
    end
    error(
        string.format(
            "guardian_duel_interactive: move %q is not one of %s",
            raw,
            table.concat(legal, ", ")
        )
    )
end

local function action_play(ctx, key)
    local session = load_session(key)
    if duel.is_over(session.g) then
        error('guardian_duel_interactive: the fight is over (call action="end" or start a new one)')
    end

    local turn = session.g.turn
    local player_action = check_move(ctx.move)
    -- A revealed answer is replayed rather than decoded again, so the
    -- board cannot show one move and the fight play another.
    local revealed = current_intent(session)
    local boss_action = revealed or decode_boss(session)

    log_turn(session, session.g, player_action, boss_action, revealed ~= nil)
    session.g = duel.apply(session.g, player_action, boss_action)

    -- The rules mark the position a poke bought; reading the answer for
    -- it now is what makes the reveal a look-ahead of the next decode
    -- rather than a prediction of its own.
    session.intent = nil
    if session.g.revealed and not duel.is_over(session.g) then
        session.intent = { turn = session.g.turn, action = decode_boss(session) }
    end
    save_session(key, session)

    local view = board_view(session)
    view.last_turn = {
        turn = turn,
        you = player_action,
        boss = boss_action,
        revealed = revealed ~= nil,
    }
    view.text = string.format(
        "turn %d: you played %s, the boss answered %s | %s",
        turn,
        player_action,
        boss_action,
        view.text
    )
    return view
end

local function action_show(_ctx, key)
    return board_view(load_session(key))
end

local function action_end(_ctx, key)
    local session = load_session(key)
    local view = board_view(session)
    view.move_log = copy_move_log(session)
    state_ns().delete(key)
    view.ended = true
    view.text = "session ended: " .. view.text
    return view
end

-- ─── Strategy entry ─────────────────────────────────────────────────

---@param ctx table `{ action, move?, style?, basis_style?, seed?, game_id? }`
---@return table view Board view for the resulting position
function M.run(ctx)
    ctx = ctx or {}
    local action = ctx.action
    if type(action) ~= "string" then
        error("guardian_duel_interactive: ctx.action must be a string, got " .. type(action))
    end
    local key = session_key(ctx)

    local view
    if action == "new" then
        view = action_new(ctx, key)
    elseif action == "play" then
        view = action_play(ctx, key)
    elseif action == "show" then
        view = action_show(ctx, key)
    elseif action == "end" then
        view = action_end(ctx, key)
    else
        error(
            string.format(
                "guardian_duel_interactive: unknown action %q (expected new / play / show / end)",
                action
            )
        )
    end

    -- `result` is the conventional output field an algocline caller
    -- reads; it mirrors `text` so the structured view stays available.
    view.result = view.text
    return view
end

return M
