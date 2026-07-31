--- card_duel_interactive — play a hand of card duel against the SLM NPC
---
--- Turns the one-shot `card_duel_npc` decision into a session a human
--- can sit in: one `alc_run` call per move, with the game kept in
--- `alc.state` between calls. The package owns no rules of its own — it
--- validates the human move against `card_duel.legal_actions`, asks the
--- NPC for the other seat and hands the round back to `card_duel.apply`.
---
--- ## Usage
---
--- ```lua
--- local game = require("card_duel_interactive")
--- game.run({ action = "new", style = "aggressive", seed = 7 })
--- game.run({ action = "play", rank = 9 })
--- game.run({ action = "show" })
--- game.run({ action = "end" })
--- ```
---
--- ## Algorithm
---
--- 1. `new` deals a game with `card_duel.new_game` and stores
---    `{ g, style, style_kind, user_seat }` under
---    `"gameai_interactive:" .. game_id`. `alc.state` is file backed, so
---    the session survives between `alc_run` calls; several sessions can
---    coexist under different `game_id` values.
--- 2. `play` validates the requested rank against the legal actions of
---    the human seat, asks `card_duel_npc` for the other seat with
---    `mode = "decide"` and the alias `card_duel_npc_<style>`, then
---    applies both moves in seat order — the arguments of
---    `card_duel.apply` are `(p1, p2)`, so they are swapped when the
---    human sits in seat two.
--- 3. The winner is only read once `card_duel.is_over` is true, and it
---    is translated to the human's point of view (`you` / `npc` /
---    `draw`). An unfinished game has no winner; reporting it as a draw
---    would hide a session that stopped early.
--- 4. Every human move is appended to a `move_log` kept next to the
---    game: the position the human was looking at (round, hand, both
---    scores, opponent history) and the rank they chose. `end` returns
---    that log, which is the training input of
---    `examples/gameai/bake_card_duel_from_log.lua`.
--- 5. `end` deletes the session. A finished game is kept until then, so
---    the final board can still be shown.
---
--- ## Entry contract
---
--- `run(ctx)` dispatches on `ctx.action`:
---
--- - `new` — `{ style?, seed?, user_seat?, game_id? }`
--- - `play` — `{ rank, game_id? }`
--- - `show` — `{ game_id? }`
--- - `end` — `{ game_id? }`
---
--- Every action returns the same board view — `round`, `your_hand`,
--- `your_points`, `npc_points`, `npc_played`, `legal_actions`, `status`
--- and a one-line `text` — plus `style_kind` (`canonical` / `persona`),
--- `last_round` after a move, `winner` once the game is over, and
--- `ended` plus `move_log` after `end`. `result` mirrors `text`, which
--- is the field an algocline caller reads by convention.
---
--- ## Play log
---
--- The log is only handed out by `end`, because a log is a whole
--- session: rows baked from a session that is still running would be
--- relabelled by the moves that come after them. Each entry is
--- `{ round, my_hand, my_points, opp_points, opp_played, action }`,
--- which is exactly the per-player state `card_duel.encode` reads plus
--- the rank that was played, so `card_duel.rows_from_moves` turns the
--- log into training rows without a translation step. Only human moves
--- are logged; the NPC seat is already a model.
---
--- ## Styles
---
--- A *canonical* style is one `card_duel` ships a `policy_<style>` for.
--- A *persona* style is baked from a prompt by
--- `examples/gameai/bake_card_duel_persona.lua`: it leaves a Card
--- pinned to `card_duel_npc_<style>` behind and no teacher policy at
--- all, so the name is accepted once that alias resolves. Play needs
--- nothing else — every move of the NPC seat is a decode through the
--- alias — so a persona style plays exactly like a canonical one. A
--- name that is neither fails loudly rather than falling back to the
--- default style, which would seat an NPC the caller never asked for.
---
--- ## Caveats
---
--- The NPC side needs a build with the `nn` feature and a Card pinned
--- to `card_duel_npc_<style>`; without one the `play` action fails at
--- Card load rather than falling back to a hand-written policy, which
--- would quietly stop measuring the model.
---
--- The session is stored under a caller-chosen `game_id` with no
--- expiry. Sessions that are never ended stay in the state file.

local duel = require("card_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "card_duel_interactive",
    version = "0.1.0",
    description = "Interactive card duel session against the SLM NPC, persisted through alc.state",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `card_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            action = T.string:describe("One of: new / play / show / end"),
            rank = T.number:is_optional():describe("Rank to play, required by the play action"),
            style = T.string:is_optional():describe(
                "NPC style for a new game: a card_duel style or a persona style "
                    .. "pinned to card_duel_npc_<style> (default: aggressive)"
            ),
            seed = T.number:is_optional():describe("Deal seed for a new game"),
            user_seat = T.number
                :is_optional()
                :describe("Seat the human takes, 1 or 2 (default: 1)"),
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
local KEY_PREFIX = "gameai_interactive:"

--- Card alias prefix written by the training script, one per style.
local ALIAS_PREFIX = "card_duel_npc_"

local DEFAULT_STYLE = "aggressive"
local DEFAULT_SEED = 20260731
local DEFAULT_GAME_ID = "default"

-- ─── Host surface guards ────────────────────────────────────────────

--- Resolve `alc.state` at call time so a caller may stub it.
local function state_ns()
    if type(alc) ~= "table" or type(alc.state) ~= "table" then
        error("card_duel_interactive: alc.state is required to persist the session")
    end
    return alc.state
end

local function require_json()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error("card_duel_interactive: alc.json_encode is required")
    end
end

--- Report whether a Card is pinned to `alias`.
---
--- A persona style has no `policy_<style>` in `card_duel` — the bake
--- script leaves only a Card behind — so the alias is the one piece of
--- evidence that the style exists at all. A host that cannot answer is
--- reported through `reason` rather than as "no such alias": the two
--- cases need different fixes, and folding the first into the second
--- would tell the caller to check a spelling that is already right.
---@param alias string
---@return boolean found
---@return string|nil reason Why the lookup could not be made
local function alias_card_exists(alias)
    if
        type(alc) ~= "table"
        or type(alc.card) ~= "table"
        or type(alc.card.get_by_alias) ~= "function"
    then
        return false, "alc.card.get_by_alias is unavailable"
    end
    local ok, card = pcall(alc.card.get_by_alias, alias)
    if not ok then
        return false, tostring(card)
    end
    return card ~= nil, nil
end

-- ─── Request parsing ────────────────────────────────────────────────

--- Read an optional numeric field.
---
--- A field the caller omitted falls back to `default`, but a field that
--- is present and not a number fails loudly. `user_seat = "second"`
--- silently becoming seat one would hand the human the wrong hand and
--- report the wrong side as the winner.
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
                "card_duel_interactive: %s must be a number, got '%s'",
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
        error("card_duel_interactive: game_id must be a string, got " .. type(game_id))
    end
    return KEY_PREFIX .. (game_id or DEFAULT_GAME_ID)
end

--- Load the session or fail loudly.
---
--- A missing session means the caller skipped `new`, which is worth an
--- error rather than an implicit fresh deal: the deal would use a seed
--- nobody chose.
local function load_session(key)
    local session = state_ns().get(key)
    if type(session) ~= "table" or type(session.g) ~= "table" then
        error('card_duel_interactive: no active game (call action="new" first)')
    end
    return session
end

local function save_session(key, session)
    state_ns().set(key, session)
end

--- The two per-player states, from the human's point of view.
local function seats(session)
    if session.user_seat == 2 then
        return session.g.p2, session.g.p1
    end
    return session.g.p1, session.g.p2
end

-- ─── Shared helpers ─────────────────────────────────────────────────

local function copy_list(list)
    local out = {}
    for i, v in ipairs(list or {}) do
        out[i] = v
    end
    return out
end

-- ─── Play log ───────────────────────────────────────────────────────

--- Append the position the human just answered, plus their answer.
---
--- The two lists are copied on the way in. `my_hand` and `opp_played`
--- are the live tables of the current position, and a caller that
--- rewrites a hand between moves — which the session allows, since the
--- board it hands back is a view — would otherwise rewrite a move that
--- was already made.
local function log_move(session, user, rank)
    local log = session.move_log
    if log == nil then
        log = {}
        session.move_log = log
    end
    log[#log + 1] = {
        round = user.round,
        my_hand = copy_list(user.my_hand),
        my_points = user.my_points,
        opp_points = user.opp_points,
        opp_played = copy_list(user.opp_played),
        action = rank,
    }
end

--- The log as the caller receives it: a copy, entries included.
---
--- A session that never took a move returns an empty array rather than
--- nothing, so a caller can tell an unplayed session from a version of
--- this package that did not keep a log yet.
local function copy_move_log(session)
    local out = {}
    for i, move in ipairs(session.move_log or {}) do
        out[i] = {
            round = move.round,
            my_hand = copy_list(move.my_hand),
            my_points = move.my_points,
            opp_points = move.opp_points,
            opp_played = copy_list(move.opp_played),
            action = move.action,
        }
    end
    return out
end

-- ─── Board view ─────────────────────────────────────────────────────

local function join(list)
    if #list == 0 then
        return "-"
    end
    local parts = {}
    for i, v in ipairs(list) do
        parts[i] = tostring(v)
    end
    return table.concat(parts, ",")
end

--- Translate the rules verdict to the human's point of view.
---
--- Only called once `is_over` is true; `card_duel.winner` returns nil
--- for a running game and folding that into `"draw"` would report a
--- result for a game that has none.
local function human_winner(session)
    local verdict = duel.winner(session.g)
    if verdict == nil then
        error("card_duel_interactive: winner asked for a game that is not over")
    end
    if verdict == "draw" then
        return "draw"
    end
    local user_is_p1 = session.user_seat ~= 2
    if (verdict == "p1") == user_is_p1 then
        return "you"
    end
    return "npc"
end

local function render_text(view)
    if view.status == "finished" then
        local verdict = "a draw"
        if view.winner == "you" then
            verdict = "you win"
        elseif view.winner == "npc" then
            verdict = "the npc wins"
        end
        return string.format(
            "game over: you %d - %d npc, %s",
            view.your_points,
            view.npc_points,
            verdict
        )
    end
    return string.format(
        "round %d of %d: you %d - %d npc, hand %s, play one of %s",
        view.round,
        duel.HAND_SIZE,
        view.your_points,
        view.npc_points,
        join(view.your_hand),
        join(view.legal_actions)
    )
end

--- Build the board the caller sees after every action.
local function board_view(session)
    local user = seats(session)
    local over = duel.is_over(session.g)
    local view = {
        round = session.g.round,
        style = session.style,
        -- A session written before persona styles existed carries no
        -- kind, and it can only hold a canonical one: the fallback did
        -- not exist when it was dealt.
        style_kind = session.style_kind or "canonical",
        user_seat = session.user_seat,
        your_hand = copy_list(user.my_hand),
        your_points = user.my_points,
        npc_points = user.opp_points,
        npc_played = copy_list(user.opp_played),
        legal_actions = over and {} or duel.legal_actions(user),
        status = over and "finished" or "your_turn",
    }
    if over then
        view.winner = human_winner(session)
    end
    view.text = render_text(view)
    return view
end

-- ─── Actions ────────────────────────────────────────────────────────

--- Resolve the NPC style of a new game and say which kind it is.
---
--- A canonical style is decided by the `policy_<style>` field, exactly
--- as before. Only a name that has none reaches the alias lookup, so a
--- persona Card can never shadow a shipped style, and the extra lookup
--- costs nothing on the canonical path.
---@param raw any Requested style name, or nil for the default
---@return string style
---@return string kind `"canonical"` or `"persona"`
local function resolve_style(raw)
    local style = raw or DEFAULT_STYLE
    if type(style) == "string" then
        if type(duel["policy_" .. style]) == "function" then
            return style, "canonical"
        end
        local alias = ALIAS_PREFIX .. style
        local found, reason = alias_card_exists(alias)
        if found then
            return style, "persona"
        end
        error(
            string.format(
                "card_duel_interactive: unknown style %s (canonical: %s; persona alias %s: %s)",
                tostring(raw),
                table.concat(duel.STYLES, ", "),
                alias,
                reason or "no Card is pinned to it"
            )
        )
    end
    error(string.format("card_duel_interactive: style must be a string, got %s", type(style)))
end

local function resolve_seat(raw)
    local seat = math.floor(optional_number("user_seat", raw, 1))
    if seat ~= 1 and seat ~= 2 then
        error("card_duel_interactive: user_seat must be 1 or 2, got " .. tostring(raw))
    end
    return seat
end

local function action_new(ctx, key)
    local style, kind = resolve_style(ctx.style)
    local session = {
        g = duel.new_game(math.floor(optional_number("seed", ctx.seed, DEFAULT_SEED))),
        style = style,
        style_kind = kind,
        user_seat = resolve_seat(ctx.user_seat),
    }
    save_session(key, session)
    return board_view(session)
end

--- Validate the human move against the legal actions of its seat.
local function check_rank(user, raw)
    local rank = math.floor(tonumber(raw) or 0)
    local legal = duel.legal_actions(user)
    for _, candidate in ipairs(legal) do
        if candidate == rank then
            return rank
        end
    end
    error(
        string.format(
            "card_duel_interactive: rank %s is not legal here (legal: %s)",
            tostring(raw),
            join(legal)
        )
    )
end

--- Ask the NPC package for the move of the other seat.
local function npc_rank(session, npc_state)
    require_json()
    local npc = require("card_duel_npc")
    local out = npc.run({
        task = alc.json_encode({ mode = "decide", state = npc_state }),
        card_alias = ALIAS_PREFIX .. session.style,
    })
    local text = type(out) == "table" and out.result or out
    if type(text) ~= "string" then
        error("card_duel_interactive: NPC answer must be a string, got " .. type(text))
    end
    local rank = tonumber(text:match("action=(%d+)"))
    if rank == nil then
        error(string.format("card_duel_interactive: NPC answer %q carries no action", text))
    end
    return rank
end

local function action_play(ctx, key)
    local session = load_session(key)
    if duel.is_over(session.g) then
        error('card_duel_interactive: the game is over (call action="end" or start a new one)')
    end

    local user, opponent = seats(session)
    local round = session.g.round
    local your_rank = check_rank(user, ctx.rank)
    local their_rank = npc_rank(session, opponent)

    -- Logged before the round is applied, from the seat state the human
    -- was answering: after `apply` the hand is one card shorter and the
    -- opponent history one card longer, which is a position nobody was
    -- ever asked about.
    log_move(session, user, your_rank)

    -- `apply` takes the seat-one move first, so the pair is swapped
    -- when the human sits in seat two.
    if session.user_seat == 2 then
        session.g = duel.apply(session.g, their_rank, your_rank)
    else
        session.g = duel.apply(session.g, your_rank, their_rank)
    end
    save_session(key, session)

    local outcome = "a tie"
    if your_rank > their_rank then
        outcome = "you score"
    elseif their_rank > your_rank then
        outcome = "the npc scores"
    end

    local view = board_view(session)
    view.last_round = {
        round = round,
        you = your_rank,
        npc = their_rank,
        outcome = outcome,
    }
    view.text = string.format(
        "round %d: you played %d, the npc played %d, %s | %s",
        round,
        your_rank,
        their_rank,
        outcome,
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

---@param ctx table `{ action, rank?, style?, seed?, user_seat?, game_id? }`
---@return table view Board view for the resulting position
function M.run(ctx)
    ctx = ctx or {}
    local action = ctx.action
    if type(action) ~= "string" then
        error("card_duel_interactive: ctx.action must be a string, got " .. type(action))
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
                "card_duel_interactive: unknown action %q (expected new / play / show / end)",
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
