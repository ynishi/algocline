--- boss_seat — shared boss-seat decode helpers for the gameai metrics.
---
--- Every metric in this package can be read from either seat. The player
--- seat is spelled out per file (`duel.player_encode(view) .. ">"` plus a
--- mask over the four always-legal player moves), which is cheap to
--- duplicate because the mask is a module-level constant. The boss seat
--- is not: the prompt is `duel.encode(state, style) .. ">"` — it takes a
--- style basis — and the legal set is state-dependent (`t` is only legal
--- in mode 1, so a state offers five or six moves). Duplicating that
--- across three files would give three chances to drift from the rules.
---
--- ## Why this file exists at all (per-file duplication is the norm here)
---
--- The rest of this package deliberately re-states its small helpers per
--- file rather than sharing them — see `trickiness.lua:102-103`, where
--- `resolve_handle` is copied "so a spec that stubs one metric does not
--- have to reach into the other's module state". This module is an
--- intentional exception to that convention, for two reasons:
---
--- 1. `guardian_duel_npc.decide` already implements exactly this rule
---    (`guardian_duel_npc/init.lua:269-290`: legal token ids, encode with
---    the style basis, first legal id in the full ranking), but it is
---    module-local — the package's public API is `run` / `reset_cache`
---    only. It cannot be required.
--- 2. Exporting it would widen the NPC package's public API for a
---    measurement-side convenience. The NPC contract should not grow
---    because the metrics wanted a shortcut, so the implementation lives
---    here instead.
---
--- The rule this file encodes is therefore a second implementation of the
--- decode contract on purpose; it is kept byte-compatible with
--- `guardian_duel_npc.decide` (same prompt, same mask, same first-legal
--- scan) and covered by `spec/boss_seat_spec.lua`.
---
--- ## API
---
--- - `require_seat(seat, who)` — decode the `"player"` / `"boss"` opt,
---   defaulting to `"player"` so an omitted `opts` reproduces the
---   pre-seat behaviour.
--- - `require_style(style, who)` — assert a `guardian_duel.STYLES` member.
--- - `require_state(state, label)` — assert a boss state, naming a player
---   view explicitly when one is passed by mistake (seat mismatch).
--- - `legal(state)` — `{ ids, actions, by_id }` for this state.
--- - `encode(state, style)` — the decode prompt, separator included.
--- - `probs(handle, state, style, temperature?)` — softmax row over the
---   legal moves, aligned with `legal(state).ids`, plus that legal set.
--- - `decide(handle, state, style)` — one legal-gated greedy move.

local duel = require("guardian_duel")

local M = {}

--- Boss alphabet, fetched once at load time so a stub `guardian_duel`
--- that swaps it fails at require time rather than at the first decode.
local VOCAB = duel.vocab()

--- Every boss-state field `guardian_duel.encode` reads.
local STATE_FIELDS = {
    "cycle",
    "mode",
    "hp",
    "damage_since_shift",
    "last_player",
    "turn",
    "shifts",
}

--- Fields only a player view carries (`guardian_duel.player_view`).
---
--- A boss state and a player view share `turn`, `mode` and `hp`, so the
--- mismatch has to be detected on the fields that do *not* overlap.
--- Without this check a player view fed to the boss seat would be
--- rejected far away, by `guardian_duel.encode` complaining about a
--- missing `cycle`, which reads as a malformed state rather than as the
--- seat mix-up it actually is.
local PLAYER_VIEW_FIELDS = {
    "boss_hp",
    "shift_distance",
    "weakened",
    "exposed",
    "spikes",
    "intent",
}

--- Decode the seat a metric is read from.
---
--- `nil` means `"player"`: every metric in this package predates the
--- boss seat, so an omitted opt has to reproduce the old behaviour
--- exactly. An unknown spelling is refused rather than folded into the
--- default, which would silently measure the wrong seat.
---@param seat any Requested seat
---@param who string Caller name, for the message
---@return string seat `"player"` or `"boss"`
function M.require_seat(seat, who)
    if seat == nil or seat == "player" then
        return "player"
    end
    if seat == "boss" then
        return "boss"
    end
    error(
        string.format(
            '%s: seat must be "player" or "boss", got %s',
            who,
            type(seat) == "string" and string.format("%q", seat) or type(seat)
        )
    )
end

--- Assert `style` names one of `guardian_duel.STYLES`.
---
--- The boss prompt measures its distance field against the style's
--- mode-shift threshold, so a metric read under a different basis than
--- the Card was baked under reads a distance the labels never followed.
--- There is no default: guessing one would answer a question the caller
--- did not ask.
---@param style any Requested style name
---@param who string Caller name, for the message
---@return string style
function M.require_style(style, who)
    if type(style) ~= "string" or #style == 0 then
        error(
            string.format(
                '%s: seat="boss" requires a style (one of %s), got %s',
                who,
                table.concat(duel.STYLES, ", "),
                type(style) == "string" and "an empty string" or type(style)
            )
        )
    end
    for _, name in ipairs(duel.STYLES) do
        if name == style then
            return style
        end
    end
    error(
        string.format(
            "%s: unknown style %q (valid: %s)",
            who,
            style,
            table.concat(duel.STYLES, ", ")
        )
    )
end

--- Assert `state` is a boss state, not a player view.
---@param state any Candidate boss state
---@param label string What to call it in the message (e.g. `level: state`)
---@return table state
function M.require_state(state, label)
    if type(state) ~= "table" then
        error(
            string.format(
                "%s must be a boss state table (guardian_duel.new_game(seed).boss), got %s",
                label,
                type(state)
            )
        )
    end
    for _, field in ipairs(PLAYER_VIEW_FIELDS) do
        if state[field] ~= nil then
            error(
                string.format(
                    "%s carries the player-view field %q, so it is a player view rather than a "
                        .. 'boss state; seat="boss" reads boss states (guardian_duel.new_game(seed).boss)',
                    label,
                    field
                )
            )
        end
    end
    for _, field in ipairs(STATE_FIELDS) do
        if type(state[field]) ~= "number" then
            error(
                string.format(
                    "%s is missing the boss-state field %q (got %s)",
                    label,
                    field,
                    type(state[field])
                )
            )
        end
    end
    return state
end

--- Legal boss moves for one state, as ids and as characters.
---
--- The order is `guardian_duel.legal_actions`'s, which is stable for a
--- given state, so two Cards read at the same state produce rows that
--- align element by element.
---@param state table Boss state
---@return table legal `{ ids, actions, by_id }`
function M.legal(state)
    local actions = duel.legal_actions(state)
    local ids, by_id = {}, {}
    for _, action in ipairs(actions) do
        local id = VOCAB.to_id[action]
        if id == nil then
            error("boss_seat: boss move " .. tostring(action) .. " is outside the boss vocab")
        end
        ids[#ids + 1] = id
        by_id[id] = action
    end
    if #ids == 0 then
        error("boss_seat: state has no legal boss move")
    end
    return { ids = ids, actions = actions, by_id = by_id }
end

--- Decode prompt for one boss state, separator included.
---@param state table Boss state
---@param style string Distance basis the state is encoded against
---@return string prompt
function M.encode(state, style)
    return duel.encode(state, style) .. ">"
end

--- Read the next-token logits of `handle` at one boss state.
local function ranked_logits(handle, state, style)
    local session = handle:generate_session(duel.to_ids(M.encode(state, style)))
    local logits = session:next_logits()
    return logits:top(logits:vocab())
end

--- Temperature-scaled softmax over the legal boss moves.
---
--- Numerically stable via max-subtraction before `exp`, matching the
--- player-seat rows in `style_distance.lua` / `trickiness.lua` so a
--- distance or an entropy is computed the same way on both seats.
---@param handle table|userdata Card handle with `generate_session`
---@param state table Boss state
---@param style string Distance basis
---@param temperature number|nil Positive finite scale, default 1.0
---@return number[] probs Aligned with `legal(state).ids`
---@return table legal The legal set the row was masked to
function M.probs(handle, state, style, temperature)
    local t = temperature
    if t == nil then
        t = 1.0
    end
    if type(t) ~= "number" or t ~= t or t == math.huge or t <= 0 then
        error("boss_seat: temperature must be a finite positive number, got " .. tostring(t))
    end

    local legal = M.legal(state)
    local ranked = ranked_logits(handle, state, style)

    local raw = {}
    local seen = 0
    for _, entry in ipairs(ranked) do
        for i, legal_id in ipairs(legal.ids) do
            if entry.id == legal_id and raw[i] == nil then
                raw[i] = entry.value
                seen = seen + 1
                break
            end
        end
        if seen == #legal.ids then
            break
        end
    end
    if seen ~= #legal.ids then
        error("boss_seat: legal boss move id missing from the logit ranking")
    end

    local scaled = {}
    for i, l in ipairs(raw) do
        scaled[i] = l / t
    end
    local max_l = scaled[1]
    for i = 2, #scaled do
        if scaled[i] > max_l then
            max_l = scaled[i]
        end
    end
    local probs = {}
    local sum = 0.0
    for i, l in ipairs(scaled) do
        local w = math.exp(l - max_l)
        probs[i] = w
        sum = sum + w
    end
    if sum <= 0 then
        error("boss_seat: softmax over legal boss moves normalised to zero")
    end
    for i, w in ipairs(probs) do
        probs[i] = w / sum
    end
    return probs, legal
end

--- One legal-gated greedy boss decision.
---
--- The argmax may be a field letter or an illegal move, so the scan takes
--- the first legal id in the full ranking rather than the top one — the
--- rule `guardian_duel_npc.decide` applies (see the header for why it is
--- restated here rather than required).
---@param handle table|userdata Card handle with `generate_session`
---@param state table Boss state
---@param style string Distance basis
---@return string action
function M.decide(handle, state, style)
    local legal = M.legal(state)
    for _, entry in ipairs(ranked_logits(handle, state, style)) do
        local action = legal.by_id[entry.id]
        if action ~= nil then
            return action
        end
    end
    -- Unreachable: `top(vocab)` enumerates the whole vocabulary and the
    -- legal set is non-empty. Kept loud in case a future ranking change
    -- starts truncating.
    error("boss_seat: no legal boss move found in the full logit ranking")
end

return M
