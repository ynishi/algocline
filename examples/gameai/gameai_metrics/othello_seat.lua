--- othello_seat — shared decode helpers for the 6x6 Othello seat.
---
--- Othello has one seat. Both sides are played from the same line by the
--- same model — `othello6.build_corpus` labels every ply of a game with
--- the one policy it was handed — so there is no player / boss split to
--- carry here and no distance basis to encode against: the prompt is
--- `othello6.encode(state)` and nothing more.
---
--- ## Why this file exists
---
--- `othello6_npc.decide` already implements the decode contract
--- (`othello6_npc/init.lua:414-435`: legal token ids, encode, first
--- legal id in the full ranking), but it is module-local — the package's
--- public API is `run` / `reset_cache` only — so it cannot be required.
--- Exporting it would widen the NPC package's public API for a
--- measurement-side convenience, so the rule is implemented a second
--- time here on purpose, kept compatible with the NPC (same prompt, same
--- legal set, same first-legal scan) and covered by
--- `spec/othello_seat_spec.lua`.
---
--- This is the Othello sibling of `boss_seat.lua`, which does the same
--- job for `guardian_duel`. Two contracts differ deliberately:
---
--- - **No `style` / basis argument.** A `guardian_duel` boss prompt
---   measures a distance against the style's threshold, so its encode
---   takes a basis. An Othello position encodes to the move sequence
---   that reached it, which no style can change, so a basis argument
---   would be an argument with nothing to do.
--- - **`legal(state)` is the move-character array**, not
---   `{ ids, actions, by_id }`. `othello6.legal_actions` already answers
---   in characters and the token id of a character is one `vocab().to_id`
---   lookup, so the three-field table would be three ways of saying the
---   same list. `probs` still returns `probs, legal` — the two-value
---   shape `boss_seat.probs` has — with the row aligned element by
---   element with that array.
---
--- ## API
---
--- - `legal(state)` — legal move characters, in `legal_actions` order.
--- - `encode(state)` — the decode prompt.
--- - `probs(handle, state, temperature?)` — softmax row over the legal
---   moves, aligned with `legal(state)`, plus that legal set.
--- - `decide(handle, state)` — one legal-gated greedy move.

local othello = require("othello6")

local M = {}

--- Move alphabet, fetched once at load time so a stub `othello6` that
--- swaps it fails at require time rather than at the first decode.
local VOCAB = othello.vocab()

--- Legal move characters for one position, in `legal_actions` order.
---
--- The order is stable for a given position, so two Cards read at the
--- same position produce rows that align element by element.
---
--- A position with no placement answers `{ othello6.PASS }` rather than
--- an empty list; an empty answer would be a rules problem and is named
--- as one here rather than surfacing as an empty softmax further down.
---@param state table Position
---@return string[] actions
function M.legal(state)
    local actions = othello.legal_actions(state)
    if #actions == 0 then
        error(
            "othello_seat: othello6.legal_actions answered an empty set; a position with no "
                .. "placement must answer the pass"
        )
    end
    for _, action in ipairs(actions) do
        if VOCAB.to_id[action] == nil then
            error("othello_seat: move " .. tostring(action) .. " is outside the move vocab")
        end
    end
    return actions
end

--- Decode prompt for one position.
---
--- `othello6.encode` writes the whole prompt — the opening marker and
--- one character per move played — and the model answers the next
--- character directly, so there is no separator to append (contrast
--- `boss_seat.encode`, which closes a field line with `>`).
---@param state table Position
---@return string prompt
function M.encode(state)
    return othello.encode(state)
end

--- Read the next-token logits of `handle` at one position.
local function ranked_logits(handle, state)
    local session = handle:generate_session(othello.to_ids(M.encode(state)))
    local logits = session:next_logits()
    return logits:top(logits:vocab())
end

--- Temperature-scaled softmax over the legal moves.
---
--- The mass outside the legal set is not redistributed, it is never
--- read: the row is built from the legal logits alone and normalised
--- over them, which is the same restriction `decide` applies and the
--- same one the noisy decode applies through
--- `alc.nn.constraint.allow_list`. Numerically stable via
--- max-subtraction before `exp`, matching `boss_seat.probs` so a
--- distance or an entropy is computed the same way on both games.
---@param handle table|userdata Card handle with `generate_session`
---@param state table Position
---@param temperature number|nil Positive finite scale, default 1.0
---@return number[] probs Aligned with `legal(state)`
---@return string[] legal The legal set the row was masked to
function M.probs(handle, state, temperature)
    local t = temperature
    if t == nil then
        t = 1.0
    end
    if type(t) ~= "number" or t ~= t or t == math.huge or t <= 0 then
        error("othello_seat: temperature must be a finite positive number, got " .. tostring(t))
    end

    local legal = M.legal(state)
    local ranked = ranked_logits(handle, state)

    local raw = {}
    local seen = 0
    for _, entry in ipairs(ranked) do
        for i, action in ipairs(legal) do
            if entry.id == VOCAB.to_id[action] and raw[i] == nil then
                raw[i] = entry.value
                seen = seen + 1
                break
            end
        end
        if seen == #legal then
            break
        end
    end
    if seen ~= #legal then
        error("othello_seat: legal move id missing from the logit ranking")
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
        error("othello_seat: softmax over legal moves normalised to zero")
    end
    for i, w in ipairs(probs) do
        probs[i] = w / sum
    end
    return probs, legal
end

--- One legal-gated greedy decision.
---
--- The argmax may be a board character or an illegal placement, so the
--- scan takes the first legal id in the full ranking rather than the top
--- one — the rule `othello6_npc.decide` applies (see the header for why
--- it is restated here rather than required). Reading the first legal
--- entry of a descending ranking is the argmax over the legal set.
---@param handle table|userdata Card handle with `generate_session`
---@param state table Position
---@return string action
function M.decide(handle, state)
    local by_id = {}
    for _, action in ipairs(M.legal(state)) do
        by_id[VOCAB.to_id[action]] = action
    end
    for _, entry in ipairs(ranked_logits(handle, state)) do
        local action = by_id[entry.id]
        if action ~= nil then
            return action
        end
    end
    -- Unreachable: `top(vocab)` enumerates the whole vocabulary and the
    -- legal set is non-empty. Kept loud in case a future ranking change
    -- starts truncating.
    error("othello_seat: no legal move found in the full logit ranking")
end

return M
