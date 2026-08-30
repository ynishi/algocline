--- trickiness — mean Shannon entropy of one Card's temperature-scaled
--- action distribution over a shared prompt set.
---
--- ## Contract
---
--- `trickiness(card, prompt_set, temperature?, opts?) -> number|table`
---
--- - `card` — string alias, or a live handle (table or userdata) with a
---   `generate_session(ids)` method, same shape as `style_distance`
---   (see `style_distance.lua`, including the `on_ckpt` +
---   `alc.nn.card.load_ckpt` usage).
---
--- - `prompt_set` — non-empty array of positions; the element type
---   follows the seat exactly as in `style_distance.lua` (player views on
---   the player seat, boss states on the boss seat, mixing them a loud
---   error).
---
--- - `temperature` — optional positive finite number, default `1.0`. The
---   distribution the entropy is measured over is the temperature-scaled
---   softmax of the Card's next-token logits, masked to the legal moves
---   of the seat. The scaling matches what
---   `alc.nn.sampler.temperature(t, seed)` would draw from, so a
---   `trickiness` reading at `t = 1.5` predicts the entropy of the
---   noisy autoplay at the same temperature.
---
--- - `opts` — optional table, every field additive (omitting `opts`
---   reproduces the pre-seat behaviour byte for byte):
---   - `seat` — `"player"` (default) or `"boss"`.
---   - `style` — one of `guardian_duel.STYLES`, required when
---     `seat = "boss"` (the boss prompt is measured against the style's
---     mode-shift threshold).
---
--- ## Output
---
--- - `seat = "player"` — the scalar mean of `alc.math.entropy(p)`
---   (natural log, base e) across every view in `prompt_set`. Zero when
---   the Card is fully committed to one move at every position
---   (equivalent to greedy), approaches `log(4)` when the four legal
---   moves are uniformly distributed. The player mask is four moves at
---   every position, so the raw entropies are already comparable across
---   positions.
---
--- - `seat = "boss"` — a table `{ value, raw_mean }`, where `value` is
---   the mean **normalised** entropy `H / log(#legal)` and `raw_mean` is
---   the mean raw `H`. The normalisation exists because the boss legal
---   set is state-dependent: the twin slam `t` is only legal in mode 1
---   (`guardian_duel/init.lua:644-651`), so a state offers five or six
---   moves and the raw ceiling moves with it (`log 5` vs `log 6`).
---   Averaging raw entropies across a mixed prompt set would read a
---   change in the mode distribution as a change in the policy.
---   `value` is in `[0, 1]` at every position, so a mixed set averages
---   into a comparable number; `raw_mean` is kept alongside so the nats
---   are not lost.
---
--- ## Why measure it from the logits, not by sampling
---
--- Sampling `N` draws per position and estimating the empirical
--- distribution would need `N` in the hundreds to stabilise the entropy
--- of a peaky distribution — the sampler is what the entropy is *of*, so
--- reading it from the softmax that feeds the sampler is cheaper and
--- deterministic per (card, view, temperature) triple.

local duel = require("guardian_duel")
local boss_seat = require("gameai_metrics.boss_seat")

local M

local VOCAB = duel.player_vocab()
local LEGAL_ACTIONS = duel.player_legal_actions()
local LEGAL_IDS = {}
for _, action in ipairs(LEGAL_ACTIONS) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error("trickiness: legal action " .. tostring(action) .. " is outside player_vocab")
    end
    LEGAL_IDS[#LEGAL_IDS + 1] = id
end

local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("trickiness: alc.nn.card is unavailable; build algocline with --features nn")
    end
end

--- Shannon entropy is general-purpose mathematics, so it is read from
--- `alc.math` (mlua-mathlib) rather than from `alc.nn.metric`. See
--- `style_distance.require_js` for the same reasoning.
local function require_entropy()
    if not (alc and alc.math and type(alc.math.entropy) == "function") then
        error("trickiness: alc.math.entropy is unavailable; mathlib not registered")
    end
    return alc.math.entropy
end

--- Match `style_distance.has_generate_session`; see that file for why
--- the method rather than the Lua type is what the guard reads, and why
--- the read is wrapped in `pcall`.
local function has_generate_session(card)
    local ok, method = pcall(function()
        return card.generate_session
    end)
    return ok and type(method) == "function"
end

--- Match `style_distance.resolve_handle`; kept local so a spec that stubs
--- one metric does not have to reach into the other's module state.
local function resolve_handle(card)
    local kind = type(card)
    if kind == "table" or kind == "userdata" then
        if not has_generate_session(card) then
            error(
                string.format(
                    "trickiness: card is a %s but has no generate_session method; "
                        .. "expected a handle returned by alc.nn.card.load_handle "
                        .. "or alc.nn.card.load_ckpt",
                    kind
                )
            )
        end
        return card
    end
    if type(card) == "string" then
        if #card == 0 then
            error("trickiness: card must be a non-empty string alias")
        end
        require_nn_card()
        if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
            error("trickiness: alc.card.get_by_alias is unavailable")
        end
        local entry = alc.card.get_by_alias(card)
        if not entry then
            error(string.format("trickiness: alias %q is not bound to any Card", card))
        end
        local card_id = entry.card_id
        if type(card_id) ~= "string" or #card_id == 0 then
            error(string.format("trickiness: alias %q resolved to a Card without card_id", card))
        end
        return alc.nn.card.load_handle(card_id)
    end
    error(
        "trickiness: card must be a string alias or a handle (table or userdata), got "
            .. type(card)
    )
end

--- Check `temperature` is finite and strictly positive. Zero is rejected
--- rather than folded into greedy: the sampler would divide by it, and
--- a caller who means greedy already gets it by omitting the argument
--- and reading `trickiness ≈ 0` from a peaked softmax.
local function decode_temperature(raw)
    if raw == nil then
        return 1.0
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error("trickiness: temperature must be a finite positive number, got " .. tostring(raw))
    end
    return raw
end

--- Temperature-scaled softmax over the four legal moves. Numerically
--- stable via max-subtraction before `exp`.
local function probs_for_view(handle, view, temperature)
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()
    local ranked = logits:top(logits:vocab())

    local raw = {}
    for i = 1, #LEGAL_IDS do
        raw[i] = nil
    end
    local seen = 0
    for _, entry in ipairs(ranked) do
        for i, legal_id in ipairs(LEGAL_IDS) do
            if entry.id == legal_id and raw[i] == nil then
                raw[i] = entry.value
                seen = seen + 1
                break
            end
        end
        if seen == #LEGAL_IDS then
            break
        end
    end
    if seen ~= #LEGAL_IDS then
        error("trickiness: player move id missing from logits ranking")
    end

    local scaled = {}
    for i, l in ipairs(raw) do
        scaled[i] = l / temperature
    end
    -- Temperature-scaled above; `alc.math.softmax` does the
    -- max-subtraction the hand-rolled loop used to do here.
    return alc.math.softmax(scaled)
end

local function require_prompt_set(prompt_set)
    if type(prompt_set) ~= "table" then
        error(
            "trickiness: prompt_set must be a table (array of player views), got "
                .. type(prompt_set)
        )
    end
    local n = #prompt_set
    if n == 0 then
        error("trickiness: prompt_set is empty; nothing to average over")
    end
    return n
end

--- Mean raw entropy over player views (the pre-seat path).
local function player_mean(handle, prompt_set, n, t, entropy)
    local total = 0.0
    for i = 1, n do
        local view = prompt_set[i]
        if type(view) ~= "table" then
            error(
                string.format(
                    "trickiness: prompt_set[%d] must be a player view table, got %s",
                    i,
                    type(view)
                )
            )
        end
        local p = probs_for_view(handle, view, t)
        total = total + entropy(p)
    end
    return total / n
end

--- Mean normalised and raw entropy over boss states.
---
--- Each position is divided by its own `log(#legal)` before averaging,
--- so a prompt set that mixes mode-0 (five moves) and mode-1 (six moves)
--- states averages comparable quantities.
local function boss_mean(handle, prompt_set, n, t, entropy, style)
    local normalised, raw = 0.0, 0.0
    for i = 1, n do
        local state = prompt_set[i]
        boss_seat.require_state(state, string.format("trickiness: prompt_set[%d]", i))
        local p, legal = boss_seat.probs(handle, state, style, t)
        local ceiling = math.log(#legal.ids)
        if ceiling <= 0 then
            error(
                string.format(
                    "trickiness: prompt_set[%d] offers %d legal boss move(s); "
                        .. "a normalised entropy needs at least two",
                    i,
                    #legal.ids
                )
            )
        end
        local h = entropy(p)
        raw = raw + h
        normalised = normalised + h / ceiling
    end
    return { value = normalised / n, raw_mean = raw / n }
end

M = function(card, prompt_set, temperature, opts)
    if opts ~= nil and type(opts) ~= "table" then
        error("trickiness: opts must be a table, got " .. type(opts))
    end
    opts = opts or {}
    local seat = boss_seat.require_seat(opts.seat, "trickiness")
    local style
    if seat == "boss" then
        style = boss_seat.require_style(opts.style, "trickiness")
    end

    local n = require_prompt_set(prompt_set)
    local t = decode_temperature(temperature)
    local entropy = require_entropy()
    local handle = resolve_handle(card)

    if seat == "boss" then
        return boss_mean(handle, prompt_set, n, t, entropy, style)
    end
    return player_mean(handle, prompt_set, n, t, entropy)
end

return M
