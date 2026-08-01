--- style_distance — mean JS divergence between two Cards' greedy action
--- distributions over a shared prompt set.
---
--- ## Contract
---
--- `style_distance(card_a, card_b, prompt_set) -> number`
---
--- - `card_a`, `card_b` — either a Lua string (Card alias resolved via
---   `alc.card.get_by_alias` → `alc.nn.card.load_handle`) or a handle
---   table with a `generate_session(ids)` method (the shape returned by
---   `alc.nn.card.load_handle`). The trainer `on_ckpt` hook currently
---   hands in a filesystem ckpt path string, which the caller must
---   resolve to an alias before calling — direct ckpt-path→handle load
---   is out of scope for this iter (see Follow-up in the ST4 report).
---
--- - `prompt_set` — a non-empty Lua array of guardian player-view tables
---   (`{ turn, mode, boss_hp, shift_distance, hp, weakened, exposed,
---   spikes, intent }`, as emitted by `guardian_duel.player_view`).
---   Callers building one from the bundled sample play log extract the
---   `.player` field of each entry:
---
---   ```lua
---   local raw = alc.json_decode(alc.fs_read(
---       "examples/gameai/data/guardian_sample_playlog_train.json"))
---   local prompt_set = {}
---   for _, entry in ipairs(raw) do
---       prompt_set[#prompt_set + 1] = entry.player
---   end
---   ```
---
--- ## Output
---
--- The scalar mean of `alc.nn.metric.js(p_a, p_b)` over every view in
--- `prompt_set`, where `p_a` / `p_b` are the softmax-over-legal-moves
--- probability rows read from each Card's next-token logits at the
--- position (temperature 1.0, so the softmax base matches the sampler's
--- default draw).
---
--- Zero when the two Cards agree on every position, close to `log(2)`
--- when they never overlap. The mean is preferred over a sum so a
--- 100-view sweep is comparable to a 32-view one.
---
--- ## Validation
---
--- Every argument shape mismatch is a loud error: the Registry-based
--- hook path is only useful if the number the trainer reads is trustable,
--- so a silent zero (empty prompt set / missing Card / bad view) would
--- undermine the whole design.

local duel = require("guardian_duel")

local M

--- The four player moves the softmax is masked to.
---
--- Fetched at load time rather than per call so a stub `guardian_duel`
--- that swaps the alphabet fails at require time rather than at the
--- first metric evaluation.
local VOCAB = duel.player_vocab()
local LEGAL_ACTIONS = duel.player_legal_actions()
local LEGAL_IDS = {}
for _, action in ipairs(LEGAL_ACTIONS) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error("style_distance: legal action " .. tostring(action) .. " is outside player_vocab")
    end
    LEGAL_IDS[#LEGAL_IDS + 1] = id
end

--- Assert `alc.nn.card` is available on this VM. A build without the
--- `nn` feature does not even parse a Card, so a metric that read from
--- an "empty" handle would report a silent zero for every position.
local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("style_distance: alc.nn.card is unavailable; build algocline with --features nn")
    end
end

--- Assert the shared `alc.nn.metric.js` primitive is present. Loaded on
--- every call rather than cached in a local so a bridge reinstall mid
--- run (the stubs the spec injects) is honoured.
local function require_js()
    if not (alc and alc.nn and alc.nn.metric and type(alc.nn.metric.js) == "function") then
        error("style_distance: alc.nn.metric.js is unavailable; ST2 bridge missing")
    end
    return alc.nn.metric.js
end

--- Resolve `card` (alias string or handle table) to a live handle.
local function resolve_handle(card, which)
    if type(card) == "table" then
        if type(card.generate_session) ~= "function" then
            error(
                string.format(
                    "style_distance: %s is a table but has no generate_session method; "
                        .. "expected a handle returned by alc.nn.card.load_handle",
                    which
                )
            )
        end
        return card
    end
    if type(card) == "string" then
        if #card == 0 then
            error(string.format("style_distance: %s must be a non-empty string alias", which))
        end
        require_nn_card()
        if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
            error("style_distance: alc.card.get_by_alias is unavailable")
        end
        local entry = alc.card.get_by_alias(card)
        if not entry then
            error(
                string.format("style_distance: %s alias %q is not bound to any Card", which, card)
            )
        end
        local card_id = entry.card_id
        if type(card_id) ~= "string" or #card_id == 0 then
            error(
                string.format(
                    "style_distance: %s alias %q resolved to a Card without card_id",
                    which,
                    card
                )
            )
        end
        return alc.nn.card.load_handle(card_id)
    end
    error(
        string.format(
            "style_distance: %s must be a string alias or a handle table, got %s",
            which,
            type(card)
        )
    )
end

--- Softmax-over-legal-moves probability row for one Card at one view.
---
--- Reads the model's `next_logits`, keeps only the four legal move ids,
--- and normalises with the numerically-stable softmax (subtract the max
--- before `exp`). Temperature is fixed at 1.0 so the base measurement
--- matches the sampler default; a caller who wants a temperature sweep
--- can compose a wrapper metric on top.
local function probs_for_view(handle, view)
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()
    local ranked = logits:top(logits:vocab())

    -- Extract logits for the four legal move ids, keyed by ordinal
    -- position in LEGAL_IDS (so the two Cards' rows align element by
    -- element for the JS call).
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
        error("style_distance: player move id missing from logits ranking")
    end

    -- Numerically stable softmax: subtract the max before exponentiating
    -- so a large positive logit does not overflow to inf.
    local max_logit = raw[1]
    for i = 2, #raw do
        if raw[i] > max_logit then
            max_logit = raw[i]
        end
    end
    local probs = {}
    local sum = 0.0
    for i, l in ipairs(raw) do
        local w = math.exp(l - max_logit)
        probs[i] = w
        sum = sum + w
    end
    if sum <= 0 then
        error("style_distance: softmax over legal moves normalised to zero")
    end
    for i, w in ipairs(probs) do
        probs[i] = w / sum
    end
    return probs
end

--- Validate a prompt_set is a non-empty array of view tables.
local function require_prompt_set(prompt_set)
    if type(prompt_set) ~= "table" then
        error(
            "style_distance: prompt_set must be a table (array of player views), got "
                .. type(prompt_set)
        )
    end
    local n = #prompt_set
    if n == 0 then
        error("style_distance: prompt_set is empty; nothing to average over")
    end
    return n
end

--- Compute mean JS divergence between two Cards over a prompt set.
M = function(card_a, card_b, prompt_set)
    local n = require_prompt_set(prompt_set)
    local js = require_js()
    local handle_a = resolve_handle(card_a, "card_a")
    local handle_b = resolve_handle(card_b, "card_b")

    local total = 0.0
    for i = 1, n do
        local view = prompt_set[i]
        if type(view) ~= "table" then
            error(
                string.format(
                    "style_distance: prompt_set[%d] must be a player view table, got %s",
                    i,
                    type(view)
                )
            )
        end
        local p = probs_for_view(handle_a, view)
        local q = probs_for_view(handle_b, view)
        total = total + js(p, q)
    end
    return total / n
end

return M
