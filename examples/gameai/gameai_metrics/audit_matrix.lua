--- gameai_metrics.audit_matrix — per-Card baseline plus pair-wise
--- style-distance matrix runner for a boss-harvest Card collection.
---
--- ## What this is (and is not)
---
--- `harvest_collection` writes one entry per band during a training run
--- and pins the `card_id` of each baked Card into the resulting
--- manifest. `audit_matrix` reads that manifest (or an explicit alias
--- list), reloads every Card by `card_id`, and produces two views for a
--- human observer:
---
--- 1. **per_card baseline** — every Card is re-measured on the same
---    three metrics that fired during training (`level` / `trickiness`
---    / `sd_teacher`). Re-measuring at a higher `n_games` than the
---    mid-run gate could afford tightens the Wilson interval without
---    re-training. The `sd_teacher` view is only emitted when a
---    `teacher_alias` is supplied — the teacher pin is optional.
---
--- 2. **pair-wise SD matrix** — `style_distance` is evaluated on every
---    unordered pair of Cards. Jensen-Shannon divergence is symmetric
---    by construction, so the runner computes each pair only once
---    (`(i, j)` with `i < j`) and mirrors the value to `sd_matrix[j][i]`;
---    the diagonal is filled with `0.0` so a caller can look up any
---    pair without knowing the iteration order.
---
--- Nothing here is a decision: the runner never compares its numbers
--- against a target, never picks a winner. What comes out is the raw
--- material a human reads to answer two questions the audit is set up
--- to inform:
---
--- - Does the strong / mid / weak separation observed during training
---   reproduce once the per-Card metrics are re-measured at higher `n`?
--- - Are the three Cards' policies actually far enough apart (large
---   pair-wise SD) to be worth pitting against each other in a
---   Card-vs-Card fight iteration?
---
--- ## Why it lives in `gameai_metrics` (not `anymetric`)
---
--- The runner speaks in boss-harvest vocabulary (`alias`, `card_id`,
--- `teacher_alias`, "boss state" as the prompt-set element type). That
--- is gameai-specific by design — `anymetric` deliberately stays free
--- of Card / seat / style vocabulary so it can be promoted into a
--- bundled package later. Keeping the audit runner one shelf up on the
--- gameai side is the same call `harvest_collection` made for the same
--- reason.
---
--- ## Why a temperature is optional here
---
--- `fight_matrix` has no spelling for greedy: two baked Cards decoding
--- greedily replay a single game N times, so the win rate over that
--- batch is a rate over one sample. This runner measures each Card
--- against a *scripted* opponent (`policy_boss_random` /
--- `policy_player_random`), and those policies carry their own RNG. A
--- greedy Card therefore still meets a different opponent line every
--- game: the batch has real spread and its Wilson interval stays an
--- interval rather than collapsing to a point. The measured backing is
--- the greedy audit already on disk — its weak band landed on a win
--- rate of 0.075 with a Wilson interval of [0.046, 0.120] over 200
--- games, which is an interval and not a point.
---
--- So `opts.temperature` is optional here and an absent key means
--- greedy — the exact path every audit before it took, which is what
--- keeps a greedy run comparable against the ones already saved. A
--- caller who wants to read a Card on the same decode scale a fight
--- uses passes `1.0` explicitly; that is what lets an audit and a
--- fight differ in one variable at a time. The asymmetry with
--- `fight_matrix` is deliberate and runs the other way: there greedy
--- has no meaning, here it is a legitimate baseline.
---
--- A supplied value is still checked. Anything that is not a finite
--- positive number is a caller mistake rather than a request for
--- greedy, so it raises instead of quietly falling back.
---
--- ## The prompt set (and why its numbers do not compare backwards)
---
--- `trickiness` and `style_distance` are averages over the prompt set:
--- each Card is read at every state in it and the per-state values are
--- meaned. The set is therefore the measurement condition, not an
--- implementation detail — the same metric read over a different set is
--- a different quantity, and its values, its thresholds and its history
--- do not carry across.
---
--- That matters here because the set built by this runner changed. It
--- used to be `guardian_duel.new_game(seed + i).boss` repeated
--- `prompt_set_size` times, and a fight opening is the same board for
--- every seed (`new_game` varies nothing but the recorded seed), so the
--- set was one state written down N times: the average had one term and
--- the entropy / divergence axes read a single opening rather than a
--- policy. Every audit written before this change measured that. Its
--- `trickiness_norm` / `sd_teacher` / `sd_matrix` numbers are readings
--- of another quantity and **must not** be compared against the ones
--- this runner produces now — not as a delta, not as a threshold, not
--- as a trend.
---
--- What replaces it is a rollout sampler:
---
--- - **Both seats move at random**, from `alc.math.rng_create` seeded
---   off the audit `seed`. A random walk reaches mode / cycle / damage
---   combinations a scripted line never enters, which is the coverage
---   an entropy axis wants.
--- - **The RNG stream is namespaced** (`seed * 7919 + game`) so it
---   never coincides with the stream the `level` view opens on the same
---   `seed` (`alc.math.rng_create(seed)`).
--- - **The set is stratified by boss mode**: half the states come from
---   mode 0 (the style walking its cycle), half from mode 1 (rolled up,
---   the slam pending), odd sizes giving the extra slot to mode 0. The
---   two modes have systematically different answer entropy, so an
---   unstratified "first N distinct states" set would let the mode mix
---   drift with the seed — and a set whose composition moves with the
---   seed is, again, a different quantity per run. Fixing the mix is
---   what makes two runs of this audit comparable to each other.
--- - **One rollout contributes at most four states**, so a set is drawn
---   from several independent trajectories rather than from one.
--- - **States are de-duplicated on `guardian_duel.encode(state, style)`
---   under the audited `style`** — the same basis the measurement reads
---   them on. The encoded distance field is measured against the
---   style's own shift threshold, so two states that are distinct
---   prompts under one style can be the same prompt under another;
---   keying on anything but the style being audited would let duplicate
---   prompts into the average.
--- - **The search stops after 200 rollouts** and raises, naming which
---   stratum came up short. A set that could not be filled is a loud
---   error rather than a short average.
---
--- Every state in the set is one the engine produced — `new_game`
--- followed by `guardian_duel.apply` — so it carries the bookkeeping
--- (`block` / `thorns`) and the field consistency a hand-written table
--- can silently get wrong.
---
--- One state is in every set the sampler builds regardless of seed: the
--- opening `C0M0H9D3L0T1`, because every fight starts there. Two sets
--- drawn under different seeds therefore agree on that element and are
--- expected to differ on the rest.
---
--- The set the run actually used is recorded on the report:
--- `meta.prompt_set_source` is `"rollout_v1"` for a sampled set and
--- `"caller"` for one handed in through `opts.prompt_set`, and a
--- sampled set adds `meta.prompt_set_composition` with the measured
--- `mode0_count` / `mode1_count` / `games_consumed` / `distinct`. Two
--- saved audits are comparable only when both carry the same source and
--- the same composition.
---
--- ## Contract
---
--- ```lua
--- local am = require("gameai_metrics.audit_matrix")
---
--- local audit = am.new({
---     collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
---     -- or, without a manifest:
---     -- aliases = { "guardian_duel_npc_weak", "_mid", "_strong" },
---     n_games = 200,           -- per-Card baseline sample size (default)
---     prompt_set_size = 16,    -- boss states for trickiness / SD (default)
---     seed = 20260731,
---     style = "guardian",      -- required, one of guardian_duel.STYLES
---     teacher_alias = "guardian_duel_npc",  -- optional
---     temperature = 1.0,       -- optional; omitted = greedy
--- })
---
--- local report = audit:run()
--- audit:save("workspace/gameai-harvest/audit_run2.json")
--- ```
---
--- The report shape is
---
--- ```
--- {
---     per_card = {
---         [alias] = {
---             win_rate      = 0.42,
---             ci_lower      = 0.30,
---             ci_upper      = 0.55,
---             trickiness_norm = 0.41,   -- boss-seat normalised entropy
---             sd_teacher    = 0.089,    -- absent when teacher_alias is nil
---             step          = 180,      -- inherited from the harvest entry
---         },
---         ...
---     },
---     sd_matrix = {
---         [alias_i] = { [alias_j] = <number>, ... },   -- symmetric, diag 0
---         ...
---     },
---     meta = { n_games, prompt_set_size, prompt_set_source,
---              prompt_set_composition, seed, style, teacher_alias,
---              temperature, ... },   -- temperature absent when greedy
---                                    -- composition absent for a
---                                    -- caller-supplied prompt set
--- }
--- ```
---
--- `save()` writes the same object as JSON via `alc.json_encode`, after
--- calling `gameai_metrics._fs.ensure_parent_dir(path)` so a fresh
--- `workspace/` sub-tree is created on the first run rather than
--- surfacing as an obscure `io.open` `No such file or directory` (the
--- same failure mode `harvest_collection` prevented at ST-0).

-- The package init is what builds the ctx adapters this file measures
-- through (`gm.metrics.style_distance`). Requiring the submodule (this
-- file) directly does not execute the package's init.lua, so the require
-- is explicit. gameai_metrics.init does not require this submodule back,
-- so there is no cycle.
local gm = require("gameai_metrics")

local am = require("anymetric")
local duel = require("guardian_duel")
local boss_seat = require("gameai_metrics.boss_seat")
local fs = require("gameai_metrics._fs")

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics.audit_matrix",
    version = "0.1.0",
    description = "Per-Card baseline + pair-wise SD matrix runner for a boss-harvest Card collection.",
    category = "game",
}

local DEFAULT_N_GAMES = 200
local DEFAULT_PROMPT_SET_SIZE = 16
local DEFAULT_SEED = 0

--- `meta.prompt_set_source` tag of a set this runner sampled. The
--- version suffix is part of the tag on purpose: a later sampler that
--- changes the collection rule produces a different measurement
--- condition, and an artifact has to say which one it was read under
--- rather than leave two incomparable runs looking alike.
local PROMPT_SET_SOURCE_ROLLOUT = "rollout_v1"

--- `meta.prompt_set_source` tag of a set the caller handed in. The
--- runner knows nothing about how it was drawn, so it reports the fact
--- rather than a composition it did not measure.
local PROMPT_SET_SOURCE_CALLER = "caller"

--- Stride between the audit seed and the rollout RNG stream, so the
--- states are sampled from a different stream than the one the `level`
--- view opens on the same seed (`alc.math.rng_create(seed)`). The
--- constant is the one `guardian_duel.build_corpus` /
--- `guardian_duel.sample_states` already namespace their playout
--- streams with.
local ROLLOUT_RNG_STRIDE = 7919

--- Rollouts the sampler may spend before it gives up. A set of the
--- default size fills in a handful of games, so this is a backstop
--- against a stratum that never appears rather than a working budget.
local ROLLOUT_GAME_CAP = 200

--- States one rollout may contribute. A fight lasts at most
--- `guardian_duel.TURN_LIMIT` turns and its states are strongly
--- correlated, so a cap spreads a set across independent trajectories
--- instead of letting the first two fights fill it.
local ROLLOUT_STATES_PER_GAME = 4

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "audit_matrix: alc.json_encode is required to write the manifest (host bridge missing)",
            0
        )
    end
    return alc.json_encode
end

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "audit_matrix: alc.json_decode is required to read the collection manifest "
                .. "(host bridge missing)",
            0
        )
    end
    return alc.json_decode
end

local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("audit_matrix: alc.nn.card is unavailable; build algocline with --features nn")
    end
    if type(alc.nn.card.load_handle) ~= "function" then
        error("audit_matrix: alc.nn.card.load_handle is unavailable")
    end
end

--- The `style_distance` adapter this file measures pairs with.
---
--- Resolved through a function rather than captured at load time so a
--- spec that swaps `gm.metrics` after requiring this module still sees
--- its own stub, and so a missing adapter names itself here instead of
--- surfacing as an "attempt to call a nil value" inside the matrix loop.
local function style_distance_metric()
    local fn = gm.metrics and gm.metrics.style_distance
    if type(fn) ~= "function" then
        error(
            "audit_matrix: gameai_metrics.metrics.style_distance is not a function "
                .. "(build without the nn feature? a stale gameai_metrics on the path?)",
            0
        )
    end
    return fn
end

--- The RNG bridge both seats of a sampling rollout draw from.
---
--- `guardian_duel.policy_player_random` / `policy_boss_random` reach
--- for the same two functions, so checking them here names the missing
--- bridge at the layer the caller called rather than three frames down
--- inside the rules package.
local function require_rng_bridge()
    if
        type(alc) ~= "table"
        or type(alc.math) ~= "table"
        or type(alc.math.rng_create) ~= "function"
        or type(alc.math.rng_int) ~= "function"
    then
        error(
            "audit_matrix: alc.math.rng_create / alc.math.rng_int are required to sample the "
                .. "prompt set (host bridge missing); pass opts.prompt_set to supply one instead",
            0
        )
    end
    return alc.math
end

local function require_card_alias_bridge()
    if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
        error(
            "audit_matrix: alc.card.get_by_alias is unavailable "
                .. "(needed to resolve an alias to a card_id when opts.aliases is set)"
        )
    end
end

-- ─── Option validation ─────────────────────────────────────────────

local function decode_int(raw, default, field, must_be_positive)
    if raw == nil then
        return default
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge then
        error(
            string.format("audit_matrix: %s must be a finite integer, got %s", field, tostring(raw)),
            3
        )
    end
    local i = math.floor(raw)
    if must_be_positive and i <= 0 then
        error(
            string.format(
                "audit_matrix: %s must be a positive integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    if not must_be_positive and i < 0 then
        error(
            string.format(
                "audit_matrix: %s must be a non-negative integer, got %s",
                field,
                tostring(raw)
            ),
            3
        )
    end
    return i
end

--- Decode the optional audit temperature.
---
--- An absent key returns `nil` and means greedy (see the header: an
--- audit's opponent carries its own RNG, so a greedy measurement is
--- still a measurement with spread). `nil` travels on rather than
--- turning into a default, so the level view config stays byte-identical
--- to the pre-temperature audit and `level` takes its greedy path.
---
--- A key that *is* present is held to the same shape `fight_matrix`
--- demands: finite, numeric and positive. Refusing it here rather than
--- inside `level` names the layer the caller called.
local function decode_temperature(raw)
    if raw == nil then
        return nil
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error(
            string.format(
                "audit_matrix: opts.temperature must be a finite positive number "
                    .. "(omit the key for a greedy audit), got %s",
                tostring(raw)
            ),
            3
        )
    end
    return raw
end

--- Sample a prompt set of boss states from random self-play.
---
--- See the header (`## The prompt set`) for why the set is built this
--- way and why its numbers do not compare against the ones the
--- pre-rollout runner produced. The rule, in one place:
---
--- - `size` is split into `ceil(size/2)` mode-0 states and
---   `floor(size/2)` mode-1 states, so the mode mix is fixed by the
---   requested size rather than by the seed;
--- - each rollout plays both seats at random off
---   `alc.math.rng_create(seed * ROLLOUT_RNG_STRIDE + game)` and may
---   contribute at most `ROLLOUT_STATES_PER_GAME` states;
--- - a state joins its stratum only when `duel.encode(state, style)`
---   — the prompt the measurement will read it as — has not been taken
---   yet;
--- - the walk stops once both strata are full, and raises after
---   `ROLLOUT_GAME_CAP` rollouts naming the stratum that came up short.
---
--- The whole walk is a function of `(seed, size, style)`, so two runs
--- with the same three arguments produce the same set.
---
--- The caller-supplied `opts.prompt_set` overrides this entirely (so an
--- audit can reuse the exact `check_states` a training run measured
--- against).
---@param seed integer Audit seed
---@param size integer States to collect
---@param style string Audited style; the basis states are keyed on
---@return table[] prompt_set `size` boss states, mode-0 stratum first
---@return table composition `{ mode0_count, mode1_count, games_consumed, distinct }`
local function build_prompt_set(seed, size, style)
    local math_ns = require_rng_bridge()

    local targets = {}
    targets[1] = math.floor(size / 2)
    targets[0] = size - targets[1]
    local buckets = { [0] = {}, [1] = {} }

    local taken_prompts = {}
    local games = 0
    while games < ROLLOUT_GAME_CAP and (#buckets[0] < targets[0] or #buckets[1] < targets[1]) do
        games = games + 1
        local g = duel.new_game(seed + games)
        local rng = math_ns.rng_create(seed * ROLLOUT_RNG_STRIDE + games)
        local adopted = 0
        while not duel.is_over(g) and adopted < ROLLOUT_STATES_PER_GAME do
            local state = g.boss
            local bucket = buckets[state.mode]
            if #bucket < targets[state.mode] then
                local prompt = duel.encode(state, style)
                if taken_prompts[prompt] == nil then
                    taken_prompts[prompt] = true
                    bucket[#bucket + 1] = state
                    adopted = adopted + 1
                end
            end
            g = duel.apply(g, duel.policy_player_random(rng), duel.policy_boss_random(state, rng))
        end
    end

    if #buckets[0] < targets[0] or #buckets[1] < targets[1] then
        error(
            string.format(
                "audit_matrix: the prompt-set sampler spent its %d-rollout budget without "
                    .. "filling both strata (mode 0: %d/%d, mode 1: %d/%d) for style %q at "
                    .. "seed %d; a short set would silently average over fewer states, so this "
                    .. "raises instead",
                ROLLOUT_GAME_CAP,
                #buckets[0],
                targets[0],
                #buckets[1],
                targets[1],
                style,
                seed
            ),
            3
        )
    end

    local out = {}
    for mode = 0, 1 do
        for _, state in ipairs(buckets[mode]) do
            out[#out + 1] = state
        end
    end

    -- `distinct` is counted back off the finished set rather than
    -- restated from the de-duplication above: the composition is a
    -- measurement of what the run will average over, and a measurement
    -- that only echoes the code that produced it cannot catch that code
    -- going wrong.
    local seen, distinct = {}, 0
    for _, state in ipairs(out) do
        local prompt = duel.encode(state, style)
        if seen[prompt] == nil then
            seen[prompt] = true
            distinct = distinct + 1
        end
    end

    local composition = {
        mode0_count = #buckets[0],
        mode1_count = #buckets[1],
        games_consumed = games,
        distinct = distinct,
    }
    return out, composition
end

--- Normalise `opts.aliases` into a `{alias, card_id}` pair list.
--- Each `card_id` is resolved lazily through `alc.card.get_by_alias`;
--- an alias without a bound Card is a loud error (same treatment the
--- collection-path branch gives to a missing `card_id`).
local function resolve_alias_list(aliases)
    if type(aliases) ~= "table" or #aliases == 0 then
        error("audit_matrix: opts.aliases must be a non-empty array of alias strings", 3)
    end
    require_card_alias_bridge()
    local seen = {}
    local out = {}
    for i, alias in ipairs(aliases) do
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format(
                    "audit_matrix: opts.aliases[%d] must be a non-empty string, got %s",
                    i,
                    type(alias)
                ),
                3
            )
        end
        if seen[alias] then
            error(string.format("audit_matrix: opts.aliases lists %q twice", alias), 3)
        end
        seen[alias] = true
        local entry = alc.card.get_by_alias(alias)
        if type(entry) ~= "table" or type(entry.card_id) ~= "string" or entry.card_id == "" then
            error(
                string.format(
                    "audit_matrix: alias %q is not bound to a Card with a card_id "
                        .. "(alc.card.get_by_alias returned no card_id)",
                    alias
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = entry.card_id, step = nil }
    end
    return out
end

--- Read a harvest manifest and turn its `entries` array into the same
--- `{alias, card_id, step}` shape the aliases path produces. Missing
--- `card_id` on any entry is a loud error (a partial audit that
--- silently skipped Cards would produce a matrix whose axes did not
--- match the caller's expectation).
local function resolve_from_collection(path)
    if type(path) ~= "string" or path == "" then
        error("audit_matrix: opts.collection_path must be a non-empty string", 3)
    end
    local decode = require_json_decoder()
    local f, open_err = io.open(path, "r")
    if f == nil then
        error(
            string.format(
                "audit_matrix: cannot open collection_path %q: %s",
                path,
                tostring(open_err)
            ),
            3
        )
    end
    local body = f:read("a")
    f:close()
    local ok, parsed = pcall(decode, body)
    if not ok then
        error(
            string.format(
                "audit_matrix: failed to decode collection_path %q: %s",
                path,
                tostring(parsed)
            ),
            3
        )
    end
    if type(parsed) ~= "table" or type(parsed.entries) ~= "table" then
        error(
            string.format(
                "audit_matrix: collection_path %q has no entries array (schema drift?)",
                path
            ),
            3
        )
    end
    if #parsed.entries == 0 then
        error(
            string.format(
                "audit_matrix: collection_path %q has zero entries; nothing to audit",
                path
            ),
            3
        )
    end
    local seen = {}
    local out = {}
    for i, entry in ipairs(parsed.entries) do
        if type(entry) ~= "table" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] must be a table, got %s",
                    i,
                    type(entry)
                ),
                3
            )
        end
        local alias = entry.alias
        if type(alias) ~= "string" or alias == "" then
            error(
                string.format("audit_matrix: collection_path entries[%d] has no alias string", i),
                3
            )
        end
        if seen[alias] then
            error(
                string.format("audit_matrix: collection_path entries list alias %q twice", alias),
                3
            )
        end
        seen[alias] = true
        local card_id = entry.card_id
        if type(card_id) ~= "string" or card_id == "" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] (alias %q) has no card_id; "
                        .. "the audit cannot reload a Card without its stored card_id",
                    i,
                    alias
                ),
                3
            )
        end
        local step = entry.step
        if step ~= nil and type(step) ~= "number" then
            error(
                string.format(
                    "audit_matrix: collection_path entries[%d] (alias %q) has non-numeric step %s",
                    i,
                    alias,
                    tostring(step)
                ),
                3
            )
        end
        out[i] = { alias = alias, card_id = card_id, step = step }
    end
    return out
end

-- ─── Views ─────────────────────────────────────────────────────────

--- Assemble the per-Card views. `n_games` / `seed` are pushed into
--- the `level` view config so `gm.metrics.level` picks them up
--- per-fire; the trickiness / style_distance views need neither.
--- The `sd_teacher` view is only added when a `teacher_alias` was
--- supplied; without one the view would have no reference Card to
--- measure against.
---
--- `temperature` reaches the `level` view only. `trickiness` reads its
--- own `ctx.temperature or 1.0` in its ctx adapter, so putting a
--- key here would move the entropy axis off the scale every earlier
--- audit measured it on; `style_distance` compares distributions the
--- same way for the same reason. A `nil` temperature adds no key at
--- all, which is what keeps the greedy report byte-identical.
local function build_views(style, prompt_set, teacher_alias, n_games, seed, temperature)
    local level_config = {
        seat = "boss",
        opponents = { "random" },
        style = style,
        n_games = n_games,
        seed = seed,
    }
    if temperature ~= nil then
        level_config.temperature = temperature
    end
    local views = {
        am.view("level", gm.metrics.level, level_config),
        am.view("trickiness", gm.metrics.trickiness, {
            seat = "boss",
            style = style,
            prompt_set = prompt_set,
        }),
    }
    if teacher_alias ~= nil then
        views[#views + 1] = am.view("sd_teacher", gm.metrics.style_distance, {
            seat = "boss",
            style = style,
            prompt_set = prompt_set,
            card_b = teacher_alias,
        })
    end
    return views
end

--- Read a numeric field off a Record.values table. Returns `nil` when
--- the record is missing, errored, or the field is not a number — the
--- runner writes each field as absent on the report entry in that
--- case rather than emitting a spurious zero.
local function numeric_field(record, field)
    if type(record) ~= "table" or record.error ~= nil then
        return nil
    end
    local values = record.values
    if type(values) ~= "table" then
        return nil
    end
    local raw = values[field]
    if type(raw) ~= "number" then
        return nil
    end
    return raw
end

local function find_record(records, view_id)
    for _, record in ipairs(records) do
        if type(record) == "table" and rawget(record, "view_id") == view_id then
            return record
        end
    end
    return nil
end

--- Extract the baseline fields the audit report cares about, in the
--- order the caller expects to read them. Any missing / errored field
--- stays absent on the entry — a measurement gap is not a labelled
--- outcome.
local function extract_baseline(records, teacher_alias)
    local level_rec = find_record(records, "level")
    local tricky_rec = find_record(records, "trickiness")
    local sd_rec = find_record(records, "sd_teacher")

    local entry = {}
    local win_rate = numeric_field(level_rec, "win_rate")
    if win_rate ~= nil then
        entry.win_rate = win_rate
    end
    local ci_lower = numeric_field(level_rec, "ci_lower")
    if ci_lower ~= nil then
        entry.ci_lower = ci_lower
    end
    local ci_upper = numeric_field(level_rec, "ci_upper")
    if ci_upper ~= nil then
        entry.ci_upper = ci_upper
    end
    local tricky_val = numeric_field(tricky_rec, "value")
    if tricky_val ~= nil then
        entry.trickiness_norm = tricky_val
    end
    -- Only attach sd_teacher if the caller pinned a teacher_alias;
    -- otherwise the view was never built and the record is absent by
    -- design (rather than a measurement failure).
    if teacher_alias ~= nil then
        local sd_val = numeric_field(sd_rec, "value")
        if sd_val ~= nil then
            entry.sd_teacher = sd_val
        end
    end
    return entry
end

-- ─── SD matrix ─────────────────────────────────────────────────────

--- Evaluate `style_distance` for the ordered pair `(a, b)`. Any raise
--- from the metric (a mask mismatch, a bad prompt state) propagates
--- to the caller so the audit fails loudly rather than silently
--- publishing a matrix with a zero where a broken metric hid a real
--- distance.
local function evaluate_pair(metric, handle_a, handle_b, style, prompt_set)
    local raw = metric({
        card_a = handle_a,
        card_b = handle_b,
        prompt_set = prompt_set,
        seat = "boss",
        style = style,
    })
    -- `style_distance` returns a scalar; the compose loop in
    -- gameai_metrics/init.lua does not lift it. Accept both shapes so
    -- a spec that mimics the anymetric lift form still works.
    if type(raw) == "number" then
        return raw
    end
    if type(raw) == "table" and type(raw.value) == "number" then
        return raw.value
    end
    error(
        "audit_matrix: style_distance returned "
            .. type(raw)
            .. "; expected a number or a { value = <n> } table"
    )
end

--- Fill in `sd_matrix[alias_i][alias_j]` for every unordered pair
--- exactly once (`i < j`), then mirror the value onto `[alias_j][alias_i]`.
--- The diagonal is filled with `0.0` so a downstream lookup does not
--- have to special-case the identity pair.
local function build_sd_matrix(cards, style, prompt_set)
    local metric = style_distance_metric()
    local matrix = {}
    for _, card in ipairs(cards) do
        matrix[card.alias] = { [card.alias] = 0.0 }
    end
    for i = 1, #cards - 1 do
        for j = i + 1, #cards do
            local a, b = cards[i], cards[j]
            local sd = evaluate_pair(metric, a.handle, b.handle, style, prompt_set)
            matrix[a.alias][b.alias] = sd
            matrix[b.alias][a.alias] = sd
        end
    end
    return matrix
end

-- ─── Runner ────────────────────────────────────────────────────────

local Audit = {}
Audit.__index = Audit

--- Execute the audit. Loads every Card handle by `card_id`, fires the
--- per-Card views once per Card, then evaluates the pair-wise SD
--- matrix. Returns the report table and caches it on the Audit so
--- `:save()` and `:report()` can read the same object without a
--- second run.
---@return table report
function Audit:run()
    require_nn_card()

    -- Reload every Card first. Doing this up-front means a missing
    -- card_id fails before any metric runs, so an audit either
    -- completes for every Card or fails cleanly with the offending
    -- alias named.
    local cards = {}
    for _, entry in ipairs(self._cards) do
        local ok, handle = pcall(alc.nn.card.load_handle, entry.card_id)
        if not ok then
            error(
                string.format(
                    "audit_matrix: alc.nn.card.load_handle(%q) failed for alias %q: %s",
                    entry.card_id,
                    entry.alias,
                    tostring(handle)
                )
            )
        end
        if handle == nil then
            error(
                string.format(
                    "audit_matrix: alc.nn.card.load_handle(%q) returned nil for alias %q "
                        .. "(card_id likely removed from the Card store)",
                    entry.card_id,
                    entry.alias
                )
            )
        end
        cards[#cards + 1] = {
            alias = entry.alias,
            card_id = entry.card_id,
            step = entry.step,
            handle = handle,
        }
    end

    local views = build_views(
        self._style,
        self._prompt_set,
        self._teacher_alias,
        self._n_games,
        self._seed,
        self._temperature
    )

    local per_card = {}
    for _, card in ipairs(cards) do
        -- observe requires a numeric step; when the collection carried
        -- one we pass it through (the "this Card was baked at step N"
        -- label) and default to 0 otherwise so the aliases-only path
        -- still produces observable records.
        local shared_step = card.step
        if type(shared_step) ~= "number" then
            shared_step = 0
        end
        local records = am.observe(views, { card = card.handle, step = shared_step })
        local baseline = extract_baseline(records, self._teacher_alias)
        if card.step ~= nil then
            baseline.step = card.step
        end
        per_card[card.alias] = baseline
    end

    local sd_matrix = build_sd_matrix(cards, self._style, self._prompt_set)

    local meta = {
        n_games = self._n_games,
        prompt_set_size = #self._prompt_set,
        -- Which measurement condition this report was read under. Two
        -- audits are comparable only when both the source and (for a
        -- sampled set) the composition agree; see the header.
        prompt_set_source = self._prompt_set_source,
        seed = self._seed,
        style = self._style,
    }
    local composition = self._prompt_set_composition
    if composition ~= nil then
        -- Copied out so the saved report cannot be mutated through the
        -- runner (and the runner cannot be mutated through the report).
        meta.prompt_set_composition = {
            mode0_count = composition.mode0_count,
            mode1_count = composition.mode1_count,
            games_consumed = composition.games_consumed,
            distinct = composition.distinct,
        }
    end
    if self._teacher_alias ~= nil then
        meta.teacher_alias = self._teacher_alias
    end
    -- Absent means greedy. Recording a `1.0` (or a literal "greedy")
    -- for a run that never asked for one would make the two decodes
    -- indistinguishable in the saved JSON, which is the one thing a
    -- decode-effect comparison reads the meta for.
    if self._temperature ~= nil then
        meta.temperature = self._temperature
    end
    if self._collection_path ~= nil then
        meta.collection_path = self._collection_path
    end

    local report = {
        per_card = per_card,
        sd_matrix = sd_matrix,
        meta = meta,
    }
    self._report = report
    return report
end

--- Read-only accessor for the last `:run()` report. Raises when
--- called before `:run()` — a save() without a report would write an
--- empty manifest that looked like a successful audit.
---@return table report
function Audit:report()
    if self._report == nil then
        error("audit_matrix:report: no report yet; call :run() first", 2)
    end
    return self._report
end

--- Encode the report and write it to `path`. Ensures the parent
--- directory exists first (via `gameai_metrics._fs`), so a first save
--- into a fresh workspace sub-tree does not surface as an obscure
--- `io.open` error.
---@param path string
function Audit:save(path)
    if type(path) ~= "string" or path == "" then
        error("audit_matrix:save: path must be a non-empty string", 2)
    end
    if self._report == nil then
        error("audit_matrix:save: no report yet; call :run() before :save()", 2)
    end
    fs.ensure_parent_dir(path)
    local encode = require_json_encoder()
    local ok, encoded = pcall(encode, self._report)
    if not ok then
        error("audit_matrix:save: failed to encode report: " .. tostring(encoded), 2)
    end
    local f, err = io.open(path, "w")
    if f == nil then
        error(
            string.format("audit_matrix:save: cannot open %q for writing: %s", path, tostring(err)),
            2
        )
    end
    local ok_write, write_err = pcall(function()
        f:write(encoded)
    end)
    f:close()
    if not ok_write then
        error(
            string.format("audit_matrix:save: write to %q failed: %s", path, tostring(write_err)),
            2
        )
    end
end

--- Build a new audit runner. `opts` is a table:
---
--- - `collection_path` — string, harvest manifest to read (produced
---   by `gameai_metrics.harvest_collection:save()`). One of
---   `collection_path` / `aliases` is required; passing both is a
---   loud error.
--- - `aliases` — array of alias strings. Each alias is resolved
---   through `alc.card.get_by_alias`, so an alias without a bound
---   Card raises immediately.
--- - `n_games` — integer, sample size for the `level` view (default
---   `200`). Recorded on `meta.n_games` for the report; the actual
---   `level` metric reads it out of its own ctx, which is supplied
---   per-fire below via the level view config.
--- - `prompt_set_size` — integer, size of the boss-state prompt set
---   used by the `trickiness` and `style_distance` views (default
---   `16`). Half of it is drawn from mode 0 and half from mode 1 (an
---   odd size gives the extra slot to mode 0). Ignored when
---   `opts.prompt_set` is supplied.
--- - `prompt_set` — optional array of boss states. When absent the
---   runner samples one from random self-play, deterministically from
---   `seed` (see the header), so an audit remains reproducible without
---   a manifest-side prompt set. A supplied set is used verbatim and
---   reported as `meta.prompt_set_source = "caller"`; the runner
---   neither de-duplicates nor stratifies it, because a caller passing
---   `check_states` means *those* states.
--- - `seed` — integer, seeds the prompt-set rollouts, the `level` view
---   and the pair-wise SD evaluation (default `0`). The rollout stream
---   is namespaced away from the `level` one, so the two never draw the
---   same numbers.
--- - `style` — required, one of `guardian_duel.STYLES`. Passed
---   verbatim into every boss-seat view.
--- - `teacher_alias` — optional string; when set, the runner adds an
---   `sd_teacher` view and reports `sd_teacher` per Card.
--- - `temperature` — optional finite positive number, pushed into the
---   `level` view only. Absent means greedy (see the header for why
---   that is a legitimate baseline here and not on `fight_matrix`);
---   present and non-positive / non-finite raises. Recorded on
---   `meta.temperature` only when it was supplied.
---
---@param opts table
---@return table audit
function M.new(opts)
    if type(opts) ~= "table" then
        error("audit_matrix.new: opts must be a table, got " .. type(opts), 2)
    end

    if opts.collection_path ~= nil and opts.aliases ~= nil then
        error("audit_matrix.new: pass either opts.collection_path or opts.aliases, not both", 2)
    end
    if opts.collection_path == nil and opts.aliases == nil then
        error("audit_matrix.new: opts.collection_path or opts.aliases is required", 2)
    end

    local style = boss_seat.require_style(opts.style, "audit_matrix.new")

    if opts.teacher_alias ~= nil then
        if type(opts.teacher_alias) ~= "string" or opts.teacher_alias == "" then
            error(
                "audit_matrix.new: opts.teacher_alias must be a non-empty string or nil, got "
                    .. type(opts.teacher_alias),
                2
            )
        end
    end

    local n_games = decode_int(opts.n_games, DEFAULT_N_GAMES, "n_games", true)
    local prompt_set_size =
        decode_int(opts.prompt_set_size, DEFAULT_PROMPT_SET_SIZE, "prompt_set_size", true)
    local seed = decode_int(opts.seed, DEFAULT_SEED, "seed", false)
    local temperature = decode_temperature(opts.temperature)

    local prompt_set, prompt_set_source, prompt_set_composition
    if opts.prompt_set ~= nil then
        if type(opts.prompt_set) ~= "table" or #opts.prompt_set == 0 then
            error("audit_matrix.new: opts.prompt_set must be a non-empty array of boss states", 2)
        end
        prompt_set = opts.prompt_set
        prompt_set_source = PROMPT_SET_SOURCE_CALLER
    else
        prompt_set, prompt_set_composition = build_prompt_set(seed, prompt_set_size, style)
        prompt_set_source = PROMPT_SET_SOURCE_ROLLOUT
    end

    local cards
    if opts.collection_path ~= nil then
        cards = resolve_from_collection(opts.collection_path)
    else
        cards = resolve_alias_list(opts.aliases)
    end
    if #cards < 2 then
        error(
            string.format(
                "audit_matrix.new: at least two Cards are needed for a pair-wise SD matrix, got %d",
                #cards
            ),
            2
        )
    end

    return setmetatable({
        _cards = cards,
        _style = style,
        _teacher_alias = opts.teacher_alias,
        _n_games = n_games,
        _prompt_set_size = #prompt_set,
        _seed = seed,
        _temperature = temperature,
        _prompt_set = prompt_set,
        _prompt_set_source = prompt_set_source,
        _prompt_set_composition = prompt_set_composition,
        _collection_path = opts.collection_path,
        _report = nil,
    }, Audit)
end

--- Read-only accessors, mostly for specs / driver logs.
M.DEFAULT_N_GAMES = DEFAULT_N_GAMES
M.DEFAULT_PROMPT_SET_SIZE = DEFAULT_PROMPT_SET_SIZE
M.DEFAULT_SEED = DEFAULT_SEED
M.PROMPT_SET_SOURCE_ROLLOUT = PROMPT_SET_SOURCE_ROLLOUT
M.PROMPT_SET_SOURCE_CALLER = PROMPT_SET_SOURCE_CALLER
M.ROLLOUT_GAME_CAP = ROLLOUT_GAME_CAP
M.ROLLOUT_STATES_PER_GAME = ROLLOUT_STATES_PER_GAME

--- The prompt-set sampler, exposed for the spec that pins its
--- collection rule (`spec/audit_matrix_spec.lua`). It is reachable
--- through `M.new` for every other purpose; the underscore says the
--- name is not part of the runner's contract, the way
--- `card_duel_tournament._aggregate` does for the same reason.
M._build_prompt_set = build_prompt_set

return M
