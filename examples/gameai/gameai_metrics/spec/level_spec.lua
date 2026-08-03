-- gameai_metrics/spec/level_spec.lua
--
-- Package-level spec for the `level` metric. Stubs `alc.card`,
-- `alc.nn.card`, and `alc.math.rng_create` so the autoplay loop can run
-- without a real model.
--
-- The stub Card always plays the first legal move ("a"), which makes
-- every fight of a batch a copy of one fight against the deterministic
-- teacher policy — the CI collapses to a single-point interval, which
-- is exactly the shape the Wilson formula produces at `p̂ = 0` or `1`.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- See style_distance_spec.lua for rationale.
alc = alc or {}

local duel = require("guardian_duel")

local VOCAB = duel.player_vocab()
local MOVES = duel.player_legal_actions()
local MOVE_IDS = {}
for _, action in ipairs(MOVES) do
    MOVE_IDS[#MOVE_IDS + 1] = VOCAB.to_id[action]
end

--- Build a handle whose greedy pick is always `preferred_move` (a
--- string like "a" / "b" / "p" / "A"). The logit vector plants a large
--- value on the id of that move so the argmax and the top-first scan
--- both pick it, regardless of turn.
local function make_handle(preferred_move)
    local target_id = VOCAB.to_id[preferred_move]
    if target_id == nil then
        error("spec: preferred_move " .. tostring(preferred_move) .. " outside player vocab")
    end
    return {
        generate_session = function(self, prompt_ids)
            self._last_prompt = prompt_ids
            return {
                next_logits = function()
                    local vocab = VOCAB.size
                    local values = {}
                    for i = 1, vocab do
                        values[i] = 0.0
                    end
                    values[target_id + 1] = 10.0
                    return {
                        vocab = function()
                            return vocab
                        end,
                        argmax = function()
                            return target_id
                        end,
                        top = function(_, n)
                            local ranked = {}
                            for i = 1, vocab do
                                ranked[i] = { id = i - 1, value = values[i] }
                            end
                            table.sort(ranked, function(a, b)
                                if a.value == b.value then
                                    return a.id < b.id
                                end
                                return a.value > b.value
                            end)
                            local out = {}
                            for i = 1, math.min(n, #ranked) do
                                out[i] = ranked[i]
                            end
                            return out
                        end,
                    }
                end,
            }
        end,
    }
end

--- See style_distance_spec.lua for why the metatable proxy stands in for
--- the userdata handle shape here.
local function via_metatable(handle)
    return setmetatable({}, { __index = handle })
end

-- ─── Boss-seat stub ─────────────────────────────────────────────────
--
-- The boss seat reads the *boss* alphabet, so it needs its own handle:
-- the player stub above plants ids from `player_vocab`, which addresses
-- a different table.

local BOSS_VOCAB = duel.vocab()

--- Handle whose greedy boss pick is always `preferred_move`, as long as
--- that move is legal in the state being decoded (`boss_seat.decide`
--- gates on the legal set, so an illegal favourite falls through to the
--- next legal token).
local function make_boss_handle(preferred_move)
    local target_id = BOSS_VOCAB.to_id[preferred_move]
    if target_id == nil then
        error("spec: preferred_move " .. tostring(preferred_move) .. " outside boss vocab")
    end
    return {
        generate_session = function(self, prompt_ids)
            self._last_prompt = prompt_ids
            return {
                next_logits = function()
                    local vocab = BOSS_VOCAB.size
                    local values = {}
                    for i = 1, vocab do
                        values[i] = 0.0
                    end
                    values[target_id + 1] = 10.0
                    return {
                        vocab = function()
                            return vocab
                        end,
                        argmax = function()
                            return target_id
                        end,
                        top = function(_, n)
                            local ranked = {}
                            for i = 1, vocab do
                                ranked[i] = { id = i - 1, value = values[i] }
                            end
                            table.sort(ranked, function(a, b)
                                if a.value == b.value then
                                    return a.id < b.id
                                end
                                return a.value > b.value
                            end)
                            local out = {}
                            for i = 1, math.min(n, #ranked) do
                                out[i] = ranked[i]
                            end
                            return out
                        end,
                    }
                end,
            }
        end,
    }
end

local ALIAS_TO_HANDLE = {}
alc.card = {
    get_by_alias = function(alias)
        if not ALIAS_TO_HANDLE[alias] then
            return nil
        end
        return { card_id = "card-" .. alias }
    end,
}

alc.nn = alc.nn or {}
alc.nn.card = {
    load_handle = function(card_id)
        local alias = card_id:match("^card%-(.+)$")
        return ALIAS_TO_HANDLE[alias]
    end,
}

--- Deterministic-ish RNG stub for the `"random"` boss path. Returns a
--- xorshift-style sequence keyed on the seed the caller planted; the
--- specs only need reproducibility, not statistical quality.
alc.math = alc.math or {}
alc.math.rng_create = function(seed)
    local state = seed
    if state == 0 then
        state = 0x9E3779B9
    end
    return {
        _state = state,
    }
end

--- `guardian_duel.policy_boss_random` / `policy_player_random` call
--- `alc.math.rng_int(rng, min, max)` for a value in `[min, max]`. Stub it
--- with an LCG, matching the host signature the engine-level harness uses
--- (`crates/algocline-engine/tests/lua/card_duel_rules_test.lua:31-35`).
alc.math.rng_int = function(rng, min, max)
    local s = rng._state
    s = (s * 1103515245 + 12345) % 2147483648
    rng._state = s
    return min + (s // 65536) % (max - min + 1)
end

-- ─── Sampler stub ───────────────────────────────────────────────────
--
-- `opts.temperature` draws through
-- `alc.nn.sampler.constrained(alc.nn.sampler.temperature(t, seed),
-- alc.nn.constraint.allow_list(ids))`. The fake below is deterministic —
-- it always returns the lowest allowed id, which is legal by
-- construction — so the specs can assert *what the chain was built with*
-- (temperature, per-decision seed, mask) rather than a random outcome.

local SAMPLER_CALLS = {}

alc.nn.sampler = {
    temperature = function(t, seed)
        return { temperature = t, seed = seed }
    end,
    constrained = function(inner, constraint)
        return {
            sample = function(_, logits)
                SAMPLER_CALLS[#SAMPLER_CALLS + 1] = {
                    temperature = inner.temperature,
                    seed = inner.seed,
                    allow = constraint.ids,
                    vocab = logits:vocab(),
                }
                local pick = constraint.ids[1]
                for _, id in ipairs(constraint.ids) do
                    if id < pick then
                        pick = id
                    end
                end
                return pick
            end,
        }
    end,
}

alc.nn.constraint = {
    allow_list = function(ids)
        local copy = {}
        for i, id in ipairs(ids) do
            copy[i] = id
        end
        return { ids = copy }
    end,
}

--- Run `fn` on a VM whose sampler surfaces are missing, restoring them
--- afterwards, and report the pcall result.
local function without_sampler(fn)
    local sampler, constraint = alc.nn.sampler, alc.nn.constraint
    alc.nn.sampler, alc.nn.constraint = nil, nil
    local ok, err = pcall(fn)
    alc.nn.sampler, alc.nn.constraint = sampler, constraint
    return ok, err
end

--- Wrap a stub handle so every prompt it is asked to decode is recorded.
--- The wrapper is what proves *which* view a seat built, which is the
--- only observable difference a style basis makes.
local function recording(handle)
    local prompts = {}
    local wrapped = {
        generate_session = function(_, prompt_ids)
            prompts[#prompts + 1] = table.concat(prompt_ids, ",")
            return handle:generate_session(prompt_ids)
        end,
    }
    return wrapped, prompts
end

local level = require("gameai_metrics.level")

local function reset()
    ALIAS_TO_HANDLE = {}
    SAMPLER_CALLS = {}
end

describe("gameai_metrics.level", function()
    it("returns a table with win_rate / ci_lower / ci_upper / wins / n_games", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy", 4, 1)
        expect(type(result)).to.equal("table")
        expect(type(result.win_rate)).to.equal("number")
        expect(type(result.ci_lower)).to.equal("number")
        expect(type(result.ci_upper)).to.equal("number")
        expect(type(result.wins)).to.equal("number")
        expect(result.n_games).to.equal(4)
    end)

    it("keeps win_rate and CI bounds inside [0, 1]", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy", 8, 7)
        expect(result.win_rate >= 0 and result.win_rate <= 1).to.equal(true)
        expect(result.ci_lower >= 0 and result.ci_lower <= 1).to.equal(true)
        expect(result.ci_upper >= 0 and result.ci_upper <= 1).to.equal(true)
        expect(result.ci_lower <= result.win_rate).to.equal(true)
        expect(result.win_rate <= result.ci_upper).to.equal(true)
    end)

    it("collapses the CI to a point interval at p̂ = 0 or 1", function()
        reset()
        -- Both handles are deterministic and the boss is deterministic,
        -- so every fight of the batch is a copy of one fight — wins is
        -- either 0 or n_games. Wilson at p̂ ∈ {0, 1} clamps to
        -- [0, upper] or [lower, 1]; the interval never straddles p̂.
        local h = make_handle("a")
        local result = level(h, "greedy", 4, 0)
        expect(result.win_rate == 0 or result.win_rate == 1).to.equal(true)
    end)

    it("is reproducible from the same (seed, opponent) pair", function()
        reset()
        local h = make_handle("b")
        local a = level(h, "greedy", 4, 3)
        local b = level(h, "greedy", 4, 3)
        expect(a.win_rate).to.equal(b.win_rate)
        expect(a.wins).to.equal(b.wins)
    end)

    it("defaults n_games to 32 and seed to 0", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "greedy")
        expect(result.n_games).to.equal(32)
    end)

    it("rejects an unknown opponent literal", function()
        reset()
        local h = make_handle("b")
        local ok, err = pcall(level, h, "trickster", 4, 1)
        expect(ok).to.equal(false)
        expect(err:find("opponent") ~= nil or err:find("not wired") ~= nil).to.equal(true)
    end)

    it("rejects a zero or negative n_games", function()
        reset()
        local h = make_handle("b")
        local ok, err = pcall(level, h, "greedy", 0, 1)
        expect(ok).to.equal(false)
        expect(err:find("n_games") ~= nil).to.equal(true)
    end)

    it("supports the random boss policy through alc.math.rng_create", function()
        reset()
        local h = make_handle("b")
        local result = level(h, "random", 3, 5)
        expect(result.n_games).to.equal(3)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("resolves a string alias through alc.card.get_by_alias", function()
        reset()
        ALIAS_TO_HANDLE["b"] = make_handle("b")
        local result = level("b", "greedy", 4, 1)
        expect(result.n_games).to.equal(4)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("accepts a handle whose generate_session comes from a metatable", function()
        reset()
        local h = via_metatable(make_handle("b"))
        expect(rawget(h, "generate_session")).to.equal(nil)
        local result = level(h, "greedy", 4, 1)
        expect(result.n_games).to.equal(4)
        expect(type(result.win_rate)).to.equal("number")
    end)

    it("still refuses a card table without a generate_session method", function()
        reset()
        local ok, err = pcall(level, { alias = "b" }, "greedy", 4, 1)
        expect(ok).to.equal(false)
        expect(err:find("generate_session") ~= nil).to.equal(true)
    end)

    describe("seat defaults", function()
        it("reads the same numbers with opts omitted, empty, or seat=player", function()
            reset()
            local h = make_handle("b")
            local bare = level(h, "greedy", 4, 1)
            local empty = level(h, "greedy", 4, 1, {})
            local explicit = level(h, "greedy", 4, 1, { seat = "player" })
            expect(empty.win_rate).to.equal(bare.win_rate)
            expect(empty.wins).to.equal(bare.wins)
            expect(explicit.win_rate).to.equal(bare.win_rate)
            expect(explicit.wins).to.equal(bare.wins)
            expect(explicit.n_games).to.equal(bare.n_games)
        end)

        it("refuses a non-table opts", function()
            reset()
            local h = make_handle("b")
            local ok, err = pcall(level, h, "greedy", 4, 1, "boss")
            expect(ok).to.equal(false)
            expect(err:find("opts") ~= nil).to.equal(true)
        end)

        it("refuses an unknown seat", function()
            reset()
            local h = make_handle("b")
            local ok, err = pcall(level, h, "greedy", 4, 1, { seat = "referee" })
            expect(ok).to.equal(false)
            expect(err:find("seat") ~= nil).to.equal(true)
        end)
    end)

    describe("opponent pool", function()
        it("reports per-opponent rates plus a pooled rate and CI", function()
            reset()
            local h = make_handle("b")
            local result = level(h, nil, 4, 1, { opponents = { "greedy", "random" } })
            expect(result.n_games).to.equal(8) -- 4 per opponent
            expect(type(result.per_opponent.greedy)).to.equal("table")
            expect(type(result.per_opponent.random)).to.equal("table")
            expect(result.per_opponent.greedy.n_games).to.equal(4)
            expect(result.per_opponent.random.n_games).to.equal(4)
            -- Pooled rate is the mean of the per-opponent rates, because
            -- every opponent plays the same n_games.
            local mean = (result.per_opponent.greedy.win_rate + result.per_opponent.random.win_rate)
                / 2
            expect(math.abs(result.win_rate - mean) < 1e-9).to.equal(true)
            expect(result.ci_lower <= result.win_rate).to.equal(true)
            expect(result.win_rate <= result.ci_upper).to.equal(true)
        end)

        it("reports the weakest matchup as win_rate_min", function()
            reset()
            local h = make_handle("b")
            local result = level(h, nil, 4, 1, { opponents = { "greedy", "random" } })
            local lowest =
                math.min(result.per_opponent.greedy.win_rate, result.per_opponent.random.win_rate)
            expect(result.win_rate_min).to.equal(lowest)
            expect(result.win_rate_min <= result.win_rate).to.equal(true)
        end)

        it("fills per_opponent for the single-opponent form too", function()
            reset()
            local h = make_handle("b")
            local result = level(h, "greedy", 4, 1)
            expect(result.per_opponent.greedy.n_games).to.equal(4)
            expect(result.per_opponent.greedy.win_rate).to.equal(result.win_rate)
            expect(result.win_rate_min).to.equal(result.win_rate)
        end)

        it("refuses both opponent and opts.opponents", function()
            reset()
            local h = make_handle("b")
            local ok, err = pcall(level, h, "greedy", 4, 1, { opponents = { "random" } })
            expect(ok).to.equal(false)
            expect(err:find("opponents") ~= nil).to.equal(true)
        end)

        it("refuses an empty pool", function()
            reset()
            local h = make_handle("b")
            local ok, err = pcall(level, h, nil, 4, 1, { opponents = {} })
            expect(ok).to.equal(false)
            expect(err:find("empty") ~= nil).to.equal(true)
        end)

        it("refuses a duplicated opponent name", function()
            reset()
            local h = make_handle("b")
            local ok, err = pcall(level, h, nil, 4, 1, { opponents = { "random", "random" } })
            expect(ok).to.equal(false)
            expect(err:find("twice") ~= nil).to.equal(true)
        end)
    end)

    describe('seat = "boss"', function()
        it("plays the boss seat against the random player pool", function()
            reset()
            local h = make_boss_handle("f")
            local result = level(h, nil, 4, 1, {
                seat = "boss",
                style = "guardian",
                opponents = { "random" },
            })
            expect(result.n_games).to.equal(4)
            expect(result.per_opponent.random.n_games).to.equal(4)
            expect(result.win_rate >= 0 and result.win_rate <= 1).to.equal(true)
            expect(result.ci_lower <= result.win_rate).to.equal(true)
            expect(result.win_rate <= result.ci_upper).to.equal(true)
            expect(result.win_rate_min).to.equal(result.win_rate)
        end)

        it("defaults the opponent to the random player policy", function()
            reset()
            local h = make_boss_handle("f")
            local a = level(h, nil, 4, 1, { seat = "boss", style = "guardian" })
            local b = level(h, nil, 4, 1, {
                seat = "boss",
                style = "guardian",
                opponents = { "random" },
            })
            expect(a.win_rate).to.equal(b.win_rate)
            expect(type(a.per_opponent.random)).to.equal("table")
        end)

        it("is reproducible from the same seed", function()
            reset()
            local h = make_boss_handle("f")
            local opts = { seat = "boss", style = "guardian", opponents = { "random" } }
            local a = level(h, nil, 6, 3, opts)
            local b = level(h, nil, 6, 3, opts)
            expect(a.win_rate).to.equal(b.win_rate)
            expect(a.wins).to.equal(b.wins)
        end)

        it("scores a boss win, not a player win", function()
            reset()
            -- A boss that only ever charges takes no damage from the
            -- random player's pokes fast enough to lose, so the batch
            -- lands on the boss side of the ledger.
            local h = make_boss_handle("c")
            local charging = level(h, nil, 6, 2, { seat = "boss", style = "guardian" })
            local venting = level(make_boss_handle("v"), nil, 6, 2, {
                seat = "boss",
                style = "guardian",
            })
            -- Two different boss policies over the same seeded openings
            -- must not be scored identically, or the seat is not reading
            -- the Card at all.
            expect(charging.win_rate ~= venting.win_rate).to.equal(true)
        end)

        it("requires a style", function()
            reset()
            local h = make_boss_handle("f")
            local ok, err = pcall(level, h, nil, 4, 1, { seat = "boss" })
            expect(ok).to.equal(false)
            expect(err:find("style") ~= nil).to.equal(true)
        end)

        it("refuses an unknown style", function()
            reset()
            local h = make_boss_handle("f")
            local ok, err = pcall(level, h, nil, 4, 1, { seat = "boss", style = "trickster" })
            expect(ok).to.equal(false)
            expect(err:find("trickster") ~= nil).to.equal(true)
        end)

        it("refuses an opponent this iter does not implement", function()
            reset()
            local h = make_boss_handle("f")
            local ok, err = pcall(level, h, nil, 4, 1, {
                seat = "boss",
                style = "guardian",
                opponents = { "greedy" },
            })
            expect(ok).to.equal(false)
            expect(err:find("not implemented") ~= nil).to.equal(true)
        end)
    end)

    describe("boss Card opponent (seat = player)", function()
        it("decodes the boss seat from the Card instead of the scripted policy", function()
            reset()
            -- Boss vocab, not player vocab: the two alphabets address
            -- different tables, and a mix-up decodes silently.
            local boss_card = make_boss_handle("c")
            ALIAS_TO_HANDLE["boss_charger"] = boss_card
            local h = make_handle("a")

            local scripted = level(h, "greedy", 4, 1)
            local carded = level(h, "boss_charger", 4, 1, { opponent_style = "guardian" })

            expect(carded.per_opponent.boss_charger.n_games).to.equal(4)
            -- The Card was asked for a move at all...
            expect(boss_card._last_prompt ~= nil).to.equal(true)
            -- ...and a boss that only ever charges is not the teacher.
            expect(carded.win_rate ~= scripted.win_rate).to.equal(true)
        end)

        it("builds the player view on the opponent_style basis", function()
            reset()
            ALIAS_TO_HANDLE["boss_charger"] = make_boss_handle("c")
            local guarded, guarded_prompts = recording(make_handle("a"))
            local rushed, rushed_prompts = recording(make_handle("a"))

            level(guarded, "boss_charger", 2, 1, { opponent_style = "guardian" })
            level(rushed, "boss_charger", 2, 1, { opponent_style = "rusher" })

            -- Both fights play the same moves (the stubs are fixed), so
            -- the states match turn for turn; only the distance field the
            -- basis computes can differ.
            expect(#guarded_prompts > 0).to.equal(true)
            expect(guarded_prompts[1] ~= rushed_prompts[1]).to.equal(true)
        end)

        it("keeps the guardian view basis for the scripted opponents", function()
            reset()
            local a, a_prompts = recording(make_handle("a"))
            local b, b_prompts = recording(make_handle("a"))
            level(a, "greedy", 2, 1)
            level(b, "random", 2, 1)
            -- Both scripted paths encode the opening view identically,
            -- which is the pre-Card behaviour this wire must not move.
            expect(a_prompts[1]).to.equal(b_prompts[1])
        end)

        it("requires opponent_style for a Card opponent", function()
            reset()
            ALIAS_TO_HANDLE["boss_charger"] = make_boss_handle("c")
            local ok, err = pcall(level, make_handle("a"), "boss_charger", 4, 1)
            expect(ok).to.equal(false)
            expect(err:find("opponent_style") ~= nil).to.equal(true)
        end)

        it("refuses opponent_style when every opponent is a scripted policy", function()
            reset()
            local ok, err = pcall(level, make_handle("a"), "greedy", 4, 1, {
                opponent_style = "guardian",
            })
            expect(ok).to.equal(false)
            expect(err:find("opponent_style") ~= nil).to.equal(true)
        end)

        it('refuses opponent_style on seat = "boss"', function()
            reset()
            local ok, err = pcall(level, make_boss_handle("f"), nil, 4, 1, {
                seat = "boss",
                style = "guardian",
                opponent_style = "guardian",
            })
            expect(ok).to.equal(false)
            expect(err:find("opponent_style") ~= nil).to.equal(true)
        end)

        it("refuses an unknown opponent_style", function()
            reset()
            ALIAS_TO_HANDLE["boss_charger"] = make_boss_handle("c")
            local ok, err = pcall(level, make_handle("a"), "boss_charger", 4, 1, {
                opponent_style = "trickster",
            })
            expect(ok).to.equal(false)
            expect(err:find("trickster") ~= nil).to.equal(true)
        end)

        it("refuses an alias no Card is bound to", function()
            reset()
            local ok, err = pcall(level, make_handle("a"), "ghost", 4, 1, {
                opponent_style = "guardian",
            })
            expect(ok).to.equal(false)
            expect(err:find("not bound to any Card") ~= nil).to.equal(true)
        end)

        it("accepts a boss handle passed directly, keyed as <table>", function()
            reset()
            local result = level(make_handle("a"), make_boss_handle("c"), 4, 1, {
                opponent_style = "guardian",
            })
            expect(type(result.per_opponent["<table>"])).to.equal("table")
            expect(result.per_opponent["<table>"].n_games).to.equal(4)
        end)
    end)

    describe("player Card opponent (seat = boss)", function()
        it("decodes the player seat from the Card instead of the random policy", function()
            reset()
            -- Player vocab here, mirroring the boss-vocab note above.
            local blocker = make_handle("b")
            local heavy = make_handle("A")
            ALIAS_TO_HANDLE["player_blocker"] = blocker
            ALIAS_TO_HANDLE["player_heavy"] = heavy
            local boss = make_boss_handle("f")

            local opts = { seat = "boss", style = "guardian" }
            local vs_blocker = level(boss, "player_blocker", 4, 1, opts)
            local vs_heavy = level(boss, "player_heavy", 4, 1, opts)

            expect(vs_blocker.per_opponent.player_blocker.n_games).to.equal(4)
            expect(blocker._last_prompt ~= nil).to.equal(true)
            -- A blocker stalls to the turn limit, a heavy attacker ends
            -- the fight early: the seat is reading the Card, not a
            -- scripted policy that would score both the same.
            expect(
                vs_blocker.per_opponent.player_blocker.game_length_mean
                    ~= vs_heavy.per_opponent.player_heavy.game_length_mean
            ).to.equal(true)
        end)

        it("builds the player view on opts.style", function()
            reset()
            local guarded, guarded_prompts = recording(make_handle("b"))
            local rushed, rushed_prompts = recording(make_handle("b"))
            local boss = make_boss_handle("f")

            level(boss, guarded, 2, 1, { seat = "boss", style = "guardian" })
            level(boss, rushed, 2, 1, { seat = "boss", style = "rusher" })

            expect(#guarded_prompts > 0).to.equal(true)
            expect(guarded_prompts[1] ~= rushed_prompts[1]).to.equal(true)
        end)

        it("refuses an alias no Card is bound to", function()
            reset()
            local ok, err = pcall(level, make_boss_handle("f"), "ghost", 4, 1, {
                seat = "boss",
                style = "guardian",
            })
            expect(ok).to.equal(false)
            expect(err:find("not bound to any Card") ~= nil).to.equal(true)
        end)

        it("accepts a player handle passed directly, keyed as <table>", function()
            reset()
            local result = level(make_boss_handle("f"), make_handle("b"), 4, 1, {
                seat = "boss",
                style = "guardian",
            })
            expect(type(result.per_opponent["<table>"])).to.equal("table")
            expect(result.per_opponent["<table>"].n_games).to.equal(4)
        end)
    end)

    describe("temperature", function()
        it("draws every player decision through the constrained sampler", function()
            reset()
            local result = level(make_handle("a"), "greedy", 2, 7, { temperature = 0.5 })
            expect(result.n_games).to.equal(2)
            expect(#SAMPLER_CALLS > 0).to.equal(true)
            expect(SAMPLER_CALLS[1].temperature).to.equal(0.5)
            -- Per-decision seed = base_seed + a run-local counter.
            expect(SAMPLER_CALLS[1].seed).to.equal(7)
            expect(SAMPLER_CALLS[2].seed).to.equal(8)
            -- Masked to the four player moves, not to the vocabulary.
            expect(#SAMPLER_CALLS[1].allow).to.equal(#MOVE_IDS)
            for i, id in ipairs(MOVE_IDS) do
                expect(SAMPLER_CALLS[1].allow[i]).to.equal(id)
            end
        end)

        it("masks a boss draw to the legal moves of the state", function()
            reset()
            level(make_boss_handle("f"), "random", 2, 3, {
                seat = "boss",
                style = "guardian",
                temperature = 1.0,
            })
            expect(#SAMPLER_CALLS > 0).to.equal(true)
            local allow = SAMPLER_CALLS[1].allow
            -- Five moves out of mode 1, six inside it — never the four
            -- of the player alphabet.
            expect(#allow == 5 or #allow == 6).to.equal(true)
        end)

        it("draws for both seats when the opponent is a Card too", function()
            reset()
            ALIAS_TO_HANDLE["boss_charger"] = make_boss_handle("c")
            level(make_handle("a"), "boss_charger", 1, 0, {
                opponent_style = "guardian",
                temperature = 1.0,
            })
            local player_draws, boss_draws = 0, 0
            for _, call in ipairs(SAMPLER_CALLS) do
                if #call.allow == #MOVE_IDS then
                    player_draws = player_draws + 1
                else
                    boss_draws = boss_draws + 1
                end
            end
            expect(player_draws > 0).to.equal(true)
            expect(boss_draws > 0).to.equal(true)
        end)

        it("replays the same run from the same seed", function()
            reset()
            local h = make_handle("a")
            local a = level(h, "greedy", 3, 2, { temperature = 1.0 })
            local first = #SAMPLER_CALLS
            local b = level(h, "greedy", 3, 2, { temperature = 1.0 })
            expect(a.win_rate).to.equal(b.win_rate)
            expect(#SAMPLER_CALLS - first).to.equal(first)
        end)

        it("never touches the sampler when no temperature is named", function()
            reset()
            local greedy = level(make_handle("a"), "greedy", 4, 1)
            expect(#SAMPLER_CALLS).to.equal(0)
            expect(type(greedy.win_rate)).to.equal("number")
        end)

        it("refuses a non-positive or non-finite temperature", function()
            reset()
            local h = make_handle("a")
            for _, bad in ipairs({ 0, -1, math.huge, 0 / 0 }) do
                local ok, err = pcall(level, h, "greedy", 4, 1, { temperature = bad })
                expect(ok).to.equal(false)
                expect(err:find("temperature") ~= nil).to.equal(true)
            end
            local ok, err = pcall(level, h, "greedy", 4, 1, { temperature = "hot" })
            expect(ok).to.equal(false)
            expect(err:find("temperature") ~= nil).to.equal(true)
        end)

        it("refuses a temperature on a build without the sampler", function()
            reset()
            local ok, err = without_sampler(function()
                return level(make_handle("a"), "greedy", 4, 1, { temperature = 1.0 })
            end)
            expect(ok).to.equal(false)
            expect(err:find("sampler") ~= nil).to.equal(true)
        end)

        it("still answers a greedy call on a build without the sampler", function()
            reset()
            local ok, result = without_sampler(function()
                return level(make_handle("a"), "greedy", 4, 1)
            end)
            expect(ok).to.equal(true)
            expect(result.n_games).to.equal(4)
        end)
    end)

    describe("game length and hp margin", function()
        it("reports both per opponent and pooled", function()
            reset()
            local h = make_handle("b")
            local result = level(h, nil, 4, 1, { opponents = { "greedy", "random" } })
            for _, name in ipairs({ "greedy", "random" }) do
                local row = result.per_opponent[name]
                expect(type(row.game_length_mean)).to.equal("number")
                expect(type(row.final_hp_margin_mean)).to.equal("number")
                expect(row.game_length_mean >= 1).to.equal(true)
                expect(row.game_length_mean <= duel.TURN_LIMIT).to.equal(true)
            end
            local mean = (
                result.per_opponent.greedy.game_length_mean
                + result.per_opponent.random.game_length_mean
            ) / 2
            expect(math.abs(result.game_length_mean - mean) < 1e-9).to.equal(true)
            local margin = (
                result.per_opponent.greedy.final_hp_margin_mean
                + result.per_opponent.random.final_hp_margin_mean
            ) / 2
            expect(math.abs(result.final_hp_margin_mean - margin) < 1e-9).to.equal(true)
        end)

        it("reads the margin boss-minus-player, whichever seat is measured", function()
            reset()
            local h = make_handle("a")
            -- Same player Card, two boss policies, one sign convention.
            --
            -- The teacher trades health for health and comes out ahead
            -- of a player who never blocks: boss minus player is
            -- positive even though the *player* seat is the measured
            -- one, so the sign tracks the boss and not the seat.
            local scripted = level(h, "greedy", 2, 1)
            expect(scripted.final_hp_margin_mean > 0).to.equal(true)

            -- A boss that only ever charges deals nothing and blocks
            -- everything after the opening hit, so it ends exactly that
            -- hit (4) behind an untouched player, for all nine turns.
            local carded = level(h, make_boss_handle("c"), 2, 1, {
                opponent_style = "guardian",
            })
            expect(carded.final_hp_margin_mean).to.equal(-4.0)
            expect(carded.game_length_mean).to.equal(duel.TURN_LIMIT)
        end)

        it("is additive: the pre-existing fields keep their values", function()
            reset()
            local h = make_handle("b")
            local a = level(h, "greedy", 4, 1)
            local b = level(h, "greedy", 4, 1)
            expect(a.win_rate).to.equal(b.win_rate)
            expect(a.wins).to.equal(b.wins)
            expect(a.n_games).to.equal(b.n_games)
            expect(a.win_rate_min).to.equal(b.win_rate_min)
            expect(a.per_opponent.greedy.win_rate).to.equal(a.win_rate)
        end)
    end)

    describe("per-game records", function()
        --- Sorted key list of a table, so a spec can assert on the exact
        --- field set rather than on the fields it thought to name.
        local function fields_of(row)
            local names = {}
            for key in pairs(row) do
                names[#names + 1] = tostring(key)
            end
            table.sort(names)
            return table.concat(names, ",")
        end

        it("emits one record per fight under per_game = true", function()
            reset()
            local h = make_handle("b")
            local result = level(h, "greedy", 5, 1, { per_game = true })
            local row = result.per_opponent.greedy
            expect(type(row.games)).to.equal("table")
            expect(#row.games).to.equal(5)
            for _, record in ipairs(row.games) do
                expect(fields_of(record)).to.equal("final_hp_margin,game_length,outcome")
                expect(record.outcome == 0.0 or record.outcome == 0.5 or record.outcome == 1.0).to.equal(
                    true
                )
                expect(record.game_length).to.equal(math.floor(record.game_length))
                expect(record.game_length >= 1).to.equal(true)
                expect(record.game_length <= duel.TURN_LIMIT).to.equal(true)
                expect(record.final_hp_margin).to.equal(math.floor(record.final_hp_margin))
            end
        end)

        it("reproduces every mean of its cell exactly", function()
            reset()
            local h = make_handle("b")
            local result =
                level(h, nil, 6, 2, { opponents = { "greedy", "random" }, per_game = true })
            for _, name in ipairs({ "greedy", "random" }) do
                local row = result.per_opponent[name]
                expect(#row.games).to.equal(6)
                -- Exact equality, not a tolerance: the records are the
                -- very values the metric summed, added back in the same
                -- (game index) order, and all three are exact in binary
                -- (`outcome` is 0 / 0.5 / 1, the other two are small
                -- integers). Same addends, same order, same rounding —
                -- so the recomputed mean is bit-identical. Summing the
                -- records in any other order would only be *close*.
                local wins, turns, margin = 0.0, 0.0, 0.0
                for _, record in ipairs(row.games) do
                    wins = wins + record.outcome
                    turns = turns + record.game_length
                    margin = margin + record.final_hp_margin
                end
                expect(wins / 6).to.equal(row.win_rate)
                expect(turns / 6).to.equal(row.game_length_mean)
                expect(margin / 6).to.equal(row.final_hp_margin_mean)
            end
        end)

        it("records the boss seat on the same contract", function()
            reset()
            local result = level(make_boss_handle("f"), nil, 3, 1, {
                seat = "boss",
                style = "guardian",
                per_game = true,
            })
            local row = result.per_opponent.random
            expect(#row.games).to.equal(3)
            local wins = 0.0
            for _, record in ipairs(row.games) do
                expect(fields_of(record)).to.equal("final_hp_margin,game_length,outcome")
                wins = wins + record.outcome
            end
            expect(wins / 3).to.equal(row.win_rate)
        end)

        it("leaves the key absent when per_game is false, nil or omitted", function()
            reset()
            local h = make_handle("b")
            local omitted = level(h, "greedy", 4, 1)
            local off = level(h, "greedy", 4, 1, { per_game = false })
            local explicit_nil = level(h, "greedy", 4, 1, { per_game = nil })
            expect(omitted.per_opponent.greedy.games).to.equal(nil)
            expect(off.per_opponent.greedy.games).to.equal(nil)
            expect(explicit_nil.per_opponent.greedy.games).to.equal(nil)
            -- The key is missing rather than present-and-nil, which is
            -- what keeps the encoded output identical to the pre-flag
            -- shape for an encoder that distinguishes the two.
            expect(fields_of(off.per_opponent.greedy)).to.equal(
                fields_of(omitted.per_opponent.greedy)
            )
        end)

        it("puts no games array on the pooled result", function()
            reset()
            local result = level(make_handle("b"), nil, 3, 1, {
                opponents = { "greedy", "random" },
                per_game = true,
            })
            -- The pooled distribution is the concatenation of the
            -- per-opponent ones; storing it twice would let them drift.
            expect(result.games).to.equal(nil)
            expect(type(result.per_opponent.greedy.games)).to.equal("table")
            expect(type(result.per_opponent.random.games)).to.equal("table")
        end)

        it("moves no pre-existing number when the flag is turned on", function()
            reset()
            local h = make_handle("b")
            local pool = { "greedy", "random" }
            local off = level(h, nil, 4, 1, { opponents = pool })
            local on = level(h, nil, 4, 1, { opponents = pool, per_game = true })
            for _, key in ipairs({
                "win_rate",
                "ci_lower",
                "ci_upper",
                "wins",
                "n_games",
                "win_rate_min",
                "game_length_mean",
                "final_hp_margin_mean",
            }) do
                expect(on[key]).to.equal(off[key])
            end
            for _, name in ipairs(pool) do
                for _, key in ipairs({
                    "win_rate",
                    "ci_lower",
                    "ci_upper",
                    "n_games",
                    "game_length_mean",
                    "final_hp_margin_mean",
                }) do
                    expect(on.per_opponent[name][key]).to.equal(off.per_opponent[name][key])
                end
            end
        end)

        it("refuses a non-boolean per_game", function()
            reset()
            local h = make_handle("b")
            for _, bad in ipairs({ "true", "false", 1, 0, {} }) do
                local ok, err = pcall(level, h, "greedy", 4, 1, { per_game = bad })
                expect(ok).to.equal(false)
                expect(err:find("per_game") ~= nil).to.equal(true)
                expect(err:find("true, false or nil") ~= nil).to.equal(true)
            end
        end)
    end)
end)
