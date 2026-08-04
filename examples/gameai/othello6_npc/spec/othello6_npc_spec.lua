-- othello6_npc/spec/othello6_npc_spec.lua
--
-- Package-level spec for the 6x6 Othello decode surface. Run with
-- `alc_pkg_test pkg="othello6_npc"` after `alc_pkg_link` has registered
-- `othello6`, `othello6_teacher` and this package, or through the
-- `mlua-probe` test runner with `examples/gameai` on the search path.
-- The `lust` globals are pre-loaded by both runners.
--
-- No Card and no model are touched: `alc.card` and `alc.nn` are replaced
-- by stubs that hand out a fake handle whose logits rank a list of moves
-- the case chooses. The default ranking puts the pass first and then the
-- squares in index order, which makes the stub model an exact policy —
-- "the lowest-index legal placement, or the pass when there is none" —
-- and lets the self-play aggregation be asserted against an independent
-- count rather than merely parsed. It also exercises the gate on every
-- position that has a placement, since the raw argmax is the pass there.
--
-- What is left under test is the request surface: the prompt, the gate,
-- the mask the draw is made under, the alias resolution, the teacher
-- pair, the self-play aggregation and the rejection of a field no mode
-- reads.
--
-- The prompt is the opening marker and the moves that reached the
-- position, which is a training row cut off where the model is asked to
-- continue it. The marker is why the opening can be decoded from at all:
-- the bridge refuses a prompt with no token to forward. Self-play runs
-- the Card at both seats for another reason: the corpus is one policy
-- playing itself, so both colours are positions the Card was trained to
-- answer.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

-- The rules module reaches for the host RNG, and the entry decodes its
-- request through the host JSON surface. The stubs below match the
-- shapes the engine-level harness installs; a real `alc` namespace is
-- left alone wherever one is present.
_G.alc = _G.alc or {}

alc.math = alc.math or {}
alc.math.rng_create = alc.math.rng_create
    or function(seed)
        return { state = math.floor(seed) % 2147483647 }
    end
alc.math.rng_int = alc.math.rng_int
    or function(rng, min, max)
        rng.state = (rng.state * 1103515245 + 12345) % 2147483648
        return min + (rng.state // 65536) % (max - min + 1)
    end

-- A stand-in for the JSON round trip, used only when the runner has no
-- real one. It is not a JSON codec: it hands out a string key into a
-- table of deep copies. What the entry contracts on is that `ctx.task`
-- is a *string* that decodes to a fresh object, and that is exactly what
-- this preserves.
if type(alc.json_encode) ~= "function" or type(alc.json_decode) ~= "function" then
    local boxes = {}
    local function deep_copy(value)
        if type(value) ~= "table" then
            return value
        end
        local out = {}
        for key, entry in pairs(value) do
            out[key] = deep_copy(entry)
        end
        return out
    end
    alc.json_encode = function(value)
        boxes[#boxes + 1] = deep_copy(value)
        return "\1json:" .. #boxes
    end
    alc.json_decode = function(text)
        local index = type(text) == "string" and text:match("^\1json:(%d+)$") or nil
        if index == nil then
            error("spec: ctx.task reached json_decode as something the stub never encoded")
        end
        return deep_copy(boxes[tonumber(index)])
    end
end

local othello = require("othello6")
local teacher = require("othello6_teacher")

local VOCAB = othello.vocab()

--- Everything the module and the teacher did during one request, in
--- order. A `decode` entry is one model decision, a `teacher` entry one
--- call of the policy; the self-play oracle below reads the interleaving
--- of the two.
local EVENTS = {}

--- The ranking the fake logits row hands out, as `{ id = <token id> }`
--- in descending order.
local ORDER = {}

--- Rank `actions` first and everything else after them.
---
--- The tail is the rest of the vocabulary in descending id order, so the
--- gate always finds a legal token however short the preference list is.
local function set_ranking(actions)
    local order, seen = {}, {}
    for _, action in ipairs(actions) do
        local id = VOCAB.to_id[action]
        if id == nil then
            error("spec: " .. tostring(action) .. " is not a move character")
        end
        order[#order + 1] = { id = id }
        seen[id] = true
    end
    for id = VOCAB.size - 1, 0, -1 do
        if not seen[id] then
            order[#order + 1] = { id = id }
        end
    end
    ORDER = order
end

--- The default ranking: the pass, then the squares in index order.
---
--- It makes the stub model the policy "play the lowest-index legal
--- placement, or pass when there is none", which is `legal_actions[1]`
--- for every position. The self-play oracle is built on that identity.
local INDEX_ORDER = { othello.PASS }
for index = 0, othello.CELLS - 1 do
    INDEX_ORDER[#INDEX_ORDER + 1] = othello.action_of_index(index)
end

set_ranking(INDEX_ORDER)

local LOGITS = {
    argmax = function()
        return ORDER[1].id
    end,
    vocab = function()
        return VOCAB.size
    end,
    top = function(_, n)
        local out = {}
        for i = 1, math.min(n, #ORDER) do
            out[i] = ORDER[i]
        end
        return out
    end,
}

local SESSION = {
    next_logits = function()
        return LOGITS
    end,
}

--- Prompt of the most recent decode, as token ids.
local last_prompt = nil

--- Read a prompt back as the line the model saw.
local function text_of(ids)
    local chars = {}
    for _, id in ipairs(ids or {}) do
        chars[#chars + 1] = VOCAB.to_char[id]
    end
    return table.concat(chars)
end

--- The stub handle.
---
--- A prompt is the opening marker and the move sequence, so the line the
--- session is opened over *is* the encoding and the event log records it
--- unchanged. The real bridge additionally refuses an empty prompt; this
--- one does not, so the length of the shortest prompt is asserted in a
--- case of its own rather than left to the stub to enforce.
local HANDLE = {
    generate_session = function(_, ids)
        last_prompt = ids
        EVENTS[#EVENTS + 1] = { kind = "decode", encoded = text_of(ids) }
        return SESSION
    end,
}

--- Alias of the most recent Card lookup.
local last_alias = nil

alc.card = {
    get_by_alias = function(alias)
        last_alias = alias
        return { card_id = "stub-card-" .. alias }
    end,
}

-- ─── Sampler stubs ──────────────────────────────────────────────────
--
-- Ported from `guardian_duel_npc/spec`. There is no RNG of the real
-- kind, but the *contract* of the bridge is kept: a composition consumes
-- both of its arguments and a spent handle is a loud error, so a chain
-- reused across decisions fails here exactly as it would against
-- `alc.nn.sampler.constrained`. An empty allow list is rejected at
-- construction, which is the Rust-side behaviour the pass exists for.

--- Every chain built since the last reset, oldest first. Each entry is
--- `{ temperature, seed, allow, draws }`.
local CHAINS = {}

--- Times each construction surface was reached since the last reset.
local BUILDS = { temperature = 0, allow_list = 0, constrained = 0 }

--- The stand-in for a temperature draw.
---
--- Not a model of the real sampler — there is no row of logits behind
--- it — but a function of the seed and the allow list alone, which is
--- the property the specs are about: a replay that derives the same seed
--- draws the same move, and neighbouring seeds do not all land on one
--- id.
local function draw(seed, allow)
    local mixed = (seed * 2654435761 + 1013904223) % 4294967296
    return allow[math.floor(mixed / 65536) % #allow + 1]
end

--- A handle that can be moved out of, like the bridge's.
local function spendable(kind)
    return { kind = kind, spent = false }
end

local function spend(handle, kind)
    if handle.spent then
        error(
            "spec: this "
                .. kind
                .. " was moved into a constrained sampler; build a fresh one instead"
        )
    end
    handle.spent = true
end

alc.nn = {
    card = {
        load_handle = function()
            return HANDLE
        end,
    },
    sampler = {
        temperature = function(temperature, seed)
            BUILDS.temperature = BUILDS.temperature + 1
            local handle = spendable("sampler")
            handle.temperature = temperature
            handle.seed = seed
            return handle
        end,
        constrained = function(inner, constraint)
            BUILDS.constrained = BUILDS.constrained + 1
            spend(inner, "sampler")
            spend(constraint, "constraint")
            local chain = {
                temperature = inner.temperature,
                seed = inner.seed,
                allow = constraint.ids,
                draws = 0,
            }
            CHAINS[#CHAINS + 1] = chain
            return {
                sample = function(self, logits)
                    if logits ~= LOGITS then
                        error("spec: a sampler must be handed the session's own logits row")
                    end
                    self.chain.draws = self.chain.draws + 1
                    return draw(chain.seed, chain.allow)
                end,
                chain = chain,
            }
        end,
    },
    constraint = {
        allow_list = function(ids)
            BUILDS.allow_list = BUILDS.allow_list + 1
            if type(ids) ~= "table" or #ids == 0 then
                error("spec: an empty allow list must be rejected at construction")
            end
            local handle = spendable("constraint")
            local copy = {}
            for i, id in ipairs(ids) do
                copy[i] = id
            end
            handle.ids = copy
            return handle
        end,
    },
}

--- The sampler namespaces, kept aside so a case can take them away and
--- put them back (a build with `alc.nn.card` and no draw surface).
local SAMPLER_NS, CONSTRAINT_NS = alc.nn.sampler, alc.nn.constraint

local npc = require("othello6_npc")

-- ─── Teacher stubs ──────────────────────────────────────────────────

local REAL_POLICY = teacher.policy

--- The pair the most recent teacher was built with.
local last_pair = nil

--- Run `body` with `othello6_teacher.policy` answering through `pick`.
---
--- The real negamax is a correct teacher but an opaque one: a spec that
--- asserted an aggregation against it would be asserting the search as
--- well. `pick` takes the legal moves of the position and returns one of
--- them, which is enough to build a teacher that agrees with the stub
--- model everywhere or nowhere. Restored on the way out, failure or not.
local function with_teacher(pick, body)
    teacher.policy = function(depth, style)
        last_pair = { depth = depth, style = style }
        return function(state)
            local legal = othello.legal_actions(state)
            local said = pick(legal)
            EVENTS[#EVENTS + 1] = {
                kind = "teacher",
                encoded = othello.encode(state),
                side = othello.side_to_move(state),
                said = said,
                model_would = legal[1],
                pass_only = #legal == 1 and legal[1] == othello.PASS,
            }
            return said
        end
    end
    local ok, err = pcall(body)
    teacher.policy = REAL_POLICY
    if not ok then
        error(err, 0)
    end
end

--- Count the self-play summary out of the event log.
---
--- Every turn is a `decode` immediately followed by the `teacher` call
--- for the same position — the comparison — because the Card plays both
--- seats and each of its moves is scored. A lone `teacher` call would
--- mean a position the Card did not answer, so the pairing is asserted
--- here rather than assumed. The stub model plays `legal_actions[1]`, so
--- the comparison is a hit exactly when the teacher named the same move,
--- and the raw argmax (the pass) is legal exactly on a pass position.
---@return integer hits
---@return integer moves
---@return integer illegal
---@return table sides Seats the model was asked at, as a set
local function oracle()
    local hits, moves, illegal = 0, 0, 0
    local sides = {}
    for i, event in ipairs(EVENTS) do
        if event.kind == "decode" then
            moves = moves + 1
            local answer = EVENTS[i + 1]
            if answer == nil or answer.kind ~= "teacher" or answer.encoded ~= event.encoded then
                error("spec: a model decision was not followed by the teacher comparison")
            end
            sides[answer.side] = true
            if answer.said == answer.model_would then
                hits = hits + 1
            end
            if not answer.pass_only then
                illegal = illegal + 1
            end
        elseif event.kind == "teacher" then
            local asked = EVENTS[i - 1]
            if asked == nil or asked.kind ~= "decode" or asked.encoded ~= event.encoded then
                error("spec: the teacher answered a position the model was never asked")
            end
        end
    end
    return hits, moves, illegal, sides
end

-- ─── Fixtures ───────────────────────────────────────────────────────

--- The opening. Black to move, four legal placements.
---
--- Nothing has been played, so it encodes to `othello6.BOS` alone: one
--- token, which is what the bridge needs to open a session over at all
--- ("alc.nn generate_session: prompt_tokens is empty" is what an empty
--- one is answered with). The cases below decode from it freely, and the
--- prompt-shape case asserts the length rather than leaving it to the
--- stub handle, which carries no such check of its own.
local OPENING = othello.new_game(1)

--- A position three moves in, for the cases that read a prompt.
---
--- The moves are taken off the head of the legal list rather than
--- written out, so the fixture follows the rules module instead of
--- pinning a second copy of the opening's legal set.
local PLAYED = (function()
    local state = othello.new_game(2)
    for _ = 1, 3 do
        state = othello.apply(state, othello.legal_actions(state)[1])
    end
    return state
end)()

--- A position where the side to move has no placement at all.
---
--- Black holds no disc, so no direction can bracket for it and the only
--- answer the rules allow is the pass. Othello reaches such positions in
--- ordinary play, which is why the allow list must never come out empty.
local PASS_ONLY = othello.state_from_rows({
    "......",
    "......",
    "..WW..",
    "..WW..",
    "......",
    "......",
}, "black")

local function ask(payload, extra)
    local request = { task = alc.json_encode(payload) }
    for key, value in pairs(extra or {}) do
        request[key] = value
    end
    npc.reset_cache()
    last_prompt, last_alias, last_pair = nil, nil, nil
    EVENTS, CHAINS = {}, {}
    BUILDS = { temperature = 0, allow_list = 0, constrained = 0 }
    set_ranking(INDEX_ORDER)
    return npc.run(request).result
end

--- `ask` without clearing the sampler counters, for the cases that watch
--- a chain being rebuilt across requests.
local function ask_keep(payload, extra)
    local request = { task = alc.json_encode(payload) }
    for key, value in pairs(extra or {}) do
        request[key] = value
    end
    npc.reset_cache()
    return npc.run(request).result
end

-- ─── decide ─────────────────────────────────────────────────────────

describe("othello6_npc decide", function()
    it("gates the illegal argmax down to the first legal move", function()
        -- The default ranking leads with the pass, which is illegal
        -- wherever a placement exists. The gate walks past it to the
        -- lowest-index legal square, which is (1,2) in the opening.
        expect(ask({ mode = "decide", state = OPENING })).to.equal(
            "action=i legal=true raw_legal=false gated=true"
        )
        expect(othello.action_of_index(8)).to.equal("i")
    end)

    it("takes the first legal move of the ranking, not the first legal move", function()
        -- The scan is over the logit order. `a` is square 0 and illegal
        -- in the opening, `y` is square 22 and legal, `i` is square 8 and
        -- also legal but ranked below `y`.
        set_ranking({ "a", "y", "i" })
        local text =
            npc.run({ task = alc.json_encode({ mode = "decide", state = OPENING }) }).result
        expect(text).to.equal("action=y legal=true raw_legal=false gated=true")
    end)

    it("keeps the argmax once the position allows it", function()
        set_ranking({ "i", "y" })
        local text =
            npc.run({ task = alc.json_encode({ mode = "decide", state = OPENING }) }).result
        expect(text).to.equal("action=i legal=true raw_legal=true gated=false")
    end)

    it("answers the pass on a position with no placement", function()
        -- `raw_legal` holds here precisely because the pass leads the
        -- default ranking and is the one move this position allows.
        expect(ask({ mode = "decide", state = PASS_ONLY })).to.equal(
            "action=- legal=true raw_legal=true gated=false"
        )
        expect(#othello.legal_actions(PASS_ONLY)).to.equal(1)
    end)

    it("prompts with the marker and the moves that reached the position", function()
        ask({ mode = "decide", state = PLAYED })
        expect(text_of(last_prompt)).to.equal(othello.encode(PLAYED))
        expect(text_of(last_prompt):sub(1, 1)).to.equal(othello.BOS)
        expect(#text_of(last_prompt)).to.equal(4)
    end)

    it("prompts the opening with the marker alone, which the bridge accepts", function()
        -- The bridge refuses a session with no token to forward, so the
        -- number that matters is one rather than zero. Without the marker
        -- this decode could not be asked for at all.
        ask({ mode = "decide", state = OPENING })
        expect(#last_prompt).to.equal(1)
        expect(text_of(last_prompt)).to.equal(othello.BOS)
    end)

    it("puts no separator in the prompt", function()
        -- A training row is one game written move by move with nothing
        -- between them, so a prompt carrying a separator would be a line
        -- of a kind the model never read. The character is still in the
        -- vocabulary, so its absence here is a choice rather than an
        -- impossibility.
        ask({ mode = "decide", state = PLAYED })
        expect(text_of(last_prompt):find(">", 1, true)).to.equal(nil)
        expect(VOCAB.to_id[">"] ~= nil).to.equal(true)
    end)

    it("rejects a position that is not an object", function()
        expect(function()
            ask({ mode = "decide", state = othello.encode(OPENING) })
        end).to.fail()
    end)

    it("rejects a position missing a field the rules read", function()
        local state = othello.new_game(1)
        state.turn = nil
        expect(function()
            ask({ mode = "decide", state = state })
        end).to.fail()
    end)

    it("rejects the fields another mode reads", function()
        expect(function()
            ask({ mode = "decide", state = OPENING, seed = 7 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide", state = OPENING, temperature = 0.8 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide", state = OPENING, games = 2 })
        end).to.fail()
    end)
end)

-- ─── determinism ────────────────────────────────────────────────────

describe("othello6_npc determinism", function()
    it("agrees across two independent sessions", function()
        expect(ask({ mode = "determinism", state = OPENING })).to.equal(
            "deterministic=true action=i"
        )
    end)

    it("reports a disagreement when the two decodes differ", function()
        -- The check is only worth running if it can fail, so the ranking
        -- is moved between the two decodes to make it.
        local flipped = false
        local real = SESSION.next_logits
        SESSION.next_logits = function()
            if not flipped then
                flipped = true
                set_ranking({ "i" })
            else
                set_ranking({ "y" })
            end
            return LOGITS
        end
        local ok, text = pcall(ask, { mode = "determinism", state = OPENING })
        SESSION.next_logits = real
        expect(ok).to.equal(true)
        expect(text).to.equal("deterministic=false action=i")
    end)

    it("rejects a field another mode reads", function()
        expect(function()
            ask({ mode = "determinism", state = OPENING, seed = 7 })
        end).to.fail()
    end)
end)

-- ─── decide_noisy ───────────────────────────────────────────────────

--- Token ids of the moves a position allows, in `legal_actions` order —
--- the shape `alc.nn.constraint.allow_list` takes and the order the
--- module builds it in.
local function legal_ids(state)
    local ids = {}
    for _, action in ipairs(othello.legal_actions(state)) do
        ids[#ids + 1] = VOCAB.to_id[action]
    end
    return ids
end

--- The move the stub draws for `seed` at `state`, as the summary spells
--- it.
local function drawn(seed, state)
    return VOCAB.to_char[draw(seed, legal_ids(state))]
end

describe("othello6_npc decide_noisy", function()
    it("draws a legal move and reports the draw", function()
        expect(ask({ mode = "decide_noisy", state = OPENING, seed = 7 })).to.equal(
            string.format(
                "action=%s legal=true raw_legal=false noisy=true temperature=1 seed=7",
                drawn(7, OPENING)
            )
        )
    end)

    it("masks the moves the position allows and nothing else", function()
        ask({ mode = "decide_noisy", state = OPENING, seed = 7 })
        expect(#CHAINS).to.equal(1)
        local opening = legal_ids(OPENING)
        expect(#CHAINS[1].allow).to.equal(4)
        for i, id in ipairs(opening) do
            expect(CHAINS[1].allow[i]).to.equal(id)
        end
    end)

    it("masks the pass alone on a position with no placement", function()
        -- The Rust allow list is rejected when it is built from nothing,
        -- so the rules answer `{ PASS }` rather than `{}` and the mask
        -- comes out one long instead of empty.
        local text = ask({ mode = "decide_noisy", state = PASS_ONLY, seed = 7 })
        expect(#CHAINS).to.equal(1)
        expect(#CHAINS[1].allow).to.equal(1)
        expect(CHAINS[1].allow[1]).to.equal(VOCAB.to_id[othello.PASS])
        expect(text).to.equal("action=- legal=true raw_legal=true noisy=true temperature=1 seed=7")
    end)

    it("builds a fresh sampler and constraint for every decision", function()
        -- `alc.nn.sampler.constrained` moves both of its arguments, so a
        -- chain held across decisions is a spent handle on the second
        -- one. Three requests must reach all three surfaces three times.
        CHAINS = {}
        BUILDS = { temperature = 0, allow_list = 0, constrained = 0 }
        for seed = 1, 3 do
            ask_keep({ mode = "decide_noisy", state = OPENING, seed = seed })
        end
        expect(#CHAINS).to.equal(3)
        expect(BUILDS.temperature).to.equal(3)
        expect(BUILDS.allow_list).to.equal(3)
        expect(BUILDS.constrained).to.equal(3)
        for _, chain in ipairs(CHAINS) do
            expect(chain.draws).to.equal(1)
        end
    end)

    it("reports the raw argmax legality off the ungated row", function()
        set_ranking({ "i" })
        local text = npc.run({
            task = alc.json_encode({ mode = "decide_noisy", state = OPENING, seed = 7 }),
        }).result
        expect(text:match("raw_legal=(%a+)")).to.equal("true")
    end)

    it("draws the same move from the same seed", function()
        local first = ask({ mode = "decide_noisy", state = OPENING, seed = 11 })
        local second = ask({ mode = "decide_noisy", state = OPENING, seed = 11 })
        expect(first).to.equal(second)
    end)

    it("does not answer every seed with one move", function()
        local seen, distinct = {}, 0
        for seed = 1, 12 do
            local action =
                ask({ mode = "decide_noisy", state = OPENING, seed = seed }):match("action=(%a)")
            if not seen[action] then
                seen[action] = true
                distinct = distinct + 1
            end
        end
        expect(distinct > 1).to.equal(true)
    end)

    it("prompts with the marker and the moves that reached the position", function()
        ask({ mode = "decide_noisy", state = PLAYED, seed = 7 })
        expect(text_of(last_prompt)).to.equal(othello.encode(PLAYED))
        expect(text_of(last_prompt):sub(1, 1)).to.equal(othello.BOS)
        expect(text_of(last_prompt):find(">", 1, true)).to.equal(nil)
    end)

    it("carries the requested temperature into the sampler", function()
        local text = ask({ mode = "decide_noisy", state = OPENING, seed = 7, temperature = 0.75 })
        expect(CHAINS[1].temperature).to.equal(0.75)
        expect(text:match("temperature=([%d%.]+)")).to.equal("0.75")
    end)

    it("defaults the temperature to 1.0", function()
        ask({ mode = "decide_noisy", state = OPENING, seed = 7 })
        expect(CHAINS[1].temperature).to.equal(1.0)
    end)

    it("passes the seed the caller derived", function()
        ask({ mode = "decide_noisy", state = OPENING, seed = 42 })
        expect(CHAINS[1].seed).to.equal(42)
    end)

    it("floors a fractional seed and echoes the one it drew under", function()
        local text = ask({ mode = "decide_noisy", state = OPENING, seed = 7.9 })
        expect(CHAINS[1].seed).to.equal(7)
        expect(text).to.equal(
            string.format(
                "action=%s legal=true raw_legal=false noisy=true temperature=1 seed=7",
                drawn(7, OPENING)
            )
        )
    end)

    it("requires a seed", function()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING })
        end).to.fail()
    end)

    it("rejects a seed that is not a non-negative number", function()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = "7" })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = -1 })
        end).to.fail()
    end)

    it("rejects a temperature that is not a finite positive number", function()
        -- Zero is a caller who means greedy, and greedy has a mode.
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = 7, temperature = 0 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = 7, temperature = -0.5 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = 7, temperature = "hot" })
        end).to.fail()
    end)

    it("refuses the draw on a build with no sampler surface", function()
        alc.nn.sampler, alc.nn.constraint = nil, nil
        local ok = pcall(ask, { mode = "decide_noisy", state = OPENING, seed = 7 })
        local greedy = ask({ mode = "decide", state = OPENING })
        alc.nn.sampler, alc.nn.constraint = SAMPLER_NS, CONSTRAINT_NS
        expect(ok).to.equal(false)
        expect(greedy).to.equal("action=i legal=true raw_legal=false gated=true")
    end)

    it("leaves the greedy modes away from the sampler", function()
        expect(ask({ mode = "decide", state = OPENING })).to.equal(
            "action=i legal=true raw_legal=false gated=true"
        )
        expect(#CHAINS).to.equal(0)
        expect(ask({ mode = "determinism", state = OPENING })).to.equal(
            "deterministic=true action=i"
        )
        expect(#CHAINS).to.equal(0)
    end)

    it("rejects a field another mode reads", function()
        expect(function()
            ask({ mode = "decide_noisy", state = OPENING, seed = 7, games = 2 })
        end).to.fail()
    end)
end)

-- ─── alias resolution ───────────────────────────────────────────────

describe("othello6_npc card alias", function()
    it("falls back to the bare alias", function()
        ask({ mode = "decide", state = OPENING })
        expect(last_alias).to.equal("othello6_npc")
    end)

    it("reads the alias from the task JSON", function()
        ask({ mode = "decide", state = OPENING, card_alias = "othello6_npc_d4_mobility" })
        expect(last_alias).to.equal("othello6_npc_d4_mobility")
    end)

    it("prefers the ctx alias over the task one", function()
        ask({ mode = "decide", state = OPENING, card_alias = "othello6_npc_d4_mobility" }, {
            card_alias = "othello6_npc_d1_greedy",
        })
        expect(last_alias).to.equal("othello6_npc_d1_greedy")
    end)

    it("rejects an alias that is not a string", function()
        expect(function()
            ask({ mode = "decide", state = OPENING, card_alias = 7 })
        end).to.fail()
    end)

    it("rejects an empty alias", function()
        expect(function()
            ask({ mode = "decide", state = OPENING, card_alias = "" })
        end).to.fail()
    end)
end)

-- ─── teacher pair ───────────────────────────────────────────────────

describe("othello6_npc teacher pair", function()
    it("defaults to the pair the first bake trains", function()
        with_teacher(function(legal)
            return legal[1]
        end, function()
            ask({ mode = "selfplay", games = 1, seed = 3 })
            expect(last_pair.depth).to.equal(2)
            expect(last_pair.style).to.equal("corner")
        end)
    end)

    it("takes the pair the ctx names", function()
        with_teacher(function(legal)
            return legal[1]
        end, function()
            ask({ mode = "selfplay", games = 1, seed = 3 }, { depth = 4, style = "mobility" })
            expect(last_pair.depth).to.equal(4)
            expect(last_pair.style).to.equal("mobility")
        end)
    end)

    it("lets the request override the pair, one field at a time", function()
        -- This is how a cross-depth comparison is asked for: the same
        -- Card scored against a teacher it was not labelled by.
        with_teacher(function(legal)
            return legal[1]
        end, function()
            ask({ mode = "selfplay", games = 1, seed = 3, depth = 6 }, {
                depth = 2,
                style = "greedy",
            })
            expect(last_pair.depth).to.equal(6)
            expect(last_pair.style).to.equal("greedy")
        end)
    end)

    it("rejects a style outside othello6.STYLES", function()
        expect(function()
            ask({ mode = "decide", state = OPENING }, { style = "corners" })
        end).to.fail()
        expect(function()
            ask({ mode = "selfplay", games = 1, seed = 3, style = "corners" })
        end).to.fail()
    end)

    it("rejects a depth that is not a positive integer", function()
        expect(function()
            ask({ mode = "decide", state = OPENING }, { depth = 0 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide", state = OPENING }, { depth = 1.5 })
        end).to.fail()
        expect(function()
            ask({ mode = "selfplay", games = 1, seed = 3, depth = "2" })
        end).to.fail()
    end)
end)

-- ─── selfplay ───────────────────────────────────────────────────────

describe("othello6_npc selfplay", function()
    it("reports the summary fields", function()
        with_teacher(function(legal)
            return legal[1]
        end, function()
            local text = ask({ mode = "selfplay", games = 2, seed = 5 })
            local pattern = "^winrate=%d+%.%d%d illegal=%d+ style_match=[01]%.%d%d "
                .. "style_hits=%d+/%d+$"
            expect(text:match(pattern) ~= nil).to.equal(true)
        end)
    end)

    it("asks the model at both seats", function()
        -- The corpus is one policy playing itself, so a Card trained on
        -- it was asked to imitate the teacher on black and on white
        -- alike. A run that let the teacher answer half the positions
        -- would score the Card on half of what it was taught and over
        -- games it only half steered. `oracle` fails on a teacher call
        -- the model was not asked first, and both colours have to appear
        -- among the positions it was asked at.
        with_teacher(function(legal)
            return legal[1]
        end, function()
            ask({ mode = "selfplay", games = 2, seed = 5 })
            local _, moves, _, sides = oracle()
            expect(moves > 0).to.equal(true)
            expect(sides.black).to.equal(true)
            expect(sides.white).to.equal(true)
        end)
    end)

    it("scores a full match against a teacher that plays what the model plays", function()
        with_teacher(function(legal)
            return legal[1]
        end, function()
            local text = ask({ mode = "selfplay", games = 2, seed = 5 })
            local hits, moves = oracle()
            expect(moves > 0).to.equal(true)
            expect(hits).to.equal(moves)
            expect(text:match("style_match=([%d%.]+)")).to.equal("1.00")
            expect(text:match("style_hits=(%d+/%d+)")).to.equal(hits .. "/" .. moves)
        end)
    end)

    it("counts the disagreements against a teacher that answers otherwise", function()
        -- The teacher takes the last legal move, which is the model's
        -- only when the position allows exactly one. The aggregation is
        -- checked against the event log rather than against a literal:
        -- what is under test is that the ratio counts model decisions.
        with_teacher(function(legal)
            return legal[#legal]
        end, function()
            local text = ask({ mode = "selfplay", games = 2, seed = 5 })
            local hits, moves, illegal = oracle()
            expect(moves > 0).to.equal(true)
            expect(hits < moves).to.equal(true)
            expect(text:match("style_hits=(%d+/%d+)")).to.equal(hits .. "/" .. moves)
            expect(text:match("style_match=([%d%.]+)")).to.equal(
                string.format("%.2f", hits / moves)
            )
            expect(tonumber(text:match("illegal=(%d+)"))).to.equal(illegal)
        end)
    end)

    it("counts a gated answer as a raw illegal one", function()
        -- Every position with a placement is one the default ranking
        -- reaches for the pass on, so a run that reported zero would mean
        -- the telemetry stopped watching.
        with_teacher(function(legal)
            return legal[1]
        end, function()
            local text = ask({ mode = "selfplay", games = 2, seed = 5 })
            expect(tonumber(text:match("illegal=(%d+)")) > 0).to.equal(true)
        end)
    end)

    it("replays the same batch from the same seed", function()
        with_teacher(function(legal)
            return legal[1]
        end, function()
            local first = ask({ mode = "selfplay", games = 2, seed = 5 })
            local second = ask({ mode = "selfplay", games = 2, seed = 5 })
            expect(first).to.equal(second)
        end)
    end)

    it("plays different batches from different seeds", function()
        -- Othello has one opening and both seats are deterministic here,
        -- so a batch that did not randomise its opening would be one game
        -- repeated whatever the seed.
        with_teacher(function(legal)
            return legal[1]
        end, function()
            local first = ask({ mode = "selfplay", games = 4, seed = 5 })
            local second = ask({ mode = "selfplay", games = 4, seed = 9 })
            expect(first ~= second).to.equal(true)
        end)
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            ask({ mode = "selfplay", games = 0, seed = 5 })
        end).to.fail()
    end)

    it("rejects a field another mode reads", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, state = OPENING })
        end).to.fail()
    end)
end)

-- ─── entry guards ───────────────────────────────────────────────────

describe("othello6_npc run", function()
    it("rejects an unknown mode", function()
        expect(function()
            ask({ mode = "rollout", state = OPENING })
        end).to.fail()
    end)

    it("rejects a task that is not a JSON string", function()
        expect(function()
            npc.run({ task = { mode = "decide" } })
        end).to.fail()
    end)

    it("rejects a task that decodes without a mode", function()
        expect(function()
            ask({ state = OPENING })
        end).to.fail()
    end)

    it("rejects a misspelled task field", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, stlye = "corner" })
        end).to.fail()
    end)
end)
