--- pick fixture rules tests (mlua-lspec)
---
--- Fences the pure-Lua half of the nn test fixture
--- (`tests/lua/fixtures/pick/init.lua`) on the default feature set, so
--- a regression in the fixture fails `cargo test` without needing the
--- `nn` feature or a trained model. The nn half (gate, hook, userdata
--- handle) is fenced by `tests/nn_gate_smoke.rs` and
--- `tests/nn_ckpt_hook_e2e.rs`.
---
--- The module only touches the host through `alc.math.rng_create` /
--- `alc.math.rng_int`, stubbed below with a plain LCG. The properties
--- under test hold for any RNG, so the stub does not weaken them.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ── 1. Package path: make the fixture requireable ─────────────
-- ALC_TEST_FIXTURES_DIR is set by the Rust harness to the absolute
-- `tests/lua/fixtures` directory. Lua tests MUST NOT guess a relative
-- path off the process CWD, which differs between `cargo test` and
-- IDE runners.
local fixtures_dir = os.getenv("ALC_TEST_FIXTURES_DIR") or ""
package.path = fixtures_dir .. "/?/init.lua;" .. package.path

-- ── 2. Stub alc.math (linear congruential generator) ─────────
alc = {}
alc.math = {
    rng_create = function(seed)
        return { state = math.floor(seed) % 2147483647 }
    end,
    rng_int = function(rng, min, max)
        rng.state = (rng.state * 1103515245 + 12345) % 2147483648
        local span = max - min + 1
        return min + (rng.state // 65536) % span
    end,
}

local pick = require("pick")

--- Play a full episode with the teacher on seat a and random on seat b,
--- returning the finished episode plus the per-turn trace.
local function playout(seed)
    local ep = pick.new_episode(seed)
    local rng = alc.math.rng_create(seed + 1)
    local trace = {}
    while not pick.is_over(ep) do
        local va = pick.teacher(ep.a)
        local vb = pick.random_policy(ep.b, rng)
        trace[#trace + 1] = string.format("%s>%d/%d", pick.encode(ep.a), va, vb)
        ep = pick.apply(ep, va, vb)
    end
    return ep, trace
end

local function contains(list, value)
    for _, v in ipairs(list) do
        if v == value then
            return true
        end
    end
    return false
end

describe("pick.legal", function()
    it("returns a deduplicated ascending subset of the pool", function()
        local legal = pick.legal({ pool = { 7, 3, 7, 1 } })
        expect(legal).to.equal({ 1, 3, 7 })
    end)

    it("loses exactly one value from the pool per turn", function()
        local ep = pick.new_episode(5)
        for turn = 1, pick.TURNS do
            expect(#ep.a.pool).to.equal(pick.TURNS - turn + 1)
            ep = pick.apply(ep, pick.teacher(ep.a), pick.teacher(ep.b))
        end
    end)
end)

describe("pick.apply", function()
    it("scores the higher value and records the other seat's value", function()
        local ep = pick.new_episode(11)
        local va, vb = ep.a.pool[#ep.a.pool], ep.b.pool[1]
        local nxt = pick.apply(ep, va, vb)
        if va > vb then
            expect(nxt.a.my_score).to.equal(1)
            expect(nxt.b.opp_score).to.equal(1)
        elseif vb > va then
            expect(nxt.b.my_score).to.equal(1)
        else
            expect(nxt.a.my_score + nxt.b.my_score).to.equal(0)
        end
        expect(nxt.a.seen).to.equal({ vb })
        expect(nxt.b.seen).to.equal({ va })
        expect(nxt.turn).to.equal(2)
    end)

    it("does not mutate the previous state", function()
        local ep = pick.new_episode(3)
        local before = table.concat(ep.a.pool, ",")
        pick.apply(ep, ep.a.pool[1], ep.b.pool[1])
        expect(table.concat(ep.a.pool, ",")).to.equal(before)
        expect(ep.turn).to.equal(1)
    end)

    it("rejects a value that is not in the pool", function()
        local ep = pick.new_episode(3)
        expect(function()
            pick.apply(ep, 0, ep.b.pool[1])
        end).to.fail()
    end)

    it("refuses to apply past the end", function()
        local ep = playout(9)
        expect(pick.is_over(ep)).to.be.truthy()
        expect(function()
            pick.apply(ep, 1, 1)
        end).to.fail()
    end)
end)

describe("pick.encode", function()
    it("is stable and ignores pool order", function()
        local s1 = { turn = 2, pool = { 1, 5, 3 }, my_score = 1, opp_score = 0, seen = { 4 } }
        local s2 = { turn = 2, pool = { 5, 3, 1 }, my_score = 1, opp_score = 0, seen = { 4 } }
        expect(pick.encode(s1)).to.equal("T2L135S10O4")
        expect(pick.encode(s2)).to.equal(pick.encode(s1))
    end)

    it("distinguishes the score gap", function()
        local level = { turn = 1, pool = { 2 }, my_score = 0, opp_score = 0, seen = {} }
        local ahead = { turn = 1, pool = { 2 }, my_score = 1, opp_score = 0, seen = {} }
        expect(pick.encode(level) ~= pick.encode(ahead)).to.be.truthy()
    end)

    it("keeps prompt plus value inside a 16-token context on every turn", function()
        local ep = pick.new_episode(21)
        while not pick.is_over(ep) do
            local line = pick.encode(ep.a) .. ">" .. pick.teacher(ep.a) .. "\n"
            expect(#pick.to_ids(line)).to.equal(15)
            ep = pick.apply(ep, pick.teacher(ep.a), pick.teacher(ep.b))
        end
    end)
end)

describe("pick.to_ids", function()
    it("maps every alphabet char inside the tiny vocabulary", function()
        local vocab = pick.vocab()
        expect(vocab.size <= 64).to.be.truthy()
        expect(vocab.pad_id).to.equal(0)
        for ch, id in pairs(vocab.to_id) do
            expect(pick.to_ids(ch)).to.equal({ id })
        end
    end)

    it("rejects a char outside the alphabet", function()
        expect(function()
            pick.to_ids("x")
        end).to.fail()
    end)
end)

describe("pick.teacher", function()
    it("commits the highest legal value when level or behind", function()
        expect(pick.teacher({ pool = { 3, 9, 5 }, my_score = 0, opp_score = 0 })).to.equal(9)
        expect(pick.teacher({ pool = { 3, 9, 5 }, my_score = 0, opp_score = 1 })).to.equal(9)
    end)

    it("commits the lowest legal value when ahead", function()
        expect(pick.teacher({ pool = { 3, 9, 5 }, my_score = 1, opp_score = 0 })).to.equal(3)
    end)

    it("is deterministic and legal over a whole playout", function()
        local _, t1 = playout(42)
        local _, t2 = playout(42)
        expect(t1).to.equal(t2)
        local ep = pick.new_episode(42)
        while not pick.is_over(ep) do
            local va = pick.teacher(ep.a)
            expect(contains(pick.legal(ep.a), va)).to.be.truthy()
            ep = pick.apply(ep, va, pick.teacher(ep.b))
        end
    end)
end)

describe("pick.build_corpus", function()
    it("emits ROWS_PER_EPISODE rows of ctx_len tokens per episode", function()
        local rows = pick.build_corpus(pick.teacher, { ctx_len = 16, episodes = 3, seed = 7 })
        expect(#rows).to.equal(3 * pick.ROWS_PER_EPISODE)
        for _, row in ipairs(rows) do
            expect(#row).to.equal(16)
            expect(row[16]).to.equal(0)
        end
    end)

    it("is reproducible for the same seed", function()
        local a = pick.build_corpus(pick.teacher, { ctx_len = 16, episodes = 2, seed = 7 })
        local b = pick.build_corpus(pick.teacher, { ctx_len = 16, episodes = 2, seed = 7 })
        expect(a).to.equal(b)
    end)

    it("rejects a context the line does not fit", function()
        expect(function()
            pick.build_corpus(pick.teacher, { ctx_len = 8, episodes = 1 })
        end).to.fail()
    end)
end)

describe("pick.legal_ids / check_states", function()
    it("maps each legal value to the token id of its digit", function()
        local by_id, values = pick.legal_ids({ pool = { 4, 4, 8 } })
        local vocab = pick.vocab()
        expect(values).to.equal({ 4, 8 })
        expect(by_id[vocab.to_id["4"]]).to.equal(4)
        expect(by_id[vocab.to_id["8"]]).to.equal(8)
    end)

    it("hands out fresh probe states covering the score-gap branches", function()
        local a, b = pick.check_states(), pick.check_states()
        expect(#a).to.equal(3)
        expect(a[1] ~= b[1]).to.be.truthy()
        expect(pick.teacher(a[1])).to.equal(9)
        expect(pick.teacher(a[2])).to.equal(1)
        expect(pick.teacher(a[3])).to.equal(8)
    end)
end)

describe("pick.require_handle", function()
    it("accepts a table with a callable generate_session", function()
        local fake = { generate_session = function() end }
        expect(pick.require_handle(fake)).to.equal(fake)
    end)

    it("rejects a table without one, and a non-table", function()
        expect(function()
            pick.require_handle({})
        end).to.fail()
        expect(function()
            pick.require_handle("alias")
        end).to.fail()
    end)
end)
