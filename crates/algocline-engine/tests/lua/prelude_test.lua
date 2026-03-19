--- prelude.lua combinator tests (mlua-lspec)
---
--- Unit tests for alc.map, alc.reduce, alc.vote, alc.filter.
--- prelude.lua expects the alc global table to exist, so we create
--- an empty table before loading it.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- Setup: create alc table with stubs and load prelude
alc = {}

-- Stub alc.json_decode / alc.json_encode for json_extract tests
alc.json_decode = function(str)
    -- Minimal JSON parser stub: delegates to a simple pattern-based approach
    -- For test purposes, we use load() to parse JSON-like Lua tables
    -- In production, the Rust bridge provides real JSON parsing
    local fn, err = load("return " .. str:gsub("%[", "{"):gsub("%]", "}"):gsub('"(%w+)"%s*:', '["%1"]='))
    if fn then
        local ok, result = pcall(fn)
        if ok then return result end
    end
    error("json decode error: " .. tostring(err))
end

alc.json_encode = function(val)
    if type(val) == "table" then return "{}" end
    return tostring(val)
end

-- Stub alc.log for llm_safe tests
local log_entries = {}
alc.log = function(level, msg)
    log_entries[#log_entries + 1] = { level = level, msg = msg }
end

-- Stub alc.state for state.update tests
local state_store = {}
alc.state = {
    get = function(key, default)
        local val = state_store[key]
        if val == nil then return default end
        return val
    end,
    set = function(key, value)
        state_store[key] = value
    end,
}

local prelude_path = os.getenv("PRELUDE_PATH")
    or "crates/algocline-engine/src/prelude.lua"
dofile(prelude_path)

-- ─── alc.map ───

describe("alc.map", function()
    it("maps over empty array", function()
        local result = alc.map({}, function(x) return x * 2 end)
        expect(#result).to.equal(0)
    end)

    it("transforms each element", function()
        local result = alc.map({1, 2, 3}, function(x) return x * 10 end)
        expect(result[1]).to.equal(10)
        expect(result[2]).to.equal(20)
        expect(result[3]).to.equal(30)
    end)

    it("passes index as second argument", function()
        local indices = {}
        alc.map({"a", "b", "c"}, function(_, i)
            indices[#indices + 1] = i
        end)
        expect(indices[1]).to.equal(1)
        expect(indices[2]).to.equal(2)
        expect(indices[3]).to.equal(3)
    end)

    it("preserves order", function()
        local result = alc.map({"x", "y", "z"}, function(v) return v .. "!" end)
        expect(result[1]).to.equal("x!")
        expect(result[2]).to.equal("y!")
        expect(result[3]).to.equal("z!")
    end)

    it("handles single element", function()
        local result = alc.map({42}, function(x) return x + 1 end)
        expect(#result).to.equal(1)
        expect(result[1]).to.equal(43)
    end)
end)

-- ─── alc.reduce ───

describe("alc.reduce", function()
    it("reduces with init value", function()
        local result = alc.reduce({1, 2, 3}, function(acc, x) return acc + x end, 0)
        expect(result).to.equal(6)
    end)

    it("reduces without init (uses first element)", function()
        local result = alc.reduce({10, 20, 30}, function(acc, x) return acc + x end)
        expect(result).to.equal(60)
    end)

    it("single element without init returns it", function()
        local result = alc.reduce({42}, function(acc, x) return acc + x end)
        expect(result).to.equal(42)
    end)

    it("single element with init applies fn once", function()
        local result = alc.reduce({5}, function(acc, x) return acc * x end, 10)
        expect(result).to.equal(50)
    end)

    it("passes index as third argument", function()
        local indices = {}
        alc.reduce({1, 2, 3}, function(acc, x, i)
            indices[#indices + 1] = i
            return acc + x
        end, 0)
        -- with init, iterates from index 1
        expect(indices[1]).to.equal(1)
        expect(indices[2]).to.equal(2)
        expect(indices[3]).to.equal(3)
    end)

    it("passes index starting at 2 without init", function()
        local indices = {}
        alc.reduce({10, 20, 30}, function(acc, x, i)
            indices[#indices + 1] = i
            return acc + x
        end)
        -- without init, starts from index 2
        expect(indices[1]).to.equal(2)
        expect(indices[2]).to.equal(3)
    end)

    it("string concatenation", function()
        local result = alc.reduce({"a", "b", "c"}, function(acc, x)
            return acc .. x
        end)
        expect(result).to.equal("abc")
    end)
end)

-- ─── alc.vote ───

describe("alc.vote", function()
    it("majority wins", function()
        local result = alc.vote({"yes", "yes", "no", "yes"})
        expect(result.winner).to.equal("yes")
        expect(result.count).to.equal(3)
        expect(result.total).to.equal(4)
    end)

    it("single answer", function()
        local result = alc.vote({"only"})
        expect(result.winner).to.equal("only")
        expect(result.count).to.equal(1)
        expect(result.total).to.equal(1)
    end)

    it("tie returns first seen", function()
        -- "a" appears 2x, "b" appears 2x -> "a" wins (first seen)
        local result = alc.vote({"a", "b", "a", "b"})
        expect(result.winner).to.equal("a")
        expect(result.count).to.equal(2)
        expect(result.total).to.equal(4)
    end)

    it("trims whitespace", function()
        local result = alc.vote({"  yes ", "yes", " yes"})
        expect(result.winner).to.equal("yes")
        expect(result.count).to.equal(3)
    end)

    it("all different returns first", function()
        local result = alc.vote({"a", "b", "c"})
        expect(result.winner).to.equal("a")
        expect(result.count).to.equal(1)
        expect(result.total).to.equal(3)
    end)

    it("converts non-string to string via tostring", function()
        local result = alc.vote({1, 1, 2})
        expect(result.winner).to.equal("1")
        expect(result.count).to.equal(2)
    end)
end)

-- ─── alc.filter ───

describe("alc.filter", function()
    it("filters empty array", function()
        local result = alc.filter({}, function() return true end)
        expect(#result).to.equal(0)
    end)

    it("keeps matching elements", function()
        local result = alc.filter({1, 2, 3, 4, 5}, function(x) return x > 3 end)
        expect(#result).to.equal(2)
        expect(result[1]).to.equal(4)
        expect(result[2]).to.equal(5)
    end)

    it("removes all when predicate is false", function()
        local result = alc.filter({1, 2, 3}, function() return false end)
        expect(#result).to.equal(0)
    end)

    it("keeps all when predicate is true", function()
        local result = alc.filter({1, 2, 3}, function() return true end)
        expect(#result).to.equal(3)
    end)

    it("passes index as second argument", function()
        local result = alc.filter({"a", "b", "c", "d"}, function(_, i)
            return i % 2 == 0
        end)
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("b")
        expect(result[2]).to.equal("d")
    end)

    it("preserves order of kept elements", function()
        local result = alc.filter({5, 3, 8, 1, 9}, function(x) return x > 4 end)
        expect(result[1]).to.equal(5)
        expect(result[2]).to.equal(8)
        expect(result[3]).to.equal(9)
    end)
end)

-- ─── alc.json_extract ───

describe("alc.json_extract", function()
    -- Save original and install real json_decode for these tests
    local real_decode = alc.json_decode

    it("returns nil for non-string input", function()
        expect(alc.json_extract(nil)).to.equal(nil)
        expect(alc.json_extract(42)).to.equal(nil)
        expect(alc.json_extract(true)).to.equal(nil)
    end)

    it("parses raw JSON directly", function()
        -- Use a simple stub that handles this exact input
        alc.json_decode = function(s)
            if s == '{"a": 1}' then return { a = 1 } end
            error("decode error")
        end
        local result = alc.json_extract('{"a": 1}')
        expect(type(result)).to.equal("table")
        expect(result.a).to.equal(1)
        alc.json_decode = real_decode
    end)

    it("extracts from markdown json fence", function()
        alc.json_decode = function(s)
            if s == '{"ok": true}' then return { ok = true } end
            error("decode error")
        end
        local input = 'Here is the result:\n```json\n{"ok": true}\n```\nDone.'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result.ok).to.equal(true)
        alc.json_decode = real_decode
    end)

    it("extracts from plain markdown fence", function()
        alc.json_decode = function(s)
            if s == '{"x": 2}' then return { x = 2 } end
            error("decode error")
        end
        local input = '```\n{"x": 2}\n```'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result.x).to.equal(2)
        alc.json_decode = real_decode
    end)

    it("extracts embedded JSON via balanced braces", function()
        alc.json_decode = function(s)
            if s == '{"b": 3}' then return { b = 3 } end
            error("decode error")
        end
        local input = 'The answer is {"b": 3} as shown.'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result.b).to.equal(3)
        alc.json_decode = real_decode
    end)

    it("extracts JSON array via balanced brackets", function()
        alc.json_decode = function(s)
            if s == '[1, 2, 3]' then return { 1, 2, 3 } end
            error("decode error")
        end
        local input = 'Result: [1, 2, 3].'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result[1]).to.equal(1)
        expect(result[3]).to.equal(3)
        alc.json_decode = real_decode
    end)

    it("returns nil when no JSON found", function()
        alc.json_decode = function(s) error("decode error") end
        local result = alc.json_extract("no json here at all")
        expect(result).to.equal(nil)
        alc.json_decode = real_decode
    end)

    it("returns nil for empty string", function()
        alc.json_decode = function(s) error("decode error") end
        local result = alc.json_extract("")
        expect(result).to.equal(nil)
        alc.json_decode = real_decode
    end)

    it("returns nil when json_decode returns non-table", function()
        alc.json_decode = function(s) return 42 end
        local result = alc.json_extract("42")
        expect(result).to.equal(nil)
        alc.json_decode = real_decode
    end)

    it("skips non-JSON balanced braces and finds valid JSON", function()
        alc.json_decode = function(s)
            if s == '{"real": true}' then return { real = true } end
            error("decode error")
        end
        local input = 'text {not json} then {"real": true} end'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result.real).to.equal(true)
        alc.json_decode = real_decode
    end)

    it("skips non-JSON balanced brackets and finds valid array", function()
        alc.json_decode = function(s)
            if s == '[1, 2]' then return { 1, 2 } end
            error("decode error")
        end
        local input = 'see [broken then [1, 2] done'
        local result = alc.json_extract(input)
        expect(type(result)).to.equal("table")
        expect(result[1]).to.equal(1)
        alc.json_decode = real_decode
    end)
end)

-- ─── alc.state.update ───

describe("alc.state.update", function()
    it("creates new key with default", function()
        state_store = {}
        local result = alc.state.update("counter", function(n) return n + 1 end, 0)
        expect(result).to.equal(1)
        expect(state_store["counter"]).to.equal(1)
    end)

    it("updates existing key", function()
        state_store = { counter = 5 }
        local result = alc.state.update("counter", function(n) return n + 10 end, 0)
        expect(result).to.equal(15)
        expect(state_store["counter"]).to.equal(15)
    end)

    it("works with table values", function()
        state_store = {}
        alc.state.update("list", function(t)
            t[#t + 1] = "item1"
            return t
        end, {})
        alc.state.update("list", function(t)
            t[#t + 1] = "item2"
            return t
        end, {})
        expect(#state_store["list"]).to.equal(2)
        expect(state_store["list"][1]).to.equal("item1")
        expect(state_store["list"][2]).to.equal("item2")
    end)

    it("returns updated value", function()
        state_store = {}
        local result = alc.state.update("x", function() return "hello" end, nil)
        expect(result).to.equal("hello")
    end)

    it("default is nil when omitted", function()
        state_store = {}
        local result = alc.state.update("missing", function(v)
            if v == nil then return "was nil" end
            return v
        end)
        expect(result).to.equal("was nil")
    end)
end)

-- ─── alc.llm_safe ───

describe("alc.llm_safe", function()
    it("returns LLM result on success", function()
        alc.llm = function(prompt, opts) return "response" end
        local result = alc.llm_safe("test", {}, "fallback")
        expect(result).to.equal("response")
    end)

    it("returns default on LLM failure", function()
        alc.llm = function() error("network error") end
        log_entries = {}
        local result = alc.llm_safe("test", {}, "fallback")
        expect(result).to.equal("fallback")
    end)

    it("logs warning on failure", function()
        alc.llm = function() error("timeout") end
        log_entries = {}
        alc.llm_safe("test", {}, "default")
        expect(#log_entries).to.equal(1)
        expect(log_entries[1].level).to.equal("warn")
    end)

    it("returns nil default when not specified", function()
        alc.llm = function() error("fail") end
        log_entries = {}
        local result = alc.llm_safe("test", {})
        expect(result).to.equal(nil)
    end)

    it("passes opts to alc.llm", function()
        local captured_opts
        alc.llm = function(prompt, opts)
            captured_opts = opts
            return "ok"
        end
        alc.llm_safe("test", { max_tokens = 100 }, "default")
        expect(captured_opts.max_tokens).to.equal(100)
    end)
end)

-- ─── alc.fingerprint ───

describe("alc.fingerprint", function()
    it("returns 8-char hex string", function()
        local fp = alc.fingerprint("hello")
        expect(#fp).to.equal(8)
        expect(fp:match("^%x+$")).to_not.equal(nil)
    end)

    it("normalizes whitespace", function()
        local fp1 = alc.fingerprint("hello   world")
        local fp2 = alc.fingerprint("hello world")
        expect(fp1).to.equal(fp2)
    end)

    it("normalizes case", function()
        local fp1 = alc.fingerprint("Hello World")
        local fp2 = alc.fingerprint("hello world")
        expect(fp1).to.equal(fp2)
    end)

    it("trims leading/trailing whitespace", function()
        local fp1 = alc.fingerprint("  hello  ")
        local fp2 = alc.fingerprint("hello")
        expect(fp1).to.equal(fp2)
    end)

    it("different inputs produce different hashes", function()
        local fp1 = alc.fingerprint("hello")
        local fp2 = alc.fingerprint("world")
        expect(fp1).to_not.equal(fp2)
    end)

    it("same input produces same hash", function()
        local fp1 = alc.fingerprint("test string")
        local fp2 = alc.fingerprint("test string")
        expect(fp1).to.equal(fp2)
    end)

    it("handles empty string", function()
        local fp = alc.fingerprint("")
        expect(#fp).to.equal(8)
    end)

    it("converts non-string via tostring", function()
        local fp1 = alc.fingerprint(42)
        local fp2 = alc.fingerprint("42")
        expect(fp1).to.equal(fp2)
    end)

    it("comprehensive normalization", function()
        local fp1 = alc.fingerprint("  Fix  the  Login   Bug  ")
        local fp2 = alc.fingerprint("fix the login bug")
        expect(fp1).to.equal(fp2)
    end)
end)

-- ─── alc.tuning ───

describe("alc.tuning", function()
    it("returns defaults when ctx is empty", function()
        local defaults = { threshold = 3.0, name = "test" }
        local cfg = alc.tuning(defaults, {})
        expect(cfg.threshold).to.equal(3.0)
        expect(cfg.name).to.equal("test")
    end)

    it("overrides scalar with ctx value", function()
        local defaults = { threshold = 3.0, rounds = 5 }
        local ctx = { threshold = 4.5 }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.threshold).to.equal(4.5)
        expect(cfg.rounds).to.equal(5)
    end)

    it("deep merges dict-like nested tables", function()
        local defaults = { exponents = { alpha = 1.0, beta = 1.0, gamma = 2.0 } }
        local ctx = { exponents = { alpha = 2.5 } }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.exponents.alpha).to.equal(2.5)
        expect(cfg.exponents.beta).to.equal(1.0)
        expect(cfg.exponents.gamma).to.equal(2.0)
    end)

    it("shallow replaces array-like tables", function()
        local defaults = { gates = { {min = 3}, {min = 5} } }
        local ctx = { gates = { {min = 7} } }
        local cfg = alc.tuning(defaults, ctx)
        expect(#cfg.gates).to.equal(1)
        expect(cfg.gates[1].min).to.equal(7)
    end)

    it("strips _schema key", function()
        local defaults = {
            threshold = 3.0,
            _schema = { threshold = { type = "number", range = {1, 10} } },
        }
        local cfg = alc.tuning(defaults, {})
        expect(cfg.threshold).to.equal(3.0)
        expect(cfg._schema).to.equal(nil)
    end)

    it("supports prefix namespace", function()
        local defaults = { threshold = 3.0, rounds = 5 }
        local ctx = { biz = { threshold = 6.0 } }
        local cfg = alc.tuning(defaults, ctx, { prefix = "biz" })
        expect(cfg.threshold).to.equal(6.0)
        expect(cfg.rounds).to.equal(5)
    end)

    it("prefix ignores top-level ctx keys", function()
        local defaults = { threshold = 3.0 }
        local ctx = { threshold = 99, biz = {} }
        local cfg = alc.tuning(defaults, ctx, { prefix = "biz" })
        expect(cfg.threshold).to.equal(3.0)  -- not 99
    end)

    it("handles nil ctx gracefully", function()
        local defaults = { x = 1 }
        local cfg = alc.tuning(defaults, nil)
        expect(cfg.x).to.equal(1)
    end)

    it("handles non-table defaults", function()
        local result = alc.tuning("not a table", {})
        expect(result).to.equal("not a table")
    end)

    it("ctx false value overrides default", function()
        local defaults = { enabled = true }
        local ctx = { enabled = false }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.enabled).to.equal(false)
    end)

    it("does not include ctx keys absent from defaults", function()
        local defaults = { a = 1 }
        local ctx = { a = 2, extra = "leak" }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.a).to.equal(2)
        expect(cfg.extra).to.equal(nil)
    end)

    it("deep merge two levels", function()
        local defaults = {
            outer = { inner = { x = 1, y = 2 }, keep = "yes" },
        }
        local ctx = { outer = { inner = { x = 10 } } }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.outer.inner.x).to.equal(10)
        expect(cfg.outer.inner.y).to.equal(2)
        expect(cfg.outer.keep).to.equal("yes")
    end)

    it("ctx can change type of value", function()
        local defaults = { mode = "auto" }
        local ctx = { mode = 42 }
        local cfg = alc.tuning(defaults, ctx)
        expect(cfg.mode).to.equal(42)
    end)

    it("prefix with non-table value logs warning and uses defaults", function()
        log_entries = {}
        local defaults = { threshold = 3.0 }
        local ctx = { biz = "not a table", threshold = 99 }
        local cfg = alc.tuning(defaults, ctx, { prefix = "biz" })
        expect(cfg.threshold).to.equal(3.0)  -- defaults, not 99
        expect(#log_entries).to.equal(1)
        expect(log_entries[1].level).to.equal("warn")
    end)
end)
