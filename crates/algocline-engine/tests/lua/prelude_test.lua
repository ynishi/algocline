--- prelude.lua combinator tests (mlua-lspec)
---
--- Unit tests for alc.map, alc.reduce, alc.vote, alc.filter.
--- prelude.lua expects the alc global table to exist, so we create
--- an empty table before loading it.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- Setup: create alc table and load prelude
alc = {}
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
