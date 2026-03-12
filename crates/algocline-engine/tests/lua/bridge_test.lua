--- bridge.rs Lua integration tests (mlua-lspec)
---
--- Interface tests for alc.chunk, alc.json_encode/decode.
--- These functions are registered by Rust (bridge.rs), so they are not
--- available in a standalone Lua VM. Tests use pure-Lua mock
--- implementations that mirror the Rust behavior.
---
--- Purpose: verify the API contract (argument format, return structure)
--- as seen from the Lua side.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Mock setup ───
-- Pure-Lua reimplementations of register_json/register_chunk from bridge.rs.
-- Tests verify the API contract, not the Rust implementation details.

alc = {}

-- Mock alc.json_encode: Lua value -> JSON string
-- Minimal implementation for environments without cjson/dkjson
local function simple_json_encode(value)
    if type(value) == "nil" then return "null" end
    if type(value) == "boolean" then return tostring(value) end
    if type(value) == "number" then return tostring(value) end
    if type(value) == "string" then
        return '"' .. value:gsub('\\', '\\\\'):gsub('"', '\\"') .. '"'
    end
    if type(value) == "table" then
        -- Detect array vs object
        local is_array = true
        local max_idx = 0
        for k, _ in pairs(value) do
            if type(k) == "number" and k == math.floor(k) and k > 0 then
                if k > max_idx then max_idx = k end
            else
                is_array = false
                break
            end
        end
        if is_array and max_idx == #value then
            local parts = {}
            for i = 1, #value do
                parts[i] = simple_json_encode(value[i])
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            for k, v in pairs(value) do
                parts[#parts + 1] = simple_json_encode(tostring(k)) .. ":" .. simple_json_encode(v)
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

-- Mock alc.json_decode: JSON string -> Lua value (primitive types only)
local function simple_json_decode(s)
    if s == "null" then return nil end
    if s == "true" then return true end
    if s == "false" then return false end
    local num = tonumber(s)
    if num then return num end
    local str = s:match('^"(.*)"$')
    if str then return str end
    return s
end

alc.json_encode = simple_json_encode
alc.json_decode = simple_json_decode

-- Mock alc.chunk: mirrors Rust chunk_by_lines/chunk_by_chars logic
function alc.chunk(text, opts)
    opts = opts or {}
    local mode = opts.mode or "lines"
    local size = opts.size or 50
    local overlap = opts.overlap or 0

    if size == 0 then return {} end

    if mode == "chars" then
        local chars = {}
        for c in text:gmatch(".") do
            chars[#chars + 1] = c
        end
        if #chars == 0 then return {} end
        local step = (overlap < size) and (size - overlap) or 1
        local chunks = {}
        local i = 1
        while i <= #chars do
            local e = math.min(i + size - 1, #chars)
            local chunk = table.concat(chars, "", i, e)
            chunks[#chunks + 1] = chunk
            i = i + step
            if e == #chars then break end
        end
        return chunks
    else
        local lines = {}
        for line in text:gmatch("([^\n]*)\n?") do
            if line ~= "" or #lines > 0 then
                lines[#lines + 1] = line
            end
        end
        -- Remove trailing empty element from gmatch
        if #lines > 0 and lines[#lines] == "" then
            table.remove(lines)
        end
        if #lines == 0 then return {} end
        local step = (overlap < size) and (size - overlap) or 1
        local chunks = {}
        local i = 1
        while i <= #lines do
            local e = math.min(i + size - 1, #lines)
            local chunk = table.concat(lines, "\n", i, e)
            chunks[#chunks + 1] = chunk
            i = i + step
            if e == #lines then break end
        end
        return chunks
    end
end

-- ─── alc.json_encode tests ───

describe("alc.json_encode", function()
    it("encodes number", function()
        expect(alc.json_encode(42)).to.equal("42")
    end)

    it("encodes string", function()
        expect(alc.json_encode("hello")).to.equal('"hello"')
    end)

    it("encodes boolean", function()
        expect(alc.json_encode(true)).to.equal("true")
        expect(alc.json_encode(false)).to.equal("false")
    end)

    it("encodes nil as null", function()
        expect(alc.json_encode(nil)).to.equal("null")
    end)

    it("encodes array", function()
        local result = alc.json_encode({1, 2, 3})
        expect(result).to.equal("[1,2,3]")
    end)

    it("encodes string with escaped quotes", function()
        local result = alc.json_encode('say "hi"')
        expect(result).to.equal('"say \\"hi\\""')
    end)
end)

-- ─── alc.json_decode tests ───

describe("alc.json_decode", function()
    it("decodes number", function()
        expect(alc.json_decode("42")).to.equal(42)
    end)

    it("decodes string", function()
        expect(alc.json_decode('"hello"')).to.equal("hello")
    end)

    it("decodes boolean", function()
        expect(alc.json_decode("true")).to.equal(true)
        expect(alc.json_decode("false")).to.equal(false)
    end)

    it("decodes null", function()
        expect(alc.json_decode("null")).to.equal(nil)
    end)
end)

-- ─── alc.chunk tests ───

describe("alc.chunk (lines mode)", function()
    it("defaults to lines mode", function()
        local result = alc.chunk("a\nb\nc", { size = 2 })
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("a\nb")
        expect(result[2]).to.equal("c")
    end)

    it("returns empty for empty text", function()
        local result = alc.chunk("", { size = 5 })
        expect(#result).to.equal(0)
    end)

    it("handles size larger than lines", function()
        local result = alc.chunk("a\nb", { size = 100 })
        expect(#result).to.equal(1)
        expect(result[1]).to.equal("a\nb")
    end)

    it("supports overlap", function()
        local result = alc.chunk("a\nb\nc\nd\ne", { size = 3, overlap = 1 })
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("a\nb\nc")
        expect(result[2]).to.equal("c\nd\ne")
    end)

    it("size zero returns empty", function()
        local result = alc.chunk("a\nb\nc", { size = 0 })
        expect(#result).to.equal(0)
    end)
end)

describe("alc.chunk (chars mode)", function()
    it("splits by characters", function()
        local result = alc.chunk("abcdef", { mode = "chars", size = 3 })
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("abc")
        expect(result[2]).to.equal("def")
    end)

    it("handles remainder", function()
        local result = alc.chunk("abcde", { mode = "chars", size = 3 })
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("abc")
        expect(result[2]).to.equal("de")
    end)

    it("supports overlap", function()
        local result = alc.chunk("abcdef", { mode = "chars", size = 4, overlap = 2 })
        expect(#result).to.equal(2)
        expect(result[1]).to.equal("abcd")
        expect(result[2]).to.equal("cdef")
    end)

    it("size zero returns empty", function()
        local result = alc.chunk("abc", { mode = "chars", size = 0 })
        expect(#result).to.equal(0)
    end)

    it("empty text returns empty", function()
        local result = alc.chunk("", { mode = "chars", size = 5 })
        expect(#result).to.equal(0)
    end)
end)

-- ─── alc.chunk + alc.map integration ───

describe("chunk + map integration", function()
    -- Load prelude for alc.map/reduce
    local prelude_path = os.getenv("PRELUDE_PATH")
        or "crates/algocline-engine/src/prelude.lua"
    dofile(prelude_path)

    it("chunk then map over chunks", function()
        local chunks = alc.chunk("line1\nline2\nline3\nline4", { size = 2 })
        local uppered = alc.map(chunks, function(c) return c:upper() end)
        expect(#uppered).to.equal(2)
        expect(uppered[1]).to.equal("LINE1\nLINE2")
        expect(uppered[2]).to.equal("LINE3\nLINE4")
    end)

    it("chunk then reduce to concatenate", function()
        local chunks = alc.chunk("abc", { mode = "chars", size = 1 })
        local result = alc.reduce(chunks, function(acc, c) return acc .. "-" .. c end)
        expect(result).to.equal("a-b-c")
    end)
end)
