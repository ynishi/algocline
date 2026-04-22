--- tools.docs.json — minimal pure-Lua JSON decoder.
---
--- Scope: `decode` only. The bundled-packages docs pipeline reads
--- `hub_index.json` (produced by algocline's `alc_hub_reindex`) as
--- the single source of truth for package enumeration; it never
--- writes JSON back. An encoder is deliberately not provided.
---
--- Subset: RFC 8259 — object, array, string (with `\uXXXX` →
--- UTF-8 transcoding), number, `true`, `false`, `null`. `null` is
--- decoded to the sentinel `M.NULL` (distinct from Lua `nil` so
--- it survives object-key existence checks).
---
--- Error messages include the byte offset where parsing failed.

local M = {}
M.NULL = setmetatable({}, { __tostring = function() return "null" end })

local decode_value -- forward declaration

local function skip_ws(s, i)
    while i <= #s do
        local c = s:byte(i)
        if c == 32 or c == 9 or c == 10 or c == 13 then
            i = i + 1
        else
            return i
        end
    end
    return i
end

local function codepoint_to_utf8(code)
    if code < 0x80 then
        return string.char(code)
    elseif code < 0x800 then
        return string.char(
            0xC0 + (code >> 6),
            0x80 + (code & 0x3F))
    elseif code < 0x10000 then
        return string.char(
            0xE0 + (code >> 12),
            0x80 + ((code >> 6) & 0x3F),
            0x80 + (code & 0x3F))
    else
        return string.char(
            0xF0 + (code >> 18),
            0x80 + ((code >> 12) & 0x3F),
            0x80 + ((code >> 6) & 0x3F),
            0x80 + (code & 0x3F))
    end
end

local function decode_string(s, i)
    -- precondition: s:sub(i, i) == '"'
    local out = {}
    i = i + 1
    while i <= #s do
        local c = s:sub(i, i)
        if c == '"' then
            return table.concat(out), i + 1
        elseif c == "\\" then
            local nxt = s:sub(i + 1, i + 1)
            if nxt == '"' or nxt == "\\" or nxt == "/" then
                out[#out + 1] = nxt
                i = i + 2
            elseif nxt == "n" then out[#out + 1] = "\n"; i = i + 2
            elseif nxt == "t" then out[#out + 1] = "\t"; i = i + 2
            elseif nxt == "r" then out[#out + 1] = "\r"; i = i + 2
            elseif nxt == "b" then out[#out + 1] = "\b"; i = i + 2
            elseif nxt == "f" then out[#out + 1] = "\f"; i = i + 2
            elseif nxt == "u" then
                local hex = s:sub(i + 2, i + 5)
                if #hex ~= 4 or not hex:match("^%x%x%x%x$") then
                    error(string.format(
                        "json: bad \\u escape at byte %d", i), 0)
                end
                local code = tonumber(hex, 16)
                -- basic surrogate pair support (UTF-16 high/low)
                if code >= 0xD800 and code <= 0xDBFF then
                    if s:sub(i + 6, i + 7) ~= "\\u" then
                        error(string.format(
                            "json: unpaired high surrogate at byte %d", i), 0)
                    end
                    local lo_hex = s:sub(i + 8, i + 11)
                    if #lo_hex ~= 4 or not lo_hex:match("^%x%x%x%x$") then
                        error(string.format(
                            "json: bad low surrogate at byte %d", i + 6), 0)
                    end
                    local lo = tonumber(lo_hex, 16)
                    code = 0x10000 + ((code - 0xD800) << 10) + (lo - 0xDC00)
                    i = i + 12
                else
                    i = i + 6
                end
                out[#out + 1] = codepoint_to_utf8(code)
            else
                error(string.format(
                    "json: bad escape \\%s at byte %d", nxt, i), 0)
            end
        else
            out[#out + 1] = c
            i = i + 1
        end
    end
    error("json: unterminated string", 0)
end

local function decode_number(s, i)
    local j = i
    if s:sub(j, j) == "-" then j = j + 1 end
    while j <= #s and s:sub(j, j):match("[%d.eE+-]") do j = j + 1 end
    local n = tonumber(s:sub(i, j - 1))
    if not n then
        error(string.format("json: bad number at byte %d", i), 0)
    end
    return n, j
end

local function decode_array(s, i)
    -- precondition: s:sub(i, i) == '['
    local arr = {}
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "]" then return arr, i + 1 end
    while i <= #s do
        local v
        v, i = decode_value(s, i)
        arr[#arr + 1] = v
        i = skip_ws(s, i)
        local c = s:sub(i, i)
        if c == "," then
            i = skip_ws(s, i + 1)
        elseif c == "]" then
            return arr, i + 1
        else
            error(string.format(
                "json: expected ',' or ']' at byte %d, got %q", i, c), 0)
        end
    end
    error("json: unterminated array", 0)
end

local function decode_object(s, i)
    -- precondition: s:sub(i, i) == '{'
    local obj = {}
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "}" then return obj, i + 1 end
    while i <= #s do
        if s:sub(i, i) ~= '"' then
            error(string.format(
                "json: expected string key at byte %d", i), 0)
        end
        local key
        key, i = decode_string(s, i)
        i = skip_ws(s, i)
        if s:sub(i, i) ~= ":" then
            error(string.format(
                "json: expected ':' after key at byte %d", i), 0)
        end
        i = skip_ws(s, i + 1)
        local val
        val, i = decode_value(s, i)
        obj[key] = val
        i = skip_ws(s, i)
        local c = s:sub(i, i)
        if c == "," then
            i = skip_ws(s, i + 1)
        elseif c == "}" then
            return obj, i + 1
        else
            error(string.format(
                "json: expected ',' or '}' at byte %d, got %q", i, c), 0)
        end
    end
    error("json: unterminated object", 0)
end

decode_value = function(s, i)
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if     c == '"' then return decode_string(s, i)
    elseif c == "{" then return decode_object(s, i)
    elseif c == "[" then return decode_array(s, i)
    elseif c == "t" then
        if s:sub(i, i + 3) == "true" then return true, i + 4 end
        error(string.format("json: expected 'true' at byte %d", i), 0)
    elseif c == "f" then
        if s:sub(i, i + 4) == "false" then return false, i + 5 end
        error(string.format("json: expected 'false' at byte %d", i), 0)
    elseif c == "n" then
        if s:sub(i, i + 3) == "null" then return M.NULL, i + 4 end
        error(string.format("json: expected 'null' at byte %d", i), 0)
    elseif c == "-" or (c >= "0" and c <= "9") then
        return decode_number(s, i)
    else
        error(string.format(
            "json: unexpected char %q at byte %d", c, i), 0)
    end
end

--- Decode a JSON document. Returns the parsed Lua value.
--- `null` decodes to `M.NULL` (sentinel table, not `nil`).
--- Errors on malformed input, trailing garbage, or empty input.
function M.decode(s)
    if type(s) ~= "string" then
        error("json.decode: input must be string", 0)
    end
    if #s == 0 then
        error("json.decode: empty input", 0)
    end
    local v, i = decode_value(s, 1)
    i = skip_ws(s, i)
    if i <= #s then
        error(string.format(
            "json.decode: trailing garbage at byte %d", i), 0)
    end
    return v
end

return M
