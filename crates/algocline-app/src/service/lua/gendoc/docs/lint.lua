--- tools.docs.lint — V0 convention gate.
---
--- Given a `PkgInfo` + the raw docstring, enumerates violations of the
--- V0 convention (`docstring-convention.md`). Returns a list of
--- `{severity, code, msg, line?}` records.
---
--- Severities:
---   "error"   — generator MUST reject the pkg in `--strict` mode.
---   "warning" — surfaces to stderr; does not block output.
---
--- Rules:
---   E_H1_IN_DOCSTRING    : line begins with `# ` (H1 is reserved for
---                          the generator, which synthesizes it from
---                          `narrative.title`).
---   E_META_MISSING_*     : required `meta.{name,version,description,
---                          category}` field absent or empty.
---   E_NAME_MISMATCH      : `meta.name` ≠ pkg directory name.
---   E_PARAMETERS_CONFLICT: `spec.entries.run.input` declared AND the
---                          docstring already contains a `## Parameters`
---                          section (shape is the SSoT).
---   E_RESULT_CONFLICT    : `spec.entries.run.result` declared AND the
---                          docstring already contains a `## Result`
---                          section (shape is the SSoT).
---   W_FAKE_LABEL         : a line like `Usage:` / `Args:` appears at
---                          column 0 outside a `## ` heading — common
---                          pre-V0 shape that should be promoted to H2.
---   W_EMPTY_NARRATIVE    : no summary AND no sections.
---   W_DESCRIPTION_MULTILINE: `meta.description` contains a newline.

local M = {}

-- ── rule helpers ──────────────────────────────────────────────────────

local FAKE_LABELS = {
    ["Usage"]       = true,
    ["Args"]        = true,
    ["Arguments"]   = true,
    ["Parameters"]  = true,
    ["Params"]      = true,
    ["Returns"]     = true,
    ["Return"]      = true,
    ["Example"]     = true,
    ["Examples"]    = true,
    ["Notes"]       = true,
    ["Note"]        = true,
    ["See"]         = true,
    ["See also"]    = true,
}

local function split_lines(s)
    local lines = {}
    local i = 1
    local len = #s
    while i <= len do
        local j = s:find("\n", i, true)
        if not j then
            lines[#lines + 1] = s:sub(i)
            break
        end
        lines[#lines + 1] = s:sub(i, j - 1)
        i = j + 1
    end
    return lines
end

local function has_parameters_section(sections)
    for i = 1, #sections do
        if sections[i].heading == "Parameters" then return true end
    end
    return false
end

local function has_result_section(sections)
    for i = 1, #sections do
        if sections[i].heading == "Result" then return true end
    end
    return false
end

-- ── public API ────────────────────────────────────────────────────────

--- Check a PkgInfo + raw docstring against the V0 convention.
---
--- Args:
---   pkg_info   : PkgInfo (see tools.docs.pkg_info)
---   docstring  : string — raw docstring text (no `---` prefix)
---   pkg_dir    : string — directory name of the pkg (for NAME_MISMATCH)
---
--- Returns: { violations = { {severity, code, msg, line?}, ... } }
function M.check(pkg_info, docstring, pkg_dir)
    local out = {}
    local id  = pkg_info.identity
    local nar = pkg_info.narrative
    local shp = pkg_info.shape

    -- meta completeness
    if not id.name or id.name == "" then
        out[#out + 1] = {
            severity = "error", code = "E_META_MISSING_NAME",
            msg = "meta.name is required" }
    end
    if not id.version or id.version == "" then
        out[#out + 1] = {
            severity = "error", code = "E_META_MISSING_VERSION",
            msg = "meta.version is required" }
    end
    if not id.description or id.description == "" then
        out[#out + 1] = {
            severity = "error", code = "E_META_MISSING_DESCRIPTION",
            msg = "meta.description is required" }
    end
    if not id.category or id.category == "" then
        out[#out + 1] = {
            severity = "error", code = "E_META_MISSING_CATEGORY",
            msg = "meta.category is required" }
    end

    -- name vs directory
    if pkg_dir and id.name and id.name ~= "" and id.name ~= pkg_dir then
        out[#out + 1] = {
            severity = "error", code = "E_NAME_MISMATCH",
            msg = string.format(
                "meta.name='%s' does not match pkg directory '%s'",
                id.name, pkg_dir) }
    end

    -- description should be single-line
    if id.description and id.description:find("\n", 1, true) then
        out[#out + 1] = {
            severity = "warning", code = "W_DESCRIPTION_MULTILINE",
            msg = "meta.description contains a newline (keep it single-line)" }
    end

    -- docstring-level scan
    local in_fence = false
    local raw_lines = split_lines(docstring or "")
    for i = 1, #raw_lines do
        local line = raw_lines[i]
        -- Track triple-backtick fences so we don't flag code content.
        if line:match("^```") then
            in_fence = not in_fence
        elseif not in_fence then
            -- H1 detection: `# ` at column 0, but not `##`/`###`.
            if line:sub(1, 2) == "# " then
                out[#out + 1] = {
                    severity = "error", code = "E_H1_IN_DOCSTRING",
                    msg = "H1 is reserved for the generator; drop `# ` prefix",
                    line = i }
            end
            -- Fake label: "Word:" or "Word word:" with no indent.
            local label = line:match("^([%w][%w ]*):%s*$")
            if label and FAKE_LABELS[label] then
                out[#out + 1] = {
                    severity = "warning", code = "W_FAKE_LABEL",
                    msg = string.format(
                        "line looks like a section label; promote to '## %s'", label),
                    line = i }
            end
        end
    end

    -- narrative emptiness
    if (nar.summary == nil or nar.summary == "") and #nar.sections == 0 then
        out[#out + 1] = {
            severity = "warning", code = "W_EMPTY_NARRATIVE",
            msg = "docstring has no summary and no sections" }
    end

    -- Parameters conflict
    if shp.input ~= nil and has_parameters_section(nar.sections) then
        out[#out + 1] = {
            severity = "error", code = "E_PARAMETERS_CONFLICT",
            msg = "spec.entries.run.input is declared AND docstring has a ## " ..
                  "Parameters section; remove the docstring section (shape is the SSoT)" }
    end

    -- Result conflict
    if shp.result ~= nil and has_result_section(nar.sections) then
        out[#out + 1] = {
            severity = "error", code = "E_RESULT_CONFLICT",
            msg = "spec.entries.run.result is declared AND docstring has a ## " ..
                  "Result section; remove the docstring section (shape is the SSoT)" }
    end

    return { violations = out }
end

--- Format a violation list for human stderr output.
function M.format(pkg_name, violations)
    if #violations == 0 then return "" end
    local parts = { string.format("# %s", pkg_name) }
    for i = 1, #violations do
        local v = violations[i]
        local tag = v.severity == "error" and "ERROR" or "warn "
        local loc = v.line and string.format(" (line %d)", v.line) or ""
        parts[#parts + 1] = string.format(
            "  %s %s%s: %s", tag, v.code, loc, v.msg)
    end
    return table.concat(parts, "\n")
end

--- Filter violations by severity.
function M.errors(violations)
    local out = {}
    for i = 1, #violations do
        if violations[i].severity == "error" then
            out[#out + 1] = violations[i]
        end
    end
    return out
end

M._internal = {
    FAKE_LABELS = FAKE_LABELS,
    split_lines = split_lines,
}

return M
