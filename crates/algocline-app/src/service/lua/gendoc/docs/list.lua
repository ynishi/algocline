--- tools.docs.list — pkg enumeration from `hub_index.json`.
---
--- The docs pipeline treats `alc_hub_reindex` as the single source
--- of truth for which directories are packages. This module reads
--- the index JSON and produces the `{name, init_path, source_path}`
--- triple that the rest of the pipeline (`tools.docs.extract` etc.)
--- consumes.
---
--- Any drift between the index and the working tree (e.g. the index
--- lists a pkg whose `init.lua` has been deleted) is a hard error.
--- No soft-skip path exists — non-pkg directories are already excluded
--- by the indexer upstream.

local Json = require("tools.docs.json")

local M = {}

local function read_file(path)
    local f, err = io.open(path, "r")
    if not f then return nil, err end
    local content = f:read("*a")
    f:close()
    return content
end

--- Enumerate packages from a hub index.
---
--- Returns a list sorted by `name`, each entry:
---   { name = string, init_path = string, source_path = string }
---
--- Errors (raised via `error(..., 0)` so the caller sees just the
--- message) when:
---   * index file is missing or unreadable
---   * JSON is malformed
---   * `schema_version` ≠ "hub_index/v0"
---   * `packages` is not an array
---   * any entry has no non-empty `name`
---   * any listed pkg has no `init.lua` at
---     `{repo_root}/{name}/init.lua`
function M.list_pkgs(repo_root, index_path)
    local content, err = read_file(index_path)
    if not content then
        error(string.format(
            "tools.docs.list: hub_index.json not found at '%s' (%s). "
            .. "Run `alc_hub_reindex` (algocline MCP tool) first.",
            index_path, tostring(err)), 0)
    end

    local ok, index = pcall(Json.decode, content)
    if not ok then
        error(string.format(
            "tools.docs.list: hub_index.json at '%s' is malformed: %s",
            index_path, tostring(index)), 0)
    end

    if index.schema_version ~= "hub_index/v0" then
        error(string.format(
            "tools.docs.list: unsupported schema_version '%s' in '%s' "
            .. "(expected 'hub_index/v0')",
            tostring(index.schema_version), index_path), 0)
    end

    if type(index.packages) ~= "table" then
        error(string.format(
            "tools.docs.list: hub_index.json at '%s' has no "
            .. "`packages` array",
            index_path), 0)
    end

    local pkgs = {}
    for _, entry in ipairs(index.packages) do
        if type(entry.name) ~= "string" or entry.name == "" then
            error(string.format(
                "tools.docs.list: hub_index.json entry has no `name` "
                .. "(index '%s')",
                index_path), 0)
        end
        local init_path = repo_root .. "/" .. entry.name .. "/init.lua"
        local probe = io.open(init_path, "r")
        if not probe then
            error(string.format(
                "tools.docs.list: hub_index.json lists pkg '%s' but "
                .. "'%s' does not exist — run `alc_hub_reindex` to "
                .. "refresh the index",
                entry.name, init_path), 0)
        end
        probe:close()
        pkgs[#pkgs + 1] = {
            name        = entry.name,
            init_path   = init_path,
            source_path = entry.name .. "/init.lua",
        }
    end
    table.sort(pkgs, function(a, b) return a.name < b.name end)
    return pkgs
end

return M
