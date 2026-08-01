--- gameai_metrics._fs — tiny POSIX filesystem helpers used by
--- `harvest_collection` and `audit_matrix` when they write manifests.
---
--- Pure Lua; no `alc.*` host bridge dependency. The repo is macOS /
--- Linux only per its justfile so POSIX `mkdir -p` is a safe way to
--- create parent directories idempotently without pulling in a
--- LuaFileSystem-style native dep.
---
--- ## Why this exists
---
--- Both `harvest_collection:save()` and (a future) `audit_matrix:save()`
--- write JSON manifests to caller-supplied paths. The obvious failure
--- mode is a missing parent directory: the caller has to remember to
--- `mkdir -p workspace/gameai-harvest/` before the first save, and
--- forgetting it makes the trainer hook crash mid-run with an obscure
--- `io.open` `No such file or directory` error (that is exactly what
--- fired on the previous iteration's Run 1). Centralising a
--- `ensure_parent_dir` helper here removes that class of accident from
--- the whole gameai_metrics/ package with one call at the top of every
--- `:save()`.
---
--- ## Why the shell escape lives here (not `string.format("%q")`)
---
--- `string.format("%q", s)` produces a **Lua source** literal (double
--- quotes with `\"` / `\n` / `\\` escapes). It is not a shell escape:
--- it happily leaves `$` / backtick / `!` unquoted, so any path with
--- a shell metacharacter would be evaluated by `/bin/sh` when passed
--- through `os.execute`. `shell_squote` wraps the argument in single
--- quotes and rewrites embedded single quotes as `'\''`, which is the
--- canonical POSIX form and neutralises every metachar `sh` interprets
--- inside single quotes.

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics._fs",
    version = "0.1.0",
    description = "POSIX filesystem helpers (shell-safe mkdir -p) for gameai_metrics manifest writers.",
    category = "game",
}

--- POSIX single-quote escape. Wraps `s` in `'...'` and rewrites every
--- embedded `'` as `'\''`, which is the standard way to pass an
--- arbitrary string through `/bin/sh` without letting the shell
--- interpret metacharacters (`$`, backtick, `!`, `"`, space, newline).
---
--- Do NOT confuse with `string.format("%q", s)` — that emits a Lua
--- source literal (double-quoted with `\"` / `\n` escapes) which does
--- not protect against shell expansion.
---@param s string
---@return string quoted
function M.shell_squote(s)
    if type(s) ~= "string" then
        error("shell_squote: expected string, got " .. type(s), 2)
    end
    return "'" .. s:gsub("'", "'\\''") .. "'"
end

--- Ensure the parent directory of `path` exists, creating it (and any
--- missing intermediate directories) via `mkdir -p` when it does not.
--- Idempotent: an existing directory is a no-op.
---
--- When `path` has no directory component (e.g. `"foo.json"`), there
--- is nothing to create and the function returns silently.
---
--- Raises loudly when `os.execute` reports the `mkdir` invocation did
--- not succeed. Silent-failing here would let a broken filesystem
--- (read-only mount, quota exceeded, permission denied on the parent
--- prefix) hide behind the subsequent `io.open` "No such file or
--- directory" and mask the real cause.
---@param path string target file path
function M.ensure_parent_dir(path)
    if type(path) ~= "string" then
        error("ensure_parent_dir: expected string path, got " .. type(path), 2)
    end
    -- POSIX-style path extraction: the greedy match ".*/" captures the
    -- longest prefix ending in a slash, which is exactly the parent
    -- directory (including its trailing slash). A top-level relative
    -- path such as "foo.json" has no slash, and `parent` is nil.
    local parent = path:match("(.*/)")
    if parent == nil or parent == "" then
        return
    end
    local cmd = "mkdir -p " .. M.shell_squote(parent)
    local ok, exit_kind, exit_code = os.execute(cmd)
    if not ok then
        error(
            string.format(
                "ensure_parent_dir: mkdir -p %q failed (%s=%s)",
                parent,
                tostring(exit_kind),
                tostring(exit_code)
            ),
            2
        )
    end
end

return M
