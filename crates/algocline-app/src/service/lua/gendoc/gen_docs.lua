#!/usr/bin/env lua
--- tools/gen_docs.lua — publish-artifact generator for bundled-packages.
---
--- Analogous to cargo-dist: projects release-facing artifacts from an
--- upstream-produced manifest (`hub_index.json`), rather than generating
--- per-package API reference. Rich per-package RustDoc-style output is
--- intentionally out of scope — for local pkg lookup use `alc_info` /
--- `alc_hub_search` or read the source.
---
--- Input: `hub_index.json` (produced by algocline MCP's
--- `alc_hub_reindex`). Single source of truth for which pkgs exist.
--- Non-pkg directories (e.g. `alc_shapes`, which has no `M.meta.name`)
--- are already excluded by `alc_hub_reindex` in algocline 0.25.0+, so
--- gen_docs never sees them.
---
--- Output (under {out_dir}):
---   narrative/{pkg}.md                   — human-readable per pkg
---   hub/{pkg}.json (if --hub)            — machine contract per pkg
---   llms.txt / llms-full.txt             — LLM consumption index
---   {repo_root}/context7.json            — Context7 manifest (--context7)
---   {repo_root}/.devin/wiki.json         — DeepWiki manifest (--devin)
---
--- Preconditions:
---   `hub_index.json` must exist and be fresh. Run `alc_hub_reindex`
---   (algocline MCP tool) before `gen_docs`. A missing / unsupported
---   / stale index is a hard error — not a warning.
---
--- Usage:
---   lua tools/gen_docs.lua [options] [repo_root] [out_dir]
---
--- Options:
---   --lint         Run the V0 lint pass and print violations. Errors are
---                  counted but do not block output unless --strict.
---   --strict       Treat lint errors as build failures (exit 2 when any
---                  pkg reports an error-level violation).
---   --lint-only    Run lint, skip file generation.
---   --hub          Additionally emit hub_entry JSON per pkg at
---                  {out_dir}/hub/{pkg}.json (pipeline-spec §7.4).
---   --context7     Additionally emit {repo_root}/context7.json for
---                  Context7 ingestion (pipeline-spec §7.6).
---   --devin        Additionally emit {repo_root}/.devin/wiki.json for
---                  DeepWiki ingestion (pipeline-spec §7.6).
---   --hub-index=PATH  Override the path to hub_index.json. Default is
---                  "{repo_root}/hub_index.json".
---
--- Defaults: repo_root = ".", out_dir = "{repo_root}/docs"

local function setup_package_path(repo_root)
    package.path = table.concat({
        repo_root .. "/?.lua",
        repo_root .. "/?/init.lua",
        package.path,
    }, ";")
end

-- ── argv parsing ──────────────────────────────────────────────────────

local function parse_argv(argv)
    local opts = {
        lint = false, strict = false, lint_only = false,
        hub = false, context7 = false, devin = false,
        hub_index = nil,
    }
    local positional = {}
    for i = 1, #argv do
        local a = argv[i]
        if a == "--lint" then
            opts.lint = true
        elseif a == "--strict" then
            opts.strict = true
            opts.lint   = true
        elseif a == "--lint-only" then
            opts.lint_only = true
            opts.lint      = true
        elseif a == "--hub" then
            opts.hub = true
        elseif a == "--context7" then
            opts.context7 = true
        elseif a == "--devin" then
            opts.devin = true
        elseif a:sub(1, 12) == "--hub-index=" then
            opts.hub_index = a:sub(13)
        elseif a:sub(1, 2) == "--" then
            io.stderr:write("gen_docs: unknown option '" .. a .. "'\n")
            os.exit(2)
        else
            positional[#positional + 1] = a
        end
    end
    return opts, positional
end

-- ── file I/O ──────────────────────────────────────────────────────────

local function write_file(path, content)
    local f, err = io.open(path, "w")
    if not f then
        error(string.format("gen_docs: cannot write '%s': %s",
                            path, tostring(err)))
    end
    f:write(content)
    f:close()
end

local function ensure_dir(path)
    local ok = os.execute(string.format("mkdir -p %q", path))
    if not ok then
        error("gen_docs: mkdir -p failed for " .. path)
    end
end

-- ── main ──────────────────────────────────────────────────────────────

local function main(argv)
    local opts, positional = parse_argv(argv or {})
    local repo_root = positional[1] or "."
    local out_dir   = positional[2] or (repo_root .. "/docs")
    local hub_index = opts.hub_index or (repo_root .. "/hub_index.json")

    setup_package_path(repo_root)

    local List        = require("tools.docs.list")
    local Extract     = require("tools.docs.extract")
    local Projections = require("tools.docs.projections")
    local Lint        = opts.lint and require("tools.docs.lint") or nil

    local pkgs = List.list_pkgs(repo_root, hub_index)
    if #pkgs == 0 then
        io.stderr:write(
            "gen_docs: hub_index.json lists zero packages at "
            .. hub_index .. "\n")
        os.exit(1)
    end

    if not opts.lint_only then
        ensure_dir(out_dir)
        ensure_dir(out_dir .. "/narrative")
        if opts.hub then
            ensure_dir(out_dir .. "/hub")
        end
    end

    local infos         = {}
    local entries       = {}
    local failures      = {}
    local lint_errors   = 0
    local lint_warnings = 0

    for i = 1, #pkgs do
        local p = pkgs[i]
        local ok, info_or_err = pcall(
            Extract.build_pkg_info, p.name, p.init_path, p.source_path)
        if ok then
            local info = info_or_err
            local md   = Projections.narrative_md(info)
            if not opts.lint_only then
                write_file(string.format("%s/narrative/%s.md", out_dir, p.name), md)
                if opts.hub then
                    write_file(
                        string.format("%s/hub/%s.json", out_dir, p.name),
                        Projections.hub_entry(info))
                end
            end
            infos[#infos + 1]     = info
            entries[#entries + 1] = { name = p.name, narrative_md = md }

            if Lint then
                local docstring = Extract.extract_docstring(p.init_path)
                local result = Lint.check(info, docstring, p.name)
                if #result.violations > 0 then
                    for _, v in ipairs(result.violations) do
                        if v.severity == "error" then
                            lint_errors = lint_errors + 1
                        else
                            lint_warnings = lint_warnings + 1
                        end
                    end
                    io.stderr:write(Lint.format(p.name, result.violations) .. "\n")
                end
            end

            io.stdout:write(string.format("  [ok]   %s\n", p.name))
        else
            -- hub_index listed this pkg but extraction failed. This is
            -- drift between the indexer and our DSL evaluator — not
            -- a soft skip. Fail loudly so the divergence surfaces.
            local msg = tostring(info_or_err)
            failures[#failures + 1] = { name = p.name, err = msg }
            io.stderr:write(string.format("  [FAIL] %s: %s\n", p.name, msg))
        end
    end

    if not opts.lint_only then
        write_file(out_dir .. "/llms.txt",      Projections.llms_index(infos))
        write_file(out_dir .. "/llms-full.txt", Projections.llms_full(entries))
        if opts.context7 then
            local Context7Config = require("tools.docs.context7_config")
            write_file(repo_root .. "/context7.json",
                       Projections.context7_config(Context7Config))
            io.stdout:write("  [ok]   context7.json (repo root)\n")
        end
        if opts.devin then
            local DevinConfig = require("tools.docs.devin_wiki_config")
            ensure_dir(repo_root .. "/.devin")
            write_file(repo_root .. "/.devin/wiki.json",
                       Projections.devin_wiki(DevinConfig))
            io.stdout:write("  [ok]   .devin/wiki.json (repo root)\n")
        end
    end

    io.stdout:write(string.format(
        "\ngen_docs: %d generated, %d failed",
        #infos, #failures))
    if Lint then
        io.stdout:write(string.format(
            ", lint: %d error(s) / %d warning(s)",
            lint_errors, lint_warnings))
    end
    io.stdout:write("\n")

    if #failures > 0 then os.exit(1) end
    if opts.strict and lint_errors > 0 then os.exit(2) end
end

main(arg)
