--- alc_shapes.instrument — Malli-style declarative shape instrumentation.
---
--- Wraps a module entry (e.g. `M.run`) with dev-mode shape assertions
--- driven by `M.spec.entries[entry_name].{input, result}`, replacing the
--- manual pattern:
---
--- ```lua
--- function M.run(ctx) ... ; S.assert_dev(ctx.result, "voted", "sc.run"); return ctx end
--- ```
---
--- with:
---
--- ```lua
--- function M.run(ctx) ... ; return ctx end
--- M.run = S.instrument(M, "run")  -- at module tail
--- ```
---
--- ## Why a wrapper layer exists
---
--- The goal is **caller transparency**: recipe and downstream callers
--- should not need to know that shape checking exists. Three analogues
--- converge on this shape:
---
--- - **Malli `mi/instrument!`** (Clojure) — rewrites a `var` to a dev-
---   gated wrapper derived from a function schema. Production code keeps
---   calling `(my.ns/do-thing x)` unchanged; instrumentation is applied
---   at the var level so every caller gets validation for free. Source:
---   `https://github.com/metosin/malli/blob/master/docs/function-schemas.md`.
--- - **Zod v4 `z.function({ input, output }).implement(fn)`** — factory
---   returns a wrapped function. `fn` becomes the body; the wrapper
---   parses input and output on each call. Source:
---   `https://zod.dev/api#function` (v4 function API, 2025).
--- - **tRPC procedure builder** — input/output validators are declared
---   at the route; the builder wraps the handler so it only ever sees
---   validated data. Source: `https://trpc.io/docs/server/validators`.
---
--- All three enforce the boundary **once**, at the edge of the module,
--- by returning a replacement function. Caller-side wrapping (each
--- consumer inserts its own assert) is explicitly rejected as a Hyrum-
--- law hazard.
---
--- ## Relationship to spec_resolver.run
---
--- `spec_resolver.run(pkg, ctx)` is the **caller-wrap** form: consumers
--- that do not trust the pkg to self-validate route the call through
--- `SR.run` for a resolver-side pre/post check. `instrument` is the
--- **producer-wrap** form: the pkg self-decorates its entry once, and
--- every caller (`pkg.run(ctx)` direct, `SR.run(pkg, ctx)` via resolver,
--- or anything else) inherits the check for free. Both forms coexist;
--- bundled pkgs prefer `instrument` because it makes the producer the
--- single source of truth for its own contract.
---
--- ## Alc-specific adaptation
---
--- Two entry paradigms coexist. The declaration shape decides which
--- wrapper path runs.
---
--- ### (1) ctx-threading (default; `spec.entries.{e}.input`)
---
--- Every entry takes a single `ctx` table, writes `ctx.result = {...}`,
--- returns `ctx`. Applies to 90%+ of bundled pkgs.
---
--- - **Input** assertion targets the first positional argument (`ctx`).
---   Reads `M.spec.entries[entry_name].input` (string registry key or
---   inline schema; string is coerced to `T.ref` by `spec_resolver.resolve`).
--- - **Result** assertion targets `ret.result` (ctx-threading), falling
---   back to `ret` itself when the function does not return a table with
---   `.result` set. Reads `M.spec.entries[entry_name].result`.
---
--- ### (2) direct-args (library-style; `spec.entries.{e}.args`)
---
--- Pure library-style pkgs (e.g. `bft.threshold(n, f) -> number`,
--- `kemeny.aggregate(rankings) -> ranking`) take positional args and
--- return a raw value. `args` is an **array of shapes** aligned to the
--- function's positional parameters; the wrapper checks each `args[i]`
--- against the caller-supplied i-th argument.
---
--- - **Input** assertion: for each `args[i]` (non-nil slot), asserts the
---   caller's i-th argument. Slots may be `nil` to skip validation at
---   that position (useful for opaque options tables).
--- - **Result** assertion targets the **raw return value** (not
---   `ret.result`) — library functions return scalars / tables directly.
--- - **Optional args** use `T.x:is_optional()` at the corresponding slot;
---   the check handler accepts nil for optional schemas at top level.
---
--- `input` and `args` are mutually exclusive per entry. `spec_resolver`
--- raises at declaration time when both are set, so the wrapper body
--- only needs to branch on one side.
---
--- ### Shared
---
--- - **Hint** is `"<meta.name>.<entry_name>"`, e.g. `"calibrate.assess"`
---   or `"bft.threshold"`. Arg-position hints append `":arg<i>"`.
--- - **Gating** is `ALC_SHAPE_CHECK=1` (existing dev-mode env var). When
---   off, the wrapper only pays one `os.getenv` per call.
---
--- ## Override form (rare)
---
--- Most packages declare every entry under `M.spec.entries.*` so no
--- override is needed. The optional `spec` argument exists as an escape
--- hatch for callers that want to instrument an entry not declared in
--- `M.spec` (e.g. temporary test scaffolding):
---
--- ```lua
--- -- ctx-threading override:
--- M.custom = S.instrument(M, "custom", {
---     input  = T.shape({ task = T.string }, { open = true }),
---     result = "voted",
--- })
---
--- -- direct-args override:
--- M.threshold = S.instrument(M, "threshold", {
---     args   = { T.number, T.number },
---     result = T.number,
--- })
--- ```
---
--- `spec` takes precedence over `M.spec.entries[entry_name].*` when both
--- are given. This matches Zod's `.implement({ input, output })` which
--- lets the caller override the schemas declared on the factory.
--- Specifying `input` and `args` together (either via the spec override
--- or via `M.spec.entries.{e}.*`) is rejected at load time.

local check = require("alc_shapes.check")
local spec_resolver = require("alc_shapes.spec_resolver")

local M = {}

--- Wrap `mod[entry_name]` with dev-mode shape assertions.
---
--- Idempotent in dev-off mode: the wrapper calls `is_dev_mode()` once
--- per invocation and short-circuits before any schema work when off.
---
--- Loud-fails at load time when:
---   • `mod.meta.name` is missing (needed for the hint),
---   • `mod[entry_name]` is not a function (must be called AFTER the
---     function is defined — typical usage is at the module tail).
---
---@param mod table                             Module being instrumented.
---@param entry_name string                     Entry function name, e.g. "run".
---@param spec? { input?: table|string|nil, result?: table|string|nil }
---@return function                             Replacement function with identical call signature.
function M.instrument(mod, entry_name, spec)
    if type(mod) ~= "table" then
        error("alc_shapes.instrument: mod must be a table (got "
            .. type(mod) .. ")", 2)
    end
    if type(entry_name) ~= "string" or entry_name == "" then
        error("alc_shapes.instrument: entry_name must be a non-empty string", 2)
    end
    local orig = mod[entry_name]
    if type(orig) ~= "function" then
        error(string.format(
            "alc_shapes.instrument: mod[%q] is not a function (got %s); "
            .. "call instrument AFTER the function is defined",
            entry_name, type(orig)), 2)
    end
    local meta = mod.meta
    if type(meta) ~= "table"
        or type(meta.name) ~= "string"
        or meta.name == ""
    then
        error("alc_shapes.instrument: mod.meta.name (string) is required "
            .. "for hint construction", 2)
    end
    if spec ~= nil and type(spec) ~= "table" then
        error("alc_shapes.instrument: spec must be a table or nil (got "
            .. type(spec) .. ")", 2)
    end

    -- Pull declared shapes from M.spec via spec_resolver (string keys
    -- are coerced to T.ref here; `args` list slots are coerced the same
    -- way, matching the rest of the toolchain).
    local resolved = spec_resolver.resolve(mod)
    local entry = resolved.entries[entry_name] or {}

    local input_shape  = (spec and spec.input)  or entry.input
    local result_shape = (spec and spec.result) or entry.result
    local args_shapes  = (spec and spec.args)   or entry.args

    -- Mutual exclusion is already enforced by spec_resolver for the
    -- M.spec-declared form. Re-check here for the override form, where
    -- both fields can arrive through the `spec` argument directly.
    if args_shapes ~= nil and input_shape ~= nil then
        error(string.format(
            "alc_shapes.instrument: entry %q declares both `input` "
                .. "(ctx-threading) and `args` (direct-args); these modes "
                .. "are mutually exclusive",
            entry_name), 2)
    end

    local hint = meta.name .. "." .. entry_name

    if args_shapes ~= nil then
        -- Direct-args mode: library-style pkg (pure function).
        -- Validate each positional arg against args_shapes[i]; validate
        -- the raw return value against result_shape.
        local n = #args_shapes
        return function(...)
            local dev = check.is_dev_mode()
            if dev and n > 0 then
                for i = 1, n do
                    local sh = args_shapes[i]
                    if sh ~= nil then
                        check.assert(
                            (select(i, ...)),
                            sh,
                            hint .. ":arg" .. i)
                    end
                end
            end
            if dev and result_shape ~= nil then
                -- Preserve multi-return (e.g. `bool, string` patterns in
                -- bft.validate, eval_guard.self_critique, etc.). Without
                -- table.pack, `local ret = orig(...)` would drop 2nd+ values.
                local results = table.pack(orig(...))
                check.assert(results[1], result_shape, hint)
                return table.unpack(results, 1, results.n)
            end
            return orig(...)
        end
    end

    -- ctx-threading mode (default, unchanged).
    return function(ctx, ...)
        local dev = check.is_dev_mode()
        if dev and input_shape ~= nil then
            check.assert(ctx, input_shape, hint .. ":input")
        end
        local ret = orig(ctx, ...)
        if dev and result_shape ~= nil then
            local payload
            if type(ret) == "table" and ret.result ~= nil then
                payload = ret.result
            else
                payload = ret
            end
            check.assert(payload, result_shape, hint)
        end
        return ret
    end
end

return M
