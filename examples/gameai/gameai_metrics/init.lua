--- gameai_metrics — GameAI-specific metrics registered into alc.nn.metric.registry
---
--- Composes the ST1 Rust primitives (`alc.nn.metric.js` / `entropy`) with
--- gameai-domain compose logic (Card handles, prompt sets, autoplay) and
--- self-registers three named metrics into the per-VM Lua registry
--- (`alc.nn.metric.registry`, ST2) so the trainer `on_ckpt` hook (ST3) can
--- reach them by name.
---
--- ## Registered metrics
---
--- - `style_distance` — mean Jensen-Shannon divergence between the greedy
---   action distributions of two Cards over a shared prompt set. Zero when
---   two Cards answer every position identically, `log(2)` in the base-e
---   convention when they never overlap. Used by the Level Sweep learner
---   to keep a "trickiness bake" close to its teacher.
---
--- - `trickiness` — mean Shannon entropy of a single Card's noisy action
---   distribution over a prompt set. Zero for a deterministic (greedy)
---   Card, larger when temperature spreads the sampler over multiple
---   legal moves. Read as "how much the sampler is exploring".
---
--- - `level` — win rate + Wilson 95% CI against a baseline opponent over
---   N autoplay games. Not used as a train-time gate directly (Card
---   picks always converge to greedy which throws the trickiness away —
---   User Decided 2026-08-01); instead the CI lets the Level Sweep
---   learner keep only bakes whose win rate stays inside a target band
---   while `style_distance` still keeps the boss policy.
---
--- ## Registry contract
---
--- Every metric is registered as a `fn(ctx) -> value` where `ctx` is a
--- Lua table the hook passes in. Argument-shape mismatches raise a loud
--- error (nil / wrong type / empty prompt set) rather than silently
--- returning zero — the whole point of the Registry-based hook path is
--- that the trainer can trust the number it reads.
---
--- ## Usage from a trainer `on_ckpt` hook
---
--- ```lua
--- alc.nn.trainer.full_ft(handle, dataset, {
---     on_ckpt = function(info)
---         local level = alc.nn.metric.registry.evaluate("level", {
---             card = info.ckpt_path, opponent = "greedy", n_games = 32,
---         })
---         return level.win_rate >= 0.6 and "break" or "continue"
---     end,
--- })
--- ```

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics",
    version = "0.1.0",
    description = "GameAI metrics: style_distance / trickiness / level (registers into alc.nn.metric.registry)",
    category = "game",
}

M.style_distance = require("gameai_metrics.style_distance")
M.trickiness = require("gameai_metrics.trickiness")
M.level = require("gameai_metrics.level")

--- Self-register into the per-VM `alc.nn.metric.registry` so the trainer
--- `on_ckpt` hook can look every metric up by name. The registry is
--- installed empty on every VM boot by the ST2 bridge, so re-`require`ing
--- this pkg on a fresh VM re-registers the three entries. Registration
--- is idempotent within one VM: `register(name, fn)` replaces any prior
--- entry under the same name.
local function self_register()
    if not (alc and alc.nn and alc.nn.metric and alc.nn.metric.registry) then
        error(
            "gameai_metrics: alc.nn.metric.registry is not available on this VM "
                .. "(build without the nn feature? load the bridge first)"
        )
    end
    alc.nn.metric.registry.register("style_distance", function(ctx)
        ctx = ctx or {}
        return M.style_distance(ctx.card_a, ctx.card_b, ctx.prompt_set)
    end)
    alc.nn.metric.registry.register("trickiness", function(ctx)
        ctx = ctx or {}
        return M.trickiness(ctx.card, ctx.prompt_set, ctx.temperature or 1.0)
    end)
    alc.nn.metric.registry.register("level", function(ctx)
        ctx = ctx or {}
        return M.level(ctx.card, ctx.opponent, ctx.n_games or 32, ctx.seed or 0)
    end)
end

self_register()

return M
