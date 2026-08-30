--- gameai_metrics — GameAI-specific metrics for a trainer checkpoint hook
---
--- Composes the shared mathlib primitives (`alc.math.js_divergence` /
--- `alc.math.entropy`) with
--- gameai-domain compose logic (Card handles, prompt sets, autoplay) and
--- exports three ctx adapters on `M.metrics` that the trainer `on_ckpt`
--- hook (or an `anymetric` view) holds directly.
---
--- ## Measurement only
---
--- Every entry here answers "what does this Card do", on an axis of its
--- own, and stops there. None of them compares its number against a
--- target, and none of them decides whether a run should continue: that
--- reading belongs to the judgment layer that consumes the records.
--- Keeping the two apart is what lets one training run be observed
--- through several independent views at once — a metric that folded a
--- threshold in would collapse its axis into a verdict.
---
--- ## The metrics
---
--- - `style_distance` — mean Jensen-Shannon divergence between the
---   action distributions of two Cards over a shared prompt set. Zero
---   when two Cards answer every position identically, `log(2)` in the
---   base-e convention when they never overlap. Reads how far a Card has
---   moved from a reference Card (e.g. its teacher).
---
--- - `trickiness` — mean Shannon entropy of a single Card's
---   temperature-scaled action distribution over a prompt set. Zero for a
---   deterministic (greedy) Card, larger when the distribution spreads
---   over several legal moves. Reads how committed the policy is. On the
---   boss seat the value is normalised by the state's legal-move count,
---   because that count varies per state.
---
--- - `level` — win rate + Wilson 95% CI over N autoplay games against a
---   pool of opponents. Reads game-optimality from the seat the Card
---   plays, together with the uncertainty of that estimate.
---
--- ## Seats
---
--- `style_distance` / `trickiness` / `level` all take an optional
--- `seat` (`"player"`, the default, or `"boss"`). The boss seat also
--- needs a `style`, the basis its prompt encodes distances against.
--- Omitting both reproduces the pre-seat behaviour exactly. The boss-seat
--- decode contract lives in `boss_seat.lua`.
---
--- ## Adapter contract
---
--- `M.metrics.<name>` is a `fn(ctx) -> value` where `ctx` is a Lua table
--- the hook passes in; the modules underneath (`M.style_distance` and
--- friends) keep their positional signatures for direct callers.
--- Argument-shape mismatches raise a loud error (nil / wrong type /
--- empty prompt set) rather than silently returning zero — the point of
--- the hook path is that the trainer can trust the number it reads.
---
--- The adapters are plain values on a table, not entries in a name
--- registry: the hook is built in Lua and takes a Lua function, so a
--- name would be resolved back to the very function the caller already
--- had in hand.
---
--- ## Usage from a trainer `on_ckpt` hook
---
--- ```lua
--- local gm = require("gameai_metrics")
---
--- alc.nn.trainer.full_ft(handle, dataset, {
---     on_ckpt = function(info)
---         local card = alc.nn.card.load_ckpt(info.ckpt_path, { arch = "gpt2-tiny" })
---         local level = gm.metrics.level({
---             card = card, seat = "boss", style = "guardian",
---             opponents = { "random" }, n_games = 50,
---         })
---         alc.log("info", string.format("step=%d win_rate=%.3f ci_lower=%.3f",
---             info.step, level.win_rate, level.ci_lower))
---         return "continue"
---     end,
--- })
--- ```
---
--- The hook above only records; whether `ci_lower` is good enough to stop
--- on is a judgment-layer question, and the value returned to the trainer
--- is that layer's answer rather than this package's.

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics",
    version = "0.1.0",
    description = "GameAI metrics: style_distance / trickiness / level (ctx adapters on M.metrics)",
    category = "game",
}

M.boss_seat = require("gameai_metrics.boss_seat")
M.style_distance = require("gameai_metrics.style_distance")
M.trickiness = require("gameai_metrics.trickiness")
M.level = require("gameai_metrics.level")

--- Seat options every metric accepts, lifted out of an adapter ctx.
---
--- `opponent_style`, `temperature`, `per_game` and `per_move` are
--- `level`-only opts. Lifting them inside the `level` adapter instead
--- would keep this helper to the two seat keys its name promises; they
--- are lifted here anyway so the ctx keys stay in one place. A metric
--- that does not read them ignores a `nil`, whereas dropping them would
--- turn a ctx that names one into a loud error from `level` about an opt
--- the caller did pass.
---
--- The list is an allowlist, so a `level` opt missing from it is dropped
--- in silence: the ctx names it, no layer raises, and the run answers as
--- though the caller never asked. Every opt `level` accepts belongs on
--- this table.
local function seat_opts(ctx)
    return {
        seat = ctx.seat,
        style = ctx.style,
        opponent_style = ctx.opponent_style,
        temperature = ctx.temperature,
        per_game = ctx.per_game,
        per_move = ctx.per_move,
    }
end

--- The ctx adapters, one per metric.
---
--- Each is `fn(ctx) -> value`, the shape a trainer `on_ckpt` hook and an
--- `anymetric` view both call. They sit beside the positional modules
--- (`M.style_distance` and friends) rather than replacing them: a caller
--- that already has its arguments in hand calls the module, a caller
--- driving a checkpoint hook holds the adapter.
M.metrics = {}

--- Mean Jensen-Shannon divergence between two Cards over a prompt set.
---
--- `card_a` falls back to `card` so an observer that supplies the Card
--- under measurement through one shared key (the mid-run checkpoint
--- handle, identical for every view of a fire) can leave the view config
--- carrying only the reference `card_b`.
function M.metrics.style_distance(ctx)
    ctx = ctx or {}
    return M.style_distance(ctx.card_a or ctx.card, ctx.card_b, ctx.prompt_set, seat_opts(ctx))
end

--- Mean Shannon entropy of one Card's temperature-scaled action
--- distribution over a prompt set.
function M.metrics.trickiness(ctx)
    ctx = ctx or {}
    return M.trickiness(ctx.card, ctx.prompt_set, ctx.temperature or 1.0, seat_opts(ctx))
end

--- Win rate + Wilson 95% CI over N autoplay games against a pool.
function M.metrics.level(ctx)
    ctx = ctx or {}
    local opts = seat_opts(ctx)
    opts.opponents = ctx.opponents
    return M.level(ctx.card, ctx.opponent, ctx.n_games or 32, ctx.seed or 0, opts)
end

return M
