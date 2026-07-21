//! GPU smoke test — the **Lua bridge** `alc.nn.*` surface end-to-end.
//!
//! Sibling of the `algocline-nn` crate's Rust-level smoke examples
//! (`nn_tinyllama_lora_gpu_smoke.rs` et al.) which drive the candle
//! surface directly. This driver instead exercises the *engine
//! bridge* — the Lua-visible `alc.nn.{preset, data, wrap_lora,
//! trainer.*}` entries — so the GPU (A40 CUDA) run verifies the
//! whole `Lua -> bridge -> algocline-nn -> candle` path, not just the
//! Rust surface.
//!
//! It boots a full `alc.*` surface via
//! [`algocline_engine::bridge::install_for_pkg_test`] (the
//! `alc_pkg_test` sandbox bootstrap: full production `alc.*` surface
//! per the `bridge_sandbox_parity` invariant, plus a per-VM tempdir
//! backing `alc.state` / `alc.card` / the nn store). The embedded Lua
//! script then walks 7 phases, printing per-phase wall time; any
//! failure raises a Lua error that surfaces as a non-zero process
//! exit.
//!
//! Phases:
//!
//!   1. `alc.nn.preset.tinyllama(variant, { pretrained = false, … })`
//!      — build a random-init base handle (no HF download).
//!   2. `alc.nn.data.synthetic(rows, opts)` — build a token dataset
//!      from a Lua-generated palette corpus (tokenizer-free).
//!   3. `alc.nn.wrap_lora(base, opts)` — wrap-only inspect; the
//!      wrapped handle is confirmed LoRA-wrapped via the double-wrap
//!      rejection path (`is_lora_wrapped` is not a Lua-exposed
//!      method, so this is the observable proxy).
//!   4. `alc.nn.trainer.run_lora_ft(fresh_base, ds, opts)` — train +
//!      mint a `training_path="lora"` Card; print the `lora_card_id`.
//!   5. `alc.nn.card.load_wrap(lora_card_id, fresh_base)` — reload the
//!      LoRA Card onto a fresh base (round-trip).
//!   6. `alc.nn.trainer.run_full_ft(fresh_base, ds, opts)` — full
//!      fine-tune + Card; print the `card_id`.
//!   7. `alc.nn.trainer.run_distill(fresh_base, ds, opts{loss_kind="ce"})`
//!      — distillation + Card; print the `card_id`.
//!
//! Usage:
//!
//! ```bash
//! # CUDA (A40 pod): default variant tinyllama-1.1b, device cuda.
//! cargo run --release --features nn-cuda --example nn_bridge_gpu_smoke
//!
//! # CPU verify on a dev host (tiny variant, a few steps):
//! NN_SMOKE_DEVICE=cpu NN_SMOKE_VARIANT=tiny NN_SMOKE_STEPS=3 \
//!   NN_SMOKE_BATCH=2 NN_SMOKE_CTX=32 \
//!   cargo run -p algocline-engine --features nn --example nn_bridge_gpu_smoke
//! ```
//!
//! Env vars (all optional):
//!
//! - `NN_SMOKE_DEVICE`  (default `"cuda"`) — `"cpu"` / `"cuda"` /
//!   `"cuda:N"`. `"cuda"` requires a `--features nn-cuda` build.
//! - `NN_SMOKE_VARIANT` (default `"1.1b"`) — TinyLlama variant:
//!   `"1.1b"` / `"tinyllama-1.1b"` (random-init 1.1B) or `"tiny"` /
//!   `"tinyllama-tiny"` (2-layer CPU smoke shape, vocab 32, ctx 16).
//! - `NN_SMOKE_STEPS`   (default `50`) — optimizer steps per trainer.
//! - `NN_SMOKE_BATCH`   (default `2`)  — batch size.
//! - `NN_SMOKE_CTX`     (default `64`) — requested sequence length.
//!   Clamped down to the base handle's model `ctx` so the RoPE cache
//!   is never overrun (the tiny variant caps at 16).
//! - `NN_SMOKE_RANK`    (default `16`) — LoRA rank.
//! - `NN_SMOKE_ALPHA`   (default `32`) — LoRA alpha scaling.
//! - `NN_SMOKE_LR`      (default `3e-4`) — peak learning rate.
//! - `NN_SMOKE_WARMUP`  (default `min(steps, 5)`) — schedule warmup.
//!
//! Exit 0 = all 7 bridge phases completed and every Card round-trip
//! succeeded.

use std::time::Instant;

use mlua::Lua;

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

/// Embedded Lua driver. Reads the `SMOKE` config table (set from Rust
/// below) and walks the 7 bridge phases. Any `assert` failure or
/// bridge error raises a Lua error, which `exec()` propagates as a
/// non-zero process exit.
const SCRIPT: &str = r#"
local S = SMOKE

local function log(phase, desc, t0)
    local dt = alc.time() - t0
    print(string.format("[bridge-smoke] phase=%d %s ok in %.2fs", phase, desc, dt))
end

local function make_base()
    return alc.nn.preset.tinyllama(S.variant, { pretrained = false, device = S.device })
end

-- Phase 1: base handle (random init, no HF download).
local t = alc.time()
local base = make_base()
assert(base:layers() > 0, "base must report a positive layer count")
assert(base:ctx() > 0, "base must report a positive ctx")
assert(base:vocab() > 0, "base must report a positive vocab")

-- Derive the effective ctx / token palette from the concrete handle:
-- clamp ctx to the model's RoPE cache size, and bound token ids
-- strictly under vocab so the same corpus is valid for the tiny
-- (vocab 32 / ctx 16) and 1.1b (vocab 32000 / ctx 2048) variants.
local eff_ctx = math.min(S.ctx, base:ctx())
local vsafe = math.min(base:vocab(), 30)
if vsafe < 1 then
    vsafe = 1
end
local rows_needed = S.steps * S.batch + S.batch
log(1, string.format("preset.tinyllama variant=%s layers=%d ctx=%d vocab=%d", S.variant, base:layers(), base:ctx(), base:vocab()), t)

local function make_dataset()
    local rows = {}
    for r = 0, rows_needed - 1 do
        local row = {}
        for p = 0, eff_ctx - 1 do
            row[#row + 1] = (r + p * 7) % vsafe
        end
        rows[#rows + 1] = row
    end
    return alc.nn.data.synthetic(rows, { batch_size = S.batch, ctx_len = eff_ctx })
end

local function train_opts(extra)
    local o = {
        lr = S.lr,
        batch = S.batch,
        steps = S.steps,
        warmup = S.warmup,
        schedule = "CosineWithWarmup",
    }
    if extra then
        for k, v in pairs(extra) do
            o[k] = v
        end
    end
    return o
end

-- Phase 2: token dataset (single-pass; a fresh one is built per
-- trainer phase because TokenizedDataset does not cycle).
t = alc.time()
local _ds_probe = make_dataset()
assert(_ds_probe:ctx_len() == eff_ctx, "dataset ctx_len must match requested eff_ctx")
log(2, string.format("data.synthetic rows=%d batch=%d ctx=%d", rows_needed, S.batch, eff_ctx), t)

-- Phase 3: wrap_lora inspect. is_lora_wrapped() is not a Lua method,
-- so confirm the wrap took by asserting the double-wrap rejection
-- path fires on the wrapped handle (a base handle would wrap cleanly).
t = alc.time()
local wrapped = alc.nn.wrap_lora(base, { rank = S.rank, alpha = S.alpha })
assert(wrapped:arch() == "tinyllama", "wrapped handle must report tinyllama arch")
local double_ok = pcall(function()
    return alc.nn.wrap_lora(wrapped, { rank = S.rank, alpha = S.alpha })
end)
assert(not double_ok, "double-wrap must be rejected (confirms handle is LoRA-wrapped)")
base = nil
wrapped = nil
collectgarbage()
log(3, string.format("wrap_lora rank=%d alpha=%.1f double_wrap_rejected=true", S.rank, S.alpha), t)

-- Phase 4: run_lora_ft on a fresh base + fresh dataset.
t = alc.time()
local lora_base = make_base()
local lora_ds = make_dataset()
local lora_id = alc.nn.trainer.run_lora_ft(
    lora_base,
    lora_ds,
    train_opts({ rank = S.rank, alpha = S.alpha, name = "bridge-smoke-lora" })
)
assert(type(lora_id) == "string" and #lora_id > 0, "run_lora_ft must return a card id string")
lora_base = nil
lora_ds = nil
collectgarbage()
log(4, "run_lora_ft lora_card_id=" .. lora_id, t)

-- Phase 5: load_wrap round-trip (reload the LoRA Card onto a fresh base).
t = alc.time()
local reload_base = make_base()
local reloaded = alc.nn.card.load_wrap(lora_id, reload_base)
assert(reloaded:arch() == "tinyllama", "load_wrap must return a tinyllama handle")
reload_base = nil
reloaded = nil
collectgarbage()
log(5, "card.load_wrap round-trip ok for " .. lora_id, t)

-- Phase 6: run_full_ft on a fresh base + fresh dataset.
t = alc.time()
local ff_base = make_base()
local ff_ds = make_dataset()
local ff_id = alc.nn.trainer.run_full_ft(ff_base, ff_ds, train_opts({ name = "bridge-smoke-full" }))
assert(type(ff_id) == "string" and #ff_id > 0, "run_full_ft must return a card id string")
ff_base = nil
ff_ds = nil
collectgarbage()
log(6, "run_full_ft card_id=" .. ff_id, t)

-- Phase 7: run_distill on a fresh base + fresh dataset.
t = alc.time()
local rd_base = make_base()
local rd_ds = make_dataset()
local rd_id = alc.nn.trainer.run_distill(
    rd_base,
    rd_ds,
    train_opts({ loss_kind = "ce", name = "bridge-smoke-distill" })
)
assert(type(rd_id) == "string" and #rd_id > 0, "run_distill must return a card id string")
rd_base = nil
rd_ds = nil
collectgarbage()
log(7, "run_distill card_id=" .. rd_id, t)

print(string.format(
    "[bridge-smoke] summary: all 7 phases ok (variant=%s device=%s steps=%d batch=%d ctx=%d)",
    S.variant, S.device, S.steps, S.batch, eff_ctx
))
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = env_string("NN_SMOKE_DEVICE", "cuda");
    let variant = env_string("NN_SMOKE_VARIANT", "1.1b");
    let steps = env_usize("NN_SMOKE_STEPS", 50);
    let batch = env_usize("NN_SMOKE_BATCH", 2);
    let ctx = env_usize("NN_SMOKE_CTX", 64);
    let rank = env_usize("NN_SMOKE_RANK", 16);
    let alpha = env_f64("NN_SMOKE_ALPHA", 32.0);
    let lr = env_f64("NN_SMOKE_LR", 3e-4);
    let warmup = env_usize("NN_SMOKE_WARMUP", steps.min(5));

    eprintln!(
        "[bridge-smoke] config: device={device} variant={variant} steps={steps} \
         batch={batch} ctx={ctx} rank={rank} alpha={alpha} lr={lr} warmup={warmup}"
    );

    let boot_t0 = Instant::now();
    let lua = Lua::new();
    algocline_engine::bridge::install_for_pkg_test(&lua)?;
    eprintln!(
        "[bridge-smoke] alc.* surface installed in {:.2?}",
        boot_t0.elapsed()
    );

    let cfg = lua.create_table()?;
    cfg.set("device", device)?;
    cfg.set("variant", variant)?;
    cfg.set("steps", steps)?;
    cfg.set("batch", batch)?;
    cfg.set("ctx", ctx)?;
    cfg.set("rank", rank)?;
    cfg.set("alpha", alpha)?;
    cfg.set("lr", lr)?;
    cfg.set("warmup", warmup)?;
    lua.globals().set("SMOKE", cfg)?;

    lua.load(SCRIPT).set_name("@nn_bridge_gpu_smoke").exec()?;

    Ok(())
}
