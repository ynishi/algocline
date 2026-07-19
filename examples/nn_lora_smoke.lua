-- alc.nn LoRA smoke — verifies the full LoRA lifecycle end-to-end:
--   1. train a LoRA delta on top of a from-scratch tiny GPT-2 (ST-c)
--   2. save a Card with `[metadata.nn.candle.lora]` populated (ST-d
--      schema extension: target_modules / dropout / delta_path)
--   3. reload the model via `alc.nn.card.load_gpt2` against a fresh
--      base handle and confirm forward runs (ST-d load-with-merge)
--
-- Completes in a couple of seconds on CPU. Run via the Rust
-- integration harness at
-- `crates/algocline-engine/tests/nn_smoke_test.rs::lora_smoke_runs`.

local base = alc.nn.preset.gpt2("tiny", { pretrained = false, device = "cpu" })
assert(base:variant() == "tiny", "expected tiny base")

local rows = {
    { 1, 5, 12, 20, 33, 44, 51, 60 },
    { 2, 8, 15, 22, 30, 40, 55, 63 },
    { 3, 9, 17, 25, 32, 42, 50, 58 },
    { 4, 10, 18, 26, 34, 45, 52, 59 },
}
local ds = alc.nn.data.synthetic(rows, { batch_size = 1, ctx_len = 8 })

local ckpt = alc.nn.trainer.lora(base, ds, {
    lr = 3e-4,
    batch_size = 1,
    steps = 3,
    warmup = 0,
    schedule = "cosine",
    weight_decay = 0.0,
    ckpt_every = 0,
    card_id = "smoke_lora",
    rank = 4,
    alpha = 8,
    -- Wrap all six canonical targets (attention Q/K/V/O + MLP up/down).
    -- Omitting `target_modules` would fall back to the same default.
    dropout = 0.0,
})

-- Checkpoint shape assertions.
assert(type(ckpt) == "table", "ckpt must be a table")
assert(ckpt.step == 3, "expected 3 steps completed, got " .. tostring(ckpt.step))
assert(type(ckpt.train_loss) == "number", "train_loss must be number")
assert(type(ckpt.bundle_ref) == "string", "bundle_ref must be string")

-- LoRA sub-table (ST-c populates the trio + ST-d adds the extras).
assert(type(ckpt.lora) == "table", "lora ckpt must carry ckpt.lora")
assert(ckpt.lora.rank == 4, "lora rank must round-trip, got " .. tostring(ckpt.lora.rank))
assert(ckpt.lora.alpha == 8, "lora alpha must round-trip, got " .. tostring(ckpt.lora.alpha))
assert(type(ckpt.lora.base_bundle_ref) == "string", "base_bundle_ref must be string")
assert(type(ckpt.lora.target_modules) == "table", "target_modules must be table")
assert(#ckpt.lora.target_modules == 6, "default target_modules must have 6 entries")
assert(type(ckpt.lora.dropout) == "number", "dropout must be number")
assert(type(ckpt.lora.delta_path) == "string", "delta_path must be string (ST-d)")

-- Save a Card with the lora branch threaded from ckpt.lora. The
-- caller passes an empty `vars` table because the trained delta
-- already lives at `ckpt.lora.delta_path` (written by run_lora_ft
-- during the trainer call); `load_gpt2` reads from delta_path
-- directly and does not touch `<nn_dir>/<card_id>.safetensors`.
local card_id = alc.nn.card.save({}, "smoke-lora-card", {
    training_path = "lora",
    architecture = "tiny",
    candle = {
        lora = ckpt.lora,
    },
})
assert(type(card_id) == "string" and #card_id > 0, "card_id must be non-empty string")

-- Reload path: fresh base handle + load_gpt2 must return a
-- LoRA-wrapped Gpt2Handle sharing the same variant / shape as the
-- base. `wrap_lora` mutates the base's model in place, so callers
-- pass a fresh base for each reload.
local fresh_base = alc.nn.preset.gpt2("tiny", { pretrained = false, device = "cpu" })
local reloaded = alc.nn.card.load_gpt2(card_id, fresh_base)
assert(reloaded:variant() == "tiny", "reloaded handle must keep tiny variant")
assert(reloaded:layers() == 2, "reloaded handle must keep 2 layers")
assert(reloaded:vocab() == 64, "reloaded handle must keep vocab=64")

-- Shape probe — verify the wrapped model still produces logits of
-- the expected shape `[batch, seq, vocab]`. The delta has been
-- restored inside the shared model, so this exercises the
-- LoraLinear::forward code path.
local shape = reloaded:forward_shape(1, 4)
assert(
    shape[1] == 1 and shape[2] == 4 and shape[3] == 64,
    "reloaded forward_shape mismatch: "
        .. tostring(shape[1])
        .. "x"
        .. tostring(shape[2])
        .. "x"
        .. tostring(shape[3])
)

return {
    ok = true,
    variant = "lora",
    step = ckpt.step,
    train_loss = ckpt.train_loss,
    card_id = card_id,
    lora_rank = ckpt.lora.rank,
    lora_targets = #ckpt.lora.target_modules,
}
