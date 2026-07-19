-- alc.nn full FT smoke — verifies the trainer namespace end-to-end
-- on a from-scratch tiny GPT-2 (2 layers / dim=32 / vocab=64) and a
-- 4-row synthetic corpus. Completes in a couple of seconds on CPU
-- and asserts the returned checkpoint has no `lora` sub-field
-- (Card foundation invariant: only lora-trained models carry
-- `NnLoraBranch`).
--
-- Run via the Rust integration harness at
-- `crates/algocline-engine/tests/nn_smoke_test.rs::full_ft_smoke_runs`.
-- Standalone execution requires a Lua VM with `alc.nn.*` registered
-- via `algocline_engine::bridge::register` — see the harness for the
-- construction pattern.

local handle = alc.nn.preset.gpt2("tiny", { pretrained = false, device = "cpu" })
assert(handle:variant() == "tiny", "expected tiny variant, got " .. tostring(handle:variant()))
assert(handle:layers() == 2, "tiny variant must have 2 layers")
assert(handle:vocab() == 64, "tiny variant must have vocab=64")

-- Synthetic corpus: token ids in [0, 64), 8 tokens per row.
-- The 4 rows overfit meaningfully in a couple of steps on a
-- 2-layer/2-head model, so `train_loss` should drop below the
-- freshly-initialised network's baseline (crude smoke — not a
-- convergence assertion).
local rows = {
    { 1, 5, 12, 20, 33, 44, 51, 60 },
    { 2, 8, 15, 22, 30, 40, 55, 63 },
    { 3, 9, 17, 25, 32, 42, 50, 58 },
    { 4, 10, 18, 26, 34, 45, 52, 59 },
}
local ds = alc.nn.data.synthetic(rows, {
    batch_size = 1,
    ctx_len = 8,
    shuffle = false,
    pad_id = 0,
})

local ckpt = alc.nn.trainer.full_ft(handle, ds, {
    lr = 3e-4,
    batch_size = 1,
    steps = 3,
    warmup = 0,
    schedule = "cosine",
    weight_decay = 0.0,
    ckpt_every = 0,
    card_id = "smoke_full_ft",
})

-- Checkpoint shape assertions.
assert(type(ckpt) == "table", "ckpt must be a table")
assert(ckpt.step == 3, "expected 3 steps completed, got " .. tostring(ckpt.step))
assert(type(ckpt.train_loss) == "number", "train_loss must be number")
assert(type(ckpt.bundle_ref) == "string", "bundle_ref must be string")
assert(type(ckpt.metrics) == "table", "metrics must be table")

-- Invariant: full_ft never populates ckpt.lora — only alc.nn.trainer.lora does.
assert(ckpt.lora == nil, "full_ft ckpt must NOT carry a lora sub-table")

return {
    ok = true,
    variant = "full_ft",
    step = ckpt.step,
    train_loss = ckpt.train_loss,
    bundle_ref = ckpt.bundle_ref,
}
