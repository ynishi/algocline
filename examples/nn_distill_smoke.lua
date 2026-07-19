-- alc.nn distillation smoke — verifies the trainer namespace
-- `distill` binding runs end-to-end on a tiny from-scratch GPT-2
-- with a synthetic "teacher" corpus.
--
-- Distillation in v1 shares the Full-FT loop core with a
-- HardLabelDistillLoss (CE on teacher tokens) — the returned
-- checkpoint therefore has the same shape as full_ft (no lora
-- sub-table). Completes in a couple of seconds on CPU. Run via the
-- Rust integration harness at
-- `crates/algocline-engine/tests/nn_smoke_test.rs::distill_smoke_runs`.

local student = alc.nn.preset.gpt2("tiny", { pretrained = false, device = "cpu" })
assert(student:variant() == "tiny", "expected tiny student")

-- Synthetic teacher-emitted sequences. In production
-- `alc.nn.data.from_card(teacher_card_id, ...)` streams
-- `{prompt, response}` pairs through a HF tokenizer; the smoke
-- example bypasses the tokenizer via `alc.nn.data.synthetic`
-- (vocab=64 has no matching HF tokenizer, so the synthetic path is
-- the only viable dataset for tiny presets).
local teacher_rows = {
    { 5, 12, 20, 33, 44, 51, 60, 8 },
    { 8, 15, 22, 30, 40, 55, 63, 5 },
    { 9, 17, 25, 32, 42, 50, 58, 4 },
    { 10, 18, 26, 34, 45, 52, 59, 6 },
}
local ds = alc.nn.data.synthetic(teacher_rows, { batch_size = 1, ctx_len = 8 })

local ckpt = alc.nn.trainer.distill(student, ds, {
    lr = 3e-4,
    batch_size = 1,
    steps = 3,
    warmup = 0,
    schedule = "cosine",
    weight_decay = 0.0,
    ckpt_every = 0,
    card_id = "smoke_distill",
    loss_kind = "ce", -- v1: only "ce"; "kl_soft" is a v2 carry.
})

-- Checkpoint shape assertions — same shape as full_ft.
assert(type(ckpt) == "table", "ckpt must be a table")
assert(ckpt.step == 3, "expected 3 steps completed, got " .. tostring(ckpt.step))
assert(type(ckpt.train_loss) == "number", "train_loss must be number")
assert(type(ckpt.bundle_ref) == "string", "bundle_ref must be string")
assert(type(ckpt.metrics) == "table", "metrics must be table")

-- Distillation shares the full_ft loop core, so no lora branch.
assert(ckpt.lora == nil, "distill ckpt must NOT carry a lora sub-table")

return {
    ok = true,
    variant = "distill",
    step = ckpt.step,
    train_loss = ckpt.train_loss,
    bundle_ref = ckpt.bundle_ref,
}
