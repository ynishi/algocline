-- alc.nn GPU bridge smoke — Lua MCP surface on CUDA.
--
-- Complements the Rust harness `crates/algocline-nn/examples/
-- nn_medium_gpu_smoke.rs` (candle direct) with the Lua bridge path so
-- the 2026-07-30 fixes land on real cuda hardware:
--
--   custom Card reload (issue 467e6630 / commit 1b35201)
--     — a gpt2-custom Card trained on cuda/bf16 reloads standalone
--       with the same device+dtype (round-trip verified end to end).
--
--   Gpt2Handle:kv_heads accessor (issue 4205df20 / commit 172d826)
--     — GQA custom (heads=2 / kv_heads=1) reports the configured
--       value through Lua on cuda, not the head count.
--
--   trained-card device/dtype round-trip (issue bcaba507 / commit 79f7f7c)
--     — `alc.card.get(card_id).metadata.nn.candle.device` /
--       `.dtype` carry "cuda" / "bf16" and drive the reload target.
--
-- Runs the run_full_ft one-call trainer on a small custom student
-- (vocab=50257, 2L2H dim=64) so a full training + Card-register +
-- standalone reload round-trip completes in ~10-30 s on an A40.
-- Then a scale-up phase runs `alc.nn.preset.gpt2("medium", ...)`
-- (355M) for a few steps to prove the GPU path is not tiny-only.
--
-- Ten phases:
--
--   1  gpu_preset_build_custom      — cuda/bf16 custom student
--   2  gpu_kv_heads_accessor        — :kv_heads() == 1 on cuda GQA build
--   3  gpu_dataset_build            — real gpt2 tokenizer + from_card
--   4  gpu_train_custom             — run_full_ft on cuda completes
--   5  gpu_card_meta_device_dtype   — metadata.nn.candle.device=="cuda"
--                                     / dtype=="bf16"
--   6  gpu_custom_card_reload       — load_handle rebuilds config from
--                                     Card and returns a wrapped NnHandle
--   7  gpu_reloaded_kv_heads        — reloaded handle :kv_heads() == 1
--   8  gpu_reloaded_generates       — generate_session runs on cuda
--   9  gpu_medium_scale_up          — preset.gpt2("medium", ...) builds
--                                     on cuda + 3-step run_full_ft
--   10 gpu_medium_card_meta         — medium Card records cuda / bf16
--
-- Pod-side invocation:
--
--   1. cargo install --path . --features nn,nn-cuda
--   2. alc info | grep nn   -- confirm nn feature is live
--   3. alc mcp-run --code-file examples/nn_medium_gpu_bridge_smoke.lua \
--        --project-root /workspace/algocline
--
-- ctx (all optional):
--   dtype    — "bf16" (default on cuda) / "f32"
--   corpus_url / cache_dir — tiny Shakespeare (Phase 3 uses a hand-
--                            crafted 4-row corpus so no network hit).
--
-- The one-call `alc.nn.trainer.run_full_ft` requires a from-scratch
-- (pretrained=false) handle. Card reload afterwards restores the
-- exact device+dtype the training used.

local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

local DTYPE = ctx_field("dtype") or "bf16"
local CTX_LEN = 64
local BATCH = 1
local STEPS_CUSTOM = 8
local STEPS_MEDIUM = 3
local LR = 3e-4
local RANK_NA = nil -- distill / full_ft only; LoRA GPU covered elsewhere

local function log(msg)
    alc.log("info", "[nn-gpu-e2e] " .. msg)
end

local checks = {}
local function run(label, thunk)
    local ok, val = pcall(thunk)
    checks[#checks + 1] = {
        label = label,
        ok = ok,
        note = ok and (type(val) == "string" and val or "OK") or tostring(val):sub(1, 240),
    }
    return ok, val
end

-- ── Phase 1: custom student on cuda/bf16 ────────────────────────────

local student
run("gpu_preset_build_custom", function()
    student = alc.nn.preset.gpt2("custom", {
        device = "cuda",
        dtype = DTYPE,
        pretrained = false,
        vocab = 50257,
        ctx = CTX_LEN,
        layers = 2,
        heads = 2,
        kv_heads = 1, -- exercises the GQA accessor fix
        dim = 64,
    })
    if not student then
        error("preset returned nil")
    end
    return string.format(
        "variant=%s vocab=%d ctx=%d layers=%d heads=%d dim=%d",
        student:variant(),
        student:vocab(),
        student:ctx(),
        student:layers(),
        student:heads(),
        student:dim()
    )
end)

-- ── Phase 2: kv_heads accessor pin on cuda GQA build ────────────────
--
-- Regression device for issue 4205df20 fix (Gpt2Handle::meta used to
-- mirror self.heads). Same assert as the CPU-side smoke, run here to
-- prove the fix on cuda too.

run("gpu_kv_heads_accessor", function()
    if not student then
        error("no student")
    end
    if student:kv_heads() ~= 1 then
        error("kv_heads = " .. tostring(student:kv_heads()) .. " (want 1 GQA, not heads=2)")
    end
    return "kv_heads=1 as configured (heads=2)"
end)

-- ── Phase 3: teacher card + from_card dataset (no network hit) ──────
--
-- A four-row hand-crafted corpus keeps the pod offline for the
-- dataset build. The real gpt2 tokenizer json still comes from the
-- HF hub on first run (later runs are cache-hit).

local dataset
run("gpu_dataset_build", function()
    local rows = {
        { prompt = "Q: 1", response = "A: one." },
        { prompt = "Q: 2", response = "A: two." },
        { prompt = "Q: 3", response = "A: three." },
        { prompt = "Q: 4", response = "A: four." },
    }
    -- Replicate so rows >= steps * batch for TeacherCardDataset.
    local expanded = {}
    for _ = 1, math.ceil((STEPS_CUSTOM * BATCH) / #rows) + 1 do
        for _, r in ipairs(rows) do
            expanded[#expanded + 1] = r
        end
    end

    local created = alc.card.create({
        pkg = { name = "alc_nn" },
        metadata = { kind = "teacher_log", loss_mask = "response" },
    })
    if not created or not created.card_id then
        error("alc.card.create returned no card_id")
    end
    alc.card.write_samples(created.card_id, expanded)

    dataset = alc.nn.data.from_card(created.card_id, {
        tokenizer = "gpt2",
        batch_size = BATCH,
        ctx_len = CTX_LEN,
    })
    if not dataset then
        error("from_card returned nil")
    end
    return string.format("TeacherCardDataset built (%d rows, ctx=%d)", #expanded, CTX_LEN)
end)

-- ── Phase 4: run_full_ft on cuda ────────────────────────────────────

local custom_card_id
run("gpu_train_custom", function()
    if not student or not dataset then
        error("prereq missing")
    end
    log(string.format("training on cuda/%s for %d steps", DTYPE, STEPS_CUSTOM))
    custom_card_id = alc.nn.trainer.run_full_ft(student, dataset, {
        lr = LR,
        batch = BATCH,
        steps = STEPS_CUSTOM,
        warmup = 0,
        schedule = "Constant",
        name = "gpu-bridge-custom-v1",
    })
    if not custom_card_id or type(custom_card_id) ~= "string" or #custom_card_id == 0 then
        error("card_id missing/empty: " .. tostring(custom_card_id))
    end
    return "card_id = " .. custom_card_id
end)

-- ── Phase 5: Card metadata pins cuda / <DTYPE> ──────────────────────
--
-- Live regression for issue bcaba507 fix (from_training used to
-- write device=None / dtype=None regardless of the handle target).
-- Reloading a GPU-trained card silently on CPU/f32 is exactly what
-- this assert catches.

local custom_card
run("gpu_card_meta_device_dtype", function()
    if not custom_card_id then
        error("no custom_card_id")
    end
    custom_card = alc.card.get(custom_card_id)
    if not custom_card then
        error("alc.card.get returned nil")
    end
    local nn = custom_card.metadata and custom_card.metadata.nn
    local candle = nn and nn.candle
    if not candle then
        error("metadata.nn.candle missing")
    end
    if candle.device ~= "cuda" then
        error("candle.device = " .. tostring(candle.device) .. " (want cuda)")
    end
    if candle.dtype ~= DTYPE then
        error("candle.dtype = " .. tostring(candle.dtype) .. " (want " .. DTYPE .. ")")
    end
    return string.format("candle.device=%s dtype=%s recorded", candle.device, candle.dtype)
end)

-- ── Phase 6: custom Card reload standalone ──────────────────────────

local reloaded
run("gpu_custom_card_reload", function()
    if not custom_card_id then
        error("no custom_card_id")
    end
    reloaded = alc.nn.card.load_handle(custom_card_id)
    if not reloaded then
        error("load_handle returned nil")
    end
    if reloaded:vocab() ~= 50257 or reloaded:ctx() ~= CTX_LEN then
        error(string.format("shape mismatch: vocab=%d ctx=%d", reloaded:vocab(), reloaded:ctx()))
    end
    return "reloaded standalone (custom shape restored from metadata)"
end)

-- ── Phase 7: reloaded kv_heads accessor ─────────────────────────────

run("gpu_reloaded_kv_heads", function()
    if not reloaded then
        error("no reloaded handle")
    end
    if reloaded:kv_heads() ~= 1 then
        error("reloaded kv_heads = " .. tostring(reloaded:kv_heads()) .. " (want 1)")
    end
    return "reloaded kv_heads=1 (GQA preserved through Card round-trip)"
end)

-- ── Phase 8: generate on cuda ───────────────────────────────────────

run("gpu_reloaded_generates", function()
    if not reloaded then
        error("no reloaded handle")
    end
    local session = reloaded:generate_session({ 1, 2, 3 })
    local logits = session:next_logits()
    local tok = logits:argmax()
    if type(tok) ~= "number" then
        error("generate_session produced no token")
    end
    return string.format("1 greedy token = %d", tok)
end)

-- ── Phase 9: medium 355M scale-up on cuda ───────────────────────────
--
-- Proves the GPU path is not tiny-only. Uses the reference gpt2
-- medium preset (24L / 16H / 1024 dim / 50257 vocab / 1024 ctx).
-- Only 3 steps so this stays inside a few seconds on A40.

local medium_card_id
run("gpu_medium_scale_up", function()
    local medium = alc.nn.preset.gpt2("medium", {
        device = "cuda",
        dtype = DTYPE,
        pretrained = false,
    })
    if not medium then
        error("medium preset returned nil")
    end
    local medium_dataset = alc.nn.data.synthetic({
        { 100, 101, 102, 103, 104, 105, 106, 107 },
        { 110, 111, 112, 113, 114, 115, 116, 117 },
        { 120, 121, 122, 123, 124, 125, 126, 127 },
        { 130, 131, 132, 133, 134, 135, 136, 137 },
    }, {
        batch_size = 1,
        ctx_len = 8,
        shuffle = false,
        pad_id = 0,
    })
    medium_card_id = alc.nn.trainer.run_full_ft(medium, medium_dataset, {
        lr = 1e-4,
        batch = 1,
        steps = STEPS_MEDIUM,
        warmup = 0,
        schedule = "Constant",
        name = "gpu-bridge-medium-scaleup-v1",
    })
    if not medium_card_id or #medium_card_id == 0 then
        error("medium card_id missing")
    end
    return "medium card_id = " .. medium_card_id
end)

-- ── Phase 10: medium Card records cuda / <DTYPE> ────────────────────

run("gpu_medium_card_meta", function()
    if not medium_card_id then
        error("no medium_card_id")
    end
    local card = alc.card.get(medium_card_id)
    local candle = card and card.metadata and card.metadata.nn and card.metadata.nn.candle
    if not candle then
        error("medium candle branch missing")
    end
    if candle.device ~= "cuda" or candle.dtype ~= DTYPE then
        error(
            string.format(
                "medium candle.device=%s dtype=%s (want cuda / %s)",
                tostring(candle.device),
                tostring(candle.dtype),
                DTYPE
            )
        )
    end
    return string.format("medium candle.device=%s dtype=%s recorded", candle.device, candle.dtype)
end)

-- ── verdict ─────────────────────────────────────────────────────────

local green, red = 0, 0
for _, c in ipairs(checks) do
    if c.ok then
        green = green + 1
    else
        red = red + 1
    end
end

return {
    verdict = (red == 0) and "COMPLETED_GREEN" or "COMPLETED_RED_FIND",
    green_count = green,
    red_count = red,
    checks = checks,
    custom_card_id = custom_card_id,
    medium_card_id = medium_card_id,
    dtype = DTYPE,
}
