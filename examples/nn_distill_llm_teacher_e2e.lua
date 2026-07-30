-- alc.nn distillation end-to-end verification with a live LLM teacher.
--
-- Third sibling of `nn_fullft_shakespeare_e2e.lua` / `nn_lora_shakespeare_e2e.lua`.
-- This script drives the public MCP surface via `alc_run` and verifies 13
-- observable properties of the distillation path, with the teacher signal
-- collected from the host LLM through `alc.llm_batch` (one pause; the host
-- answers every prompt and resumes via `alc_continue`):
--
--   Teacher collection / Card assembly (1-4)
--     1  teacher_collect             — alc.llm_batch returns a non-empty
--                                       response for every prompt
--     2  teacher_card_created        — alc.card.create with
--                                       metadata.loss_mask="response"
--     3  samples_written             — alc.card.write_samples (write-once)
--                                       lands all rows
--     4  samples_roundtrip           — alc.card.read_samples returns the rows
--
--   Dataset / student / train (5-7)
--     5  dataset_built               — alc.nn.data.from_card builds the
--                                       mask-carrying TeacherCardDataset with
--                                       the real gpt2 tokenizer
--     6  student_built               — alc.nn.preset.gpt2("custom") with
--                                       vocab=50257 so real gpt2 token ids fit
--     7  distill_completed           — alc.nn.trainer.run_distill trains and
--                                       registers a Card in one call
--
--   Card / loss / ckpt contract (8-10)
--     8  card_meta_shape             — training_path="distillation",
--                                       hyperparams.loss_kind="ce",
--                                       bundle_ref="nn/<card_id>"
--     9  loss_descended              — final loss below the ln(50257)=10.825
--                                       random baseline
--     10 ckpt_file_exists            — <nn_dir>/<card_id>.safetensors landed
--
--   Reload / guards (11-13)
--     11 custom_load_handle_roundtrip — load_handle rebuilds the custom
--                                       config from metadata.nn.candle.custom
--                                       and reloads the trained weights
--                                       standalone (shape + generate verified)
--     12 mask_boundary_guard         — from_card refuses a row whose prompt
--                                       exhausts ctx_len (FullyMaskedRow
--                                       protection, loud)
--     13 strict_validation_prefixes  — run_distill opts.lr missing / unknown
--                                       loss_kind / unknown schedule all BLOCK
--                                       with the surface-specific prefix
--                                       "alc.nn.trainer.run_distill"
--
-- Driving recipe: `alc_run` pauses at Phase 1 with status "needs_response"
-- (one llm_batch pause carrying all teacher prompts). Answer the queries —
-- e.g. dispatch the @alc-eval agent with the desired teacher model — and
-- the script resumes through Phase 13 without further pauses. See
-- `docs/nn-e2e-runbook.md` for the invocation recipe.
--
-- First run fetches the gpt2 tokenizer json from the HF hub into
-- <nn_dir>/tokenizers/gpt2.json; later runs are fully offline.

local N_PROMPTS = 12
local REPLICATE = 4 -- rows = N_PROMPTS * REPLICATE
local CTX_LEN = 64
local STEPS = 40
local LR = 1e-3
local BASELINE = 10.825 -- ln(50257)
local LOSS_THRESHOLD = 10.5

local PROMPTS = {
    "Q: What is the capital of France? Answer in at most 8 words.",
    "Q: What color is a ripe banana? Answer in at most 8 words.",
    "Q: How many legs does a spider have? Answer in at most 8 words.",
    "Q: What is water made of? Answer in at most 8 words.",
    "Q: Which planet is closest to the sun? Answer in at most 8 words.",
    "Q: What sound does a cat make? Answer in at most 8 words.",
    "Q: How many days are in a week? Answer in at most 8 words.",
    "Q: What season comes after winter? Answer in at most 8 words.",
    "Q: What do bees produce? Answer in at most 8 words.",
    "Q: What is the opposite of hot? Answer in at most 8 words.",
    "Q: How many sides does a triangle have? Answer in at most 8 words.",
    "Q: What language is spoken in Japan? Answer in at most 8 words.",
}

local function log(msg)
    alc.log("info", "[nn-distill-e2e] " .. msg)
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

-- Keep each teacher response to a single trimmed line so the
-- prompt.."\n"..response joint stays inside ctx and the response
-- region cannot smuggle extra separators.
local function sanitize(resp)
    local first = tostring(resp):match("[^\r\n]+") or ""
    first = first:gsub("^%s+", ""):gsub("%s+$", "")
    return first:sub(1, 200)
end

-- ── Phase 1: teacher collection (single llm_batch pause) ────────────

local pairs_pq = {}
run("teacher_collect", function()
    local items = {}
    for i, p in ipairs(PROMPTS) do
        items[i] = { prompt = p, max_tokens = 100 }
    end
    log(string.format("collecting %d teacher responses via alc.llm_batch", #items))
    local responses = alc.llm_batch(items)
    for i, p in ipairs(PROMPTS) do
        local r = sanitize(responses[i])
        if #r == 0 then
            error(string.format("teacher response %d empty", i))
        end
        pairs_pq[i] = { prompt = p, response = r }
    end
    return string.format("%d prompt/response pairs collected", #pairs_pq)
end)

-- ── Phase 2: teacher Card creation ──────────────────────────────────
--
-- metadata.loss_mask = "response" is the Tier 1 declaration that
-- switches alc.nn.data.from_card to the mask-carrying
-- TeacherCardDataset (prompt region masked out of the loss).

local teacher_card_id
run("teacher_card_created", function()
    local created = alc.card.create({
        pkg = { name = "alc_nn" },
        metadata = { kind = "teacher_log", loss_mask = "response" },
    })
    if not created or not created.card_id then
        error("alc.card.create returned no card_id")
    end
    teacher_card_id = created.card_id
    return "teacher card_id = " .. teacher_card_id
end)

-- ── Phase 3: samples written (write-once, all rows in one call) ─────
--
-- write_samples is write-once per Card, so rows are accumulated in
-- Lua and written in a single call. Pairs are replicated so
-- rows >= steps * batch (TeacherCardDataset does not wrap).

local n_rows = 0
run("samples_written", function()
    if not teacher_card_id or #pairs_pq == 0 then
        error("prereq missing")
    end
    local rows = {}
    for _ = 1, REPLICATE do
        for _, pq in ipairs(pairs_pq) do
            rows[#rows + 1] = { prompt = pq.prompt, response = pq.response }
        end
    end
    local res = alc.card.write_samples(teacher_card_id, rows)
    if not res or res.count ~= #rows then
        error(
            string.format("write_samples count = %s (want %d)", tostring(res and res.count), #rows)
        )
    end
    n_rows = res.count
    return string.format("%d rows written (%d pairs x %d)", n_rows, #pairs_pq, REPLICATE)
end)

-- ── Phase 4: samples round-trip ─────────────────────────────────────

run("samples_roundtrip", function()
    if not teacher_card_id then
        error("no teacher_card_id")
    end
    local samples = alc.card.read_samples(teacher_card_id)
    if not samples or #samples ~= n_rows then
        error(
            string.format(
                "read_samples count = %s (want %d)",
                tostring(samples and #samples),
                n_rows
            )
        )
    end
    local first = samples[1]
    if type(first.prompt) ~= "string" or type(first.response) ~= "string" then
        error("sample fields missing prompt/response")
    end
    return string.format("%d samples round-tripped", #samples)
end)

-- ── Phase 5: mask-carrying dataset via from_card ────────────────────

local dataset
run("dataset_built", function()
    if not teacher_card_id then
        error("no teacher_card_id")
    end
    dataset = alc.nn.data.from_card(teacher_card_id, {
        tokenizer = "gpt2",
        batch_size = 1,
        ctx_len = CTX_LEN,
    })
    if not dataset then
        error("from_card returned nil")
    end
    return string.format("TeacherCardDataset built (ctx=%d, batch=1)", CTX_LEN)
end)

-- ── Phase 6: custom student (vocab=50257 fits real gpt2 ids) ────────
--
-- The tiny preset (vocab=64) cannot hold real gpt2 token ids; medium
-- (355M) is CPU-hostile. The custom variant takes a vocab override
-- and stays small: 2L2H dim=64 over vocab 50257 ≈ 6.5M params.

local student
run("student_built", function()
    student = alc.nn.preset.gpt2("custom", {
        device = "cpu",
        dtype = "f32",
        pretrained = false,
        vocab = 50257,
        ctx = CTX_LEN,
        layers = 2,
        heads = 2,
        dim = 64,
    })
    if not student then
        error("preset returned nil")
    end
    return string.format(
        "variant=%s vocab=%d ctx=%d layers=%d dim=%d",
        student:variant(),
        student:vocab(),
        student:ctx(),
        student:layers(),
        student:dim()
    )
end)

-- ── Phase 7: distill train (one-call surface, run_distill) ──────────

local distill_card_id
run("distill_completed", function()
    if not student or not dataset then
        error("prereq missing")
    end
    log(string.format("distill: %d steps, lr=%g, batch=1, ctx=%d", STEPS, LR, CTX_LEN))
    distill_card_id = alc.nn.trainer.run_distill(student, dataset, {
        lr = LR,
        batch = 1,
        steps = STEPS,
        warmup = 0,
        schedule = "Constant",
        loss_kind = "ce",
        name = "distill-llm-teacher-v1",
    })
    if not distill_card_id or type(distill_card_id) ~= "string" or #distill_card_id == 0 then
        error("card_id missing/empty: " .. tostring(distill_card_id))
    end
    return "card_id = " .. distill_card_id
end)

-- ── Phase 8: distill Card metadata shape ────────────────────────────

local distill_card
run("card_meta_shape", function()
    if not distill_card_id then
        error("no distill_card_id")
    end
    distill_card = alc.card.get(distill_card_id)
    if not distill_card then
        error("alc.card.get returned nil")
    end
    local nn = distill_card.metadata and distill_card.metadata.nn
    if not nn then
        error("metadata.nn missing")
    end
    if nn.training_path ~= "distillation" then
        error("training_path = " .. tostring(nn.training_path) .. " (want distillation)")
    end
    local lk = nn.hyperparams and nn.hyperparams.loss_kind
    if lk ~= "ce" then
        error("hyperparams.loss_kind = " .. tostring(lk) .. " (want ce)")
    end
    local expect_bundle = "nn/" .. distill_card_id
    if not nn.candle or nn.candle.bundle_ref ~= expect_bundle then
        error(
            "bundle_ref = "
                .. tostring(nn.candle and nn.candle.bundle_ref)
                .. " (want "
                .. expect_bundle
                .. ")"
        )
    end
    return "training_path=distillation loss_kind=ce bundle_ref ok"
end)

-- ── Phase 9: loss curve inspection ──────────────────────────────────
--
-- Random init over a vocab-50257 categorical starts near
-- ln(50257) = 10.825. 40 steps over 12 replicated teacher pairs must
-- pull the final-step loss clearly below the baseline.

run("loss_descended", function()
    if not distill_card then
        error("no distill_card")
    end
    local metrics = distill_card.metadata.nn.metrics
    local final = metrics and metrics.train_loss
    if not final then
        error("metadata.nn.metrics.train_loss missing")
    end
    if final >= LOSS_THRESHOLD then
        error(
            string.format(
                "no descent from random baseline (ln(50257)=%.3f): final=%.3f (threshold %.2f)",
                BASELINE,
                final,
                LOSS_THRESHOLD
            )
        )
    end
    return string.format(
        "final=%.3f (< %.2f threshold, baseline %.3f)",
        final,
        LOSS_THRESHOLD,
        BASELINE
    )
end)

-- ── Phase 10: full-weight safetensors on disk ───────────────────────
--
-- run_distill writes full student weights at <nn_dir>/<card_id>
-- .safetensors (unlike the LoRA delta-only layout). nn_dir defaults
-- to ~/.algocline/nn for a production MCP server.

run("ckpt_file_exists", function()
    if not distill_card_id then
        error("no distill_card_id")
    end
    local home = os.getenv("HOME")
    if not home then
        error("HOME unset")
    end
    local path = home .. "/.algocline/nn/" .. distill_card_id .. ".safetensors"
    local f = io.open(path, "r")
    if not f then
        error("safetensors not on disk: " .. path)
    end
    local size = f:seek("end")
    f:close()
    if not size or size < 1024 then
        error("safetensors too small: " .. tostring(size))
    end
    return string.format("%d bytes at %s", size, path)
end)

-- ── Phase 11: custom-variant Card reload round-trip ─────────────────
--
-- Custom cards record their full shape under metadata.nn.candle
-- .custom, so load_handle rebuilds the config from the Card and
-- reloads the trained weights standalone (no base handle needed).

run("custom_load_handle_roundtrip", function()
    if not distill_card_id then
        error("no distill_card_id")
    end
    local reloaded = alc.nn.card.load_handle(distill_card_id)
    if not reloaded then
        error("load_handle returned nil")
    end
    if reloaded:vocab() ~= 50257 or reloaded:ctx() ~= CTX_LEN then
        error(
            string.format(
                "reloaded shape mismatch: vocab=%d ctx=%d (want 50257/%d)",
                reloaded:vocab(),
                reloaded:ctx(),
                CTX_LEN
            )
        )
    end
    if reloaded:layers() ~= 2 or reloaded:dim() ~= 64 then
        error(
            string.format(
                "reloaded shape mismatch: layers=%d dim=%d (want 2/64)",
                reloaded:layers(),
                reloaded:dim()
            )
        )
    end
    local session = reloaded:generate_session({ 1, 2, 3 })
    local logits = session:next_logits()
    local tok = logits:argmax()
    if type(tok) ~= "number" then
        error("generate_session produced no token")
    end
    return string.format("reloaded standalone, shape ok, 1 greedy token = %d", tok)
end)

-- ── Phase 12: mask boundary guard (FullyMaskedRow protection) ───────
--
-- A row whose prompt exhausts ctx_len leaves no scored position, so
-- from_card must refuse it loudly instead of training on nothing.

run("mask_boundary_guard", function()
    local created = alc.card.create({
        pkg = { name = "alc_nn" },
        metadata = { kind = "teacher_log", loss_mask = "response" },
    })
    local long_prompt = string.rep("alpha beta gamma delta ", 20)
    alc.card.write_samples(created.card_id, {
        { prompt = long_prompt, response = "short answer" },
    })
    local ok, err = pcall(function()
        alc.nn.data.from_card(created.card_id, {
            tokenizer = "gpt2",
            batch_size = 1,
            ctx_len = 8,
        })
    end)
    if ok then
        error("prompt-exhausted row should have BLOCKED")
    end
    if not tostring(err):find("alc.nn.data.from_card", 1, true) then
        error("guard error lacked from_card prefix: " .. tostring(err):sub(1, 200))
    end
    return "prompt-exhausted row refused with from_card prefix"
end)

-- ── Phase 13: strict validation error path (loud-error contract) ────

run("strict_validation_prefixes", function()
    if not student or not dataset then
        error("prereq missing")
    end
    local PREFIX = "alc.nn.trainer.run_distill"

    -- (a) missing lr -> prefix + opts.lr cited
    local ok1, err1 = pcall(function()
        alc.nn.trainer.run_distill(student, dataset, {
            batch = 1,
            steps = 5,
        })
    end)
    if ok1 then
        error("missing lr should have BLOCKED")
    end
    if not tostring(err1):find(PREFIX, 1, true) or not tostring(err1):find("opts.lr", 1, true) then
        error("missing-lr error contract broke: " .. tostring(err1):sub(1, 200))
    end

    -- (b) unknown loss_kind -> prefix + 'ce' vocabulary hint
    local ok2, err2 = pcall(function()
        alc.nn.trainer.run_distill(student, dataset, {
            lr = LR,
            batch = 1,
            steps = 5,
            loss_kind = "kl",
        })
    end)
    if ok2 then
        error("loss_kind=kl should have BLOCKED")
    end
    if not tostring(err2):find(PREFIX, 1, true) or not tostring(err2):find("'ce'", 1, true) then
        error("unknown-loss_kind error contract broke: " .. tostring(err2):sub(1, 200))
    end

    -- (c) unknown schedule -> prefix + vocabulary hint
    local ok3, err3 = pcall(function()
        alc.nn.trainer.run_distill(student, dataset, {
            lr = LR,
            batch = 1,
            steps = 5,
            schedule = "linear",
        })
    end)
    if ok3 then
        error("schedule=linear should have BLOCKED")
    end
    if
        not tostring(err3):find(PREFIX, 1, true)
        or not tostring(err3):find("must be one of", 1, true)
    then
        error("unknown-schedule error contract broke: " .. tostring(err3):sub(1, 200))
    end

    return "3/3 strict rules BLOCKED with run_distill prefix"
end)

-- ── verdict aggregation ─────────────────────────────────────────────

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
    teacher_card_id = teacher_card_id,
    distill_card_id = distill_card_id,
}
