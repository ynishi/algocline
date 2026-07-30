-- alc.nn Full FT end-to-end verification on Tiny Shakespeare.
--
-- Unlike the in-harness `nn_full_ft_smoke.lua` (which runs inside a
-- Rust integration test with a synthetic 4-row corpus), this script
-- drives the public MCP surface via `alc_run` and verifies 15
-- observable properties on a real corpus:
--
--   Setup / correctness (1-10)
--     1  corpus_fetch                — Tiny Shakespeare cached / fetched via curl
--     2  tokenize_and_chunk          — 62-codepoint char vocab that fits vocab=64
--     3  preset_build                — alc.nn.preset.gpt2("tiny") from scratch
--     4  dataset_build               — alc.nn.data.synthetic on 10k rows
--     5  train_completed             — alc.nn.trainer.full_ft 500 steps
--     6  loss_descended              — final loss < 3.2 (< ln(64) baseline 4.158)
--     7  ckpt_file_exists            — safetensors landed on disk
--     8  card_saved                  — alc.nn.trainer.run_full_ft registers a Card
--     9  card_reload                 — alc.nn.card.load_handle round-trips
--     10 reloaded_generates          — reloaded handle:generate_session works
--
--   Runtime-param reflection (11-13)
--     11 greedy_determinism          — same prompt → byte-identical tokens
--     12 sampler_param_reflected     — greedy vs alc.nn.sampler.temperature(1.5)
--                                       diverges within a few tokens
--     13 top_k_1_equals_greedy       — alc.nn.sampler.top_k_top_p(1, ...) == greedy
--
--   Training hyperparam reflection (14)
--     14 train_lr_reflected          — lr=1e-5 vs lr=3e-3 produce measurably
--                                       different final losses (Δ >= 0.05)
--
--   Loud-error contract (15)
--     15 strict_validation_prefixes  — run_full_ft opts.lr missing / opts.schedule
--                                       unknown / opts.warmup < 0 all BLOCK with
--                                       the surface-specific error prefix
--                                       "alc.nn.trainer.run_full_ft"
--
-- Runs in ~5 s on CPU (M-series). Requires `alc` built with feature
-- `nn` and the MCP server up. See `docs/nn-e2e-runbook.md` for the
-- full invocation recipe and expected output.
--
-- ctx (all optional, JSON object passed to alc_run):
--   cache_dir  — directory for the fetched corpus + safetensors ckpts
--                (default: "target/nn-e2e")
--   corpus_url — override the Tiny Shakespeare URL (default: karpathy
--                char-rnn mirror on GitHub raw)

-- ctx is exposed by `alc_run`; when the caller passes no `ctx`
-- argument the global comes through as a non-table userdata that
-- cannot be indexed. Read fields via pcall so both cases work.
local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

local CACHE_DIR = ctx_field("cache_dir") or "target/nn-e2e"
local CORPUS_URL = ctx_field("corpus_url")
    or "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
local CORPUS_CACHE = CACHE_DIR .. "/tiny-shakespeare.txt"
local CKPT_DIR = CACHE_DIR .. "/ckpts"

local STEPS = 500
local BATCH_SIZE = 16
local CTX_LEN = 16
local ROW_LEN = CTX_LEN
local MAX_ROWS = 10000
local LR = 3e-4

local function log(msg)
    alc.log("info", "[nn-e2e] " .. msg)
end

-- Char vocabulary (62 chars + PAD=0 + BOS=1) fits in vocab=64.
local vocab = { [0] = "\0", [1] = "\1", [2] = "\n", [3] = " " }
vocab[4], vocab[5], vocab[6], vocab[7] = ".", ",", "!", "?"
vocab[8], vocab[9], vocab[10] = ":", ";", "'"
for i = 0, 25 do
    vocab[11 + i] = string.char(65 + i)
end
for i = 0, 25 do
    vocab[37 + i] = string.char(97 + i)
end
local char_to_id = {}
for id, ch in pairs(vocab) do
    if id >= 2 then
        char_to_id[ch] = id
    end
end
local SPACE_ID = char_to_id[" "]

-- Ensure cache dirs exist (best-effort, ignore failures).
os.execute("mkdir -p " .. CACHE_DIR)
os.execute("mkdir -p " .. CKPT_DIR)

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

-- ── Phase 1: corpus fetch ───────────────────────────────────────────

local corpus
run("corpus_fetch", function()
    local f = io.open(CORPUS_CACHE, "r")
    if f then
        local c = f:read("*a")
        f:close()
        if c and #c > 1024 then
            corpus = c
            return string.format("cached %d bytes", #c)
        end
    end
    log("fetching corpus from " .. CORPUS_URL)
    local pipe = io.popen("curl -sL '" .. CORPUS_URL .. "'")
    if not pipe then
        error("curl unavailable")
    end
    local c = pipe:read("*a")
    pipe:close()
    if not c or #c < 1024 then
        error("fetched too small: " .. tostring(#c or 0))
    end
    local out = io.open(CORPUS_CACHE, "w")
    if out then
        out:write(c)
        out:close()
    end
    corpus = c
    return string.format("fetched %d bytes", #c)
end)

-- ── Phase 2: tokenize + row chunk ───────────────────────────────────

local rows
run("tokenize_and_chunk", function()
    if not corpus then
        error("no corpus")
    end
    local ids = {}
    for i = 1, #corpus do
        local ch = corpus:sub(i, i)
        ids[#ids + 1] = char_to_id[ch] or SPACE_ID
    end
    local r = {}
    for start = 1, #ids - ROW_LEN + 1, ROW_LEN do
        local row = {}
        for j = 0, ROW_LEN - 1 do
            row[j + 1] = ids[start + j]
        end
        r[#r + 1] = row
        if #r >= MAX_ROWS then
            break
        end
    end
    rows = r
    return string.format("%d ids -> %d rows x %d", #ids, #r, ROW_LEN)
end)

-- ── Phase 3: preset build ───────────────────────────────────────────

local handle
run("preset_build", function()
    handle = alc.nn.preset.gpt2("tiny", {
        device = "cpu",
        dtype = "f32",
        pretrained = false,
    })
    if not handle then
        error("preset returned nil")
    end
    return string.format(
        "variant=%s vocab=%d ctx=%d layers=%d heads=%d dim=%d",
        handle:variant(),
        handle:vocab(),
        handle:ctx(),
        handle:layers(),
        handle:heads(),
        handle:dim()
    )
end)

-- ── Phase 4: dataset build ──────────────────────────────────────────

local dataset
run("dataset_build", function()
    if not rows then
        error("no rows")
    end
    dataset = alc.nn.data.synthetic(rows, {
        batch_size = BATCH_SIZE,
        ctx_len = CTX_LEN,
        shuffle = true,
        pad_id = 0,
    })
    if not dataset then
        error("dataset returned nil")
    end
    return string.format("synthetic %d rows batch=%d", #rows, BATCH_SIZE)
end)

-- ── Phase 5: train loop ─────────────────────────────────────────────

local ckpt
run("train_completed", function()
    if not handle or not dataset then
        error("prereq missing")
    end
    log(string.format("Full FT: %d steps, lr=%g, batch=%d, ctx=%d", STEPS, LR, BATCH_SIZE, CTX_LEN))
    ckpt = alc.nn.trainer.full_ft(handle, dataset, {
        steps = STEPS,
        lr = LR,
        ckpt_dir = CKPT_DIR,
        ckpt_prefix = "shakespeare-tiny",
        ckpt_keep = 3,
        log_every = 50,
        schedule = "cosine_with_warmup",
        warmup_steps = 50,
    })
    if not ckpt then
        error("full_ft returned nil")
    end
    return "ckpt table returned"
end)

-- ── Phase 6: loss curve inspection ──────────────────────────────────
--
-- A random-init tiny model over a vocab-64 categorical starts near
-- ln(64) ~= 4.158. A final loss below 3.2 is comfortable evidence
-- that gradients flowed. The threshold accounts for shuffle-induced
-- run-to-run variance (~+/-0.1 on tiny CPU).

run("loss_descended", function()
    if not ckpt or type(ckpt) ~= "table" then
        error("no ckpt table")
    end
    local final = ckpt.train_loss
    local best = ckpt.metrics and ckpt.metrics.min_train_loss
    if not final then
        error("ckpt.train_loss missing")
    end
    if final >= 3.2 then
        error(
            string.format(
                "no descent from random baseline (ln(64)=4.158): "
                    .. "final=%.3f best=%s (threshold 3.2)",
                final,
                tostring(best)
            )
        )
    end
    return string.format(
        "final=%.3f best=%s (< 3.2 threshold, ln(vocab)=4.158)",
        final,
        tostring(best)
    )
end)

-- ── Phase 7: ckpt file on disk ──────────────────────────────────────

run("ckpt_file_exists", function()
    local p = io.popen("ls " .. CKPT_DIR .. "/*.safetensors 2>/dev/null | head -3")
    if not p then
        error("ls failed")
    end
    local out = p:read("*a")
    p:close()
    if not out or #out < 5 then
        error("no safetensors in " .. CKPT_DIR)
    end
    local count = 0
    for _ in out:gmatch("%.safetensors\n?") do
        count = count + 1
    end
    return string.format("%d safetensors present", count)
end)

-- ── Phase 8: Card save (one-call surface, run_full_ft) ──────────────
--
-- alc.nn.trainer.full_ft (Phase 5) writes raw safetensors but does
-- not register a Card. alc.nn.trainer.run_full_ft trains, writes
-- <nn_dir>/<card_id>.safetensors, and registers the Card in one call
-- -- the designed path for "train and get a loadable Card". Here it
-- runs a few extra steps on the already-trained handle so the Card
-- holds the trained weights.

local card_id
run("card_saved", function()
    if not handle or not dataset then
        error("prereq missing")
    end
    card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
        lr = LR,
        batch = BATCH_SIZE,
        steps = 10,
        warmup = 0,
        schedule = "Constant",
        name = "shakespeare-tiny-v1",
    })
    if not card_id or type(card_id) ~= "string" or #card_id == 0 then
        error("card_id missing/empty: " .. tostring(card_id))
    end
    return "card_id = " .. card_id
end)

-- ── Phase 9: Card round-trip (load_handle) ──────────────────────────

run("card_reload", function()
    if not card_id then
        error("no card_id")
    end
    local reloaded = alc.nn.card.load_handle(card_id)
    if not reloaded then
        error("load_handle returned nil")
    end
    return "NnHandle reloaded"
end)

-- ── Phase 10: generation from reloaded Card (stateless session) ─────
--
-- Trainable handles (Gpt2 / TinyLlama) expose `generate_session` on
-- a stateless backend -- every next_logits re-forwards the full
-- history (no KV cache), capped at the model context window.

run("reloaded_generates", function()
    if not card_id then
        error("no card_id")
    end
    local reloaded = alc.nn.card.load_handle(card_id)
    local prompt = { char_to_id["T"], char_to_id["h"], char_to_id["e"], SPACE_ID }
    local session = reloaded:generate_session(prompt)
    local out = ""
    for _ = 1, 8 do
        local logits = session:next_logits()
        local tok = logits:argmax()
        session:append(tok)
        out = out .. (vocab[tok] or "?")
    end
    return string.format("8 greedy tokens after %q: %q", "The ", out)
end)

-- ── Phase 11: greedy determinism ────────────────────────────────────
--
-- The greedy stateless backend is fully deterministic: two
-- independent sessions with the same card + same prompt must produce
-- byte-identical tokens. Pins down the "no hidden randomness in the
-- stateless backend" contract.

local function greedy_run(prompt, n)
    local reloaded = alc.nn.card.load_handle(card_id)
    local session = reloaded:generate_session(prompt)
    local ids = {}
    for _ = 1, n do
        local logits = session:next_logits()
        local tok = logits:argmax()
        session:append(tok)
        ids[#ids + 1] = tok
    end
    return ids
end

run("greedy_determinism", function()
    if not card_id then
        error("no card_id")
    end
    local prompt = { char_to_id["T"], char_to_id["h"], char_to_id["e"], SPACE_ID }
    local a = greedy_run(prompt, 12)
    local b = greedy_run(prompt, 12)
    for i = 1, 12 do
        if a[i] ~= b[i] then
            error(string.format("greedy diverged at pos %d: %d vs %d", i, a[i], b[i]))
        end
    end
    return "12 tokens identical across 2 runs"
end)

-- ── Phase 12: sampler param reflection (greedy vs temperature) ──────
--
-- Same prompt, two sampler configurations. Greedy = deterministic
-- argmax. Temperature sampler with a fixed seed diverges from greedy
-- within a handful of tokens on a non-degenerate distribution. Same
-- output on both = sampler param not reflected, or distribution
-- collapsed to a single token (regression from Phase 6).
--
-- 8-token window keeps the total session (prompt 4 + gen 8 = 12)
-- inside the tiny preset's ctx=16.

run("sampler_param_reflected", function()
    if not card_id then
        error("no card_id")
    end
    local prompt = { char_to_id["T"], char_to_id["h"], char_to_id["e"], SPACE_ID }

    local h1 = alc.nn.card.load_handle(card_id)
    local s1 = h1:generate_session(prompt)
    local greedy = alc.nn.sampler.greedy()
    local greedy_ids = {}
    for _ = 1, 8 do
        local logits = s1:next_logits()
        local tok = greedy:sample(logits)
        s1:append(tok)
        greedy_ids[#greedy_ids + 1] = tok
    end

    local h2 = alc.nn.card.load_handle(card_id)
    local s2 = h2:generate_session(prompt)
    local temp = alc.nn.sampler.temperature(1.5, 12345)
    local sampled_ids = {}
    for _ = 1, 8 do
        local logits = s2:next_logits()
        local tok = temp:sample(logits)
        s2:append(tok)
        sampled_ids[#sampled_ids + 1] = tok
    end

    local diverge_at = nil
    for i = 1, 8 do
        if greedy_ids[i] ~= sampled_ids[i] then
            diverge_at = i
            break
        end
    end
    if not diverge_at then
        error(
            "greedy == temperature(1.5) over 8 tokens - sampler param not reflected or degenerate distribution"
        )
    end
    return string.format(
        "greedy vs temperature(1.5) diverge at pos %d (within 8 tokens)",
        diverge_at
    )
end)

-- ── Phase 13: sampler top_k contract ────────────────────────────────
--
-- top_k = 1 must be equivalent to greedy for the same prompt (the
-- top-1 truncation collapses the distribution to a single token
-- regardless of temperature / seed).

run("top_k_1_equals_greedy", function()
    if not card_id then
        error("no card_id")
    end
    local prompt = { char_to_id["T"], char_to_id["h"], char_to_id["e"], SPACE_ID }
    local greedy_ids = greedy_run(prompt, 12)

    local h = alc.nn.card.load_handle(card_id)
    local s = h:generate_session(prompt)
    local top1 = alc.nn.sampler.top_k_top_p(1, 1.0, 1.0, 99)
    local ids = {}
    for _ = 1, 12 do
        local logits = s:next_logits()
        local tok = top1:sample(logits)
        s:append(tok)
        ids[#ids + 1] = tok
    end
    for i = 1, 12 do
        if greedy_ids[i] ~= ids[i] then
            error(
                string.format(
                    "top_k=1 diverged from greedy at pos %d: %d vs %d",
                    i,
                    greedy_ids[i],
                    ids[i]
                )
            )
        end
    end
    return "top_k=1 identical to greedy over 12 tokens"
end)

-- ── Phase 14: train hyperparam reflection (lr axis) ─────────────────
--
-- Two fresh handles, identical dataset + steps + prompt, only lr
-- differs. Final losses must differ meaningfully -- the "hyperparam
-- 1 axis swept, behavior changes" signal applied to the
-- run_full_ft path.
--
-- Fresh datasets are constructed here because the shared `dataset`
-- above has been consumed by Phase 5 (500 steps) + Phase 8 (10
-- steps) and cannot serve another 100 steps.

local function final_loss_of(card_id_str)
    local card = alc.card.get(card_id_str)
    if not card then
        error("alc.card.get returned nil for " .. tostring(card_id_str))
    end
    local metrics = card.metadata and card.metadata.nn and card.metadata.nn.metrics
    return metrics and metrics.train_loss
end

run("train_lr_reflected", function()
    if not rows then
        error("no rows")
    end
    local ds_lo = alc.nn.data.synthetic(rows, {
        batch_size = BATCH_SIZE,
        ctx_len = CTX_LEN,
        shuffle = true,
        pad_id = 0,
    })
    local h_lo = alc.nn.preset.gpt2("tiny", { device = "cpu", dtype = "f32", pretrained = false })
    local id_lo = alc.nn.trainer.run_full_ft(h_lo, ds_lo, {
        lr = 1e-5,
        batch = BATCH_SIZE,
        steps = 100,
        warmup = 0,
        schedule = "Constant",
        name = "shakespeare-lr-lo-" .. tostring(os.time()),
    })
    local loss_lo = final_loss_of(id_lo)

    local ds_hi = alc.nn.data.synthetic(rows, {
        batch_size = BATCH_SIZE,
        ctx_len = CTX_LEN,
        shuffle = true,
        pad_id = 0,
    })
    local h_hi = alc.nn.preset.gpt2("tiny", { device = "cpu", dtype = "f32", pretrained = false })
    local id_hi = alc.nn.trainer.run_full_ft(h_hi, ds_hi, {
        lr = 3e-3,
        batch = BATCH_SIZE,
        steps = 100,
        warmup = 0,
        schedule = "Constant",
        name = "shakespeare-lr-hi-" .. tostring(os.time()),
    })
    local loss_hi = final_loss_of(id_hi)

    if not loss_lo or not loss_hi then
        error(
            string.format(
                "metadata.nn.metrics.train_loss missing (lo=%s hi=%s)",
                tostring(loss_lo),
                tostring(loss_hi)
            )
        )
    end
    local delta = math.abs(loss_hi - loss_lo)
    if delta < 0.05 then
        error(
            string.format(
                "lr axis had no measurable effect: lo=%.4f hi=%.4f delta=%.4f",
                loss_lo,
                loss_hi,
                delta
            )
        )
    end
    return string.format("lr=1e-5 -> %.4f  lr=3e-3 -> %.4f  delta=%.4f", loss_lo, loss_hi, delta)
end)

-- ── Phase 15: strict validation error path (loud-error contract) ────
--
-- run_full_ft's opts validation must emit the surface-specific error
-- prefix "alc.nn.trainer.run_full_ft" (one prefix per surface).
-- Drives each strict rule and matches the message.

run("strict_validation_prefixes", function()
    if not handle or not dataset then
        error("prereq missing")
    end

    -- (a) missing lr -> prefix + opts.lr cited
    local ok1, err1 = pcall(function()
        alc.nn.trainer.run_full_ft(handle, dataset, {
            batch = BATCH_SIZE,
            steps = 5,
            warmup = 0,
            schedule = "Constant",
        })
    end)
    if ok1 then
        error("missing lr should have BLOCKED")
    end
    if not tostring(err1):find("alc%.nn%.trainer%.run_full_ft", 1, false) then
        error("missing-lr error lacked run_full_ft prefix: " .. tostring(err1):sub(1, 200))
    end
    if not tostring(err1):find("opts%.lr", 1, false) then
        error("missing-lr error did not cite opts.lr: " .. tostring(err1):sub(1, 200))
    end

    -- (b) unknown schedule -> prefix + "must be one of"
    local ok2, err2 = pcall(function()
        alc.nn.trainer.run_full_ft(handle, dataset, {
            lr = 1e-4,
            batch = BATCH_SIZE,
            steps = 5,
            warmup = 0,
            schedule = "linear",
        })
    end)
    if ok2 then
        error("unknown schedule should have BLOCKED")
    end
    if not tostring(err2):find("alc%.nn%.trainer%.run_full_ft", 1, false) then
        error("unknown-schedule error lacked prefix: " .. tostring(err2):sub(1, 200))
    end
    if not tostring(err2):find("must be one of", 1, false) then
        error("unknown-schedule error missing vocab hint: " .. tostring(err2):sub(1, 200))
    end

    -- (c) negative warmup -> prefix + warmup field
    local ok3, err3 = pcall(function()
        alc.nn.trainer.run_full_ft(handle, dataset, {
            lr = 1e-4,
            batch = BATCH_SIZE,
            steps = 5,
            warmup = -1,
            schedule = "Constant",
        })
    end)
    if ok3 then
        error("negative warmup should have BLOCKED")
    end
    if not tostring(err3):find("alc%.nn%.trainer%.run_full_ft", 1, false) then
        error("negative-warmup error lacked prefix: " .. tostring(err3):sub(1, 200))
    end
    if not tostring(err3):find("warmup", 1, false) then
        error("negative-warmup error missing field name: " .. tostring(err3):sub(1, 200))
    end

    return "3/3 strict rules BLOCKED with run_full_ft prefix"
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
    card_id = card_id,
    cache_dir = CACHE_DIR,
}
