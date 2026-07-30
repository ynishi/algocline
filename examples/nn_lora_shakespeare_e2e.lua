-- alc.nn LoRA end-to-end verification on Tiny Shakespeare.
--
-- Sibling of `nn_fullft_shakespeare_e2e.lua` (Full FT path). This
-- script drives the public MCP surface via `alc_run` and verifies 15
-- observable properties of the LoRA training path on a real corpus:
--
--   Setup / correctness (1-8)
--     1  corpus_fetch                — Tiny Shakespeare cached / fetched via curl
--     2  tokenize_and_chunk          — 62-codepoint char vocab that fits vocab=64
--     3  dataset_build               — alc.nn.data.synthetic on 10k rows
--     4  base_preset_build           — alc.nn.preset.gpt2("tiny") from scratch
--     5  lora_train_completed        — alc.nn.trainer.run_lora_ft trains and
--                                       registers a Card in one call
--     6  lora_card_meta_shape        — training_path="lora", candle.lora carries
--                                       rank / alpha / base_bundle_ref / the
--                                       6-entry default target_modules
--     7  lora_loss_descended         — final loss below the random baseline
--     8  delta_file_exists           — the Δ-only safetensors landed at
--                                       candle.lora.delta_path
--
--   Load / compose contract (9-11)
--     9  load_handle_refusal         — load_handle(lora_card_id) is a designed
--                                       refusal directing to load_wrap
--     10 load_wrap_roundtrip         — load_wrap(lora_card_id, fresh_base)
--                                       returns a wrapped NnHandle
--     11 wrapped_generates           — wrapped handle:generate_session works
--
--   Merge export (12-14)
--     12 merge_export                — merge_lora(wrapped, {name, lora_card})
--                                       registers a "merged" Card with
--                                       lineage.parent = lora_card_id
--     13 merged_load_generates       — load_handle(merged_card_id) works and
--                                       generates (self-contained, no base)
--     14 merged_parity_with_wrapped  — greedy tokens from the merged card are
--                                       byte-identical to the wrapped
--                                       (base + Δ) composition
--
--   Loud-error contract (15)
--     15 strict_validation_prefixes  — run_lora_ft opts.rank missing / unknown
--                                       target module / dropout out of range /
--                                       already-wrapped base all BLOCK with the
--                                       surface-specific error prefix
--                                       "alc.nn.trainer.run_lora_ft"
--
-- Runs in ~5 s on CPU (M-series). Requires `alc` built with feature
-- `nn` and the MCP server up. See `docs/nn-e2e-runbook.md` for the
-- full invocation recipe and expected output.
--
-- ctx (all optional, JSON object passed to alc_run):
--   cache_dir  — directory for the fetched corpus (default: "target/nn-e2e")
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

local STEPS = 500
local BATCH_SIZE = 16
local CTX_LEN = 16
local ROW_LEN = CTX_LEN
local MAX_ROWS = 10000
local LR = 3e-3
local RANK = 4
local ALPHA = 8

local function log(msg)
    alc.log("info", "[nn-lora-e2e] " .. msg)
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

-- Ensure the cache dir exists (best-effort, ignore failures).
os.execute("mkdir -p " .. CACHE_DIR)

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

-- ── Phase 3: dataset build ──────────────────────────────────────────

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

-- ── Phase 4: base preset build ──────────────────────────────────────
--
-- NOTE: run_lora_ft wraps the base model *in place* (the Lua-side
-- userdata keeps reporting unwrapped). This handle is dedicated to
-- Phase 5 and never reused afterwards — later phases build fresh
-- bases per operation.

local base
run("base_preset_build", function()
    base = alc.nn.preset.gpt2("tiny", {
        device = "cpu",
        dtype = "f32",
        pretrained = false,
    })
    if not base then
        error("preset returned nil")
    end
    return string.format(
        "variant=%s vocab=%d ctx=%d layers=%d heads=%d dim=%d",
        base:variant(),
        base:vocab(),
        base:ctx(),
        base:layers(),
        base:heads(),
        base:dim()
    )
end)

-- ── Phase 5: LoRA train (one-call surface, run_lora_ft) ─────────────
--
-- alc.nn.trainer.run_lora_ft trains the Δ, writes the Δ-only
-- safetensors under <nn_dir>/nn/, and registers the Card in one
-- call, returning the card_id. target_modules is omitted so the
-- per-arch canonical default (6 entries for gpt2) applies.

local lora_card_id
run("lora_train_completed", function()
    if not base or not dataset then
        error("prereq missing")
    end
    log(
        string.format(
            "LoRA FT: %d steps, rank=%d, alpha=%d, lr=%g, batch=%d",
            STEPS,
            RANK,
            ALPHA,
            LR,
            BATCH_SIZE
        )
    )
    lora_card_id = alc.nn.trainer.run_lora_ft(base, dataset, {
        rank = RANK,
        alpha = ALPHA,
        lr = LR,
        batch = BATCH_SIZE,
        steps = STEPS,
        warmup = 0,
        schedule = "Constant",
        name = "shakespeare-lora-v1",
    })
    if not lora_card_id or type(lora_card_id) ~= "string" or #lora_card_id == 0 then
        error("card_id missing/empty: " .. tostring(lora_card_id))
    end
    return "card_id = " .. lora_card_id
end)

-- ── Phase 6: LoRA Card metadata shape ───────────────────────────────

local lora_card
run("lora_card_meta_shape", function()
    if not lora_card_id then
        error("no lora_card_id")
    end
    lora_card = alc.card.get(lora_card_id)
    if not lora_card then
        error("alc.card.get returned nil")
    end
    local nn = lora_card.metadata and lora_card.metadata.nn
    if not nn then
        error("metadata.nn missing")
    end
    if nn.training_path ~= "lora" then
        error("training_path = " .. tostring(nn.training_path) .. " (want lora)")
    end
    local lora = nn.candle and nn.candle.lora
    if not lora then
        error("candle.lora branch missing")
    end
    if lora.rank ~= RANK then
        error("rank = " .. tostring(lora.rank) .. " (want " .. RANK .. ")")
    end
    if lora.alpha ~= ALPHA then
        error("alpha = " .. tostring(lora.alpha) .. " (want " .. ALPHA .. ")")
    end
    if lora.base_bundle_ref ~= "nn/gpt2-tiny" then
        error("base_bundle_ref = " .. tostring(lora.base_bundle_ref))
    end
    local n_targets = lora.target_modules and #lora.target_modules or 0
    if n_targets ~= 6 then
        error("target_modules count = " .. n_targets .. " (want 6 gpt2 defaults)")
    end
    local expect_bundle = "nn/" .. lora_card_id
    if nn.candle.bundle_ref ~= expect_bundle then
        error(
            "bundle_ref = " .. tostring(nn.candle.bundle_ref) .. " (want " .. expect_bundle .. ")"
        )
    end
    return string.format(
        "training_path=lora rank=%d alpha=%d targets=%d base=%s",
        lora.rank,
        lora.alpha,
        n_targets,
        lora.base_bundle_ref
    )
end)

-- ── Phase 7: loss curve inspection ──────────────────────────────────
--
-- A random-init tiny model over a vocab-64 categorical starts near
-- ln(64) ~= 4.158. LoRA has less capacity than Full FT here: only
-- the 6 projection targets carry Δ while the embedding and LM head
-- stay frozen at random init, so the achievable descent is smaller.
-- Any final loss below 4.05 is evidence that Δ gradients flowed.

run("lora_loss_descended", function()
    if not lora_card then
        error("no lora_card")
    end
    local metrics = lora_card.metadata.nn.metrics
    local final = metrics and metrics.train_loss
    if not final then
        error("metadata.nn.metrics.train_loss missing")
    end
    if final >= 4.05 then
        error(
            string.format(
                "no descent from random baseline (ln(64)=4.158): final=%.3f (threshold 4.05)",
                final
            )
        )
    end
    return string.format("final=%.3f (< 4.05 threshold, ln(vocab)=4.158)", final)
end)

-- ── Phase 8: Δ-only safetensors on disk ─────────────────────────────
--
-- A LoRA card has *no* <nn_dir>/<card_id>.safetensors — the
-- candle.bundle_ref is a logical reference only. The actual file is
-- the Δ at candle.lora.delta_path (absolute).

run("delta_file_exists", function()
    if not lora_card then
        error("no lora_card")
    end
    local delta_path = lora_card.metadata.nn.candle.lora.delta_path
    if not delta_path or type(delta_path) ~= "string" then
        error("candle.lora.delta_path missing")
    end
    local f = io.open(delta_path, "r")
    if not f then
        error("delta file not on disk: " .. delta_path)
    end
    local size = f:seek("end")
    f:close()
    if not size or size < 64 then
        error("delta file too small: " .. tostring(size))
    end
    return string.format("delta %d bytes at %s", size, delta_path)
end)

-- ── Phase 9: load_handle refusal (designed error path) ──────────────
--
-- load_handle on a lora card must refuse with a typed message that
-- names the training_path and directs the caller to load_wrap.

run("load_handle_refusal", function()
    if not lora_card_id then
        error("no lora_card_id")
    end
    local ok, err = pcall(function()
        alc.nn.card.load_handle(lora_card_id)
    end)
    if ok then
        error("load_handle(lora_card) should have BLOCKED")
    end
    local msg = tostring(err)
    if not msg:find('training_path="lora"', 1, true) then
        error("refusal did not name training_path: " .. msg:sub(1, 200))
    end
    if not msg:find("load_wrap", 1, true) then
        error("refusal did not direct to load_wrap: " .. msg:sub(1, 200))
    end
    return "refused with load_wrap direction"
end)

-- ── Phase 10: load_wrap round-trip ──────────────────────────────────
--
-- A *fresh* base is required: load_wrap mutates the base model in
-- place, so the Phase 5 handle (already wrapped internally) would
-- double-wrap and fail.

local wrapped
run("load_wrap_roundtrip", function()
    if not lora_card_id then
        error("no lora_card_id")
    end
    local fresh_base = alc.nn.preset.gpt2("tiny", {
        device = "cpu",
        dtype = "f32",
        pretrained = false,
    })
    wrapped = alc.nn.card.load_wrap(lora_card_id, fresh_base)
    if not wrapped then
        error("load_wrap returned nil")
    end
    return "wrapped NnHandle returned"
end)

-- ── Phase 11: generation from the wrapped handle ────────────────────

local function greedy_tokens(handle, prompt, n)
    local session = handle:generate_session(prompt)
    local ids = {}
    for _ = 1, n do
        local logits = session:next_logits()
        local tok = logits:argmax()
        session:append(tok)
        ids[#ids + 1] = tok
    end
    return ids
end

local PROMPT = { char_to_id["T"], char_to_id["h"], char_to_id["e"], SPACE_ID }

run("wrapped_generates", function()
    if not wrapped then
        error("no wrapped handle")
    end
    local ids = greedy_tokens(wrapped, PROMPT, 8)
    local out = ""
    for _, tok in ipairs(ids) do
        out = out .. (vocab[tok] or "?")
    end
    return string.format("8 greedy tokens after %q: %q", "The ", out)
end)

-- ── Phase 12: merge export ──────────────────────────────────────────
--
-- merge_lora consumes the *load_wrap output* (the handle fed into
-- run_lora_ft never flips its Lua-side wrapped flag and is refused).
-- The merged card is self-contained: training_path="merged",
-- lineage.parent = the lora card.

local merged_card_id
run("merge_export", function()
    if not wrapped or not lora_card_id then
        error("prereq missing")
    end
    merged_card_id = alc.nn.card.merge_lora(wrapped, {
        name = "shakespeare-lora-merged-v1",
        lora_card = lora_card_id,
    })
    if not merged_card_id or type(merged_card_id) ~= "string" or #merged_card_id == 0 then
        error("merged card_id missing/empty: " .. tostring(merged_card_id))
    end
    local card = alc.card.get(merged_card_id)
    if not card then
        error("alc.card.get(merged) returned nil")
    end
    local nn = card.metadata and card.metadata.nn
    if not nn or nn.training_path ~= "merged" then
        error("training_path = " .. tostring(nn and nn.training_path) .. " (want merged)")
    end
    local parent = nn.lineage and nn.lineage.parent
    if parent ~= lora_card_id then
        error("lineage.parent = " .. tostring(parent) .. " (want " .. lora_card_id .. ")")
    end
    return string.format("merged card_id = %s (parent = %s)", merged_card_id, parent)
end)

-- ── Phase 13: merged card loads standalone ──────────────────────────
--
-- The merged card goes through load_handle (no base needed) — the
-- Δ has been materialised into plain weights.

local merged_handle
run("merged_load_generates", function()
    if not merged_card_id then
        error("no merged_card_id")
    end
    merged_handle = alc.nn.card.load_handle(merged_card_id)
    if not merged_handle then
        error("load_handle(merged) returned nil")
    end
    local ids = greedy_tokens(merged_handle, PROMPT, 8)
    local out = ""
    for _, tok in ipairs(ids) do
        out = out .. (vocab[tok] or "?")
    end
    return string.format("8 greedy tokens after %q: %q", "The ", out)
end)

-- ── Phase 14: merged ≡ wrapped parity ───────────────────────────────
--
-- The merged (base + Δ materialised) weights must produce the same
-- greedy trajectory as the wrapped (base + Δ composed at forward
-- time) handle. Divergence means the merge lost or distorted the Δ.

run("merged_parity_with_wrapped", function()
    if not wrapped or not merged_handle then
        error("prereq missing")
    end
    local a = greedy_tokens(wrapped, PROMPT, 12)
    local b = greedy_tokens(merged_handle, PROMPT, 12)
    for i = 1, 12 do
        if a[i] ~= b[i] then
            error(string.format("parity broke at pos %d: wrapped=%d merged=%d", i, a[i], b[i]))
        end
    end
    return "12 greedy tokens identical (wrapped vs merged)"
end)

-- ── Phase 15: strict validation error path (loud-error contract) ────
--
-- run_lora_ft's opts validation must emit the surface-specific error
-- prefix "alc.nn.trainer.run_lora_ft" (one prefix per surface).
-- Drives each strict rule and matches the message.

run("strict_validation_prefixes", function()
    if not dataset then
        error("prereq missing")
    end
    local fresh = alc.nn.preset.gpt2("tiny", {
        device = "cpu",
        dtype = "f32",
        pretrained = false,
    })
    local PREFIX = "alc.nn.trainer.run_lora_ft"

    -- (a) missing rank -> prefix + opts.rank cited
    local ok1, err1 = pcall(function()
        alc.nn.trainer.run_lora_ft(fresh, dataset, {
            alpha = ALPHA,
            lr = LR,
            batch = BATCH_SIZE,
            steps = 5,
        })
    end)
    if ok1 then
        error("missing rank should have BLOCKED")
    end
    if
        not tostring(err1):find(PREFIX, 1, true) or not tostring(err1):find("opts.rank", 1, true)
    then
        error("missing-rank error contract broke: " .. tostring(err1):sub(1, 200))
    end

    -- (b) unknown target module -> prefix + "unknown target module"
    local ok2, err2 = pcall(function()
        alc.nn.trainer.run_lora_ft(fresh, dataset, {
            rank = RANK,
            alpha = ALPHA,
            target_modules = { "bogus_proj" },
            lr = LR,
            batch = BATCH_SIZE,
            steps = 5,
        })
    end)
    if ok2 then
        error("unknown target module should have BLOCKED")
    end
    if
        not tostring(err2):find(PREFIX, 1, true)
        or not tostring(err2):find("unknown target module", 1, true)
    then
        error("unknown-target error contract broke: " .. tostring(err2):sub(1, 200))
    end

    -- (c) dropout out of range -> prefix + "[0.0, 1.0)"
    local ok3, err3 = pcall(function()
        alc.nn.trainer.run_lora_ft(fresh, dataset, {
            rank = RANK,
            alpha = ALPHA,
            dropout = 1.5,
            lr = LR,
            batch = BATCH_SIZE,
            steps = 5,
        })
    end)
    if ok3 then
        error("dropout=1.5 should have BLOCKED")
    end
    if
        not tostring(err3):find(PREFIX, 1, true)
        or not tostring(err3):find("[0.0, 1.0)", 1, true)
    then
        error("dropout-range error contract broke: " .. tostring(err3):sub(1, 200))
    end

    -- (d) already-wrapped base -> prefix + "drop the wrap first"
    local ok4, err4 = pcall(function()
        alc.nn.trainer.run_lora_ft(wrapped, dataset, {
            rank = RANK,
            alpha = ALPHA,
            lr = LR,
            batch = BATCH_SIZE,
            steps = 5,
        })
    end)
    if ok4 then
        error("wrapped base should have BLOCKED")
    end
    if
        not tostring(err4):find(PREFIX, 1, true)
        or not tostring(err4):find("drop the wrap first", 1, true)
    then
        error("wrapped-base error contract broke: " .. tostring(err4):sub(1, 200))
    end

    return "4/4 strict rules BLOCKED with run_lora_ft prefix"
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
    lora_card_id = lora_card_id,
    merged_card_id = merged_card_id,
    cache_dir = CACHE_DIR,
}
