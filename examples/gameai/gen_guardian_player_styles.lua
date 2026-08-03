-- Generate the guardian-duel player-style play logs for the player pool.
--
-- Self-contained script for `alc_run` (`code_file` form). It writes two
-- new corpora — "bruiser" and "shieldbearer" — in the same entry shape
-- the bake path reads, so `bake_guardian_player_from_log.lua` can turn
-- each into a Card without a single change on its side.
--
-- ## Why a second generator instead of extending the first one
--
-- `gen_guardian_sample_playlog.lua` drives an interactive session and
-- lets the boss seat be decoded from a Card. That is the right shape for
-- the sample corpus, but it drags a Card dependency into a job that has
-- no learning in it: these two styles are rules over the player view and
-- the boss they meet is a teacher policy. So this script walks the rules
-- directly — `new_game` / `player_view` / `apply` plus
-- `policy_<boss>` — and stays pure Lua and deterministic: no Card, no
-- `alc.llm`, no RNG anywhere on the path.
--
-- The two paths are not asserted to be equivalent by hand-waving. This
-- script carries the sentinel rule of the first generator verbatim and
-- can replay it over the same twelve openings; passing
-- `ctx.verify_sentinel_path` makes the run compare its own sentinel
-- output against the committed
-- `data/guardian_sample_playlog_train.json` entry by entry (all nine
-- view fields plus the move). A mismatch is loud, which turns "the new
-- loop reproduces the old corpus" into a checked property rather than a
-- claim. The comparison is on decoded values, not bytes: JSON key order
-- is not part of the corpus meaning.
--
-- ## The three styles
--
-- Each style is a list of rules over the player view, walked in order,
-- first match wins. A rule whose move is `"script"` yields to the
-- per-game opening script — the only source of trajectory diversity, as
-- in the sample corpus.
--
--   sentinel (reproduction only, no file written)
--     R1  intent == "t"           -> "b"   block the telegraphed slam
--     R2  intent == "f" or "w"    -> "b"   block the heavy cycle hits
--     R3  intent == "c" or "d"    -> "A"   boss deals 0 that turn
--     R4  intent == "v"           -> "a"   attack through the weaken
--     R5  no intent, mode == 1    -> "p"   poke the rolled-up boss
--     R6  no intent, mode == 0    -> script
--
--   bruiser — intent-blind attacker. It never pokes, so its intent
--   field is always "-" and it cannot know when the slam lands.
--     B1  spikes                  -> "A"   swing inside the mode-1 window
--     B2  otherwise               -> script over {a, A}
--
--   shieldbearer — reads what it is shown, never steps in.
--     S1  intent in {t, f, w, v}  -> "b"   soak the visible attack
--     S2  intent in {c, d}        -> "a"   pay a small price on a free turn
--     S3  no intent               -> script over {a, p}
--
-- B1 is the interesting one. `spikes` is up exactly while the boss is in
-- mode 1 (only `d` raises the thorns and enters the mode, only `t` clears
-- both), and a mode-1 boss plays the fixed sequence {d, d, t}: two turns
-- of zero damage and then the slam. An attacker that cannot see the slam
-- coming has nothing to buy by defending, so it spends the window on the
-- heavy attack, pays the 3 thorns and comes out ahead. Blocking there
-- instead measured *worse* than having no rule at all.
--
-- Neither new style reads `weakened` or `exposed`; those two view fields
-- are unused by all three rule sets, which is worth stating out loud
-- because the bake still encodes them.
--
-- ## Openings
--
-- The twelve train openings are the ones the sample corpus uses,
-- character-substituted per style so that each style's script slot stays
-- inside that style's own alphabet: bruiser replaces `p` with `a` (it
-- does not poke), shieldbearer replaces `A` with `a` (it does not step
-- in), sentinel takes them unchanged. Deriving the openings instead of
-- inventing twelve new strings keeps the three corpora comparable: the
-- trajectory diversity has the same source in all of them.
--
-- Twelve openings x three canonical bosses = 36 games per style. Every
-- game gets its own `new_game` seed (`style index * 1000 + boss index *
-- 100 + variant index`). The seed does not change how a fight plays —
-- the engine has no RNG and the opening is fixed — it is carried so a
-- logged game can be named.
--
-- ## Contract
--
--   alc_run(
--     code_file = "<repo>/examples/gameai/gen_guardian_player_styles.lua",
--     ctx = {
--       -- required
--       output_dir = "<repo>/examples/gameai/data",
--       -- optional
--       verify_sentinel_path =
--         "<repo>/examples/gameai/data/guardian_sample_playlog_train.json",
--     },
--   )
--
-- Writes `<output_dir>/guardian_bruiser_playlog_train.json` and
-- `<output_dir>/guardian_shield_playlog_train.json`, each a bare array
-- of `{ player = <view>, player_action = <move> }` entries — exactly the
-- two fields the bake path reads, one entry per line, as the committed
-- sample corpus is laid out. The sentinel run writes no file: its output
-- exists to be compared, not to be shipped a second time.
--
-- Returns `{ bruiser_entries, shield_entries, sentinel_entries,
-- sentinel_match, games, outputs, slot_counts, rule_maps }`.
-- `rule_maps` is the same table the rules are evaluated from, so the
-- attribution side can read the slot definitions off the run that
-- produced the corpus instead of keeping a hand copy in step with it.
--
-- Prerequisites: the `guardian_duel` package linked or installed. No
-- Card, no `--features nn`, no host LLM.

local duel = require("guardian_duel")

local BOSSES = { "guardian", "rusher", "turtle" }

-- The twelve train openings of `gen_guardian_sample_playlog.lua`, copied
-- verbatim; the per-style alphabets are derived from them below.
local TRAIN_VARIANTS = {
    "papapapap",
    "apapapapa",
    "pAapAapAa",
    "aapaapaap",
    "pppaaAAap",
    "ApaApaApa",
    "aaapppaaA",
    "paAapaAap",
    "AApaApaap",
    "apppaappp",
    "aApaApaAp",
    "ppApaapAa",
}

--- Rule sets, in evaluation order. Each rule is
--- `{ id, when, match, move }`:
---
--- - `id` is the slot name the attribution side reports against.
--- - `when` is the human-readable condition, kept next to the machine
---   one so a reader never has to decode `match` to know what fires.
--- - `match` is the machine condition. Keys are ANDed: `intent` is a
---   list the (normalised) intent must be in, `no_intent` demands the
---   board showed nothing, `mode` / `spikes` compare the view fields,
---   `otherwise` matches unconditionally.
--- - `move` is the played move, or `"script"` for the opening slot.
---
--- This table is the single source: the loop below evaluates it and the
--- run returns it, so a corpus and its slot definitions cannot drift
--- apart within a run.
local RULE_MAPS = {
    sentinel = {
        { id = "R1", when = 'intent == "t"', match = { intent = { "t" } }, move = "b" },
        { id = "R2", when = 'intent == "f" or "w"', match = { intent = { "f", "w" } }, move = "b" },
        { id = "R3", when = 'intent == "c" or "d"', match = { intent = { "c", "d" } }, move = "A" },
        { id = "R4", when = 'intent == "v"', match = { intent = { "v" } }, move = "a" },
        {
            id = "R5",
            when = "no intent, mode == 1",
            match = { no_intent = true, mode = 1 },
            move = "p",
        },
        {
            id = "R6",
            when = "no intent, mode == 0",
            match = { no_intent = true, mode = 0 },
            move = "script",
        },
    },
    bruiser = {
        { id = "B1", when = "spikes", match = { spikes = true }, move = "A" },
        { id = "B2", when = "otherwise", match = { otherwise = true }, move = "script" },
    },
    shield = {
        {
            id = "S1",
            when = "intent in {t, f, w, v}",
            match = { intent = { "t", "f", "w", "v" } },
            move = "b",
        },
        { id = "S2", when = "intent in {c, d}", match = { intent = { "c", "d" } }, move = "a" },
        { id = "S3", when = 'intent == "-"', match = { no_intent = true }, move = "script" },
    },
}

--- Style table, in the order the run walks them. `seed_index` fixes the
--- per-style seed block, `opening` maps a sample opening onto the
--- style's own alphabet, `alphabet` is that alphabet as a set so a
--- mis-substituted opening fails on the first game rather than showing
--- up as a strange move in the corpus, and `output` names the file
--- (absent for the sentinel replay, which is compared and discarded).
local STYLES = {
    {
        name = "sentinel",
        seed_index = 1,
        opening = function(variant)
            return variant
        end,
        alphabet = { a = true, A = true, p = true },
        output = nil,
    },
    {
        name = "bruiser",
        seed_index = 2,
        opening = function(variant)
            return (variant:gsub("p", "a"))
        end,
        alphabet = { a = true, A = true },
        output = "guardian_bruiser_playlog_train.json",
    },
    {
        name = "shield",
        seed_index = 3,
        opening = function(variant)
            return (variant:gsub("A", "a"))
        end,
        alphabet = { a = true, p = true },
        output = "guardian_shield_playlog_train.json",
    },
}

local VIEW_FIELDS = {
    "turn",
    "mode",
    "boss_hp",
    "shift_distance",
    "hp",
    "weakened",
    "exposed",
    "spikes",
    "intent",
}

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "gen_guardian_player_styles: alc.json_encode is required to write a corpus "
                .. "(host bridge missing)",
            0
        )
    end
    return alc.json_encode
end

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "gen_guardian_player_styles: alc.json_decode is required to read the sentinel "
                .. "corpus (host bridge missing)",
            0
        )
    end
    return alc.json_decode
end

local function log(msg)
    alc.log("info", "[gen-styles] " .. msg)
end

-- ─── ctx ────────────────────────────────────────────────────────────

-- `ctx` is a global injected by alc_run. When the caller passes no ctx
-- the global is a non-table userdata that cannot be indexed, so every
-- read goes through pcall.
local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

local function require_string(field)
    local v = ctx_field(field)
    if type(v) ~= "string" or v == "" then
        error(
            string.format(
                "gen_guardian_player_styles: ctx.%s is required and must be a non-empty "
                    .. "string, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

local function optional_string(field)
    local v = ctx_field(field)
    if v == nil then
        return nil
    end
    if type(v) ~= "string" or v == "" then
        error(
            string.format(
                "gen_guardian_player_styles: ctx.%s must be a non-empty string or nil, got %s",
                field,
                type(v)
            )
        )
    end
    return v
end

--- Join a directory and a file name with exactly one separator, so a
--- caller may pass the directory with or without a trailing slash.
local function join_path(dir, name)
    return (dir:gsub("/+$", "")) .. "/" .. name
end

-- ─── Rules ──────────────────────────────────────────────────────────

--- The intent as the rules read it: the board's "nothing was shown"
--- marker becomes nil, which is what lets `no_intent` and the intent
--- lists be written without repeating the marker in every rule. This is
--- the same normalisation `gen_guardian_sample_playlog.lua` does at the
--- top of its `core_rule`.
local function normalise_intent(view)
    if view.intent == duel.NO_INTENT then
        return nil
    end
    return view.intent
end

local function rule_matches(match, view, intent)
    if match.otherwise then
        return true
    end
    if match.no_intent and intent ~= nil then
        return false
    end
    if match.intent ~= nil then
        if intent == nil then
            return false
        end
        local found = false
        for _, ch in ipairs(match.intent) do
            if ch == intent then
                found = true
                break
            end
        end
        if not found then
            return false
        end
    end
    if match.mode ~= nil and view.mode ~= match.mode then
        return false
    end
    if match.spikes ~= nil and view.spikes ~= match.spikes then
        return false
    end
    return true
end

--- The opening character for this turn, checked against the style's
--- alphabet. A short opening or a character outside the alphabet means
--- the substitution rule and the style disagree, and that is a bug in
--- this file rather than a position worth logging.
local function script_move(style, opening, turn, label)
    local move = opening:sub(turn, turn)
    if move == "" then
        error(
            string.format(
                "gen_guardian_player_styles: opening %q of %s has no character for turn %d",
                opening,
                label,
                turn
            )
        )
    end
    if not style.alphabet[move] then
        error(
            string.format(
                "gen_guardian_player_styles: opening %q of %s plays %q on turn %d, which is "
                    .. "outside the %s alphabet",
                opening,
                label,
                move,
                turn,
                style.name
            )
        )
    end
    return move
end

--- Walk the style's rules and return the move plus the slot that chose
--- it. Falling off the end is impossible for the three rule sets here
--- (every one of them ends in a slot that covers the remaining views),
--- so it is raised rather than defaulted: a silent fallback would put a
--- move in the corpus that no slot accounts for.
local function decide(style, rules, view, opening, label)
    local intent = normalise_intent(view)
    for _, rule in ipairs(rules) do
        if rule_matches(rule.match, view, intent) then
            if rule.move == "script" then
                return script_move(style, opening, view.turn, label), rule.id
            end
            return rule.move, rule.id
        end
    end
    error(
        string.format(
            "gen_guardian_player_styles: %s has no rule for the view of %s "
                .. "(turn=%d mode=%d spikes=%s intent=%s)",
            style.name,
            label,
            view.turn,
            view.mode,
            tostring(view.spikes),
            tostring(view.intent)
        )
    )
end

-- ─── Play ───────────────────────────────────────────────────────────

--- Play one game and return the player half of every turn.
---
--- The boss answer is decided before the view is built, because the view
--- has to carry it: a poke on the previous turn set `revealed`, and
--- `player_view` refuses a revealed turn without the answer it bought
--- (and an answer on a turn nobody paid for). The same answer is then
--- the one `apply` plays, so the intent in the log is never a look at a
--- move the boss did not make.
local function play_game(style, rules, boss, opening, seed, label, slot_counts)
    local policy = duel["policy_" .. boss]
    if type(policy) ~= "function" then
        error("gen_guardian_player_styles: no policy for boss style " .. tostring(boss))
    end
    local state = duel.new_game(seed)
    local entries, guard = {}, 0
    while not duel.is_over(state) do
        local boss_action = policy(state.boss)
        local view = duel.player_view(state, boss, state.revealed and boss_action or nil)
        local player_action, slot = decide(style, rules, view, opening, label)
        slot_counts[slot] = (slot_counts[slot] or 0) + 1
        entries[#entries + 1] = { player = view, player_action = player_action }
        state = duel.apply(state, player_action, boss_action)
        guard = guard + 1
        if guard > duel.TURN_LIMIT + 1 then
            error("gen_guardian_player_styles: runaway fight: " .. label)
        end
    end
    return entries
end

--- Collect one style over all three canonical bosses and all twelve
--- openings, keeping only the player half of each logged turn. The boss
--- half is dropped on purpose: the bake path never reads it, and a
--- corpus should carry exactly what its consumer consumes.
local function collect(style)
    local rules = RULE_MAPS[style.name]
    if rules == nil then
        error("gen_guardian_player_styles: no rule map for style " .. tostring(style.name))
    end
    local entries, games, slot_counts = {}, 0, {}
    for bi, boss in ipairs(BOSSES) do
        for vi, variant in ipairs(TRAIN_VARIANTS) do
            local opening = style.opening(variant)
            local label = string.format("%s/%s/%02d", style.name, boss, vi)
            local seed = style.seed_index * 1000 + bi * 100 + vi
            local game = play_game(style, rules, boss, opening, seed, label, slot_counts)
            games = games + 1
            for _, entry in ipairs(game) do
                entries[#entries + 1] = entry
            end
        end
    end
    log(string.format("%s: %d games, %d entries", style.name, games, #entries))
    return entries, games, slot_counts
end

-- ─── Output ─────────────────────────────────────────────────────────

--- Serialise a corpus one entry per line, matching the layout of the
--- committed sample corpus: a bare JSON array whose elements each sit on
--- their own line, which keeps a regenerated file readable in a diff.
local function encode_entries(entries)
    if #entries == 0 then
        error("gen_guardian_player_styles: refusing to write an empty corpus")
    end
    local encode = require_json_encoder()
    local parts = {}
    for i, entry in ipairs(entries) do
        local ok, encoded = pcall(encode, entry)
        if not ok then
            error(
                string.format(
                    "gen_guardian_player_styles: failed to encode entry %d: %s",
                    i,
                    tostring(encoded)
                )
            )
        end
        parts[i] = encoded
    end
    return "[\n" .. table.concat(parts, ",\n") .. "\n]\n"
end

local function write_corpus(path, entries)
    local body = encode_entries(entries)
    local f, err = io.open(path, "w")
    if f == nil then
        error(
            string.format(
                "gen_guardian_player_styles: cannot open %q for writing: %s",
                path,
                tostring(err)
            )
        )
    end
    local ok, write_err = pcall(function()
        f:write(body)
    end)
    f:close()
    if not ok then
        error(
            string.format(
                "gen_guardian_player_styles: write to %q failed: %s",
                path,
                tostring(write_err)
            )
        )
    end
end

-- ─── Sentinel reproduction ──────────────────────────────────────────

local function read_entries(path)
    local decode = require_json_decoder()
    local f, open_err = io.open(path, "r")
    if f == nil then
        error(
            string.format(
                "gen_guardian_player_styles: cannot open verify_sentinel_path %q: %s",
                path,
                tostring(open_err)
            )
        )
    end
    local body = f:read("a")
    f:close()
    local ok, parsed = pcall(decode, body)
    if not ok then
        error(
            string.format(
                "gen_guardian_player_styles: failed to decode verify_sentinel_path %q: %s",
                path,
                tostring(parsed)
            )
        )
    end
    if type(parsed) ~= "table" then
        error(
            string.format(
                "gen_guardian_player_styles: verify_sentinel_path %q must hold an array of "
                    .. "entries, got %s",
                path,
                type(parsed)
            )
        )
    end
    return parsed
end

--- Compare the replayed sentinel corpus against the committed one, entry
--- by entry. The comparison is on decoded values rather than bytes: the
--- two files are written by different encoders and JSON key order is not
--- part of what a corpus means. The first difference is raised with the
--- entry index and the field name, because "corpus differs" without a
--- position is not a lead.
local function verify_sentinel(entries, path)
    local expected = read_entries(path)
    if #expected ~= #entries then
        error(
            string.format(
                "gen_guardian_player_styles: sentinel replay produced %d entries but %q holds "
                    .. "%d",
                #entries,
                path,
                #expected
            )
        )
    end
    for i = 1, #expected do
        local want, got = expected[i], entries[i]
        if type(want.player) ~= "table" then
            error(
                string.format(
                    "gen_guardian_player_styles: entry %d of %q carries no player view",
                    i,
                    path
                )
            )
        end
        for _, field in ipairs(VIEW_FIELDS) do
            if want.player[field] ~= got.player[field] then
                error(
                    string.format(
                        "gen_guardian_player_styles: sentinel replay differs at entry %d, "
                            .. "player.%s: committed %s, replayed %s",
                        i,
                        field,
                        tostring(want.player[field]),
                        tostring(got.player[field])
                    )
                )
            end
        end
        if want.player_action ~= got.player_action then
            error(
                string.format(
                    "gen_guardian_player_styles: sentinel replay differs at entry %d, "
                        .. "player_action: committed %s, replayed %s",
                    i,
                    tostring(want.player_action),
                    tostring(got.player_action)
                )
            )
        end
    end
    return true
end

-- ─── Run ────────────────────────────────────────────────────────────

local OUTPUT_DIR = require_string("output_dir")
local VERIFY_SENTINEL_PATH = optional_string("verify_sentinel_path")

local result = {
    games = #BOSSES * #TRAIN_VARIANTS,
    outputs = {},
    slot_counts = {},
    rule_maps = RULE_MAPS,
}

for _, style in ipairs(STYLES) do
    local entries, games, slot_counts = collect(style)
    if games ~= result.games then
        error(
            string.format(
                "gen_guardian_player_styles: %s played %d games, expected %d",
                style.name,
                games,
                result.games
            )
        )
    end
    result[style.name .. "_entries"] = #entries
    result.slot_counts[style.name] = slot_counts
    if style.output ~= nil then
        local path = join_path(OUTPUT_DIR, style.output)
        write_corpus(path, entries)
        result.outputs[style.name] = path
        log(string.format("%s: wrote %d entries to %s", style.name, #entries, path))
    elseif VERIFY_SENTINEL_PATH ~= nil then
        result.sentinel_match = verify_sentinel(entries, VERIFY_SENTINEL_PATH)
        log(
            string.format(
                "sentinel: %d entries match %s field for field",
                #entries,
                VERIFY_SENTINEL_PATH
            )
        )
    else
        log("sentinel: replayed but not compared (ctx.verify_sentinel_path was not passed)")
    end
end

return result
