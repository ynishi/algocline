-- Generate the bundled guardian-duel sample play logs.
--
-- Self-contained script for `alc_run` (`code_file` form). It replays the
-- deterministic collection behind `data/guardian_sample_playlog_train.json`
-- and `data/guardian_sample_playlog_holdout.json`, and returns both sets so
-- the committed files can be regenerated or verified byte-for-byte at any
-- time (the engine has no RNG: a fight is a pure function of the boss seat
-- and the player move sequence).
--
-- The sample player is a fixed conditional style, "sentinel":
--
--   R1: intent == "t"          -> "b"  (block the telegraphed slam)
--   R2: intent == "f" or "w"   -> "b"  (block the heavy cycle hits)
--   R3: intent == "c" or "d"   -> "A"  (boss deals 0 that turn: free heavy)
--   R4: intent == "v"          -> "a"  (attack through the weaken)
--   R5: no intent, mode == 1   -> "p"  (poke the rolled-up boss to reveal)
--   R6: no intent, mode == 0   -> per-game opening script (free slot)
--
-- R1-R5 are functions of the player view alone, so every logged position
-- has a computable ground truth — that is what makes the held-out set a
-- real generalisation test for `eval_guardian_player_generalization.lua`.
-- R6 varies per game and is the only source of trajectory diversity; its
-- scripts use `{a, A, p}` only, so `b` is (almost) exclusively rule-caused.
--
-- The train and holdout sets share the style but no opening script, and
-- each set walks all three canonical bosses.
--
-- Prerequisites: the four gameai packages linked/installed, and the boss
-- Cards pinned by `train_guardian_npc.lua` (the interactive session seats
-- the boss from `guardian_duel_npc_<style>`).
--
-- Returns `{ train = <entries>, holdout = <entries>, ... }` where each
-- entry is the player half of a logged turn — `{ player = <view>,
-- player_action = <move> }` — exactly the two fields the bake path reads.

local duel = require("guardian_duel")
local interactive = require("guardian_duel_interactive")

local BOSSES = { "guardian", "rusher", "turtle" }

-- 12 train openings x 3 bosses = 36 games.
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

-- 6 holdout openings x 3 bosses = 18 games. None appear above.
local HELDOUT_VARIANTS = {
    "pApApApAp",
    "aappaappa",
    "AapappaAp",
    "papAAppaa",
    "apAppApaa",
    "aAAppaApa",
}

local function core_rule(mode, intent)
    if intent == duel.NO_INTENT then
        intent = nil
    end
    if intent == "t" then
        return "b"
    end
    if intent == "f" or intent == "w" then
        return "b"
    end
    if intent == "c" or intent == "d" then
        return "A"
    end
    if intent == "v" then
        return "a"
    end
    if mode == 1 then
        return "p"
    end
    return nil
end

local function play_game(boss, variant, game_id)
    local view = interactive.run({ action = "new", style = boss, game_id = game_id })
    local guard = 0
    while view.status == "your_turn" do
        local move = core_rule(view.boss_mode, view.intent)
        if move == nil then
            move = variant:sub(view.turn, view.turn)
            if move == "" then
                error(string.format("variant %q has no char for turn %d", variant, view.turn))
            end
        end
        view = interactive.run({ action = "play", move = move, game_id = game_id })
        guard = guard + 1
        if guard > duel.TURN_LIMIT + 1 then
            error("runaway fight: " .. game_id)
        end
    end
    local ended = interactive.run({ action = "end", game_id = game_id })
    return ended.move_log
end

--- Collect one set, keeping only the player half of each logged turn:
--- the view the board showed and the move chosen from it. The boss half
--- is dropped on purpose — the bake path never reads it, and a sample
--- data file should carry exactly what its consumer consumes.
local function collect(variants, tag)
    local entries, games = {}, 0
    for _, boss in ipairs(BOSSES) do
        for vi, variant in ipairs(variants) do
            local game_id = string.format("gen_sample_%s_%s_%02d", tag, boss, vi)
            local move_log = play_game(boss, variant, game_id)
            games = games + 1
            for _, entry in ipairs(move_log) do
                entries[#entries + 1] = {
                    player = entry.player,
                    player_action = entry.player_action,
                }
            end
        end
    end
    alc.log("info", string.format("[gen-sample] %s: %d games, %d entries", tag, games, #entries))
    return entries, games
end

local train, train_games = collect(TRAIN_VARIANTS, "train")
local holdout, holdout_games = collect(HELDOUT_VARIANTS, "holdout")

return {
    train = train,
    holdout = holdout,
    train_games = train_games,
    train_entries = #train,
    holdout_games = holdout_games,
    holdout_entries = #holdout,
}
