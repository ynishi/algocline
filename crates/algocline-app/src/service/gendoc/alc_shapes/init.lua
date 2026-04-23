--- alc_shapes — SSoT for the result shape convention.
---
--- Usage:
---   local S = require("alc_shapes")
---   local ok, reason = S.check(value, S.voted)
---   local x = S.assert(value, "voted", "where")
---
--- Core concept: Schema-as-Data (after Malli's TypeSchemaAsData doctrine).
--- Schemas are plain kind-tagged Lua tables. They are the single AST used
--- by every consumer (validator / reflector / codegen / docs projector)
--- and are persistable without loss — JSON-encode, JSON-decode, still
--- valid. Metatables carry combinator sugar only; `rawget` is the
--- universal reader. Parallel representations (e.g. a separate "TypeExpr"
--- AST) are prohibited: downstream consumers must project, not mirror.
--- See README.md §Core concept.
---
--- NOTE: bundled packages declare their I/O contract under
--- `M.spec.entries.run.{input, result}`. Each field accepts either a
--- string (registry lookup key) or an inline `T.shape(...)` schema.
--- Packages with multiple entry points (e.g. calibrate.run vs
--- calibrate.assess) add further `M.spec.entries.{name}` blocks.
--- Packages without `M.spec` are treated as opaque by spec_resolver.
--- Producers self-decorate at module tail with `M.run = S.instrument(
--- M, "run")`; the wrapper reads `M.spec.entries.<entry>.{input,
--- result}` and asserts on each call when `ALC_SHAPE_CHECK=1`.
--- See README.md §Producer usage and §Producer-wrap vs caller-wrap.

local T = require("alc_shapes.t")
local check = require("alc_shapes.check")
local reflect = require("alc_shapes.reflect")
local luacats = require("alc_shapes.luacats")
local spec_resolver = require("alc_shapes.spec_resolver")
local instrument = require("alc_shapes.instrument")

local M = {}

-- ── shape dictionary ─────────────────────────────────────────────────

M.voted = T.shape({
    consensus       = T.string:describe("LLM-synthesized majority summary"),
    answer          = T.string:is_optional():describe("Majority answer (nil when no paths converge)"),
    answer_norm     = T.string:is_optional():describe("Normalized vote key"),
    paths           = T.array_of(T.shape({
        reasoning = T.string,
        answer    = T.string,
    })):describe("Per-path reasoning + extracted answer"),
    votes           = T.array_of(T.string):describe("Normalized vote per path, 1-indexed"),
    vote_counts     = T.map_of(T.string, T.number):describe("{ [norm] = count } tally"),
    n_sampled       = T.number:describe("Number of sampled paths"),
    total_llm_calls = T.number,
}, { open = true })

M.paneled = T.shape({
    arguments = T.array_of(T.shape({
        role = T.string,
        text = T.string,
    })):describe("Per-role position statements"),
    synthesis = T.string:describe("Moderator synthesis"),
}, { open = true })

M.assessed = T.shape({
    answer          = T.string,
    confidence      = T.number:describe("Self-assessed confidence 0.0–1.0"),
    total_llm_calls = T.number,
}, { open = true })

M.calibrated = T.shape({
    answer          = T.string,
    confidence      = T.number:describe("Initial self-assessed confidence"),
    escalated       = T.boolean:describe("Whether fallback was triggered"),
    strategy        = T.one_of({ "direct", "retry", "panel", "ensemble" }),
    total_llm_calls = T.number,
    fallback_detail = T.table:is_optional():describe("Fallback strategy result (voted/paneled)"),
}, { open = true })

-- conformal_vote (Wang et al. 2026, arXiv:2604.07667):
-- linear opinion pool + split conformal prediction with the three-way
-- decision rule (Proposition 3). `card_id` is populated only when the
-- caller opts into Card emission via ctx.auto_card = true.
M.conformal_decided = T.shape({
    action         = T.one_of({ "commit", "escalate", "anomaly" })
        :describe("Three-way decision per Proposition 3"),
    selected       = T.string:is_optional()
        :describe("Committed label (nil when action != 'commit')"),
    prediction_set = T.array_of(T.string)
        :describe("Labels y with P_social(y|x) >= tau"),
    p_social       = T.map_of(T.string, T.number)
        :describe("Linear opinion pool output { [label] = prob }"),
    coverage_level = T.number:describe("1 - alpha (finite-sample guarantee)"),
    q_hat          = T.number:describe("Calibration quantile of nonconformity scores"),
    tau            = T.number:describe("1 - q_hat (prediction-set threshold)"),
    card_id        = T.string:is_optional()
        :describe("Emitted Card id (only when auto_card=true)"),
}, { open = true })

-- dci (Prakash 2026, arXiv:2603.11781):
-- Deliberative Collective Intelligence (DCI-CF). 4 roles × 14 typed
-- epistemic acts × 8-stage convergence, emitting a decision_packet with
-- 5 non-nil components (selected_option / residual_objections /
-- minority_report / next_actions / reopen_triggers). `card_id` is
-- populated only when the caller opts into Card emission via
-- ctx.auto_card = true.
M.deliberated = T.shape({
    answer          = T.string:describe("Selected option's final answer text"),
    decision_packet = T.shape({
        selected_option     = T.shape({
            answer    = T.string,
            rationale = T.string,
            evidence  = T.array_of(T.string),
        }, { open = true }):describe("Chosen option with rationale and cited evidence"),
        residual_objections = T.array_of(T.string)
            :describe("Objections not fully resolved (empty array allowed, nil禁止)"),
        minority_report     = T.array_of(T.shape({
            position   = T.string,
            rationale  = T.string,
            confidence = T.number,
        }, { open = true }))
            :describe("Dissenting positions with confidence (empty array allowed, nil禁止)"),
        next_actions        = T.array_of(T.string)
            :describe("Concrete follow-up actions (empty array allowed, nil禁止)"),
        reopen_triggers     = T.array_of(T.string)
            :describe("Conditions to reopen deliberation (empty array allowed, nil禁止)"),
    }, { open = true }):describe("5-component decision packet; all 5 fields MUST be non-nil"),
    workspace       = T.shape({
        problem_view          = T.string,
        key_frames            = T.array_of(T.string),
        emerging_ideas        = T.array_of(T.string),
        tensions              = T.array_of(T.string),
        synthesis_in_progress = T.string,
        next_actions          = T.array_of(T.string),
    }, { open = true }):describe("Shared workspace 6 fields after finalization"),
    history         = T.array_of(T.table):describe("Per-stage typed-act log (14-act typed)"),
    convergence     = T.one_of({ "dominance", "no_blocking", "fallback" })
        :describe("How the session converged"),
    stats           = T.shape({
        rounds_used     = T.number,
        total_acts      = T.number,
        options_count   = T.number,
        total_llm_calls = T.number,
    }, { open = true }):describe("Execution statistics"),
    card_id         = T.string:is_optional()
        :describe("Emitted Card id (only when auto_card=true)"),
}, { open = true })

-- smc_sample (Markovic-Voronov et al. 2026, arXiv:2604.16453):
-- Block-level Sequential Monte Carlo sampling (Target I specialization).
-- Reward-weighted importance sampling with ESS-triggered multinomial
-- resampling and Metropolis-Hastings rejuvenation. 1 particle = 1
-- complete answer, ψ_t = exp(α · r(answer)) driven by a caller-injected
-- reward_fn. `card_id` is populated only when the caller opts into
-- Card emission via ctx.auto_card = true.
M.smc_sampled = T.shape({
    answer         = T.string:describe("Argmax-weight particle's answer text"),
    particles      = T.array_of(T.shape({
        answer  = T.string:describe("Particle answer text"),
        weight  = T.number:describe("Final normalized importance weight"),
        reward  = T.number:describe("Final reward under caller's reward_fn"),
        history = T.array_of(T.table):describe("Per-iteration trace entries (1 slot in v1)"),
    }, { open = true }))
        :describe("All N particles in their final state"),
    weights        = T.array_of(T.number)
        :describe("Final normalized weights (Σ ≈ 1)"),
    iterations     = T.number:describe("K SMC rounds actually executed"),
    resample_count = T.number:describe("Number of iterations that triggered multinomial resample"),
    ess_trace      = T.array_of(T.number)
        :describe("ESS recorded at the start of each iteration (length K)"),
    stats          = T.shape({
        total_llm_calls    = T.number:describe("alc.llm invocations issued by the pkg"),
        total_reward_calls = T.number:describe("reward_fn invocations (caller-provided)"),
    }, { open = true }):describe("Execution counters (open for diagnostics like mh_rejected)"),
    card_id        = T.string:is_optional()
        :describe("Emitted Card id (only when auto_card=true)"),
}, { open = true })

-- ── ranking shapes ───────────────────────────────────────────────────

local ranked_item_3 = T.shape({
    rank  = T.number,
    index = T.number,
    text  = T.string,
})

local ranked_item_4 = T.shape({
    rank  = T.number,
    index = T.number,
    score = T.number,
    text  = T.string,
})

M.tournament = T.shape({
    best        = T.string:describe("Winner text"),
    best_index  = T.number:describe("Winner original index (1-based)"),
    total_wins  = T.number:describe("Winner's win count"),
    candidates  = T.array_of(T.string):describe("Input candidate texts"),
    matches     = T.array_of(T.shape({
        a      = T.number:describe("Index of first candidate"),
        b      = T.number:describe("Index of second candidate"),
        winner = T.number:describe("Index of the winner"),
        reason = T.string:describe("Judge verdict explanation"),
    })):describe("Pairwise match log"),
}, { open = true })

M.listwise_ranked = T.shape({
    ranked      = T.array_of(ranked_item_3):describe("Full ranking"),
    top_k       = T.array_of(ranked_item_3):describe("Top-k subset"),
    killed      = T.array_of(ranked_item_3):describe("Eliminated candidates"),
    best        = T.string:describe("Top-ranked text"),
    best_index  = T.number:describe("Top-ranked original index (1-based)"),
    n_candidates    = T.number,
    total_llm_calls = T.number,
}, { open = true })

M.pairwise_ranked = T.shape({
    ranked      = T.array_of(ranked_item_4):describe("Full ranking with scores"),
    top_k       = T.array_of(ranked_item_4):describe("Top-k subset"),
    killed      = T.array_of(ranked_item_4):describe("Eliminated candidates"),
    best        = T.string:describe("Top-ranked text"),
    best_index  = T.number:describe("Top-ranked original index (1-based)"),
    method          = T.one_of({ "allpair", "sorting" }):describe("Comparison strategy"),
    score_semantics = T.one_of({ "copeland", "rank_inverse" }):describe("Score interpretation"),
    n_candidates        = T.number,
    total_llm_calls     = T.number,
    position_bias_splits = T.number:describe("Position-bias correction splits"),
    both_tie_pairs       = T.number:describe("Pairs that tied in both directions"),
}, { open = true })

-- ── recipe shapes ────────────────────────────────────────────────────

local funnel_stage = T.discriminated("name", {
    listwise_rank = T.shape({
        name         = T.one_of({ "listwise_rank" }),
        input_count  = T.number,
        output_count = T.number,
        llm_calls    = T.number,
        window_size  = T.number,
    }),
    listwise_skipped = T.shape({
        name         = T.one_of({ "listwise_skipped" }),
        input_count  = T.number,
        output_count = T.number,
        llm_calls    = T.number,
        reason       = T.string,
    }),
    multi_axis_scoring = T.shape({
        name           = T.one_of({ "multi_axis_scoring" }),
        input_count    = T.number,
        output_count   = T.number,
        llm_calls      = T.number,
        axes_count     = T.number,
        parse_failures = T.number,
        score_range    = T.shape({ min = T.number, max = T.number }),
    }),
    scoring_skipped = T.shape({
        name         = T.one_of({ "scoring_skipped" }),
        input_count  = T.number,
        output_count = T.number,
        llm_calls    = T.number,
        reason       = T.string,
    }),
    pairwise_rank_allpair = T.shape({
        name                 = T.one_of({ "pairwise_rank_allpair" }),
        input_count          = T.number,
        output_count         = T.number,
        llm_calls            = T.number,
        position_bias_splits = T.number,
        both_tie_pairs       = T.number,
    }),
    direct_pairwise = T.shape({
        name                 = T.one_of({ "direct_pairwise" }),
        input_count          = T.number,
        output_count         = T.number,
        llm_calls            = T.number,
        position_bias_splits = T.number,
        both_tie_pairs       = T.number,
    }),
})

M.funnel_ranked = T.shape({
    ranking     = T.array_of(T.shape({
        rank           = T.number,
        text           = T.string,
        original_index = T.number:describe("Pre-funnel candidate index (1-based)"),
        pairwise_score = T.number:describe("Copeland score from pairwise stage"),
    })):describe("Final ranking"),
    best            = T.string:describe("Top-ranked text"),
    best_index      = T.number:describe("Top-ranked original index (1-based)"),
    funnel_bypassed = T.boolean:describe("True when N < 6 bypasses funnel stages"),
    bypass_reason   = T.string:is_optional():describe("Reason for bypass (nil when not bypassed)"),
    total_llm_calls     = T.number,
    naive_baseline_calls = T.number:describe("Hypothetical full-pairwise call count"),
    naive_baseline_kind  = T.string:describe("Baseline method identifier"),
    savings_percent = T.number:is_optional():describe("LLM call savings vs baseline (nil on bypass)"),
    warnings        = T.array_of(T.shape({
        code     = T.string:describe("Machine-readable warning identifier"),
        severity = T.one_of({ "warn", "critical" }),
        data     = T.table:describe("Diagnostic payload (structure varies by code)"),
        message  = T.string:describe("Human-readable summary"),
    })):describe("Diagnostic warnings"),
    stages       = T.array_of(funnel_stage):describe("Per-stage detail (discriminated by name)"),
    funnel_shape = T.array_of(T.number):describe("Candidate counts per stage [N, s1_out, s2_out]"),
}, { open = true })

local safe_panel_stage = T.discriminated("name", {
    condorcet = T.shape({
        name              = T.one_of({ "condorcet" }),
        p_estimate        = T.number,
        recommended_n     = T.number,
        expected_accuracy = T.number,
        target_accuracy   = T.number,
        target_met        = T.boolean,
    }),
    condorcet_anti_jury = T.shape({
        name         = T.one_of({ "condorcet_anti_jury" }),
        p_estimate   = T.number,
        is_anti_jury = T.boolean,
    }),
    condorcet_coin_flip = T.shape({
        name         = T.one_of({ "condorcet_coin_flip" }),
        p_estimate   = T.number,
        is_anti_jury = T.boolean,
    }),
    sc = T.shape({
        name               = T.one_of({ "sc" }),
        panel_size         = T.number,
        answer             = T.string,
        plurality_fraction = T.number,
        margin_gap         = T.number,
        n_distinct_answers = T.number,
        unanimous          = T.boolean,
        llm_calls          = T.number,
    }),
    vote_prefix_stability = T.shape({
        name              = T.one_of({ "vote_prefix_stability" }),
        signal_type       = T.string,
        series_length     = T.number,
        is_safe           = T.boolean,
        peak_idx          = T.number,
        consecutive_drops = T.number,
    }),
    vote_prefix_stability_skipped = T.shape({
        name          = T.one_of({ "vote_prefix_stability_skipped" }),
        signal_type   = T.string,
        reason        = T.string,
        series_length = T.number,
        is_safe       = T.boolean,
    }),
    calibrate = T.shape({
        name                = T.one_of({ "calibrate" }),
        confidence          = T.number,
        threshold           = T.number,
        needs_investigation = T.boolean,
        escalated           = T.boolean,
        llm_calls           = T.number,
    }),
})

M.safe_paneled = T.shape({
    answer       = T.string:is_optional():describe("Consensus answer (nil on abort)"),
    confidence   = T.number:describe("Meta-confidence estimate"),
    panel_size   = T.number:describe("Actual panel size used"),
    plurality_fraction = T.number:describe("Top-answer vote fraction"),
    margin_gap   = T.number:describe("(top - runner_up) / n"),
    vote_counts  = T.map_of(T.string, T.number):describe("{ [normalized_answer] = count } tally"),
    n_distinct_answers = T.number:describe("Count of unique answers"),
    expected_accuracy  = T.number:describe("Condorcet expected majority accuracy"),
    target_met         = T.boolean:describe("Whether expected accuracy >= target"),
    is_safe            = T.boolean:describe("Vote-prefix stability safe flag"),
    anti_jury          = T.boolean:describe("Condorcet anti-jury detection"),
    aborted            = T.boolean:describe("True if early-abort triggered"),
    needs_investigation = T.boolean:describe("True if meta-confidence below threshold"),
    unanimous          = T.boolean:describe("All votes identical"),
    total_llm_calls    = T.number,
    abort_reason       = T.string:is_optional():describe("Abort reason (nil when not aborted)"),
    stages             = T.array_of(safe_panel_stage):describe("Per-stage detail (discriminated by name)"),
}, { open = true })

-- ── recipe_quick_vote shape ──────────────────────────────────────────
-- Adaptive-stop majority vote with SPRT gate. Three outcome branches
-- ("confirmed" / "rejected" / "truncated") share the same result shape
-- so consumers can read any field without branching on `outcome`.
local quick_vote_sample = T.shape({
    reasoning = T.string,
    answer    = T.string,
    norm      = T.string,
}, { open = true })

local quick_vote_sprt = T.shape({
    log_lr  = T.number,
    n       = T.number,
    a_bound = T.number,
    b_bound = T.number,
}, { open = true })

local quick_vote_params = T.shape({
    p0    = T.number,
    p1    = T.number,
    alpha = T.number,
    beta  = T.number,
    min_n = T.number,
    max_n = T.number,
}, { open = true })

M.quick_voted = T.shape({
    answer      = T.string:describe("Leader answer from sample 1 (cleaned, not normalized)"),
    leader_norm = T.string:describe("Normalized leader key used for agreement tests"),
    outcome     = T.one_of({ "confirmed", "rejected", "truncated" })
        :describe("Terminal state: confirmed=H1 accepted, rejected=H0 accepted, truncated=no verdict at max_n"),
    verdict     = T.one_of({ "accept_h1", "accept_h0", "continue" })
        :describe("Underlying SPRT verdict from the final decide()"),
    n_samples   = T.number:describe("Total samples drawn (1 leader + k agreement observations)"),
    vote_counts = T.map_of(T.string, T.number):describe("{ [norm] = count } tally across all samples"),
    samples     = T.array_of(quick_vote_sample):describe("Per-sample reasoning + extracted answer"),
    sprt        = quick_vote_sprt:describe("Final SPRT state snapshot"),
    params      = quick_vote_params:describe("Echoed parameter values"),
    total_llm_calls     = T.number:describe("2 × n_samples (reasoning + extract per sample)"),
    needs_investigation = T.boolean:describe("True only when outcome == 'truncated' (evidence inconclusive at declared α/β). 'rejected' is a conclusive verdict and does NOT set this flag."),
}, { open = true })

-- ── recipe_deep_panel shape ──────────────────────────────────────────
-- Stages are heterogeneous per invocation (Stage 1 may be the abort
-- branch; Stage 3's decomp is nullable; Stage 5 always present on
-- main path). Keep the per-stage sub-shape open-ended as T.table so
-- the recipe can evolve per-stage fields without breaking consumers
-- that don't inspect them. open = true at the top level allows
-- forward-compat additions such as future Stage-specific diagnostics.
M.deep_paneled = T.shape({
    answer       = T.any:describe("Plurality answer (nil on abort)"),
    confidence   = T.number:describe("Meta-confidence estimate"),
    panel_size   = T.number:describe("Requested panel size"),
    n_branches_completed = T.number:describe("Branches actually finished"),
    plurality_fraction   = T.number:describe("Top-answer vote fraction"),
    margin_gap           = T.number:describe("(top - runner_up) / n"),
    vote_counts          = T.map_of(T.string, T.number):describe("{ [normalized_answer] = count } tally"),
    n_distinct_answers   = T.number:describe("Count of unique normalized answers"),
    branches             = T.table:describe("{ [bkey] = { approach, answer, best_score, tree_stats } }"),
    expected_accuracy    = T.number:describe("Condorcet expected majority accuracy"),
    target_met           = T.boolean:describe("Whether expected accuracy >= ctx.target_accuracy"),
    anti_jury            = T.boolean:describe("Condorcet anti-jury detection at Stage 1"),
    aborted              = T.boolean:describe("True if early-abort triggered"),
    needs_investigation  = T.boolean:describe("True if meta-confidence below threshold"),
    unanimous            = T.boolean:describe("All normalized votes identical"),
    diversity            = T.table:is_optional():describe("{ n_distinct, distinctness, decomp_status }"),
    decomp               = T.table:is_optional():describe("ensemble_div.decompose output (nil if Stage 3b skipped)"),
    total_llm_calls      = T.number,
    abort_reason         = T.string:is_optional():describe("Abort reason (nil when not aborted)"),
    stages               = T.array_of(T.table):describe("Per-stage detail (heterogeneous)"),
}, { open = true })

-- ── public API re-export ─────────────────────────────────────────────
M.check        = check.check
M.assert       = check.assert
M.assert_dev   = check.assert_dev
M.is_dev_mode  = check.is_dev_mode
M.fields       = reflect.fields
M.walk         = reflect.walk
M.is_schema    = T._internal.is_schema

-- Combinator namespace (so callers can write `S.T.string` without a
-- separate require).
M.T = T

-- Codegen namespace (used by scripts/gen_shapes_luacats.lua).
M.LuaCats = luacats

-- Spec resolver namespace (routing / recipe layer uses this to treat
-- typed bundled pkgs and opaque external pkgs uniformly).
M.spec_resolver = spec_resolver

-- Malli-style producer-wrap instrumentation (see alc_shapes/instrument.lua).
-- Bundled pkgs self-decorate with `M.run = S.instrument(M, "run")` at
-- module tail, reading shapes from `M.spec.entries[entry_name]`.
M.instrument   = instrument.instrument

-- ── reserved-name guard ──────────────────────────────────────────────
-- Certain names collide with `check.assert` shortcut semantics:
--   `M.assert(v, "any")` is always a no-op pass-through. Registering a
-- shape under such a name would silently shadow the shortcut (check.lua).
-- tableshape / Zod avoid this by namespace-separating built-ins from user
-- schemas; we enforce the same invariant via a load-time loud-fail.
-- Re-exported functions / combinator namespaces (M.T, M.LuaCats, etc.)
-- are not shape-kind and therefore never trip this check.
local RESERVED_SHAPE_NAMES = {
    "any", "check", "assert", "assert_dev", "is_dev_mode",
    "fields", "walk", "is_schema", "T", "LuaCats", "spec_resolver",
    "instrument", "_internal",
}

local function assert_no_reserved_shapes(mod)
    for i = 1, #RESERVED_SHAPE_NAMES do
        local name = RESERVED_SHAPE_NAMES[i]
        local v = rawget(mod, name)
        if type(v) == "table" and rawget(v, "kind") == "shape" then
            error(string.format(
                "alc_shapes: '%s' is reserved (assert shortcut); cannot register a shape under this name",
                name), 2)
        end
    end
end

assert_no_reserved_shapes(M)

M._internal = {
    assert_no_reserved_shapes = assert_no_reserved_shapes,
    RESERVED_SHAPE_NAMES      = RESERVED_SHAPE_NAMES,
}

return M
