---@meta
--- algocline Lua StdLib — LuaCats type definitions
--- This file provides type information for editor completion and static analysis.
--- It is NOT executed at runtime. Place in workspace.library for LuaLS.
---
--- Usage: Add this directory to `workspace.library` in .luarc.json:
---   { "workspace": { "library": ["types"] } }
---
--- Layer 0: Runtime Primitives (Rust-backed)
--- Layer 1: Prelude Combinators (Pure Lua)

---@class alc
alc = {}

-- ============================================================
-- Layer 0: Runtime Primitives
-- ============================================================

-- LLM ---

---@class AlcLlmOpts
---@field system? string System prompt
---@field max_tokens? integer Max tokens (default: 1024)
---@field grounded? boolean Request grounded response (default: false)
---@field underspecified? boolean Signal underspecified prompt (default: false)

--- Call the Host LLM. Yields the coroutine until the host responds.
---@param prompt string The prompt to send
---@param opts? AlcLlmOpts Options
---@return string response LLM response text
function alc.llm(prompt, opts) end

---@class AlcLlmBatchItem
---@field prompt string The prompt
---@field system? string System prompt
---@field max_tokens? integer Max tokens (default: 1024)
---@field grounded? boolean Request grounded response
---@field underspecified? boolean Signal underspecified prompt

--- Send multiple LLM calls as a single batch. All queries dispatched concurrently.
---@param items AlcLlmBatchItem[] Array of query tables
---@return string[] responses Responses in same order as input
function alc.llm_batch(items) end

---@class AlcForkResult
---@field strategy string Package name
---@field result? string Result (on success)
---@field error? string Error message (on failure)

---@class AlcForkOpts
---@field on_error? "skip"|"fail" Error handling (default: "skip")

--- Spawn N independent Lua VMs, each running one strategy with the same ctx.
---@param strategies string[] Array of package names
---@param ctx table Context passed to each strategy's run(ctx)
---@param opts? AlcForkOpts Options
---@return AlcForkResult[] results Per-strategy results
function alc.fork(strategies, ctx, opts) end

-- JSON ---

--- Serialize a Lua value to JSON string.
---@param value any Lua value to serialize
---@return string json JSON string
function alc.json_encode(value) end

--- Deserialize a JSON string to a Lua value.
---@param str string JSON string
---@return any value Lua value
function alc.json_decode(str) end

-- Fuzzy Matching ---

---@class AlcMatchEnumOpts
---@field threshold? number Minimum similarity for fuzzy fallback (default: 0.7)

--- Find which candidate appears in LLM output (case-insensitive substring).
--- If multiple match, returns the one whose last occurrence is latest.
--- Falls back to fuzzy matching (Jaro-Winkler) for typos.
---@param text string LLM response text
---@param candidates string[] Valid values to search for
---@param opts? AlcMatchEnumOpts Options
---@return string|nil matched Matched candidate or nil
function alc.match_enum(text, candidates, opts) end

--- Normalize yes/no-style LLM responses.
--- Scans for affirmative/negative keywords and returns the polarity of the last-occurring keyword.
---@param text string LLM response text
---@return boolean|nil result true (affirmative), false (negative), or nil (ambiguous)
function alc.match_bool(text) end

-- Logging ---

--- Emit a log message via tracing.
---@param level "error"|"warn"|"info"|"debug" Log level
---@param msg string Log message
function alc.log(level, msg) end

-- State ---

---@class alc.state
alc.state = {}

--- Read a value from the persistent key-value store.
---@param key string Key to read
---@param default? any Default value if key does not exist
---@return any value Stored value or default
function alc.state.get(key, default) end

--- Write a value to the persistent key-value store.
---@param key string Key to write
---@param value any JSON-serializable value
function alc.state.set(key, value) end

--- List all keys in the current namespace.
---@return string[] keys Array of key names
function alc.state.keys() end

--- Remove a key from the store.
---@param key string Key to delete
function alc.state.delete(key) end

-- Text ---

---@class AlcChunkOpts
---@field mode? "lines"|"chars" Chunking mode (default: "lines")
---@field size? integer Chunk size (default: 50)
---@field overlap? integer Overlap between chunks (default: 0)

--- Split text into chunks by lines or characters.
---@param text string Text to split
---@param opts? AlcChunkOpts Options
---@return string[] chunks Array of text chunks
function alc.chunk(text, opts) end

-- Metrics ---

---@class alc.stats
alc.stats = {}

--- Record a custom metric.
---@param key string Metric name
---@param value any JSON-serializable value
function alc.stats.record(key, value) end

--- Retrieve a recorded metric.
---@param key string Metric name
---@return any|nil value Metric value or nil
function alc.stats.get(key) end

-- Time ---

--- Wall-clock time in fractional seconds since Unix epoch.
---@return number seconds Sub-millisecond precision
function alc.time() end

-- Budget ---

---@class AlcBudgetRemaining
---@field llm_calls? integer Remaining LLM calls (if limit set)
---@field elapsed_ms? integer Remaining time in ms (if limit set)

--- Query raw remaining budget.
---@return AlcBudgetRemaining|nil remaining nil if no budget set
function alc.budget_remaining() end

-- Progress ---

--- Report structured progress, readable via alc_status MCP tool.
---@param step integer Current step number
---@param total integer Total number of steps
---@param msg? string Optional progress message
function alc.progress(step, total, msg) end

-- ============================================================
-- Layer 1: Prelude Combinators
-- ============================================================

-- LLM Wrappers ---

---@class AlcCacheOpts : AlcLlmOpts
---@field cache_key? string Explicit cache key (overrides auto-fingerprint)
---@field cache_skip? boolean Bypass cache, always call LLM

--- Memoized LLM call. Session-scoped cache, max 256 entries.
---@param prompt string The prompt
---@param opts? AlcCacheOpts Options (same as alc.llm plus cache control)
---@return string response Cached or fresh LLM response
function alc.cache(prompt, opts) end

---@class AlcCacheInfo
---@field entries integer Current cache entries
---@field hits integer Cache hit count
---@field misses integer Cache miss count
---@field max_entries integer Maximum cache capacity

--- Return cache statistics.
---@return AlcCacheInfo info
function alc.cache_info() end

--- Clear all cached responses and reset counters.
function alc.cache_clear() end

--- Call alc.llm, returning default on failure instead of raising.
---@param prompt string The prompt
---@param opts? AlcLlmOpts Options
---@param default string Fallback value on error
---@return string response LLM response or default
function alc.llm_safe(prompt, opts, default) end

--- Convenience wrapper: calls alc.llm with grounded = true.
---@param claim string Claim to ground
---@param opts? AlcLlmOpts Options
---@return string response Grounded response
function alc.ground(claim, opts) end

--- Convenience wrapper: calls alc.llm with underspecified = true.
---@param prompt string Underspecified prompt
---@param opts? AlcLlmOpts Options
---@return string response Resolved response
function alc.specify(prompt, opts) end

-- Collection ---

--- Apply fn(item, index) to each item.
---@generic T, R
---@param items T[] Array of items
---@param fn fun(item: T, index: integer): R Transform function
---@return R[] results Array of results
function alc.map(items, fn) end

--- Fold array to single value.
---@generic T, R
---@param items T[] Array of items
---@param fn fun(acc: R, item: T, index: integer): R Reducer function
---@param init? R Initial value (default: items[1])
---@return R result Final accumulated value
function alc.reduce(items, fn, init) end

--- Keep items where fn(item, index) returns truthy.
---@generic T
---@param items T[] Array of items
---@param fn fun(item: T, index: integer): any Predicate function
---@return T[] filtered Filtered array
function alc.filter(items, fn) end

---@class AlcParallelOpts
---@field system? string Shared system prompt
---@field max_tokens? integer Shared max_tokens
---@field post_fn? fun(response: string, item: any, index: integer): any Post-processing function

--- Batch-parallel LLM calls over an array (single round-trip).
---@param items any[] Array of items
---@param prompt_fn fun(item: any, index: integer): string|AlcLlmBatchItem Prompt builder
---@param opts? AlcParallelOpts Options
---@return string[]|any[] results Responses (or post_fn results)
function alc.parallel(items, prompt_fn, opts) end

-- Aggregation ---

---@class AlcVoteResult
---@field winner string Most frequent answer
---@field count integer Count of winner
---@field total integer Total answers

--- Majority vote over an array of string answers.
---@param answers string[] Array of answers
---@return AlcVoteResult result Vote result
function alc.vote(answers) end

--- Extract the first integer from a string. Clamps to 1-10.
---@param str string String containing a number
---@param default? integer Fallback value (default: 5)
---@return integer score Score in 1-10 range
function alc.parse_score(str, default) end

--- Extract a number from LLM output.
--- If pattern is given, uses it as a Lua pattern with a capture group.
--- Otherwise extracts the first number (integer or decimal, optionally negative).
---@param text string Text to extract from
---@param pattern? string Lua pattern with capture group
---@return number|nil value Extracted number or nil
function alc.parse_number(text, pattern) end

-- JSON ---

--- Extract JSON object or array from LLM output (3-stage fallback).
---@param raw string Raw LLM output
---@return table|nil data Parsed JSON or nil
function alc.json_extract(raw) end

-- State ---

--- Read-modify-write for state.
---@param key string Key to update
---@param fn fun(current: any): any Transform function
---@param default? any Initial value if key does not exist
---@return any updated Updated value
function alc.state.update(key, fn, default) end

-- Pipeline ---

---@class AlcPipeOpts
---@field on_stage? fun(i: integer, name: string, ctx: table) Callback after each stage (not called on error)
---@field on_error? "abort"|"skip"|"continue" Error handling mode (default: "abort"). "abort": propagate error; "skip"/"continue": log and advance to next stage

--- Sequential pipeline: run multiple strategies in order.
---@param strategies (string|fun(ctx: table): table)[] Package names or inline functions
---@param ctx table Initial context
---@param opts? AlcPipeOpts Options
---@return table ctx Context with .result and .pipe_history
function alc.pipe(strategies, ctx, opts) end

-- Tuning ---

---@class AlcTuningOpts
---@field prefix? string Namespace key in ctx

--- Merge tuning defaults with ctx overrides. Deep-merges dicts, shallow-replaces arrays.
---@param defaults table Default parameter table
---@param ctx table Context with potential overrides
---@param opts? AlcTuningOpts Options
---@return table merged Merged parameters
function alc.tuning(defaults, ctx, opts) end

-- Utility ---

--- Normalize text and return 8-char hex hash (DJB2). For dedup, not crypto.
---@param str string Text to fingerprint
---@return string hash 8-character hex string
function alc.fingerprint(str) end

--- Returns true if budget has remaining capacity.
---@return boolean ok True if safe to continue
function alc.budget_check() end

-- ============================================================
-- alc.math — Numeric Computing (mlua-mathlib v0.2)
-- ============================================================

---@class alc.math
alc.math = {}

---@class LuaRng
--- Opaque RNG handle (ChaCha12). Created via alc.math.rng_create().

-- RNG ---

--- Create a new seeded RNG instance.
---@param seed integer 64-bit seed value
---@return LuaRng rng New RNG instance
function alc.math.rng_create(seed) end

--- Sample a uniform float in [0, 1).
---@param rng LuaRng RNG instance
---@return number value Random float
function alc.math.rng_float(rng) end

--- Sample a uniform integer in [min, max].
---@param rng LuaRng RNG instance
---@param min integer Minimum value (inclusive)
---@param max integer Maximum value (inclusive)
---@return integer value Random integer
function alc.math.rng_int(rng, min, max) end

-- Distribution Sampling: Continuous ---

--- Sample from Normal (Gaussian) distribution.
---@param rng LuaRng RNG instance
---@param mean number Mean
---@param stddev number Standard deviation
---@return number value Sampled value
function alc.math.normal_sample(rng, mean, stddev) end

--- Sample from Beta distribution.
---@param rng LuaRng RNG instance
---@param alpha number Alpha parameter (> 0)
---@param beta number Beta parameter (> 0)
---@return number value Sampled value in (0, 1)
function alc.math.beta_sample(rng, alpha, beta) end

--- Sample from Gamma distribution.
---@param rng LuaRng RNG instance
---@param shape number Shape parameter (> 0)
---@param scale number Scale parameter (> 0)
---@return number value Sampled value
function alc.math.gamma_sample(rng, shape, scale) end

--- Sample from Exponential distribution.
---@param rng LuaRng RNG instance
---@param lambda number Rate parameter (> 0)
---@return number value Sampled value
function alc.math.exp_sample(rng, lambda) end

--- Sample from continuous Uniform distribution [low, high).
---@param rng LuaRng RNG instance
---@param low number Lower bound (inclusive)
---@param high number Upper bound (exclusive)
---@return number value Sampled value
function alc.math.uniform_sample(rng, low, high) end

--- Sample from Log-normal distribution.
---@param rng LuaRng RNG instance
---@param mu number Mean of the underlying normal
---@param sigma number Std dev of the underlying normal
---@return number value Sampled value
function alc.math.lognormal_sample(rng, mu, sigma) end

--- Sample from Student's t-distribution.
---@param rng LuaRng RNG instance
---@param df number Degrees of freedom (> 0)
---@return number value Sampled value
function alc.math.student_t_sample(rng, df) end

--- Sample from Chi-squared distribution.
---@param rng LuaRng RNG instance
---@param df number Degrees of freedom (> 0)
---@return number value Sampled value
function alc.math.chi_squared_sample(rng, df) end

-- Distribution Sampling: Discrete ---

--- Sample from Poisson distribution.
---@param rng LuaRng RNG instance
---@param lambda number Rate parameter (> 0)
---@return integer value Sampled count
function alc.math.poisson_sample(rng, lambda) end

--- Sample from Binomial distribution.
---@param rng LuaRng RNG instance
---@param n integer Number of trials
---@param p number Success probability [0, 1]
---@return integer value Sampled count
function alc.math.binomial_sample(rng, n, p) end

-- Distribution Sampling: Multivariate ---

--- Sample from Dirichlet distribution.
---@param rng LuaRng RNG instance
---@param alphas number[] Alpha parameters (≥ 2 elements, all > 0)
---@return number[] values Probability vector summing to 1
function alc.math.dirichlet_sample(rng, alphas) end

--- Sample from weighted categorical distribution.
---@param rng LuaRng RNG instance
---@param weights number[] Non-negative weights (≥ 1 element)
---@return integer index 1-based sampled index
function alc.math.categorical_sample(rng, weights) end

-- Descriptive Statistics ---

--- Arithmetic mean.
---@param data number[] Non-empty array
---@return number mean
function alc.math.mean(data) end

--- Sample variance (Welford's algorithm).
---@param data number[] Non-empty array
---@return number variance
function alc.math.variance(data) end

--- Sample standard deviation.
---@param data number[] Non-empty array
---@return number stddev
function alc.math.stddev(data) end

--- Median (linear interpolation).
---@param data number[] Non-empty array
---@return number median
function alc.math.median(data) end

--- Percentile with linear interpolation.
---@param data number[] Non-empty array
---@param p number Percentile in [0, 100]
---@return number value
function alc.math.percentile(data, p) end

--- Interquartile range (Q3 - Q1).
---@param data number[] Non-empty array
---@return number iqr
function alc.math.iqr(data) end

-- Bivariate Statistics ---

--- Sample covariance.
---@param xs number[] First variable (≥ 2 elements)
---@param ys number[] Second variable (same length as xs)
---@return number covariance
function alc.math.covariance(xs, ys) end

--- Pearson correlation coefficient.
---@param xs number[] First variable (≥ 2 elements)
---@param ys number[] Second variable (same length as xs)
---@return number correlation In [-1, 1]
function alc.math.correlation(xs, ys) end

-- Transforms & Utilities ---

--- Numerically stable softmax.
---@param data number[] Input values
---@return number[] probabilities Sum to 1.0
function alc.math.softmax(data) end

--- Log-normalize positive values to [0, 100] scale.
---@param data number[] All values must be > 0
---@return number[] normalized Values in [0, 100]
function alc.math.log_normalize(data) end

---@class AlcHistogramResult
---@field counts integer[] Bin counts
---@field edges number[] Bin edges (#edges == bins + 1)

--- Compute histogram.
---@param data number[] Non-empty array
---@param bins integer Number of bins (> 0)
---@return AlcHistogramResult result
function alc.math.histogram(data, bins) end

---@class AlcWilsonCiResult
---@field lower number Lower bound
---@field upper number Upper bound
---@field center number Center value

--- Wilson score confidence interval for binomial proportions.
---@param successes number Number of successes
---@param total number Total trials (> 0)
---@param confidence number Confidence level in [0, 1]
---@return AlcWilsonCiResult result
function alc.math.wilson_ci(successes, total, confidence) end

-- CDF & PPF ---

--- Normal CDF.
---@param x number Value
---@param mu number Mean
---@param sigma number Standard deviation
---@return number probability
function alc.math.normal_cdf(x, mu, sigma) end

--- Beta CDF.
---@param x number Value in [0, 1]
---@param alpha number Alpha parameter
---@param beta number Beta parameter
---@return number probability
function alc.math.beta_cdf(x, alpha, beta) end

--- Gamma CDF (scale parameterization).
---@param x number Value
---@param shape number Shape parameter
---@param scale number Scale parameter (> 0)
---@return number probability
function alc.math.gamma_cdf(x, shape, scale) end

--- Poisson CDF.
---@param k integer Value
---@param lambda number Rate parameter
---@return number probability P(X ≤ k)
function alc.math.poisson_cdf(k, lambda) end

--- Normal inverse CDF (PPF) for N(mu, sigma).
---@param p number Probability in [0, 1]
---@param mu number Mean
---@param sigma number Standard deviation
---@return number value
function alc.math.normal_inverse_cdf(p, mu, sigma) end

--- Standard normal PPF (N(0,1)).
---@param p number Probability in [0, 1]
---@return number value
function alc.math.normal_ppf(p) end

--- Beta inverse CDF (PPF).
---@param p number Probability in [0, 1]
---@param alpha number Alpha parameter
---@param beta number Beta parameter
---@return number value
function alc.math.beta_ppf(p, alpha, beta) end

-- Distribution Utilities ---

--- Mean of Beta distribution.
---@param alpha number Alpha (> 0)
---@param beta number Beta (> 0)
---@return number mean
function alc.math.beta_mean(alpha, beta) end

--- Variance of Beta distribution.
---@param alpha number Alpha (> 0)
---@param beta number Beta (> 0)
---@return number variance
function alc.math.beta_variance(alpha, beta) end

-- Special Functions ---

--- Error function.
---@param x number Input
---@return number value erf(x)
function alc.math.erf(x) end

--- Complementary error function.
---@param x number Input
---@return number value erfc(x) = 1 - erf(x)
function alc.math.erfc(x) end

--- Log-gamma function.
---@param x number Input
---@return number value ln(Γ(x))
function alc.math.lgamma(x) end

--- Beta function B(a, b).
---@param a number Parameter a
---@param b number Parameter b
---@return number value
function alc.math.beta(a, b) end

--- Log-beta function.
---@param a number Parameter a
---@param b number Parameter b
---@return number value ln(B(a, b))
function alc.math.ln_beta(a, b) end

--- Regularized incomplete beta function I_x(a, b).
---@param x number Value in [0, 1]
---@param a number Parameter a
---@param b number Parameter b
---@return number value
function alc.math.regularized_incomplete_beta(x, a, b) end

--- Regularized lower incomplete gamma P(a, x).
---@param a number Shape parameter
---@param x number Value
---@return number value
function alc.math.regularized_incomplete_gamma(a, x) end

--- Digamma function ψ(x).
---@param x number Input
---@return number value
function alc.math.digamma(x) end

--- Factorial n! (max n=170, overflows f64 beyond).
---@param n integer Non-negative integer (0-170)
---@return number value
function alc.math.factorial(n) end

--- Natural log of factorial ln(n!).
---@param n integer Non-negative integer
---@return number value
function alc.math.ln_factorial(n) end
