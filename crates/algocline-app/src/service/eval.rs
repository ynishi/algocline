use super::eval_store::{
    escape_for_lua_sq, evals_dir, extract_strategy_from_id, list_eval_history,
    save_compare_result, save_eval_result,
};
use super::path::ContainedPath;
use super::resolve::{is_package_installed, resolve_scenario_code};
use super::AppService;

/// Lua shim that bridges algocline's `alc.*` primitives to the `std` global
/// expected by evalframe.std. Injected once before any evalframe code runs.
const STD_SHIM: &str = r#"
std = {
  json = {
    decode = alc.json_decode,
    encode = alc.json_encode,
  },
  fs = {
    read = function(path)
      local f, err = io.open(path, "r")
      if not f then error("std.fs.read: " .. (err or path), 2) end
      local content = f:read("*a")
      f:close()
      return content
    end,
    is_file = function(path)
      local f = io.open(path, "r")
      if f then f:close(); return true end
      return false
    end,
  },
  time = {
    now = alc.time,
  },
}
"#;

impl AppService {
    /// Run an evalframe evaluation suite via `alc.eval()`.
    ///
    /// Resolves the scenario from one of three input modes (inline/file/name),
    /// injects the `std` global shim, and delegates to `alc.eval()` in prelude
    /// which handles evalframe loading, provider wiring, and optional Card
    /// emission.
    ///
    /// # Security: `strategy` is not sanitized
    ///
    /// `strategy` is interpolated into a Lua string literal without escaping.
    /// This is intentional — algocline runs Lua in the caller's own process
    /// with full ambient authority, so Lua injection does not cross a trust
    /// boundary.
    pub async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        scenario_name: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
        auto_card: bool,
    ) -> Result<String, String> {
        // Auto-install bundled packages if evalframe is missing
        if !is_package_installed("evalframe") {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed("evalframe") {
                return Err(
                    "Package 'evalframe' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                        .into(),
                );
            }
        }

        let scenario_code =
            resolve_scenario_code(scenario, scenario_file, scenario_name.clone())?;

        // Build strategy opts Lua table literal
        let opts_lua = match &strategy_opts {
            Some(v) if !v.is_null() => format!("alc.json_decode('{}')", v),
            _ => "nil".to_string(),
        };

        let auto_card_lua = if auto_card { "true" } else { "false" };

        // Delegate to alc.eval() in prelude.
        // The shim injects `std` for evalframe, then the scenario code is
        // evaluated into a table and passed to alc.eval() along with opts.
        let wrapped = format!(
            r#"{std_shim}

local scenario = (function()
{scenario_code}
end)()

return alc.eval(scenario, "{strategy}", {{
  strategy_opts = {opts_lua},
  auto_card = {auto_card_lua},
}})
"#,
            std_shim = STD_SHIM,
        );

        let ctx = serde_json::Value::Null;
        let result = self
            .start_and_tick(wrapped, ctx, Some(strategy), vec![])
            .await?;

        // Persist eval result for history/comparison.
        // Card emission is handled by alc.eval() Lua-side when auto_card=true.
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
            match parsed.get("status").and_then(|s| s.as_str()) {
                Some("completed") => {
                    save_eval_result(strategy, &result);
                }
                Some("needs_response") => {
                    if let Some(sid) = parsed.get("session_id").and_then(|s| s.as_str()) {
                        if let Ok(mut map) = self.eval_sessions.lock() {
                            map.insert(sid.to_string(), strategy.to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(result)
    }

    /// List eval history, optionally filtered by strategy.
    pub fn eval_history(&self, strategy: Option<&str>, limit: usize) -> Result<String, String> {
        let dir = evals_dir()?;
        list_eval_history(&dir, strategy, limit)
    }

    /// View a specific eval result by ID.
    pub fn eval_detail(&self, eval_id: &str) -> Result<String, String> {
        let evals_dir = evals_dir()?;
        let path = ContainedPath::child(&evals_dir, &format!("{eval_id}.json"))
            .map_err(|e| format!("Invalid eval_id: {e}"))?;
        if !path.exists() {
            return Err(format!("Eval result not found: {eval_id}"));
        }
        std::fs::read_to_string(&*path).map_err(|e| format!("Failed to read eval: {e}"))
    }

    /// Compare two eval results with statistical significance testing.
    ///
    /// Delegates to evalframe's `stats.welch_t` (single source of truth for
    /// t-distribution table and test logic). Reads persisted `aggregated.scores`
    /// from each eval result — no re-computation of descriptive statistics.
    ///
    /// The comparison result is persisted to `~/.algocline/evals/` so repeated
    /// lookups of the same pair are file reads only.
    pub async fn eval_compare(&self, eval_id_a: &str, eval_id_b: &str) -> Result<String, String> {
        // Check for cached comparison
        let cache_filename = format!("compare_{eval_id_a}_vs_{eval_id_b}.json");
        if let Ok(dir) = evals_dir() {
            if let Ok(cached_path) = ContainedPath::child(&dir, &cache_filename) {
                if cached_path.exists() {
                    return std::fs::read_to_string(&*cached_path)
                        .map_err(|e| format!("Failed to read cached comparison: {e}"));
                }
            }
        }

        // Auto-install bundled packages if evalframe is missing
        if !is_package_installed("evalframe") {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed("evalframe") {
                return Err(
                    "Package 'evalframe' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                        .into(),
                );
            }
        }

        let result_a = self.eval_detail(eval_id_a)?;
        let result_b = self.eval_detail(eval_id_b)?;

        // Build Lua snippet that uses evalframe's stats module
        // to compute welch_t from the persisted aggregated scores.
        let lua_code = format!(
            r#"
std = {{
  json = {{
    decode = alc.json_decode,
    encode = alc.json_encode,
  }},
  fs = {{ read = function() end, is_file = function() return false end }},
  time = {{ now = alc.time }},
}}

local stats = require("evalframe.eval.stats")

local result_a = alc.json_decode('{result_a_escaped}')
local result_b = alc.json_decode('{result_b_escaped}')

local agg_a = result_a.result and result_a.result.aggregated
local agg_b = result_b.result and result_b.result.aggregated

if not agg_a or not agg_a.scores then
  error("No aggregated scores in {eval_id_a}")
end
if not agg_b or not agg_b.scores then
  error("No aggregated scores in {eval_id_b}")
end

local welch = stats.welch_t(agg_a.scores, agg_b.scores)

local strategy_a = (result_a.result and result_a.result.name) or "{strategy_a_fallback}"
local strategy_b = (result_b.result and result_b.result.name) or "{strategy_b_fallback}"

local delta = agg_a.scores.mean - agg_b.scores.mean
local winner = "none"
if welch.significant then
  winner = delta > 0 and "a" or "b"
end

-- Build summary text
local parts = {{}}
if welch.significant then
  local w, l, d = strategy_a, strategy_b, delta
  if delta < 0 then w, l, d = strategy_b, strategy_a, -delta end
  parts[#parts + 1] = string.format(
    "%s outperforms %s by %.4f (mean score), statistically significant (t=%.3f, df=%.1f).",
    w, l, d, math.abs(welch.t_stat), welch.df
  )
else
  parts[#parts + 1] = string.format(
    "No statistically significant difference between %s and %s (t=%.3f, df=%.1f).",
    strategy_a, strategy_b, math.abs(welch.t_stat), welch.df
  )
end
if agg_a.pass_rate and agg_b.pass_rate then
  local dp = agg_a.pass_rate - agg_b.pass_rate
  if math.abs(dp) > 1e-9 then
    local h = dp > 0 and strategy_a or strategy_b
    parts[#parts + 1] = string.format("Pass rate: %s +%.1fpp.", h, math.abs(dp) * 100)
  else
    parts[#parts + 1] = string.format("Pass rate: identical (%.1f%%).", agg_a.pass_rate * 100)
  end
end

return {{
  a = {{
    eval_id = "{eval_id_a}",
    strategy = strategy_a,
    scores = agg_a.scores,
    pass_rate = agg_a.pass_rate,
    pass_at_1 = agg_a.pass_at_1,
    ci_95 = agg_a.ci_95,
  }},
  b = {{
    eval_id = "{eval_id_b}",
    strategy = strategy_b,
    scores = agg_b.scores,
    pass_rate = agg_b.pass_rate,
    pass_at_1 = agg_b.pass_at_1,
    ci_95 = agg_b.ci_95,
  }},
  comparison = {{
    delta_mean = delta,
    welch_t = {{
      t_stat = welch.t_stat,
      df = welch.df,
      significant = welch.significant,
      direction = welch.direction,
    }},
    winner = winner,
    summary = table.concat(parts, " "),
  }},
}}
"#,
            result_a_escaped = escape_for_lua_sq(&result_a),
            result_b_escaped = escape_for_lua_sq(&result_b),
            eval_id_a = eval_id_a,
            eval_id_b = eval_id_b,
            strategy_a_fallback = extract_strategy_from_id(eval_id_a).unwrap_or("A"),
            strategy_b_fallback = extract_strategy_from_id(eval_id_b).unwrap_or("B"),
        );

        let ctx = serde_json::Value::Null;
        let raw_result = self.start_and_tick(lua_code, ctx, None, vec![]).await?;

        // Persist comparison result
        save_compare_result(eval_id_a, eval_id_b, &raw_result);

        Ok(raw_result)
    }
}
