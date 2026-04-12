import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { getSettingSync } from "./storage/configStorage.js";

/**
 * Configuration — Single Source of Truth: prism-config.db
 *
 * All settings are read from prism-config.db via getSettingSync().
 * The Mind Palace dashboard (http://localhost:3333) is the UI for managing
 * these settings. On first run, env vars are bootstrapped into the DB
 * by configStorage.ts — after that, env vars are ignored.
 */

// ─── Server Identity ──────────────────────────────────────────

function resolveServerVersion(): string {
  try {
    const moduleDir = dirname(fileURLToPath(import.meta.url));
    const packageJsonPath = resolve(moduleDir, "../package.json");
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf-8"));
    if (typeof packageJson?.version === "string" && packageJson.version.trim()) {
      return packageJson.version.trim();
    }
  } catch {
    // Fallback below keeps server booting even if package metadata is unavailable.
  }
  return "0.0.0";
}

// REVIEWER NOTE: derive version from package.json so MCP handshake,
// dashboard badge, and package metadata stay in sync.
export const SERVER_CONFIG = {
  name: "prism-mcp",
  version: resolveServerVersion(),
};

// ─── API Keys ─────────────────────────────────────────────────
// All keys are read from prism-config.db (managed via dashboard).
// On first run, env vars are bootstrapped into the DB by configStorage.

export const BRAVE_API_KEY = getSettingSync("brave_api_key", "");
export const GOOGLE_API_KEY = getSettingSync("google_api_key", "");
export const BRAVE_ANSWERS_API_KEY = getSettingSync("brave_answers_api_key", "");
export const VOYAGE_API_KEY = getSettingSync("voyage_api_key", "");

// ─── v2.0: Storage Backend Selection ─────────────────────────
// REVIEWER NOTE: Step 1 of v2.0 introduces a storage abstraction.
// Currently only "supabase" is implemented. "local" (SQLite) is
// coming in Step 2. Default is "supabase" for backward compat.
//
// Set PRISM_STORAGE=local to use SQLite (once implemented).
// Set PRISM_STORAGE=supabase to use Supabase REST API (default).

export const PRISM_STORAGE: "local" | "supabase" =
  (getSettingSync("prism_storage", "local") as "local" | "supabase");

// ─── Optional: Supabase (Session Memory Module) ───────────────
// When both SUPABASE_URL and SUPABASE_KEY are set, session memory tools
// are registered. These tools allow AI agents to persist and recover
// context between sessions.

function sanitizeEnv(value?: string): string | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  // Treat unresolved template placeholders as unset (e.g. "${SUPABASE_URL}")
  if (!trimmed || trimmed.includes("${")) return undefined;
  return trimmed;
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export const SUPABASE_URL = sanitizeEnv(getSettingSync("supabase_url", ""));
export const SUPABASE_KEY = sanitizeEnv(getSettingSync("supabase_key", ""));
export const SUPABASE_CONFIGURED =
  !!SUPABASE_URL &&
  !!SUPABASE_KEY &&
  isHttpUrl(SUPABASE_URL);

if (SUPABASE_URL && !isHttpUrl(SUPABASE_URL)) {
  console.error("Warning: SUPABASE_URL is not a valid http(s) URL. Falling back to local storage.");
}
// Session memory remains core-enabled in both local and Supabase modes.
export const SESSION_MEMORY_ENABLED = true;

// Optional multi-tenant scope ID (used by storage queries and handoffs).
export const PRISM_USER_ID = getSettingSync("user_id", "default");


// ─── v2.1: Auto-Capture Feature ─────────────────────────────
// REVIEWER NOTE: Automatically captures HTML snapshots of local dev servers
// when handoffs are saved. Prevents UI context loss between sessions.
// Opt-in only — set PRISM_AUTO_CAPTURE=true to enable.

export const PRISM_AUTO_CAPTURE = getSettingSync("auto_capture_enabled", "false") === "true";
export const PRISM_CAPTURE_PORTS = getSettingSync("capture_ports", "3000,3001,5173,8080")
  .split(",")
  .map(p => parseInt(p.trim(), 10))
  .filter(p => !isNaN(p));

// ─── v2.3: Debug Logging ──────────────────────────────────────
// Optionally enable verbose output (stderr) for Prism initialization,
// memory indexing, and background tasks.

export const PRISM_DEBUG_LOGGING = getSettingSync("debug_logging", "false") === "true";

// ─── v3.0: Agent Hivemind Feature Flag ───────────────────────
// When enabled, registers 3 additional MCP tools for multi-agent
// coordination: agent_register, agent_heartbeat, agent_list_team.
// The role parameter on existing tools (session_save_ledger, etc.)
// is always available regardless of this flag — adding a parameter
// doesn't increase tool count.
//
// SOURCE OF TRUTH: The Mind Palace dashboard (Settings → Hivemind Mode)
// persists this flag to prism-config.db via getSettingSync() at call time.
//
export const PRISM_ENABLE_HIVEMIND_ENV = getSettingSync("hivemind_enabled", "false") === "true";

// ─── v3.0: Task Router Feature Flag ──────────────────────────
// Routes tasks to the local Claw agent when enabled.
// SOURCE OF TRUTH: dashboard (Settings → Task Router) → prism-config.db.
// Same _ENV pattern: use getSettingSync() at call sites.
// REMOVED: PRISM_TASK_ROUTER_ENABLED used to call getSettingSync() at import time,
// which always returned the fallback due to the ESM race condition (settingsCache=null).
// Use PRISM_TASK_ROUTER_ENABLED_ENV (line ~368) and getSettingSync() at call sites instead.

// ─── v4.1: Auto-Load Projects ────────────────────────────────
// Auto-load is configured exclusively via the Mind Palace dashboard
// ("Auto-Load Projects" checkboxes in Settings). The setting is stored
// in prism-config.db and read at startup via getSettingSync().
//
// The PRISM_AUTOLOAD_PROJECTS env var has been removed — the dashboard
// is the single source of truth. This prevents mismatches between
// env var and dashboard values causing duplicate project loads.

if (PRISM_AUTO_CAPTURE) {
  // Use console.error instead of debugLog here to prevent circular dependency
  if (PRISM_DEBUG_LOGGING) {
    console.error(`[AutoCapture] Enabled for ports: ${PRISM_CAPTURE_PORTS.join(", ")}`);
  }
}

// ─── v5.3: Hivemind Watchdog Thresholds ──────────────────────
// All values have sane defaults. Override via env vars only for
// testing or production tuning. Dashboard UI exposure deferred to v5.4.
export const WATCHDOG_INTERVAL_MS = parseInt(
  getSettingSync("watchdog_interval_ms", "60000"), 10
);
export const WATCHDOG_STALE_MIN = parseInt(
  getSettingSync("watchdog_stale_min", "5"), 10
);
export const WATCHDOG_FROZEN_MIN = parseInt(
  getSettingSync("watchdog_frozen_min", "15"), 10
);
export const WATCHDOG_OFFLINE_MIN = parseInt(
  getSettingSync("watchdog_offline_min", "30"), 10
);
export const WATCHDOG_LOOP_THRESHOLD = parseInt(
  getSettingSync("watchdog_loop_threshold", "5"), 10
);

// ─── v5.4: Background Purge Scheduler ────────────────────────
// Automated background maintenance: TTL sweep, importance decay,
// compaction, and deep storage purge. Runs independently from
// the Watchdog (different cadence: 12h vs 60s).
export const PRISM_SCHEDULER_ENABLED =
  getSettingSync("scheduler_enabled", "true") !== "false";
export const PRISM_SCHEDULER_INTERVAL_MS = parseInt(
  getSettingSync("scheduler_interval_ms", "43200000"), 10
);

// ─── v5.4: Autonomous Web Scholar ─────────────────────────────
// Background LLM research pipeline powered by Brave Search + Firecrawl.
// Tavily can be used as an alternative when TAVILY_API_KEY is set.
// Defaults are conservative to prevent runaway API costs.

export const FIRECRAWL_API_KEY = getSettingSync("firecrawl_api_key", "");
export const TAVILY_API_KEY = getSettingSync("tavily_api_key", "");
export const PRISM_SCHOLAR_ENABLED = getSettingSync("scholar_enabled", "false") === "true";

export const PRISM_SCHOLAR_INTERVAL_MS = parseInt(
  getSettingSync("scholar_interval_ms", "0"), 10
);
export const PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN = parseInt(
  getSettingSync("scholar_max_articles", "3"), 10
);
export const PRISM_SCHOLAR_TOPICS = getSettingSync("scholar_topics", "ai,agents")
  .split(",")
  .map(t => t.trim());

// ─── v6.0: Associative Memory Graph ──────────────────────────
// Controls the age threshold for link strength decay.
// Links not traversed in the last N days lose 0.1 strength per sweep.
export const PRISM_LINK_DECAY_DAYS = parseInt(
  getSettingSync("link_decay_days", "30"), 10
);

// ─── v6.5: Cognitive Architecture (HDC Policy Gateway) ─────────────
// Master feature flag for HDC-driven cognitive routing APIs.
export const PRISM_HDC_ENABLED = getSettingSync("hdc_enabled", "false") === "true";

// Explainability payload toggle for cognitive routing responses.
export const PRISM_HDC_EXPLAINABILITY_ENABLED =
  getSettingSync("hdc_explainability_enabled", "true") !== "false";

const DEFAULT_HDC_FALLBACK_THRESHOLD = 0.85;
const DEFAULT_HDC_CLARIFY_THRESHOLD = 0.95;

const rawHdcFallbackThreshold = parseFloat(
  getSettingSync("hdc_policy_fallback_threshold", String(DEFAULT_HDC_FALLBACK_THRESHOLD))
);
const rawHdcClarifyThreshold = parseFloat(
  getSettingSync("hdc_policy_clarify_threshold", String(DEFAULT_HDC_CLARIFY_THRESHOLD))
);

const hdcThresholdsValid =
  Number.isFinite(rawHdcFallbackThreshold) &&
  Number.isFinite(rawHdcClarifyThreshold) &&
  rawHdcFallbackThreshold >= 0 &&
  rawHdcFallbackThreshold < rawHdcClarifyThreshold &&
  rawHdcClarifyThreshold <= 1;

if (!hdcThresholdsValid) {
  console.error(
    "Warning: Invalid HDC policy thresholds. Falling back to defaults " +
    `(fallback=${DEFAULT_HDC_FALLBACK_THRESHOLD}, clarify=${DEFAULT_HDC_CLARIFY_THRESHOLD}).`
  );
}

export const PRISM_HDC_POLICY_FALLBACK_THRESHOLD = hdcThresholdsValid
  ? rawHdcFallbackThreshold
  : DEFAULT_HDC_FALLBACK_THRESHOLD;

export const PRISM_HDC_POLICY_CLARIFY_THRESHOLD = hdcThresholdsValid
  ? rawHdcClarifyThreshold
  : DEFAULT_HDC_CLARIFY_THRESHOLD;

// ─── v6.2: Graph Soft-Pruning ───────────────────────────────
// Soft-pruning filters weak links from graph/retrieval reads while preserving
// underlying rows for provenance. This does NOT delete links.
export const PRISM_GRAPH_PRUNING_ENABLED = getSettingSync("graph_pruning_enabled", "false") === "true";
export const PRISM_GRAPH_PRUNE_MIN_STRENGTH = parseFloat(
  getSettingSync("graph_prune_min_strength", "0.15")
);

// Scheduler-driven prune sweep controls (WS3)
export const PRISM_GRAPH_PRUNE_PROJECT_COOLDOWN_MS = parseInt(
  getSettingSync("graph_prune_project_cooldown_ms", "600000"), 10
);
export const PRISM_GRAPH_PRUNE_SWEEP_BUDGET_MS = parseInt(
  getSettingSync("graph_prune_sweep_budget_ms", "30000"), 10
);
export const PRISM_GRAPH_PRUNE_MAX_PROJECTS_PER_SWEEP = parseInt(
  getSettingSync("graph_prune_max_projects_per_sweep", "25"), 10
);

// ─── v7.0: ACT-R Cognitive Memory Activation ────────────────
// Scientifically-grounded retrieval re-ranking based on the ACT-R
// cognitive architecture. Replaces simple Ebbinghaus decay with
// a composite similarity + activation model.

/** Master switch for ACT-R activation-based re-ranking. */
export const PRISM_ACTR_ENABLED = getSettingSync("actr_enabled", "false") === "true";

/** ACT-R decay parameter d in t^(-d). Higher = faster forgetting. (Paper default: 0.5) */
export const PRISM_ACTR_DECAY = parseFloat(getSettingSync("actr_decay", "0.5"));

/** Weight of cosine similarity in composite score. (Default: 0.7 — similarity dominates) */
export const PRISM_ACTR_WEIGHT_SIMILARITY = parseFloat(
  getSettingSync("actr_weight_similarity", "0.7")
);

/** Weight of activation boost in composite score. (Default: 0.3 — activation re-ranks) */
export const PRISM_ACTR_WEIGHT_ACTIVATION = parseFloat(
  getSettingSync("actr_weight_activation", "0.3")
);

/** Sigmoid midpoint: activation value that maps to 0.5 boost. (Default: -2.0) */
export const PRISM_ACTR_SIGMOID_MIDPOINT = parseFloat(
  getSettingSync("actr_sigmoid_midpoint", "-2.0")
);

/** Sigmoid steepness k. Higher = sharper discrimination. (Default: 1.0) */
export const PRISM_ACTR_SIGMOID_STEEPNESS = parseFloat(
  getSettingSync("actr_sigmoid_steepness", "1.0")
);

/** Max access log entries per entry for base-level activation. (Default: 50) */
export const PRISM_ACTR_MAX_ACCESSES_PER_ENTRY = parseInt(
  getSettingSync("actr_max_accesses_per_entry", "50"), 10
);

/** AccessLogBuffer flush interval in milliseconds. (Default: 5000ms) */
export const PRISM_ACTR_BUFFER_FLUSH_MS = parseInt(
  getSettingSync("actr_buffer_flush_ms", "5000"), 10
);

/** Days to retain access log entries before pruning. (Default: 90) */
export const PRISM_ACTR_ACCESS_LOG_RETENTION_DAYS = parseInt(
  getSettingSync("actr_access_log_retention_days", "90"), 10
);

// ─── v7.1: Task Router Configuration ─────────────────────────
// Deterministic heuristic-based routing for delegating coding tasks
// between the host cloud model and the local claw-code-agent.
// Set PRISM_TASK_ROUTER_ENABLED=true to unlock the session_task_route tool.

/** Master switch for the task router tool. */
export const PRISM_TASK_ROUTER_ENABLED_ENV = getSettingSync("task_router_enabled", "false") === "true";

/** Confidence threshold below which routing defaults to the host model. (Default: 0.6) */
export const PRISM_TASK_ROUTER_CONFIDENCE_THRESHOLD = parseFloat(
  getSettingSync("task_router_confidence_threshold", "0.6")
);

/** Maximum complexity score (1-10) that Claw can handle. Tasks above this → host. (Default: 4) */
export const PRISM_TASK_ROUTER_MAX_CLAW_COMPLEXITY = parseInt(
  getSettingSync("task_router_max_claw_complexity", "4"), 10
);

// ─── v7.2: Verification Harness ──────────────────────────────

/** Master switch for the v7.2.0 enhanced verification harness. */
export const PRISM_VERIFICATION_HARNESS_ENABLED =
  getSettingSync("verification_enabled", "false") === "true";

/** Comma-separated list of verification layers to run. */
export const PRISM_VERIFICATION_LAYERS =
  getSettingSync("verification_layers", "data,agent,pipeline")
    .split(",").map(l => l.trim()).filter(Boolean);

/** Default severity floor for all assertions. Overrides individual assertion severity when higher. */
export const PRISM_VERIFICATION_DEFAULT_SEVERITY =
  getSettingSync("verification_default_severity", "warn") as "warn" | "gate" | "abort";

// ─── v7.3: Dark Factory Orchestration ─────────────────────────
// Autonomous pipeline runner: PLAN → EXECUTE → VERIFY → iterate.
// Opt-in because it executes LLM calls in the background.

/** Master switch for the Dark Factory background runner. */
export const PRISM_DARK_FACTORY_ENABLED_ENV =
  getSettingSync("dark_factory_enabled", "false") === "true";

/** Poll interval for the runner loop (ms). Default: 30s. */
export const PRISM_DARK_FACTORY_POLL_MS = parseInt(
  getSettingSync("dark_factory_poll_ms", "30000"), 10
);

/** Default max wall-clock time per pipeline (ms). Default: 15 minutes. */
export const PRISM_DARK_FACTORY_MAX_RUNTIME_MS = parseInt(
  getSettingSync("dark_factory_max_runtime_ms", "900000"), 10
);

// ─── v8.0: Synapse — Spreading Activation Engine ──────────────
// Multi-hop energy propagation through memory_links graph.
// Enabled by default. Set PRISM_SYNAPSE_ENABLED=false to fall back
// to 1-hop candidateScopedSpreadingActivation (v7.0 behavior).

/** Master switch for the Synapse engine. Enabled by default (opt-out). */
export const PRISM_SYNAPSE_ENABLED =
  getSettingSync("synapse_enabled", "true") !== "false";

/** Number of propagation iterations (depth). Higher = deeper traversal, more latency. (Default: 3) */
export const PRISM_SYNAPSE_ITERATIONS = parseInt(
  getSettingSync("synapse_iterations", "3"), 10
);

/** Energy attenuation per hop. Must be < 1.0 for convergence. (Default: 0.8) */
export const PRISM_SYNAPSE_SPREAD_FACTOR = parseFloat(
  getSettingSync("synapse_spread_factor", "0.8")
);

/** Hard cap on final output nodes (lateral inhibition). (Default: 7) */
export const PRISM_SYNAPSE_LATERAL_INHIBITION = parseInt(
  getSettingSync("synapse_lateral_inhibition", "7"), 10
);

/** Soft cap on active nodes per iteration (prevents explosion). (Default: 20) */
export const PRISM_SYNAPSE_SOFT_CAP = parseInt(
  getSettingSync("synapse_soft_cap", "20"), 10
);

// ─── v9.0: Affect-Tagged Memory (Valence Engine) ─────────────
// Derives emotional valence from experience events and uses
// Affective Salience (|valence| boosts retrieval) for ranking.

/** Master switch for affect-tagged memory. (Default: true) */
export const PRISM_VALENCE_ENABLED =
  getSettingSync("valence_enabled", "true") !== "false";

/** Weight of |valence| in hybrid scoring formula. (Default: 0.1) */
export const PRISM_VALENCE_WEIGHT = parseFloat(
  getSettingSync("valence_weight", "0.1")
);

/** Average valence below this threshold triggers a UX warning. (Default: -0.3) */
export const PRISM_VALENCE_WARNING_THRESHOLD = parseFloat(
  getSettingSync("valence_warning_threshold", "-0.3")
);

// ─── v9.0: Token-Economic RL (Cognitive Budget) ──────────────
// Implements a strict token economy for agent memory operations.
// Budget is persistent (stored in session_handoffs.cognitive_budget).

/** Master switch for the cognitive budget system. (Default: true) */
export const PRISM_COGNITIVE_BUDGET_ENABLED =
  getSettingSync("cognitive_budget_enabled", "true") !== "false";

/** Initial budget size per project in tokens. (Default: 2000) */
export const PRISM_COGNITIVE_BUDGET_SIZE = parseInt(
  getSettingSync("cognitive_budget_size", "2000"), 10
);

/** Master switch for the surprisal gate. (Default: true) */
export const PRISM_SURPRISAL_GATE_ENABLED =
  getSettingSync("surprisal_gate_enabled", "true") !== "false";

