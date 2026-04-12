This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/config.ts, src/backgroundScheduler.ts, src/scholar/webScholar.ts, src/dashboard/server.ts
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
src/
  dashboard/
    server.ts
  scholar/
    webScholar.ts
  backgroundScheduler.ts
  config.ts
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="src/scholar/webScholar.ts">
import { 
  BRAVE_API_KEY, 
  FIRECRAWL_API_KEY, 
  PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN, 
  PRISM_USER_ID, 
  PRISM_SCHOLAR_TOPICS 
} from "../config.js";
import { getStorage } from "../storage/index.js";
import { debugLog } from "../utils/logger.js";
import { getLLMProvider } from "../utils/llm/factory.js";
import { randomUUID } from "node:crypto";
import { performWebSearchRaw } from "../utils/braveApi.js";
import { getTracer } from "../utils/telemetry.js";

interface FirecrawlScrapeResponse {
  success: boolean;
  data: {
    markdown?: string;
  };
}

/**
 * Runs the Web Scholar pipeline:
 * 1. Picks a random topic from configuration
 * 2. Searches Brave for recent articles
 * 3. Scrapes articles as markdown using Firecrawl
 * 4. Summarizes the findings via LLM
 * 5. Injects the summary directly into Prism's semantic ledger
 */
export async function runWebScholar(): Promise<void> {
  const tracer = getTracer();
  const span = tracer.startSpan("background.web_scholar");
  
  try {
    if (!BRAVE_API_KEY || !FIRECRAWL_API_KEY) {
      debugLog("[WebScholar] Skipped: Missing BRAVE_API_KEY or FIRECRAWL_API_KEY");
      span.setAttribute("scholar.skipped_reason", "missing_keys");
      return;
    }

    if (!PRISM_SCHOLAR_TOPICS || PRISM_SCHOLAR_TOPICS.length === 0) {
      debugLog("[WebScholar] Skipped: No topics configured in PRISM_SCHOLAR_TOPICS");
      span.setAttribute("scholar.skipped_reason", "no_topics");
      return;
    }

    // 1. Pick a random topic to research
    const topic = PRISM_SCHOLAR_TOPICS[Math.floor(Math.random() * PRISM_SCHOLAR_TOPICS.length)];
    debugLog(`[WebScholar] 🧠 Starting research on topic: "${topic}"`);
    span.setAttribute("scholar.topic", topic);

    // 2. Search Brave for articles
    const braveResponse = await performWebSearchRaw(topic, PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN);
    const braveData = JSON.parse(braveResponse);
    const urls = (braveData.web?.results || []).map((r: any) => r.url).filter(Boolean);

    if (urls.length === 0) {
      debugLog(`[WebScholar] No articles found for "${topic}"`);
      span.setAttribute("scholar.skipped_reason", "no_search_results");
      return;
    }

    debugLog(`[WebScholar] Found ${urls.length} articles. Scraping with Firecrawl...`);
    span.setAttribute("scholar.articles_found", urls.length);

    // 3. Scrape each URL with Firecrawl
    const scrapedTexts: string[] = [];
    for (const url of urls) {
      try {
        debugLog(`[WebScholar] Scraping: ${url}`);
        const scrapeRes = await fetch("https://api.firecrawl.dev/v1/scrape", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${FIRECRAWL_API_KEY}`
          },
          body: JSON.stringify({
            url,
            formats: ["markdown"],
          })
        });

        if (!scrapeRes.ok) {
          console.error(`[WebScholar] Firecrawl failed for ${url}: ${scrapeRes.status}`);
          continue;
        }

        const result = (await scrapeRes.json()) as FirecrawlScrapeResponse;
        if (result.success && result.data?.markdown) {
          scrapedTexts.push(`Source: ${url}\n\n${result.data.markdown}\n\n---\n`);
        }
      } catch (err) {
        console.error(`[WebScholar] Failed to scrape ${url}:`, err);
      }
    }

    if (scrapedTexts.length === 0) {
      debugLog(`[WebScholar] Could not extract markdown from any articles.`);
      span.setAttribute("scholar.skipped_reason", "all_scrapes_failed");
      return;
    }

    span.setAttribute("scholar.articles_scraped", scrapedTexts.length);

    // 4. Summarize findings using LLM
    debugLog(`[WebScholar] Summarizing ${scrapedTexts.length} articles...`);
    const combinedText = scrapedTexts.join("\n");
    const prompt = `You are an AI research assistant. You have been asked to research the topic: "${topic}".
Read the following scraped web articles and write a comprehensive, markdown-formatted report summarizing the key findings, trends, and actionable insights. Focus heavily on facts, data, and actual content. Do NOT just list the articles. Synthesize the information.

### Scraped Articles:
${combinedText}`;

    const llm = getLLMProvider();
    const summary = await llm.generateText(prompt);

    // 5. Inject the summary back into Prism memory
    const storage = await getStorage();
    await storage.saveLedger({
      id: randomUUID(),
      project: "prism-scholar",
      conversation_id: "scholar-bg-" + Date.now(),
      user_id: PRISM_USER_ID, // Use the configured user ID for multi-tenant isolation
      role: "scholar",
      summary: `Autonomous Web Scholar Research: ${topic}\n\n${summary}`,
      keywords: [topic, "research", "autonomous", "scholar"],
      event_type: "learning",
      importance: 7, // Auto-graduate scholar findings as important insights
      created_at: new Date().toISOString()
    });

    debugLog(`[WebScholar] ✅ Research complete and saved to ledger under project 'prism-scholar'.`);
    span.setAttribute("scholar.success", true);

  } catch (err) {
    console.error("[WebScholar] Pipeline failed:", err);
    span.setAttribute("scholar.error", String(err));
  } finally {
    span.end();
  }
}
</file>

<file path="src/backgroundScheduler.ts">
/**
 * Background Purge Scheduler (v5.4) — Unified Maintenance Automation
 *
 * Automates all storage maintenance tasks that were previously manual-only:
 *   1. TTL Sweep     — expireByTTL() for all projects with configured TTL
 *   2. Importance Decay — decayImportance() across all projects
 *   3. Compaction    — auto-compact projects exceeding entry threshold
 *   4. Deep Purge    — purge float32 embeddings for old compressed entries
 *
 * Architecture:
 *   - Single `setInterval` loop (default: 12 hours)
 *   - Independent from Hivemind Watchdog (60s loop) — different cadence, different concerns
 *   - Non-blocking: all errors caught and logged, sweep never crashes the server
 *   - Configurable via env vars: PRISM_SCHEDULER_ENABLED, PRISM_SCHEDULER_INTERVAL_MS
 *
 * Each sweep runs tasks sequentially (not parallel) to avoid overloading
 * storage backends during maintenance windows.
 */

import { getStorage } from "./storage/index.js";
import { PRISM_USER_ID } from "./config.js";
import { debugLog } from "./utils/logger.js";

// ─── Configuration ───────────────────────────────────────────

export interface SchedulerConfig {
  /** Sweep interval in milliseconds (default: 43_200_000 = 12 hours) */
  intervalMs: number;
  /** Run TTL expiry sweep for projects with configured policies (default: true) */
  enableTTLSweep: boolean;
  /** Run importance decay for behavioral memory entries (default: true) */
  enableDecay: boolean;
  /** Auto-compact projects exceeding compaction threshold (default: true) */
  enableCompaction: boolean;
  /** Purge float32 embeddings for old entries with compressed blobs (default: true) */
  enableDeepPurge: boolean;
  /** Minimum age in days for deep purge eligibility (default: 30) */
  purgeOlderThanDays: number;
  /** Minimum entries before compaction triggers (default: 50) */
  compactionThreshold: number;
  /** Number of recent entries to keep intact during compaction (default: 10) */
  compactionKeepRecent: number;
  /** Days before importance decay applies to behavioral entries (default: 30) */
  decayDays: number;
}

export const DEFAULT_SCHEDULER_CONFIG: SchedulerConfig = {
  intervalMs: 43_200_000,       // 12 hours
  enableTTLSweep: true,
  enableDecay: true,
  enableCompaction: true,
  enableDeepPurge: true,
  purgeOlderThanDays: 30,
  compactionThreshold: 50,
  compactionKeepRecent: 10,
  decayDays: 30,
};

// ─── Scheduler State ─────────────────────────────────────────

let schedulerInterval: ReturnType<typeof setInterval> | null = null;

/** Tracks the last completed sweep for dashboard status */
let lastSweepResult: SchedulerSweepResult | null = null;

/** When the scheduler was started */
let schedulerStartedAt: string | null = null;

export interface SchedulerSweepResult {
  /** ISO timestamp of sweep start */
  startedAt: string;
  /** ISO timestamp of sweep completion */
  completedAt: string;
  /** Duration in milliseconds */
  durationMs: number;
  /** Results per task */
  tasks: {
    ttlSweep: { ran: boolean; projectsSwept: number; totalExpired: number; error?: string };
    importanceDecay: { ran: boolean; projectsDecayed: number; error?: string };
    compaction: { ran: boolean; projectsCompacted: number; error?: string };
    deepPurge: { ran: boolean; purged: number; reclaimedBytes: number; error?: string };
  };
}

// ─── Public API ──────────────────────────────────────────────

/**
 * Start the background scheduler.
 * Returns a cleanup function that stops the interval.
 *
 * @param config - Override defaults for testing or production tuning
 */
export function startScheduler(config?: Partial<SchedulerConfig>): () => void {
  const cfg: SchedulerConfig = { ...DEFAULT_SCHEDULER_CONFIG, ...config };

  if (schedulerInterval) {
    clearInterval(schedulerInterval);
  }

  schedulerStartedAt = new Date().toISOString();

  schedulerInterval = setInterval(() => {
    runSchedulerSweep(cfg).catch(err => {
      console.error(`[Scheduler] Sweep error (non-fatal): ${err}`);
    });
  }, cfg.intervalMs);

  // Run an immediate first sweep (after a short delay to let storage fully warm up)
  setTimeout(() => {
    runSchedulerSweep(cfg).catch(err => {
      console.error(`[Scheduler] Initial sweep error (non-fatal): ${err}`);
    });
  }, 5_000);

  const enabledTasks = [
    cfg.enableTTLSweep && "TTL",
    cfg.enableDecay && "Decay",
    cfg.enableCompaction && "Compaction",
    cfg.enableDeepPurge && "DeepPurge",
  ].filter(Boolean).join(", ");

  console.error(
    `[Scheduler] ⏰ Started (interval=${formatDuration(cfg.intervalMs)}, tasks=[${enabledTasks}])`
  );

  return () => {
    if (schedulerInterval) {
      clearInterval(schedulerInterval);
      schedulerInterval = null;
      console.error("[Scheduler] Stopped");
    }
  };
}

import { PRISM_SCHOLAR_ENABLED, PRISM_SCHOLAR_INTERVAL_MS } from "./config.js";
import { runWebScholar } from "./scholar/webScholar.js";

/**
 * Get scheduler status for the dashboard.
 */
export function getSchedulerStatus(): {
  running: boolean;
  startedAt: string | null;
  intervalMs: number;
  lastSweep: SchedulerSweepResult | null;
  scholarRunning: boolean;
  scholarIntervalMs: number;
} {
  return {
    running: schedulerInterval !== null,
    startedAt: schedulerStartedAt,
    intervalMs: DEFAULT_SCHEDULER_CONFIG.intervalMs,
    lastSweep: lastSweepResult,
    scholarRunning: scholarInterval !== null,
    scholarIntervalMs: PRISM_SCHOLAR_INTERVAL_MS,
  };
}

// ─── Scholar State ───────────────────────────────────────────
let scholarInterval: ReturnType<typeof setInterval> | null = null;

export function startScholarScheduler(): () => void {
  if (scholarInterval) {
    clearInterval(scholarInterval);
  }

  if (!PRISM_SCHOLAR_ENABLED || PRISM_SCHOLAR_INTERVAL_MS <= 0) {
    debugLog("[WebScholar] 🕒 Scheduler disabled (PRISM_SCHOLAR_ENABLED=false or PRISM_SCHOLAR_INTERVAL_MS=0)");
    return () => {};
  }

  // Initial trigger after 30s to avoid thundering herd on boot
  setTimeout(() => {
    runWebScholar().catch(err => {
        console.error(`[WebScholar] Initial run error: ${err}`);
    });
  }, 30_000);

  scholarInterval = setInterval(() => {
    runWebScholar().catch(err => {
      console.error(`[WebScholar] Sweep error: ${err}`);
    });
  }, PRISM_SCHOLAR_INTERVAL_MS);

  console.error(
    `[WebScholar] ⏰ Started (interval=${formatDuration(PRISM_SCHOLAR_INTERVAL_MS)})`
  );

  return () => {
    if (scholarInterval) {
      clearInterval(scholarInterval);
      scholarInterval = null;
      console.error("[WebScholar] Stopped");
    }
  };
}

// ─── Core Sweep Logic ────────────────────────────────────────

/**
 * Single scheduler sweep — orchestrates all maintenance tasks.
 * Exported for testing.
 *
 * Execution order:
 *   1. TTL Sweep     — lightweight SQL UPDATEs (fast)
 *   2. Importance Decay — lightweight SQL UPDATEs (fast)
 *   3. Compaction    — LLM-powered summarization (slow, expensive)
 *   4. Deep Purge    — SQL UPDATEs to NULL embeddings (moderate)
 */
export async function runSchedulerSweep(
  cfg: SchedulerConfig = DEFAULT_SCHEDULER_CONFIG
): Promise<SchedulerSweepResult> {
  const sweepStart = Date.now();
  const startedAt = new Date().toISOString();

  const result: SchedulerSweepResult = {
    startedAt,
    completedAt: "",
    durationMs: 0,
    tasks: {
      ttlSweep: { ran: false, projectsSwept: 0, totalExpired: 0 },
      importanceDecay: { ran: false, projectsDecayed: 0 },
      compaction: { ran: false, projectsCompacted: 0 },
      deepPurge: { ran: false, purged: 0, reclaimedBytes: 0 },
    },
  };

  debugLog("[Scheduler] 🔄 Sweep starting...");

  const storage = await getStorage();

  // ── Task 1: TTL Sweep ──────────────────────────────────────
  if (cfg.enableTTLSweep) {
    try {
      result.tasks.ttlSweep.ran = true;
      const projects = await storage.listProjects();
      const settings = await storage.getAllSettings();

      for (const project of projects) {
        const ttlKey = `ttl:${project}`;
        const ttlValue = settings[ttlKey];
        if (!ttlValue) continue;

        const ttlDays = parseInt(ttlValue, 10);
        if (isNaN(ttlDays) || ttlDays <= 0) continue;

        try {
          const { expired } = await storage.expireByTTL(project, ttlDays, PRISM_USER_ID);
          result.tasks.ttlSweep.projectsSwept++;
          result.tasks.ttlSweep.totalExpired += expired;
          if (expired > 0) {
            debugLog(`[Scheduler] TTL: expired ${expired} entries for "${project}" (>${ttlDays}d)`);
          }
        } catch (err) {
          debugLog(`[Scheduler] TTL sweep failed for "${project}": ${err}`);
        }
      }
    } catch (err) {
      result.tasks.ttlSweep.error = err instanceof Error ? err.message : String(err);
      console.error(`[Scheduler] TTL sweep error: ${err}`);
    }
  }

  // ── Task 2: Importance Decay ───────────────────────────────
  if (cfg.enableDecay) {
    try {
      result.tasks.importanceDecay.ran = true;
      const projects = await storage.listProjects();

      for (const project of projects) {
        try {
          await storage.decayImportance(project, PRISM_USER_ID, cfg.decayDays);
          result.tasks.importanceDecay.projectsDecayed++;
        } catch (err) {
          debugLog(`[Scheduler] Decay failed for "${project}": ${err}`);
        }
      }
    } catch (err) {
      result.tasks.importanceDecay.error = err instanceof Error ? err.message : String(err);
      console.error(`[Scheduler] Importance decay error: ${err}`);
    }
  }

  // ── Task 3: Compaction ─────────────────────────────────────
  // NOTE: Compaction uses LLM summarization which is expensive.
  // We only trigger the candidate detection here — actual compaction
  // is deferred to avoid blocking the sweep with long LLM calls.
  // Instead, we log which projects need compaction for dashboard visibility.
  if (cfg.enableCompaction) {
    try {
      result.tasks.compaction.ran = true;
      const candidates = await storage.getCompactionCandidates(
        cfg.compactionThreshold, cfg.compactionKeepRecent, PRISM_USER_ID
      );

      if (candidates.length > 0) {
        // Import compaction handler dynamically to avoid circular deps
        const { compactLedgerHandler } = await import("./tools/compactionHandler.js");

        for (const candidate of candidates) {
          try {
            debugLog(
              `[Scheduler] Compacting "${candidate.project}": ` +
              `${candidate.total_entries} entries (${candidate.to_compact} to compact)`
            );
            await compactLedgerHandler({
              project: candidate.project,
              threshold: cfg.compactionThreshold,
              keep_recent: cfg.compactionKeepRecent,
              dry_run: false,
            });
            result.tasks.compaction.projectsCompacted++;
          } catch (err) {
            debugLog(`[Scheduler] Compaction failed for "${candidate.project}": ${err}`);
          }
        }
      }
    } catch (err) {
      result.tasks.compaction.error = err instanceof Error ? err.message : String(err);
      console.error(`[Scheduler] Compaction error: ${err}`);
    }
  }

  // ── Task 4: Deep Purge ─────────────────────────────────────
  if (cfg.enableDeepPurge) {
    try {
      result.tasks.deepPurge.ran = true;
      const purgeResult = await storage.purgeHighPrecisionEmbeddings({
        olderThanDays: cfg.purgeOlderThanDays,
        dryRun: false,
        userId: PRISM_USER_ID,
      });
      result.tasks.deepPurge.purged = purgeResult.purged;
      result.tasks.deepPurge.reclaimedBytes = purgeResult.reclaimedBytes;

      if (purgeResult.purged > 0) {
        debugLog(
          `[Scheduler] Deep purge: freed ${formatBytes(purgeResult.reclaimedBytes)} ` +
          `(${purgeResult.purged} entries purged)`
        );
      }
    } catch (err) {
      result.tasks.deepPurge.error = err instanceof Error ? err.message : String(err);
      console.error(`[Scheduler] Deep purge error: ${err}`);
    }
  }

  // ── Finalize ───────────────────────────────────────────────
  result.completedAt = new Date().toISOString();
  result.durationMs = Date.now() - sweepStart;
  lastSweepResult = result;

  // Build summary line
  const parts: string[] = [];
  if (result.tasks.ttlSweep.ran && result.tasks.ttlSweep.totalExpired > 0) {
    parts.push(`TTL:${result.tasks.ttlSweep.totalExpired} expired`);
  }
  if (result.tasks.importanceDecay.ran) {
    parts.push(`Decay:${result.tasks.importanceDecay.projectsDecayed} projects`);
  }
  if (result.tasks.compaction.ran && result.tasks.compaction.projectsCompacted > 0) {
    parts.push(`Compact:${result.tasks.compaction.projectsCompacted} projects`);
  }
  if (result.tasks.deepPurge.ran && result.tasks.deepPurge.purged > 0) {
    parts.push(`Purge:${result.tasks.deepPurge.purged} entries (${formatBytes(result.tasks.deepPurge.reclaimedBytes)})`);
  }

  const summaryLine = parts.length > 0
    ? parts.join(" | ")
    : "no maintenance actions needed";

  debugLog(`[Scheduler] ✅ Sweep completed in ${result.durationMs}ms — ${summaryLine}`);

  return result;
}

// ─── Helpers ─────────────────────────────────────────────────

function formatDuration(ms: number): string {
  if (ms < 60_000) return `${ms}ms`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m`;
  return `${Math.round(ms / 3_600_000)}h`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1_048_576).toFixed(1)}MB`;
}
</file>

<file path="src/config.ts">
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Configuration & Environment Variables
 *
 * This file is loaded once at startup. It reads environment variables,
 * validates required ones, and exports them for use throughout the server.
 *
 * Environment variable guide:
 *   BRAVE_API_KEY          — (required) API key for Brave Search Pro. Get one at https://brave.com/search/api/
 *   GOOGLE_API_KEY         — (optional) API key for Google AI Studio / Gemini. Enables paper analysis.
 *   BRAVE_ANSWERS_API_KEY  — (optional) API key for Brave Answers (AI grounding). Enables brave_answers tool.
 *   SUPABASE_URL           — (optional) Your Supabase project URL. Enables session memory tools.
 *   SUPABASE_KEY           — (optional) Your Supabase anon/service key. Enables session memory tools.
 *   PRISM_USER_ID          — (optional) Unique tenant ID for multi-user Supabase instances.
 *                            Defaults to "default". Set per-user in Claude Desktop config.
 *
 * If a required key is missing, the process exits immediately.
 * If an optional key is missing, a warning is logged but the server continues
 * with reduced functionality (the corresponding tools will be unavailable).
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

// ─── Required: Brave Search API Key ───────────────────────────

export const BRAVE_API_KEY = process.env.BRAVE_API_KEY;
if (!BRAVE_API_KEY) {
  console.error("Warning: BRAVE_API_KEY environment variable is missing. Search tools will return errors when called.");
}

// ─── Optional: Google Gemini API Key ──────────────────────────
// Used by the gemini_research_paper_analysis tool.
// Without this, the tool will still appear but will error when called.

export const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
if (!GOOGLE_API_KEY) {
  console.error("Warning: GOOGLE_API_KEY environment variable is missing. Gemini research features will be unavailable.");
}

// ─── Optional: Brave Answers API Key ──────────────────────────
// Used by the brave_answers tool for AI-grounded answers.
// This is a separate API key from the main Brave Search key.

export const BRAVE_ANSWERS_API_KEY = process.env.BRAVE_ANSWERS_API_KEY;
if (!BRAVE_ANSWERS_API_KEY) {
  console.error("Warning: BRAVE_ANSWERS_API_KEY environment variable is missing. Brave Answers tool will be unavailable.");
}

// ─── Optional: Supabase (Session Memory Module) ───────────────
// When both SUPABASE_URL and SUPABASE_KEY are set, session memory tools
// are registered. These tools allow AI agents to persist and recover
// context between sessions.

export const SUPABASE_URL = process.env.SUPABASE_URL;
export const SUPABASE_KEY = process.env.SUPABASE_KEY;
export const SESSION_MEMORY_ENABLED = !!(SUPABASE_URL && SUPABASE_KEY);
// Note: debug() is defined at the bottom of this file; these lines
// execute at import time after the full module is loaded by Node.
if (!SESSION_MEMORY_ENABLED) {
  console.error("Info: Session memory disabled (set SUPABASE_URL + SUPABASE_KEY to enable)");
}

// ─── v2.0: Storage Backend Selection ─────────────────────────
// REVIEWER NOTE: Step 1 of v2.0 introduces a storage abstraction.
// Currently only "supabase" is implemented. "local" (SQLite) is
// coming in Step 2. Default is "supabase" for backward compat.
//
// Set PRISM_STORAGE=local to use SQLite (once implemented).
// Set PRISM_STORAGE=supabase to use Supabase REST API (default).

export const PRISM_STORAGE: "local" | "supabase" =
  (process.env.PRISM_STORAGE as "local" | "supabase") || "supabase";
// Logged at debug level — see debug() at bottom of file

// ─── Optional: Multi-Tenant User ID ──────────────────────────
// REVIEWER NOTE: When multiple users share the same Supabase instance,
// PRISM_USER_ID isolates their data. Each user sets a unique ID in their
// Claude Desktop config. All queries are scoped to this user_id.
//
// Defaults to "default" for backward compatibility — existing single-user
// installations work without any config changes.
//
// For enterprise: use a stable unique identifier (UUID, email hash, etc.)
// For personal use: any unique string works (e.g., "alice", "bob")

export const PRISM_USER_ID = process.env.PRISM_USER_ID || "default";
// Multi-tenant info logged at debug level in startServer()

// ─── v2.1: Auto-Capture Feature ─────────────────────────────
// REVIEWER NOTE: Automatically captures HTML snapshots of local dev servers
// when handoffs are saved. Prevents UI context loss between sessions.
// Opt-in only — set PRISM_AUTO_CAPTURE=true to enable.

export const PRISM_AUTO_CAPTURE = process.env.PRISM_AUTO_CAPTURE === "true";
export const PRISM_CAPTURE_PORTS = (process.env.PRISM_CAPTURE_PORTS || "3000,3001,5173,8080")
  .split(",")
  .map(p => parseInt(p.trim(), 10))
  .filter(p => !isNaN(p));

// ─── v2.3: Debug Logging ──────────────────────────────────────
// Optionally enable verbose output (stderr) for Prism initialization,
// memory indexing, and background tasks.

export const PRISM_DEBUG_LOGGING = process.env.PRISM_DEBUG_LOGGING === "true";

// ─── v3.0: Agent Hivemind Feature Flag ───────────────────────
// When enabled, registers 3 additional MCP tools for multi-agent
// coordination: agent_register, agent_heartbeat, agent_list_team.
// The role parameter on existing tools (session_save_ledger, etc.)
// is always available regardless of this flag — adding a parameter
// doesn't increase tool count.
// Set PRISM_ENABLE_HIVEMIND=true to unlock the Agent Registry tools.

export const PRISM_ENABLE_HIVEMIND = process.env.PRISM_ENABLE_HIVEMIND === "true";

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
  process.env.PRISM_WATCHDOG_INTERVAL_MS || "60000", 10
);
export const WATCHDOG_STALE_MIN = parseInt(
  process.env.PRISM_WATCHDOG_STALE_MIN || "5", 10
);
export const WATCHDOG_FROZEN_MIN = parseInt(
  process.env.PRISM_WATCHDOG_FROZEN_MIN || "15", 10
);
export const WATCHDOG_OFFLINE_MIN = parseInt(
  process.env.PRISM_WATCHDOG_OFFLINE_MIN || "30", 10
);
export const WATCHDOG_LOOP_THRESHOLD = parseInt(
  process.env.PRISM_WATCHDOG_LOOP_THRESHOLD || "5", 10
);

// ─── v5.4: Background Purge Scheduler ────────────────────────
// Automated background maintenance: TTL sweep, importance decay,
// compaction, and deep storage purge. Runs independently from
// the Watchdog (different cadence: 12h vs 60s).
export const PRISM_SCHEDULER_ENABLED =
  process.env.PRISM_SCHEDULER_ENABLED !== "false"; // Default: true
export const PRISM_SCHEDULER_INTERVAL_MS = parseInt(
  process.env.PRISM_SCHEDULER_INTERVAL_MS || "43200000", 10  // 12 hours
);

// ─── v5.4: Autonomous Web Scholar ─────────────────────────────
// Background LLM research pipeline powered by Brave Search + Firecrawl.
// Defaults are conservative to prevent runaway API costs.

export const FIRECRAWL_API_KEY = process.env.FIRECRAWL_API_KEY;
if (!FIRECRAWL_API_KEY) {
  console.error("Warning: FIRECRAWL_API_KEY environment variable is missing. Web Scholar will be unavailable.");
}

export const PRISM_SCHOLAR_ENABLED = process.env.PRISM_SCHOLAR_ENABLED === "true"; // Opt-in
export const PRISM_SCHOLAR_INTERVAL_MS = parseInt(
  process.env.PRISM_SCHOLAR_INTERVAL_MS || "0", 10  // Default manual-only
);
export const PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN = parseInt(
  process.env.PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN || "3", 10
);
export const PRISM_SCHOLAR_TOPICS = (process.env.PRISM_SCHOLAR_TOPICS || "ai,agents")
  .split(",")
  .map(t => t.trim());
</file>

<file path="src/dashboard/server.ts">
/**
 * Mind Palace Dashboard — HTTP Server (v2.0 — Step 8)
 *
 * Zero-dependency HTTP server serving the Prism Mind Palace UI.
 * Runs alongside the MCP stdio server on a separate port.
 *
 * ═══════════════════════════════════════════════════════════════════
 * CRITICAL MCP SAFETY:
 *   The MCP server communicates via stdout. ANY console.log() here
 *   will corrupt the JSON-RPC stream and crash the agent.
 *   All logging uses console.error() exclusively.
 *
 * ENDPOINTS:
 *   GET /                   → Dashboard UI (HTML)
 *   GET /api/projects       → List all projects with handoff data
 *   GET /api/project?name=  → Full project data (context, ledger, history)
 * ═══════════════════════════════════════════════════════════════════
 */

import * as http from "http";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";
import { exec } from "child_process";
import { getStorage } from "../storage/index.js";
import { PRISM_USER_ID, SERVER_CONFIG } from "../config.js";
import { renderDashboardHTML } from "./ui.js";
import { getAllSettings, setSetting, getSetting } from "../storage/configStorage.js";
import { compactLedgerHandler } from "../tools/compactionHandler.js";


const PORT = parseInt(process.env.PRISM_DASHBOARD_PORT || "3000", 10);

/** Read HTTP request body as string */
function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", chunk => { data += chunk; });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

/**
 * Kill any existing process holding the dashboard port.
 * This prevents zombie dashboard processes from surviving IDE restarts
 * and serving stale versions of the UI.
 *
 * CRITICAL: Uses async exec() instead of execSync() to avoid blocking
 * the Node.js event loop. Blocking during startup prevents the MCP
 * stdio transport from responding to the initialize handshake in time,
 * causing Antigravity to report MCP_SERVER_INIT_ERROR.
 */
async function killPortHolder(port: number): Promise<void> {
  return new Promise((resolve) => {
    exec(`lsof -ti tcp:${port}`, { encoding: "utf-8" }, (err, stdout) => {
      if (err) {
        // lsof exits with code 1 when no matches found — that's expected.
        // Any other failure (lsof missing, permission denied, etc.) gets a warning.
        const isNoMatch = err.code === 1;
        if (!isNoMatch) {
          console.error(
            `[Dashboard] killPortHolder: could not check port ${port} (lsof may not be installed) — skipping.`
          );
        }
        return resolve();
      }

      const pids = stdout.trim().split("\n").filter(Boolean);
      if (pids.length === 0) return resolve();

      // Don't kill ourselves
      const myPid = String(process.pid);
      const stalePids = pids.filter(p => p !== myPid);

      if (stalePids.length > 0) {
        console.error(`[Dashboard] Killing stale process(es) on port ${port}: ${stalePids.join(", ")}`);
        exec(`kill ${stalePids.join(" ")}`, () => {
          // Brief pause to let the OS release the port
          setTimeout(resolve, 300);
        });
      } else {
        resolve();
      }
    });
  });
}

export async function startDashboardServer(): Promise<void> {
  // Await port cleanup before binding. This adds ~300ms from lsof + setTimeout,
  // but is safe because startDashboardServer() is already deferred to
  // setTimeout(0) in server.ts — the MCP stdio handshake is long finished.
  // The old fire-and-forget approach caused a deadly race condition:
  //   1. listen() fired BEFORE killPortHolder cleared the port → EADDRINUSE
  //   2. killPortHolder then killed the OTHER instance's entire process
  //   3. Result: no instance ever held port 3000
  await killPortHolder(PORT).catch(() => {});

  // Lazy storage accessor — returns null if storage isn't ready yet.
  // API routes gracefully degrade with 503 instead of blocking startup.
  let _storage: Awaited<ReturnType<typeof getStorage>> | null = null;
  const getStorageSafe = async (): Promise<Awaited<ReturnType<typeof getStorage>> | null> => {
    if (_storage) return _storage;
    try {
      _storage = await getStorage();
      return _storage;
    } catch {
      return null;
    }
  };

  /**
   * v5.1: Optional HTTP Basic Auth for remote dashboard access.
   *
   * HOW IT WORKS:
   *   1. If PRISM_DASHBOARD_USER and PRISM_DASHBOARD_PASS are NOT set → auth is disabled (backward compatible)
   *   2. If set → every request must provide Basic Auth credentials OR a valid session cookie
   *   3. On successful auth → a session cookie (24h) is set so users don't re-authenticate on every request
   *   4. On failure → a styled login page is shown (not a raw 401 popup)
   *
   * SECURITY NOTES:
   *   - This is HTTP Basic Auth — suitable for LAN/VPN access, NOT public internet without HTTPS
   *   - Session tokens are random 64-char hex strings stored in-memory (cleared on server restart)
   *   - Timing-safe comparison prevents credential timing attacks
   */
  const AUTH_USER = process.env.PRISM_DASHBOARD_USER || "";
  const AUTH_PASS = process.env.PRISM_DASHBOARD_PASS || "";
  const AUTH_ENABLED = AUTH_USER.length > 0 && AUTH_PASS.length > 0;
  const SESSION_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
  const activeSessions = new Map<string, number>(); // token → expiry timestamp

  /** Generate a random session token */
  function generateToken(): string {
    const chars = "abcdef0123456789";
    let token = "";
    for (let i = 0; i < 64; i++) {
      token += chars[Math.floor(Math.random() * chars.length)];
    }
    return token;
  }

  /** Timing-safe string comparison to prevent timing attacks */
  function safeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }
    return result === 0;
  }

  /** Check if request is authenticated (returns true if auth is disabled) */
  function isAuthenticated(req: http.IncomingMessage): boolean {
    if (!AUTH_ENABLED) return true;

    // Check session cookie first
    const cookies = req.headers.cookie || "";
    const match = cookies.match(/prism_session=([a-f0-9]{64})/);
    if (match) {
      const token = match[1];
      const expiry = activeSessions.get(token);
      if (expiry && expiry > Date.now()) return true;
      // Expired — clean up
      if (expiry) activeSessions.delete(token);
    }

    // Check Basic Auth header
    const authHeader = req.headers.authorization || "";
    if (authHeader.startsWith("Basic ")) {
      const decoded = Buffer.from(authHeader.slice(6), "base64").toString("utf-8");
      const [user, pass] = decoded.split(":");
      return safeCompare(user || "", AUTH_USER) && safeCompare(pass || "", AUTH_PASS);
    }

    return false;
  }

  /** Render a styled login page matching the Mind Palace theme */
  function renderLoginPage(): string {
    return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Prism MCP — Login</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;color:#f1f5f9;font-family:'Inter',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
.bg{position:fixed;inset:0;background-image:radial-gradient(circle at 20% 30%,rgba(139,92,246,0.08) 0%,transparent 50%),radial-gradient(circle at 80% 70%,rgba(59,130,246,0.06) 0%,transparent 50%)}
.login-card{position:relative;z-index:1;background:rgba(17,24,39,0.6);backdrop-filter:blur(16px);border:1px solid rgba(139,92,246,0.15);border-radius:16px;padding:2.5rem;width:380px;max-width:90vw;text-align:center}
.logo{font-size:1.75rem;font-weight:700;background:linear-gradient(135deg,#8b5cf6,#3b82f6,#06b6d4);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem}
.subtitle{color:#64748b;font-size:0.85rem;margin-bottom:2rem}
.field{margin-bottom:1rem}
.field input{width:100%;padding:0.7rem 1rem;background:#111827;border:1px solid rgba(139,92,246,0.15);border-radius:10px;color:#f1f5f9;font-size:0.9rem;font-family:'Inter',sans-serif;outline:none;transition:border-color 0.2s}
.field input:focus{border-color:rgba(139,92,246,0.5)}
.field input::placeholder{color:#475569}
.login-btn{width:100%;padding:0.75rem;background:linear-gradient(135deg,#8b5cf6,#3b82f6);color:white;border:none;border-radius:10px;font-size:0.95rem;font-weight:600;cursor:pointer;transition:opacity 0.2s;margin-top:0.5rem}
.login-btn:hover{opacity:0.9}
.error{color:#f43f5e;font-size:0.8rem;margin-top:1rem;display:none}
.lock{font-size:2rem;margin-bottom:1rem}
</style></head><body>
<div class="bg"></div>
<div class="login-card">
<div class="lock">🔒</div>
<div class="logo">🧠 Prism Mind Palace</div>
<div class="subtitle">Authentication required for remote access</div>
<form id="loginForm" onsubmit="return handleLogin(event)">
<div class="field"><input type="text" id="user" placeholder="Username" autocomplete="username" required></div>
<div class="field"><input type="password" id="pass" placeholder="Password" autocomplete="current-password" required></div>
<button type="submit" class="login-btn">Sign In</button>
</form>
<div class="error" id="error">Invalid credentials</div>
</div>
<script>
async function handleLogin(e){e.preventDefault();
var u=document.getElementById('user').value,p=document.getElementById('pass').value;
var r=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user:u,pass:p})});
if(r.ok){window.location.reload();}else{document.getElementById('error').style.display='block';}
return false;}
</script></body></html>`;
  }

  if (AUTH_ENABLED) {
    console.error(`[Dashboard] 🔒 Auth enabled for user "${AUTH_USER}"`);
  }

  const httpServer = http.createServer(async (req, res) => {
    // CORS headers for local dev
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      return res.end();
    }

    // ─── v5.1: Auth login endpoint (always accessible) ───
    const reqUrl = new URL(req.url || "/", `http://${req.headers.host}`);
    if (AUTH_ENABLED && reqUrl.pathname === "/api/auth/login" && req.method === "POST") {
      const body = await readBody(req);
      try {
        const { user, pass } = JSON.parse(body);
        if (safeCompare(user || "", AUTH_USER) && safeCompare(pass || "", AUTH_PASS)) {
          const token = generateToken();
          activeSessions.set(token, Date.now() + SESSION_TTL_MS);
          res.writeHead(200, {
            "Content-Type": "application/json",
            "Set-Cookie": `prism_session=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${SESSION_TTL_MS / 1000}`,
          });
          return res.end(JSON.stringify({ ok: true }));
        }
      } catch { /* fall through to 401 */ }
      res.writeHead(401, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ error: "Invalid credentials" }));
    }

    // ─── v5.1: Auth gate — block unauthenticated requests ───
    if (AUTH_ENABLED && !isAuthenticated(req)) {
      // For API calls, return 401 JSON
      if (reqUrl.pathname.startsWith("/api/")) {
        res.writeHead(401, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ error: "Authentication required" }));
      }
      // For page requests, show login page
      res.writeHead(401, { "Content-Type": "text/html; charset=utf-8" });
      return res.end(renderLoginPage());
    }

    try {
      const url = new URL(req.url || "/", `http://${req.headers.host}`);

      // ─── Serve the Dashboard UI ───
      if (url.pathname === "/" || url.pathname === "/index.html") {
        res.writeHead(200, {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "no-store, no-cache, must-revalidate",
        });
        return res.end(renderDashboardHTML(SERVER_CONFIG.version));
      }

      // ─── API: List all projects ───
      if (url.pathname === "/api/projects") {
        const s = await getStorageSafe();
        if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
        const projects = await s.listProjects();
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ projects }));
      }

      // ─── API: Get full project data ───
      if (url.pathname === "/api/project") {
        const projectName = url.searchParams.get("name");
        if (!projectName) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Missing ?name= parameter" }));
        }

        const s = await getStorageSafe();
        if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
        const context = await s.loadContext(projectName, "deep", PRISM_USER_ID);
        const ledger = await s.getLedgerEntries({
          project: `eq.${projectName}`,
          order: "created_at.desc",
          limit: "20",
        });
        let history: unknown[] = [];
        try {
          history = await s.getHistory(projectName, PRISM_USER_ID, 10);
        } catch {
          // History may not exist for all projects
        }

        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ context, ledger, history }));
      }

      // ─── API: Brain Health Check (v2.2.0) ───
      if (url.pathname === "/api/health" && req.method === "GET") {
        try {
          const { runHealthCheck } = await import("../utils/healthCheck.js");
          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
          const stats = await s.getHealthStats(PRISM_USER_ID);
          const report = runHealthCheck(stats);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify(report));
        } catch (err) {
          console.error("[Dashboard] Health check error:", err);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({
            status: "unknown",
            summary: "Health check unavailable",
            issues: [],
            counts: { errors: 0, warnings: 0, infos: 0 },
            totals: { activeEntries: 0, handoffs: 0, rollups: 0 },
            timestamp: new Date().toISOString(),
          }));
        }
      }

      // ─── API: Brain Health Cleanup (v3.1) ───
      // Deletes orphaned handoffs (handoffs with no backing ledger entries).
      if (url.pathname === "/api/health/cleanup" && req.method === "POST") {
        try {
          const { runHealthCheck } = await import("../utils/healthCheck.js");
          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
          const stats = await s.getHealthStats(PRISM_USER_ID);
          const report = runHealthCheck(stats);

          // Collect orphaned handoff projects from the health issues
          const orphaned = stats.orphanedHandoffs || [];
          const cleaned: string[] = [];

          for (const { project } of orphaned) {
            try {
              await s.deleteHandoff(project, PRISM_USER_ID);
              cleaned.push(project);
              console.error(`[Dashboard] Cleaned up orphaned handoff: ${project}`);
            } catch (delErr) {
              console.error(`[Dashboard] Failed to delete handoff for ${project}:`, delErr);
            }
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({
            ok: true,
            cleaned,
            count: cleaned.length,
            message: cleaned.length > 0
              ? `Cleaned up ${cleaned.length} orphaned handoff(s): ${cleaned.join(", ")}`
              : "No orphaned handoffs to clean up.",
          }));
        } catch (err) {
          console.error("[Dashboard] Health cleanup error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: false, error: "Cleanup failed" }));
        }
      }

      // ─── API: Role-Scoped Skills (v3.1) ───

      // GET /api/skills → { skills: { dev: "...", qa: "..." } }
      if (url.pathname === "/api/skills" && req.method === "GET") {
        const all = await getAllSettings();
        const skills: Record<string, string> = {};
        for (const [k, v] of Object.entries(all)) {
          if (k.startsWith("skill:") && v) {
            skills[k.replace("skill:", "")] = v;
          }
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ skills }));
      }

      // POST /api/skills → { role, content } saves skill:<role>
      if (url.pathname === "/api/skills" && req.method === "POST") {
        const body = await new Promise<string>(resolve => {
          let data = ""; req.on("data", c => data += c); req.on("end", () => resolve(data));
        });
        const { role, content } = JSON.parse(body || "{}");
        if (!role) { res.writeHead(400); return res.end(JSON.stringify({ error: "role required" })); }
        await setSetting(`skill:${role}`, content || "");
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ ok: true, role }));
      }

      // DELETE /api/skills/:role → clears skill:<role>
      if (url.pathname.startsWith("/api/skills/") && req.method === "DELETE") {
        const role = url.pathname.replace("/api/skills/", "");
        await setSetting(`skill:${role}`, "");
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ ok: true, role }));
      }

      // ─── API: Knowledge Graph Data (v2.3.0 / v5.1) ───
      if (url.pathname === "/api/graph" && req.method === "GET") {
        const project = url.searchParams.get("project") || undefined;
        const days = url.searchParams.get("days") || undefined;
        const min_importance = url.searchParams.get("min_importance") || undefined;

        // Fetch recent ledger entries to build the graph
        // We look at the last 100 entries to keep the graph relevant but performant
        const s = await getStorageSafe();
        if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }

        const params: any = {
          order: "created_at.desc",
          select: "project,keywords,created_at,importance",
        };

        if (!project && !days && !min_importance) {
          params.limit = "30";  // Keep default small to prevent Vis.js stack overflow (426 nodes @ 100 entries)
        } else {
          params.limit = "200"; // Bump limit when exploring specific filters (capped by frontend maxNodes)
        }

        if (project) {
          params.project = `eq.${project}`;
        }
        if (days) {
          const past = new Date();
          past.setDate(past.getDate() - parseInt(days, 10));
          params.created_at = `gte.${past.toISOString()}`;
        }
        if (min_importance) {
          params.importance = `gte.${parseInt(min_importance, 10)}`;
        }

        const entries = await s.getLedgerEntries(params);

        // Deduplication sets for nodes and edges
        const nodes: { id: string; label: string; group: string }[] = [];
        const edges: { from: string; to: string }[] = [];
        const nodeIds = new Set<string>();   // track unique node IDs
        const edgeIds = new Set<string>();   // track unique edges

        // Helper: add a node only if it doesn't already exist
        const addNode = (id: string, group: string, label?: string) => {
          if (!nodeIds.has(id)) {
            nodes.push({ id, label: label || id, group });
            nodeIds.add(id);
          }
        };

        // Helper: add an edge only if it doesn't already exist
        const addEdge = (from: string, to: string) => {
          const id = `${from}-${to}`;  // deterministic edge ID
          if (!edgeIds.has(id)) {
            edges.push({ from, to });
            edgeIds.add(id);
          }
        };

        // Transform relational data into graph nodes & edges
        (entries as any[]).forEach(row => {
          if (!row.project) return;  // skip rows without project

          // 1. Project node (hub — large purple dot)
          addNode(row.project, "project");

          // 2. Keyword nodes (spokes — small dots)
          let keywords: string[] = [];

          // Handle SQLite (JSON string) vs Supabase (native array)
          if (Array.isArray(row.keywords)) {
            keywords = row.keywords;
          } else if (typeof row.keywords === "string") {
            try { keywords = JSON.parse(row.keywords); } catch { /* skip malformed */ }
          }

          // Create nodes + edges for each keyword
          keywords.forEach((kw: string) => {
            if (kw.length < 3) return;  // skip noise like "a", "is"

            // Handle categories (cat:debugging) vs raw keywords
            const isCat = kw.startsWith("cat:");
            const group = isCat ? "category" : "keyword";
            const label = isCat ? kw.replace("cat:", "") : kw;

            addNode(kw, group, label);  // keyword/category node
            addEdge(row.project, kw);   // edge: project → keyword
          });
        });

        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ nodes, edges }));
      }

      // ─── API: Edit Knowledge Graph Node (v5.1) ───
      // Surgically patches keywords in the session_ledger.
      // Supports two operations:
      //   1. RENAME: old keyword → new keyword across all entries
      //   2. DELETE: remove a keyword from all entries (newId = null)
      //
      // HOW IT WORKS:
      //   - Reconstructs the full PostgREST-style keyword (e.g. cat:debugging)
      //   - Uses LIKE-based search to find candidate entries
      //   - Validates exact array membership in JS (prevents substring matches)
      //   - Idempotently strips or replaces the keyword via patchLedger()
      //
      // SECURITY: Protected by the v5.1 Dashboard Auth gate above.
      if (url.pathname === "/api/graph/node" && req.method === "POST") {
        try {
          const body = await readBody(req);
          const { oldId, newId, group } = JSON.parse(body || "{}");

          if (!oldId || !group || (group !== "keyword" && group !== "category")) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "Invalid request" }));
          }

          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage not ready" })); }

          // 1. Reconstruct the full string as stored in DB
          //    Categories are prefixed with "cat:" (e.g. cat:debugging)
          //    Keywords are stored as bare strings (e.g. authentication)
          const searchKw = group === "category" ? `cat:${oldId}` : oldId;
          const newKw = newId ? (group === "category" ? `cat:${newId}` : newId) : null;

          // 2. Fetch all entries containing the old keyword (LIKE search)
          //    Note: LIKE '%auth%' would also match 'authentication',
          //    so we verify exact array membership in the JS loop below.
          const entries = await s.getLedgerEntries({
            keywords: `cs.{${searchKw}}`,
            select: "id,keywords",
          }) as Array<{ id: string; keywords: unknown }>;

          let updated = 0;
          for (const entry of entries) {
            // Parse keywords — handle both SQLite (JSON string) and Supabase (array)
            let kws: string[] = [];
            if (Array.isArray(entry.keywords)) kws = entry.keywords as string[];
            else if (typeof entry.keywords === "string") {
              try { kws = JSON.parse(entry.keywords); } catch { continue; }
            }

            // Exact match check — guards against substring false positives
            if (!kws.includes(searchKw)) continue;

            // Remove the old keyword
            const newKws = kws.filter(k => k !== searchKw);

            // If renaming (not deleting), add the new keyword (no duplicates)
            if (newKw && !newKws.includes(newKw)) {
              newKws.push(newKw);
            }

            // 3. Patch the entry — patchLedger handles JSON serialization
            await s.patchLedger(entry.id, { keywords: newKws });
            updated++;
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: true, updated }));
        } catch (err) {
          console.error("[Dashboard] Node edit error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Edit failed" }));
        }
      }

      // ─── API: Hivemind Team Roster (v3.0) ───
      if (url.pathname === "/api/team") {
        const projectName = url.searchParams.get("project");
        if (!projectName) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Missing ?project= parameter" }));
        }
        try {
          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
          const team = await s.listTeam(projectName, PRISM_USER_ID);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ team }));
        } catch {
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ team: [] }));
        }
      }

      // ─── API: Settings — GET (v3.0 Dashboard Settings) ───
      if (url.pathname === "/api/settings" && req.method === "GET") {
        try {
          const { getAllSettings } = await import("../storage/configStorage.js");
          const settings = await getAllSettings();
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ settings }));
        } catch {
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ settings: {} }));
        }
      }

      // ─── API: Settings — POST (v3.0 Dashboard Settings) ───
      if (url.pathname === "/api/settings" && req.method === "POST") {
        try {
          const body = await readBody(req);
          const parsed = JSON.parse(body);
          if (parsed.key && parsed.value !== undefined) {
            const { setSetting } = await import("../storage/configStorage.js");
            await setSetting(parsed.key, String(parsed.value));
            res.writeHead(200, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ ok: true, key: parsed.key, value: parsed.value }));
          }
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Missing key or value" }));
        } catch (err) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Invalid JSON body" }));
        }

      }

      // ─── API: Memory Analytics (v3.1) ────────────────────
      if (url.pathname === "/api/analytics" && req.method === "GET") {
        const projectName = url.searchParams.get("project");
        if (!projectName) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Missing ?project= parameter" }));
        }
        try {
          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
          const analytics = await s.getAnalytics(projectName, PRISM_USER_ID);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify(analytics));
        } catch (err) {
          console.error("[Dashboard] Analytics error:", err);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({
            totalEntries: 0, totalRollups: 0, rollupSavings: 0,
            avgSummaryLength: 0, sessionsByDay: [],
          }));
        }
      }

      // ─── API: Retention (TTL) Settings (v3.1) ──────────────
      // GET /api/retention?project= → current TTL setting
      // POST /api/retention → { project, ttl_days } → saves + runs sweep
      if (url.pathname === "/api/retention") {
        if (req.method === "GET") {
          const projectName = url.searchParams.get("project");
          if (!projectName) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "Missing ?project= parameter" }));
          }
          const ttlRaw = await getSetting(`ttl:${projectName}`, "0");
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ project: projectName, ttl_days: parseInt(ttlRaw, 10) || 0 }));
        }

        if (req.method === "POST") {
          const body = await readBody(req);
          const { project, ttl_days } = JSON.parse(body || "{}");
          if (!project || ttl_days === undefined) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "project and ttl_days required" }));
          }
          if (ttl_days > 0 && ttl_days < 7) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "Minimum TTL is 7 days" }));
          }
          await setSetting(`ttl:${project}`, String(ttl_days));
          let expired = 0;
          if (ttl_days > 0) {
            const s = await getStorageSafe();
            if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
            const result = await s.expireByTTL(project, ttl_days, PRISM_USER_ID);
            expired = result.expired;
          }
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: true, project, ttl_days, expired }));
        }
      }

      // ─── API: Compact Now (v3.1 — Dashboard button) ──────────
      if (url.pathname === "/api/compact" && req.method === "POST") {
        const body = await readBody(req);
        const { project } = JSON.parse(body || "{}");
        if (!project) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "project required" }));
        }
        try {
          const result = await compactLedgerHandler({ project });
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: true, result }));
        } catch (err) {
          console.error("[Dashboard] Compact error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: false, error: "Compaction failed" }));
        }
      }

      // ─── API: PKM Export — Obsidian/Logseq ZIP (v3.1) ──────
      if (url.pathname === "/api/export" && req.method === "GET") {
        const projectName = url.searchParams.get("project");
        if (!projectName) {
          res.writeHead(400, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Missing ?project= parameter" }));
        }
        try {
          // Lazy-import fflate to keep startup fast
          const { strToU8, zipSync } = await import("fflate");

          // Fetch all active ledger entries for this project
          const s = await getStorageSafe();
          if (!s) { res.writeHead(503, { "Content-Type": "application/json" }); return res.end(JSON.stringify({ error: "Storage initializing..." })); }
          const entries = await s.getLedgerEntries({
            project: `eq.${projectName}`,
            order: "created_at.asc",
            limit: "1000",
          }) as Array<Record<string, unknown>>;

          const files: Record<string, Uint8Array> = {};

          // One MD file per session
          for (const entry of entries) {
            const date = (entry.created_at as string | undefined)?.slice(0, 10) ?? "unknown";
            const id = (entry.id as string | undefined)?.slice(0, 8) ?? "xxxxxxxx";
            const filename = `${projectName}/${date}-${id}.md`;

            const todos = Array.isArray(entry.todos) ? (entry.todos as string[]) : [];
            const decisions = Array.isArray(entry.decisions) ? (entry.decisions as string[]) : [];
            const files_changed = Array.isArray(entry.files_changed) ? (entry.files_changed as string[]) : [];
            const tags = ((Array.isArray(entry.keywords) ? entry.keywords : []) as string[]).slice(0, 10);

            const content = [
              `# Session: ${date}`,
              ``,
              `**Project:** ${projectName}`,
              `**Date:** ${date}`,
              `**Role:** ${(entry.role as string) || "global"}`,
              tags.length ? `**Tags:** ${tags.map(t => `#${t.replace(/\s+/g, "_")}`).join(" ")}` : "",
              ``,
              `## Summary`,
              ``,
              entry.summary as string,
              ``,
              todos.length ? `## TODOs\n\n${todos.map(t => `- [ ] ${t}`).join("\n")}` : "",
              decisions.length ? `## Decisions\n\n${decisions.map(d => `- ${d}`).join("\n")}` : "",
              files_changed.length ? `## Files Changed\n\n${files_changed.map(f => `- \`${f}\``).join("\n")}` : "",
            ].filter(Boolean).join("\n");

            files[filename] = strToU8(content);
          }

          // Index file linking all sessions
          const indexLines = [
            `# ${projectName} — Session Index`,
            ``,
            `> Exported from Prism MCP on ${new Date().toISOString().slice(0, 10)}`,
            ``,
            ...entries.map(e => {
              const d = (e.created_at as string | undefined)?.slice(0, 10) ?? "unknown";
              const i = (e.id as string | undefined)?.slice(0, 8) ?? "xxxxxxxx";
              return `- [[${projectName}/${d}-${i}]]`;
            }),
          ];
          files[`${projectName}/_index.md`] = strToU8(indexLines.join("\n"));

          const zipped = zipSync(files, { level: 6 });

          res.writeHead(200, {
            "Content-Type": "application/zip",
            "Content-Disposition": `attachment; filename="prism-export-${projectName}-${Date.now()}.zip"`,
            "Content-Length": String(zipped.byteLength),
          });
          return res.end(Buffer.from(zipped));
        } catch (err) {
          console.error("[Dashboard] PKM export error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "Export failed" }));
        }
      }

      // ─── API: Universal History Import (v5.2) ───
      if (url.pathname === "/api/import" && req.method === "POST") {
        try {
          const body = await new Promise<string>(resolve => {
            let data = ""; req.on("data", c => data += c); req.on("end", () => resolve(data));
          });
          const { path: filePath, format, project, dryRun } = JSON.parse(body || "{}");
          if (!filePath) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "path is required" }));
          }

          // Verify file exists before starting import
          if (!fs.existsSync(filePath)) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: `File not found: ${filePath}` }));
          }

          const { universalImporter } = await import("../utils/universalImporter.js");
          const result = await universalImporter({
            path: filePath,
            format: format || undefined,
            project: project || undefined,
            dryRun: !!dryRun,
            verbose: false,
          });

          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({
            ok: true,
            ...result,
            message: `Imported ${result.conversationCount} conversations (${result.successCount} turns)${result.skipCount > 0 ? `, ${result.skipCount} skipped (dup)` : ""}${result.failCount > 0 ? `, ${result.failCount} failed` : ""}${dryRun ? " [DRY RUN]" : ""}`,
          }));
        } catch (err: any) {
          console.error("[Dashboard] Import error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: err.message || "Import failed" }));
        }
      }

      // ─── API: Universal History Import via File Upload (v5.2) ───
      if (url.pathname === "/api/import-upload" && req.method === "POST") {
        try {
          const body = await new Promise<string>(resolve => {
            let data = ""; req.on("data", c => data += c); req.on("end", () => resolve(data));
          });
          const { filename, content, format, project, dryRun } = JSON.parse(body || "{}");
          if (!content || !filename) {
            res.writeHead(400, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ error: "filename and content are required" }));
          }

          // Write uploaded content to a temp file
          const tmpDir = path.join(os.tmpdir(), "prism-import");
          fs.mkdirSync(tmpDir, { recursive: true });
          const tmpFile = path.join(tmpDir, `upload-${Date.now()}-${filename}`);
          fs.writeFileSync(tmpFile, content, "utf-8");

          try {
            const { universalImporter } = await import("../utils/universalImporter.js");
            const result = await universalImporter({
              path: tmpFile,
              format: format || undefined,
              project: project || undefined,
              dryRun: !!dryRun,
              verbose: false,
            });

            res.writeHead(200, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({
              ok: true,
              ...result,
              message: `Imported ${result.conversationCount} conversations (${result.successCount} turns)${result.skipCount > 0 ? `, ${result.skipCount} skipped (dup)` : ""}${result.failCount > 0 ? `, ${result.failCount} failed` : ""}${dryRun ? " [DRY RUN]" : ""} from ${filename}`,
            }));
          } finally {
            // Clean up temp file
            try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
          }
        } catch (err: any) {
          console.error("[Dashboard] Import upload error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: err.message || "Import failed" }));
        }
      }

      // ─── API: Background Scheduler Status (v5.4) ────────────
      if (url.pathname === "/api/scheduler" && req.method === "GET") {
        try {
          const { getSchedulerStatus } = await import("../backgroundScheduler.js");
          const status = getSchedulerStatus();
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify(status));
        } catch (err) {
          console.error("[Dashboard] Scheduler status error:", err);
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({
            running: false, startedAt: null, intervalMs: 0, lastSweep: null,
          }));
        }
      }

      // ─── API: Autonomous Web Scholar Trigger (v5.4) ─────────
      if (url.pathname === "/api/scholar/trigger" && req.method === "POST") {
        try {
          const { runWebScholar } = await import("../scholar/webScholar.js");
          
          // Fire and forget, don't block the request
          runWebScholar().catch(err => {
            console.error("[Dashboard] Web Scholar async trigger failed:", err);
          });
          
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ ok: true, message: "Autonomous research started in background" }));
        } catch (err: any) {
          console.error("[Dashboard] Web Scholar trigger error:", err);
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: err.message || "Failed to trigger Web Scholar" }));
        }
      }

      // ─── PWA: Manifest (v5.4) ───
      if (url.pathname === "/manifest.json" && req.method === "GET") {
        const manifest = {
          name: "Prism Mind Palace",
          short_name: "Prism",
          description: "Prism MCP Mobile Dashboard",
          start_url: "/",
          display: "standalone",
          background_color: "#0a0e1a",
          theme_color: "#0a0e1a",
          icons: [
            { src: "/icon-192.svg", sizes: "192x192", type: "image/svg+xml" },
            { src: "/icon-512.svg", sizes: "512x512", type: "image/svg+xml" }
          ]
        };
        res.writeHead(200, {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=86400"
        });
        return res.end(JSON.stringify(manifest));
      }

      // ─── PWA: Service Worker (v5.4) ───
      if (url.pathname === "/sw.js" && req.method === "GET") {
        const swContent = `
const CACHE_NAME = 'prism-pwa-v1';
const ASSETS = [
  '/',
  '/manifest.json',
  '/icon-192.svg',
  '/icon-512.svg'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => {
    return Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)));
  }));
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  // Network-first for API requests, Cache-first for Assets
  if (e.request.url.includes('/api/')) {
    e.respondWith(fetch(e.request).catch(() => new Response(JSON.stringify({ error: "Offline" }), { headers: { "Content-Type": "application/json" }, status: 503 })));
  } else {
    e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request).then((fres) => {
      // Cache dynamically fetched non-API assets
      return caches.open(CACHE_NAME).then(c => { c.put(e.request, fres.clone()); return fres; });
    })));
  }
});
        `.trim();
        res.writeHead(200, {
          "Content-Type": "application/javascript",
          "Cache-Control": "no-cache"
        });
        return res.end(swContent);
      }

      // ─── PWA: Dynamic SVG Icons (v5.4) ───
      if ((url.pathname === "/icon-192.svg" || url.pathname === "/icon-512.svg") && req.method === "GET") {
        const size = url.pathname === "/icon-192.svg" ? 192 : 512;
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8b5cf6" />
      <stop offset="50%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#06b6d4" />
    </linearGradient>
  </defs>
  <rect width="${size}" height="${size}" rx="${Math.floor(size * 0.2)}" fill="#0a0e1a"/>
  <path d="M${size * 0.5} ${size * 0.25} L${size * 0.75} ${size * 0.75} L${size * 0.25} ${size * 0.75} Z" fill="url(#grad)" opacity="0.9"/>
  <circle cx="${size * 0.5}" cy="${size * 0.55}" r="${size * 0.15}" fill="#ffffff" opacity="0.1" />
</svg>`;
        res.writeHead(200, {
          "Content-Type": "image/svg+xml",
          "Cache-Control": "public, max-age=86400"
        });
        return res.end(svg);
      }

      // ─── 404 ───
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");


    } catch (error) {
      console.error("[Dashboard] Error handling request:", error);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Internal Server Error" }));
    }
  });

  // ─── Resilient port binding with retry ───
  // Wraps listen() in a Promise to detect EADDRINUSE failures and retry
  // with a delay (gives OS time to release the port after killPortHolder).
  // Falls back to PORT+1, PORT+2 if the preferred port is permanently taken.
  const MAX_RETRIES = 3;
  const RETRY_DELAY_MS = 500;

  const tryListen = (port: number): Promise<number> =>
    new Promise((resolve, reject) => {
      const onError = (err: NodeJS.ErrnoException) => {
        httpServer.removeListener("error", onError);
        reject(err);
      };
      httpServer.on("error", onError);
      httpServer.listen(port, () => {
        httpServer.removeListener("error", onError);
        // Re-register a permanent error handler for runtime errors
        httpServer.on("error", (err: NodeJS.ErrnoException) => {
          console.error(`[Dashboard] HTTP server error: ${err.message}`);
        });
        resolve(port);
      });
    });

  let boundPort = PORT;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      boundPort = await tryListen(PORT + attempt);
      break; // Success
    } catch (err: any) {
      if (err.code === "EADDRINUSE") {
        console.error(
          `[Dashboard] Port ${PORT + attempt} is in use (attempt ${attempt + 1}/${MAX_RETRIES}).`
        );
        if (attempt < MAX_RETRIES - 1) {
          // Wait for OS to release the port, then try next port
          await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
        } else {
          console.error(
            `[Dashboard] All ports ${PORT}–${PORT + MAX_RETRIES - 1} in use — Mind Palace disabled. ` +
            `Set PRISM_DASHBOARD_PORT to use a different port.`
          );
          return; // Give up — MCP server keeps running
        }
      } else {
        console.error(`[Dashboard] HTTP server error: ${err.message}`);
        return; // Non-retryable error
      }
    }
  }

  // Write the active port to a file for discoverability
  try {
    const portFile = path.join(os.homedir(), ".prism-mcp", "dashboard.port");
    fs.writeFileSync(portFile, String(boundPort), "utf8");
  } catch {
    // Non-fatal — just means the user has to know the port
  }

  console.error(`[Prism] 🧠 Mind Palace Dashboard → http://localhost:${boundPort}`);

  // ─── v3.1: TTL Sweep — runs at startup + every 12 hours ───────────
  // NOTE (v5.4): The Background Scheduler in server.ts now also handles
  // TTL sweeps. This dashboard sweep is kept as a legacy fallback for
  // deployments where the scheduler is disabled. Both are idempotent.
  async function runTtlSweep() {
    try {
      const allSettings = await getAllSettings();
      for (const [key, val] of Object.entries(allSettings)) {
        if (!key.startsWith("ttl:")) continue;
        const project = key.replace("ttl:", "");
        const ttlDays = parseInt(val, 10);
        if (!ttlDays || ttlDays <= 0) continue;
        const s = await getStorageSafe();
        if (!s) continue;
        const result = await s.expireByTTL(project, ttlDays, PRISM_USER_ID);
        if (result.expired > 0) {
          console.error(`[Dashboard] TTL sweep: expired ${result.expired} entries for "${project}" (ttl=${ttlDays}d)`);
        }
      }
    } catch (err) {
      console.error("[Dashboard] TTL sweep error (non-fatal):", err);
    }
  }

  // Run immediately on startup, then every 12 hours
  runTtlSweep().catch(() => {});
  setInterval(() => { runTtlSweep().catch(() => {}); }, 12 * 60 * 60 * 1000);
}
</file>

</files>
