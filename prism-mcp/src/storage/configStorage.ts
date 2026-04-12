import { createClient } from "@libsql/client";
import { resolve, dirname } from "path";
import { homedir } from "os";
import { existsSync, mkdirSync } from "fs";

// We use a small, dedicated DB just for configuration settings.
// This solves the chicken-and-egg problem: we need to know WHICH
// storage backend to boot *before* we can use that backend.
//
// Stored in ~/.prism-mcp/prism-config.db — the same root directory
// used by sqlite.ts and autoCapture.ts for all Prism files.
//
// ⚡ BOOT SETTINGS NOTE:
//   Settings in this store that affect server initialization (e.g.
//   PRISM_STORAGE, PRISM_ENABLE_HIVEMIND) are read only at startup.
//   Changing them at runtime requires a server restart to take effect.
//   Runtime-only settings (e.g. dashboard_theme) take effect immediately.
const CONFIG_PATH = resolve(homedir(), ".prism-mcp", "prism-config.db");

let configClient: ReturnType<typeof createClient> | null = null;
let initialized = false;

// ─── In-memory settings cache ──────────────────────────────────────
// Preloaded during initConfigStorage() so that hot-path MCP handlers
// (e.g. ReadResourceRequestSchema) can read settings synchronously
// without opening an additional SQLite round-trip and stalling the
// MCP stdio handshake (which causes a black-screen on startup).
let settingsCache: Record<string, string> | null = null;

function getClient() {
  if (!configClient) {
    // Ensure the directory exists before opening the DB.
    // In Docker/CI (e.g. Glama), ~/.prism-mcp/ doesn't exist yet,
    // and libSQL throws SQLITE_CANTOPEN (error 14) without it.
    const dir = dirname(CONFIG_PATH);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    configClient = createClient({
      url: `file:${CONFIG_PATH}`,
    });
  }
  return configClient;
}

// ── Env → setting key mapping ─────────────────────────────────────
// Used during first-run bootstrap: if a setting has no value in the DB
// and the corresponding env var is set, we write it INTO the DB so the
// dashboard becomes the single source of truth from that point forward.
const ENV_TO_SETTING: Record<string, string> = {
  // Provider selection
  TEXT_PROVIDER:              "text_provider",
  EMBEDDING_PROVIDER:         "embedding_provider",

  // Llama.cpp
  LLAMACPP_TEXT_URL:          "llamacpp_text_url",
  LLAMACPP_TEXT_MODEL:        "llamacpp_text_model",
  LLAMACPP_EMBEDDING_URL:     "llamacpp_embedding_url",
  LLAMACPP_EMBEDDING_MODEL:   "llamacpp_embedding_model",

  // OpenAI / Ollama
  OPENAI_API_KEY:             "openai_api_key",
  OPENAI_BASE_URL:            "openai_base_url",
  OPENAI_MODEL:               "openai_model",
  OPENAI_EMBEDDING_MODEL:     "openai_embedding_model",

  // Google / Gemini
  GOOGLE_API_KEY:             "google_api_key",

  // Anthropic
  ANTHROPIC_API_KEY:          "anthropic_api_key",
  ANTHROPIC_MODEL:            "anthropic_model",

  // Voyage
  VOYAGE_API_KEY:             "voyage_api_key",

  // Brave Search
  BRAVE_API_KEY:              "brave_api_key",
  BRAVE_ANSWERS_API_KEY:      "brave_answers_api_key",

  // Firecrawl / Tavily (Web Scholar)
  FIRECRAWL_API_KEY:          "firecrawl_api_key",
  TAVILY_API_KEY:             "tavily_api_key",

  // Storage
  PRISM_STORAGE:              "prism_storage",
  SUPABASE_URL:               "supabase_url",
  SUPABASE_KEY:               "supabase_key",

  // Dashboard
  PRISM_DASHBOARD_PORT:       "dashboard_port",

  // User identity
  PRISM_USER_ID:              "user_id",
  PRISM_INSTANCE:             "instance_name",

  // Feature flags
  PRISM_SCHOLAR_ENABLED:      "scholar_enabled",
  PRISM_SCHOLAR_INTERVAL_MS:  "scholar_interval_ms",
  PRISM_SCHOLAR_TOPICS:       "scholar_topics",
  PRISM_SCHOLAR_MAX_ARTICLES_PER_RUN: "scholar_max_articles",
  PRISM_AUTO_CAPTURE:         "auto_capture_enabled",
  PRISM_DARK_FACTORY_ENABLED: "dark_factory_enabled",
  PRISM_DEBUG_LOGGING:        "debug_logging",
  PRISM_SCHEDULER_ENABLED:    "scheduler_enabled",
  PRISM_SCHEDULER_INTERVAL_MS: "scheduler_interval_ms",
  PRISM_HDC_ENABLED:          "hdc_enabled",
  PRISM_GRAPH_PRUNING_ENABLED: "graph_pruning_enabled",
  PRISM_ACTR_ENABLED:         "actr_enabled",
  PRISM_SYNAPSE_ENABLED:      "synapse_enabled",
  PRISM_VALENCE_ENABLED:      "valence_enabled",
  PRISM_COGNITIVE_BUDGET_ENABLED: "cognitive_budget_enabled",
  PRISM_SURPRISAL_GATE_ENABLED: "surprisal_gate_enabled",
  PRISM_VERIFICATION_HARNESS_ENABLED: "verification_enabled",
  PRISM_TASK_ROUTER_ENABLED_ENV: "task_router_enabled",
};

export async function initConfigStorage() {
  if (initialized) return;

  try {
    const client = getClient();
    await client.execute(`
      CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Preload all rows into the cache so subsequent reads are zero-cost.
    const rs = await client.execute("SELECT key, value FROM system_settings");
    settingsCache = {};
    for (const row of rs.rows) {
      settingsCache[row.key as string] = row.value as string;
    }

    // ── First-run bootstrap ───────────────────────────────────────────
    // On first run (no _prism_bootstrapped flag in DB), seed the DB from
    // process.env so that MCP harness "env" blocks and .env files get
    // written into the DB as initial values. After bootstrap, env vars
    // are ignored — the dashboard is the single source of truth.
    if (!settingsCache["_prism_bootstrapped"]) {
      let seeded = 0;
      for (const [envKey, settingKey] of Object.entries(ENV_TO_SETTING)) {
        const val = process.env[envKey];
        if (val !== undefined && val !== "" && !settingsCache[settingKey]) {
          await client.execute({
            sql: `INSERT INTO system_settings (key, value, updated_at)
                  VALUES (?, ?, CURRENT_TIMESTAMP)
                  ON CONFLICT(key) DO NOTHING`,
            args: [settingKey, val],
          });
          settingsCache[settingKey] = val;
          seeded++;
        }
      }
      // Mark bootstrap complete so we never re-seed from env.
      await client.execute({
        sql: `INSERT INTO system_settings (key, value, updated_at)
              VALUES ('_prism_bootstrapped', 'true', CURRENT_TIMESTAMP)
              ON CONFLICT(key) DO NOTHING`,
        args: [],
      });
      settingsCache["_prism_bootstrapped"] = "true";
      if (seeded > 0) {
        console.error(`[configStorage] Bootstrap: seeded ${seeded} settings from env vars`);
      }
    }
  } catch (err) {
    // Graceful degradation: if the DB can't be opened (e.g. read-only
    // filesystem in a sandboxed container), fall back to an empty cache.
    // getSettingSync() will return defaults; getSetting()/setSetting()
    // will attempt to re-open the DB on first call.
    console.error(`[configStorage] Failed to initialize (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
    settingsCache = {};
  }

  initialized = true;
}

/**
 * Synchronous setting read — served from the in-memory cache.
 * Returns defaultValue if the cache hasn't been populated yet (e.g. very
 * early startup before initConfigStorage() has been called) or if the key
 * doesn't exist. Safe to call from any MCP request handler without triggering
 * a SQLite round-trip.
 */
export function getSettingSync(key: string, defaultValue = ""): string {
  if (!settingsCache) return defaultValue;
  return settingsCache[key] ?? defaultValue;
}

export async function getSetting(key: string, defaultValue = ""): Promise<string> {
  await initConfigStorage();
  // Serve from cache when warm (the common case after startup).
  if (settingsCache && key in settingsCache) {
    return settingsCache[key];
  }
  const client = getClient();
  const rs = await client.execute({
    sql: "SELECT value FROM system_settings WHERE key = ?",
    args: [key],
  });

  if (rs.rows.length > 0) {
    const value = rs.rows[0].value as string;
    // Populate cache entry for future reads.
    if (settingsCache) settingsCache[key] = value;
    return value;
  }
  return defaultValue;
}

export async function setSetting(key: string, value: string): Promise<void> {
  await initConfigStorage();
  const client = getClient();

  // Retry with exponential backoff for SQLITE_BUSY (concurrent writes).
  // The dashboard and load tests can fire many parallel setting saves.
  const MAX_RETRIES = 5;
  const BASE_DELAY_MS = 20;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      await client.execute({
        sql: `
          INSERT INTO system_settings (key, value, updated_at) 
          VALUES (?, ?, CURRENT_TIMESTAMP)
          ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        `,
        args: [key, value],
      });
      // Keep the cache in sync so getSettingSync() reflects the new value immediately.
      if (settingsCache) {
        settingsCache[key] = value;
      }
      return; // Success — exit
    } catch (err: any) {
      const isBusy = err?.code === "SQLITE_BUSY" || err?.rawCode === 5;
      if (isBusy && attempt < MAX_RETRIES) {
        // Exponential backoff + jitter
        const delay = BASE_DELAY_MS * Math.pow(2, attempt) + Math.random() * 10;
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      throw err; // Not SQLITE_BUSY or retries exhausted
    }
  }
}

export async function getAllSettings(): Promise<Record<string, string>> {
  await initConfigStorage();
  // Return a snapshot of the cache (avoids a redundant DB round-trip).
  if (settingsCache) {
    return { ...settingsCache };
  }
  const client = getClient();
  const rs = await client.execute("SELECT key, value FROM system_settings");

  const settings: Record<string, string> = {};
  for (const row of rs.rows) {
    settings[row.key as string] = row.value as string;
  }
  return settings;
}

/**
 * Closes the config SQLite client to release the file handle on prism-config.db.
 * Called by the lifecycle module during graceful shutdown.
 */
export function closeConfigStorage() {
  if (configClient) {
    try {
      configClient.close();
    } catch (e) {
      console.error(`[ConfigStorage] Error closing db:`, e);
    }
  }
}
