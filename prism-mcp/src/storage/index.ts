
import { debugLog } from "../utils/logger.js";
import { SupabaseStorage } from "./supabase.js";
import type { StorageBackend } from "./interface.js";
import { getSetting } from "./configStorage.js";

let storageInstance: StorageBackend | null = null;
export let activeStorageBackend: string = "local";

/** Validate that a string is an http(s) URL (mirrors logic in config.ts). */
function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Returns the singleton storage backend.
 *
 * On first call: creates and initializes the appropriate backend.
 * On subsequent calls: returns the cached instance.
 *
 * SUPABASE CREDENTIAL RESOLUTION ORDER:
 *   Single source of truth: prism-config.db (set via Mind Palace dashboard).
 *   Env vars are bootstrapped into the DB on first run by configStorage.ts.
 */
export async function getStorage(): Promise<StorageBackend> {
  if (storageInstance) return storageInstance;

  // Single source of truth: prism-config.db (dashboard).
  const dbStorage = await getSetting("prism_storage", "");
  const requestedBackend = (dbStorage || "local") as "supabase" | "local";

  if (requestedBackend === "supabase") {
    // ─── Resolve credentials: configStorage → env var fallback ──────────
    // v9.2: DB (dashboard) is the source of truth for Supabase credentials,
    // consistent with PRISM_STORAGE resolution above. If the user configured
    // Supabase via the dashboard, the values live in configStorage. Env vars
    // are only used as a fallback for users who haven't migrated yet.
    const resolvedUrl =
      await getSetting("supabase_url", "") ||
      await getSetting("SUPABASE_URL", "") ||  // legacy key compat
      "";
    const resolvedKey =
      await getSetting("supabase_key", "") ||
      await getSetting("SUPABASE_KEY", "") ||  // legacy key compat
      await getSetting("SUPABASE_SERVICE_ROLE_KEY", "") ||
      "";

    const isConfigured = !!resolvedUrl && !!resolvedKey && isHttpUrl(resolvedUrl);

    if (!isConfigured) {
      activeStorageBackend = "local";
      console.error(
        "[Prism Storage] Supabase backend requested but credentials are missing or invalid " +
        "(prism-config.db). Falling back to local storage.\n" +
        "  → Configure via Mind Palace dashboard (Settings → Storage Backend → Supabase)."
      );
    } else {
      activeStorageBackend = "supabase";
      debugLog(`[Prism Storage] Supabase credentials resolved from configStorage`);
    }
  } else {
    activeStorageBackend = requestedBackend;
  }

  debugLog(`[Prism Storage] Initializing backend: ${activeStorageBackend}`);

  if (activeStorageBackend === "local") {
    const { SqliteStorage } = await import("./sqlite.js");
    storageInstance = new SqliteStorage();
  } else if (activeStorageBackend === "supabase") {
    storageInstance = new SupabaseStorage();
  } else {
    throw new Error(
      `Unknown PRISM_STORAGE value: "${activeStorageBackend}". ` +
      `Must be "local" or "supabase".`
    );
  }

  await storageInstance.initialize();
  return storageInstance;
}


/**
 * Closes the active storage backend and resets the singleton.
 * Used for testing and graceful shutdown.
 */
export async function closeStorage(): Promise<void> {
  if (storageInstance) {
    await storageInstance.close();
    storageInstance = null;
  }
}

// Re-export the interface types for convenience
export type { StorageBackend } from "./interface.js";
export type {
  LedgerEntry,
  HandoffEntry,
  SaveHandoffResult,
  ContextResult,
  KnowledgeSearchResult,
  SemanticSearchResult,
  PipelineState,
  PipelineStatus,
} from "./interface.js";
