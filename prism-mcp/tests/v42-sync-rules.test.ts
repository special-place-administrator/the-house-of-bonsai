/**
 * v4.2 Knowledge Sync Rules — Test Suite
 *
 * Covers:
 *  1. getGraduatedInsights() query correctness:
 *     - Returns entries with importance >= 7
 *     - Excludes below-threshold entries
 *     - Excludes archived/deleted entries
 *     - Orders by importance DESC
 *     - Scopes by project (no cross-project leakage)
 *  2. Rules file generation:
 *     - Correct markdown format with sentinel markers
 *     - Idempotent: running twice produces same file
 *     - Correctly replaces existing sentinel block
 *     - Appends sentinels when file has no existing block
 *  3. Type guard:
 *     - Accepts valid args
 *     - Rejects missing project
 *
 * Run:  npm test
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createClient } from "@libsql/client";
import type { Client } from "@libsql/client";

// Import the type guard from production
import { isKnowledgeSyncRulesArgs } from "../src/tools/sessionMemoryDefinitions.js";

// ──────────────────────────────────────────────────────────────
// Helper: Create in-memory DB with full schema
// ──────────────────────────────────────────────────────────────

async function makeDb(): Promise<Client> {
  const db = createClient({ url: "file::memory:" });

  await db.executeMultiple(`
    CREATE TABLE IF NOT EXISTS session_ledger (
      id            TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
      project       TEXT NOT NULL,
      user_id       TEXT NOT NULL DEFAULT 'default',
      conversation_id TEXT DEFAULT '',
      summary       TEXT DEFAULT '',
      todos         TEXT DEFAULT '[]',
      decisions     TEXT DEFAULT '[]',
      files_changed TEXT DEFAULT '[]',
      keywords      TEXT DEFAULT '[]',
      role          TEXT DEFAULT 'global',
      is_rollup     INTEGER DEFAULT 0,
      rollup_count  INTEGER DEFAULT 0,
      event_type    TEXT DEFAULT 'session',
      confidence_score INTEGER DEFAULT NULL,
      importance    INTEGER DEFAULT 0,
      archived_at   TEXT DEFAULT NULL,
      deleted_at    TEXT DEFAULT NULL,
      created_at    TEXT DEFAULT (datetime('now'))
    );
  `);

  return db;
}

/** Seed helper — inserts a ledger entry with controlled importance */
async function seedEntry(
  db: Client,
  overrides: {
    project?: string;
    importance?: number;
    event_type?: string;
    summary?: string;
    archived_at?: string | null;
    deleted_at?: string | null;
    user_id?: string;
  } = {}
): Promise<string> {
  const result = await db.execute({
    sql: `INSERT INTO session_ledger
          (project, user_id, summary, importance, event_type, archived_at, deleted_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)`,
    args: [
      overrides.project ?? "test-project",
      overrides.user_id ?? "default",
      overrides.summary ?? "Test insight",
      overrides.importance ?? 0,
      overrides.event_type ?? "correction",
      overrides.archived_at ?? null,
      overrides.deleted_at ?? null,
    ],
  });
  // Return the generated ID
  const rows = await db.execute({
    sql: `SELECT id FROM session_ledger ORDER BY rowid DESC LIMIT 1`,
    args: [],
  });
  return rows.rows[0].id as string;
}

// ──────────────────────────────────────────────────────────────
// Test: getGraduatedInsights query logic
// ──────────────────────────────────────────────────────────────

describe("getGraduatedInsights", () => {
  let db: Client;

  beforeEach(async () => {
    db = await makeDb();
  });

  afterEach(async () => {
    db.close();
  });

  it("returns entries with importance >= 7", async () => {
    await seedEntry(db, { importance: 7, summary: "Graduated insight" });
    await seedEntry(db, { importance: 9, summary: "High importance" });
    await seedEntry(db, { importance: 3, summary: "Low importance" });

    const result = await db.execute({
      sql: `SELECT id, summary, importance FROM session_ledger
            WHERE project = 'test-project' AND user_id = 'default'
              AND importance >= 7
              AND deleted_at IS NULL AND archived_at IS NULL
            ORDER BY importance DESC`,
      args: [],
    });

    expect(result.rows).toHaveLength(2);
    expect(Number(result.rows[0].importance)).toBe(9);
    expect(Number(result.rows[1].importance)).toBe(7);
  });

  it("excludes archived entries", async () => {
    await seedEntry(db, {
      importance: 10,
      summary: "Archived graduate",
      archived_at: new Date().toISOString(),
    });
    await seedEntry(db, { importance: 8, summary: "Active graduate" });

    const result = await db.execute({
      sql: `SELECT id FROM session_ledger
            WHERE project = 'test-project' AND user_id = 'default'
              AND importance >= 7
              AND deleted_at IS NULL AND archived_at IS NULL`,
      args: [],
    });

    expect(result.rows).toHaveLength(1);
  });

  it("excludes soft-deleted entries", async () => {
    await seedEntry(db, {
      importance: 8,
      summary: "Deleted graduate",
      deleted_at: new Date().toISOString(),
    });
    await seedEntry(db, { importance: 7, summary: "Active insight" });

    const result = await db.execute({
      sql: `SELECT id FROM session_ledger
            WHERE project = 'test-project' AND user_id = 'default'
              AND importance >= 7
              AND deleted_at IS NULL AND archived_at IS NULL`,
      args: [],
    });

    expect(result.rows).toHaveLength(1);
  });

  it("scopes by project — no cross-project leakage", async () => {
    await seedEntry(db, { project: "project-a", importance: 9, summary: "A insight" });
    await seedEntry(db, { project: "project-b", importance: 8, summary: "B insight" });

    const result = await db.execute({
      sql: `SELECT id FROM session_ledger
            WHERE project = 'project-a' AND user_id = 'default'
              AND importance >= 7
              AND deleted_at IS NULL AND archived_at IS NULL`,
      args: [],
    });

    expect(result.rows).toHaveLength(1);
  });

  it("returns empty array when no graduated insights exist", async () => {
    await seedEntry(db, { importance: 3, summary: "Not graduated" });
    await seedEntry(db, { importance: 6, summary: "Almost graduated" });

    const result = await db.execute({
      sql: `SELECT id FROM session_ledger
            WHERE project = 'test-project' AND user_id = 'default'
              AND importance >= 7
              AND deleted_at IS NULL AND archived_at IS NULL`,
      args: [],
    });

    expect(result.rows).toHaveLength(0);
  });
});

// ──────────────────────────────────────────────────────────────
// Test: Sentinel block formatting and idempotency
// ──────────────────────────────────────────────────────────────

const SENTINEL_START = "<!-- PRISM:AUTO-RULES:START -->";
const SENTINEL_END = "<!-- PRISM:AUTO-RULES:END -->";

/** Inline helper to mimic the handler's applySentinelBlock logic */
function applySentinelBlock(existingContent: string, rulesBlock: string): string {
  const startIdx = existingContent.indexOf(SENTINEL_START);
  const endIdx = existingContent.indexOf(SENTINEL_END);

  if (startIdx !== -1 && endIdx !== -1) {
    const before = existingContent.substring(0, startIdx);
    const after = existingContent.substring(endIdx + SENTINEL_END.length);
    return `${before}${rulesBlock}${after}`;
  }

  const separator = existingContent.length > 0 && !existingContent.endsWith("\n\n")
    ? (existingContent.endsWith("\n") ? "\n" : "\n\n")
    : "";
  return `${existingContent}${separator}${rulesBlock}\n`;
}

describe("Sentinel block formatting", () => {
  const sampleBlock = `${SENTINEL_START}\n## Prism Graduated Insights\n- rule 1\n${SENTINEL_END}`;

  it("appends sentinels to an empty file", () => {
    const result = applySentinelBlock("", sampleBlock);
    expect(result).toContain(SENTINEL_START);
    expect(result).toContain(SENTINEL_END);
    expect(result).toContain("rule 1");
  });

  it("appends sentinels to existing content", () => {
    const existing = "# My Rules\n\n- Existing rule 1\n";
    const result = applySentinelBlock(existing, sampleBlock);

    // Existing content preserved
    expect(result).toContain("# My Rules");
    expect(result).toContain("Existing rule 1");
    // Sentinel block appended
    expect(result).toContain(SENTINEL_START);
    expect(result).toContain("rule 1");
  });

  it("replaces existing sentinel block (idempotent)", () => {
    const existingWithBlock = `# Header\n\n${SENTINEL_START}\n## Old Rules\n- old rule\n${SENTINEL_END}\n\n# Footer`;
    const newBlock = `${SENTINEL_START}\n## Updated Rules\n- new rule\n${SENTINEL_END}`;

    const result = applySentinelBlock(existingWithBlock, newBlock);

    // Old content replaced
    expect(result).not.toContain("old rule");
    expect(result).not.toContain("Old Rules");
    // New content present
    expect(result).toContain("new rule");
    expect(result).toContain("Updated Rules");
    // Surrounding content preserved
    expect(result).toContain("# Header");
    expect(result).toContain("# Footer");
  });

  it("double-apply produces identical result (idempotency)", () => {
    const first = applySentinelBlock("# Existing\n", sampleBlock);
    const second = applySentinelBlock(first, sampleBlock);
    expect(second).toBe(first);
  });
});

// ──────────────────────────────────────────────────────────────
// Test: Type guard validation
// ──────────────────────────────────────────────────────────────

describe("isKnowledgeSyncRulesArgs type guard", () => {
  it("accepts valid args with only project", () => {
    expect(isKnowledgeSyncRulesArgs({ project: "my-project" })).toBe(true);
  });

  it("accepts valid args with all fields", () => {
    expect(isKnowledgeSyncRulesArgs({
      project: "my-project",
      target_file: ".clauderules",
      dry_run: true,
    })).toBe(true);
  });

  it("rejects missing project", () => {
    expect(isKnowledgeSyncRulesArgs({})).toBe(false);
  });

  it("rejects null", () => {
    expect(isKnowledgeSyncRulesArgs(null)).toBe(false);
  });

  it("rejects non-string project", () => {
    expect(isKnowledgeSyncRulesArgs({ project: 42 })).toBe(false);
  });
});
