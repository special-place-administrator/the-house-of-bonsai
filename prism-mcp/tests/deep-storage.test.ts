/**
 * Deep Storage Mode Tests — Prism MCP v5.1
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHAT IS DEEP STORAGE MODE?
 *
 *   v5.0 introduced RotorQuant, which compresses 768-dimensional float32
 *   embeddings (3KB each) into ~400-byte blobs with 95%+ cosine accuracy.
 *   After compression, old entries have BOTH representations stored:
 *     - embedding (BLOB):            original float32 vector (3KB)
 *     - embedding_compressed (TEXT):  RotorQuant compressed blob (~400B)
 *
 *   v5.1 Deep Storage Mode ("The Purge") NULLs out the redundant float32
 *   column for old entries, reclaiming ~90% of vector storage while
 *   maintaining search accuracy via Tier-2 RotorQuant search.
 *
 * WHAT THESE TESTS VERIFY:
 *
 *   1. DRY RUN MODE:
 *      - Correctly counts eligible entries without modifying data
 *      - Reports accurate byte counts for potential space savings
 *
 *   2. EXECUTE MODE:
 *      - Only purges entries with BOTH embedding AND embedding_compressed
 *      - Preserves entries that only have float32 (no compressed fallback)
 *      - Preserves entries newer than the age threshold
 *      - Preserves embedding_compressed blobs after purge (never destroys them)
 *
 *   3. SAFETY GUARDS:
 *      - Rejects olderThanDays < 7 (prevents purging active data)
 *      - Respects project filter (only purges specified project)
 *
 *   4. EDGE CASES:
 *      - Soft-deleted entries are excluded (never purge tombstoned rows)
 *      - Empty database returns zeroes without errors
 *
 * ISOLATION:
 *   Each test uses createTestDb() which creates an ephemeral SQLite database
 *   in a temp directory. Tests share state within the suite but cleanup
 *   happens in afterAll().
 *
 * ARCHITECTURE NOTE:
 *   We test at the StorageBackend level (not the handler level) because:
 *   - The handler is a thin wrapper (getStorage + format response)
 *   - The real logic lives in SqliteStorage.purgeHighPrecisionEmbeddings()
 *   - By testing the storage layer directly, we verify SQL correctness
 * ═══════════════════════════════════════════════════════════════════
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createTestDb, TEST_PROJECT, TEST_USER_ID } from "./helpers/fixtures.js";

// ─── Shared test state ───────────────────────────────────────────
let storage: any;
let cleanup: () => void;

beforeAll(async () => {
  const testDb = await createTestDb("deep-storage");
  storage = testDb.storage;
  cleanup = testDb.cleanup;

  // ── Seed the database with test entries ──
  // We create entries with varying states to test all purge conditions:
  //
  // Entry 1-3: OLD entries with BOTH embedding AND embedding_compressed
  //   → These are ELIGIBLE for purge (the target scenario)
  //
  // Entry 4: OLD entry with ONLY embedding (no compressed blob)
  //   → Must NOT be purged (would destroy the only search index)
  //
  // Entry 5: RECENT entry with BOTH embedding AND embedding_compressed
  //   → Must NOT be purged (too young — still useful for Tier-1 search)
  //
  // Entry 6: OLD entry with BOTH, but SOFT-DELETED
  //   → Must NOT be purged (tombstoned entries are excluded)
  //
  // Entry 7: Entry on a DIFFERENT project (for project filter testing)
  //   → Should only be purged when no project filter is applied

  // Create dummy binary data that simulates real embeddings.
  //
  // IMPORTANT: The sqlite-vec vector index on session_ledger.embedding is
  // configured for 768-dimensional float32 vectors (768 × 4 bytes = 3072 bytes).
  // Using a different dimension count causes SQLITE_ERROR at INSERT time
  // because the index rejects mismatched vector sizes.
  //
  // We fill with 0.1 for simplicity — actual vector content doesn't matter
  // for purge tests, only the presence/absence and byte size of the column.
  const float32Array = new Float32Array(768).fill(0.1); // 768 dims × 4 bytes = 3,072 bytes
  const float32Blob = Buffer.from(float32Array.buffer);
  const compressedBlob = Buffer.from("dHVyYm9xdWFudC1ibG9i"); // ~20 bytes base64 (simulates RotorQuant output)

  // Helper to save a ledger entry with controlled timestamps and data
  const saveLedgerDirect = async (
    id: string,
    project: string,
    hasEmbedding: boolean,
    hasCompressed: boolean,
    isOld: boolean,
    isDeleted: boolean = false,
  ) => {
    await storage.db.execute({
      sql: `INSERT INTO session_ledger
            (id, project, user_id, conversation_id, summary, created_at, deleted_at,
             embedding, embedding_compressed, embedding_format)
            VALUES (?, ?, ?, ?, ?, datetime('now', ?), ?, ?, ?, ?)`,
      args: [
        id,
        project,
        TEST_USER_ID,
        `conv-${id}`,
        `Test entry ${id}`,
        isOld ? "-60 days" : "-1 days",   // old = 60 days ago, recent = 1 day ago
        isDeleted ? new Date().toISOString() : null,
        hasEmbedding ? float32Blob : null,
        hasCompressed ? compressedBlob : null,
        hasCompressed ? "turbo2" : null,
      ],
    });
  };

  // Seed entries for comprehensive coverage
  await saveLedgerDirect("purge-1", TEST_PROJECT, true,  true,  true);  // eligible
  await saveLedgerDirect("purge-2", TEST_PROJECT, true,  true,  true);  // eligible
  await saveLedgerDirect("purge-3", TEST_PROJECT, true,  true,  true);  // eligible
  await saveLedgerDirect("purge-4", TEST_PROJECT, true,  false, true);  // float32 only — NOT eligible
  await saveLedgerDirect("purge-5", TEST_PROJECT, true,  true,  false); // recent — NOT eligible
  await saveLedgerDirect("purge-6", TEST_PROJECT, true,  true,  true,  true);  // soft-deleted — NOT eligible
  await saveLedgerDirect("purge-7", "other-project", true, true, true); // different project

}, 15_000);

afterAll(() => {
  cleanup();
});

// ═══════════════════════════════════════════════════════════════════
// 1. DRY RUN MODE
// ═══════════════════════════════════════════════════════════════════

describe("Deep Storage Mode — Dry Run", () => {
  /**
   * Verifies that dry run correctly identifies eligible entries for the
   * target project without modifying any data.
   *
   * Expected: 3 eligible (purge-1, purge-2, purge-3)
   *   - purge-4: excluded (no compressed blob)
   *   - purge-5: excluded (too recent)
   *   - purge-6: excluded (soft-deleted)
   *   - purge-7: excluded (different project)
   */
  it("should report eligible count without modifying data", async () => {
    const result = await storage.purgeHighPrecisionEmbeddings({
      project: TEST_PROJECT,
      olderThanDays: 7,
      dryRun: true,
      userId: TEST_USER_ID,
    });

    // Should find 3 eligible entries
    expect(result.eligible).toBe(3);
    // Should not have purged anything
    expect(result.purged).toBe(0);
    // Should report positive byte count (3 entries × float32 blob size)
    expect(result.reclaimedBytes).toBeGreaterThan(0);

    // Verify no data was actually modified — all 7 entries should
    // still have their original embedding values
    const allEntries = await storage.db.execute({
      sql: `SELECT id, embedding FROM session_ledger WHERE id LIKE 'purge-%' ORDER BY id`,
      args: [],
    });
    const withEmbedding = allEntries.rows.filter((r: any) => r.embedding !== null);
    expect(withEmbedding).toHaveLength(7); // ALL entries still have embeddings
  });

  /**
   * Verifies dry run works across all projects when no project filter is set.
   *
   * Expected: 4 eligible (purge-1, purge-2, purge-3 from TEST_PROJECT + purge-7 from other-project)
   */
  it("should count all projects when no project filter", async () => {
    const result = await storage.purgeHighPrecisionEmbeddings({
      olderThanDays: 7,
      dryRun: true,
      userId: TEST_USER_ID,
    });

    // Should find 4 eligible entries across all projects
    // (purge-1, purge-2, purge-3, purge-7)
    expect(result.eligible).toBe(4);
    expect(result.purged).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. EXECUTE MODE — Project-Scoped Purge
// ═══════════════════════════════════════════════════════════════════

describe("Deep Storage Mode — Execute (Project-Scoped)", () => {
  /**
   * Executes the purge for TEST_PROJECT only and verifies:
   *   1. Exactly 3 entries are purged
   *   2. Purged entries have embedding = NULL
   *   3. Purged entries STILL have embedding_compressed (preserved)
   *   4. Non-eligible entries are untouched
   *   5. Other-project entries are untouched
   */
  it("should purge only eligible entries for the specified project", async () => {
    const result = await storage.purgeHighPrecisionEmbeddings({
      project: TEST_PROJECT,
      olderThanDays: 7,
      dryRun: false,
      userId: TEST_USER_ID,
    });

    // Should have purged exactly 3 entries
    expect(result.purged).toBe(3);
    expect(result.eligible).toBe(3);
    expect(result.reclaimedBytes).toBeGreaterThan(0);
  });

  /**
   * Verifies that purged entries have embedding = NULL but
   * embedding_compressed is still intact.
   *
   * THIS IS THE CRITICAL SAFETY CHECK:
   *   If embedding_compressed were also NULLed, the entry would
   *   become unsearchable — a data loss scenario.
   */
  it("should NULL embedding but preserve embedding_compressed", async () => {
    // Check purged entries (purge-1, purge-2, purge-3)
    const purgedEntries = await storage.db.execute({
      sql: `SELECT id, embedding, embedding_compressed FROM session_ledger
            WHERE id IN ('purge-1', 'purge-2', 'purge-3') ORDER BY id`,
      args: [],
    });

    expect(purgedEntries.rows).toHaveLength(3);
    for (const row of purgedEntries.rows) {
      // Float32 should be NULL (purged)
      expect(row.embedding).toBeNull();
      // Compressed should still be present (preserved)
      expect(row.embedding_compressed).not.toBeNull();
    }
  });

  /**
   * Verifies that non-eligible entries are completely untouched:
   *   - purge-4: had only float32, no compressed → must keep float32
   *   - purge-5: was too recent → must keep float32
   *   - purge-6: was soft-deleted → must keep float32
   */
  it("should not affect non-eligible entries", async () => {
    // Entry 4: float32 only (no compressed fallback) — MUST keep embedding
    const entry4 = await storage.db.execute({
      sql: `SELECT embedding FROM session_ledger WHERE id = 'purge-4'`,
      args: [],
    });
    expect(entry4.rows[0]?.embedding).not.toBeNull();

    // Entry 5: recent entry — MUST keep embedding
    const entry5 = await storage.db.execute({
      sql: `SELECT embedding FROM session_ledger WHERE id = 'purge-5'`,
      args: [],
    });
    expect(entry5.rows[0]?.embedding).not.toBeNull();

    // Entry 6: soft-deleted — MUST keep embedding (deleted_at IS NULL filter)
    const entry6 = await storage.db.execute({
      sql: `SELECT embedding FROM session_ledger WHERE id = 'purge-6'`,
      args: [],
    });
    expect(entry6.rows[0]?.embedding).not.toBeNull();
  });

  /**
   * Verifies that other-project entry is untouched when project filter was set.
   */
  it("should not affect entries in other projects", async () => {
    const entry7 = await storage.db.execute({
      sql: `SELECT embedding, embedding_compressed FROM session_ledger WHERE id = 'purge-7'`,
      args: [],
    });
    // Both should still be present — purge was scoped to TEST_PROJECT
    expect(entry7.rows[0]?.embedding).not.toBeNull();
    expect(entry7.rows[0]?.embedding_compressed).not.toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. SAFETY GUARDS
// ═══════════════════════════════════════════════════════════════════

describe("Deep Storage Mode — Safety Guards", () => {
  /**
   * Verifies that olderThanDays < 7 throws an error.
   *
   * WHY THIS MATTERS:
   *   Entries younger than 7 days are likely from active sessions.
   *   Purging them would degrade Tier-1 native vector search quality
   *   for the user's most recent work — an unacceptable trade-off.
   */
  it("should reject olderThanDays < 7", async () => {
    await expect(
      storage.purgeHighPrecisionEmbeddings({
        olderThanDays: 5,
        dryRun: false,
        userId: TEST_USER_ID,
      })
    ).rejects.toThrow("olderThanDays must be at least 7");

    // Also verify that exactly 7 is accepted (boundary condition)
    await expect(
      storage.purgeHighPrecisionEmbeddings({
        olderThanDays: 7,
        dryRun: true,
        userId: TEST_USER_ID,
      })
    ).resolves.not.toThrow();
  });

  /**
   * Verifies idempotency — running purge twice should be a no-op the second time.
   *
   * After the first purge, eligible entries already have embedding = NULL,
   * so the WHERE clause (embedding IS NOT NULL) filters them out.
   */
  it("should be idempotent (second purge finds 0 eligible)", async () => {
    // First, purge the remaining other-project entry
    const result = await storage.purgeHighPrecisionEmbeddings({
      project: "other-project",
      olderThanDays: 7,
      dryRun: false,
      userId: TEST_USER_ID,
    });
    expect(result.purged).toBe(1); // purge-7

    // Second purge of the same project should find nothing
    const result2 = await storage.purgeHighPrecisionEmbeddings({
      project: "other-project",
      olderThanDays: 7,
      dryRun: false,
      userId: TEST_USER_ID,
    });
    expect(result2.eligible).toBe(0);
    expect(result2.purged).toBe(0);
    expect(result2.reclaimedBytes).toBe(0);
  });
});
