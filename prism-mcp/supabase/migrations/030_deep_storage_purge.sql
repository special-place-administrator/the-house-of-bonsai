-- ═══════════════════════════════════════════════════════════════════
-- Migration 030: Deep Storage Mode — Purge RPC
-- Prism MCP v5.1
-- ═══════════════════════════════════════════════════════════════════
--
-- PURPOSE:
--   Provides a server-side Postgres function that NULLs out the bulky
--   float32 `embedding` column for session_ledger entries that already
--   have a TurboQuant `embedding_compressed` blob. This reclaims ~90%
--   of vector storage while preserving Tier-2 search accuracy (95%+).
--
-- CONTEXT:
--   v5.0 introduced TurboQuant compression, storing a ~400-byte
--   compressed blob alongside the original 3KB float32 embedding.
--   Once entries are old enough, the float32 is pure redundancy —
--   Tier-2 asymmetric search uses the compressed blob only.
--
--   This RPC is the Supabase counterpart to the SQLite implementation
--   in SqliteStorage.purgeHighPrecisionEmbeddings().
--
-- SAFETY GUARDS:
--   1. Only purges entries where embedding_compressed IS NOT NULL
--      (never destroys the last searchable vector)
--   2. Requires p_older_than_days >= 7 (CHECK enforced in-function)
--   3. Multi-tenant: scoped to p_user_id
--   4. Skips soft-deleted entries (deleted_at IS NOT NULL)
--   5. Optional project filter (NULL = all projects)
--
-- USAGE:
--   SELECT * FROM prism_purge_embeddings(
--     p_project := 'my-project',   -- optional, NULL for all projects
--     p_user_id := 'default',
--     p_older_than_days := 30,
--     p_dry_run := true             -- preview mode
--   );
--
-- RETURNS:
--   Single row with: eligible (INT), purged (INT), reclaimed_bytes (BIGINT)
--
-- IMPORTANT: Run this AFTER migration 027 (auto_migration_infra).
-- For local SQLite users, this is unnecessary — SQLite handles purging
-- natively in sqlite.ts.
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION prism_purge_embeddings(
  p_project         TEXT    DEFAULT NULL,    -- NULL = all projects
  p_user_id         TEXT    DEFAULT 'default',
  p_older_than_days INTEGER DEFAULT 30,
  p_dry_run         BOOLEAN DEFAULT false
)
RETURNS TABLE(eligible INTEGER, purged INTEGER, reclaimed_bytes BIGINT)
LANGUAGE plpgsql
SECURITY DEFINER          -- Runs with owner (postgres) privileges for UPDATE
SET search_path = public  -- Prevent search_path hijacking
AS $$
DECLARE
  v_eligible    INTEGER;
  v_bytes       BIGINT;
  v_cutoff      TIMESTAMPTZ;
BEGIN
  -- ── Safety guard: reject olderThanDays < 7 ──
  -- Entries younger than 7 days may still benefit from Tier-1 native
  -- vector search (pgvector/Supabase Vecs). Purging them would silently
  -- degrade search quality for active projects.
  IF p_older_than_days < 7 THEN
    RAISE EXCEPTION 'p_older_than_days must be at least 7 to prevent purging recent entries';
  END IF;

  -- Calculate the cutoff timestamp once (deterministic within the function)
  v_cutoff := now() - (p_older_than_days || ' days')::interval;

  -- ── Step 1: Count eligible entries and estimate reclaimed bytes ──
  -- octet_length(embedding::text) approximates the storage cost of
  -- the embedding column. For binary BYTEA columns, this is the exact
  -- byte count. For TEXT-serialized vectors, it's a close approximation.
  SELECT
    COUNT(*)::INTEGER,
    COALESCE(SUM(octet_length(embedding::text)), 0)::BIGINT
  INTO v_eligible, v_bytes
  FROM session_ledger
  WHERE embedding IS NOT NULL                 -- has a float32 vector to purge
    AND embedding_compressed IS NOT NULL       -- CRITICAL: has TurboQuant fallback
    AND deleted_at IS NULL                     -- skip tombstoned entries
    AND created_at < v_cutoff                  -- only old entries
    AND user_id = p_user_id                    -- multi-tenant guard
    AND (p_project IS NULL OR project = p_project);  -- optional project filter

  -- ── Dry run: return counts without modifying data ──
  IF p_dry_run THEN
    RETURN QUERY SELECT v_eligible, 0::INTEGER, v_bytes;
    RETURN;
  END IF;

  -- ── Step 2: Execute the purge — NULL out the float32 column ──
  -- Single UPDATE is atomic — either all eligible entries are purged
  -- or none are (in case of a database error).
  IF v_eligible > 0 THEN
    UPDATE session_ledger
    SET embedding = NULL
    WHERE embedding IS NOT NULL
      AND embedding_compressed IS NOT NULL
      AND deleted_at IS NULL
      AND created_at < v_cutoff
      AND user_id = p_user_id
      AND (p_project IS NULL OR project = p_project);
  END IF;

  RETURN QUERY SELECT v_eligible, v_eligible, v_bytes;
END;
$$;

-- ── Document the function for pg_catalog introspection ──
COMMENT ON FUNCTION prism_purge_embeddings(TEXT, TEXT, INTEGER, BOOLEAN) IS
  'v5.1 Deep Storage Mode: Purge float32 embeddings for entries with '
  'TurboQuant compressed blobs (embedding_compressed). Reclaims ~90% '
  'of vector storage. Use p_dry_run=true to preview. '
  'Safety: requires p_older_than_days >= 7, scoped to p_user_id.';
