-- ═══════════════════════════════════════════════════════════════════
-- Prism MCP v3.0.1: Repair — Fix session_handoffs unique constraint
-- ═══════════════════════════════════════════════════════════════════
--
-- WHAT THIS FIXES:
--   Migration 024 tried to drop old unique constraints using
--   `DROP INDEX IF EXISTS session_handoffs_project_user_id_key`,
--   but migration 020 created the constraint via:
--     ALTER TABLE ADD CONSTRAINT uq_handoffs_user_project UNIQUE (user_id, project)
--
--   Since DROP INDEX doesn't affect named constraints, the old
--   2-column constraint survived, causing:
--     - Duplicate key errors when saving handoffs with role
--     - get_session_context returning 0 rows (function signature mismatch
--       if old 3-param version lingered alongside new 4-param version)
--     - Memory appearing "erased" on session_load_context
--
-- THIS MIGRATION IS SAFE TO RE-RUN (all operations are idempotent).
-- ═══════════════════════════════════════════════════════════════════

-- ─── Step 1: Drop ALL old unique constraints on session_handoffs ──
DO $$
DECLARE
  constraint_rec RECORD;
BEGIN
  FOR constraint_rec IN
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'session_handoffs'::regclass
      AND contype = 'u'
  LOOP
    EXECUTE format('ALTER TABLE session_handoffs DROP CONSTRAINT %I', constraint_rec.conname);
    RAISE NOTICE 'Dropped constraint: %', constraint_rec.conname;
  END LOOP;
END $$;

-- ─── Step 2: Drop any lingering unique indexes ───────────────────
DROP INDEX IF EXISTS session_handoffs_project_user_id_key;
DROP INDEX IF EXISTS session_handoffs_project_user_id_role_key;
DROP INDEX IF EXISTS idx_handoffs_user_project;

-- ─── Step 3: Ensure role column exists with correct default ──────
ALTER TABLE session_handoffs
  ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'global';

-- Backfill any NULL roles (shouldn't happen, but safety net)
UPDATE session_handoffs SET role = 'global' WHERE role IS NULL;

-- ─── Step 4: Create the correct 3-column unique index ────────────
CREATE UNIQUE INDEX IF NOT EXISTS session_handoffs_project_user_id_role_key
  ON session_handoffs (project, user_id, role);

-- ─── Step 5: Ensure RPC functions have correct 4-param signatures ─
-- Re-drop old function signatures to clean up any lingering overloads
DROP FUNCTION IF EXISTS get_session_context(TEXT, TEXT, TEXT);
DROP FUNCTION IF EXISTS save_handoff_with_version(TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[], TEXT, TEXT, TEXT);

-- Verify the v3.0 functions exist (024 should have created them)
-- If they don't exist, this will raise a clear error
DO $$
BEGIN
  -- Check get_session_context has 4 params
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc
    WHERE proname = 'get_session_context'
      AND pronargs = 4
  ) THEN
    RAISE EXCEPTION 'get_session_context(TEXT,TEXT,TEXT,TEXT) not found — re-run migration 024 first';
  END IF;

  -- Check save_handoff_with_version has 10 params
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc
    WHERE proname = 'save_handoff_with_version'
      AND pronargs = 10
  ) THEN
    RAISE EXCEPTION 'save_handoff_with_version with 10 params not found — re-run migration 024 first';
  END IF;

  RAISE NOTICE 'All RPC functions verified ✓';
END $$;

-- ═══════════════════════════════════════════════════════════════════
-- REPAIR COMPLETE
-- ═══════════════════════════════════════════════════════════════════
-- After running this migration:
--   1. Existing handoff rows keep their data (role='global' backfilled)
--   2. The unique constraint is now (project, user_id, role)
--   3. get_session_context and save_handoff_with_version use v3.0 signatures
--   4. session_load_context should return your memory again
-- ═══════════════════════════════════════════════════════════════════
