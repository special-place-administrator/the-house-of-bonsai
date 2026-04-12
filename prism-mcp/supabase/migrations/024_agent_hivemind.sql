-- ═══════════════════════════════════════════════════════════════════
-- Prism MCP v3.0: Agent Hivemind — Supabase Migration
-- ═══════════════════════════════════════════════════════════════════
--
-- This migration adds multi-agent (Hivemind) support:
--   1. Add 'role' column to session_ledger (defaults to 'global')
--   2. Rebuild session_handoffs with role in UNIQUE constraint
--   3. Create agent_registry table for team coordination
--   4. Create system_settings table for dashboard config
--   5. Apply RLS policies for multi-tenant security
--
-- IMPORTANT: Run this AFTER all previous migrations (015-023).
-- For local SQLite users, these changes are applied automatically
-- on server startup — this file is only needed for Supabase/Postgres.
-- ═══════════════════════════════════════════════════════════════════

-- ─── 1. session_ledger: Add role column ──────────────────────────
-- Existing entries get 'global' — backward compatible.
ALTER TABLE session_ledger
  ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'global';

-- ─── 2. session_handoffs: Add role column + rebuild UNIQUE ───────
-- Step 1: Add the role column with default
ALTER TABLE session_handoffs
  ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'global';

-- Step 2: Drop ALL old UNIQUE constraints/indexes on session_handoffs,
-- then create the new 3-column unique index (project, user_id, role).
--
-- Migration 020 created: ALTER TABLE ADD CONSTRAINT uq_handoffs_user_project UNIQUE (user_id, project)
-- The old code tried DROP INDEX by guessed names — which silently missed named constraints.
-- This fix dynamically discovers and drops ALL unique constraints regardless of name.

-- 2a: Drop any named UNIQUE constraints (from ALTER TABLE ADD CONSTRAINT)
DO $$
DECLARE
  constraint_rec RECORD;
BEGIN
  FOR constraint_rec IN
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'session_handoffs'::regclass
      AND contype = 'u'  -- unique constraints only
  LOOP
    EXECUTE format('ALTER TABLE session_handoffs DROP CONSTRAINT %I', constraint_rec.conname);
    RAISE NOTICE 'Dropped constraint: %', constraint_rec.conname;
  END LOOP;
END $$;

-- 2b: Drop any unique indexes (from CREATE UNIQUE INDEX)
DROP INDEX IF EXISTS session_handoffs_project_user_id_key;
DROP INDEX IF EXISTS session_handoffs_project_user_id_role_key;
DROP INDEX IF EXISTS idx_handoffs_user_project;

-- 2c: Create the new 3-column unique index
CREATE UNIQUE INDEX IF NOT EXISTS session_handoffs_project_user_id_role_key
  ON session_handoffs (project, user_id, role);

-- ─── 3. agent_registry: New table for Hivemind coordination ─────
-- Agents register here to announce presence, role, and current task.
-- Stale entries are auto-pruned by the server (30-minute window).
CREATE TABLE IF NOT EXISTS agent_registry (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project       TEXT NOT NULL,
  user_id       TEXT NOT NULL,
  role          TEXT NOT NULL,
  agent_name    TEXT,
  status        TEXT NOT NULL DEFAULT 'active',
  current_task  TEXT,
  last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Each user can only have one agent per role per project
  UNIQUE (project, user_id, role)
);

-- Index for fast team lookups (used by agent_list_team)
CREATE INDEX IF NOT EXISTS idx_agent_registry_project_user
  ON agent_registry (project, user_id);

-- Index for stale agent pruning (WHERE last_heartbeat < threshold)
CREATE INDEX IF NOT EXISTS idx_agent_registry_heartbeat
  ON agent_registry (last_heartbeat);

-- ─── 4. system_settings: Key-value store for dashboard config ───
-- Runtime settings that can be changed from the dashboard UI
-- without restarting the server.
CREATE TABLE IF NOT EXISTS system_settings (
  key         TEXT PRIMARY KEY,
  value       TEXT NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── 5. Update RPC functions to accept p_role parameter ─────────
-- v3.0: The TypeScript storage layer now sends p_role to these RPCs.
-- Without this update, Supabase returns 404 (function signature mismatch).

-- Drop old function signatures to avoid overload conflicts
DROP FUNCTION IF EXISTS get_session_context(TEXT, TEXT, TEXT);
DROP FUNCTION IF EXISTS save_handoff_with_version(TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[], TEXT, TEXT, TEXT);

-- 5a. get_session_context — add p_role parameter
CREATE OR REPLACE FUNCTION get_session_context(
    p_project TEXT DEFAULT 'default',
    p_level TEXT DEFAULT 'standard',
    p_user_id TEXT DEFAULT 'default',
    p_role TEXT DEFAULT 'global'
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB := '{}'::jsonb;
    handoff RECORD;
    ledger_entries JSONB;
    knowledge_cache JSONB;
    hot_keywords TEXT[];
    top_categories TEXT[];
    related_count INT;
BEGIN
    -- Get the handoff record for this user + project + role
    SELECT * INTO handoff
    FROM session_handoffs
    WHERE project = p_project
      AND user_id = p_user_id
      AND role = p_role;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'level', p_level,
            'project', p_project,
            'status', 'no_previous_session',
            'message', 'No previous session found for this project.'
        );
    END IF;

    -- quick: Keywords + TODO + VERSION
    result := jsonb_build_object(
        'level', p_level,
        'project', p_project,
        'last_agent', handoff.last_agent,
        'keywords', to_jsonb(handoff.keywords),
        'pending_todo', to_jsonb(handoff.pending_todo),
        'updated_at', handoff.updated_at,
        'version', handoff.version
    );

    -- standard: + Summary + Decisions + Knowledge Cache
    IF p_level IN ('standard', 'deep') THEN
        result := result || jsonb_build_object(
            'last_title', handoff.last_title,
            'last_summary', handoff.last_summary,
            'active_decisions', to_jsonb(handoff.active_decisions)
        );

        SELECT ARRAY(
            SELECT kw
            FROM session_ledger sl, unnest(sl.keywords) AS kw
            WHERE sl.project = p_project
              AND sl.user_id = p_user_id
              AND sl.created_at >= NOW() - INTERVAL '7 days'
              AND sl.archived_at IS NULL
              AND kw NOT LIKE 'cat:%'
            GROUP BY kw
            ORDER BY COUNT(*) DESC
            LIMIT 5
        ) INTO hot_keywords;

        SELECT ARRAY(
            SELECT REPLACE(kw, 'cat:', '')
            FROM session_ledger sl, unnest(sl.keywords) AS kw
            WHERE sl.project = p_project
              AND sl.user_id = p_user_id
              AND sl.created_at >= NOW() - INTERVAL '7 days'
              AND sl.archived_at IS NULL
              AND kw LIKE 'cat:%'
            GROUP BY kw
            ORDER BY COUNT(*) DESC
            LIMIT 3
        ) INTO top_categories;

        SELECT COUNT(*) INTO related_count
        FROM session_ledger
        WHERE project = p_project
          AND user_id = p_user_id
          AND archived_at IS NULL;

        knowledge_cache := jsonb_build_object(
            'hot_keywords', COALESCE(to_jsonb(hot_keywords), '[]'::jsonb),
            'top_categories', COALESCE(to_jsonb(top_categories), '[]'::jsonb),
            'total_sessions', COALESCE(related_count, 0)
        );

        result := result || jsonb_build_object('knowledge_cache', knowledge_cache);
    END IF;

    -- deep: + Last 5 ledger entries + cross-project knowledge
    IF p_level = 'deep' THEN
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'date', sub.session_date,
                'agent', sub.agent_name,
                'title', sub.title,
                'summary', sub.summary,
                'keywords', to_jsonb(sub.keywords),
                'files_changed', to_jsonb(sub.files_changed),
                'decisions', to_jsonb(sub.decisions),
                'todo_next', to_jsonb(sub.todo_next)
            )
        ), '[]'::jsonb) INTO ledger_entries
        FROM (
            SELECT sl.session_date, sl.agent_name, sl.title, sl.summary,
                   sl.keywords, sl.files_changed, sl.decisions, sl.todo_next, sl.created_at
            FROM session_ledger sl
            WHERE sl.project = p_project
              AND sl.user_id = p_user_id
              AND sl.archived_at IS NULL
            ORDER BY sl.created_at DESC
            LIMIT 5
        ) sub;

        result := result || jsonb_build_object(
            'recent_sessions', ledger_entries
        );

        -- Cross-project knowledge: SCOPED to same user_id
        IF array_length(handoff.keywords, 1) > 0 THEN
            result := result || jsonb_build_object(
                'cross_project_knowledge', (
                    SELECT COALESCE(jsonb_agg(
                        jsonb_build_object(
                            'project', sl2.project,
                            'summary', sl2.summary,
                            'keywords', to_jsonb(sl2.keywords),
                            'date', sl2.session_date,
                            'overlap_count', (
                                SELECT COUNT(*)
                                FROM unnest(sl2.keywords) k
                                WHERE k = ANY(handoff.keywords)
                            )
                        )
                    ), '[]'::jsonb)
                    FROM (
                        SELECT sl3.project, sl3.summary, sl3.keywords, sl3.session_date
                        FROM session_ledger sl3
                        WHERE sl3.project != p_project
                          AND sl3.user_id = p_user_id
                          AND sl3.keywords && handoff.keywords
                          AND sl3.archived_at IS NULL
                        ORDER BY (
                            SELECT COUNT(*)
                            FROM unnest(sl3.keywords) k
                            WHERE k = ANY(handoff.keywords)
                        ) DESC
                        LIMIT 3
                    ) sl2
                )
            );
        END IF;
    END IF;

    RETURN result;
END;
$$;

COMMENT ON FUNCTION get_session_context(TEXT, TEXT, TEXT, TEXT) IS
    'Progressive context loading with OCC, knowledge cache, multi-tenant + role isolation. '
    'v3.0: p_role scopes handoff queries by agent role. '
    'quick=keywords+todo+version, standard=+summary+decisions+cache, '
    'deep=+recent sessions+cross-project knowledge (same user only).';

-- 5b. save_handoff_with_version — add p_role parameter
CREATE OR REPLACE FUNCTION save_handoff_with_version(
  p_project TEXT,
  p_expected_version INT DEFAULT NULL,
  p_last_summary TEXT DEFAULT NULL,
  p_pending_todo TEXT[] DEFAULT NULL,
  p_active_decisions TEXT[] DEFAULT NULL,
  p_keywords TEXT[] DEFAULT NULL,
  p_key_context TEXT DEFAULT NULL,
  p_active_branch TEXT DEFAULT NULL,
  p_user_id TEXT DEFAULT 'default',
  p_role TEXT DEFAULT 'global'
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
  current_version INT;
  new_version INT;
BEGIN
  -- Scope to user + project + role
  SELECT version INTO current_version
  FROM session_handoffs
  WHERE project = p_project
    AND user_id = p_user_id
    AND role = p_role
  FOR UPDATE;

  -- CASE 1: No existing handoff → create
  IF NOT FOUND THEN
    INSERT INTO session_handoffs (
      project, user_id, role, last_summary, pending_todo, active_decisions,
      keywords, version, updated_at
    ) VALUES (
      p_project,
      p_user_id,
      p_role,
      p_last_summary,
      COALESCE(p_pending_todo, '{}'),
      COALESCE(p_active_decisions, '{}'),
      COALESCE(p_keywords, '{}'),
      1,
      NOW()
    );

    RETURN jsonb_build_object(
      'status', 'created',
      'project', p_project,
      'version', 1
    );
  END IF;

  -- CASE 2: Version mismatch → REJECT
  IF p_expected_version IS NOT NULL
     AND p_expected_version != current_version THEN
    RETURN jsonb_build_object(
      'status', 'conflict',
      'error', format(
        'Version conflict: you sent version %s but the current version is %s.',
        p_expected_version, current_version
      ),
      'current_version', current_version,
      'expected_version', p_expected_version
    );
  END IF;

  -- CASE 3: Version matches → UPDATE
  new_version := current_version + 1;

  UPDATE session_handoffs SET
    last_summary = COALESCE(p_last_summary, last_summary),
    pending_todo = COALESCE(p_pending_todo, pending_todo),
    active_decisions = COALESCE(p_active_decisions, active_decisions),
    keywords = COALESCE(p_keywords, keywords),
    version = new_version,
    updated_at = NOW()
  WHERE project = p_project
    AND user_id = p_user_id
    AND role = p_role;

  RETURN jsonb_build_object(
    'status', 'updated',
    'project', p_project,
    'version', new_version
  );
END;
$$;

COMMENT ON FUNCTION save_handoff_with_version(TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[], TEXT, TEXT, TEXT, TEXT) IS
  'OCC handoff save with multi-tenant + role isolation. '
  'v3.0: Scoped by user_id + project + role for Hivemind. '
  'Returns: created | updated | conflict.';

-- ─── 6. Row Level Security (RLS) ────────────────────────────────
-- All Prism tables use RLS to isolate users in multi-tenant mode.
--
-- IMPORTANT: These policies use a permissive "true" check because
-- the actual user_id filtering is done at the APPLICATION level
-- (in SupabaseStorage via PostgREST query filters like
-- `user_id=eq.{PRISM_USER_ID}`). This matches the established
-- pattern from 020_multi_tenant_rls.sql.
--
-- We do NOT use current_setting('app.current_user_id') because
-- Prism connects via Supabase REST API (PostgREST), which does
-- not set Postgres config variables — that would cause all queries
-- to silently return 0 rows.

ALTER TABLE agent_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "agent_registry_isolation" ON agent_registry;
CREATE POLICY "agent_registry_isolation" ON agent_registry
  FOR ALL
  USING (true)
  WITH CHECK (true);

DROP POLICY IF EXISTS "system_settings_isolation" ON system_settings;
CREATE POLICY "system_settings_isolation" ON system_settings
  FOR ALL
  USING (true)
  WITH CHECK (true);
