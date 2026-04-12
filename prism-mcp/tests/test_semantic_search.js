#!/usr/bin/env node
/**
 * Semantic Search Test Suite — Prism MCP v2.1
 *
 * Tests the embedding-based semantic search pipeline end-to-end:
 *   1. Embedding API (unit): dimension, truncation, empty text
 *   2. Semantic Search Lifecycle (integration): save → embed → search → cleanup
 *
 * Usage:
 *   node tests/test_semantic_search.js                    # unit tests only (needs GOOGLE_API_KEY)
 *   node tests/test_semantic_search.js --integration      # + Supabase integration tests
 *
 * Env vars required for unit tests:
 *   GOOGLE_API_KEY   — Gemini API key for embedding generation
 *
 * Additional env vars for integration tests:
 *   SUPABASE_URL     — Supabase project URL
 *   SUPABASE_KEY     — Supabase service-role key (bypasses RLS)
 *
 * Exit code 0 = all passed, 1 = failures
 */

import { generateEmbedding } from "../dist/utils/embeddingApi.js";

const INTEGRATION = process.argv.includes("--integration");
let passed = 0;
let failed = 0;
let skipped = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  ✅ ${message}`);
    passed++;
  } else {
    console.error(`  ❌ FAIL: ${message}`);
    failed++;
  }
}

function skip(message) {
  console.log(`  ⏭️  SKIP: ${message}`);
  skipped++;
}

// ─── Banner ───────────────────────────────────────────────────

console.log("\n═══════════════════════════════════════════════════════");
console.log("  🧪 Semantic Search Test Suite — Prism MCP v2.1");
console.log("═══════════════════════════════════════════════════════\n");

// ─── Unit Tests: Embedding API ────────────────────────────────

console.log("1️⃣  Embedding API (Unit Tests)");
console.log("─────────────────────────────────────");

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

if (!GOOGLE_API_KEY) {
  console.error("  ⚠️  GOOGLE_API_KEY not set — skipping embedding API tests");
  skip("generateEmbedding returns 768 dimensions");
  skip("generateEmbedding handles long text with truncation");
  skip("generateEmbedding rejects empty text");
  skip("Embeddings of similar texts have high cosine similarity");
  skip("Embeddings of unrelated texts have low cosine similarity");
} else {
  // Test 1: Basic embedding generation returns 768-dimension vector
  try {
    console.log("\n  📐 Test 1: Generating embedding for sample text...");
    const start = performance.now();
    const embedding = await generateEmbedding(
      "Implemented Stripe webhook handler with idempotency keys for payment processing"
    );
    const elapsed = performance.now() - start;

    assert(Array.isArray(embedding), "generateEmbedding returns an array");
    assert(embedding.length === 768, `Embedding has 768 dimensions (got ${embedding.length})`);
    assert(
      embedding.every(v => typeof v === "number" && !isNaN(v)),
      "All embedding values are valid numbers"
    );
    assert(
      embedding.some(v => v !== 0),
      "Embedding is not a zero vector"
    );
    console.log(`  📊 Embedding generated in ${elapsed.toFixed(0)}ms`);
  } catch (err) {
    console.error(`  ❌ Embedding generation failed: ${err.message}`);
    failed += 4;
  }

  // Test 2: Truncation guard for long text
  try {
    console.log("\n  ✂️  Test 2: Truncation guard for long text...");
    const longText = "x ".repeat(5000); // 10,000 chars
    const embedding = await generateEmbedding(longText);
    assert(embedding.length === 768, `Long text embedding has 768 dims (truncated safely)`);
  } catch (err) {
    console.error(`  ❌ Long text embedding failed: ${err.message}`);
    failed++;
  }

  // Test 3: Empty text rejection
  try {
    console.log("\n  🚫 Test 3: Empty text rejection...");
    await generateEmbedding("");
    assert(false, "Should have thrown for empty text");
  } catch (err) {
    assert(
      err.message.includes("empty"),
      `Rejects empty text with proper error: "${err.message.substring(0, 60)}"`
    );
  }

  // Test 4: Whitespace-only rejection
  try {
    console.log("\n  🚫 Test 4: Whitespace-only rejection...");
    await generateEmbedding("   \n\t  ");
    assert(false, "Should have thrown for whitespace-only text");
  } catch (err) {
    assert(
      err.message.includes("empty"),
      `Rejects whitespace-only text with proper error`
    );
  }

  // Test 5: Semantic similarity — similar texts should have high cosine similarity
  try {
    console.log("\n  🔗 Test 5: Semantic similarity between related texts...");
    const emb1 = await generateEmbedding(
      "Fixed authentication bug by rotating API keys"
    );
    const emb2 = await generateEmbedding(
      "Resolved auth failure by updating credentials"
    );
    const emb3 = await generateEmbedding(
      "Implemented payment gateway with Stripe webhooks"
    );

    // Cosine similarity helper
    function cosineSimilarity(a, b) {
      let dotProduct = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    const simRelated = cosineSimilarity(emb1, emb2);
    const simUnrelated = cosineSimilarity(emb1, emb3);

    console.log(`  📊 Related text similarity: ${(simRelated * 100).toFixed(1)}%`);
    console.log(`  📊 Unrelated text similarity: ${(simUnrelated * 100).toFixed(1)}%`);

    assert(
      simRelated > 0.7,
      `Related texts have high similarity (${(simRelated * 100).toFixed(1)}% > 70%)`
    );
    assert(
      simRelated > simUnrelated,
      `Related texts are more similar than unrelated (${(simRelated * 100).toFixed(1)}% > ${(simUnrelated * 100).toFixed(1)}%)`
    );
  } catch (err) {
    console.error(`  ❌ Similarity test failed: ${err.message}`);
    failed += 2;
  }
}

// ─── Integration Tests: Supabase Semantic Search ──────────────

console.log("\n\n2️⃣  Semantic Search Integration (Supabase)");
console.log("─────────────────────────────────────");

if (!INTEGRATION) {
  skip("Supabase semantic search lifecycle (run with --integration)");
  skip("semantic_search_ledger RPC returns matching entries");
  skip("semantic_search_ledger respects similarity threshold");
  skip("Cleanup test data");
} else if (!GOOGLE_API_KEY) {
  skip("Integration tests require GOOGLE_API_KEY");
} else {
  const SUPABASE_URL = process.env.SUPABASE_URL;
  const SUPABASE_KEY = process.env.SUPABASE_KEY;

  if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error("  ❌ SUPABASE_URL and SUPABASE_KEY must be set for integration tests");
    process.exit(1);
  }

  const headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": `Bearer ${SUPABASE_KEY}`,
    "Content-Type": "application/json",
    "Prefer": "return=representation",
  };

  const rpc = async (fn, body) => {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`RPC ${fn} failed (${res.status}): ${text}`);
    }
    return res.json();
  };

  const post = async (table, body) => {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`POST ${table} failed (${res.status}): ${text}`);
    }
    return res.json();
  };

  const patch = async (table, body, params) => {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${qs}`, {
      method: "PATCH",
      headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`PATCH ${table} failed (${res.status}): ${text}`);
    }
    return res.json();
  };

  const del = async (table, params) => {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${qs}`, {
      method: "DELETE",
      headers,
    });
    if (res.status === 204) return [];
    return res.json();
  };

  const TEST_PROJECT = "__test_semantic_search__";
  const TEST_USER = "__test_user_semantic__";

  try {
    // Cleanup from previous runs
    await del("session_ledger", { project: `eq.${TEST_PROJECT}` });

    // ── Test A: Save ledger entry ────────────────────────────
    console.log("\n  📝 Test A: Save a ledger entry with known text...");
    const testSummary = "Fixed critical authentication failure by rotating expired API credentials and updating the OAuth token refresh flow";
    const saveResult = await post("session_ledger", {
      project: TEST_PROJECT,
      conversation_id: "test-semantic-001",
      user_id: TEST_USER,
      summary: testSummary,
      keywords: ["authentication", "oauth", "api", "credentials", "cat:security"],
      decisions: ["Rotated API keys", "Updated OAuth refresh flow"],
      files_changed: ["src/auth/oauth.ts", "src/config/keys.ts"],
    });
    assert(!saveResult.error, `Saved test ledger entry: ${saveResult.error || "OK"}`);
    const entryId = Array.isArray(saveResult) ? saveResult[0]?.id : saveResult?.id;
    assert(!!entryId, `Got entry ID: ${entryId ? entryId.substring(0, 8) + "..." : "MISSING"}`);

    // ── Test B: Generate and store embedding ─────────────────
    console.log("\n  🧮 Test B: Generate embedding and patch into entry...");
    const embeddingText = [testSummary, "Rotated API keys", "Updated OAuth refresh flow"].join(" | ");
    const embedding = await generateEmbedding(embeddingText);
    assert(embedding.length === 768, `Generated 768-dim embedding`);

    // Patch embedding into the entry
    const embeddingJson = JSON.stringify(embedding);
    await patch("session_ledger", { embedding: embeddingJson }, { id: `eq.${entryId}` });
    console.log("  ✅ Embedding patched into ledger entry");

    // ── Test C: Semantic search with similar query ────────────
    console.log("\n  🔍 Test C: Semantic search with a similar query...");
    const searchQuery = "authentication bug with expired credentials";
    const queryEmbedding = await generateEmbedding(searchQuery);
    const queryEmbeddingJson = JSON.stringify(queryEmbedding);

    const searchResults = await rpc("semantic_search_ledger", {
      p_query_embedding: queryEmbeddingJson,
      p_project: TEST_PROJECT,
      p_limit: 5,
      p_similarity_threshold: 0.5,
      p_user_id: TEST_USER,
    });

    assert(Array.isArray(searchResults), "RPC returns an array");
    assert(searchResults.length > 0, `Found ${searchResults.length} semantically similar entries`);

    if (searchResults.length > 0) {
      const topResult = searchResults[0];
      assert(
        topResult.summary?.includes("authentication"),
        `Top result matches: "${topResult.summary?.substring(0, 60)}..."`
      );
      assert(
        typeof topResult.similarity === "number" && topResult.similarity > 0.5,
        `Similarity score is above threshold: ${(topResult.similarity * 100).toFixed(1)}%`
      );
      console.log(`  📊 Top similarity: ${(topResult.similarity * 100).toFixed(1)}%`);
    }

    // ── Test D: Search with unrelated query (should return 0) ─
    console.log("\n  🚫 Test D: Search with unrelated query (expect 0 results)...");
    const unrelatedEmbedding = await generateEmbedding(
      "cooking recipe for chocolate cake with vanilla frosting"
    );
    const unrelatedResults = await rpc("semantic_search_ledger", {
      p_query_embedding: JSON.stringify(unrelatedEmbedding),
      p_project: TEST_PROJECT,
      p_limit: 5,
      p_similarity_threshold: 0.8,  // High threshold to ensure unrelated content is filtered
      p_user_id: TEST_USER,
    });

    assert(
      unrelatedResults.length === 0,
      `Unrelated query returns 0 results at threshold 0.8 (got ${unrelatedResults.length})`
    );

    // ── Test E: Search with different user_id (multi-tenant isolation) ──
    console.log("\n  🔒 Test E: Multi-tenant isolation (different user_id)...");
    const otherUserResults = await rpc("semantic_search_ledger", {
      p_query_embedding: queryEmbeddingJson,
      p_project: TEST_PROJECT,
      p_limit: 5,
      p_similarity_threshold: 0.3,
      p_user_id: "some_other_user",
    });

    assert(
      otherUserResults.length === 0,
      `Different user_id returns 0 results (tenant isolation works)`
    );

    // ── Cleanup ──────────────────────────────────────────────
    console.log("\n  🧹 Cleaning up test data...");
    await del("session_ledger", { project: `eq.${TEST_PROJECT}` });
    console.log("  ✅ Cleanup complete");

  } catch (err) {
    console.error(`  ❌ Integration test error: ${err.message}`);
    console.error(`  Stack: ${err.stack}`);
    failed++;
    // Cleanup on error
    try {
      await del("session_ledger", { project: `eq.${TEST_PROJECT}` });
    } catch (_) {}
  }
}

// ─── Summary ──────────────────────────────────────────────────

console.log("\n═══════════════════════════════════════════════════════");
console.log(`  📊 Results: ${passed} passed, ${failed} failed, ${skipped} skipped`);
console.log("═══════════════════════════════════════════════════════\n");

process.exit(failed > 0 ? 1 : 0);
