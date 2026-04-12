#!/usr/bin/env npx ts-node
/**
 * Hybrid Search Pipeline Test
 * 
 * Validates the combined MCP + Vertex AI Discovery Engine search pipeline:
 *   1. Brave Search (real-time web) via MCP server
 *   2. Discovery Engine (curated enterprise index) via Vertex AI
 *   3. code_mode_transform (context reduction via sandboxed JS)
 *   4. Gemini analysis (LLM post-processing of merged results)
 * 
 * Usage:
 *   npx ts-node vertex-ai/test_hybrid_search_pipeline.ts
 * 
 * Prerequisites:
 *   - BRAVE_API_KEY environment variable
 *   - GCP ADC configured (gcloud auth application-default login)
 *   - DISCOVERY_ENGINE_* environment variables set
 *   - GEMINI_API_KEY or GOOGLE_API_KEY environment variable
 */

import { SearchServiceClient } from '@google-cloud/discoveryengine';
import { GoogleGenerativeAI } from '@google/generative-ai';

// ─── Configuration ───────────────────────────────────────────────

const BRAVE_API_KEY = process.env.BRAVE_API_KEY || '';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '';

const DE_PROJECT_ID = process.env.DISCOVERY_ENGINE_PROJECT_ID || process.env.GCP_PROJECT_ID || '';
const DE_LOCATION = process.env.DISCOVERY_ENGINE_LOCATION || 'global';
const DE_COLLECTION = process.env.DISCOVERY_ENGINE_COLLECTION || 'default_collection';
const DE_ENGINE_ID = process.env.DISCOVERY_ENGINE_ENGINE_ID || '';
const DE_SERVING_CONFIG = process.env.DISCOVERY_ENGINE_SERVING_CONFIG || 'default_serving_config';

const TEST_QUERY = 'machine learning model optimization techniques';

// ─── Interfaces ──────────────────────────────────────────────────

interface SearchResult {
  source: 'brave' | 'discovery_engine';
  title: string;
  url: string;
  snippet: string;
}

interface PipelineMetrics {
  stage: string;
  latencyMs: number;
  inputSizeKB: number;
  outputSizeKB: number;
  reductionPct: number;
}

// ─── Stage 1: Brave Web Search ───────────────────────────────────

async function braveWebSearch(query: string, count: number = 5): Promise<{ results: SearchResult[]; rawSize: number; latencyMs: number }> {
  if (!BRAVE_API_KEY) {
    console.log('  ⚠️  BRAVE_API_KEY not set — skipping Brave Search stage');
    return { results: [], rawSize: 0, latencyMs: 0 };
  }

  const start = Date.now();
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}`;

  const response = await fetch(url, {
    headers: {
      'Accept': 'application/json',
      'Accept-Encoding': 'gzip',
      'X-Subscription-Token': BRAVE_API_KEY,
    },
  });

  const data = await response.json();
  const latencyMs = Date.now() - start;
  const rawJson = JSON.stringify(data);
  const rawSize = Buffer.byteLength(rawJson, 'utf8');

  const results: SearchResult[] = (data.web?.results || []).map((r: any) => ({
    source: 'brave' as const,
    title: r.title || '',
    url: r.url || '',
    snippet: (r.description || '').substring(0, 200),
  }));

  return { results, rawSize, latencyMs };
}

// ─── Stage 2: Discovery Engine Search ────────────────────────────

async function discoveryEngineSearch(query: string, pageSize: number = 5): Promise<{ results: SearchResult[]; rawSize: number; latencyMs: number }> {
  if (!DE_PROJECT_ID || !DE_ENGINE_ID) {
    console.log('  ⚠️  Discovery Engine env vars not set — skipping DE stage');
    return { results: [], rawSize: 0, latencyMs: 0 };
  }

  const start = Date.now();
  const client = new SearchServiceClient();

  const servingConfig = `projects/${DE_PROJECT_ID}/locations/${DE_LOCATION}/collections/${DE_COLLECTION}/engines/${DE_ENGINE_ID}/servingConfigs/${DE_SERVING_CONFIG}`;

  try {
    const [response] = await client.search({
      servingConfig,
      query,
      pageSize,
    }, { autoPaginate: false });

    const latencyMs = Date.now() - start;
    
    // With autoPaginate: false, response is an array of ISearchResult
    const resultItems = Array.isArray(response) ? response : (response as any).results || [];
    const rawJson = JSON.stringify(resultItems);
    const rawSize = Buffer.byteLength(rawJson, 'utf8');

    // Helper to extract string from protobuf Value
    const getField = (structData: any, field: string): string => {
      if (!structData) return '';
      // Direct access (already decoded)
      if (typeof structData[field] === 'string') return structData[field];
      // Protobuf fields format
      if (structData.fields?.[field]?.stringValue) return structData.fields[field].stringValue;
      return '';
    };

    const results: SearchResult[] = resultItems.map((r: any) => ({
      source: 'discovery_engine' as const,
      title: getField(r.document?.derivedStructData, 'title') || 
             getField(r.document?.derivedStructData, 'displayLink') || 
             r.document?.name?.split('/').pop() || 'Untitled',
      url: getField(r.document?.derivedStructData, 'link') || 
           getField(r.document?.derivedStructData, 'htmlFormattedUrl') || '',
      snippet: getField(r.document?.derivedStructData, 'snippet') || 
               getField(r.document?.derivedStructData, 'htmlSnippet') || '',
    }));

    return { results, rawSize, latencyMs };
  } catch (error: any) {
    console.log(`  ⚠️  Discovery Engine error: ${error.message}`);
    return { results: [], rawSize: 0, latencyMs: Date.now() - start };
  }
}

// ─── Stage 3: Context Reduction (simulates code_mode_transform) ──

function contextReduction(braveResults: SearchResult[], deResults: SearchResult[]): { merged: SearchResult[]; metrics: PipelineMetrics } {
  const start = Date.now();

  // Merge and deduplicate by URL
  const allResults = [...braveResults, ...deResults];
  const seen = new Set<string>();
  const merged: SearchResult[] = [];

  for (const result of allResults) {
    const key = result.url.toLowerCase().replace(/\/$/, '');
    if (!seen.has(key) && result.url) {
      seen.add(key);
      merged.push(result);
    }
  }

  const inputJson = JSON.stringify(allResults);
  const outputJson = JSON.stringify(merged.map(r => ({
    source: r.source,
    title: r.title,
    url: r.url,
  })));

  const inputSizeKB = Buffer.byteLength(inputJson, 'utf8') / 1024;
  const outputSizeKB = Buffer.byteLength(outputJson, 'utf8') / 1024;

  return {
    merged,
    metrics: {
      stage: 'context_reduction',
      latencyMs: Date.now() - start,
      inputSizeKB,
      outputSizeKB,
      reductionPct: inputSizeKB > 0 ? Number((100 - (outputSizeKB / inputSizeKB) * 100).toFixed(1)) : 0,
    },
  };
}

// ─── Stage 4: Gemini Analysis ────────────────────────────────────

async function geminiAnalysis(mergedResults: SearchResult[], query: string): Promise<{ analysis: string; latencyMs: number }> {
  if (!GEMINI_API_KEY) {
    console.log('  ⚠️  GEMINI_API_KEY not set — skipping Gemini analysis stage');
    return { analysis: '[Skipped - no API key]', latencyMs: 0 };
  }

  const start = Date.now();
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

  const sourceSummary = mergedResults.map((r, i) =>
    `[${i + 1}] (${r.source}) ${r.title}\n    URL: ${r.url}`
  ).join('\n');

  const prompt = `Given the following search results from a hybrid pipeline (web search + enterprise Discovery Engine) for the query "${query}", provide a brief analytical summary highlighting the most relevant findings and how the two sources complement each other:\n\n${sourceSummary}\n\nKeep the summary concise (3-5 sentences).`;

  try {
    const result = await model.generateContent(prompt);
    const text = result.response.text();
    return { analysis: text, latencyMs: Date.now() - start };
  } catch (error: any) {
    return { analysis: `[Error: ${error.message}]`, latencyMs: Date.now() - start };
  }
}

// ─── Main Pipeline ───────────────────────────────────────────────

async function runHybridPipeline() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║     Hybrid Search Pipeline Test: MCP + Vertex AI       ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  console.log(`Query: "${TEST_QUERY}"\n`);

  const metrics: PipelineMetrics[] = [];

  // Stage 1: Brave Search
  console.log('── Stage 1: Brave Web Search (MCP) ──');
  const brave = await braveWebSearch(TEST_QUERY);
  console.log(`   Results: ${brave.results.length} | Latency: ${brave.latencyMs}ms | Raw: ${(brave.rawSize / 1024).toFixed(1)}KB`);
  metrics.push({
    stage: 'brave_web_search',
    latencyMs: brave.latencyMs,
    inputSizeKB: 0,
    outputSizeKB: brave.rawSize / 1024,
    reductionPct: 0,
  });

  // Stage 2: Discovery Engine
  console.log('\n── Stage 2: Vertex AI Discovery Engine ──');
  const de = await discoveryEngineSearch(TEST_QUERY);
  console.log(`   Results: ${de.results.length} | Latency: ${de.latencyMs}ms | Raw: ${(de.rawSize / 1024).toFixed(1)}KB`);
  metrics.push({
    stage: 'discovery_engine',
    latencyMs: de.latencyMs,
    inputSizeKB: 0,
    outputSizeKB: de.rawSize / 1024,
    reductionPct: 0,
  });

  // Stage 3: Context Reduction
  console.log('\n── Stage 3: Context Reduction (code_mode_transform) ──');
  const { merged, metrics: reductionMetrics } = contextReduction(brave.results, de.results);
  console.log(`   Merged: ${merged.length} unique results (from ${brave.results.length + de.results.length} total)`);
  console.log(`   Reduction: ${reductionMetrics.inputSizeKB.toFixed(1)}KB → ${reductionMetrics.outputSizeKB.toFixed(1)}KB (${reductionMetrics.reductionPct}%)`);
  metrics.push(reductionMetrics);

  // Stage 4: Gemini Analysis
  console.log('\n── Stage 4: Gemini LLM Analysis ──');
  const analysis = await geminiAnalysis(merged, TEST_QUERY);
  console.log(`   Latency: ${analysis.latencyMs}ms`);
  console.log(`   Analysis:\n${analysis.analysis.split('\n').map(l => `   ${l}`).join('\n')}`);
  metrics.push({
    stage: 'gemini_analysis',
    latencyMs: analysis.latencyMs,
    inputSizeKB: 0,
    outputSizeKB: Buffer.byteLength(analysis.analysis, 'utf8') / 1024,
    reductionPct: 0,
  });

  // Summary
  const totalLatency = metrics.reduce((sum, m) => sum + m.latencyMs, 0);

  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║                   PIPELINE SUMMARY                      ║');
  console.log('╠══════════════════════════════════════════════════════════╣');
  console.log(`║  Total latency: ${totalLatency}ms                       `);
  console.log(`║  Sources queried: ${[brave.results.length > 0 ? 'Brave' : null, de.results.length > 0 ? 'Discovery Engine' : null].filter(Boolean).join(' + ') || 'None'}`);
  console.log(`║  Unique results: ${merged.length}                       `);
  console.log(`║  Context reduction: ${reductionMetrics.reductionPct}%    `);
  console.log('╚══════════════════════════════════════════════════════════╝');

  // Detailed results table
  console.log('\n── Merged Results ──');
  merged.forEach((r, i) => {
    console.log(`  [${i + 1}] (${r.source === 'brave' ? '🌐 Brave' : '🔍 DE'}) ${r.title}`);
    console.log(`      ${r.url}`);
  });

  // Pass/fail
  const passed = brave.results.length > 0 || de.results.length > 0;
  console.log(`\n${passed ? '✅ PIPELINE TEST PASSED' : '⚠️  No results from either source — check API keys and env vars'}`);

  return passed;
}

runHybridPipeline()
  .then(passed => process.exit(passed ? 0 : 1))
  .catch(err => {
    console.error('Pipeline error:', err);
    process.exit(1);
  });
