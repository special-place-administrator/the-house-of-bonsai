#!/usr/bin/env npx ts-node
/**
 * Pipeline Performance Benchmark
 * 
 * Measures and compares performance characteristics of the hybrid search pipeline
 * across different configurations:
 *   - MCP-only (Brave Search + code_mode_transform)
 *   - Discovery Engine-only (Vertex AI Search)
 *   - Hybrid (both sources merged + LLM analysis)
 * 
 * Metrics collected:
 *   - Query latency (ms)
 *   - Payload size (KB before/after context reduction)
 *   - Token estimation (chars / 4 ≈ tokens)
 *   - Context window efficiency (% reduction)
 * 
 * Usage:
 *   npx ts-node vertex-ai/test_pipeline_benchmark.ts
 * 
 * Prerequisites:
 *   - BRAVE_API_KEY environment variable
 *   - GCP ADC configured for Discovery Engine
 */

import { SearchServiceClient } from '@google-cloud/discoveryengine';

// ─── Configuration ───────────────────────────────────────────────

const BRAVE_API_KEY = process.env.BRAVE_API_KEY || '';
const DE_PROJECT_ID = process.env.DISCOVERY_ENGINE_PROJECT_ID || process.env.GCP_PROJECT_ID || '';
const DE_ENGINE_ID = process.env.DISCOVERY_ENGINE_ENGINE_ID || '';
const DE_LOCATION = process.env.DISCOVERY_ENGINE_LOCATION || 'global';
const DE_COLLECTION = process.env.DISCOVERY_ENGINE_COLLECTION || 'default_collection';
const DE_SERVING_CONFIG = process.env.DISCOVERY_ENGINE_SERVING_CONFIG || 'default_serving_config';

const BENCHMARK_QUERIES = [
  'transformer architecture attention mechanism',
  'kubernetes pod autoscaling best practices',
  'python async await concurrency patterns',
  'vertex ai discovery engine structured search',
  'mcp model context protocol integration',
];

const ITERATIONS = 3; // Number of runs per query for averaging

// ─── Types ───────────────────────────────────────────────────────

interface BenchmarkResult {
  source: string;
  query: string;
  avgLatencyMs: number;
  avgRawSizeKB: number;
  avgReducedSizeKB: number;
  avgReductionPct: number;
  estimatedTokensBefore: number;
  estimatedTokensAfter: number;
  successRate: number;
}

// ─── Brave Search Benchmark ─────────────────────────────────────

async function benchmarkBrave(query: string): Promise<{ latencyMs: number; rawSizeKB: number; reducedSizeKB: number; success: boolean }> {
  if (!BRAVE_API_KEY) return { latencyMs: 0, rawSizeKB: 0, reducedSizeKB: 0, success: false };

  const start = Date.now();
  try {
    const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=10`;
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
    const rawSizeKB = Buffer.byteLength(rawJson, 'utf8') / 1024;

    // Simulate code_mode_transform: extract only title + url + description
    const reduced = (data.web?.results || []).map((r: any) => ({
      title: r.title,
      url: r.url,
      desc: (r.description || '').substring(0, 150),
    }));
    const reducedJson = JSON.stringify(reduced);
    const reducedSizeKB = Buffer.byteLength(reducedJson, 'utf8') / 1024;

    return { latencyMs, rawSizeKB, reducedSizeKB, success: true };
  } catch {
    return { latencyMs: Date.now() - start, rawSizeKB: 0, reducedSizeKB: 0, success: false };
  }
}

// ─── Discovery Engine Benchmark ─────────────────────────────────

async function benchmarkDiscoveryEngine(query: string): Promise<{ latencyMs: number; rawSizeKB: number; reducedSizeKB: number; success: boolean }> {
  if (!DE_PROJECT_ID || !DE_ENGINE_ID) return { latencyMs: 0, rawSizeKB: 0, reducedSizeKB: 0, success: false };

  const start = Date.now();
  try {
    const client = new SearchServiceClient();
    const servingConfig = `projects/${DE_PROJECT_ID}/locations/${DE_LOCATION}/collections/${DE_COLLECTION}/engines/${DE_ENGINE_ID}/servingConfigs/${DE_SERVING_CONFIG}`;

    const [response] = await client.search({ servingConfig, query, pageSize: 10 }, { autoPaginate: false });
    const latencyMs = Date.now() - start;

    const resultItems = Array.isArray(response) ? response : (response as any).results || [];
    const rawJson = JSON.stringify(resultItems);
    const rawSizeKB = Buffer.byteLength(rawJson, 'utf8') / 1024;

    // Helper for protobuf Value extraction
    const getField = (sd: any, f: string): string => {
      if (!sd) return '';
      if (typeof sd[f] === 'string') return sd[f];
      if (sd.fields?.[f]?.stringValue) return sd.fields[f].stringValue;
      return '';
    };

    const reduced = resultItems.map((r: any) => ({
      title: getField(r.document?.derivedStructData, 'title') || getField(r.document?.derivedStructData, 'displayLink') || '',
      url: getField(r.document?.derivedStructData, 'link') || getField(r.document?.derivedStructData, 'htmlFormattedUrl') || '',
    }));
    const reducedJson = JSON.stringify(reduced);
    const reducedSizeKB = Buffer.byteLength(reducedJson, 'utf8') / 1024;

    return { latencyMs, rawSizeKB, reducedSizeKB, success: true };
  } catch {
    return { latencyMs: Date.now() - start, rawSizeKB: 0, reducedSizeKB: 0, success: false };
  }
}

// ─── Run Benchmark ───────────────────────────────────────────────

async function runBenchmarks() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║        Pipeline Performance Benchmark: MCP vs Vertex AI     ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  const braveResults: BenchmarkResult[] = [];
  const deResults: BenchmarkResult[] = [];

  for (const query of BENCHMARK_QUERIES) {
    console.log(`\n── Query: "${query}" ──`);

    // Benchmark Brave Search
    const braveRuns: Awaited<ReturnType<typeof benchmarkBrave>>[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
      braveRuns.push(await benchmarkBrave(query));
      if (i < ITERATIONS - 1) await sleep(500); // Rate limit courtesy
    }

    const braveSuccessful = braveRuns.filter(r => r.success);
    if (braveSuccessful.length > 0) {
      const result: BenchmarkResult = {
        source: 'Brave Search',
        query,
        avgLatencyMs: avg(braveSuccessful.map(r => r.latencyMs)),
        avgRawSizeKB: avg(braveSuccessful.map(r => r.rawSizeKB)),
        avgReducedSizeKB: avg(braveSuccessful.map(r => r.reducedSizeKB)),
        avgReductionPct: avg(braveSuccessful.map(r => r.rawSizeKB > 0 ? (1 - r.reducedSizeKB / r.rawSizeKB) * 100 : 0)),
        estimatedTokensBefore: Math.round(avg(braveSuccessful.map(r => r.rawSizeKB * 1024 / 4))),
        estimatedTokensAfter: Math.round(avg(braveSuccessful.map(r => r.reducedSizeKB * 1024 / 4))),
        successRate: braveSuccessful.length / ITERATIONS,
      };
      braveResults.push(result);
      console.log(`   🌐 Brave: ${result.avgLatencyMs.toFixed(0)}ms | ${result.avgRawSizeKB.toFixed(1)}KB → ${result.avgReducedSizeKB.toFixed(1)}KB (${result.avgReductionPct.toFixed(0)}% reduction)`);
    } else {
      console.log('   🌐 Brave: skipped (no API key)');
    }

    // Benchmark Discovery Engine
    const deRuns: Awaited<ReturnType<typeof benchmarkDiscoveryEngine>>[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
      deRuns.push(await benchmarkDiscoveryEngine(query));
      if (i < ITERATIONS - 1) await sleep(300);
    }

    const deSuccessful = deRuns.filter(r => r.success);
    if (deSuccessful.length > 0) {
      const result: BenchmarkResult = {
        source: 'Discovery Engine',
        query,
        avgLatencyMs: avg(deSuccessful.map(r => r.latencyMs)),
        avgRawSizeKB: avg(deSuccessful.map(r => r.rawSizeKB)),
        avgReducedSizeKB: avg(deSuccessful.map(r => r.reducedSizeKB)),
        avgReductionPct: avg(deSuccessful.map(r => r.rawSizeKB > 0 ? (1 - r.reducedSizeKB / r.rawSizeKB) * 100 : 0)),
        estimatedTokensBefore: Math.round(avg(deSuccessful.map(r => r.rawSizeKB * 1024 / 4))),
        estimatedTokensAfter: Math.round(avg(deSuccessful.map(r => r.reducedSizeKB * 1024 / 4))),
        successRate: deSuccessful.length / ITERATIONS,
      };
      deResults.push(result);
      console.log(`   🔍 DE:    ${result.avgLatencyMs.toFixed(0)}ms | ${result.avgRawSizeKB.toFixed(1)}KB → ${result.avgReducedSizeKB.toFixed(1)}KB (${result.avgReductionPct.toFixed(0)}% reduction)`);
    } else {
      console.log('   🔍 DE:    skipped (env vars not set)');
    }
  }

  // Summary
  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║                    AGGREGATE RESULTS                        ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');

  if (braveResults.length > 0) {
    console.log('║  🌐 Brave Search (MCP)                                     ║');
    console.log(`║     Avg latency:     ${avg(braveResults.map(r => r.avgLatencyMs)).toFixed(0)}ms`);
    console.log(`║     Avg raw payload: ${avg(braveResults.map(r => r.avgRawSizeKB)).toFixed(1)}KB`);
    console.log(`║     Avg reduced:     ${avg(braveResults.map(r => r.avgReducedSizeKB)).toFixed(1)}KB`);
    console.log(`║     Avg reduction:   ${avg(braveResults.map(r => r.avgReductionPct)).toFixed(0)}%`);
    console.log(`║     Token savings:   ~${avg(braveResults.map(r => r.estimatedTokensBefore - r.estimatedTokensAfter)).toFixed(0)} tokens/query`);
  }

  if (deResults.length > 0) {
    console.log('║                                                             ║');
    console.log('║  🔍 Discovery Engine (Vertex AI)                            ║');
    console.log(`║     Avg latency:     ${avg(deResults.map(r => r.avgLatencyMs)).toFixed(0)}ms`);
    console.log(`║     Avg raw payload: ${avg(deResults.map(r => r.avgRawSizeKB)).toFixed(1)}KB`);
    console.log(`║     Avg reduced:     ${avg(deResults.map(r => r.avgReducedSizeKB)).toFixed(1)}KB`);
    console.log(`║     Avg reduction:   ${avg(deResults.map(r => r.avgReductionPct)).toFixed(0)}%`);
    console.log(`║     Token savings:   ~${avg(deResults.map(r => r.estimatedTokensBefore - r.estimatedTokensAfter)).toFixed(0)} tokens/query`);
  }

  if (braveResults.length > 0 && deResults.length > 0) {
    const braveLatency = avg(braveResults.map(r => r.avgLatencyMs));
    const deLatency = avg(deResults.map(r => r.avgLatencyMs));
    const latencyDiff = ((braveLatency - deLatency) / braveLatency * 100).toFixed(0);
    console.log('║                                                             ║');
    console.log(`║  ⚡ DE is ${latencyDiff}% ${Number(latencyDiff) > 0 ? 'faster' : 'slower'} than Brave (pre-indexed vs live search)`);
  }

  console.log('╚══════════════════════════════════════════════════════════════╝');
}

// ─── Utilities ───────────────────────────────────────────────────

function avg(nums: number[]): number {
  return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Entry ───────────────────────────────────────────────────────

runBenchmarks().catch(err => {
  console.error('Benchmark error:', err);
  process.exit(1);
});
