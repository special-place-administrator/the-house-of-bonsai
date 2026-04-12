#!/usr/bin/env npx ts-node
/**
 * Real-World Comparison: Brave Search Only vs Hybrid Pipeline
 * 
 * Demonstrates WHY the hybrid pipeline matters by running the same
 * domain-specific queries through both approaches and comparing:
 *   - Result coverage (unique sources found)
 *   - Source quality (academic vs generic web)
 *   - Information depth (domain-specific findings)
 *   - Token efficiency (context sent to LLM)
 * 
 * Usage:
 *   npx ts-node vertex-ai/test_realworld_comparison.ts
 */

import { SearchServiceClient } from '@google-cloud/discoveryengine';

// ─── Configuration ───────────────────────────────────────────────

const BRAVE_API_KEY = process.env.BRAVE_API_KEY || '';
const DE_PROJECT_ID = process.env.DISCOVERY_ENGINE_PROJECT_ID || process.env.GCP_PROJECT_ID || '';
const DE_ENGINE_ID = process.env.DISCOVERY_ENGINE_ENGINE_ID || '';
const DE_LOCATION = process.env.DISCOVERY_ENGINE_LOCATION || 'global';
const DE_COLLECTION = process.env.DISCOVERY_ENGINE_COLLECTION || 'default_collection';
const DE_SERVING_CONFIG = process.env.DISCOVERY_ENGINE_SERVING_CONFIG || 'default_serving_config';

// Real-world queries a developer or researcher would actually ask
const REAL_WORLD_QUERIES = [
  {
    query: 'reinforcement learning from human feedback RLHF implementation',
    context: 'AI Engineer researching LLM alignment techniques',
  },
  {
    query: 'transformer model quantization INT8 inference optimization',
    context: 'ML Engineer deploying models to production',
  },
  {
    query: 'retrieval augmented generation RAG vector database architecture',
    context: 'Developer building an enterprise RAG pipeline',
  },
];

// ─── Types ───────────────────────────────────────────────────────

interface Result {
  title: string;
  url: string;
  domain: string;
  isAcademic: boolean;
}

interface ComparisonResult {
  query: string;
  context: string;
  braveOnly: {
    results: Result[];
    latencyMs: number;
    rawSizeKB: number;
    reducedSizeKB: number;
    academicCount: number;
  };
  hybrid: {
    results: Result[];
    latencyMs: number;
    rawSizeKB: number;
    reducedSizeKB: number;
    academicCount: number;
    uniqueFromDE: number;
    overlapCount: number;
  };
}

// ─── Helpers ─────────────────────────────────────────────────────

const ACADEMIC_DOMAINS = ['arxiv.org', 'scholar.google', 'ieee.org', 'acm.org', 'springer.com', 'nature.com', 'sciencedirect.com', 'openreview.net', 'proceedings.mlr.press', 'aclweb.org', 'neurips.cc'];

function extractDomain(url: string): string {
  try { return new URL(url).hostname.replace('www.', ''); } catch { return url; }
}

function isAcademicSource(url: string): boolean {
  const domain = extractDomain(url);
  return ACADEMIC_DOMAINS.some(ad => domain.includes(ad));
}

function estimateTokens(sizeKB: number): number {
  return Math.round(sizeKB * 1024 / 4);
}

// ─── Brave Search ────────────────────────────────────────────────

async function searchBrave(query: string): Promise<{ results: Result[]; latencyMs: number; rawSizeKB: number }> {
  if (!BRAVE_API_KEY) return { results: [], latencyMs: 0, rawSizeKB: 0 };

  const start = Date.now();
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=10`;
  const response = await fetch(url, {
    headers: { 'Accept': 'application/json', 'X-Subscription-Token': BRAVE_API_KEY },
  });
  const data = await response.json();
  const latencyMs = Date.now() - start;
  const rawSizeKB = Buffer.byteLength(JSON.stringify(data), 'utf8') / 1024;

  const results: Result[] = (data.web?.results || []).map((r: any) => ({
    title: r.title || '',
    url: r.url || '',
    domain: extractDomain(r.url || ''),
    isAcademic: isAcademicSource(r.url || ''),
  }));

  return { results, latencyMs, rawSizeKB };
}

// ─── Discovery Engine Search ─────────────────────────────────────

async function searchDE(query: string): Promise<{ results: Result[]; latencyMs: number; rawSizeKB: number }> {
  if (!DE_PROJECT_ID || !DE_ENGINE_ID) return { results: [], latencyMs: 0, rawSizeKB: 0 };

  const start = Date.now();
  const client = new SearchServiceClient();
  const servingConfig = `projects/${DE_PROJECT_ID}/locations/${DE_LOCATION}/collections/${DE_COLLECTION}/engines/${DE_ENGINE_ID}/servingConfigs/${DE_SERVING_CONFIG}`;

  try {
    const [response] = await client.search({ servingConfig, query, pageSize: 10 }, { autoPaginate: false });
    const latencyMs = Date.now() - start;

    const resultItems = Array.isArray(response) ? response : [];
    const rawSizeKB = Buffer.byteLength(JSON.stringify(resultItems), 'utf8') / 1024;

    const getField = (sd: any, f: string): string => {
      if (!sd) return '';
      if (typeof sd[f] === 'string') return sd[f];
      if (sd.fields?.[f]?.stringValue) return sd.fields[f].stringValue;
      return '';
    };

    const results: Result[] = resultItems.map((r: any) => {
      const urlVal = getField(r.document?.derivedStructData, 'link') ||
                     getField(r.document?.derivedStructData, 'htmlFormattedUrl') || '';
      return {
        title: getField(r.document?.derivedStructData, 'title') ||
               getField(r.document?.derivedStructData, 'displayLink') || 'Untitled',
        url: urlVal,
        domain: extractDomain(urlVal),
        isAcademic: isAcademicSource(urlVal),
      };
    }).filter(r => r.url);

    return { results, latencyMs, rawSizeKB };
  } catch (error: any) {
    return { results: [], latencyMs: Date.now() - start, rawSizeKB: 0 };
  }
}

// ─── Run Comparison ──────────────────────────────────────────────

async function runComparison() {
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║   Real-World Comparison: Brave Only vs Hybrid (Brave + Vertex AI)  ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

  const comparisons: ComparisonResult[] = [];

  for (const { query, context } of REAL_WORLD_QUERIES) {
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`📋 Scenario: ${context}`);
    console.log(`🔍 Query: "${query}"`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    // Search both sources
    const brave = await searchBrave(query);
    const de = await searchDE(query);

    // Brave-only approach (what you'd get without Vertex AI)
    const braveReduced = brave.results.map(r => ({ title: r.title, url: r.url }));
    const braveReducedSizeKB = Buffer.byteLength(JSON.stringify(braveReduced), 'utf8') / 1024;

    // Hybrid approach: merge + dedup
    const seenUrls = new Set<string>();
    const mergedResults: Result[] = [];
    for (const r of [...brave.results, ...de.results]) {
      const key = r.url.toLowerCase().replace(/\/$/, '');
      if (!seenUrls.has(key) && r.url) {
        seenUrls.add(key);
        mergedResults.push(r);
      }
    }

    const hybridReduced = mergedResults.map(r => ({ title: r.title, url: r.url }));
    const hybridReducedSizeKB = Buffer.byteLength(JSON.stringify(hybridReduced), 'utf8') / 1024;

    // Count unique DE contributions
    const braveUrls = new Set(brave.results.map(r => r.url.toLowerCase().replace(/\/$/, '')));
    const uniqueFromDE = de.results.filter(r => !braveUrls.has(r.url.toLowerCase().replace(/\/$/, ''))).length;
    const overlapCount = de.results.length - uniqueFromDE;

    const comparison: ComparisonResult = {
      query,
      context,
      braveOnly: {
        results: brave.results,
        latencyMs: brave.latencyMs,
        rawSizeKB: brave.rawSizeKB,
        reducedSizeKB: braveReducedSizeKB,
        academicCount: brave.results.filter(r => r.isAcademic).length,
      },
      hybrid: {
        results: mergedResults,
        latencyMs: brave.latencyMs + de.latencyMs,
        rawSizeKB: brave.rawSizeKB + de.rawSizeKB,
        reducedSizeKB: hybridReducedSizeKB,
        academicCount: mergedResults.filter(r => r.isAcademic).length,
        uniqueFromDE,
        overlapCount,
      },
    };
    comparisons.push(comparison);

    // Print results
    console.log('  ┌─ BRAVE ONLY ────────────────────────────────────────────────');
    console.log(`  │ Results: ${brave.results.length} | Academic: ${comparison.braveOnly.academicCount} | Latency: ${brave.latencyMs}ms`);
    brave.results.forEach((r, i) => {
      const tag = r.isAcademic ? '📚' : '🌐';
      console.log(`  │  [${i + 1}] ${tag} ${r.title.substring(0, 65)}`);
      console.log(`  │      ${r.domain}`);
    });

    console.log('  │');
    console.log('  ├─ DISCOVERY ENGINE ADDITIONS ──────────────────────────────────');
    console.log(`  │ Results: ${de.results.length} | Unique (not in Brave): ${uniqueFromDE} | Overlap: ${overlapCount}`);
    
    const deUnique = de.results.filter(r => !braveUrls.has(r.url.toLowerCase().replace(/\/$/, '')));
    deUnique.forEach((r, i) => {
      const tag = r.isAcademic ? '📚' : '🔍';
      console.log(`  │  [+${i + 1}] ${tag} ${r.title.substring(0, 65)}`);
      console.log(`  │       ${r.domain}`);
    });

    if (deUnique.length === 0 && de.results.length > 0) {
      console.log('  │  (all DE results overlapped with Brave)');
    }

    console.log('  │');
    console.log('  └─ HYBRID ADVANTAGE ────────────────────────────────────────────');
    console.log(`  │ Total unique results: ${mergedResults.length} (${brave.results.length} Brave + ${uniqueFromDE} unique DE)`);
    console.log(`  │ Academic sources:     ${comparison.braveOnly.academicCount} (Brave) → ${comparison.hybrid.academicCount} (Hybrid) [+${comparison.hybrid.academicCount - comparison.braveOnly.academicCount}]`);
    console.log(`  │ Token estimate:       ${estimateTokens(braveReducedSizeKB)} (Brave) vs ${estimateTokens(hybridReducedSizeKB)} (Hybrid) for ${mergedResults.length - brave.results.length} more results`);
    console.log('');
  }

  // ─── Summary ───────────────────────────────────────────────────
  console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║                      AGGREGATE COMPARISON                          ║');
  console.log('╠══════════════════════════════════════════════════════════════════════╣');

  const avgBraveResults = avg(comparisons.map(c => c.braveOnly.results.length));
  const avgHybridResults = avg(comparisons.map(c => c.hybrid.results.length));
  const avgBraveAcademic = avg(comparisons.map(c => c.braveOnly.academicCount));
  const avgHybridAcademic = avg(comparisons.map(c => c.hybrid.academicCount));
  const avgUniqueDE = avg(comparisons.map(c => c.hybrid.uniqueFromDE));
  const totalUniqueDE = comparisons.reduce((s, c) => s + c.hybrid.uniqueFromDE, 0);
  const totalBraveResults = comparisons.reduce((s, c) => s + c.braveOnly.results.length, 0);

  console.log(`║                                                                    ║`);
  console.log(`║  Metric                    Brave Only    Hybrid     Improvement     ║`);
  console.log(`║  ─────────────────────     ──────────    ──────     ───────────     ║`);
  console.log(`║  Avg results / query       ${avgBraveResults.toFixed(1).padEnd(13)} ${avgHybridResults.toFixed(1).padEnd(11)} +${(avgHybridResults - avgBraveResults).toFixed(1)} results       ║`);
  console.log(`║  Avg academic sources      ${avgBraveAcademic.toFixed(1).padEnd(13)} ${avgHybridAcademic.toFixed(1).padEnd(11)} +${(avgHybridAcademic - avgBraveAcademic).toFixed(1)} sources        ║`);
  console.log(`║  Unique DE contributions   —             ${avgUniqueDE.toFixed(1).padEnd(11)} ${totalUniqueDE} across ${comparisons.length} queries  ║`);
  console.log(`║  Coverage improvement      —             —          +${((avgHybridResults / avgBraveResults - 1) * 100).toFixed(0)}% more results  ║`);
  console.log(`║                                                                    ║`);
  console.log('╚══════════════════════════════════════════════════════════════════════╝');

  // Verdict
  if (totalUniqueDE > 0) {
    console.log(`\n✅ Discovery Engine added ${totalUniqueDE} unique results across ${comparisons.length} queries that Brave Search did not find.`);
    console.log(`   This represents a ${((avgHybridResults / avgBraveResults - 1) * 100).toFixed(0)}% increase in result coverage.`);
  } else {
    console.log(`\n⚠️  No unique DE results found in this run — results may vary by query and index content.`);
  }
}

function avg(nums: number[]): number {
  return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

runComparison().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
