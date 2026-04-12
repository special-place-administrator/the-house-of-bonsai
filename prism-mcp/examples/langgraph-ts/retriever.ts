/**
 * Prism Memory Retriever — LangGraph-Compatible Node
 * ═══════════════════════════════════════════════════════════════════
 * A retriever node for LangGraph that searches Prism MCP's memory
 * and returns results as structured state updates.
 *
 * This is the TypeScript equivalent of the Python PrismMemoryRetriever
 * in ../langgraph-agent/prism_retriever.py.
 *
 * KEY DIFFERENCE FROM PYTHON VERSION:
 *   TypeScript/LangGraph.js uses graph nodes instead of BaseRetriever.
 *   Each node is an async function that reads/writes to the graph state.
 *   This is simpler than the Python LangChain adapter pattern.
 *
 * USAGE IN A LANGGRAPH GRAPH:
 *   const graph = new StateGraph(AgentState)
 *     .addNode("search_memory", searchMemoryNode)
 *     .addNode("research", researchNode)
 *     .addEdge("search_memory", "research");
 * ═══════════════════════════════════════════════════════════════════
 */

import { PrismClient } from "./prism-client.js";

/**
 * The MEMORY TRACE marker — must match src/utils/tracing.ts.
 * When enable_trace=true, Prism appends a second content block
 * with this marker, containing latency and scoring metadata.
 */
const TRACE_MARKER = "=== MEMORY TRACE ===";

/**
 * A single memory search result parsed from Prism's response.
 */
export interface MemoryResult {
  /** The matched summary text from the ledger */
  summary: string;
  /** Similarity score (0-1) — higher is more relevant */
  score?: number;
  /** Project the memory belongs to */
  project?: string;
  /** Optional parsed MemoryTrace for observability */
  trace?: Record<string, unknown>;
}

/**
 * Parse MemoryTrace JSON from the second content block.
 *
 * DESIGN NOTE:
 *   Prism returns traces in content[1] (content[0] is the results text).
 *   The trace is prefixed with "=== MEMORY TRACE ===" as a sentinel.
 *   This function safely extracts the JSON, returning null if unavailable.
 *
 * @param contentBlocks - Raw MCP content block array
 * @returns Parsed trace object, or null
 */
function parseTraceBlock(
  contentBlocks: Array<{ type: string; text: string }>
): Record<string, unknown> | null {
  if (contentBlocks.length < 2) return null;

  const traceText = contentBlocks[1]?.text ?? "";
  if (!traceText.includes(TRACE_MARKER)) return null;

  try {
    const rawJson = traceText.replace(TRACE_MARKER, "").trim();
    return JSON.parse(rawJson);
  } catch {
    console.warn("[PrismRetriever] Failed to parse MemoryTrace JSON");
    return null;
  }
}

/**
 * Create a LangGraph-compatible memory search node.
 *
 * This factory creates an async function suitable for use as a
 * LangGraph node. It searches Prism's semantic memory and returns
 * the results as an array of MemoryResult objects.
 *
 * WHY A FACTORY?
 *   LangGraph nodes are functions with a specific signature:
 *     (state: S) => Promise<Partial<S>>
 *   The factory captures the PrismClient in a closure, allowing
 *   the node to be stateless while still having MCP access.
 *
 * @param client - Connected PrismClient instance
 * @param project - Optional project filter
 * @param options - Search configuration
 * @returns Async function suitable for graph.addNode()
 *
 * @example
 * const node = createMemorySearchNode(client, "my-project");
 * const results = await node({ query: "auth patterns" });
 */
export function createMemorySearchNode(
  client: PrismClient,
  project?: string,
  options: { limit?: number; threshold?: number; enableTrace?: boolean } = {}
) {
  const { limit = 5, threshold = 0.7, enableTrace = true } = options;

  return async (state: { query: string }): Promise<{ memories: MemoryResult[] }> => {
    const result = await client.callTool("session_search_memory", {
      query: state.query,
      project,
      limit,
      similarity_threshold: threshold,
      enable_trace: enableTrace,
    });

    if (result.isError) {
      console.error("[PrismRetriever] Search failed:", result.content?.[0]?.text);
      return { memories: [] };
    }

    const text = result.content?.[0]?.text ?? "";
    const trace = enableTrace ? parseTraceBlock(result.content) : null;

    // Parse the text response into individual results
    // Prism returns results as numbered items: "1. [project] summary (score: 0.85)"
    const memories: MemoryResult[] = text
      .split("\n")
      .filter((line: string) => line.match(/^\d+\./))
      .map((line: string) => {
        const scoreMatch = line.match(/\(score:\s*([\d.]+)\)/);
        const projectMatch = line.match(/\[([^\]]+)\]/);
        return {
          summary: line.replace(/^\d+\.\s*/, "").replace(/\(score:.*\)/, "").trim(),
          score: scoreMatch ? parseFloat(scoreMatch[1]) : undefined,
          project: projectMatch ? projectMatch[1] : project,
          trace: trace ?? undefined,
        };
      });

    return { memories };
  };
}

/**
 * Create a LangGraph-compatible knowledge search node.
 *
 * Searches Prism's keyword-based knowledge (FTS5 full-text search).
 * Complements the semantic memory search — use both for hybrid retrieval.
 *
 * @param client - Connected PrismClient instance
 * @param project - Optional project filter
 * @param category - Optional category filter (e.g., "debugging", "architecture")
 */
export function createKnowledgeSearchNode(
  client: PrismClient,
  project?: string,
  category?: string
) {
  return async (state: { query: string }): Promise<{ knowledge: string[] }> => {
    const result = await client.callTool("knowledge_search", {
      query: state.query,
      project,
      category,
      limit: 10,
    });

    if (result.isError) {
      console.error("[PrismRetriever] Knowledge search failed:", result.content?.[0]?.text);
      return { knowledge: [] };
    }

    const text = result.content?.[0]?.text ?? "";
    const knowledge = text
      .split("\n")
      .filter((line: string) => line.trim().length > 0);

    return { knowledge };
  };
}
