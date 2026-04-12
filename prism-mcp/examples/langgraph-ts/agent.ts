/**
 * Prism Memory-Augmented Agent — TypeScript LangGraph Example
 * ═══════════════════════════════════════════════════════════════════
 * A reference implementation of a LangGraph agent that uses Prism MCP
 * as its persistent memory backend. This is the TypeScript companion
 * to the Python example in ../langgraph-agent/.
 *
 * THE AGENT LOOP:
 *   1. Load context — recover previous session state from Prism
 *   2. Search memory — check if similar work was done before
 *   3. Research — perform new work (simulated here)
 *   4. Save session — persist findings back to Prism
 *
 * WHY THIS MATTERS:
 *   Without persistent memory, every LangGraph run starts from zero.
 *   With Prism, the agent remembers past sessions, avoids redundant
 *   work, and accumulates institutional knowledge across runs.
 *
 * PREREQUISITES:
 *   npm install @langchain/core @langchain/langgraph @modelcontextprotocol/sdk
 *
 * RUN:
 *   npx tsx agent.ts
 * ═══════════════════════════════════════════════════════════════════
 */

import { PrismClient } from "./prism-client.js";
import { createMemorySearchNode, createKnowledgeSearchNode } from "./retriever.js";

// ─── Configuration ───────────────────────────────────────────────
const PROJECT = process.env.PRISM_PROJECT || "langgraph-demo";

/**
 * Main agent loop demonstrating memory-augmented research.
 *
 * DESIGN NOTES:
 *   This is a simplified agent for demonstration purposes. In production,
 *   you would use LangGraph's StateGraph to define the flow declaratively.
 *   The core pattern remains the same:
 *     load context → search memory → work → save session
 */
async function main() {
  const client = new PrismClient();

  try {
    // ── Step 1: Connect to Prism ──────────────────────────────────
    console.log("🧠 Connecting to Prism MCP server...");
    await client.connect();

    // ── Step 2: Load previous context ──────────────────────────────
    console.log(`\n📚 Loading context for project: ${PROJECT}`);
    const context = await client.loadContext(PROJECT, "standard");
    if (context) {
      console.log("Previous context found:");
      console.log(context.slice(0, 500));  // Show first 500 chars
    } else {
      console.log("No previous context — this is a fresh project.");
    }

    // ── Step 3: Search memory for related work ─────────────────────
    const query = process.argv[2] || "recent architectural decisions";
    console.log(`\n🔍 Searching memory: "${query}"`);

    const memoryNode = createMemorySearchNode(client, PROJECT, {
      limit: 3,
      threshold: 0.65,
      enableTrace: true,
    });

    const memoryResults = await memoryNode({ query });
    if (memoryResults.memories.length > 0) {
      console.log(`Found ${memoryResults.memories.length} relevant memories:`);
      memoryResults.memories.forEach((m, i) => {
        console.log(`  ${i + 1}. ${m.summary} (score: ${m.score?.toFixed(2) ?? "N/A"})`);
      });
    } else {
      console.log("No relevant memories found — doing fresh research.");
    }

    // ── Step 4: Search knowledge base ──────────────────────────────
    console.log(`\n📖 Searching knowledge: "${query}"`);
    const knowledgeNode = createKnowledgeSearchNode(client, PROJECT);
    const knowledgeResults = await knowledgeNode({ query });
    if (knowledgeResults.knowledge.length > 0) {
      console.log(`Found ${knowledgeResults.knowledge.length} knowledge entries:`);
      knowledgeResults.knowledge.slice(0, 3).forEach((k) => {
        console.log(`  • ${k.slice(0, 120)}`);
      });
    }

    // ── Step 5: Do research (simulated) ────────────────────────────
    console.log("\n🔬 Performing research... (simulated)");
    const researchFindings = `Investigated: ${query}. Found patterns in existing codebase.`;
    console.log(`  Result: ${researchFindings}`);

    // ── Step 6: Save session to Prism ──────────────────────────────
    console.log("\n💾 Saving session to Prism ledger...");
    await client.saveLedger({
      project: PROJECT,
      summary: `LangGraph agent researched: ${query}. ${researchFindings}`,
      decisions: ["Used Prism memory-augmented research pattern"],
      todos: ["Follow up on research findings"],
    });
    console.log("  ✅ Session saved successfully!");

    // ── Summary ───────────────────────────────────────────────────
    console.log("\n" + "═".repeat(60));
    console.log("🎉 Agent run complete! Memory persisted to Prism.");
    console.log("  Next run will recover this session's findings.");
    console.log("═".repeat(60));

  } catch (error) {
    console.error("Agent error:", error);
    process.exitCode = 1;
  } finally {
    await client.disconnect();
  }
}

// Run the agent
main();
