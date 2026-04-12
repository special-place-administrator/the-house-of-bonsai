/**
 * Morning Briefing Generator (v2.0 — Step 7)
 *
 * Synthesizes a punchy, 3-bullet-point action plan using Gemini
 * when the developer opens Prism for the first time in > 4 hours.
 *
 * ═══════════════════════════════════════════════════════════════════
 * DESIGN DECISIONS:
 *   - Uses gemini-2.5-flash for max speed (~2-3s generation)
 *   - Graceful fallback if no API key or Gemini call fails
 *   - Prompt is tuned for brevity — exactly 3 bullets, no fluff
 *   - Reuses GOOGLE_API_KEY from config.ts (same key as embeddings)
 * ═══════════════════════════════════════════════════════════════════
 */

import { getLLMProvider } from "./llm/factory.js";
import { debugLog } from "./logger.js";

export interface BriefingContext {
  project: string;
  lastSummary?: string | null;
  pendingTodos?: string[] | null;
  keyContext?: string | null;
  activeBranch?: string | null;
}

export interface LedgerSummary {
  type: string;
  summary: string;
}

/**
 * Generates a 3-bullet Morning Briefing using Gemini.
 *
 * @param context - The loaded handoff state (summary, todos, etc.)
 * @param recentEntries - The 10 most recent ledger entries
 * @returns Formatted briefing text, or a graceful fallback string
 */
export async function generateMorningBriefing(
  context: BriefingContext,
  recentEntries: LedgerSummary[]
): Promise<string> {
  let llm;
  try {
    llm = getLLMProvider();
  } catch {
    return "☀️ Good morning! (Morning Briefing unavailable — LLM provider not configured)";
  }

  const todosBlock = context.pendingTodos?.length
    ? `Pending TODOs:\n${context.pendingTodos.map(t => `  • ${t}`).join("\n")}`
    : "No pending TODOs.";

  const recentBlock = recentEntries.length
    ? recentEntries
        .map(e => `  [${e.type}] ${e.summary}`)
        .join("\n")
    : "No recent activity logged.";

  const prompt = `You are an elite AI engineering assistant. A developer is starting their first session of the day on project "${context.project}".

Project Summary: ${context.lastSummary || "No summary available."}
${todosBlock}
Key Context: ${context.keyContext || "None."}
Active Branch: ${context.activeBranch || "unknown"}

Most recent session activity:
${recentBlock}

Write a "Morning Briefing" — EXACTLY 3 short, punchy bullet points:
• Bullet 1: Where we left off (1 sentence max).
• Bullet 2: The immediate next step based on the TODOs/context (1 sentence max).
• Bullet 3: A quick tip, potential gotcha, or priority call-out for today (1 sentence max).

Rules:
- Be direct, energetic, and actionable.
- No preamble, no closing remarks — just the 3 bullets.
- Each bullet starts with a relevant emoji.`;

  try {
    const text = (await llm.generateText(prompt)).trim();
    debugLog(`[Morning Briefing] Generated for "${context.project}" (${text.length} chars)`);
    return text;
  } catch (error) {
    console.error(
      `[Morning Briefing] LLM call failed: ${error instanceof Error ? error.message : String(error)}`
    );
    return "☀️ Good morning! Ready to continue — check your TODOs above to pick up where you left off.";
  }
}
