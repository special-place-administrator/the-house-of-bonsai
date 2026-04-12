/**
 * Prism MCP Client — TypeScript Wrapper
 * ═══════════════════════════════════════════════════════════════════
 * Lightweight MCP client that connects to the Prism MCP server via
 * stdio transport and exposes tool calling as simple async functions.
 *
 * USAGE:
 *   const client = new PrismClient();
 *   await client.connect();
 *   const result = await client.callTool("session_search_memory", { query: "auth" });
 *
 * DESIGN DECISIONS:
 *   - Uses the official @modelcontextprotocol/sdk for protocol compliance
 *   - Connects via stdio (same transport as Claude Desktop / Cursor)
 *   - Auto-discovers Prism server binary from node_modules or PATH
 * ═══════════════════════════════════════════════════════════════════
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

/**
 * Result from an MCP tool call.
 * content[0] = text results, content[1] = optional MemoryTrace (if enable_trace=true)
 */
export interface ToolResult {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
}

/**
 * Wrapper around the MCP SDK client for Prism-specific tool calls.
 *
 * WHY A WRAPPER?
 *   The raw MCP SDK requires verbose setup (transport init, capability negotiation,
 *   JSON-RPC framing). This class reduces it to 2 lines: connect() + callTool().
 *
 *   It also handles the Prism-specific patterns:
 *   - MemoryTrace parsing from content[1]
 *   - Error extraction from isError responses
 *   - Type-safe argument passing
 */
export class PrismClient {
  private client: Client;
  private transport: StdioClientTransport | null = null;

  constructor() {
    this.client = new Client(
      { name: "langgraph-ts-agent", version: "1.0.0" },
      { capabilities: {} }
    );
  }

  /**
   * Connect to the Prism MCP server via stdio.
   *
   * The server binary is discovered from the prism-mcp-server npm package.
   * Make sure prism-mcp-server is installed: npm install prism-mcp-server
   */
  async connect(): Promise<void> {
    this.transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "prism-mcp-server"],
    });

    await this.client.connect(this.transport);
    console.log("[PrismClient] Connected to Prism MCP server");
  }

  /**
   * Call a Prism MCP tool by name.
   *
   * @param toolName - The MCP tool name (e.g., "session_search_memory")
   * @param args - Tool arguments as a plain object
   * @returns The raw MCP tool result with content blocks
   *
   * @example
   * const result = await client.callTool("session_search_memory", {
   *   query: "authentication patterns",
   *   project: "my-app",
   *   enable_trace: true
   * });
   */
  async callTool(toolName: string, args: Record<string, unknown>): Promise<ToolResult> {
    const result = await this.client.callTool({
      name: toolName,
      arguments: args,
    });
    return result as unknown as ToolResult;
  }

  /**
   * Convenience: Search Prism's semantic memory.
   * Wraps session_search_memory with sane defaults.
   */
  async searchMemory(
    query: string,
    options: { project?: string; limit?: number; threshold?: number } = {}
  ): Promise<string> {
    const result = await this.callTool("session_search_memory", {
      query,
      project: options.project,
      limit: options.limit ?? 5,
      similarity_threshold: options.threshold ?? 0.7,
    });
    return result.content?.[0]?.text ?? "";
  }

  /**
   * Convenience: Search Prism's keyword-based knowledge.
   * Wraps knowledge_search with sane defaults.
   */
  async searchKnowledge(
    query: string,
    options: { project?: string; category?: string; limit?: number } = {}
  ): Promise<string> {
    const result = await this.callTool("knowledge_search", {
      query,
      project: options.project,
      category: options.category,
      limit: options.limit ?? 10,
    });
    return result.content?.[0]?.text ?? "";
  }

  /**
   * Convenience: Save a session to the Prism ledger.
   * Wraps session_save_ledger with required fields.
   */
  async saveLedger(params: {
    project: string;
    summary: string;
    conversationId?: string;
    todos?: string[];
    decisions?: string[];
    filesChanged?: string[];
  }): Promise<void> {
    await this.callTool("session_save_ledger", {
      project: params.project,
      summary: params.summary,
      conversation_id: params.conversationId ?? `langgraph-${Date.now()}`,
      todos: params.todos,
      decisions: params.decisions,
      files_changed: params.filesChanged,
    });
  }

  /**
   * Convenience: Load project context for session resumption.
   */
  async loadContext(
    project: string,
    level: "quick" | "standard" | "deep" = "standard"
  ): Promise<string> {
    const result = await this.callTool("session_load_context", {
      project,
      level,
    });
    return result.content?.[0]?.text ?? "";
  }

  /** Disconnect from the MCP server */
  async disconnect(): Promise<void> {
    if (this.transport) {
      await this.transport.close();
      console.log("[PrismClient] Disconnected");
    }
  }
}
