#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  ALL_TOOLS,
  bonsaiCheckHealthHandler,
  bonsaiListModelsHandler,
  bonsaiStartModelHandler,
  bonsaiStopModelHandler,
  bonsaiGpuStatusHandler,
  bonsaiGetConfigHandler,
} from "./tools/index.js";

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new Server(
  {
    name: "bonsai-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// ---------------------------------------------------------------------------
// Tool listing
// ---------------------------------------------------------------------------

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: ALL_TOOLS };
});

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "bonsai_check_health":
      return bonsaiCheckHealthHandler(args);
    case "bonsai_list_models":
      return bonsaiListModelsHandler(args);
    case "bonsai_start_model":
      return bonsaiStartModelHandler(args);
    case "bonsai_stop_model":
      return bonsaiStopModelHandler(args);
    case "bonsai_gpu_status":
      return bonsaiGpuStatusHandler(args);
    case "bonsai_get_config":
      return bonsaiGetConfigHandler(args);
    default:
      return {
        content: [{ type: "text" as const, text: `Unknown tool: ${name}` }],
        isError: true,
      };
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("bonsai-mcp fatal:", err);
  process.exit(1);
});
