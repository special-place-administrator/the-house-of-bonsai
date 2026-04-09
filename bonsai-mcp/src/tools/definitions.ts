import type { Tool } from "@modelcontextprotocol/sdk/types.js";

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

export const BONSAI_CHECK_HEALTH_TOOL: Tool = {
  name: "bonsai_check_health",
  description:
    "Check if the Bonsai inference stack is running. Returns the status of all model slots including which models are loaded, their ports, and whether they are healthy.",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

export const BONSAI_LIST_MODELS_TOOL: Tool = {
  name: "bonsai_list_models",
  description:
    "List all available GGUF models on disk that can be loaded into Bonsai. Returns model names, file paths, sizes, and sources (local file, Ollama, or recent).",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

export const BONSAI_START_MODEL_TOOL: Tool = {
  name: "bonsai_start_model",
  description:
    "Start a model in the Bonsai inference stack. Specify the model by name/alias (matched case-insensitively against configured slot aliases and model filenames) or by explicit slot index.",
  inputSchema: {
    type: "object",
    properties: {
      model: {
        type: "string",
        description:
          "Model name or alias to start (e.g. 'Bonsai-8B', 'nomic-embed-text-v2-moe'). Matched against slot aliases and model path filenames.",
      },
      slot: {
        type: "number",
        description:
          "Explicit slot index (0-based). Use this instead of model name if you know the exact slot.",
      },
    },
  },
};

export const BONSAI_STOP_MODEL_TOOL: Tool = {
  name: "bonsai_stop_model",
  description:
    "Stop a running model in the Bonsai inference stack. Specify the model by name/alias (matched case-insensitively) or by explicit slot index.",
  inputSchema: {
    type: "object",
    properties: {
      model: {
        type: "string",
        description: "Model name or alias to stop.",
      },
      slot: {
        type: "number",
        description: "Explicit slot index (0-based).",
      },
    },
  },
};

export const BONSAI_GPU_STATUS_TOOL: Tool = {
  name: "bonsai_gpu_status",
  description:
    "Get GPU and system resource information including GPU name, VRAM (total and free), RAM, and CPU details.",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

export const BONSAI_GET_CONFIG_TOOL: Tool = {
  name: "bonsai_get_config",
  description:
    "Get the current Bonsai launcher configuration including all slot settings, host, ports, and extra arguments.",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

// ---------------------------------------------------------------------------
// All tools array
// ---------------------------------------------------------------------------

export const ALL_TOOLS: Tool[] = [
  BONSAI_CHECK_HEALTH_TOOL,
  BONSAI_LIST_MODELS_TOOL,
  BONSAI_START_MODEL_TOOL,
  BONSAI_STOP_MODEL_TOOL,
  BONSAI_GPU_STATUS_TOOL,
  BONSAI_GET_CONFIG_TOOL,
];

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

export function isStartModelArgs(
  args: unknown
): args is { model?: string; slot?: number } {
  if (typeof args !== "object" || args === null) return false;
  const a = args as Record<string, unknown>;
  if (a.model !== undefined && typeof a.model !== "string") return false;
  if (a.slot !== undefined) {
    if (typeof a.slot !== "number" || !Number.isInteger(a.slot) || a.slot < 0) return false;
  }
  return a.model !== undefined || a.slot !== undefined;
}

export function isStopModelArgs(
  args: unknown
): args is { model?: string; slot?: number } {
  if (typeof args !== "object" || args === null) return false;
  const a = args as Record<string, unknown>;
  if (a.model !== undefined && typeof a.model !== "string") return false;
  if (a.slot !== undefined) {
    if (typeof a.slot !== "number" || !Number.isInteger(a.slot) || a.slot < 0) return false;
  }
  return a.model !== undefined || a.slot !== undefined;
}
