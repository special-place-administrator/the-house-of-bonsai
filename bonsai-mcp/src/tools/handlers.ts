import { isStartModelArgs, isStopModelArgs } from "./definitions.js";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const HOST = process.env.BONSAI_API_HOST ?? "127.0.0.1";
const PORT = process.env.BONSAI_API_PORT ?? "9876";
const SECRET = process.env.BONSAI_API_SECRET ?? "";
const BASE = `http://${HOST}:${PORT}`;

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (SECRET) {
    headers["X-Bonsai-Secret"] = SECRET;
  }
  return headers;
}

async function apiGet(path: string): Promise<unknown> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Bonsai API ${path} returned ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function apiPost(path: string): Promise<unknown> {
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Bonsai API ${path} returned ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Model name → slot index resolution
// ---------------------------------------------------------------------------

interface SlotInfo {
  index: number;
  alias: string;
  model_path: string;
  status: string;
}

async function resolveModelToSlot(model: string): Promise<number> {
  const data = (await apiGet("/api/status")) as {
    ok: boolean;
    data: { slots: SlotInfo[] };
  };
  const slots = data.data.slots;
  const needle = model.toLowerCase();

  const idx = slots.findIndex(
    (s) =>
      s.alias?.toLowerCase() === needle ||
      s.model_path?.toLowerCase().includes(needle)
  );

  if (idx === -1) {
    const available = slots
      .map((s) => `  slot ${s.index}: ${s.alias || "(no alias)"}`)
      .join("\n");
    throw new Error(
      `Model "${model}" not found in any slot.\nAvailable slots:\n${available}`
    );
  }
  return idx;
}

// ---------------------------------------------------------------------------
// MCP response helper
// ---------------------------------------------------------------------------

function ok(data: unknown) {
  return {
    content: [
      { type: "text" as const, text: JSON.stringify(data, null, 2) },
    ],
  };
}

function err(message: string) {
  return {
    content: [{ type: "text" as const, text: message }],
    isError: true,
  };
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

export async function bonsaiCheckHealthHandler(_args: unknown) {
  try {
    const [health, status] = await Promise.all([
      apiGet("/api/health"),
      apiGet("/api/status"),
    ]);
    return ok({ health, status });
  } catch (e) {
    return err(
      `Bonsai is not reachable at ${BASE}. Is the launcher running?\n${e}`
    );
  }
}

export async function bonsaiListModelsHandler(_args: unknown) {
  try {
    const data = await apiGet("/api/models");
    return ok(data);
  } catch (e) {
    return err(`Failed to list models: ${e}`);
  }
}

export async function bonsaiStartModelHandler(args: unknown) {
  if (!isStartModelArgs(args)) {
    return err(
      'Invalid arguments. Provide { "model": "name" } or { "slot": 0 }.'
    );
  }

  try {
    let slotIndex: number;
    if (args.slot !== undefined) {
      slotIndex = args.slot;
    } else {
      slotIndex = await resolveModelToSlot(args.model!);
    }
    const data = await apiPost(`/api/start/${slotIndex}`);
    return ok(data);
  } catch (e) {
    return err(`Failed to start model: ${e}`);
  }
}

export async function bonsaiStopModelHandler(args: unknown) {
  if (!isStopModelArgs(args)) {
    return err(
      'Invalid arguments. Provide { "model": "name" } or { "slot": 0 }.'
    );
  }

  try {
    let slotIndex: number;
    if (args.slot !== undefined) {
      slotIndex = args.slot;
    } else {
      slotIndex = await resolveModelToSlot(args.model!);
    }
    const data = await apiPost(`/api/stop/${slotIndex}`);
    return ok(data);
  } catch (e) {
    return err(`Failed to stop model: ${e}`);
  }
}

export async function bonsaiGpuStatusHandler(_args: unknown) {
  try {
    const data = await apiGet("/api/gpu");
    return ok(data);
  } catch (e) {
    return err(`Failed to get GPU status: ${e}`);
  }
}

export async function bonsaiGetConfigHandler(_args: unknown) {
  try {
    const data = await apiGet("/api/config");
    return ok(data);
  } catch (e) {
    return err(`Failed to get config: ${e}`);
  }
}
