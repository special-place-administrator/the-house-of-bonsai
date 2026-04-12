# Planning Handoff: Bonsai MCP Integration

## Load context first
Run `session_load_context(project: "the-house-of-bonsai", level: "deep")` to get full session history.

---

## Project
**the-house-of-bonsai** — A local AI launcher (Rust/Tauri) that bundles llama.cpp and manages model inference on consumer GPUs. Currently at `C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai`.

## Current State
- Bonsai launcher runs two llama-server instances: **Bonsai-8B** (text, port 8080) and **nomic-embed-text-v2-moe** (embeddings, port 8081, 768 dims)
- **Prism MCP** (v7.8.2) is installed at `C:\Users\poslj\prism-mcp` from `dcostenco/prism-mcp` GitHub
- We already patched Prism with a **Llama.cpp provider** — new adapter at `src/utils/llm/adapters/llamacpp.ts`, registered in `factory.ts`, with dashboard UI fields in `ui.ts`
- Verified: ALL Prism features marked "Cloud only" (semantic search, briefings, compaction, VLM, Dark Factory, Web Scholar) actually use `getLLMProvider()` abstraction — they work with our local Llama.cpp provider
- npm registry has Prism 8.0.3 but GitHub source is 7.8.2 — newer versions published from private source

## Vision
Transform Bonsai from a model launcher into a **complete local AI stack** by deeply integrating Prism MCP:

### Phase 1: Fork & Submodule
- Fork `dcostenco/prism-mcp` to `special-place-administrator/prism-mcp`
- Push the Llama.cpp provider patch
- Add fork as git submodule in `the-house-of-bonsai` (like `llama-cpp/` already is)

### Phase 2: MCP Auto-Installer
Build into Bonsai launcher a one-click MCP installer that:
1. **Detects AI harnesses** — scans for Claude Code, Cursor, Windsurf, VS Code + Continue, etc. by checking known config file paths
2. **Checks model readiness** — verifies llama-server instances running, text model responds to `/v1/chat/completions`, embedding model responds to `/v1/embeddings`, embedding dims == 768
3. **Configures MCP** — writes `prism-mcp` entry into each harness's MCP config (global or project scope — ask user)
4. **Pre-flight validation** — end-to-end test: generate embedding, run text completion, verify dashboard reachable
5. **Green light** — only marks ready when all checks pass

### Phase 3: Two-Way MCP
Expose Bonsai management as MCP tools so LLMs can control the inference stack:
- `bonsai_start_model` / `bonsai_stop_model`
- `bonsai_check_health`
- `bonsai_swap_model`
- `bonsai_list_models`
- `bonsai_gpu_status`
- etc.

### Phase 4: Unified Dashboard
Merge Prism's web dashboard with Bonsai model management:
- Single web UI for memory (Prism) + models (Bonsai) + inference settings (llama-server)
- Strip unnecessary cloud-only paths (Supabase, Gemini-specific cruft)
- "Mind Palace" button in Bonsai launcher GUI: grey when not ready, green when all checks pass, click opens dashboard

### Phase 5: Documentation
- Updated capability matrix showing all features work offline with Llama.cpp
- Integration guide for users who want to add Prism to their own Bonsai install
- Prompt template for Claude Code users to patch their own Prism (already drafted at `prism-llamacpp-patch-prompt.md`)

## Key Technical Facts
- Bonsai launcher is **Rust/Tauri** (`launcher/src/`)
- llama-server exposes **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/embeddings`)
- Prism's LLM abstraction: `src/utils/llm/factory.ts` → `getLLMProvider()` → adapters (gemini, openai, anthropic, voyage, llamacpp)
- Prism dashboard: `src/dashboard/ui.ts` (inline HTML in TypeScript), served on port 3333
- Prism storage: SQLite + FTS5 + RotorQuant locally, pgvector on Supabase (cloud)
- Embedding dimension contract: exactly **768 dims** (nomic-embed-text-v2-moe outputs 768 natively)
- GPU: RTX 5090 (32GB VRAM)

## Constraints
- Must remain fully offline-capable — no cloud API keys required for core functionality
- Brave API key still needed for Web Scholar web fetching (that's expected — it's a web search)
- VLM image captioning won't work with Bonsai-8B (no vision) — needs llava or similar vision model as a future addition
- Must not break existing Prism functionality for users who still use Gemini/OpenAI cloud providers

## What to plan
Create a phased implementation plan covering all 5 phases above. Identify file changes, new files, dependencies between phases, and verification steps for each phase.
