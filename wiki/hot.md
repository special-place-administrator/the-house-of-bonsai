---
type: meta
title: "Hot Cache"
created: 2026-04-08
updated: 2026-04-09
tags: [meta, cache]
status: active
---

# Hot Cache

**Last Updated:** 2026-04-09T01:30

## Key Recent Facts
- Launcher renamed: turboquant-launcher → bonsai-launcher. All source refs, config dir (BonsaiLauncher) updated
- Q1_0 patch: type IDs Q1_0_g128=44, Q1_0=45 are SACRED (GGUF has 44 hardcoded). TCQ bumped to 46-47. Patch doesn't apply cleanly — 3 hunks need manual fix (ggml.h type enum, mmvq.cu switch, generate_cu_files.py TYPES_KV)
- Bonsai-8B model: `Bonsai-8B.gguf` from `huggingface.co/prism-ml/Bonsai-8B-gguf` (1.15 GB, 254 tensors, qwen3). Download script filename is STALE
- `digitsflow_bonsai-8b_latest.gguf` from Ollama is NOT compatible — different GGUF tensor layout
- Build: VS 2026 (v18), CUDA 13.2 (DLLs at `bin/x64/`), Vulkan 1.4.341.1, VS-bundled CMake required
- Package: `C:\TOOLS\Bonzai\` — bonsai-launcher.exe, system/, models/, prism-mcp/
- Prism settings: prism-config.db is single source of truth, dashboard controls everything (8-step cleanup done)
- Auto-tune: MUST derive ALL settings from GGUF metadata + VRAM math. NO hardcoded model-specific values
- Embedder settings (empirically validated): parallel=2, context=8192, turbo3 cache, FA on. Higher parallel breaks Prism MCP
- Sampling params (temp, top_p, top_k, min_p, reasoning_budget) added to ModelSlot, read from GGUF `general.sampling.*`
- Config migration: load() copies from LlamaTurboQuantLauncher → BonsaiLauncher on first run
- Deploy Prism: finds prism-mcp next to exe, npm via `where npm.cmd`, `claude mcp add` arg order fixed (name before -e flags)
- Process cleanup: ctrlc handler + post-launch kill_all_llama_servers() as safety net. Job Object may fail on nested jobs

## Critical Lessons Learned
- **Type ID 44 is sacred** — Bonsai-8B.gguf has it hardcoded. TCQ types were at 44-45 and displaced Q1_0, causing tensor offset mismatch
- **Never fork over local changes** — forking llama-cpp from upstream wiped the parent-child process fix that was only in local working copy
- **Always verify builds work end-to-end** — build succeeding ≠ model loading. Must test actual model load after any llama-cpp change
- **Embedder parallel=8 breaks Prism MCP** — starves context per slot. Validated limit is parallel=2
- **Don't hardcode model settings** — auto-tune must use GGUF metadata + VRAM computation, not if/else per model type

## Recent Changes
- 2026-04-08: Prism settings cleanup (8 steps), launcher rename, Q1_0 patch applied, build pipeline fixed
- 2026-04-08: Package assembled at C:\TOOLS\Bonzai\ with proper structure
- 2026-04-08: claude-obsidian Stop hook fixed — infinite loop (type:prompt → response → Stop → loop)
- 2026-04-09: Auto-tune rewritten — derives cache/FA/parallel from VRAM+metadata, no hardcoded values
- 2026-04-09: Sampling params added to ModelSlot (temp, top_p, top_k, min_p, reasoning_budget)
- 2026-04-09: Deploy Prism fixed — prism-mcp search path, npm detection via `where`, claude mcp add arg order
- 2026-04-09: Process cleanup — ctrlc handler + post-launch kill_all added

## Active Threads
- download-model.ps1 needs filename fix (Bonsai-8B.gguf not bonsai-8b-q1_0_g128.gguf)
- "unknown type q1_0_g128" warning on model load — cosmetic, model loads fine
- Sampling param UI fields not yet in Dioxus — values persist in config, passed to server, but not editable in GUI
- Parent-child process watchdog was lost in fork — ctrlc+kill_all is current workaround
- Uncommitted changes in launcher and llama-cpp — need to commit and push
