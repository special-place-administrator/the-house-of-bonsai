---
type: meta
title: "Hot Cache"
created: 2026-04-08
updated: 2026-04-09
tags: [meta, cache]
status: active
---

# Hot Cache

**Last Updated:** 2026-04-09T02:30

## Key Recent Facts
- Launcher renamed: turboquant-launcher → bonsai-launcher. All source refs, config dir (BonsaiLauncher) updated
- Q1_0 patch: type IDs Q1_0_g128=44, Q1_0=45 are SACRED (GGUF has 44 hardcoded). TCQ bumped to 46-47
- Bonsai-8B model: `Bonsai-8B.gguf` from `huggingface.co/prism-ml/Bonsai-8B-gguf` (1.15 GB, 254 tensors, qwen3)
- Package: `C:\TOOLS\Bonzai\` — bonsai-launcher.exe, system/, models/, prism-mcp/
- Prism settings: prism-config.db is single source of truth, dashboard controls everything
- Auto-tune: derives ALL settings from GGUF metadata + VRAM math. NO hardcoded values
- Embedder settings (empirically validated): parallel=2, context=8192, turbo3 cache, FA on
- Sampling params (temp, top_p, top_k, min_p, reasoning_budget) in ModelSlot, read from GGUF `general.sampling.*`
- Dashboard provider dropdowns are DYNAMIC — show actual model alias from settings (e.g. "Llama.cpp (Bonsai-8B)")
- Alias is user-defined identity — launcher GUI sets alias, Deploy Prism passes it to LLAMACPP_TEXT_MODEL / LLAMACPP_EMBEDDING_MODEL env vars
- Deploy Prism: all paths derived from current_exe() location. NEVER hardcode install paths
- Process cleanup: ctrlc handler + post-launch kill_all_llama_servers() as safety net

## Critical Lessons Learned
- **Type ID 44 is sacred** — Bonsai-8B.gguf has it hardcoded. TCQ displaced Q1_0, causing tensor offset mismatch
- **Never fork over local changes** — forking llama-cpp from upstream wiped local fixes
- **Always verify builds end-to-end** — build succeeding ≠ model loading
- **Embedder parallel=8 breaks Prism MCP** — starves context per slot. Validated limit is parallel=2
- **Don't hardcode model settings** — auto-tune uses GGUF metadata + VRAM, not if/else per model type
- **Don't hardcode install paths** — Deploy Prism, prism-mcp search, model detection all derive from exe location
- **Alias = identity** — model alias flows from launcher GUI → llama-server -a flag → Prism env vars → dashboard labels. All dynamic

## Recent Changes
- 2026-04-08: Prism settings cleanup (8 steps), launcher rename, Q1_0 patch, build pipeline fixed
- 2026-04-08: Package assembled, claude-obsidian Stop hook loop fixed
- 2026-04-09: Auto-tune rewritten — algorithmic, no hardcoded values
- 2026-04-09: Sampling params added to ModelSlot
- 2026-04-09: Deploy Prism fixed — path detection, npm via `where`, claude mcp add arg order
- 2026-04-09: Dashboard provider dropdowns now show dynamic model aliases from settings
- 2026-04-09: Removed all hardcoded "Bonsai" labels from provider dropdowns — labels read from DB

## Active Threads
- download-model.ps1 needs filename fix (Bonsai-8B.gguf not bonsai-8b-q1_0_g128.gguf)
- "unknown type q1_0_g128" warning on model load — cosmetic, model loads fine
- Sampling param UI fields not yet in Dioxus — values persist in config, passed to server, but not editable in GUI
- Parent-child process watchdog was lost in fork — ctrlc+kill_all is current workaround
