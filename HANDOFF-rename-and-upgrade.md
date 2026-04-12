# Handoff: Rename + Deploy Prism Upgrade Path

## Task 1: Rename the-house-of-bonsai → the-house-of-bonsai

### What to change
- `launcher/Cargo.toml` — change `name` from `"the-house-of-bonsai"` to `"the-house-of-bonsai"`, update `[[bin]]` if present
- `launcher/src/main.rs` — any window title or app name strings (rename already completed)
- `launcher/src/prism.rs` — any references to the binary name in deploy paths
- Output binary changes from `the-house-of-bonsai.exe` to `the-house-of-bonsai.exe`
- The installed location stays at `C:\TOOLS\Bonzai\` but the exe filename changes

### What NOT to change
- Do NOT rename the `launcher/` directory itself
- Do NOT rename the git repo
- Do NOT touch `C:\TOOLS\Bonzai\bin\` (that's llama-server binaries)

### After rename
- `cargo build --release` must succeed
- Copy new binary to `C:\TOOLS\Bonzai\the-house-of-bonsai.exe`
- Delete old `C:\TOOLS\Bonzai\the-house-of-bonsai.exe`

---

## Task 2: Fix bonsai-mcp build

`bonsai-mcp` fails to build without `@types/node`. Add it:

```bash
cd bonsai-mcp && npm install --save-dev @types/node
```

Commit the updated `package.json` and `package-lock.json`.

---

## Task 3: Deploy Prism Upgrade Path (Phase 2 Enhancement)

### Problem
Users may already have prism-mcp installed from a standalone clone (e.g., `C:\Users\poslj\prism-mcp`). The Deploy Prism modal currently creates/overwrites entries but doesn't handle the upgrade scenario properly. Two prism-mcp instances pointing at the same `~/.prism-mcp/data.db` will fight over SQLite locks.

### Required changes to `launcher/src/prism.rs`

#### 1. Detect existing prism-mcp installations
Add a function:
```rust
pub fn detect_existing_prism(probes: &[HarnessProbe]) -> Vec<(Harness, PathBuf)>
```
For each found harness config, read it and check if `mcpServers.prism-mcp` already exists. Extract the old `dist/server.js` path. Return a list of (harness, old_path) pairs.

For Claude Code: run `claude mcp list` and parse output to find existing prism-mcp entry.

#### 2. Show existing installation in the modal
In `DeployPrismModal` (main.rs), after harness detection completes, also run `detect_existing_prism()`. If found, show:

```
⚠ Existing Prism MCP found:
  Claude Desktop: C:\Users\poslj\prism-mcp\dist\server.js
  Claude Code:    C:\Users\poslj\prism-mcp\dist\server.js

Will be updated to:
  C:\AI_STUFF\...\the-house-of-bonsai\prism-mcp\dist\server.js
```

#### 3. Update env vars on upgrade
The old installation might have Gemini/OpenAI provider settings. The new entry MUST include:
- `TEXT_PROVIDER=llamacpp`
- `EMBEDDING_PROVIDER=llamacpp`
- All `LLAMACPP_*` env vars

The `build_prism_entry()` function already does this — just make sure it fully replaces the old entry's env block, not merges with it.

#### 4. Cleanup option
After successful deployment, offer to remove the old installation directory:

Add a signal:
```rust
cleanup_old_prism: Signal<Option<PathBuf>>  // set if old install detected and different from submodule
```

Show in modal after deploy completes:
```
☐ Remove old Prism installation (C:\Users\poslj\prism-mcp\)
  ℹ Your data at ~/.prism-mcp/ will NOT be affected
```

On click: delete the old directory (NOT `~/.prism-mcp/` — that's the data).

Safety checks before cleanup:
- Old path must NOT be inside the bonsai repo (don't delete the submodule)
- Old path must NOT be `~/.prism-mcp/` (don't delete the data)
- Old path must contain `dist/server.js` (confirm it's actually a prism install)
- Warn if old path is inside a git repo (user might have local changes)

#### 5. Warn about unmanaged harnesses
If some harnesses still point to the old path after deploy (because user unchecked them), show a warning:
```
⚠ Cursor still points to old Prism installation.
  Deploy to Cursor too, or manually update its config.
```

### Data is safe
Prism stores all data at `~/.prism-mcp/` (specifically `C:\Users\<user>\.prism-mcp\`):
- `data.db` — sessions, embeddings, ledger
- `prism-config.db` — settings
- `media/` — saved images

This directory is home-based, NOT inside the install directory. Both old and new prism-mcp installs share the same database. No migration needed. NEVER delete `~/.prism-mcp/`.

---

## Execution order
1. Rename (Task 1) — do first since deploy paths depend on binary name
2. Fix bonsai-mcp build (Task 2) — quick commit
3. Upgrade path (Task 3) — enhance prism.rs and main.rs modal
4. Rebuild + test Deploy Prism end-to-end
