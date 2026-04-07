# Prompt: Patch Prism MCP to add Llama.cpp Provider

Paste this entire prompt into Claude Code to have it patch your prism-mcp installation with a dedicated Llama.cpp provider. This adds a "🦙 Llama.cpp" option to both the Text Provider and Embedding Provider dropdowns in the Prism dashboard, with separate URL and model fields for each.

---

## Task

I need you to add a **Llama.cpp provider** to my prism-mcp installation. This provider supports llama-server (llama.cpp's OpenAI-compatible HTTP server) with **separate base URLs** for text and embedding models — which is the standard setup when running two llama-server instances on different ports.

### Step 0: Find my prism-mcp installation

Check my Claude Code MCP config to find where prism-mcp is installed. Look in:
- `~/.claude/settings.json` or `~/.claude.json` (check for mcpServers entries)
- The MCP config will have an entry for `prism-mcp` with a command/args pointing to the install path
- Common locations: the `args` array usually contains a path ending in `dist/server.js` — the parent of `dist/` is the prism-mcp root

Once found, confirm the path exists and has `src/utils/llm/adapters/` and `src/dashboard/ui.ts`.

### Step 1: Create the Llama.cpp adapter

Create a new file at `src/utils/llm/adapters/llamacpp.ts` with this content:

```typescript
/**
 * Llama.cpp Adapter
 * ─────────────────────────────────────────────────────────────────────────────
 * Implements LLMProvider for llama-server (llama.cpp's OpenAI-compatible
 * HTTP server). Supports SEPARATE base URLs for text and embedding models.
 *
 * CONFIG KEYS (Prism dashboard "AI Providers" tab):
 *   llamacpp_text_url       — Base URL for text/chat model (default: http://127.0.0.1:8080/v1)
 *   llamacpp_text_model     — Chat model alias (default: Bonsai-8B)
 *   llamacpp_embedding_url  — Base URL for embedding model (default: http://127.0.0.1:8081/v1)
 *   llamacpp_embedding_model — Embedding model alias (default: nomic-embed-text-v2-moe)
 */

import OpenAI from "openai";
import { getSettingSync } from "../../../storage/configStorage.js";
import { debugLog } from "../../logger.js";
import type { LLMProvider } from "../provider.js";

const EMBEDDING_DIMS = 768;
const MAX_EMBEDDING_CHARS = 8000;

export class LlamaCppAdapter implements LLMProvider {
  private textClient: OpenAI;
  private embeddingClient: OpenAI;

  constructor() {
    const textURL = getSettingSync("llamacpp_text_url", "http://127.0.0.1:8080/v1");
    const embedURL = getSettingSync("llamacpp_embedding_url", "http://127.0.0.1:8081/v1");

    this.textClient = new OpenAI({ apiKey: "none", baseURL: textURL });
    this.embeddingClient = new OpenAI({ apiKey: "none", baseURL: embedURL });

    debugLog(`[LlamaCppAdapter] Initialized — text=${textURL}, embedding=${embedURL}`);
  }

  async generateText(prompt: string, systemInstruction?: string): Promise<string> {
    const model = getSettingSync("llamacpp_text_model", "Bonsai-8B");

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    if (systemInstruction) {
      messages.push({ role: "system", content: systemInstruction });
    }
    messages.push({ role: "user", content: prompt });

    debugLog(`[LlamaCppAdapter] generateText — model=${model}, messages=${messages.length}`);

    const response = await this.textClient.chat.completions.create({ model, messages });
    return response.choices[0]?.message?.content ?? "";
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!text || !text.trim()) {
      throw new Error("Cannot generate embedding for empty text.");
    }

    const model = getSettingSync("llamacpp_embedding_model", "nomic-embed-text-v2-moe");

    let inputText = text;
    if (inputText.length > MAX_EMBEDDING_CHARS) {
      debugLog(
        `[LlamaCppAdapter] Embedding input truncated from ${inputText.length}` +
        ` to ~${MAX_EMBEDDING_CHARS} chars (word-safe)`
      );
      inputText = inputText.substring(0, MAX_EMBEDDING_CHARS);
      const lastSpace = inputText.lastIndexOf(" ");
      if (lastSpace > 0) inputText = inputText.substring(0, lastSpace);
    }

    debugLog(`[LlamaCppAdapter] generateEmbedding — model=${model}`);

    const response = await this.embeddingClient.embeddings.create({
      model,
      input: inputText,
    });

    const embedding = response.data[0]?.embedding;

    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error(`[LlamaCppAdapter] Embedding response is empty for model "${model}"`);
    }

    if (embedding.length !== EMBEDDING_DIMS) {
      console.warn(
        `[LlamaCppAdapter] Embedding dimension mismatch: expected ${EMBEDDING_DIMS}, ` +
        `got ${embedding.length}. Ensure your embedding model outputs ${EMBEDDING_DIMS} dims.`
      );
    }

    return embedding;
  }
}
```

### Step 2: Update the factory (`src/utils/llm/factory.ts`)

1. Add this import near the other adapter imports:
```typescript
import { LlamaCppAdapter } from "./adapters/llamacpp.js";
```

2. In `buildTextAdapter()`, add this case before the `default`:
```typescript
case "llamacpp":  return new LlamaCppAdapter();
```

3. In `buildEmbeddingAdapter()`, add this case before the `default`:
```typescript
case "llamacpp":  return new LlamaCppAdapter();
```

### Step 3: Update the dashboard UI (`src/dashboard/ui.ts`)

This is the largest change. You need to add "Llama.cpp" to both provider dropdowns and add config fields. Search for each location carefully.

#### 3a. Text Provider dropdown
Find the `<select id="select-text-provider"` block. Add this option after the "openai" option:
```html
<option value="llamacpp">🦙 Llama.cpp</option>
```

#### 3b. Llama.cpp text config fields
Right before the `<!-- Gemini text fields -->` comment, insert this HTML block:
```html
<!-- Llama.cpp text fields -->
<div id="provider-fields-llamacpp" style="display:none">
  <div class="setting-row">
    <div>
      <div class="setting-label">Text Model URL</div>
      <div class="setting-desc">llama-server endpoint for chat/completions</div>
    </div>
    <input type="text" id="input-llamacpp-text-url"
      placeholder="http://127.0.0.1:8080/v1"
      style="padding: 0.2rem 0.5rem; background: var(--bg-hover); color: var(--text-primary); border: 1px solid var(--border-color); border-radius: 4px; font-size: 0.85rem; font-family: var(--font-mono); width: 210px;"
      onchange="saveBootSetting('llamacpp_text_url', this.value)"
      oninput="clearTimeout(this._ltu); var self=this; this._ltu=setTimeout(function(){saveBootSetting('llamacpp_text_url',self.value)},800)" />
  </div>
  <div class="setting-row">
    <div>
      <div class="setting-label">Text Model</div>
      <div class="setting-desc">Model alias (as set in llama-server -a flag)</div>
    </div>
    <input type="text" id="input-llamacpp-text-model"
      placeholder="Bonsai-8B"
      style="padding: 0.2rem 0.5rem; background: var(--bg-hover); color: var(--text-primary); border: 1px solid var(--border-color); border-radius: 4px; font-size: 0.85rem; font-family: var(--font-mono); width: 210px;"
      onchange="saveBootSetting('llamacpp_text_model', this.value)"
      oninput="clearTimeout(this._ltm); var self=this; this._ltm=setTimeout(function(){saveBootSetting('llamacpp_text_model',self.value)},800)" />
  </div>
</div>
```

#### 3c. Embedding Provider dropdown
Find the `<select id="select-embedding-provider"` block. Add this option after "openai":
```html
<option value="llamacpp">🦙 Llama.cpp</option>
```

#### 3d. Llama.cpp embedding config fields
Right before the `<!-- Anthropic + auto warning -->` comment, insert:
```html
<!-- Llama.cpp embedding fields (shown when embedding_provider = llamacpp) -->
<div id="embed-fields-llamacpp" style="display:none">
  <div class="setting-row">
    <div>
      <div class="setting-label">Embedding Model URL</div>
      <div class="setting-desc">llama-server endpoint for embeddings (separate port from text)</div>
    </div>
    <input type="text" id="input-llamacpp-embedding-url"
      placeholder="http://127.0.0.1:8081/v1"
      style="padding: 0.2rem 0.5rem; background: var(--bg-hover); color: var(--text-primary); border: 1px solid var(--border-color); border-radius: 4px; font-size: 0.85rem; font-family: var(--font-mono); width: 210px;"
      onchange="saveBootSetting('llamacpp_embedding_url', this.value)"
      oninput="clearTimeout(this._leu); var self=this; this._leu=setTimeout(function(){saveBootSetting('llamacpp_embedding_url',self.value)},800)" />
  </div>
  <div class="setting-row">
    <div>
      <div class="setting-label">Embedding Model</div>
      <div class="setting-desc">Must output 768 dims. e.g. nomic-embed-text-v2-moe</div>
    </div>
    <input type="text" id="input-llamacpp-embedding-model"
      placeholder="nomic-embed-text-v2-moe"
      style="padding: 0.2rem 0.5rem; background: var(--bg-hover); color: var(--text-primary); border: 1px solid var(--border-color); border-radius: 4px; font-size: 0.85rem; font-family: var(--font-mono); width: 210px;"
      onchange="saveBootSetting('llamacpp_embedding_model', this.value)"
      oninput="clearTimeout(this._lem); var self=this; self._lem=setTimeout(function(){saveBootSetting('llamacpp_embedding_model',self.value)},800)" />
  </div>
</div>
```

#### 3e. Show/hide logic for `onTextProviderChange` function
Find the `onTextProviderChange` function. Add this line alongside the other `provider-fields-*` display toggles:
```javascript
document.getElementById('provider-fields-llamacpp').style.display = value === 'llamacpp' ? '' : 'none';
```

#### 3f. Show/hide logic for `onEmbeddingProviderChange` function
Find the `onEmbeddingProviderChange` function. Add this line alongside the `embed-fields-openai` toggle:
```javascript
document.getElementById('embed-fields-llamacpp').style.display = value === 'llamacpp' ? '' : 'none';
```

#### 3g. Settings loader (page load)
Find the section where saved settings are loaded on page load (look for `provider-fields-gemini` display toggle in the settings loader, NOT in onTextProviderChange). Add these lines:

After the `provider-fields-anthropic` display line:
```javascript
document.getElementById('provider-fields-llamacpp').style.display = textProvider === 'llamacpp' ? '' : 'none';
```

After the `embed-fields-openai` display line:
```javascript
document.getElementById('embed-fields-llamacpp').style.display = embedProvider === 'llamacpp' ? '' : 'none';
```

And in the same settings loader block, after the OpenAI embedding model loader, add:
```javascript
// Llama.cpp fields
var lcTextUrl = document.getElementById('input-llamacpp-text-url');
if (lcTextUrl && s.llamacpp_text_url)
    lcTextUrl.value = s.llamacpp_text_url;
var lcTextModel = document.getElementById('input-llamacpp-text-model');
if (lcTextModel && s.llamacpp_text_model)
    lcTextModel.value = s.llamacpp_text_model;
var lcEmbedUrl = document.getElementById('input-llamacpp-embedding-url');
if (lcEmbedUrl && s.llamacpp_embedding_url)
    lcEmbedUrl.value = s.llamacpp_embedding_url;
var lcEmbedModel = document.getElementById('input-llamacpp-embedding-model');
if (lcEmbedModel && s.llamacpp_embedding_model)
    lcEmbedModel.value = s.llamacpp_embedding_model;
```

### Step 4: Build

Run `npm run build` in the prism-mcp directory. There should be zero errors.

### Step 5: Verify

After building, restart Claude Code. Then:
1. Open the Prism dashboard (usually `http://localhost:3333`)
2. Go to the AI Providers / settings section
3. You should see "🦙 Llama.cpp" in both Text Provider and Embedding Provider dropdowns
4. Select it and configure your llama-server URLs and model aliases
5. Test with a `session_save_ledger` call to verify embeddings generate

### Important notes

- **Embedding dimensions**: Prism requires exactly 768-dim embeddings. `nomic-embed-text-v2-moe` outputs 768 natively. If you use a different embedding model, make sure it outputs 768 dims.
- **No API key needed**: The adapter hardcodes `apiKey: "none"` since llama-server doesn't require auth on localhost.
- **Model alias**: The model name you enter must match the `-a` (alias) flag you set when launching llama-server. If you didn't set an alias, use the model filename.
- **Two ports required**: You need two separate llama-server instances — one for text (with `--embedding` disabled) and one for embeddings (with `--embedding` enabled).
