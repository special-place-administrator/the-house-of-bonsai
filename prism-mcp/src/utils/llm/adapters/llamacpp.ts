/**
 * Llama.cpp Adapter
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *   Implements LLMProvider for llama-server (llama.cpp's OpenAI-compatible
 *   HTTP server). Supports SEPARATE base URLs for text and embedding models,
 *   which is the typical setup when running two llama-server instances on
 *   different ports (e.g. :8080 for text, :8081 for embeddings).
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
// nomic-embed-text-v2-moe has a 512 token training context.
// llama-server caps slot context to the model's training limit.
// 1800 chars ≈ 450 tokens — safely under the 512 token ceiling.
const MAX_EMBEDDING_CHARS = 1800;

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
