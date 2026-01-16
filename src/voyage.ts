/**
 * voyage.ts - Voyage AI embedding provider for QMD
 *
 * Implements the LLM interface using Voyage AI's API for embeddings and reranking.
 * Requires VOYAGE_API_KEY environment variable.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  RerankDocumentResult,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

const VOYAGE_API_BASE = "https://api.voyageai.com/v1";

// Default models - voyage-4 series is latest
const DEFAULT_EMBED_MODEL = "voyage-3-lite";
const DEFAULT_RERANK_MODEL = "rerank-2";

// =============================================================================
// Types
// =============================================================================

interface VoyageEmbedRequest {
  input: string[];
  model: string;
  input_type?: "query" | "document";
  truncation?: boolean;
  output_dimension?: number;
  output_dtype?: "float" | "int8" | "uint8" | "binary" | "ubinary";
}

interface VoyageEmbedResponse {
  object: "list";
  data: Array<{
    object: "embedding";
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    total_tokens: number;
  };
}

interface VoyageRerankRequest {
  query: string;
  documents: string[];
  model: string;
  top_k?: number;
  truncation?: boolean;
}

interface VoyageRerankResponse {
  object: "list";
  data: Array<{
    index: number;
    relevance_score: number;
    document?: string;
  }>;
  model: string;
  usage: {
    total_tokens: number;
  };
}

// =============================================================================
// Voyage LLM Implementation
// =============================================================================

export type VoyageConfig = {
  apiKey?: string;
  embedModel?: string;
  rerankModel?: string;
  baseUrl?: string;
};

/**
 * LLM implementation using Voyage AI's API
 */
export class VoyageLLM implements LLM {
  private apiKey: string;
  private embedModel: string;
  private rerankModel: string;
  private baseUrl: string;

  constructor(config: VoyageConfig = {}) {
    this.apiKey = config.apiKey || process.env.VOYAGE_API_KEY || "";
    if (!this.apiKey) {
      throw new Error(
        "Voyage API key required. Set VOYAGE_API_KEY environment variable or pass apiKey in config."
      );
    }
    this.embedModel = config.embedModel || process.env.VOYAGE_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.rerankModel = config.rerankModel || process.env.VOYAGE_RERANK_MODEL || DEFAULT_RERANK_MODEL;
    this.baseUrl = config.baseUrl || VOYAGE_API_BASE;
  }

  /**
   * Make a request to Voyage API
   */
  private async request<T>(endpoint: string, body: object): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Voyage API error (${response.status}): ${error}`);
    }

    return response.json() as Promise<T>;
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const inputType = options.isQuery ? "query" : "document";

      const response = await this.request<VoyageEmbedResponse>("/embeddings", {
        input: [text],
        model: options.model || this.embedModel,
        input_type: inputType,
      } satisfies VoyageEmbedRequest);

      if (!response.data || response.data.length === 0) {
        return null;
      }

      return {
        embedding: response.data[0].embedding,
        model: response.model,
      };
    } catch (error) {
      console.error("Voyage embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Voyage supports up to 128 texts per request
   */
  async embedBatch(
    texts: string[],
    options: EmbedOptions = {}
  ): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const BATCH_SIZE = 128;
    const results: (EmbeddingResult | null)[] = [];
    const inputType = options.isQuery ? "query" : "document";

    // Process in batches
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);

      try {
        const response = await this.request<VoyageEmbedResponse>("/embeddings", {
          input: batch,
          model: options.model || this.embedModel,
          input_type: inputType,
        } satisfies VoyageEmbedRequest);

        // Map results back by index
        const batchResults: (EmbeddingResult | null)[] = new Array(batch.length).fill(null);
        for (const item of response.data) {
          batchResults[item.index] = {
            embedding: item.embedding,
            model: response.model,
          };
        }
        results.push(...batchResults);
      } catch (error) {
        console.error("Voyage batch embedding error:", error);
        // Fill with nulls for failed batch
        results.push(...new Array(batch.length).fill(null));
      }
    }

    return results;
  }

  /**
   * Text generation - not supported by Voyage
   * Falls back to simple query expansion without LLM
   */
  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult | null> {
    // Voyage doesn't do text generation
    // Return null to signal caller should handle differently
    console.warn("Voyage does not support text generation. Use a different provider for expandQuery.");
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // Voyage models are always "available" if API key works
    return {
      name: model,
      exists: true,
    };
  }

  /**
   * Expand query - simplified version without LLM generation
   * Returns the original query for vector search
   */
  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const results: Queryable[] = [];

    if (includeLexical) {
      results.push({ type: "lex", text: query });
    }
    results.push({ type: "vec", text: query });

    return results;
  }

  /**
   * Rerank documents using Voyage's reranker
   */
  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel };
    }

    try {
      const docTexts = documents.map((d) => d.text);

      const response = await this.request<VoyageRerankResponse>("/rerank", {
        query,
        documents: docTexts,
        model: options.model || this.rerankModel,
      } satisfies VoyageRerankRequest);

      const results: RerankDocumentResult[] = response.data.map((item) => ({
        file: documents[item.index].file,
        score: item.relevance_score,
        index: item.index,
      }));

      // Sort by score descending
      results.sort((a, b) => b.score - a.score);

      return {
        results,
        model: response.model,
      };
    } catch (error) {
      console.error("Voyage rerank error:", error);
      // Return original order with zero scores on error
      return {
        results: documents.map((d, i) => ({
          file: d.file,
          score: 0,
          index: i,
        })),
        model: this.rerankModel,
      };
    }
  }

  /**
   * Dispose - no-op for API-based provider
   */
  async dispose(): Promise<void> {
    // Nothing to dispose for HTTP-based provider
  }
}

// =============================================================================
// Singleton
// =============================================================================

let defaultVoyage: VoyageLLM | null = null;

/**
 * Get the default Voyage instance (creates one if needed)
 */
export function getDefaultVoyage(): VoyageLLM {
  if (!defaultVoyage) {
    defaultVoyage = new VoyageLLM();
  }
  return defaultVoyage;
}

/**
 * Set a custom default Voyage instance
 */
export function setDefaultVoyage(llm: VoyageLLM | null): void {
  defaultVoyage = llm;
}

/**
 * Dispose the default Voyage instance
 */
export async function disposeDefaultVoyage(): Promise<void> {
  if (defaultVoyage) {
    await defaultVoyage.dispose();
    defaultVoyage = null;
  }
}
