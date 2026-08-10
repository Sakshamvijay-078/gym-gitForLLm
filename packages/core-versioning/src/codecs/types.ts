import type { ModelWeights } from "../model.ts";

/**
 * A ModelCodec translates between on-disk bytes for one specific format
 * and the canonical ModelWeights representation. This is what lets
 * BlobStore/ManifestStore stay completely format-agnostic (they only ever
 * see bytes) while merge strategies stay completely format-agnostic in
 * the other direction (they only ever see ModelWeights). Adding a new
 * format later — ONNX, GGUF — means writing one file implementing this
 * interface and registering it; nothing else in the system changes.
 */
export interface ModelCodec {
  format: string;
  /** File extensions this codec claims, lowercase, with the leading dot. */
  extensions: string[];
  encode(weights: ModelWeights): Buffer;
  decode(bytes: Buffer): ModelWeights;
}
