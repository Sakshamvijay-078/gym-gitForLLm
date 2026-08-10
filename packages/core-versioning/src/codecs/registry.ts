import type { ModelCodec } from "./types.ts";
import { jsonCodec } from "./json.ts";
import { safetensorsCodec } from "./safetensors.ts";

/**
 * Adding a new format (ONNX, GGUF, ...) later: write one file implementing
 * ModelCodec, add it to this array. Nothing that calls getCodecByFormat()
 * or getCodecByExtension() needs to change — same plug-and-play shape as
 * the merge strategy registry and the allocate() policy interface.
 */
const codecs: ModelCodec[] = [jsonCodec, safetensorsCodec];

export function getCodecByFormat(format: string): ModelCodec {
  const codec = codecs.find((c) => c.format === format);
  if (!codec) {
    throw new Error(`Unknown model format "${format}". Available: ${listFormats().join(", ")}`);
  }
  return codec;
}

export function getCodecByExtension(filePath: string): ModelCodec | null {
  const lower = filePath.toLowerCase();
  return codecs.find((c) => c.extensions.some((ext) => lower.endsWith(ext))) ?? null;
}

export function listFormats(): string[] {
  return codecs.map((c) => c.format);
}
