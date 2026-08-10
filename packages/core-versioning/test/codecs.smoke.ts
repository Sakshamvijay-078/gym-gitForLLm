import { safetensorsCodec } from "../src/codecs/safetensors.ts";
import { jsonCodec } from "../src/codecs/json.ts";
import { getCodecByExtension, getCodecByFormat, listFormats } from "../src/codecs/registry.ts";
import type { ModelWeights } from "../src/model.ts";

function assert(condition: boolean, message: string) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  } else {
    console.log(`ok - ${message}`);
  }
}

function approxEqual(a: ArrayLike<number>, b: number[], eps = 1e-5): boolean {
  return a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) < eps);
}

function float32ToUint16Bits(f: number): number {
  // Reference F16 encoder used only to build a test fixture buffer —
  // not part of the codec itself, which only needs to decode F16.
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = f;
  const x = int32View[0];
  const sign = (x >> 16) & 0x8000;
  let exponent = ((x >> 23) & 0xff) - 127 + 15;
  let mantissa = x & 0x7fffff;
  if (exponent <= 0) return sign;
  if (exponent >= 0x1f) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >> 13);
}

function main() {
  // --- safetensors: encode then decode returns the same tensors ---
  const original: ModelWeights = {
    "layer.weight": { dtype: "float32", shape: [2, 2], data: new Float32Array([1.5, -2.25, 3, 0]) },
    "layer.bias": { dtype: "float32", shape: [2], data: new Float32Array([0.1, -0.1]) },
  };
  const encoded = safetensorsCodec.encode(original);
  assert(encoded.length > 8, "safetensors encode() produces a non-trivial byte buffer");

  const headerLen = Number(encoded.readBigUInt64LE(0));
  assert(headerLen > 0 && headerLen < encoded.length, "the 8-byte header-length prefix correctly points inside the buffer");

  const decoded = safetensorsCodec.decode(encoded);
  assert(
    approxEqual(decoded["layer.weight"].data, [1.5, -2.25, 3, 0]),
    "safetensors round-trip preserves layer.weight values exactly",
  );
  assert(
    JSON.stringify(decoded["layer.weight"].shape) === JSON.stringify([2, 2]),
    "safetensors round-trip preserves tensor shape",
  );
  assert(approxEqual(decoded["layer.bias"].data, [0.1, -0.1]), "safetensors round-trip preserves layer.bias values");
  assert(decoded["layer.weight"].dtype === "float32", "safetensors round-trip preserves float32 dtype");

  // --- safetensors: float64 tensors round-trip too ---
  const withDoubles: ModelWeights = {
    precise: { dtype: "float64", shape: [3], data: new Float64Array([1 / 3, 2 / 3, 1]) },
  };
  const decodedDoubles = safetensorsCodec.decode(safetensorsCodec.encode(withDoubles));
  assert(
    approxEqual(decodedDoubles.precise.data, [1 / 3, 2 / 3, 1], 1e-12),
    "safetensors preserves float64 precision, not just float32",
  );
  assert(decodedDoubles.precise.dtype === "float64", "safetensors round-trip preserves float64 dtype specifically");

  // --- safetensors: F16 read support (build a real F16 fixture by hand) ---
  const f16Values = [1, -1, 0.5, 2.5];
  const dataBuf = Buffer.alloc(f16Values.length * 2);
  f16Values.forEach((v, i) => dataBuf.writeUInt16LE(float32ToUint16Bits(v), i * 2));
  const header = { "half.tensor": { dtype: "F16", shape: [4], data_offsets: [0, dataBuf.length] } };
  const headerBytes = Buffer.from(JSON.stringify(header));
  const headerLenBuf = Buffer.alloc(8);
  headerLenBuf.writeBigUInt64LE(BigInt(headerBytes.length));
  const f16File = Buffer.concat([headerLenBuf, headerBytes, dataBuf]);

  const decodedF16 = safetensorsCodec.decode(f16File);
  assert(approxEqual(decodedF16["half.tensor"].data, f16Values, 1e-2), "safetensors decodes F16 tensors, upcasting to float32");
  assert(decodedF16["half.tensor"].dtype === "float32", "F16 tensors are recorded as float32 after upcast, not left as F16");

  // --- safetensors: malformed input fails loudly, not silently ---
  let threwOnGarbage = false;
  try {
    safetensorsCodec.decode(Buffer.from([1, 2, 3]));
  } catch {
    threwOnGarbage = true;
  }
  assert(threwOnGarbage, "decode() throws on a buffer too short to be a valid safetensors file");

  // --- JSON codec round-trip, for parity ---
  const jsonDecoded = jsonCodec.decode(jsonCodec.encode(original));
  assert(approxEqual(jsonDecoded["layer.weight"].data, [1.5, -2.25, 3, 0]), "json codec round-trips values correctly too");

  // --- Registry ---
  assert(listFormats().sort().join(",") === "json,safetensors", "registry lists both codecs");
  assert(getCodecByFormat("safetensors").format === "safetensors", "getCodecByFormat resolves by exact name");
  assert(getCodecByExtension("model.safetensors")?.format === "safetensors", "getCodecByExtension resolves .safetensors files");
  assert(getCodecByExtension("weights.json")?.format === "json", "getCodecByExtension resolves .json files");
  assert(getCodecByExtension("checkpoint.pt") === null, "getCodecByExtension returns null for an unsupported format, not a guess");

  let threwUnknownFormat = false;
  try {
    getCodecByFormat("onnx");
  } catch {
    threwUnknownFormat = true;
  }
  assert(threwUnknownFormat, "getCodecByFormat throws clearly on a format with no codec yet, rather than pretending to support it");

  if (process.exitCode === 1) {
    console.error("\nSome checks failed.");
  } else {
    console.log("\nAll checks passed.");
  }
}

main();
