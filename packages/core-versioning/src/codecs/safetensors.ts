import type { ModelCodec } from "./types.ts";
import type { ModelWeights, DType } from "../model.ts";

/**
 * .safetensors layout (github.com/huggingface/safetensors):
 *   bytes[0..8)   — u64 little-endian: length N of the header in bytes
 *   bytes[8..8+N) — UTF-8 JSON header: tensor name -> { dtype, shape, data_offsets }
 *                   (plus an optional "__metadata__" string map, ignored here)
 *   bytes[8+N..]  — raw tensor bytes, back to back, at the offsets the
 *                   header specifies (relative to the start of this region)
 *
 * This is the format the whole model-merging ecosystem (mergekit, HF)
 * already treats as the interchange standard — real .pt/.onnx checkpoints
 * get converted TO this before merging, rather than merged directly.
 */

type OnDiskDType = "F64" | "F32" | "F16";

const ELEMENT_BYTES: Record<OnDiskDType, number> = { F64: 8, F32: 4, F16: 2 };

function float16ToFloat32(half: number): number {
  const sign = (half & 0x8000) >> 15;
  const exponent = (half & 0x7c00) >> 10;
  const fraction = half & 0x03ff;
  let value: number;
  if (exponent === 0) {
    value = fraction === 0 ? 0 : (fraction / 1024) * Math.pow(2, -14);
  } else if (exponent === 0x1f) {
    value = fraction ? NaN : Infinity;
  } else {
    value = (1 + fraction / 1024) * Math.pow(2, exponent - 15);
  }
  return sign ? -value : value;
}

export const safetensorsCodec: ModelCodec = {
  format: "safetensors",
  extensions: [".safetensors"],

  decode(bytes) {
    if (bytes.length < 8) {
      throw new Error("Not a valid .safetensors file: too short to contain a header length");
    }
    const headerLen = Number(bytes.readBigUInt64LE(0));
    if (8 + headerLen > bytes.length) {
      throw new Error("Not a valid .safetensors file: declared header length exceeds file size");
    }

    const header = JSON.parse(bytes.subarray(8, 8 + headerLen).toString("utf-8")) as Record<
      string,
      { dtype: OnDiskDType; shape: number[]; data_offsets: [number, number] } | Record<string, string>
    >;
    const dataStart = 8 + headerLen;

    const weights: ModelWeights = {};
    for (const [name, meta] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const entry = meta as { dtype: OnDiskDType; shape: number[]; data_offsets: [number, number] };
      const [start, end] = entry.data_offsets;
      const slice = bytes.subarray(dataStart + start, dataStart + end);

      if (entry.dtype === "F32") {
        const floats = new Float32Array(slice.byteLength / 4);
        for (let i = 0; i < floats.length; i++) floats[i] = slice.readFloatLE(i * 4);
        weights[name] = { dtype: "float32", shape: entry.shape, data: floats };
      } else if (entry.dtype === "F64") {
        const doubles = new Float64Array(slice.byteLength / 8);
        for (let i = 0; i < doubles.length; i++) doubles[i] = slice.readDoubleLE(i * 8);
        weights[name] = { dtype: "float64", shape: entry.shape, data: doubles };
      } else if (entry.dtype === "F16") {
        // Upcast to float32 on read — see the note on DType in model.ts.
        const count = slice.byteLength / 2;
        const floats = new Float32Array(count);
        for (let i = 0; i < count; i++) floats[i] = float16ToFloat32(slice.readUInt16LE(i * 2));
        weights[name] = { dtype: "float32", shape: entry.shape, data: floats };
      } else {
        throw new Error(
          `Unsupported safetensors dtype "${entry.dtype}" for tensor "${name}". ` +
            `Supported today: F32, F64, F16 (read-only, upcast to float32). ` +
            `BF16/int types aren't decoded yet — this is a real gap, not a silent one.`,
        );
      }
    }
    return weights;
  },

  encode(weights) {
    // Merge output is always written as F32 or F64 matching each tensor's
    // in-memory dtype — never re-quantized to F16 here. Downcasting for
    // deployment (e.g. to F16 or GGUF) is a separate, later concern.
    const header: Record<string, { dtype: OnDiskDType; shape: number[]; data_offsets: [number, number] }> = {};
    const chunks: Buffer[] = [];
    let offset = 0;

    for (const [name, tensor] of Object.entries(weights)) {
      const dtype: OnDiskDType = tensor.dtype === "float64" ? "F64" : "F32";
      const bytesPerEl = ELEMENT_BYTES[dtype];
      const byteLength = tensor.data.length * bytesPerEl;

      const buf = Buffer.alloc(byteLength);
      if (dtype === "F64") {
        for (let i = 0; i < tensor.data.length; i++) buf.writeDoubleLE(tensor.data[i], i * 8);
      } else {
        for (let i = 0; i < tensor.data.length; i++) buf.writeFloatLE(tensor.data[i], i * 4);
      }
      chunks.push(buf);

      header[name] = { dtype, shape: tensor.shape, data_offsets: [offset, offset + byteLength] };
      offset += byteLength;
    }

    const headerBytes = Buffer.from(JSON.stringify(header), "utf-8");
    const headerLenBuf = Buffer.alloc(8);
    headerLenBuf.writeBigUInt64LE(BigInt(headerBytes.length));

    return Buffer.concat([headerLenBuf, headerBytes, ...chunks]);
  },
};
