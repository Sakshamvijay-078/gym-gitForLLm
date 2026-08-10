import type { ModelCodec } from "./types.ts";
import type { ModelWeights, DType } from "../model.ts";

interface JsonTensor {
  dtype: DType;
  shape: number[];
  data: number[];
}

/**
 * Plain JSON — not a real ML interchange format, kept deliberately for
 * synthetic test fixtures and quick demos where hand-writing a binary
 * .safetensors file by hand would be painful. Anything that matters
 * should be committed as .safetensors instead.
 */
export const jsonCodec: ModelCodec = {
  format: "json",
  extensions: [".json"],

  encode(weights) {
    const out: Record<string, JsonTensor> = {};
    for (const [name, tensor] of Object.entries(weights)) {
      out[name] = { dtype: tensor.dtype, shape: tensor.shape, data: Array.from(tensor.data) };
    }
    return Buffer.from(JSON.stringify(out));
  },

  decode(bytes) {
    const raw = JSON.parse(bytes.toString("utf-8")) as Record<string, JsonTensor>;
    const weights: ModelWeights = {};
    for (const [name, t] of Object.entries(raw)) {
      const Ctor = t.dtype === "float64" ? Float64Array : Float32Array;
      weights[name] = { dtype: t.dtype ?? "float32", shape: t.shape, data: new Ctor(t.data) };
    }
    return weights;
  },
};
