/**
 * The canonical in-memory representation every codec decodes INTO and every
 * merge strategy operates ON. Codecs (safetensors, json, later onnx/gguf)
 * translate between this and a specific on-disk byte layout; strategies
 * never see raw bytes or know which format a model came from.
 *
 * dtype is deliberately narrow: math happens in float32 or float64. Reading
 * lower-precision on-disk formats (like safetensors' F16) upcasts to
 * float32 at decode time — same convention real merge tools (mergekit,
 * etc.) use, since averaging/interpolating in half precision loses more
 * than it's worth.
 */
export type DType = "float32" | "float64";

export interface Tensor {
  dtype: DType;
  shape: number[];
  data: Float32Array | Float64Array;
}

export type ModelWeights = Record<string, Tensor>;

export function shapesEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function tensorElementCount(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Build a new Tensor with the same dtype/shape as `reference` but new
 * data. Used by every merge strategy to wrap plain-number-array math
 * results back into a properly typed tensor without repeating the
 * dtype-to-constructor logic in four places.
 */
export function makeTensorLike(reference: Tensor, data: number[]): Tensor {
  const Ctor = reference.dtype === "float64" ? Float64Array : Float32Array;
  return { dtype: reference.dtype, shape: reference.shape, data: new Ctor(data) };
}
