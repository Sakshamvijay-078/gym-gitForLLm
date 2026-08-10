import type { ModelWeights } from "../model.ts";
export type { ModelWeights, Tensor, DType } from "../model.ts";
export { makeTensorLike } from "../model.ts";

export interface WeightedModel {
  /** The manifest hash this model's weights came from. */
  hash: string;
  weights: ModelWeights;
}

export interface MergeOptions {
  /** Common ancestor weights — required by task-arithmetic and TIES. */
  base?: ModelWeights;
  /** Global scale applied to the combined task vector (task-arithmetic, TIES). Default varies by strategy. */
  lambda?: number;
  /** Fraction of each task vector's parameters to keep by magnitude (TIES). Default 0.2. */
  trimFraction?: number;
  /** Interpolation factor between the two models, 0..1 (SLERP). Default 0.5. */
  t?: number;
}

export interface MergeStrategy {
  name: string;
  requiresBase: boolean;
  minModels: number;
  maxModels: number | null;
  merge(models: WeightedModel[], options: MergeOptions): ModelWeights;
}

/**
 * Every model (and the base, if present) must share the exact same tensor
 * names, shapes, and element counts. Shape-aware now, not just length —
 * a real safetensors checkpoint carries shape info and a merge across
 * mismatched architectures should fail loudly, not silently misalign.
 */
export function assertCompatible(modelsList: ModelWeights[]): void {
  if (modelsList.length === 0) return;
  const reference = modelsList[0];
  const keys = Object.keys(reference).sort();

  for (const weights of modelsList.slice(1)) {
    const otherKeys = Object.keys(weights).sort();
    if (JSON.stringify(keys) !== JSON.stringify(otherKeys)) {
      throw new Error(`Model tensor names don't match: [${keys}] vs [${otherKeys}]`);
    }
    for (const key of keys) {
      const a = reference[key];
      const b = weights[key];
      if (JSON.stringify(a.shape) !== JSON.stringify(b.shape)) {
        throw new Error(`Shape mismatch on "${key}": ${JSON.stringify(a.shape)} vs ${JSON.stringify(b.shape)}`);
      }
      if (a.data.length !== b.data.length) {
        throw new Error(`Element count mismatch on "${key}": ${a.data.length} vs ${b.data.length}`);
      }
    }
  }
}
