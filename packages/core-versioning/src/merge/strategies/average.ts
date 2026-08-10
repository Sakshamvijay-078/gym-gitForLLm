import type { MergeStrategy, ModelWeights } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { mean } from "../vectorMath.ts";

/**
 * Plain elementwise averaging (FedAvg-style). No base needed, no direction
 * information used — this is the null hypothesis for merging, the same
 * role round-robin plays for allocation. If a smarter strategy can't beat
 * this, it isn't earning its complexity.
 */
export const average: MergeStrategy = {
  name: "average",
  requiresBase: false,
  minModels: 2,
  maxModels: null,

  merge(models) {
    const weightsList = models.map((m) => m.weights);
    assertCompatible(weightsList);
    const keys = Object.keys(weightsList[0]);
    const result: ModelWeights = {};
    for (const key of keys) {
      const vectors = weightsList.map((w) => Array.from(w[key].data));
      result[key] = makeTensorLike(weightsList[0][key], mean(vectors));
    }
    return result;
  },
};
