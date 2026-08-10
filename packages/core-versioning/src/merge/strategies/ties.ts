import type { MergeStrategy, ModelWeights } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, scale, sub } from "../vectorMath.ts";

/**
 * TIES-merging (Yadav et al., 2023): task vectors from different branches
 * often disagree on sign for a given parameter — one branch pushes a
 * weight up, another pushes it down — and plain averaging cancels that
 * signal out instead of resolving it. TIES fixes this in three steps per
 * parameter:
 *
 *   1. Trim   — zero out all but the top-k% largest-magnitude entries of
 *               each task vector (drops noise, keeps only what each branch
 *               actually changed meaningfully).
 *   2. Elect  — for each parameter position, pick the sign with the larger
 *               total magnitude across branches ("majority vote weighted
 *               by how much each branch cares").
 *   3. Merge  — average only the trimmed values that agree with the
 *               elected sign; disagreeing/zeroed values are excluded, not
 *               just averaged in and diluted.
 *
 * Same base-from-lineage story as task-arithmetic: base is normally
 * commonAncestor(branchA, branchB).
 */
export const ties: MergeStrategy = {
  name: "ties",
  requiresBase: true,
  minModels: 2,
  maxModels: null,

  merge(models, options) {
    if (!options.base) {
      throw new Error("ties requires a base (the common-ancestor checkpoint)");
    }
    const base = options.base;
    const lambda = options.lambda ?? 1;
    const trimFraction = options.trimFraction ?? 0.25;

    assertCompatible([base, ...models.map((m) => m.weights)]);

    const keys = Object.keys(base);
    const result: ModelWeights = {};

    for (const key of keys) {
      const baseVec = Array.from(base[key].data);
      const taskVectors = models.map((m) => sub(Array.from(m.weights[key].data), baseVec));
      const trimmed = taskVectors.map((tv) => trim(tv, trimFraction));

      const len = baseVec.length;
      const merged = new Array(len).fill(0);

      for (let i = 0; i < len; i++) {
        const valuesAtI = trimmed.map((tv) => tv[i]);
        const electedSign = electSign(valuesAtI);
        if (electedSign === 0) {
          merged[i] = 0;
          continue;
        }
        const agreeing = valuesAtI.filter((v) => Math.sign(v) === electedSign);
        merged[i] = agreeing.length > 0 ? agreeing.reduce((s, v) => s + v, 0) / agreeing.length : 0;
      }

      result[key] = makeTensorLike(base[key], add(baseVec, scale(merged, lambda)));
    }

    return result;
  },
};

/** Keep only the top-k% largest-magnitude entries of a vector; zero the rest. */
function trim(vector: number[], keepFraction: number): number[] {
  const n = vector.length;
  const keepCount = Math.max(1, Math.round(n * keepFraction));
  const sortedIndices = vector
    .map((v, i) => [Math.abs(v), i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, keepCount)
    .map(([, i]) => i);
  const keepSet = new Set(sortedIndices);
  return vector.map((v, i) => (keepSet.has(i) ? v : 0));
}

/** Sign with the larger total magnitude at this position; 0 if everything is 0. */
function electSign(values: number[]): number {
  let positiveMass = 0;
  let negativeMass = 0;
  for (const v of values) {
    if (v > 0) positiveMass += v;
    else if (v < 0) negativeMass += -v;
  }
  if (positiveMass === 0 && negativeMass === 0) return 0;
  return positiveMass >= negativeMass ? 1 : -1;
}
