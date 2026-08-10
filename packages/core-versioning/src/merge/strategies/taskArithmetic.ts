import type { MergeStrategy, ModelWeights } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, scale, sub } from "../vectorMath.ts";

/**
 * Task Arithmetic (Ilharco et al., 2022): treat each branch's change from
 * the shared base as a "task vector" — a direction in weight space that
 * encodes what that branch learned — then combine task vectors and apply
 * them on top of the base.
 *
 *   task_i = model_i - base
 *   merged = base + lambda * sum(task_i)
 *
 * This is where the gym's lineage graph earns its keep: the "base" here
 * is exactly commonAncestor(branchA, branchB) from the manifest DAG — the
 * checkpoint both branches actually diverged from — so it never has to be
 * supplied by guesswork.
 */
export const taskArithmetic: MergeStrategy = {
  name: "task-arithmetic",
  requiresBase: true,
  minModels: 1,
  maxModels: null,

  merge(models, options) {
    if (!options.base) {
      throw new Error("task-arithmetic requires a base (the common-ancestor checkpoint)");
    }
    const base = options.base;
    const lambda = options.lambda ?? 1 / models.length;

    assertCompatible([base, ...models.map((m) => m.weights)]);

    const keys = Object.keys(base);
    const result: ModelWeights = {};
    for (const key of keys) {
      const baseVec = Array.from(base[key].data);
      const taskVectors = models.map((m) => sub(Array.from(m.weights[key].data), baseVec));
      let combined = new Array(baseVec.length).fill(0);
      for (const tv of taskVectors) combined = add(combined, tv);
      result[key] = makeTensorLike(base[key], add(baseVec, scale(combined, lambda)));
    }
    return result;
  },
};
