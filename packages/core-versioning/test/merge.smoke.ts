import { average } from "../src/merge/strategies/average.ts";
import { taskArithmetic } from "../src/merge/strategies/taskArithmetic.ts";
import { ties } from "../src/merge/strategies/ties.ts";
import { slerp } from "../src/merge/strategies/slerp.ts";
import { getMergeStrategy, listMergeStrategies } from "../src/merge/registry.ts";
import type { WeightedModel } from "../src/merge/types.ts";
import type { ModelWeights } from "../src/model.ts";

function assert(condition: boolean, message: string) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  } else {
    console.log(`ok - ${message}`);
  }
}

function approxEqual(a: Float32Array | Float64Array | number[], b: number[], eps = 1e-6): boolean {
  return a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) < eps);
}

/** Small helper: wrap a plain array into a 1-D float32 tensor for test fixtures. */
function t(data: number[]): { dtype: "float32"; shape: number[]; data: Float32Array } {
  return { dtype: "float32", shape: [data.length], data: new Float32Array(data) };
}

function model(hash: string, w: Record<string, number[]>): WeightedModel {
  const weights: ModelWeights = {};
  for (const [k, v] of Object.entries(w)) weights[k] = t(v);
  return { hash, weights };
}

function baseWeights(w: Record<string, number[]>): ModelWeights {
  const weights: ModelWeights = {};
  for (const [k, v] of Object.entries(w)) weights[k] = t(v);
  return weights;
}

function main() {
  // --- Registry ---
  assert(listMergeStrategies().sort().join(",") === "average,confidence-weighted,slerp,task-arithmetic,ties", "registry lists all five strategies");
  assert(getMergeStrategy("average").name === "average", "getMergeStrategy() resolves a known strategy by name");
  let threwUnknown = false;
  try {
    getMergeStrategy("nonsense");
  } catch {
    threwUnknown = true;
  }
  assert(threwUnknown, "getMergeStrategy() throws on an unknown strategy name");

  // --- Average: hand-computable, now with real Tensor wrapping ---
  const modelA = model("a", { w: [1, 2, 3] });
  const modelB = model("b", { w: [3, 4, 5] });
  const avgResult = average.merge([modelA, modelB], {});
  assert(approxEqual(avgResult.w.data, [2, 3, 4]), "average of [1,2,3] and [3,4,5] is [2,3,4]");
  assert(avgResult.w.dtype === "float32", "average output preserves dtype from the input tensors");
  assert(JSON.stringify(avgResult.w.shape) === JSON.stringify([3]), "average output preserves shape from the input tensors");

  // --- Task arithmetic: base + lambda * sum(deltas) ---
  const base = baseWeights({ w: [0, 0, 0] });
  const branchA = model("a", { w: [2, 0, -2] }); // delta = [2,0,-2]
  const branchB = model("b", { w: [0, 4, 2] }); // delta = [0,4,2]
  const taResult = taskArithmetic.merge([branchA, branchB], { base });
  assert(approxEqual(taResult.w.data, [1, 2, 0]), "task-arithmetic with default lambda=1/n merges deltas correctly");

  const taResultLambda1 = taskArithmetic.merge([branchA, branchB], { base, lambda: 1 });
  assert(approxEqual(taResultLambda1.w.data, [2, 4, 0]), "task-arithmetic with lambda=1 sums deltas without averaging");

  let taThrew = false;
  try {
    taskArithmetic.merge([branchA], {});
  } catch {
    taThrew = true;
  }
  assert(taThrew, "task-arithmetic throws when no base is provided");

  // --- TIES: sign election + disjoint merge, hand-computable ---
  const tiesBase = baseWeights({ w: [0, 0] });
  const tiesA = model("a", { w: [5, 1] }); // delta [5, 1]
  const tiesB = model("b", { w: [-1, 3] }); // delta [-1, 3]
  const tiesResult = ties.merge([tiesA, tiesB], { base: tiesBase, trimFraction: 1, lambda: 1 });
  assert(approxEqual(tiesResult.w.data, [5, 2]), "ties elects the higher-magnitude sign and disjoint-merges agreeing values");

  let tiesThrew = false;
  try {
    ties.merge([tiesA, tiesB], {});
  } catch {
    tiesThrew = true;
  }
  assert(tiesThrew, "ties throws when no base is provided");

  // --- SLERP: t=0 returns model A, t=1 returns model B ---
  const slerpA = model("a", { w: [1, 0] });
  const slerpB = model("b", { w: [0, 1] });
  const slerpT0 = slerp.merge([slerpA, slerpB], { t: 0 });
  assert(approxEqual(slerpT0.w.data, [1, 0], 1e-4), "slerp at t=0 returns model A");
  const slerpT1 = slerp.merge([slerpA, slerpB], { t: 1 });
  assert(approxEqual(slerpT1.w.data, [0, 1], 1e-4), "slerp at t=1 returns model B");
  const slerpMid = slerp.merge([slerpA, slerpB], { t: 0.5 });
  const midNorm = Math.sqrt(slerpMid.w.data[0] ** 2 + slerpMid.w.data[1] ** 2);
  assert(Math.abs(midNorm - 1) < 1e-6, "slerp at t=0.5 between two unit vectors stays on the unit sphere (norm 1)");
  assert(Math.abs(slerpMid.w.data[0] - slerpMid.w.data[1]) < 1e-6, "slerp at t=0.5 between orthogonal unit vectors is symmetric");

  const parallelA = model("a", { w: [1, 0] });
  const parallelB = model("b", { w: [1, 1e-9] });
  const parallelResult = slerp.merge([parallelA, parallelB], { t: 0.5 });
  assert(
    Number.isFinite(parallelResult.w.data[0]) && Number.isFinite(parallelResult.w.data[1]),
    "slerp on nearly-parallel vectors doesn't produce NaN/Infinity",
  );

  let slerpThrew = false;
  try {
    slerp.merge([slerpA, slerpB, slerpA], { t: 0.5 });
  } catch {
    slerpThrew = true;
  }
  assert(slerpThrew, "slerp throws when given anything other than exactly 2 models");

  // --- Shape mismatch guard, shared across strategies ---
  let mismatchThrew = false;
  try {
    average.merge([model("a", { w: [1, 2] }), model("b", { w: [1, 2, 3] })], {});
  } catch {
    mismatchThrew = true;
  }
  assert(mismatchThrew, "merging models with mismatched shapes throws instead of silently misaligning");

  if (process.exitCode === 1) {
    console.error("\nSome checks failed.");
  } else {
    console.log("\nAll checks passed.");
  }
}

main();
