import { confidenceWeighted } from "../src/merge/strategies/confidenceWeighted.ts";
import { ties } from "../src/merge/strategies/ties.ts";
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

function approxEqual(a: Float32Array | Float64Array | number[], b: number[], eps = 1e-5): boolean {
  return a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) < eps);
}

function t(data: number[]) {
  return { dtype: "float32" as const, shape: [data.length], data: new Float32Array(data) };
}

function model(hash: string, w: Record<string, number[]>, info?: WeightedModel["info"]): WeightedModel {
  const weights: ModelWeights = {};
  for (const [k, v] of Object.entries(w)) weights[k] = t(v);
  return { hash, weights, info };
}

function baseWeights(w: Record<string, number[]>): ModelWeights {
  const weights: ModelWeights = {};
  for (const [k, v] of Object.entries(w)) weights[k] = t(v);
  return weights;
}

function main() {
  // --- No info at all -> falls back to plain equal weighting (average) ---
  const plainA = model("a", { w: [1, 2, 3] });
  const plainB = model("b", { w: [3, 4, 5] });
  const noInfoResult = confidenceWeighted.merge([plainA, plainB], {});
  assert(approxEqual(noInfoResult.w.data, [2, 3, 4]), "with no branch info at all, confidence-weighted falls back to plain averaging");

  // --- Dataset size drives the weighting proportionally (metric disabled via metricWeight: 0) ---
  const bigBranch = model("big", { w: [10, 10] }, { datasetSize: 300 });
  const smallBranch = model("small", { w: [0, 0] }, { datasetSize: 100 });
  // 300:100 -> confidence 0.75 : 0.25 -> weighted mean = 0.75*10 + 0.25*0 = 7.5
  const sizeResult = confidenceWeighted.merge([bigBranch, smallBranch], { metricWeight: 0 });
  assert(approxEqual(sizeResult.w.data, [7.5, 7.5]), "dataset size proportionally weights the merge (300 vs 100 examples -> 75/25 split)");

  // --- Validation metric drives the weighting proportionally (size disabled via sizeWeight: 0) ---
  const goodBranch = model("good", { w: [1, 1] }, { validationMetric: 0.9 });
  const weakBranch = model("weak", { w: [0, 0] }, { validationMetric: 0.1 });
  // 0.9:0.1 proportional -> confidence 0.9 : 0.1 -> weighted mean = 0.9*1 + 0.1*0 = 0.9
  const metricResult = confidenceWeighted.merge([goodBranch, weakBranch], { sizeWeight: 0 });
  assert(approxEqual(metricResult.w.data, [0.9, 0.9]), "validation metric proportionally weights the merge (0.9 vs 0.1 -> 90/10 split)");

  // --- typeTrust is a manual override, applied on top of the other signals ---
  const trustedType = model("trusted", { w: [1, 1] }, { datasetType: "clean" });
  const untrustedType = model("untrusted", { w: [0, 0] }, { datasetType: "noisy" });
  const trustResult = confidenceWeighted.merge([trustedType, untrustedType], {
    sizeWeight: 0,
    metricWeight: 0,
    typeTrust: { clean: 4, noisy: 1 },
  });
  // no size/metric info -> equal base shares (0.5/0.5) -> raw confidence 4*0.5=2 vs 1*0.5=0.5 -> normalized 0.8/0.2
  assert(approxEqual(trustResult.w.data, [0.8, 0.8]), "typeTrust manually biases the merge toward the more-trusted dataset type (4x trust -> 80/20 split)");

  // --- The TIES sign-flip case: confidence can override raw magnitude ---
  // Same fixture as the plain-ties test (delta A=[5,1], delta B=[-1,3]), but
  // now branch A has only 10 examples and branch B has 90 -> confidence 0.1/0.9.
  const tiesBase = baseWeights({ w: [0, 0] });
  const lowConfidenceA = model("a", { w: [5, 1] }, { datasetSize: 10 }); // delta [5, 1], but low confidence
  const highConfidenceB = model("b", { w: [-1, 3] }, { datasetSize: 90 }); // delta [-1, 3], high confidence

  // Sanity: plain (unweighted) TIES on this same fixture elects the POSITIVE sign at position 0
  // (raw magnitude 5 > 1) and produces [5, 2] — this is the baseline the confidence version must differ from.
  const plainTiesResult = ties.merge(
    [
      { hash: "a", weights: baseWeights({ w: [5, 1] }) },
      { hash: "b", weights: baseWeights({ w: [-1, 3] }) },
    ],
    { base: tiesBase, trimFraction: 1, lambda: 1 },
  );
  assert(approxEqual(plainTiesResult.w.data, [5, 2]), "sanity check: plain ties on this fixture gives [5,2] (positive sign wins on raw magnitude)");

  const cwTiesResult = confidenceWeighted.merge([lowConfidenceA, highConfidenceB], {
    base: tiesBase,
    trimFraction: 1,
    lambda: 1,
    metricWeight: 0,
  });
  // position 0: positive mass = 0.1*5 = 0.5, negative mass = 0.9*1 = 0.9 -> NEGATIVE wins (flipped from plain ties!)
  //   only B agrees with elected sign -> merged[0] = B's value = -1
  // position 1: both positive, weighted mean = (0.1*1 + 0.9*3)/(0.1+0.9) = 2.8
  assert(
    approxEqual(cwTiesResult.w.data, [-1, 2.8]),
    "confidence-weighted ties FLIPS the elected sign vs plain ties when the low-magnitude branch has much higher confidence — this is the whole point of the strategy",
  );

  // --- Equal confidence reduces exactly to plain ties (backward-compatibility check) ---
  const equalA = model("a", { w: [5, 1] }, { datasetSize: 50 });
  const equalB = model("b", { w: [-1, 3] }, { datasetSize: 50 });
  const equalConfidenceResult = confidenceWeighted.merge([equalA, equalB], {
    base: tiesBase,
    trimFraction: 1,
    lambda: 1,
    metricWeight: 0,
  });
  assert(
    approxEqual(equalConfidenceResult.w.data, [5, 2]),
    "with equal confidence (50/50 dataset sizes), confidence-weighted ties matches plain ties exactly",
  );

  // --- Shape/compatibility guard still applies ---
  let mismatchThrew = false;
  try {
    confidenceWeighted.merge([model("a", { w: [1, 2] }), model("b", { w: [1, 2, 3] })], {});
  } catch {
    mismatchThrew = true;
  }
  assert(mismatchThrew, "confidence-weighted merge still rejects mismatched shapes");

  if (process.exitCode === 1) {
    console.error("\nSome checks failed.");
  } else {
    console.log("\nAll checks passed.");
  }
}

main();
