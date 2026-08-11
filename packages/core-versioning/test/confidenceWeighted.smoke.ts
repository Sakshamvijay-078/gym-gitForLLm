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

  // --- Dataset size drives the weighting proportionally when scoreMode='proportional' ---
  const bigBranch = model("big", { w: [10, 10] }, { datasetSize: 300 });
  const smallBranch = model("small", { w: [0, 0] }, { datasetSize: 100 });
  // 300:100 -> proportional: confidence 0.75 : 0.25 -> weighted mean = 0.75*10 + 0.25*0 = 7.5
  const sizeResult = confidenceWeighted.merge([bigBranch, smallBranch], { metricWeight: 0, scoreMode: "proportional" });
  assert(approxEqual(sizeResult.w.data, [7.5, 7.5]), "dataset size proportionally weights the merge (300 vs 100 examples -> 75/25 split) when scoreMode=proportional");

  // --- sqrt mode (new default): 100:1 ratio does NOT collapse to near-zero confidence ---
  // With datasetSize 30000 vs 300 (100:1 imbalance ratio, like real Split-MNIST experiment):
  //   proportional: confidence_B = 300/30300 = 0.0099 (collapses to near-zero!)
  //   sqrt:         confidence_B = sqrt(300)/(sqrt(30000)+sqrt(300)) = 17.32/190.71 = 0.091
  // The sqrt mode ensures the small branch has at least ~9% influence, not 1%.
  const hugeBranch = model("huge", { w: [10, 10] }, { datasetSize: 30000 });
  const tinyBranch = model("tiny", { w: [0, 0] }, { datasetSize: 300 });
  // With proportional: result ≈ [9.9, 9.9] (B contributes 1%)
  // With sqrt (default): B should contribute ~9% -> result should be meaningfully < 9.9
  const sqrtResult = confidenceWeighted.merge([hugeBranch, tinyBranch], { metricWeight: 0 });
  const propResult = confidenceWeighted.merge([hugeBranch, tinyBranch], { metricWeight: 0, scoreMode: "proportional" });
  // sqrt result gives B more voice: sqrtResult[0] < propResult[0] (B pulls toward 0 more)
  assert(
    sqrtResult.w.data[0] < propResult.w.data[0],
    "sqrt scoreMode gives the small branch (100:1 ratio) more influence than proportional — prevents confidence collapse",
  );
  // Also verify the proportional mode collapses to near-zero (the bug we fixed):
  assert(
    propResult.w.data[0] > 9.8,
    "proportional mode correctly collapses small branch to near-zero influence at 100:1 ratio (the v2 bug, preserved for documentation)",
  );

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

  // With scoreMode='proportional': datasetSize 10:90 gives confidence 0.1:0.9
  // position 0: positiveMass=0.1*5=0.5, negativeMass=0.9*1=0.9 -> NEGATIVE wins (flipped!)
  // With scoreMode='sqrt' (default): sqrt(10)/(sqrt(10)+sqrt(90))=0.25, confidence 0.25:0.75
  // position 0: positiveMass=0.25*5=1.25, negativeMass=0.75*1=0.75 -> positive still wins (no flip at this ratio)
  // To flip with sqrt, need even stronger confidence imbalance: use 1:99 dataset sizes
  const tiesBase2 = baseWeights({ w: [0, 0] });
  const lowConfA_prop = model("a", { w: [5, 1] }, { datasetSize: 10 });
  const highConfB_prop = model("b", { w: [-1, 3] }, { datasetSize: 90 });
  const cwTiesResult = confidenceWeighted.merge([lowConfA_prop, highConfB_prop], {
    base: tiesBase2,
    trimFraction: 1,
    lambda: 1,
    metricWeight: 0,
    scoreMode: "proportional", // explicitly use proportional to get 10%/90% confidence
  });
  // position 0: positive mass = 0.1*5 = 0.5, negative mass = 0.9*1 = 0.9 -> NEGATIVE wins (flipped from plain ties!)
  //   only B agrees with elected sign -> merged[0] = B's value = -1
  // position 1: both positive, weighted mean = (0.1*1 + 0.9*3)/(0.1+0.9) = 2.8
  assert(
    approxEqual(cwTiesResult.w.data, [-1, 2.8]),
    "confidence-weighted ties FLIPS the elected sign vs plain ties when the low-magnitude branch has much higher confidence (scoreMode=proportional) — this is the whole point of the strategy",
  );

  // --- sqrt mode: verify sign-flip still works for extreme enough imbalance (1:999 ratio) ---
  // sqrt(1)/(sqrt(1)+sqrt(999)) = 1/32.6 = 0.031 confidence for A
  // position 0: positiveMass=0.031*5=0.155, negativeMass=0.969*1=0.969 -> NEGATIVE wins even with sqrt!
  const veryLowA = model("a", { w: [5, 1] }, { datasetSize: 1 });
  const veryHighB = model("b", { w: [-1, 3] }, { datasetSize: 999 });
  const cwTiesSqrtResult = confidenceWeighted.merge([veryLowA, veryHighB], {
    base: tiesBase2,
    trimFraction: 1,
    lambda: 1,
    metricWeight: 0,
    // default scoreMode = 'sqrt'
  });
  assert(
    cwTiesSqrtResult.w.data[0] < 0,
    "confidence-weighted ties still flips sign with sqrt scoreMode when imbalance is extreme enough (1:999 ratio)",
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

  // --- Norm-equalized mode: rescales small task vectors to prevent magnitude collapse ---
  const normBase = baseWeights({ w: [0, 0] });
  const largeDelta = model("large", { w: [10, 10] }, { datasetSize: 30000 }); // norm = sqrt(200) = 14.14
  const smallDelta = model("small", { w: [1, 1] }, { datasetSize: 300 });   // norm = sqrt(2) = 1.41
  // Without norm-equalization, 50/50 TV merge gives: base + 0.5*[10,10] + 0.5*[1,1] = [5.5, 5.5]
  // With normEqualizePower=1.0: meanNorm = (14.14+1.41)/2 = 7.77
  //   largeDelta gets scaled by 7.77/14.14 = 0.55 -> 0.55 * [10,10] = [5.5, 5.5]
  //   smallDelta gets scaled by 7.77/1.41 = 5.5 -> 5.5 * [1,1] = [5.5, 5.5]
  //   50/50 TV merge gives: 0.5*[5.5, 5.5] + 0.5*[5.5, 5.5] = [5.5, 5.5]
  const normEqResult = confidenceWeighted.merge([largeDelta, smallDelta], {
    base: normBase,
    ties: false,
    scoreMode: "norm-equalized",
    normEqualizePower: 1.0,
    metricWeight: 0,
  });
  assert(
    approxEqual(normEqResult.w.data, [5.5, 5.5]),
    "norm-equalized mode (power 1.0) equalizes task vector magnitudes so starved branch has equal energy in merge",
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
