import type { MergeOptions, MergeStrategy, ModelWeights, WeightedModel } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, scale, sub } from "../vectorMath.ts";

/**
 * Confidence-Weighted Merge — decides how much each branch's weights
 * should count based on THAT branch's own training signals, instead of
 * treating every branch equally (plain average / plain TIES).
 *
 * Three signals, combined multiplicatively into one confidence score per
 * branch, then renormalized to sum to 1:
 *
 *   - dataset size   — proportional share of total examples trained on.
 *                       Same principle FedAvg uses to weight client
 *                       updates by local dataset size.
 *   - validation metric — proportional share of total score. A branch
 *                       that measurably performs better should count for
 *                       more, not be averaged down by a weaker one.
 *   - dataset type    — NOT a magnitude, so not auto-computed. A manual
 *                       trust multiplier via `typeTrust`, since "this
 *                       data source is more reliable than that one" is a
 *                       judgment call, not something derivable from the
 *                       numbers alone.
 *
 * Both `sizeWeight` and `metricWeight` are exponents on the normalized
 * share (0 disables that signal entirely, 1 is proportional, >1 sharpens
 * it toward whichever branch already leads). Missing info on a branch
 * falls back to neutral — with no info supplied for any branch, this
 * degrades exactly to equal weighting.
 *
 * Adapts its OWN behavior depending on whether a base is given:
 *   - no base  -> confidence-weighted average (generalizes `average`)
 *   - base given -> confidence-weighted TIES: trim, then elect sign and
 *     disjoint-merge using confidence instead of raw magnitude, so a
 *     branch with more/better data can win a sign conflict even against
 *     a branch with a locally larger raw delta.
 */
export const confidenceWeighted: MergeStrategy = {
  name: "confidence-weighted",
  requiresBase: false,
  minModels: 2,
  maxModels: null,

  merge(models, options) {
    const confidences = computeConfidences(models, options);

    if (!options.base) {
      return weightedAverage(models, confidences);
    }
    return confidenceWeightedTies(models, options.base, confidences, options);
  },
};

/** Proportional share of a signal across branches; missing values default to `fallback`. */
function normalizedShares(values: (number | undefined)[], fallback: number): number[] {
  const raw = values.map((v) => (v !== undefined && v > 0 ? v : fallback));
  const total = raw.reduce((a, b) => a + b, 0);
  if (total <= 0) return raw.map(() => 1 / raw.length);
  return raw.map((v) => v / total);
}

/** One confidence score per branch, normalized to sum to 1. */
function computeConfidences(models: WeightedModel[], options: MergeOptions): number[] {
  const sizeWeight = options.sizeWeight ?? 1;
  const metricWeight = options.metricWeight ?? 1;
  const typeTrust = options.typeTrust ?? {};

  const sizeShares = normalizedShares(models.map((m) => m.info?.datasetSize), 1);
  const metricShares = normalizedShares(models.map((m) => m.info?.validationMetric), 1);

  const raw = models.map((m, i) => {
    const trust = m.info?.datasetType !== undefined ? (typeTrust[m.info.datasetType] ?? 1) : 1;
    return trust * Math.pow(sizeShares[i], sizeWeight) * Math.pow(metricShares[i], metricWeight);
  });

  const total = raw.reduce((a, b) => a + b, 0);
  if (total <= 0) return models.map(() => 1 / models.length);
  return raw.map((v) => v / total);
}

/** No base: a straight confidence-weighted mean per tensor. */
function weightedAverage(models: WeightedModel[], confidences: number[]): ModelWeights {
  const weightsList = models.map((m) => m.weights);
  assertCompatible(weightsList);
  const keys = Object.keys(weightsList[0]);
  const result: ModelWeights = {};

  for (const key of keys) {
    const len = weightsList[0][key].data.length;
    const merged = new Array(len).fill(0);
    for (let m = 0; m < models.length; m++) {
      const vec = weightsList[m][key].data;
      for (let i = 0; i < len; i++) merged[i] += confidences[m] * vec[i];
    }
    result[key] = makeTensorLike(weightsList[0][key], merged);
  }
  return result;
}

/** Base given: TIES' trim/elect/disjoint-merge, but weighted by confidence instead of uniform. */
function confidenceWeightedTies(
  models: WeightedModel[],
  base: ModelWeights,
  confidences: number[],
  options: MergeOptions,
): ModelWeights {
  const lambda = options.lambda ?? 1;
  const trimFraction = options.trimFraction ?? 0.2;

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
      const electedSign = electSignWeighted(valuesAtI, confidences);
      if (electedSign === 0) {
        merged[i] = 0;
        continue;
      }
      let weightedSum = 0;
      let weightSum = 0;
      for (let m = 0; m < valuesAtI.length; m++) {
        if (Math.sign(valuesAtI[m]) === electedSign) {
          weightedSum += confidences[m] * valuesAtI[m];
          weightSum += confidences[m];
        }
      }
      merged[i] = weightSum > 0 ? weightedSum / weightSum : 0;
    }

    result[key] = makeTensorLike(base[key], add(baseVec, scale(merged, lambda)));
  }

  return result;
}

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

/** Sign with the larger CONFIDENCE-weighted magnitude, not just raw magnitude. */
function electSignWeighted(values: number[], confidences: number[]): number {
  let positiveMass = 0;
  let negativeMass = 0;
  for (let m = 0; m < values.length; m++) {
    const v = values[m];
    if (v > 0) positiveMass += confidences[m] * v;
    else if (v < 0) negativeMass += confidences[m] * -v;
  }
  if (positiveMass === 0 && negativeMass === 0) return 0;
  return positiveMass >= negativeMass ? 1 : -1;
}
