import type { MergeOptions, MergeStrategy, ModelWeights, WeightedModel } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, norm, scale, softmax, sub, weightedMean } from "../vectorMath.ts";

/**
 * Confidence-Weighted Merge — a smarter-than-average blending strategy
 * that weights each branch's contribution by three multiplicative signals
 * (dataset size, validation metric, and a manual dataset-type trust factor),
 * then routes to the right merge mode based on what information is available.
 *
 * ─── THE CORE FIX (v2) ──────────────────────────────────────────────────────
 *
 * The original `weightedAverage` blended raw absolute weight tensors.  This
 * made the imbalanced case degenerate: a branch trained for only ~15 steps
 * barely moves its weights from the random init, so even a 50/50 average is
 * really `0.5 · A + 0.5 · (≈ root)` — confidence-weighting the coefficients
 * made no observable difference because the quantity being weighted was already
 * negligible.
 *
 * Fix: when a base is available, BOTH merge paths operate on task vectors
 * (Δ = model − base), not raw weights.  Task vectors make "how much did this
 * branch actually change" visible and separable from initialization noise.
 * Only when no base is present do we fall back to raw-weight averaging, which
 * is the honest no-information baseline (identical to FedAvg).
 *
 * ─── THREE MERGE MODES ──────────────────────────────────────────────────────
 *
 *  1. no base  →  confidence-weighted FedAvg over raw weights (null hypothesis)
 *  2. base, no TIES  →  confidence-weighted task-vector average:
 *       merged = base + λ · Σᵢ cᵢ · Δᵢ       (where cᵢ are normalized confidences)
 *  3. base + TIES  →  confidence-weighted TIES (trim → elect sign → disjoint merge)
 *       sign election uses confidence-weighted mass, not raw magnitude,
 *       so a branch with more/better data can win even against a larger raw Δ.
 *
 * ─── CONFIDENCE SCORING ─────────────────────────────────────────────────────
 *
 *  raw_i = typeTrust_i · (sizeShare_i)^sizeWeight · (metricShare_i)^metricWeight
 *
 *  These raw scores are fed through a TEMPERATURE-SCALED SOFTMAX before being
 *  used as blending coefficients.  This replaces the old direct normalization,
 *  which caused extreme collapse (0.993 / 0.007 ratios) when one branch
 *  dominated both signals at once.  With `confidenceTemp > 1` the mix stays
 *  meaningful even for strongly imbalanced branches; set `confidenceTemp = 0.5`
 *  to sharpen it when you want near-winner-takes-all behaviour.
 *
 *  Missing signals → neutral (1), so with zero metadata this degrades exactly
 *  to equal weighting.
 *
 * ─── PER-BRANCH ADAPTIVE TRIM ───────────────────────────────────────────────
 *
 *  A fixed global trimFraction treats an under-trained branch (small Δ) the
 *  same as a well-trained one (large Δ) — both keep the top-20% by magnitude.
 *  But a branch that barely moved keeps only noise in its top-20%.
 *
 *  With `adaptiveTrim = true` (default when base is given), each branch's keep
 *  fraction is scaled by its relative delta norm:
 *
 *    keepFrac_i = trimFraction · (‖Δᵢ‖ / max_j ‖Δⱼ‖)
 *
 *  This means a branch with 10% of the max delta norm keeps only 10% * 20% = 2%
 *  of its parameters — correctly representing that most of its entries are noise
 *  relative to the branches that actually trained more.
 *
 * ─── L2-NORMALIZED TASK VECTORS (optional) ──────────────────────────────────
 *
 *  Set `normalizeTaskVectors = true` to unit-normalize each branch's Δ before
 *  merging.  This separates "direction of learning" from "magnitude of update"
 *  — useful when branches have very different learning rates or step counts and
 *  you want confidence to be the sole magnitude signal.  The combined task vector
 *  is then re-scaled by the confidence-weighted mean of the original norms.
 */
export const confidenceWeighted: MergeStrategy = {
  name: "confidence-weighted",
  requiresBase: false,
  minModels: 2,
  maxModels: null,

  merge(models, options) {
    const confidences = computeConfidences(models, options);

    if (!options.base) {
      // Mode 1 — no lineage info: honest FedAvg over raw weights.
      return rawWeightedAverage(models, confidences);
    }

    if (!options.ties) {
      // Mode 2 — task-vector average (THE main fix for imbalanced branches).
      return taskVectorAverage(models, options.base, confidences, options);
    }

    // Mode 3 — confidence-weighted TIES.
    return confidenceWeightedTies(models, options.base, confidences, options);
  },
};

// ─── Confidence computation ────────────────────────────────────────────────

/** Proportional share of a signal across branches; missing values default to `fallback`. */
function normalizedShares(values: (number | undefined)[], fallback: number): number[] {
  const raw = values.map((v) => (v !== undefined && v > 0 ? v : fallback));
  const total = raw.reduce((a, b) => a + b, 0);
  if (total <= 0) return raw.map(() => 1 / raw.length);
  return raw.map((v) => v / total);
}

/**
 * One confidence score per branch, normalized via temperature-scaled softmax.
 *
 * Key change from v1: the final normalization uses softmax(temperature) instead
 * of simple division.  At temperature=1 (default) this is nearly identical for
 * balanced branches, but avoids the extreme collapse (e.g. 0.993 / 0.007) that
 * occurs when one branch strongly dominates both size AND metric signals.
 *
 * confidenceTemp > 1 → flatter (safer for imbalanced experiments)
 * confidenceTemp < 1 → sharper (for deliberate winner-takes-all blending)
 */
function computeConfidences(models: WeightedModel[], options: MergeOptions): number[] {
  const sizeWeight = options.sizeWeight ?? 1;
  const metricWeight = options.metricWeight ?? 1;
  const typeTrust = options.typeTrust ?? {};
  const temperature = options.confidenceTemp ?? 1.0;

  const sizeShares = normalizedShares(
    models.map((m) => m.info?.datasetSize),
    1,
  );
  const metricShares = normalizedShares(
    models.map((m) => m.info?.validationMetric),
    1,
  );

  // Raw log-scores before softmax (product becomes sum in log space).
  const logScores = models.map((m, i) => {
    const trust = m.info?.datasetType !== undefined ? (typeTrust[m.info.datasetType] ?? 1) : 1;
    // log of: trust * sizeShare^sizeWeight * metricShare^metricWeight
    return (
      Math.log(Math.max(trust, 1e-9)) +
      sizeWeight * Math.log(Math.max(sizeShares[i], 1e-9)) +
      metricWeight * Math.log(Math.max(metricShares[i], 1e-9))
    );
  });

  return softmax(logScores, temperature);
}

// ─── Mode 1: raw-weight FedAvg (no base) ──────────────────────────────────

/** No base available: plain confidence-weighted mean of raw weight tensors. */
function rawWeightedAverage(models: WeightedModel[], confidences: number[]): ModelWeights {
  const weightsList = models.map((m) => m.weights);
  assertCompatible(weightsList);
  const keys = Object.keys(weightsList[0]);
  const result: ModelWeights = {};

  for (const key of keys) {
    const vecs = weightsList.map((w) => Array.from(w[key].data));
    result[key] = makeTensorLike(weightsList[0][key], weightedMean(vecs, confidences));
  }
  return result;
}

// ─── Mode 2: task-vector average (THE core fix) ────────────────────────────

/**
 * Confidence-weighted average in TASK-VECTOR space.
 *
 *   merged = base + λ · Σᵢ cᵢ · Δᵢ
 *
 * This is the correct formulation when a base is known.  Operating on Δ
 * (model − base) instead of raw weights makes a branch's actual training
 * signal visible: a branch that only moved a tiny amount from init will
 * have a small-norm Δ, and therefore a small influence on the merge result
 * REGARDLESS of its raw weight values (which are dominated by initialization).
 *
 * Optional features:
 *  - adaptiveTrim (default true): scale each branch's trim fraction by its
 *    relative delta norm so barely-trained branches are trimmed more aggressively.
 *  - normalizeTaskVectors: unit-normalize each Δ before blending (separates
 *    direction from magnitude; re-scales by confidence-weighted original norm).
 */
function taskVectorAverage(
  models: WeightedModel[],
  base: ModelWeights,
  confidences: number[],
  options: MergeOptions,
): ModelWeights {
  const lambda = options.lambda ?? 1;
  const adaptiveTrim = options.adaptiveTrim ?? true;
  const trimFraction = options.trimFraction ?? 0; // 0 = no trim by default for this mode
  const normalize = options.normalizeTaskVectors ?? false;

  assertCompatible([base, ...models.map((m) => m.weights)]);

  const keys = Object.keys(base);
  const result: ModelWeights = {};

  for (const key of keys) {
    const baseVec = Array.from(base[key].data);
    const taskVectors = models.map((m) => sub(Array.from(m.weights[key].data), baseVec));

    // Per-branch delta norms (needed for adaptive trim and normalization).
    const deltaNorms = taskVectors.map((tv) => norm(tv));
    const maxNorm = Math.max(...deltaNorms, 1e-12);

    let processedVectors = taskVectors;

    // Optional: unit-normalize each task vector so confidence is the only
    // magnitude signal.  Re-scale the result by the confidence-weighted norm.
    if (normalize) {
      const normalizedNorm = confidences.reduce((s, c, i) => s + c * deltaNorms[i], 0);
      processedVectors = taskVectors.map((tv, i) => {
        const n = deltaNorms[i];
        return n > 1e-12 ? scale(tv, normalizedNorm / n) : tv;
      });
    }

    // Optional trim per branch, adapted to its delta magnitude.
    if (trimFraction > 0) {
      processedVectors = processedVectors.map((tv, i) => {
        const branchFrac = adaptiveTrim
          ? trimFraction * (deltaNorms[i] / maxNorm) // shrink trim for small deltas
          : trimFraction;
        return branchFrac > 0 ? trimVector(tv, branchFrac) : tv;
      });
    }

    // Confidence-weighted sum of task vectors.
    const combined = weightedMean(processedVectors, confidences);
    result[key] = makeTensorLike(base[key], add(baseVec, scale(combined, lambda)));
  }
  return result;
}

// ─── Mode 3: confidence-weighted TIES ──────────────────────────────────────

/**
 * TIES-merging with confidence-weighted sign election and disjoint merge.
 *
 * Improvements vs v1:
 *  - Adaptive per-branch trim fraction (same logic as taskVectorAverage).
 *  - electSignWeighted uses confidence-weighted mass, unchanged — this is
 *    already correct for the balanced case.
 *  - Disjoint merge denominator uses confidence weights, not raw counts,
 *    so a high-confidence branch that agrees on sign gets proportionally more
 *    influence on the final merged value.
 */
function confidenceWeightedTies(
  models: WeightedModel[],
  base: ModelWeights,
  confidences: number[],
  options: MergeOptions,
): ModelWeights {
  const lambda = options.lambda ?? 1;
  const trimFraction = options.trimFraction ?? 0.2;
  const adaptiveTrim = options.adaptiveTrim ?? true;

  assertCompatible([base, ...models.map((m) => m.weights)]);

  const keys = Object.keys(base);
  const result: ModelWeights = {};

  for (const key of keys) {
    const baseVec = Array.from(base[key].data);
    const taskVectors = models.map((m) => sub(Array.from(m.weights[key].data), baseVec));

    const deltaNorms = taskVectors.map((tv) => norm(tv));
    const maxNorm = Math.max(...deltaNorms, 1e-12);

    // Adaptive trim: branches with smaller deltas are trimmed more aggressively.
    const trimmed = taskVectors.map((tv, i) => {
      const branchFrac = adaptiveTrim
        ? trimFraction * (deltaNorms[i] / maxNorm)
        : trimFraction;
      return branchFrac > 0 ? trimVector(tv, branchFrac) : tv;
    });

    const len = baseVec.length;
    const merged = new Array(len).fill(0);

    for (let i = 0; i < len; i++) {
      const valuesAtI = trimmed.map((tv) => tv[i]);
      const electedSign = electSignWeighted(valuesAtI, confidences);
      if (electedSign === 0) {
        merged[i] = 0;
        continue;
      }

      // Disjoint merge: only include branches that agree on sign.
      // Weight by confidence, not by count — high-confidence branches
      // contribute proportionally more to the final value.
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

// ─── Shared helpers ─────────────────────────────────────────────────────────

/**
 * Keep only the top-`keepFraction` largest-magnitude entries; zero the rest.
 * `keepFraction = 0` is a no-op (keeps everything), `keepFraction = 1` drops
 * everything except the single largest element.
 *
 * Note: the parameter name is KEEP fraction (not trim fraction) — pass 0.2 to
 * keep 20% (trim 80%).  This matches the TIES paper convention.
 */
function trimVector(vector: number[], keepFraction: number): number[] {
  const n = vector.length;
  const keepCount = Math.max(1, Math.round(n * keepFraction));
  if (keepCount >= n) return vector; // nothing to trim

  const sortedIndices = vector
    .map((v, i) => [Math.abs(v), i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, keepCount)
    .map(([, i]) => i);

  const keepSet = new Set(sortedIndices);
  return vector.map((v, i) => (keepSet.has(i) ? v : 0));
}

/**
 * Elect the sign at position i using CONFIDENCE-WEIGHTED total mass,
 * not raw magnitude.  A branch with 0.9 confidence that says "positive"
 * beats three branches each with 0.033 confidence saying "negative",
 * which is the correct outcome when branch quality differs strongly.
 */
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
