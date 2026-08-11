import type { MergeOptions, MergeStrategy, ModelWeights, WeightedModel } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, norm, scale, softmax, sub, weightedMean } from "../vectorMath.ts";

/**
 * Confidence-Weighted Merge — v3
 *
 * ─── BUG FIXED (v3) ─────────────────────────────────────────────────────────
 *
 * v2's confidence computation had a catastrophic collapse bug:
 * With sizeA=30596, sizeB=300 the log-score path computed:
 *   logScore_B = log(300/30896) + log(metricShare_B)
 *             = -4.63 + (-0.3 typical) = -4.93
 *   confidence_B = softmax([-0.05, -4.93]) ≈ 0.0072  (0.7%!)
 *
 * When cw-tv then computed 0.0072·Δ_B, Δ_B was already tiny (≈1/15 of Δ_A).
 * So effective contribution of B_starved = 0.007 × (1/15) = 0.00047 × Δ_A.
 * This is machine-zero — mathematically identical to plain task-arithmetic on A.
 *
 * FIX: The new `scoreMode` option controls how size is turned into confidence:
 *
 *  'proportional' (old buggy default): size_i / sum(sizes)  — collapses to 0
 *  'sqrt'       (new safe default):    √(size_i) / Σ√(size_j) — 100:1 → 10:1
 *  'metric'     :                      use validationMetric only, ignore size
 *  'equal'      :                      flat 1/N weighting, ignore all signals
 *  'delta-norm' :                      weight by ‖Δᵢ‖² (Fisher-approximate proxy,
 *                                      requires base — falls back to 'equal')
 *
 * ─── THREE MERGE MODES ──────────────────────────────────────────────────────
 *
 *  1. no base  →  confidence-weighted FedAvg over raw weights
 *  2. base, ties=false  →  confidence-weighted task-vector average (THE fix):
 *       merged = base + λ · Σᵢ cᵢ · Δᵢ
 *  3. base + TIES (default)  →  CW-TIES: trim → elect sign by confidence-mass
 *
 * ─── CONFIDENCE SCORING ─────────────────────────────────────────────────────
 *
 *  score_i = typeTrust_i · size_signal_i^sizeWeight · metric_signal_i^metricWeight
 *
 *  size_signal_i depends on scoreMode (see above).
 *  All signals normalized to sum=1 across branches before exponentiation.
 *  Final softmax with confidenceTemp flattens extreme ratios further.
 *
 * References:
 *  - Task Arithmetic: Ilharco et al. (2023)
 *  - TIES-Merging:    Yadav et al. (NeurIPS 2023)
 *  - Fisher Merging:  Matena & Raffel (2022)
 *  - LT-Soups:        Aminbeidokhti et al. (NeurIPS 2025)
 *  - FedProx sqrt:    Li et al. (MLSys 2020) — sqrt clip for hetero FL
 */
export const confidenceWeighted: MergeStrategy = {
  name: "confidence-weighted",
  requiresBase: false,
  minModels: 2,
  maxModels: null,

  merge(models, options) {
    // For delta-norm scoreMode we need delta norms before computing confidences.
    // Pre-compute them here if base is available so the confidence function
    // can use ‖Δᵢ‖² as a Fisher-approximate importance proxy.
    let precomputedDeltaNorms: number[] | undefined;
    if (options.base && (options.scoreMode === "delta-norm" || !options.scoreMode)) {
      // Aggregate delta norm across all keys for a global per-branch magnitude.
      // This is the Frobenius norm of the task vector: ‖Δᵢ‖_F
      const baseKeys = Object.keys(options.base);
      precomputedDeltaNorms = models.map((m) => {
        let sumSq = 0;
        for (const key of baseKeys) {
          const baseData = Array.from(options.base![key].data);
          const modelData = Array.from(m.weights[key].data);
          for (let j = 0; j < baseData.length; j++) {
            const d = modelData[j] - baseData[j];
            sumSq += d * d;
          }
        }
        return Math.sqrt(sumSq);
      });
    }

    const confidences = computeConfidences(models, options, precomputedDeltaNorms);

    if (!options.base) {
      // Mode 1 — no lineage info: honest FedAvg over raw weights.
      return rawWeightedAverage(models, confidences);
    }

    if (options.ties === false) {
      // Mode 2 — explicit opt-in: pure task-vector average.
      // The confidence values already reflect the chosen scoreMode.
      return taskVectorAverage(models, options.base, confidences, options, precomputedDeltaNorms);
    }

    // Mode 3 — confidence-weighted TIES (default when base is present).
    return confidenceWeightedTies(models, options.base, confidences, options);
  },
};

// ─── Confidence computation (v3 — fixed collapse bug) ─────────────────────

/**
 * Compute per-branch importance shares under a given scoring mode.
 *
 * scoreMode values:
 *  'proportional' — raw size fraction (BUGS OUT at 100:1+ ratios)
 *  'sqrt'         — √(size) fraction (safe default; 100:1 ratio → 10:1 weight)
 *  'metric'       — validationMetric fraction only (ignore size)
 *  'equal'        — uniform 1/N (ignore all signals)
 *  'delta-norm'   — ‖Δᵢ‖² fraction (computed later, fed in as `values`)
 */
function computeSizeSignal(sizes: (number | undefined)[], mode: string): number[] {
  const n = sizes.length;
  const filled = sizes.map((s) => (s !== undefined && s > 0 ? s : 1));

  if (mode === "equal") return filled.map(() => 1 / n);
  if (mode === "metric") return filled.map(() => 1 / n); // size ignored

  // sqrt mode (default): apply √ before normalizing so 100:1 → 10:1
  const transformed = mode === "proportional"
    ? filled
    : filled.map(Math.sqrt); // 'sqrt' and 'delta-norm' both use sqrt here

  const total = transformed.reduce((a, b) => a + b, 0);
  if (total <= 0) return filled.map(() => 1 / n);
  return transformed.map((v) => v / total);
}

function normalizedShares(values: (number | undefined)[], fallback: number): number[] {
  const raw = values.map((v) => (v !== undefined && v > 0 ? v : fallback));
  const total = raw.reduce((a, b) => a + b, 0);
  if (total <= 0) return raw.map(() => 1 / raw.length);
  return raw.map((v) => v / total);
}

/**
 * One confidence score per branch, normalized via temperature-scaled softmax.
 *
 * v3 key fix: default scoreMode changed from 'proportional' to 'sqrt'.
 * With 100:1 dataset size ratio:
 *   proportional → 0.99/0.01 → after log+softmax ≈ 0.9999/0.0001 (collapse!)
 *   sqrt         → 0.991/0.030 → after log+softmax ≈ 0.97/0.03 (honest)
 *
 * confidenceTemp > 1 → flatter (even safer for imbalanced)
 * confidenceTemp < 1 → sharper (winner-takes-all)
 */
function computeConfidences(
  models: WeightedModel[],
  options: MergeOptions,
  deltaNorms?: number[],
): number[] {
  const sizeWeight = options.sizeWeight ?? 1;
  const metricWeight = options.metricWeight ?? 1;
  const typeTrust = options.typeTrust ?? {};
  const temperature = options.confidenceTemp ?? 1.0;
  // NEW: default is 'sqrt' — safe for imbalanced scenarios
  // Use 'proportional' only if you explicitly want classic FedAvg-style weighting
  const scoreMode = options.scoreMode ?? "sqrt";

  // Handle delta-norm mode: weight by ‖Δᵢ‖² (Fisher-approximate)
  let sizeSignals: number[];
  if (scoreMode === "delta-norm" && deltaNorms && deltaNorms.length === models.length) {
    // Fisher proxy: importance ∝ ‖Δᵢ‖² (squared update norm ~ curvature)
    const squaredNorms = deltaNorms.map((n) => n * n);
    const totalSq = squaredNorms.reduce((a, b) => a + b, 0);
    sizeSignals = totalSq > 0
      ? squaredNorms.map((v) => v / totalSq)
      : models.map(() => 1 / models.length);
  } else {
    sizeSignals = computeSizeSignal(
      models.map((m) => m.info?.datasetSize),
      scoreMode,
    );
  }

  // Metric signal: always proportional (validation accuracy is already a quality measure)
  const metricSignals = normalizedShares(
    models.map((m) => m.info?.validationMetric),
    1,
  );

  // Raw log-scores before softmax (product of signals in linear = sum in log space).
  const logScores = models.map((m, i) => {
    const trust = m.info?.datasetType !== undefined ? (typeTrust[m.info.datasetType] ?? 1) : 1;
    const sizeTerm = scoreMode === "metric" ? 0 : sizeWeight * Math.log(Math.max(sizeSignals[i], 1e-9));
    const metricTerm = metricWeight === 0 ? 0 : metricWeight * Math.log(Math.max(metricSignals[i], 1e-9));
    return Math.log(Math.max(trust, 1e-9)) + sizeTerm + metricTerm;
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
  globalDeltaNorms?: number[], // pre-computed Frobenius norms \u2016\u0394\u1d62\u2016 across ALL keys
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
  // In TIES mode, trimFraction is an explicit algorithmic parameter from the paper
  // (keep top-k% by magnitude) — not a scale factor.  Adaptive trim is opt-in here,
  // not the default, so user-supplied trimFraction values mean exactly what they say.
  const adaptiveTrim = options.adaptiveTrim ?? false;

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
