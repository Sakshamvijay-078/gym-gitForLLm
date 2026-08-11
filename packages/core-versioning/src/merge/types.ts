import type { ModelWeights } from "../model.ts";
export type { ModelWeights, Tensor, DType } from "../model.ts";
export { makeTensorLike } from "../model.ts";

/**
 * Signals used to decide how much a branch's weights should count in a
 * merge. Absent fields fall back to neutral defaults — a strategy that
 * uses this should degrade gracefully (to equal weighting) when nothing
 * is known about a branch, not error out.
 */
export interface BranchInfo {
  /** Number of training examples this branch trained on. */
  datasetSize?: number;
  /** A validation score for this branch, higher = better, any consistent scale. */
  validationMetric?: number;
  /** Free-form label for what kind of data this branch trained on — not a magnitude, used only via typeTrust. */
  datasetType?: string;
}

export interface WeightedModel {
  /** The manifest hash this model's weights came from. */
  hash: string;
  weights: ModelWeights;
  /** Optional — only read by confidence-aware strategies. */
  info?: BranchInfo;
}

export interface MergeOptions {
  /** Common ancestor weights — required by task-arithmetic and TIES. */
  base?: ModelWeights;
  /** Global scale applied to the combined task vector (task-arithmetic, TIES, confidence-weighted). Default varies by strategy. */
  lambda?: number;
  /** Fraction of each task vector's parameters to keep by magnitude (TIES, confidence-weighted). Default 0.2. */
  trimFraction?: number;
  /** Interpolation factor between the two models, 0..1 (SLERP). Default 0.5. */
  t?: number;

  // --- confidence-weighted strategy options ---
  /** How strongly dataset size influences confidence, as an exponent on its normalized share. 0 disables it, 1 (default) is proportional, >1 sharpens it. */
  sizeWeight?: number;
  /** Same, for validationMetric. */
  metricWeight?: number;
  /** Manual trust multiplier per datasetType label, e.g. {"clean-labels": 2, "noisy-scrape": 0.5}. Missing types default to 1 (neutral). */
  typeTrust?: Record<string, number>;
  /**
   * Softmax temperature applied to the raw confidence log-scores before
   * normalization.  > 1 flattens the distribution toward uniform (safer
   * for imbalanced experiments); < 1 sharpens it toward winner-takes-all.
   * Default 1.0 (equivalent to direct softmax; replaces the old hard normalize).
   */
  confidenceTemp?: number;
  /**
   * When `false`, use a plain confidence-weighted task-vector average instead
   * of TIES.  Default `true` (TIES when base is present) — set to `false`
   * for the imbalanced-delta experiment where you want pure weighted-Δ blending
   * without trim/sign-election.
   */
  ties?: boolean;
  /**
   * When true (default when base is given), scale each branch's keep fraction
   * by its relative delta norm so barely-trained branches are trimmed more
   * aggressively than well-trained ones.
   */
  adaptiveTrim?: boolean;
  /**
   * When true, unit-normalize each branch's task vector before blending, then
   * re-scale by the confidence-weighted mean of the original norms.  This
   * separates learning direction from magnitude, so confidence scores alone
   * govern how much each branch contributes.  Default false.
   */
  normalizeTaskVectors?: boolean;

  /**
   * Controls how dataset size is converted into a confidence signal.
   * This is the most important option for imbalanced datasets.
   *
   *  'proportional' — raw size fraction: size_i / Σ size_j
   *                   At 100:1 ratio this gives 0.99/0.01 → after log+softmax
   *                   confidence collapses to ~0.0001 for the small branch.
   *                   Only use when datasets are balanced (<5:1 ratio).
   *
   *  'sqrt'         — √(size_i) / Σ√(size_j)  ← NEW DEFAULT
   *                   100:1 ratio → 10:1 confidence. Much safer for imbalanced
   *                   scenarios. Inspired by FedProx heterogeneous FL scaling.
   *
   *  'metric'       — use validationMetric only; size has no influence.
   *                   Good when metric accurately captures quality regardless
   *                   of how much data was used to achieve it.
   *
   *  'equal'        — flat 1/N weighting; ignore all signals.
   *                   Use when you have no reliable metadata but want task-vector
   *                   merging (better than raw-weight average regardless).
   *
   *  'delta-norm'   — weight by ‖Δᵢ‖² (Fisher-approximate: squared update norm
   *                   is a proxy for the diagonal Fisher information matrix).
   *                   A branch that actually moved its weights more is weighted
   *                   more. Requires base. Falls back to 'equal' without base.
   *                   Ref: Matena & Raffel (2022) "Merging Models with Fisher..."
   */
  scoreMode?: "proportional" | "sqrt" | "metric" | "equal" | "delta-norm";

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
