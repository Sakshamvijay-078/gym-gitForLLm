import type { MergeStrategy, ModelWeights } from "../types.ts";
import { assertCompatible, makeTensorLike } from "../types.ts";
import { add, dot, norm, scale } from "../vectorMath.ts";

/**
 * SLERP — Spherical Linear Interpolation. Unlike task-arithmetic/TIES,
 * this doesn't work in "delta from a base" space — it interpolates
 * directly between the two models' weight vectors along the arc of the
 * hypersphere connecting them, rather than a straight line through
 * weight space. This tends to preserve each model's weight *magnitude*
 * (norm) better than linear interpolation does.
 *
 * Only defined for exactly two models — there's no natural "arc through
 * three points on a sphere," which is why maxModels is 2 here while every
 * other strategy generalizes to N branches.
 *
 * Per tensor:
 *   Ω = angle between vector a and vector b
 *   result = (sin((1-t)Ω)/sin Ω) * a + (sin(tΩ)/sin Ω) * b
 * Falls back to linear interpolation when Ω is near 0 (nearly-parallel
 * vectors), where the SLERP formula divides by ~0.
 */
export const slerp: MergeStrategy = {
  name: "slerp",
  requiresBase: false,
  minModels: 2,
  maxModels: 2,

  merge(models, options) {
    if (models.length !== 2) {
      throw new Error(`slerp requires exactly 2 models, got ${models.length}`);
    }
    const t = options.t ?? 0.5;
    if (t < 0 || t > 1) {
      throw new Error(`slerp t must be in [0, 1], got ${t}`);
    }

    const [a, b] = models;
    assertCompatible([a.weights, b.weights]);

    const keys = Object.keys(a.weights);
    const result: ModelWeights = {};

    for (const key of keys) {
      const vecA = Array.from(a.weights[key].data);
      const vecB = Array.from(b.weights[key].data);
      result[key] = makeTensorLike(a.weights[key], slerpVector(vecA, vecB, t));
    }
    return result;
  },
};

function slerpVector(a: number[], b: number[], t: number): number[] {
  const normA = norm(a);
  const normB = norm(b);

  if (normA === 0 || normB === 0) {
    // Degenerate: at least one vector is all zeros, angle is undefined.
    return add(scale(a, 1 - t), scale(b, t));
  }

  const cosOmega = Math.min(1, Math.max(-1, dot(a, b) / (normA * normB)));
  const omega = Math.acos(cosOmega);
  const sinOmega = Math.sin(omega);

  if (sinOmega < 1e-6) {
    // Nearly parallel — SLERP's coefficients blow up, fall back to lerp.
    return add(scale(a, 1 - t), scale(b, t));
  }

  const coeffA = Math.sin((1 - t) * omega) / sinOmega;
  const coeffB = Math.sin(t * omega) / sinOmega;
  return add(scale(a, coeffA), scale(b, coeffB));
}
