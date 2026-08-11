export function add(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i]);
}

export function sub(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - b[i]);
}

export function scale(a: number[], s: number): number[] {
  return a.map((v) => v * s);
}

export function mean(vectors: number[][]): number[] {
  const n = vectors.length;
  const len = vectors[0].length;
  const out = new Array(len).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < len; i++) out[i] += v[i] / n;
  }
  return out;
}

/** Elementwise weighted mean: sum_i(w_i * v_i) where weights already sum to 1. */
export function weightedMean(vectors: number[][], weights: number[]): number[] {
  const len = vectors[0].length;
  const out = new Array(len).fill(0);
  for (let m = 0; m < vectors.length; m++) {
    const w = weights[m];
    const v = vectors[m];
    for (let i = 0; i < len; i++) out[i] += w * v[i];
  }
  return out;
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Temperature-scaled softmax over raw scores.
 * temperature > 1 flattens toward uniform; < 1 sharpens toward the max.
 * Numerically stable (shifts by max before exp).
 */
export function softmax(scores: number[], temperature = 1): number[] {
  const scaled = scores.map((s) => s / temperature);
  const maxS = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - maxS));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0) return scores.map(() => 1 / scores.length);
  return exps.map((e) => e / sum);
}

/** Linear interpolation between two vectors: (1-t)*a + t*b */
export function lerpVec(a: number[], b: number[], t: number): number[] {
  return a.map((v, i) => v * (1 - t) + b[i] * t);
}
