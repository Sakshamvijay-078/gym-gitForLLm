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

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}
