/**
 * flash-attention-wgsl: Numerical Stability & Precision Utilities
 * Safe exponentiation, log-sum-exp, and tensor difference metrics.
 */

export const NEG_INFINITY = -1e9; // WGSL compatible large negative sentinel
export const EPSILON = 1e-6;

/**
 * Calculates standard scale factor tau = 1 / sqrt(d).
 */
export function defaultScale(headDim: number): number {
  if (headDim <= 0) {
    throw new Error(`headDim must be positive, got ${headDim}`);
  }
  return 1.0 / Math.sqrt(headDim);
}

/**
 * Computes numerically stable Softmax in-place over a 1D slice.
 * Softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
 */
export function stableSoftmax(vector: Float32Array): Float32Array {
  const len = vector.length;
  if (len === 0) return vector;

  let maxVal = -Infinity;
  for (let i = 0; i < len; i++) {
    if (vector[i] > maxVal) {
      maxVal = vector[i];
    }
  }

  // Handle all -Infinity case (e.g. fully masked tokens)
  if (!isFinite(maxVal) || maxVal <= NEG_INFINITY) {
    vector.fill(0);
    return vector;
  }

  let sumExp = 0.0;
  for (let i = 0; i < len; i++) {
    const val = Math.exp(vector[i] - maxVal);
    vector[i] = val;
    sumExp += val;
  }

  const invSum = sumExp > 0 ? 1.0 / sumExp : 0.0;
  for (let i = 0; i < len; i++) {
    vector[i] *= invSum;
  }

  return vector;
}

/**
 * Calculates Maximum Absolute Error: max_i |a_i - b_i|
 */
export function maxAbsoluteError(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Array lengths must match: ${a.length} vs ${b.length}`);
  }
  let maxDiff = 0.0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }
  return maxDiff;
}

/**
 * Calculates Mean Squared Error: (1/N) * sum_i (a_i - b_i)^2
 */
export function meanSquaredError(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Array lengths must match: ${a.length} vs ${b.length}`);
  }
  let sumSq = 0.0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  return sumSq / a.length;
}

/**
 * Calculates Relative Error: max_i (|a_i - b_i| / (max(|a_i|, |b_i|) + eps))
 */
export function maxRelativeError(a: Float32Array, b: Float32Array, eps = 1e-5): number {
  if (a.length !== b.length) {
    throw new Error(`Array lengths must match: ${a.length} vs ${b.length}`);
  }
  let maxRel = 0.0;
  for (let i = 0; i < a.length; i++) {
    const absDiff = Math.abs(a[i] - b[i]);
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i])) + eps;
    const rel = absDiff / denom;
    if (rel > maxRel) {
      maxRel = rel;
    }
  }
  return maxRel;
}
