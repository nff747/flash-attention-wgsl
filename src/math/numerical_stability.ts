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
