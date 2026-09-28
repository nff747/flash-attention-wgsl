/**
 * flash-attention-wgsl: Naive Scaled Dot-Product Attention (Baseline Reference)
 * Exact O(N^2) reference implementation with full intermediate matrix materialization.
 */

import {
  Tensor4D,
  AttentionConfig,
  createTensor4D,
  tensorGet,
  tensorSet
} from './types';
import { defaultScale, NEG_INFINITY, stableSoftmax } from './numerical_stability';

export interface NaiveAttentionResult {
  readonly output: Tensor4D;
  /** Materialized attention score logits S = (Q K^T) / sqrt(d) */
  readonly scores: Float32Array;
  /** Materialized attention probability matrix P = softmax(S) */
  readonly attentionWeights: Float32Array;
}

/**
 * Computes standard multi-head attention with explicit O(N^2) score matrices.
 */
export function naiveAttention(
  Q: Tensor4D,
  K: Tensor4D,
  V: Tensor4D,
  config: AttentionConfig
): NaiveAttentionResult {
  const { batch, heads, seqLen, headDim } = Q.shape;

  // Validate tensor shapes match
  if (
    K.shape.batch !== batch ||
    K.shape.heads !== heads ||
    K.shape.seqLen !== seqLen ||
    K.shape.headDim !== headDim ||
    V.shape.batch !== batch ||
    V.shape.heads !== heads ||
    V.shape.seqLen !== seqLen ||
    V.shape.headDim !== headDim
  ) {
    throw new Error('Tensor shape mismatch between Q, K, and V');
  }

  const scale = config.scale ?? defaultScale(headDim);
  const O = createTensor4D(batch, heads, seqLen, headDim);

  // Materialized intermediate buffers [batch * heads * seqLen * seqLen]
  const matrixSize = batch * heads * seqLen * seqLen;
  const scores = new Float32Array(matrixSize);
  const attentionWeights = new Float32Array(matrixSize);

  const rowBuffer = new Float32Array(seqLen);

  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < heads; h++) {
      const bhOffset = (b * heads + h) * seqLen * seqLen;

        return { output: O, scores, attentionWeights };
}