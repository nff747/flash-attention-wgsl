import { describe, it, expect } from 'vitest';
import {
  createTensor4D,
  createRandomTensor4D,
  naiveAttention,
  tensorGet
} from '../src/math';

describe('Naive Attention Reference Engine', () => {
  it('should compute valid attention output with uniform probabilities for identity keys', () => {
    const batch = 1;
    const heads = 1;
    const seqLen = 4;
    const headDim = 4;

    // All queries identical, keys identical, values distinct
    const Q = createTensor4D(batch, heads, seqLen, headDim, () => 1.0);
    const K = createTensor4D(batch, heads, seqLen, headDim, () => 1.0);
    const V = createTensor4D(batch, heads, seqLen, headDim, (_b, _h, i, d) => (i + 1) * 10 + d);

    const result = naiveAttention(Q, K, V, { causal: false });

    // Since all Q_i . K_j are identical, softmax weights should be uniform = 1/4 = 0.25
    // For non-causal attention, each row output should be the mean of all V rows
    for (let d = 0; d < headDim; d++) {
      let expectedMean = 0;
      for (let j = 0; j < seqLen; j++) {
        expectedMean += (j + 1) * 10 + d;
      }
      expectedMean /= seqLen;

      for (let i = 0; i < seqLen; i++) {
        const val = tensorGet(result.output, 0, 0, i, d);
        expect(val).toBeCloseTo(expectedMean, 4);
      }
    }
  });

  it('should zero out future tokens in causal mode', () => {
    const batch = 1;
    const heads = 1;
    const seqLen = 3;
    const headDim = 2;

    const Q = createRandomTensor4D(batch, heads, seqLen, headDim, 101);
    const K = createRandomTensor4D(batch, heads, seqLen, headDim, 102);
    const V = createRandomTensor4D(batch, heads, seqLen, headDim, 103);

    const result = naiveAttention(Q, K, V, { causal: true });

    // In causal mode:
    // First token (i=0) only attends to j=0, so its output must exactly equal V[0, :]
    for (let d = 0; d < headDim; d++) {
      const outVal = tensorGet(result.output, 0, 0, 0, d);
      const vVal = tensorGet(V, 0, 0, 0, d);
      expect(outVal).toBeCloseTo(vVal, 5);
    }

    // Verify upper triangular attention weights are 0
    const pMatrix = result.attentionWeights;
    expect(pMatrix[0 * seqLen + 1]).toBe(0);
    expect(pMatrix[0 * seqLen + 2]).toBe(0);
    expect(pMatrix[1 * seqLen + 2]).toBe(0);
    expect(pMatrix[0 * seqLen + 0]).toBeCloseTo(1.0, 5);
  });
});
