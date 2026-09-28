import { describe, it, expect } from 'vitest';
import {
  createTensor4D,
  flashAttentionCPU,
  naiveAttention,
  maxAbsoluteError
} from '../src/math';

describe('Numerical Stability Safeguards', () => {
  it('should handle large dynamic range logits without overflowing into NaN or Infinity', () => {
    const batch = 1;
    const heads = 1;
    const seqLen = 8;
    const headDim = 8;

    // Queries and keys with large values (can cause exp(x) > 1e38 overflow in naive un-normalized softmax)
    const Q = createTensor4D(batch, heads, seqLen, headDim, (_b, _h, i, d) => (i + 1) * 20.0 + d);
    const K = createTensor4D(batch, heads, seqLen, headDim, (_b, _h, i, d) => (i + 1) * 20.0 + d);
    const V = createTensor4D(batch, heads, seqLen, headDim, (_b, _h, i, d) => Math.sin(i + d));

    const resNaive = naiveAttention(Q, K, V, { causal: false });
    const resFlash = flashAttentionCPU(Q, K, V, { causal: false, blockSizeR: 4, blockSizeC: 4 });

    // Verify all outputs are finite
    for (let i = 0; i < resFlash.output.data.length; i++) {
      expect(isFinite(resFlash.output.data[i])).toBe(true);
      expect(isNaN(resFlash.output.data[i])).toBe(false);
    }

    const err = maxAbsoluteError(resNaive.output.data, resFlash.output.data);
    expect(err).toBeLessThan(1e-5);
  });
});
