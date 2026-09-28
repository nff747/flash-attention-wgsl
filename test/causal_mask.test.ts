import { describe, it, expect } from 'vitest';
import {
  createRandomTensor4D,
  flashAttentionCPU,
  tensorGet
} from '../src/math';

describe('Causal Masking Autoregressive Invariant', () => {
  it('modifying future tokens (t > i) must not affect attention output at token i', () => {
    const batch = 1;
    const heads = 1;
    const seqLen = 16;
    const headDim = 16;

    const Q = createRandomTensor4D(batch, heads, seqLen, headDim, 1001);
    const K = createRandomTensor4D(batch, heads, seqLen, headDim, 1002);
    const V = createRandomTensor4D(batch, heads, seqLen, headDim, 1003);

    // Initial causal attention run
    const resA = flashAttentionCPU(Q, K, V, { causal: true, blockSizeR: 8, blockSizeC: 8 });

    // Mutate future tokens in K and V at position index 10..15
    const K_mutated = createRandomTensor4D(batch, heads, seqLen, headDim, 1002);
    const V_mutated = createRandomTensor4D(batch, heads, seqLen, headDim, 1003);

    for (let i = 10; i < seqLen; i++) {
      for (let d = 0; d < headDim; d++) {
        K_mutated.data[i * headDim + d] += 50.0;
        V_mutated.data[i * headDim + d] += 50.0;
      }
    }

    const resB = flashAttentionCPU(Q, K_mutated, V_mutated, { causal: true, blockSizeR: 8, blockSizeC: 8 });

    // Tokens at indices 0..9 must remain strictly identical because they cannot attend to future tokens!
    for (let i = 0; i < 10; i++) {
      for (let d = 0; d < headDim; d++) {
        const valA = tensorGet(resA.output, 0, 0, i, d);
        const valB = tensorGet(resB.output, 0, 0, i, d);
        expect(valA).toBeCloseTo(valB, 5);
      }
    }

    // Token 10 must be different because its own V was changed
    const diff10 = Math.abs(tensorGet(resA.output, 0, 0, 10, 0) - tensorGet(resB.output, 0, 0, 10, 0));
    expect(diff10).toBeGreaterThan(0.1);
  });
});
