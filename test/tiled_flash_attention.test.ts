import { describe, it, expect } from 'vitest';
import {
  createRandomTensor4D,
  naiveAttention,
  flashAttentionCPU,
  maxAbsoluteError,
  maxRelativeError
} from '../src/math';

describe('Tiled FlashAttention-2 vs Naive Attention Equivalence', () => {
  const testCases = [
    { name: 'Small power-of-two (Seq=32, Dim=32, Causal=false)', batch: 1, heads: 2, seqLen: 32, headDim: 32, causal: false, br: 16, bc: 16 },
    { name: 'Small power-of-two (Seq=32, Dim=32, Causal=true)', batch: 1, heads: 2, seqLen: 32, headDim: 32, causal: true, br: 16, bc: 16 },
    { name: 'Medium power-of-two (Seq=64, Dim=64, Causal=false)', batch: 2, heads: 4, seqLen: 64, headDim: 64, causal: false, br: 32, bc: 32 },
    { name: 'Medium power-of-two (Seq=64, Dim=64, Causal=true)', batch: 2, heads: 4, seqLen: 64, headDim: 64, causal: true, br: 32, bc: 32 },
    { name: 'Non-power-of-two boundary check (Seq=47, Dim=24)', batch: 1, heads: 2, seqLen: 47, headDim: 24, causal: false, br: 16, bc: 16 },
    { name: 'Non-power-of-two causal check (Seq=53, Dim=32)', batch: 1, heads: 2, seqLen: 53, headDim: 32, causal: true, br: 16, bc: 16 },
    { name: 'Asymmetric block sizes (Br=32, Bc=16, Seq=64, Dim=32)', batch: 1, heads: 2, seqLen: 64, headDim: 32, causal: true, br: 32, bc: 16 }
  ];

  testCases.forEach((tc) => {
    it(`should match naive attention: ${tc.name}`, () => {
      const Q = createRandomTensor4D(tc.batch, tc.heads, tc.seqLen, tc.headDim, 42);
      const K = createRandomTensor4D(tc.batch, tc.heads, tc.seqLen, tc.headDim, 43);
      const V = createRandomTensor4D(tc.batch, tc.heads, tc.seqLen, tc.headDim, 44);

      const config = {
        causal: tc.causal,
        blockSizeR: tc.br,
        blockSizeC: tc.bc
      };

      const naiveRes = naiveAttention(Q, K, V, config);
      const flashRes = flashAttentionCPU(Q, K, V, config);

      const maxErr = maxAbsoluteError(naiveRes.output.data, flashRes.output.data);
      const relErr = maxRelativeError(naiveRes.output.data, flashRes.output.data);

      // Numerical tolerance for 32-bit floating point arithmetic
      expect(maxErr).toBeLessThan(1e-5);
      expect(relErr).toBeLessThan(5e-4);

      // In causal mode, verify causal blocks were indeed skipped
      if (tc.causal) {
        expect(flashRes.tilingMetrics.skippedCausalBlocks).toBeGreaterThan(0);
      }
    });
  });
});
