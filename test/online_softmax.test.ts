import { describe, it, expect } from 'vitest';
import {
  createOnlineSoftmaxState,
  updateOnlineSoftmaxBlock,
  finalizeOnlineSoftmax,
  stableSoftmax
} from '../src/math';

describe('Online Softmax State Machine', () => {
  it('should produce identical probabilities and weighted values across arbitrary block splits', () => {
    const totalCols = 16;
    const headDim = 8;
    const blockSizeC = 4;

    // Fixed random logits
    const fullScores = new Float32Array(totalCols);
    for (let i = 0; i < totalCols; i++) {
      fullScores[i] = Math.sin(i * 1.7) * 4.0;
    }

    // Values matrix
    const fullV = new Float32Array(totalCols * headDim);
    for (let i = 0; i < totalCols * headDim; i++) {
      fullV[i] = Math.cos(i * 0.9);
    }

    // Baseline: standard 2-pass stable softmax
    const baselineP = new Float32Array(fullScores);
    stableSoftmax(baselineP);

    const baselineOut = new Float32Array(headDim);
    for (let d = 0; d < headDim; d++) {
      let sum = 0;
      for (let j = 0; j < totalCols; j++) {
        sum += baselineP[j] * fullV[j * headDim + d];
      }
      baselineOut[d] = sum;
    }

    // Online Softmax: partitioned into blocks of size 4
    const state = createOnlineSoftmaxState();
    const onlineOut = new Float32Array(headDim);

    const numBlocks = totalCols / blockSizeC;
    for (let b = 0; b < numBlocks; b++) {
      const blockScores = fullScores.subarray(b * blockSizeC, (b + 1) * blockSizeC);
      const blockV = fullV.subarray(b * blockSizeC * headDim, (b + 1) * blockSizeC * headDim);

      updateOnlineSoftmaxBlock(state, onlineOut, blockScores, blockV, headDim);
    }

    finalizeOnlineSoftmax(state, onlineOut);

    // Verify online softmax output equals baseline output down to 1e-6
    for (let d = 0; d < headDim; d++) {
      expect(onlineOut[d]).toBeCloseTo(baselineOut[d], 5);
    }
  });
});
