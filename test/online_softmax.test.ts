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

import { mergeOnlineSoftmaxStates } from '../src/math/online_softmax';

describe('Split-K Online Softmax State Merging', () => {
  it('should merge two disjoint block states and match monolithic online softmax', () => {
    const headDim = 4;
    // Worker A processes block 1: logits [1.0, 2.0]
    const stateA = createOnlineSoftmaxState();
    const outA = new Float32Array(headDim);
    const scoresA = new Float32Array([1.0, 2.0]);
    const vA = new Float32Array([1, 1, 1, 1, 2, 2, 2, 2]);
    updateOnlineSoftmaxBlock(stateA, outA, scoresA, vA, headDim);

    // Worker B processes block 2: logits [3.0, 0.5]
    const stateB = createOnlineSoftmaxState();
    const outB = new Float32Array(headDim);
    const scoresB = new Float32Array([3.0, 0.5]);
    const vB = new Float32Array([3, 3, 3, 3, 0.5, 0.5, 0.5, 0.5]);
    updateOnlineSoftmaxBlock(stateB, outB, scoresB, vB, headDim);

    // Merge A and B
    const merged = mergeOnlineSoftmaxStates(stateA, outA, stateB, outB);
    finalizeOnlineSoftmax(merged.state, merged.output);

    // Compute monolithic baseline: 4 logits sequentially
    const stateMono = createOnlineSoftmaxState();
    const outMono = new Float32Array(headDim);
    const allScores = new Float32Array([1.0, 2.0, 3.0, 0.5]);
    const allV = new Float32Array([...vA, ...vB]);
    updateOnlineSoftmaxBlock(stateMono, outMono, allScores, allV, headDim);
    finalizeOnlineSoftmax(stateMono, outMono);

    for (let d = 0; d < headDim; d++) {
      expect(merged.output[d]).toBeCloseTo(outMono[d], 5);
    }
  });
});
