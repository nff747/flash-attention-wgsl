/**
 * flash-attention-wgsl: Tiled FlashAttention-2 CPU Reference Engine
 * Hardware-faithful SRAM tiling simulation with outer Q-block and inner KV-block loop.
 */

import {
  Tensor4D,
  AttentionConfig,
  createTensor4D,
  tensorGet,
  tensorSet
} from './types';
import { defaultScale, NEG_INFINITY } from './numerical_stability';
import {
  createOnlineSoftmaxState,
  updateOnlineSoftmaxBlock,
  finalizeOnlineSoftmax
} from './online_softmax';

export interface FlashAttentionTilingMetrics {
  readonly numRowBlocks: number;
  readonly numColBlocks: number;
  readonly skippedCausalBlocks: number;
  readonly executedBlocks: number;
}

export interface FlashAttentionResult {
  readonly output: Tensor4D;
  readonly tilingMetrics: FlashAttentionTilingMetrics;
}

/**
 * Executes FlashAttention-2 with block-by-block SRAM tiling and online softmax.
 */
export function flashAttentionCPU(
  Q: Tensor4D,
  K: Tensor4D,
  V: Tensor4D,
  config: AttentionConfig
): FlashAttentionResult {
  const { batch, heads, seqLen, headDim } = Q.shape;

  // Validate tensor shapes
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
  const Br = config.blockSizeR ?? 32;
  const Bc = config.blockSizeC ?? 32;

  const Tr = Math.ceil(seqLen / Br);
  const Tc = Math.ceil(seqLen / Bc);

  const O = createTensor4D(batch, heads, seqLen, headDim);

  let skippedCausalBlocks = 0;
  let executedBlocks = 0;

  // Optimized SRAM cache tiles for inner loop
  const sBlock = new Float32Array(Bc);
  const vBlock = new Float32Array(Bc * headDim);
  const rowOutputAcc = new Float32Array(headDim);

  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < heads; h++) {

      // Outer Loop: iterate over Q blocks (Tr)
      for (let tr = 0; tr < Tr; tr++) {
        const rowStart = tr * Br;
        const rowEnd = Math.min(seqLen, rowStart + Br);
        const actualBr = rowEnd - rowStart;

        for (let r = 0; r < actualBr; r++) {
          const globalRowIdx = rowStart + r;
          const state = createOnlineSoftmaxState();
          rowOutputAcc.fill(0.0);

          // Inner Loop: iterate over K, V blocks (Tc)
          for (let tc = 0; tc < Tc; tc++) {
            const colStart = tc * Bc;
            const colEnd = Math.min(seqLen, colStart + Bc);
            const actualBc = colEnd - colStart;

            // Causal optimization: Skip blocks strictly past the diagonal
            if (config.causal && colStart > globalRowIdx) {
              skippedCausalBlocks++;
              continue;
            }

            executedBlocks++;

            // Load V block into SRAM tile buffer
            for (let c = 0; c < actualBc; c++) {
              const globalColIdx = colStart + c;
              for (let d = 0; d < headDim; d++) {
                vBlock[c * headDim + d] = tensorGet(V, b, h, globalColIdx, d);
              }
            }

            // Compute S_ij block tile: S_ij = (Q_i . K_j) * scale
            const currentScores = sBlock.subarray(0, actualBc);
            for (let c = 0; c < actualBc; c++) {
              const globalColIdx = colStart + c;

              if (config.causal && globalColIdx > globalRowIdx) {
                currentScores[c] = NEG_INFINITY;
                continue;
              }

              let dot = 0.0;
              for (let d = 0; d < headDim; d++) {
                dot += tensorGet(Q, b, h, globalRowIdx, d) * tensorGet(K, b, h, globalColIdx, d);
              }
              currentScores[c] = dot * scale;
            }

            // Perform Online Softmax update step on this block
            updateOnlineSoftmaxBlock(
              state,
              rowOutputAcc,
              currentScores,
              vBlock,
              headDim
            );
          }

          // Epilogue: normalize accumulator by running sum of exponentials (l_i)
          finalizeOnlineSoftmax(state, rowOutputAcc);

          // Write back normalized output row to global memory
          for (let d = 0; d < headDim; d++) {
            tensorSet(O, b, h, globalRowIdx, d, rowOutputAcc[d]);
          }
        }
      }
    }
  }

  return {
    output: O,
    tilingMetrics: {
      numRowBlocks: Tr,
      numColBlocks: Tc,
      skippedCausalBlocks,
      executedBlocks
    }
  };
}
