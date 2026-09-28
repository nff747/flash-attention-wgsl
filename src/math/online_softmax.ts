/**
 * flash-attention-wgsl: Online Softmax Accumulator & State Machine
 * Mathematical formulations and incremental state transitions for FlashAttention.
 */

export interface OnlineSoftmaxState {
  /** Running maximum of logits seen so far (m_i) */
  m: number;
  /** Running normalizer / sum of exponentials (l_i) */
  l: number;
}

/**
 * Creates initial Online Softmax state with m = -Infinity, l = 0.
 */
export function createOnlineSoftmaxState(): OnlineSoftmaxState {
  return {
    m: -Infinity,
    l: 0.0
  };
}

/**
 * Updates running online softmax state and rescales the output accumulator.
 *
 * Given current state (m_prev, l_prev), accumulator O_prev,
 * and a new block of logits S_block with its row-max m_block:
 *
 * 1. m_new = max(m_prev, m_block)
 * 2. alpha = exp(m_prev - m_new)   // factor to rescale previous running sums
 * 3. P_tilde = exp(S_block - m_new)
 * 4. l_new = alpha * l_prev + sum(P_tilde)
 * 5. O_new = alpha * O_prev + P_tilde @ V_block
 *
 * @param state Running state {m, l} (updated in-place)
 * @param rowOutput Running output vector slice of length headDim (updated in-place)
 * @param blockScores Logits for current block S_ij
 * @param blockV Values matrix for current block V_j [blockSizeC, headDim]
 * @param headDim Head dimension
 */
export function updateOnlineSoftmaxBlock(
  state: OnlineSoftmaxState,
  rowOutput: Float32Array,
  blockScores: Float32Array,
  blockV: Float32Array,
  headDim: number
): void {
  const blockSizeC = blockScores.length;

  // 1. Find max of current block logits
  let mBlock = -Infinity;
  for (let j = 0; j < blockSizeC; j++) {
    if (blockScores[j] > mBlock) {
      mBlock = blockScores[j];
    }
  }

  // If block is completely masked (-Infinity), state is unchanged
  if (!isFinite(mBlock) || mBlock < -1e8) {
    return;
  }

  // 2. Compute new row maximum
  const mNew = Math.max(state.m, mBlock);

  // 3. Compute rescale factor for previous accumulator
  // When state.m is -Infinity (first valid block), alpha = 0
  const alpha = isFinite(state.m) ? Math.exp(state.m - mNew) : 0.0;

  // 4. Compute unnormalized probabilities P_tilde = exp(S - mNew) and their sum
  let blockSum = 0.0;
  // Temporary buffer for unnormalized P
  const pTilde = new Float32Array(blockSizeC);
  for (let j = 0; j < blockSizeC; j++) {
    const s = blockScores[j];
    if (s > -1e8) {
      const p = Math.exp(s - mNew);
      pTilde[j] = p;
      blockSum += p;
    } else {
      pTilde[j] = 0.0;
    }
  }

  // 5. Rescale previous output accumulator and accumulate P_tilde @ V_block
  for (let d = 0; d < headDim; d++) {
    let pDotV = 0.0;
    for (let j = 0; j < blockSizeC; j++) {
      const p = pTilde[j];
      if (p > 0.0) {
        pDotV += p * blockV[j * headDim + d];
      }
    }
    rowOutput[d] = alpha * rowOutput[d] + pDotV;
  }

  // 6. Update running state
  state.l = alpha * state.l + blockSum;
  state.m = mNew;
}

/**
 * Normalizes the final output accumulator: O = O / l.
 */
export function finalizeOnlineSoftmax(
  state: OnlineSoftmaxState,
  rowOutput: Float32Array
): void {
  if (state.l > 0.0) {
    const invL = 1.0 / state.l;
    for (let d = 0; d < rowOutput.length; d++) {
      rowOutput[d] *= invL;
    }
  } else {
    rowOutput.fill(0.0);
  }
}
