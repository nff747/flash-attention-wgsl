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
