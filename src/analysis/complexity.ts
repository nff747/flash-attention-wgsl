/**
 * flash-attention-wgsl: FLOPs & IO Complexity Analyzer
 * Precise computational complexity and High-Bandwidth Memory (HBM) IO modeling.
 */

export interface ComplexityAnalysis {
  readonly batch: number;
  readonly heads: number;
  readonly seqLen: number;
  readonly headDim: number;
  readonly isCausal: boolean;
  /** Total theoretical floating point operations */
  readonly totalFlops: number;
  /** Flops per token */
  readonly flopsPerToken: number;
  /** Naive Attention HBM bytes transferred (loading & storing Q, K, V, S, P, O) */
  readonly naiveHbmBytes: number;
  /** FlashAttention HBM bytes transferred (no intermediate S & P materialization) */
  readonly flashHbmBytes: number;
  /** Arithmetic intensity: FLOPs / Byte for Naive vs Flash */
  readonly naiveArithmeticIntensity: number;
  readonly flashArithmeticIntensity: number;
  /** Memory bandwidth reduction multiplier */
  readonly hbmSpeedupFactor: number;
}

export function computeAttentionComplexity(
  batch: number,
  heads: number,
  seqLen: number,
  headDim: number,
  isCausal = false,
  blockSizeR = 32,
  bytesPerFloat = 4
): ComplexityAnalysis {
  const causalFactor = isCausal ? 0.5 : 1.0;

  // 1. Precise FLOPs Calculation:
  // Q * K^T: 2 * seqLen^2 * headDim FLOPs per head
  // Softmax: 3 * seqLen^2 FLOPs per head (max, exp, sum/div)
  // P * V:   2 * seqLen^2 * headDim FLOPs per head
  const flopsPerHead = causalFactor * (4 * seqLen * seqLen * headDim + 3 * seqLen * seqLen);
  const totalFlops = batch * heads * flopsPerHead;
  const flopsPerToken = totalFlops / (batch * seqLen);

  // 2. Naive Attention Memory Traffic (in Bytes):
  // Read Q (N*d), Read K (N*d)
  // Write S (N*N), Read S (N*N)
  // Write P (N*N), Read P (N*N)
  // Read V (N*d), Write O (N*d)
  const qkvElements = 4 * seqLen * headDim; // Q + K + V + O
  const intermediateElements = 4 * seqLen * seqLen; // S write+read, P write+read
  const naiveBytesPerHead = (qkvElements + intermediateElements) * bytesPerFloat;
  const naiveHbmBytes = batch * heads * naiveBytesPerHead;

  // 3. FlashAttention Memory Traffic (in Bytes):
  // Q is loaded once per row block: N * d
  // O is written once at the end: N * d
  // K and V are loaded Tr = ceil(N / Br) times from global memory
  const Tr = Math.ceil(seqLen / blockSizeR);
  const kvReadsPerHead = causalFactor * Tr * (2 * seqLen * headDim);
  const qoTrafficPerHead = 2 * seqLen * headDim; // Read Q once, write O once
  const flashBytesPerHead = (qoTrafficPerHead + kvReadsPerHead) * bytesPerFloat;
  const flashHbmBytes = batch * heads * flashBytesPerHead;

  const naiveArithmeticIntensity = totalFlops / Math.max(1, naiveHbmBytes);
  const flashArithmeticIntensity = totalFlops / Math.max(1, flashHbmBytes);
  const hbmSpeedupFactor = naiveHbmBytes / Math.max(1, flashHbmBytes);

  return {
    batch,
    heads,
    seqLen,
    headDim,
    isCausal,
    totalFlops,
    flopsPerToken,
    naiveHbmBytes,
    flashHbmBytes,
    naiveArithmeticIntensity,
    flashArithmeticIntensity,
    hbmSpeedupFactor
  };
}
