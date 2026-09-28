/**
 * flash-attention-wgsl: Tensor & Attention Types
 * Multi-head attention tensor representation with 4D strided layout: [batch, heads, seqLen, headDim].
 */

export interface TensorShape {
  readonly batch: number;
  readonly heads: number;
  readonly seqLen: number;
  readonly headDim: number;
}

export interface Tensor4D {
  readonly data: Float32Array;
  readonly shape: TensorShape;
  readonly strides: readonly [number, number, number, number];
}

export interface AttentionConfig {
  /** Enable autoregressive causal masking (upper-triangular -infinity) */
  readonly causal: boolean;
  /** Scaling factor tau, defaults to 1 / sqrt(headDim) */
  readonly scale?: number;
  /** Workgroup SRAM tile row block size (Br), typically 32 or 64 */
  readonly blockSizeR?: number;
  /** Workgroup SRAM tile col block size (Bc), typically 32 or 64 */
  readonly blockSizeC?: number;
}

export interface AttentionMetrics {
  readonly seqLen: number;
  readonly headDim: number;
  readonly numHeads: number;
  readonly batchSize: number;
  readonly isCausal: boolean;
  /** Theoretical floating point operations (FLOPs) */
  readonly totalFlops: number;
  /** Theoretical global memory bytes transferred (Naive vs Flash) */
  readonly naiveHbmBytes: number;
  readonly flashHbmBytes: number;
  /** HBM Memory bandwidth reduction factor (naiveBytes / flashBytes) */
  readonly ioReductionRatio: number;
}

/**
 * Creates a contiguous row-major 4D tensor [batch, heads, seqLen, headDim].
 */
