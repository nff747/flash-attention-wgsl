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
export function createTensor4D(
  batch: number,
  heads: number,
  seqLen: number,
  headDim: number,
  init?: Float32Array | number[] | ((b: number, h: number, i: number, d: number) => number)
): Tensor4D {
  const size = batch * heads * seqLen * headDim;
  const data = new Float32Array(size);

  const strideD = 1;
  const strideS = headDim;
  const strideH = seqLen * strideS;
  const strideB = heads * strideH;
  const strides: [number, number, number, number] = [strideB, strideH, strideS, strideD];

  if (typeof init === 'function') {
    let idx = 0;
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < heads; h++) {
        for (let i = 0; i < seqLen; i++) {
          for (let d = 0; d < headDim; d++) {
            data[idx++] = init(b, h, i, d);
          }
        }
      }
    }
  } else if (init) {
    data.set(init);
  }

  return {
    data,
    shape: { batch, heads, seqLen, headDim },
    strides
  };
}

/**
 * Fast flat offset computation for [b, h, i, d].
 */
export function tensorOffset(
  tensor: Tensor4D,
  b: number,
  h: number,
  i: number,
  d: number
): number {
  return (
    b * tensor.strides[0] +
    h * tensor.strides[1] +
    i * tensor.strides[2] +
    d * tensor.strides[3]
  );
}

export function tensorGet(
  tensor: Tensor4D,
  b: number,
  h: number,
  i: number,
  d: number
): number {
  return tensor.data[tensorOffset(tensor, b, h, i, d)];
}

export function tensorSet(
  tensor: Tensor4D,
  b: number,
  h: number,
  i: number,
  d: number,
  val: number
): void {
  tensor.data[tensorOffset(tensor, b, h, i, d)] = val;
}

/**
 * Generates reproducible deterministic pseudo-random tensor for testing.
 */
export function createRandomTensor4D(
  batch: number,
  heads: number,
  seqLen: number,
  headDim: number,
  seed = 42
): Tensor4D {
  let state = seed;
  const nextRand = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state / 4294967296) * 2 - 1; // uniformly distributed in [-1, 1]
  };

  return createTensor4D(batch, heads, seqLen, headDim, () => nextRand());
}
