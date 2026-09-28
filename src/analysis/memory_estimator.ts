/**
 * flash-attention-wgsl: VRAM Footprint & OOM Threshold Estimator
 * Models hardware buffer limits and predicts WebGPU memory overflow points.
 */

export interface MemoryComparisonPoint {
  readonly seqLen: number;
  /** VRAM consumed by Naive Attention (Bytes) */
  readonly naiveVramBytes: number;
  /** VRAM consumed by FlashAttention (Bytes) */
  readonly flashVramBytes: number;
  /** Percentage of VRAM saved: (1 - flash / naive) * 100 */
  readonly savingsPercent: number;
  /** Exceeds standard WebGPU maxStorageBufferBindingSize (usually 128 MB or 2 GB max) */
  readonly naiveExceedsMaxBindingSize: boolean;
  readonly flashExceedsMaxBindingSize: boolean;
}

export const WEBGPU_DEFAULT_MAX_STORAGE_BUFFER = 134217728; // 128 MB default limit
export const WEBGPU_EXTENDED_MAX_STORAGE_BUFFER = 2147483648; // 2 GB upper spec limit

export function estimateMemoryScaling(
  batch: number,
  heads: number,
  headDim: number,
  seqLengths: number[] = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
): MemoryComparisonPoint[] {
  const bytesPerFloat = 4;

  return seqLengths.map((seqLen) => {
    // Q, K, V, O inputs/outputs exist in both engines:
    const qkvoBytes = batch * heads * seqLen * headDim * bytesPerFloat * 4;

    // Naive intermediate matrix S and P (each batch * heads * seqLen^2 * 4 bytes)
    const naiveIntermediates = batch * heads * seqLen * seqLen * bytesPerFloat * 2;
    const naiveVramBytes = qkvoBytes + naiveIntermediates;

    // FlashAttention requires NO intermediate global memory allocations!
    const flashVramBytes = qkvoBytes;

    const savingsPercent = ((naiveVramBytes - flashVramBytes) / naiveVramBytes) * 100.0;

    // A single buffer in naive attention (S or P matrix) size:
    const singleNaiveBuffer = batch * heads * seqLen * seqLen * bytesPerFloat;
    const singleFlashBuffer = batch * heads * seqLen * headDim * bytesPerFloat;

    return {
      seqLen,
      naiveVramBytes,
      flashVramBytes,
      savingsPercent,
      naiveExceedsMaxBindingSize: singleNaiveBuffer > WEBGPU_DEFAULT_MAX_STORAGE_BUFFER,
      flashExceedsMaxBindingSize: singleFlashBuffer > WEBGPU_DEFAULT_MAX_STORAGE_BUFFER
    };
  });
}
