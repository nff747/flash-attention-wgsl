import { describe, it, expect } from 'vitest';
import {
  computeAttentionComplexity,
  estimateMemoryScaling
  
} from '../src/analysis';

describe('Attention Complexity & Memory Profiling', () => {
  it('should accurately calculate quadratic FLOPs and causal reduction', () => {
    const batch = 1;
    const heads = 1;
    const seqLen = 4096;
    const headDim = 32;
    const blockSizeR = 64;

    const full = computeAttentionComplexity(batch, heads, seqLen, headDim, false, blockSizeR);
    const causal = computeAttentionComplexity(batch, heads, seqLen, headDim, true, blockSizeR);

    // In causal mode, FLOPs should be exactly half of non-causal
    expect(causal.totalFlops).toBeCloseTo(full.totalFlops * 0.5, 0);

    // For seqLen=4096 with Br=64, FlashAttention requires substantially less HBM memory traffic
    expect(full.naiveHbmBytes).toBeGreaterThan(full.flashHbmBytes);
    expect(full.hbmSpeedupFactor).toBeGreaterThan(1.5);
  });

  it('should predict WebGPU buffer overflow in naive attention as sequence length scales', () => {
    const batch = 1;
    const heads = 8;
    const headDim = 64;

    const points = estimateMemoryScaling(batch, heads, headDim);

    // At short sequences (128), naive attention fits easily
    const p128 = points.find((p) => p.seqLen === 128)!;
    expect(p128.naiveExceedsMaxBindingSize).toBe(false);

    // At sequence length 8192, intermediate matrix S (8 * 8192^2 * 4 bytes = 2.14 GB)
    // overwhelmingly exceeds default WebGPU 128 MB max storage buffer limit!
    const p8192 = points.find((p) => p.seqLen === 8192)!;
    expect(p8192.naiveExceedsMaxBindingSize).toBe(true);

    // Meanwhile, FlashAttention does not materialize S or P:
    // Q, K, V, O each only need 8 * 8192 * 64 * 4 bytes = 16.7 MB, well below 128 MB!
    expect(p8192.flashExceedsMaxBindingSize).toBe(false);
    expect(p8192.savingsPercent).toBeGreaterThan(90.0);
  });
});

import { validateSRAMTileConfiguration } from '../src/analysis/memory_estimator';

describe('SRAM Tile Storage Validation', () => {
  it('validates whether SRAM tile sizes fit within 16 KB WebGPU workgroup limit', () => {
    // 16x16 tile with headDim 32:
    // Q: 16*32*4 = 2048
    // K: 16*32*4 = 2048
    // V: 16*32*4 = 2048
    // S: 16*16*4 = 1024
    // Total = 7168 bytes (< 16384 bytes) -> fits!
    const validConfig = validateSRAMTileConfiguration(16, 16, 32);
    expect(validConfig.workgroupSharedBytes).toBe(7168);
    expect(validConfig.fitsInMaxComputeWorkgroupStorage).toBe(true);

    // Oversized 64x64 tile with headDim 64:
    // Q: 64*64*4 = 16384
    // Exceeds 16 KB by itself!
    const invalidConfig = validateSRAMTileConfiguration(64, 64, 64);
    expect(invalidConfig.workgroupSharedBytes).toBeGreaterThan(16384);
    expect(invalidConfig.fitsInMaxComputeWorkgroupStorage).toBe(false);
  });
});
