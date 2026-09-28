/**
 * flash-attention-wgsl: Hardware Benchmark & Performance Profiler
 * Measures execution latency, FLOPS throughput, and memory bandwidth.
 */

import { Tensor4D, AttentionConfig } from '../math/types';
import { computeAttentionComplexity, ComplexityAnalysis } from '../analysis/complexity';
import { flashAttentionCPU } from '../math/tiled_flash_attention';
import { naiveAttention } from '../math/naive_attention';

export interface BenchmarkMetrics {
  readonly name: string;
  readonly meanMs: number;
  readonly medianMs: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly gflops: number;
  readonly complexity: ComplexityAnalysis;
}

export function benchmarkCPUAttention(
  Q: Tensor4D,
  K: Tensor4D,
  V: Tensor4D,
  config: AttentionConfig,
  type: 'flash' | 'naive',
  iterations = 5,
  warmup = 2
): BenchmarkMetrics {
  const { batch, heads, seqLen, headDim } = Q.shape;
  const complexity = computeAttentionComplexity(
    batch,
    heads,
    seqLen,
    headDim,
    config.causal,
    config.blockSizeR ?? 32
  );

  const runner = type === 'flash'
    ? () => flashAttentionCPU(Q, K, V, config)
    : () => naiveAttention(Q, K, V, config);

  // Warmup
  for (let i = 0; i < warmup; i++) {
    runner();
  }

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    runner();
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  times.sort((a, b) => a - b);
  const minMs = times[0];
  const maxMs = times[times.length - 1];
  const medianMs = times[Math.floor(times.length / 2)];
  const meanMs = times.reduce((acc, t) => acc + t, 0) / times.length;

  const seconds = meanMs / 1000.0;
  const gflops = (complexity.totalFlops / seconds) / 1e9;

  return {
    name: type === 'flash' ? 'FlashAttention-2 (CPU Tiled)' : 'Naive Attention (O(N^2) Baseline)',
    meanMs,
    medianMs,
    minMs,
    maxMs,
    gflops,
    complexity
  };
}
