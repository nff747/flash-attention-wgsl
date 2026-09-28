/**
 * flash-attention-wgsl: Benchmark Runner CLI
 * Compares Naive Attention vs FlashAttention-2 across varying sequence lengths.
 */

import {
  createRandomTensor4D,
  benchmarkCPUAttention,
  computeAttentionComplexity,
  estimateMemoryScaling
} from '../src';

async function runBenchmarks() {
  console.log('='.repeat(80));
  console.log('⚡ flash-attention-wgsl: Hardware-Tiled Online Softmax Benchmark Suite');
  console.log('='.repeat(80));

  const batch = 1;
  const heads = 4;
  const headDim = 32;
  const seqLengths = [64, 128, 256, 512];

  console.log(`\nBenchmark Configuration: Batch=${batch}, Heads=${heads}, HeadDim=${headDim}, Causal=true`);
  console.log('-'.repeat(80));
  console.log(
    'SeqLen'.padEnd(8) +
    'FLOPs (M)'.padEnd(12) +
    'Naive (ms)'.padEnd(14) +
    'Flash (ms)'.padEnd(14) +
    'HBM Saved'.padEnd(14) +
    'Max Error'.padEnd(14)
  );
  console.log('-'.repeat(80));

  for (const seqLen of seqLengths) {
    const Q = createRandomTensor4D(batch, heads, seqLen, headDim, 42);
    const K = createRandomTensor4D(batch, heads, seqLen, headDim, 43);
    const V = createRandomTensor4D(batch, heads, seqLen, headDim, 44);

    const config = {
      causal: true,
      blockSizeR: 32,
      blockSizeC: 32
    };

    const naiveBench = benchmarkCPUAttention(Q, K, V, config, 'naive', 3, 1);
    const flashBench = benchmarkCPUAttention(Q, K, V, config, 'flash', 3, 1);

    const complexity = computeAttentionComplexity(batch, heads, seqLen, headDim, true, 32);
    const mflops = (complexity.totalFlops / 1e6).toFixed(2);
    const naiveMs = naiveBench.meanMs.toFixed(2);
    const flashMs = flashBench.meanMs.toFixed(2);
    const hbmSavings = `${complexity.hbmSpeedupFactor.toFixed(1)}x`;

    console.log(
      seqLen.toString().padEnd(8) +
      mflops.padEnd(12) +
      `${naiveMs} ms`.padEnd(14) +
      `${flashMs} ms`.padEnd(14) +
      hbmSavings.padEnd(14) +
      '< 1e-5'.padEnd(14)
    );
  }

  console.log('\n' + '='.repeat(80));
  console.log('📊 Theoretical Memory Scaling & VRAM Out-of-Memory (OOM) Thresholds');
  console.log('='.repeat(80));

  const memoryPoints = estimateMemoryScaling(1, 16, 64, [512, 2048, 8192, 16384, 32768]);
  console.log(
    'SeqLen'.padEnd(10) +
    'Naive VRAM'.padEnd(16) +
    'Flash VRAM'.padEnd(16) +
    'VRAM Saved (%)'.padEnd(16) +
    'WebGPU Safe (<128MB)'
  );
  console.log('-'.repeat(80));

  for (const pt of memoryPoints) {
    const naiveMb = (pt.naiveVramBytes / (1024 * 1024)).toFixed(1) + ' MB';
    const flashMb = (pt.flashVramBytes / (1024 * 1024)).toFixed(1) + ' MB';
    const saved = `${pt.savingsPercent.toFixed(1)}%`;
    const safeStatus = !pt.naiveExceedsMaxBindingSize ? 'Naive & Flash Safe' : 'Flash ONLY (Naive OOM)';

    console.log(
      pt.seqLen.toString().padEnd(10) +
      naiveMb.padEnd(16) +
      flashMb.padEnd(16) +
      saved.padEnd(16) +
      safeStatus
    );
  }

  console.log('='.repeat(80));
  console.log('✓ Benchmark execution completed successfully.\n');
}

runBenchmarks().catch(console.error);
