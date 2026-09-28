# flash-attention-wgsl

[![CI](https://github.com/nff747/flash-attention-wgsl/actions/workflows/ci.yml/badge.svg)](https://github.com/nff747/flash-attention-wgsl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Architecture: FlashAttention-2](https://img.shields.io/badge/Architecture-FlashAttention--2-emerald.svg)]
[![WebGPU](https://img.shields.io/badge/WebGPU-WGSL-orange.svg)](https://www.w3.org/TR/webgpu/)

Hardware-tiled, online-softmax **FlashAttention-2** compute engine in WebGPU & WGSL. Eliminates $O(N^2)$ intermediate attention matrix materialization by computing scaled dot-product attention in high-speed on-chip SRAM (`var<workgroup>` memory) with exact mathematical parity to standard attention ($\epsilon < 10^{-5}$).

---

## The Problem: High-Bandwidth Memory (HBM) Bottlenecks

Standard Multi-Head Attention computes:
$$S = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{N \times N}$$
$$P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$$
$$O = P V \in \mathbb{R}^{N \times d}$$

In conventional GPU implementations, materializing the intermediate $N \times N$ matrices $S$ and $P$ creates two severe bottlenecks:
1. **$O(N^2)$ VRAM Footprint**: At sequence length $N = 16,384$ with $H = 16$ heads, storing $S$ and $P$ in 32-bit floats consumes **32 GB of VRAM**, exceeding WebGPU buffer binding limits (`maxStorageBufferBindingSize` default: 128 MB, max: 2 GB).
2. **Memory Bandwidth Throttling**: Standard attention spends over 80% of its execution time reading and writing $S$ and $P$ to global device memory rather than performing arithmetic computation.

---

## FlashAttention-2 Architecture & WGSL Tiling

`flash-attention-wgsl` tiles the inputs $Q, K, V$ into blocks that fit entirely within workgroup shared memory (`var<workgroup>`).

```
 Global Memory (VRAM)                         Workgroup Shared Memory (SRAM)
┌─────────────────────────────────┐           ┌─────────────────────────────┐
│  Q: [Batch, Heads, N, d]        │──Tile Br─▶│  s_q: array<f32, Br * d>    │
│  K: [Batch, Heads, N, d]        │──Tile Bc─▶│  s_k: array<f32, Bc * d>    │
│  V: [Batch, Heads, N, d]        │──Tile Bc─▶│  s_v: array<f32, Bc * d>    │
└─────────────────────────────────┘           └─────────────────────────────┘
                                                             │
                                              Tile GEMM: S_ij = Q_i @ K_j^T
                                                             │
                                              Online Softmax Scaling (m_i, l_i)
                                                             │
                                              Accumulate O_i += P_ij @ V_j
                                                             │
┌─────────────────────────────────┐                          ▼
│  O: [Batch, Heads, N, d]        │◀───────Final Normalization (O_i / l_i)
└─────────────────────────────────┘
```

### Online Softmax Mathematical Formulation

Rather than requiring the full sequence of logits to evaluate the softmax normalizer $\sum_j e^{S_{i, j}}$, FlashAttention updates running statistics incrementally:

For a given row $i$ and block step $j \in [1 \dots T_c]$:

1. **Local Block Max**:
   $$m_{\text{block}} = \max_{k \in [0, B_c - 1]} S_{i, k}^{(j)}$$

2. **Updated Running Max**:
   $$m^{(j)} = \max(m^{(j-1)}, m_{\text{block}})$$

3. **Rescaling Factor**:
   $$\alpha = e^{m^{(j-1)} - m^{(j)}}$$

4. **Running Denominator Normalizer**:
   $$l^{(j)} = \alpha \cdot l^{(j-1)} + \sum_{k} e^{S_{i, k}^{(j)} - m^{(j)}}$$

5. **Running Output Vector Accumulator**:
   $$O^{(j)} = \alpha \cdot O^{(j-1)} + \tilde{P}^{(j)} V^{(j)}$$
   $$\text{where } \tilde{P}^{(j)} = \exp(S^{(j)} - m^{(j)})$$

6. **Final Epilogue**:
   $$O = \frac{1}{l^{(T_c)}} O^{(T_c)}$$

---

## Memory Footprint & Out-of-Memory (OOM) Thresholds

Measured VRAM footprint across sequence lengths ($B = 1, H = 16, d = 64, \text{float32}$):

| Sequence Length ($N$) | Naive Attention VRAM | FlashAttention-2 VRAM | VRAM Saved (%) | WebGPU 128 MB Limit Status |
|:---:|:---:|:---:|:---:|:---:|
| **512** | 40.0 MB | 8.0 MB | **80.0%** | Both Safe |
| **1,024** | 144.0 MB | 16.0 MB | **88.9%** | Flash Only (Naive OOM) |
| **2,048** | 544.0 MB | 32.0 MB | **94.1%** | Flash Only (Naive OOM) |
| **4,096** | 2,112.0 MB | 64.0 MB | **97.0%** | Flash Only (Naive OOM) |
| **8,192** | 8,320.0 MB | 128.0 MB | **98.5%** | Flash Only (Naive OOM) |
| **16,384** | 33,024.0 MB (33 GB) | 256.0 MB | **99.2%** | Flash Only (Naive OOM) |
| **32,768** | 131,584.0 MB (131 GB)| 512.0 MB | **99.6%** | Flash Only (Naive OOM) |

---

## Features

- **WebGPU WGSL Compute Kernels**:
  - `flash_attention_v2.wgsl`: Parametric workgroup tiling (`workgroup_size(Br, 1, 1)`), collaborative SRAM loading, barrier synchronization.
  - `flash_attention_vec4.wgsl`: 128-bit vectorized `vec4<f32>` memory loads and dot products.
  - `naive_attention.wgsl`: Baseline global-memory shader for profiling and verification.
- **Autoregressive Causal Masking**: Automatically skips strictly upper-triangular tiles ($t_c \cdot B_c > (t_r + 1) \cdot B_r$), saving 50% of FLOPs and memory transactions.
- **CPU Reference Engines**:
  - `naiveAttention()`: Exact $O(N^2)$ baseline.
  - `flashAttentionCPU()`: Hardware-faithful block tiling simulation with running accumulators.
- **Dynamic Shader Generator**: Compiles WGSL at runtime with model-tailored $B_r, B_c, d$ constants.
- **Analysis Suite**: Exact FLOPs calculators, memory IO bandwidth models, and OOM threshold estimators.

---

## Installation

```bash
npm install flash-attention-wgsl
```

---

## Quickstart

### 1. CPU Reference Engine

```typescript
import {
  createRandomTensor4D,
  flashAttentionCPU,
  naiveAttention,
  maxAbsoluteError
} from 'flash-attention-wgsl';

// Shape: [batch=1, heads=8, seqLen=512, headDim=64]
const Q = createRandomTensor4D(1, 8, 512, 64);
const K = createRandomTensor4D(1, 8, 512, 64);
const V = createRandomTensor4D(1, 8, 512, 64);

const config = {
  causal: true,
  blockSizeR: 32,
  blockSizeC: 32
};

const result = flashAttentionCPU(Q, K, V, config);
console.log(`Executed blocks: ${result.tilingMetrics.executedBlocks}`);
console.log(`Skipped causal blocks: ${result.tilingMetrics.skippedCausalBlocks}`);
```

### 2. WebGPU Pipeline Execution

```typescript
import {
  initWebGPUContext,
  FlashAttentionPipeline,
  createRandomTensor4D
} from 'flash-attention-wgsl';

const gpu = await initWebGPUContext();
if (gpu) {
  const pipeline = new FlashAttentionPipeline(gpu.device);

  const Q = createRandomTensor4D(1, 4, 256, 64);
  const K = createRandomTensor4D(1, 4, 256, 64);
  const V = createRandomTensor4D(1, 4, 256, 64);

  const output = await pipeline.execute(Q, K, V, { causal: true });
  console.log('FlashAttention computed on WebGPU:', output.length);
}
```

---

## Verification & Testing

Every implementation is numerically validated against naive scaled dot-product attention:

```bash
# Run Vitest verification suite
npm test

# Run typecheck
npm run typecheck

# Run benchmark suite
npm run bench
```

---

## License

MIT © [nff747](https://github.com/nff747)
