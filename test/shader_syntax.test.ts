import { describe, it, expect } from 'vitest';
import {
  compileFlashAttentionV2WGSL,
  compileFlashAttentionVec4WGSL,
  getNaiveAttentionWGSL
} from '../src/shaders';

describe('WGSL Shader Compilation & Grammar Checks', () => {
  it('should compile standard FlashAttention-2 shader with correct tile parameters', () => {
    const wgsl = compileFlashAttentionV2WGSL({
      blockSizeR: 32,
      blockSizeC: 32,
      headDim: 64
    });

    // Check placeholders were cleanly replaced
    expect(wgsl).not.toContain('{{BLOCK_R}}');
    expect(wgsl).not.toContain('{{BLOCK_C}}');
    expect(wgsl).not.toContain('{{HEAD_DIM}}');

    // Check critical WGSL syntax elements
    expect(wgsl).toContain('@compute @workgroup_size(32, 1, 1)');
    expect(wgsl).toContain('var<workgroup> s_k: array<f32, 32 * 64>;');
    expect(wgsl).toContain('var<workgroup> s_v: array<f32, 32 * 64>;');
    expect(wgsl).toContain('workgroupBarrier();');
    expect(wgsl).toContain('@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;');
    expect(wgsl).toContain('@group(0) @binding(1) var<storage, read> Q: array<f32>;');
    expect(wgsl).toContain('@group(0) @binding(4) var<storage, read_write> O: array<f32>;');
  });

  it('should compile vectorized vec4 FlashAttention shader', () => {
    const wgsl = compileFlashAttentionVec4WGSL({
      blockSizeR: 32,
      blockSizeC: 32,
      headDim: 64
    });

    // For headDim=64, headDimVec4 = 16
    expect(wgsl).not.toContain('{{HEAD_DIM_VEC4}}');
    expect(wgsl).toContain('var<workgroup> s_k: array<vec4<f32>, 32 * 16>;');
    expect(wgsl).toContain('var<workgroup> s_v: array<vec4<f32>, 32 * 16>;');
    expect(wgsl).toContain('dot(reg_q[d], s_k[k_tile_offset + d])');
  });

  it('should reject vec4 compilation when headDim is not divisible by 4', () => {
    expect(() => {
      compileFlashAttentionVec4WGSL({
        blockSizeR: 32,
        blockSizeC: 32,
        headDim: 50 // Not divisible by 4
      });
    }).toThrowError(/divisible by 4/);
  });

  it('should return valid Naive Attention baseline shader', () => {
    const wgsl = getNaiveAttentionWGSL();
    expect(wgsl).toContain('fn compute_scores');
    expect(wgsl).toContain('fn softmax_and_pv');
    expect(wgsl).toContain('var<storage, read_write> S_matrix: array<f32>;');
  });
});
