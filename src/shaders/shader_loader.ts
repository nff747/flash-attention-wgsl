/**
 * flash-attention-wgsl: Dynamic WGSL Shader Generator
 * Compiles parametric templates with optimal workgroup sizes and hardware tile bounds.
 */

import { FLASH_ATTENTION_V2_WGSL_TEMPLATE } from './flash_attention_v2.wgsl';
import { FLASH_ATTENTION_VEC4_WGSL_TEMPLATE } from './flash_attention_vec4.wgsl';
import { NAIVE_ATTENTION_WGSL } from './naive_attention.wgsl';

export interface ShaderCompilationConfig {
  blockSizeR: number;
  blockSizeC: number;
  headDim: number;
}

export function compileFlashAttentionV2WGSL(config: ShaderCompilationConfig): string {
  const { blockSizeR, blockSizeC, headDim } = config;

  if (blockSizeR <= 0 || blockSizeC <= 0 || headDim <= 0) {
    throw new Error(`Invalid tile dimensions: Br=${blockSizeR}, Bc=${blockSizeC}, d=${headDim}`);
  }

  return FLASH_ATTENTION_V2_WGSL_TEMPLATE
    .replace(/\{\{BLOCK_R\}\}/g, blockSizeR.toString())
    .replace(/\{\{BLOCK_C\}\}/g, blockSizeC.toString())
    .replace(/\{\{HEAD_DIM\}\}/g, headDim.toString());
}

export function compileFlashAttentionVec4WGSL(config: ShaderCompilationConfig): string {
  const { blockSizeR, blockSizeC, headDim } = config;

  if (headDim % 4 !== 0) {
    throw new Error(`Vectorized vec4 kernel requires headDim divisible by 4, got ${headDim}`);
  }

  const headDimVec4 = headDim / 4;

  return FLASH_ATTENTION_VEC4_WGSL_TEMPLATE
    .replace(/\{\{BLOCK_R\}\}/g, blockSizeR.toString())
    .replace(/\{\{BLOCK_C\}\}/g, blockSizeC.toString())
    .replace(/\{\{HEAD_DIM_VEC4\}\}/g, headDimVec4.toString());
}

export function getNaiveAttentionWGSL(): string {
  return NAIVE_ATTENTION_WGSL;
}
