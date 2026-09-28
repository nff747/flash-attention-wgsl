/**
 * flash-attention-wgsl: Naive O(N^2) Scaled Dot-Product Attention Shader
 * Baseline shader allocating full N x N intermediate attention matrix in global VRAM.
 */

export const NAIVE_ATTENTION_WGSL = `
struct AttentionUniforms {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  block_r: u32,
  block_c: u32,
};

@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<storage, read_write> S_matrix: array<f32>; // Full N x N materialized matrix

const NEG_INF: f32 = -1e9;

// Stage 1: Compute S = (Q K^T) * scale in global VRAM
@compute @workgroup_size(16, 16, 1)
fn compute_scores(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let row = global_id.x;
  let col = global_id.y;
  let bh  = global_id.z; // (b * num_heads + h)

  let seq_len = uniforms.seq_len;
  let head_dim = uniforms.head_dim;

  if (row >= seq_len || col >= seq_len) {
    return;
  }

  let s_offset = bh * seq_len * seq_len + row * seq_len + col;

  if (uniforms.is_causal != 0u && col > row) {
    S_matrix[s_offset] = NEG_INF;
    return;
  }

  let q_offset = bh * seq_len * head_dim + row * head_dim;
  let k_offset = bh * seq_len * head_dim + col * head_dim;

  var dot: f32 = 0.0;
  for (var d = 0u; d < head_dim; d = d + 1u) {
    dot = dot + Q[q_offset + d] * K[k_offset + d];
  }

  S_matrix[s_offset] = dot * uniforms.scale;
}

// Stage 2: Softmax along rows and multiply by V to write O
@compute @workgroup_size(64, 1, 1)
fn softmax_and_pv(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let row = global_id.x;
  let bh  = global_id.y;

  let seq_len = uniforms.seq_len;
  let head_dim = uniforms.head_dim;

  if (row >= seq_len) {
    return;
  }

  let s_row_offset = bh * seq_len * seq_len + row * seq_len;

  // Pass 1: Find row max
  var m: f32 = NEG_INF;
  for (var j = 0u; j < seq_len; j = j + 1u) {
    let s = S_matrix[s_row_offset + j];
    if (s > m) {
      m = s;
    }
  }

  // Pass 2: Exponentiate and sum
  var sum_exp: f32 = 0.0;
  for (var j = 0u; j < seq_len; j = j + 1u) {
    let s = S_matrix[s_row_offset + j];
    var p: f32 = 0.0;
    if (s > -1e8) {
      p = exp(s - m);
    }
    S_matrix[s_row_offset + j] = p;
    sum_exp = sum_exp + p;
  }

  let inv_sum = select(1.0 / sum_exp, 0.0, sum_exp <= 0.0);

  // Pass 3: Multiply normalized P by V
  let out_row_offset = bh * seq_len * head_dim + row * head_dim;
  let v_bh_offset = bh * seq_len * head_dim;

  for (var d = 0u; d < head_dim; d = d + 1u) {
    var acc: f32 = 0.0;
    for (var j = 0u; j < seq_len; j = j + 1u) {
      let p_norm = S_matrix[s_row_offset + j] * inv_sum;
      if (p_norm > 0.0) {
        acc = acc + p_norm * V[v_bh_offset + j * head_dim + d];
      }
    }
    O[out_row_offset + d] = acc;
  }
}
`;
