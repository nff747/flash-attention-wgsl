/**
 * flash-attention-wgsl: Vectorized FlashAttention-2 (128-bit vec4<f32> Loads & FMAs)
 * Optimized memory bandwidth and instruction throughput for head dimensions divisible by 4.
 */

export const FLASH_ATTENTION_VEC4_WGSL_TEMPLATE = `
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
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;

// Shared workgroup memory storing vec4 elements
var<workgroup> s_k: array<vec4<f32>, {{BLOCK_C}} * {{HEAD_DIM_VEC4}}>;
var<workgroup> s_v: array<vec4<f32>, {{BLOCK_C}} * {{HEAD_DIM_VEC4}}>;

const NEG_INF: f32 = -1e9;

@compute @workgroup_size({{BLOCK_R}}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tr = workgroup_id.x;
  let h  = workgroup_id.y;
  let b  = workgroup_id.z;

  let r = local_id.x;
  let global_row = tr * uniforms.block_r + r;

  let head_dim_v4 = uniforms.head_dim / 4u;
  let seq_len = uniforms.seq_len;
  let num_heads = uniforms.num_heads;

  let bh_offset_v4 = (b * num_heads + h) * seq_len * head_dim_v4;

  var m_prev: f32 = NEG_INF;
  var l_prev: f32 = 0.0;
  var acc_o: array<vec4<f32>, {{HEAD_DIM_VEC4}}>;

  for (var d = 0u; d < head_dim_v4; d = d + 1u) {
    acc_o[d] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  var reg_q: array<vec4<f32>, {{HEAD_DIM_VEC4}}>;
  let valid_row = global_row < seq_len;
  if (valid_row) {
    let q_offset = bh_offset_v4 + global_row * head_dim_v4;
    for (var d = 0u; d < head_dim_v4; d = d + 1u) {
      reg_q[d] = Q[q_offset + d];
    }
  }

  let Tc = (seq_len + uniforms.block_c - 1u) / uniforms.block_c;

  for (var tc = 0u; tc < Tc; tc = tc + 1u) {
    let col_start = tc * uniforms.block_c;

    if (uniforms.is_causal != 0u && col_start > (tr + 1u) * uniforms.block_r) {
      continue;
    }

    let total_v4_elements = uniforms.block_c * head_dim_v4;
    let threads_in_wg = uniforms.block_r;

    for (var idx = r; idx < total_v4_elements; idx = idx + threads_in_wg) {
      let c = idx / head_dim_v4;
      let d = idx % head_dim_v4;
      let global_col = col_start + c;

      if (global_col < seq_len) {
        let kv_offset = bh_offset_v4 + global_col * head_dim_v4 + d;
        s_k[idx] = K[kv_offset];
        s_v[idx] = V[kv_offset];
      } else {
        s_k[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        s_v[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      }
    }

    workgroupBarrier();

    if (valid_row) {
      var m_block: f32 = NEG_INF;
      var s_tile: array<f32, {{BLOCK_C}}>;

      let actual_bc = min(uniforms.block_c, seq_len - col_start);

      for (var c = 0u; c < actual_bc; c = c + 1u) {
        let global_col = col_start + c;

        if (uniforms.is_causal != 0u && global_col > global_row) {
          s_tile[c] = NEG_INF;
          continue;
        }

        var dot_sum: f32 = 0.0;
        let k_tile_offset = c * head_dim_v4;
        for (var d = 0u; d < head_dim_v4; d = d + 1u) {
          dot_sum = dot_sum + dot(reg_q[d], s_k[k_tile_offset + d]);
        }

        let score = dot_sum * uniforms.scale;
        s_tile[c] = score;
        if (score > m_block) {
          m_block = score;
        }
      }

      if (m_block > -1e8) {
        let m_new = max(m_prev, m_block);
        let alpha = select(exp(m_prev - m_new), 0.0, m_prev <= NEG_INF);

        var p_sum: f32 = 0.0;
        var p_tile: array<f32, {{BLOCK_C}}>;

        for (var c = 0u; c < actual_bc; c = c + 1u) {
          let s = s_tile[c];
          if (s > -1e8) {
            let p = exp(s - m_new);
            p_tile[c] = p;
            p_sum = p_sum + p;
          } else {
            p_tile[c] = 0.0;
          }
        }

        for (var d = 0u; d < head_dim_v4; d = d + 1u) {
          var p_dot_v: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
          for (var c = 0u; c < actual_bc; c = c + 1u) {
            let p = p_tile[c];
            if (p > 0.0) {
              p_dot_v = p_dot_v + s_v[c * head_dim_v4 + d] * p;
            }
          }
          acc_o[d] = acc_o[d] * alpha + p_dot_v;
        }

        l_prev = alpha * l_prev + p_sum;
        m_prev = m_new;
      }
    }

    workgroupBarrier();
  }

  if (valid_row) {
    let out_offset = bh_offset_v4 + global_row * head_dim_v4;
    let inv_l = select(1.0 / l_prev, 0.0, l_prev <= 0.0);

    for (var d = 0u; d < head_dim_v4; d = d + 1u) {
      O[out_offset + d] = acc_o[d] * inv_l;
    }
  }
}
`;
