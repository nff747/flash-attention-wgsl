/**
 * flash-attention-wgsl: FlashAttention-2 Workgroup Tiled WGSL Compute Shader
 * Online Softmax reduction with collaborative SRAM tile loading.
 */

export const FLASH_ATTENTION_V2_WGSL_TEMPLATE = `
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

// Workgroup shared memory (SRAM) for K and V tiles
var<workgroup> s_k: array<f32, {{BLOCK_C}} * {{HEAD_DIM}}>;
var<workgroup> s_v: array<f32, {{BLOCK_C}} * {{HEAD_DIM}}>;

const NEG_INF: f32 = -1e9;

@compute @workgroup_size({{BLOCK_R}}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tr = workgroup_id.x; // Row block index [0, Tr - 1]
  let h  = workgroup_id.y; // Head index [0, num_heads - 1]
  let b  = workgroup_id.z; // Batch index [0, batch_size - 1]

  let r = local_id.x; // Thread index within row block [0, Br - 1]
  let global_row = tr * uniforms.block_r + r;

  let head_dim = uniforms.head_dim;
  let seq_len = uniforms.seq_len;
  let num_heads = uniforms.num_heads;

  let bh_offset = (b * num_heads + h) * seq_len * head_dim;

  // Thread-local register state for Online Softmax
  var m_prev: f32 = NEG_INF;
  var l_prev: f32 = 0.0;
  var acc_o: array<f32, {{HEAD_DIM}}>;

  // Initialize output register accumulator
  for (var d = 0u; d < head_dim; d = d + 1u) {
    acc_o[d] = 0.0;
  }

  // Load thread's own Q row into registers (persists throughout inner loop)
  var reg_q: array<f32, {{HEAD_DIM}}>;
  let valid_row = global_row < seq_len;
  if (valid_row) {
    let q_offset = bh_offset + global_row * head_dim;
    for (var d = 0u; d < head_dim; d = d + 1u) {
      reg_q[d] = Q[q_offset + d];
    }
  }

  let Tc = (seq_len + uniforms.block_c - 1u) / uniforms.block_c;

  // Inner Loop: Iterate through K, V blocks in SRAM
  for (var tc = 0u; tc < Tc; tc = tc + 1u) {
    let col_start = tc * uniforms.block_c;

    // Causal Masking optimization: completely skip strictly upper-triangular tiles
    if (uniforms.is_causal != 0u && col_start > (tr + 1u) * uniforms.block_r) {
      continue;
    }

    // Collaborative loading of K and V into shared workgroup memory
    let total_elements = uniforms.block_c * head_dim;
    let threads_in_wg = uniforms.block_r;

    for (var idx = r; idx < total_elements; idx = idx + threads_in_wg) {
      let c = idx / head_dim;
      let d = idx % head_dim;
      let global_col = col_start + c;

      if (global_col < seq_len) {
        let kv_offset = bh_offset + global_col * head_dim + d;
        s_k[idx] = K[kv_offset];
        s_v[idx] = V[kv_offset];
      } else {
        s_k[idx] = 0.0;
        s_v[idx] = 0.0;
      }
    }

    workgroupBarrier();

    if (valid_row) {
      // Find block max of S_ij = (Q_i . K_j) * scale
      var m_block: f32 = NEG_INF;
      var s_tile: array<f32, {{BLOCK_C}}>;

      let actual_bc = min(uniforms.block_c, seq_len - col_start);

      for (var c = 0u; c < actual_bc; c = c + 1u) {
        let global_col = col_start + c;

        if (uniforms.is_causal != 0u && global_col > global_row) {
          s_tile[c] = NEG_INF;
          continue;
        }

        var dot: f32 = 0.0;
        let k_tile_offset = c * head_dim;
        for (var d = 0u; d < head_dim; d = d + 1u) {
          dot = dot + reg_q[d] * s_k[k_tile_offset + d];
        }

        let score = dot * uniforms.scale;
        s_tile[c] = score;
        if (score > m_block) {
          m_block = score;
        }
      }

      // Online Softmax update if at least one unmasked element exists
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

        // Rescale previous output accumulator and accumulate P_tilde @ V_tile
        for (var d = 0u; d < head_dim; d = d + 1u) {
          var p_dot_v: f32 = 0.0;
          for (var c = 0u; c < actual_bc; c = c + 1u) {
            let p = p_tile[c];
            if (p > 0.0) {
              p_dot_v = p_dot_v + p * s_v[c * head_dim + d];
            }
          }
          acc_o[d] = alpha * acc_o[d] + p_dot_v;
        }

        l_prev = alpha * l_prev + p_sum;
        m_prev = m_new;
      }
    }

    workgroupBarrier();
  }

  // Epilogue: final normalization and store to global memory
  if (valid_row) {
    let out_offset = bh_offset + global_row * head_dim;
    let inv_l = select(1.0 / l_prev, 0.0, l_prev <= 0.0);

    for (var d = 0u; d < head_dim; d = d + 1u) {
      O[out_offset + d] = acc_o[d] * inv_l;
    }
  }
}
`;
