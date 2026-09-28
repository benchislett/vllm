/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */
#pragma once

template <int Threads, int MinBlocks, bool FuseFp8>
__global__
__launch_bounds__(Threads, MinBlocks) void gdn_decode_post_conv_wide_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens,
    __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, void* __restrict__ out, int H, int HV,
    int state_indices_width, int dt_bias_type, bool norm_weight_is_bf16,
    float scale, float norm_eps, GdnDecodeStrides strides) {
  constexpr int kThreads = Threads;
  constexpr int kWarps = kThreads / 32;
  constexpr int kChunkV = kWarps * 4;
  constexpr int kNumChunks = kDimV / kChunkV;
  constexpr int kRowsPerWarp = 4;
  constexpr int kStages = kNumChunks == 1 ? 1 : 2;
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  // The final request also owns the padded output tail.
  if (request == gridDim.x - 1) {
    for (int64_t linear = tid; linear < (strides.output_tokens - eos) * kDimV;
         linear += kThreads) {
      const int64_t token = eos + linear / kDimV;
      store_gdn_output<FuseFp8>(
          out, strides, (token * HV + value_head) * kDimV + linear % kDimV,
          __float2bfloat16(0.0f));
    }
  }
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || num_tokens > 16) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      store_gdn_output<FuseFp8>(out, strides, out_offset,
                                __float2bfloat16(0.0f));
    }
    return;
  }

  const int key_head = value_head / 8;
  extern __shared__ __align__(16) unsigned char dynamic_shared_state[];
  auto shared_state =
      reinterpret_cast<__nv_bfloat16(*)[kChunkV][kDimK]>(dynamic_shared_state);
  __shared__ float shared_q[16][kDimK];
  __shared__ float shared_k[16][kDimK];
  __shared__ __nv_bfloat16 shared_v[16][kDimV];
  __shared__ __nv_bfloat16 shared_out[16][kDimV];
  __shared__ float shared_decay[16];
  __shared__ float shared_beta[16];

  __nv_bfloat16* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<__nv_bfloat16, kChunkV, kDimK, kStages>(
      &shared_state[0][0][0], source_state, 0, tid, kThreads);

  for (int t = warp; t < num_tokens; t += kWarps) {
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<__nv_bfloat16, kChunkV, kDimK, kStages>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

#pragma unroll 8
    for (int t = 0; t < 16; ++t) {
      if (t >= num_tokens) {
        break;
      }
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }

      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        __nv_bfloat16* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          const float4 updated =
              make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
          store_state4_vectorized(destination_state + value * kDimK + k_base,
                                  updated);
        }
      }
    }
  }
  __syncthreads();

  for (int t = warp; t < num_tokens; t += kWarps) {
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate = silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      store_gdn_output<FuseFp8>(
          out, strides, out_offset,
          __float2bfloat16(output_values[i] * rstd * weight * gate));
    }
  }
}

template <bool FuseFp8>
void launch_gdn_decode_wide(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& norm_weight,
    torch::stable::Tensor& out, const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* output_gate, int num_key_heads, int num_value_heads,
    double scale, double norm_eps, GdnDecodeStrides strides) {
  torch::stable::accelerator::DeviceGuard const guard(
      mixed_qkv.get_device_index());
  const auto stream = get_current_cuda_stream(mixed_qkv.get_device_index());
  const int num_requests = state_indices.size(0);
  auto launch_wide = [&]<int Threads, int MinBlocks>() {
    constexpr int Chunk = Threads / 32 * 4;
    constexpr int Stages = Chunk == 128 ? 1 : 2;
    constexpr int SharedBytes =
        Threads > 256 ? Stages * Chunk * 128 * sizeof(__nv_bfloat16) : 0;
    if constexpr (SharedBytes > 0) {
      static thread_local int configured_device = -1;
      if (configured_device != mixed_qkv.get_device_index()) {
        const auto error = cudaFuncSetAttribute(
            gdn_decode_post_conv_wide_kernel<Threads, MinBlocks, FuseFp8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
        STD_TORCH_CHECK(
            error == cudaSuccess,
            "Wide GDN shared memory setup failed: ", cudaGetErrorString(error));
        configured_device = mixed_qkv.get_device_index();
      }
    }
    // The generic convolution predecessor is ordered on this stream; no PDL
    // wait is needed.
    gdn_decode_post_conv_wide_kernel<Threads, MinBlocks, FuseFp8>
        <<<dim3(num_requests, num_value_heads), Threads, SharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), a, b,
            static_cast<const float*>(a_log.data_ptr()), dt_bias.data_ptr(),
            static_cast<const int*>(state_indices.data_ptr()),
            static_cast<const int*>(cu_seqlens.data_ptr()),
            static_cast<const int*>(num_accepted_tokens.data_ptr()),
            static_cast<__nv_bfloat16*>(state.data_ptr()), output_gate,
            norm_weight.data_ptr(), out.data_ptr(), num_key_heads,
            num_value_heads, static_cast<int>(state_indices.size(1)),
            kDtBiasBFloat16, true, static_cast<float>(scale),
            static_cast<float>(norm_eps), strides);
  };

  // Whole-head CTAs favor the smallest grids; split heads bound registers
  // once enough independent heads are available to fill the device.
  const int width = state_indices.size(1);
  if ((num_requests == 2 && width == 12) ||
      (num_value_heads == 16 && num_requests == 1 && width == 16)) {
    launch_wide.template operator()<1024, 1>();
  } else if (num_requests * num_value_heads <= 128) {
    launch_wide.template operator()<1024, 2>();
  } else {
    launch_wide.template operator()<512, 2>();
  }
  const auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "Wide GDN launch failed: ", cudaGetErrorString(error));
}
