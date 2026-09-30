/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#pragma once

// Included inside the decode implementation's anonymous namespace, after its
// state-copy, BF16-store, and warp-reduction helpers.
struct GdnClusterParams {
  const __nv_bfloat16* mixed;
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  const float* a_log;
  const __nv_bfloat16* dt_bias;
  const int* indices;
  const int* cu_seqlens;
  const int* accepted;
  __nv_bfloat16* state;
  const __nv_bfloat16* gate;
  const __nv_bfloat16* norm_weight;
  void* out;
  int state_indices_width;
  GdnDecodeStrides strides;
};

template <bool EnablePdl>
__device__ __forceinline__ void gdn_dependency_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (EnablePdl) {
    cudaGridDependencySynchronize();
  }
#endif
}

// BS1: 16 CTAs per value head expose enough parallelism for a small batch.
// Each CTA owns eight V rows; each warp keeps one row's FP32 recurrence in
// registers. All eight BF16 state snapshots are still written to the cache.
template <bool EnablePdl, bool FuseFp8 = false>
__global__ __launch_bounds__(256) void gdn_decode_mtp_cluster_kernel(
    GdnClusterParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kRows = 8;
  constexpr int kSplits = kDimV / kRows;
  const int split = blockIdx.x % kSplits;
  const int head = blockIdx.x / kSplits;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int bos = p.cu_seqlens[0];
  const int num_tokens = p.cu_seqlens[1] - bos;
  const int accepted = p.accepted[0];
  const int source_slot = accepted > 0 && accepted <= p.state_indices_width
                              ? p.indices[accepted - 1]
                              : 0;
  // These branches are uniform across the entire cluster, so no member can
  // return while another member is waiting at a cluster barrier.
  if (num_tokens <= 0) {
    gdn_dependency_wait<EnablePdl>();
    return;
  }
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    gdn_dependency_wait<EnablePdl>();
    for (int i = tid; i < num_tokens * kRows; i += 256) {
      store_gdn_output<FuseFp8>(
          p.out, p.strides,
          ((bos + i / kRows) * 16 + head) * kDimV + split * kRows + i % kRows,
          __float2bfloat16(0.0f));
    }
    return;
  }

  __shared__ float q[kMaxMtpTokens][kDimK];
  __shared__ float k[kMaxMtpTokens][kDimK];
  __shared__ float decay[kMaxMtpTokens], beta[kMaxMtpTokens];
  __shared__ float kq[kMaxMtpTokens];
  __shared__ __nv_bfloat16 v[kMaxMtpTokens][kRows];
  __shared__ __nv_bfloat16 raw[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_state[kRows][kDimK];

  const __nv_bfloat16* source =
      p.state + static_cast<int64_t>(source_slot) * p.strides.state_slot +
      head * kDimV * kDimK + split * kRows * kDimK;
  copy_state_chunk<__nv_bfloat16, kRows, kDimK, 1>(&shared_state[0][0], source,
                                                   0, tid, 256);
  gdn_dependency_wait<EnablePdl>();
  if (warp < num_tokens) {
    const int t = warp;
    const int64_t base = static_cast<int64_t>(bos + t) * p.strides.mixed_row;
    float qv[4], kv[4];
    float qs = 0.0f, ks = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      qv[i] = __bfloat162float(p.mixed[base + (head / 8) * kDimK + d]);
      kv[i] =
          __bfloat162float(p.mixed[base + 2 * kDimK + (head / 8) * kDimK + d]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_reduce_sum(qs);
    ks = warp_reduce_sum(ks);
    const float qscale = __shfl_sync(
        0xffffffffu,
        lane == 0 ? rsqrtf(qs + 1e-6f) * 0.08838834764831845f : 0.0f, 0);
    const float kscale =
        __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(ks + 1e-6f) : 0.0f, 0);
    float dot = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[t][lane + i * 32] = qv[i] * qscale;
      k[t][lane + i * 32] = kv[i] * kscale;
      dot += (qv[i] * qscale) * (kv[i] * kscale);
    }
    dot = warp_reduce_sum(dot);
    if (lane < kRows) {
      v[t][lane] =
          p.mixed[base + 4 * kDimK + head * kDimV + split * kRows + lane];
    }
    if (lane == 0) {
      kq[t] = dot;
      const float av =
          __bfloat162float(p.a[(bos + t) * p.strides.a_row + head]);
      const float bv =
          __bfloat162float(p.b[(bos + t) * p.strides.b_row + head]);
      decay[t] = __expf(-__expf(p.a_log[head]) *
                        softplus_fast(av + __bfloat162float(p.dt_bias[head])));
      beta[t] = sigmoid_fast(bv);
    }
  }
  cp_async_wait_all();
  auto cluster = cooperative_groups::this_cluster();
  // Establish shared-memory lifetime before any remote stores to the leader.
  cluster.sync();
  float h[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    h[i] = __bfloat162float(shared_state[warp][lane * 4 + i]);
  }
  #pragma unroll
  for (int t = 0; t < kMaxMtpTokens; ++t) {
    if (t >= num_tokens) {
      break;
    }
    float qv[4], kv[4];
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      qv[i] = q[t][lane * 4 + i];
      kv[i] = k[t][lane * 4 + i];
    }
    float hk = 0.0f, hq = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[i] *= decay[t];
      hk += h[i] * kv[i];
      hq += h[i] * qv[i];
    }
    // (dH + delta*k)q = (dH)q + delta*(kq). The two independent
    // reductions overlap, and kq is shared by all rows of this head.
    const Sum2 dots = warp_reduce_sum_pair(hk, hq);
    const float delta = (__bfloat162float(v[t][warp]) - dots.x) * beta[t];
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[i] += kv[i] * delta;
    }
    if (lane == 0) {
      __nv_bfloat16* leader = cluster.map_shared_rank(&raw[0][0], 0);
      // Preserve BF16 rounding between the recurrence and RMS normalization.
      leader[t * kDimV + split * kRows + warp] =
          __float2bfloat16(dots.y + delta * kq[t]);
    }
    const int destination_slot = p.indices[t];
    if (destination_slot > 0) {
      __nv_bfloat16* destination =
          p.state +
          static_cast<int64_t>(destination_slot) * p.strides.state_slot +
          head * kDimV * kDimK + (split * kRows + warp) * kDimK + lane * 4;
      store_state4_vectorized(destination, make_float4(h[0], h[1], h[2], h[3]));
    }
  }
  // Complete all remote writes before the leader reads raw. After this
  // barrier it accesses only its own shared memory, so other CTAs may exit.
  cluster.sync();
  if (split != 0) {
    return;
  }
  if (warp < num_tokens) {
    const int t = warp;
    float x[4], ss = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[i] = __bfloat162float(raw[t][lane + i * 32]);
      ss += x[i] * x[i];
    }
    const float rstd = rsqrtf(warp_reduce_sum(ss) / kDimV + 1e-6f);
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      const float gate = __bfloat162float(
          p.gate[(bos + t) * p.strides.gate_row + head * kDimV + d]);
      store_gdn_output<FuseFp8>(
          p.out, p.strides, ((bos + t) * 16 + head) * kDimV + d,
          __float2bfloat16(x[i] * rstd * __bfloat162float(p.norm_weight[d]) *
                           silu_fast(gate)));
    }
  }
#endif
}

// A 16-CTA cluster is nonportable: query occupancy once per device, including
// restricted partitions, and retain the original kernel if it cannot launch.
template <bool EnablePdl, bool FuseFp8 = false>
bool gdn_supports_cluster16(int device_index, cudaStream_t stream) {
  struct Support {
    std::once_flag flag;
    bool available = false;
  };
  static std::deque<Support> support(device_properties.size());
  auto& device = support[device_index];
  std::call_once(device.flag, [&] {
    cudaError_t error =
        cudaFuncSetAttribute(gdn_decode_mtp_cluster_kernel<EnablePdl, FuseFp8>,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    STD_TORCH_CHECK(error == cudaSuccess, "GDN cluster configuration failed: ",
                    cudaGetErrorString(error));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(256);
    config.blockDim = dim3(256);
    config.stream = stream;
    int max_cluster_size = 0;
    error = cudaOccupancyMaxPotentialClusterSize(
        &max_cluster_size, gdn_decode_mtp_cluster_kernel<EnablePdl, FuseFp8>,
        &config);
    STD_TORCH_CHECK(
        error == cudaSuccess,
        "GDN cluster occupancy query failed: ", cudaGetErrorString(error));
    device.available = max_cluster_size >= 16;
  });
  return device.available;
}

template <bool EnablePdl, bool FuseFp8 = false>
void launch_gdn_decode_mtp_cluster(GdnClusterParams params,
                                   cudaStream_t stream) {
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim = {16, 1, 1};
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = EnablePdl;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(256);
  config.blockDim = dim3(256);
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;
  const cudaError_t error = cudaLaunchKernelEx(
      &config, gdn_decode_mtp_cluster_kernel<EnablePdl, FuseFp8>, params);
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "GDN decode MTP cluster launch failed: ", cudaGetErrorString(error));
}
