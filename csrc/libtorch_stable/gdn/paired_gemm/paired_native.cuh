/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu
 * which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
 *
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Local experiment: native FP8 mma.sync, asynchronous FP8 copies, explicit PDL
// and CTA split-K.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#ifndef GDN_TOKEN_TILE
  #define GDN_TOKEN_TILE 8
#endif
namespace paired_native {
using bf16_t = __nv_bfloat16;
using fp8_t = __nv_fp8_e4m3;

struct ProjectionScales {
  const float* input;
  const float* weight;
  __device__ float operator[](int index) const {
    return index == 0 ? *input : *weight;
  }
};

__device__ void mma_16_8_32_f32acc_fp8ab(float (&d_reg)[4],
                                         const fp8_t (&a_reg)[16],
                                         const fp8_t (&b_reg)[8],
                                         float const (&c_reg)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a0 = *reinterpret_cast<uint32_t const*>(a_reg + 0);
  uint32_t a1 = *reinterpret_cast<uint32_t const*>(a_reg + 4);
  uint32_t a2 = *reinterpret_cast<uint32_t const*>(a_reg + 8);
  uint32_t a3 = *reinterpret_cast<uint32_t const*>(a_reg + 12);
  uint32_t b0 = *reinterpret_cast<uint32_t const*>(b_reg + 0);
  uint32_t b1 = *reinterpret_cast<uint32_t const*>(b_reg + 4);
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d_reg[0]),
        "f"(d_reg[1]), "f"(d_reg[2]), "f"(d_reg[3]));
#endif
}

extern "C" {
__device__ uint32_t __nvvm_get_smem_pointer(void*);
}

__device__ void ldgsts_128(void const* gPtr, void* sPtr, uint32_t pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pred) {
    uint32_t smemPtrAsUint32 = __nvvm_get_smem_pointer(sPtr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                     smemPtrAsUint32),
                 "l"(gPtr), "n"(16));
  }
#endif
}

__device__ void ldsm_x4(void* smem_ptr, uint32_t* reg_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
      : "r"(__nvvm_get_smem_pointer(smem_ptr)));
#endif
}

template <class Type>
__device__ int apply_swizzle_343_on_elem_row_col(int row_idx_, int col_idx_) {
  uint32_t row_idx = *reinterpret_cast<uint32_t*>(&row_idx_);
  uint32_t col_idx = *reinterpret_cast<uint32_t*>(&col_idx_);
  row_idx = row_idx % 8;
  row_idx = row_idx * (16 / sizeof(Type));
  col_idx = col_idx ^ row_idx;
  return *reinterpret_cast<int*>(&col_idx);
}

__device__ void initialize_barrier(
    uint64_t* smem_barrier,  // 64 bits user-manged barrier in smem
    int thread_count =
        1)  // Thread count expected to arrive/wait on this barrier
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
#endif
}

// Barrier wait
__device__ void wait_barrier(
    uint64_t* smem_barrier,  // 64 bits user-manged barrier in smem
    int phase_bit)           // Current phase bit the barrier waiting to flip
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra DONE;\n"
      "bra                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
#endif
}

__device__ bool try_wait_barrier(uint64_t* smem_ptr, int phase_bit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t wait_complete;
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(wait_complete)
      : "r"(smem_int_ptr), "r"(phase_bit));
  return static_cast<bool>(wait_complete);
#endif
  return false;
}

// Barrier arrive
__device__ void arrive_barrier(
    uint64_t* smem_barrier)  // 64 bits user-manged barrier in smem
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
      "}\n" ::"r"(smem_int_ptr));
#endif
}

__device__ void ldgsts_arrive(uint64_t* smem_barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"
               :
               : "r"(smem_int_ptr));
#endif
}

template <int gemm_k, int tile_m, int tile_k, int stage_cnt, int k_splits>
struct GmemLoaderA {
  static constexpr int elem_bytes = 1;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_m * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int a_inst_cnt_per_iter =
      (tile_m * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  // Extra params to keep the order of k reduction...
  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) +
           (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderA(fp8_t const* gmem_a_local_, fp8_t* smem_a_,
                         uint64_t* smem_barrier_)
      : gmem_a(gmem_a_local_),
        smem_a(smem_a_),
        smem_barrier(smem_barrier_),
        local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // swizzle, that's what we want.
  #pragma unroll
    for (int i = 0; i < a_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int m_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<fp8_t>(m_idx, k_idx);
      a_smem_offsets[i] = m_idx * tile_k + k_idx;
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit =
          next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2,
                                      next_phase_bit);
      }

  #pragma unroll
      for (int i = 0; i < a_inst_cnt_per_iter; i++) {
        int smem_offset = a_smem_offsets[i];
        fp8_t* smem_ptr_this_iter =
            smem_a + stage_idx * tile_m * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int m_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = m_idx * gemm_k * k_splits + k_project(k_idx);
        fp8_t const* gmem_ptr_this_iter = gmem_a + gmem_offset;
        ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, true);
      }
      ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_a += per_mma_warp_k;
    }
#endif
  }

  fp8_t const* gmem_a;
  fp8_t* smem_a;
  uint64_t* smem_barrier;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  // per smem_stage, store with swizzle information
  int a_smem_offsets[a_inst_cnt_per_iter];
};

template <int gemm_k, int tile_n, int tile_k, int stage_cnt, int k_splits,
          int QuantMode>
struct GmemLoaderB {
  using InputT = std::conditional_t<(QuantMode != 0), bf16_t, fp8_t>;
  static constexpr int elem_bytes = 1;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_n * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int b_inst_cnt_per_iter =
      (tile_n * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  // Extra params to keep the order of k reduction...
  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) +
           (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderB(InputT const* gmem_b_local_, fp8_t* smem_b_,
                         uint64_t* smem_barrier_, int gemm_n_,
                         ProjectionScales scales)
      : gmem_b(gmem_b_local_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        gemm_n(gemm_n_),
        input_scale(scales[0]),
        input_inverse(1.f / scales[0]),
        local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // swizzle, that's what we want.
  #pragma unroll
    for (int i = 0; i < b_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int n_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<fp8_t>(n_idx, k_idx);
      b_smem_offsets[i] = n_idx * tile_k + k_idx;
      preds[i] = n_idx < gemm_n;
    }
#endif
  }

  __device__ void issue_mainloop(int pdl_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl_mode == 2) cudaGridDependencySynchronize();
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit =
          next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2,
                                      next_phase_bit);
      }
  #pragma unroll
      for (int i = 0; i < b_inst_cnt_per_iter; i++) {
        int smem_offset = b_smem_offsets[i];
        fp8_t* smem_ptr_this_iter =
            smem_b + stage_idx * tile_n * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int n_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = n_idx * gemm_k * k_splits + k_project(k_idx);
        InputT const* gmem_ptr_this_iter = gmem_b + gmem_offset;
        if constexpr (QuantMode) {
          // A BF16 input vector is loaded once by the activation loader,
          // quantized for this projection, then shared by all four compute
          // warps.
          if (preds[i]) {
            uint4 source[2];
            source[0] = reinterpret_cast<const uint4*>(gmem_ptr_this_iter)[0];
            source[1] = reinterpret_cast<const uint4*>(gmem_ptr_this_iter)[1];
            uint4 result;
            auto values = reinterpret_cast<const bf16_t*>(source);
            auto quantized = reinterpret_cast<fp8_t*>(&result);
            if constexpr (QuantMode == 2) {
  #pragma unroll
              for (int v = 0; v < 8; ++v) {
                float2 pair = __bfloat1622float2(
                    reinterpret_cast<const __nv_bfloat162*>(values)[v]);
                pair.x *= input_inverse;
                pair.y *= input_inverse;
                reinterpret_cast<__nv_fp8x2_storage_t*>(&result)[v] =
                    __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
              }
            } else {
  #pragma unroll
              for (int v = 0; v < 16; ++v)
                quantized[v] = fp8_t(fminf(
                    448.f,
                    fmaxf(-448.f, __bfloat162float(values[v]) / input_scale)));
            }
            *reinterpret_cast<uint4*>(smem_ptr_this_iter) = result;
          }
        } else {
          ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, preds[i]);
        }
      }
      if constexpr (QuantMode)
        arrive_barrier(smem_barrier + stage_idx * 2);
      else
        ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_b += per_mma_warp_k;
    }
#endif
  }

  InputT const* gmem_b;
  fp8_t* smem_b;
  uint64_t* smem_barrier;
  int gemm_n;
  float input_scale;
  float input_inverse;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  // per smem_stage, store with swizzle information
  int b_smem_offsets[b_inst_cnt_per_iter];
  uint32_t preds[b_inst_cnt_per_iter];
};

template <int gemm_m, int gemm_k, int tile_m, int tile_n, int tile_k,
          int stage_cnt, typename OutT>
struct MmaComputer {
  static constexpr int elem_bytes = 1;
  static constexpr int thread_cnt = 128;
  static_assert(gemm_k % tile_k == 0);
  static_assert(tile_k % (thread_cnt / 32) == 0);
  static constexpr int per_warp_tile_k = tile_k / (thread_cnt / 32);
  static constexpr int k_iter_cnt = gemm_k / tile_k;
  static constexpr int k_phase_cnt = per_warp_tile_k / 32;
  static constexpr int m_iter_cnt = (tile_m + 15) / 16;
  static constexpr int n_iter_cnt =
      (tile_n + 7) /
      8;  // Possible to have non-1 n_iter_cnt for ab_swap m16 case.
  static_assert(m_iter_cnt == 1 || m_iter_cnt == 2);
  static_assert(n_iter_cnt >= 1 && n_iter_cnt <= 8);

  __device__ MmaComputer(OutT* gmem_c_local_, fp8_t* smem_a_, fp8_t* smem_b_,
                         uint64_t* smem_barrier_, int warp_idx_, int gemm_n_,
                         ProjectionScales scales_, int output_stride_)
      : gmem_c(gmem_c_local_),
        smem_a(smem_a_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        warp_idx(warp_idx_ - (thread_cnt / 32)),
        gemm_n(gemm_n_),
        scales(scales_),
        output_stride(output_stride_) {}

 private:
  __device__ constexpr int internal_b_atom_func(int tid) {
    if constexpr (tile_n < 8) {
      return (tid % tile_n) + ((tid % 8) / tile_n * 0) + tid / 8 * 8 * tile_n;
    } else {
      return (tid % 8) + ((tid % 32) / 8 * (tile_n * 8));
    }
  }

 public:
  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll
    for (int m = 0; m < m_iter_cnt; m++) {
  #pragma unroll
      for (int i = 0; i < k_phase_cnt; i++) {
        int linear_idx = (lane_idx % 16) + (lane_idx / 16) * 128 + i * 256;
        int m_idx = linear_idx % 16 + m * 16;
        int k_idx = 2 * (linear_idx / 16) + warp_k_offset_in_tile_k;
        k_idx = apply_swizzle_343_on_elem_row_col<fp8_t>(m_idx, k_idx);
        a_smem_offsets[m][i] = m_idx * tile_k + k_idx;
      }
    }
  #pragma unroll
    for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
  #pragma unroll
      for (int i = 0; i < k_phase_cnt; i += 2) {  // Special i+=2 for B.
        int linear_idx =
            internal_b_atom_func(lane_idx) + i * tile_n * 16 + n_iter_idx * 8;
        int n_idx = linear_idx % tile_n;
        int k_idx = 2 * (linear_idx / tile_n) + warp_k_offset_in_tile_k;
        k_idx = apply_swizzle_343_on_elem_row_col<fp8_t>(n_idx, k_idx);
        b_smem_offsets[n_iter_idx][i] = n_idx * tile_k + k_idx;
      }
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      wait_barrier(smem_barrier + 0 + stage_idx * 2, phase_bit);

  #pragma unroll
      for (int m = 0; m < m_iter_cnt; m++) {
  #pragma unroll
        for (int i = 0; i < k_phase_cnt; i++) {
          int smem_offset = a_smem_offsets[m][i];
          fp8_t* smem_ptr_this_iter =
              smem_a + stage_idx * tile_m * tile_k + smem_offset;
          ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(a_reg[m][i]));
        }
      }

  #pragma unroll
      for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
  #pragma unroll
        for (int i = 0; i < k_phase_cnt; i += 2) {
          int smem_offset = b_smem_offsets[n_iter_idx][i];
          fp8_t* smem_ptr_this_iter =
              smem_b + stage_idx * tile_n * tile_k + smem_offset;
          ldsm_x4(smem_ptr_this_iter,
                  reinterpret_cast<uint32_t*>(b_reg[n_iter_idx][i]));
        }
      }

  #pragma unroll
      for (int k_iter_idx = 0; k_iter_idx < k_phase_cnt; k_iter_idx++) {
  #pragma unroll
        for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
  #pragma unroll
          for (int m = 0; m < m_iter_cnt; m++) {
            mma_16_8_32_f32acc_fp8ab(
                acc_reg[m][n_iter_idx], a_reg[m][k_iter_idx],
                b_reg[n_iter_idx][k_iter_idx], acc_reg[m][n_iter_idx]);
          }
        }
      }
      arrive_barrier(smem_barrier + 1 + stage_idx * 2);
      stage_idx += 1;
      phase_bit = stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      stage_idx = stage_idx == stage_cnt ? 0 : stage_idx;
    }
#endif
  }

  __device__ void epi() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));
    // reorganize the acc_reg
    constexpr int thread_m = 2 * m_iter_cnt;
    constexpr int thread_n = 2 * n_iter_cnt;
    constexpr int cta_mma_n = n_iter_cnt * 8;
    float acc_reg_reorg[thread_m][thread_n];

    for (int i = 0; i < thread_m; i++) {
      for (int j = 0; j < thread_n; j++) {
        acc_reg_reorg[i][j] = acc_reg[i / 2][j / 2][(j % 2) + (i % 2) * 2];
      }
    }

    // 4 x cosize(smem_c_layout)
    float* smem_c = reinterpret_cast<float*>(smem_a);
    // coord -> index
    auto smem_c_index_func = [&](int m_idx, int n_idx) {
      constexpr int group_rows = cta_mma_n <= 32 ? 32 / cta_mma_n : 1;
      constexpr int group_width = cta_mma_n <= 32 ? 32 : cta_mma_n;
      int group_cnt = 2;
      return (m_idx % group_rows * cta_mma_n) +
             (m_idx / group_rows * (group_width + group_cnt)) + n_idx;
    };
    constexpr int group_rows = cta_mma_n <= 32 ? 32 / cta_mma_n : 1;
    constexpr int group_width = cta_mma_n <= 32 ? 32 : cta_mma_n;
    constexpr int cosize_smem_c = (tile_m / group_rows) * (group_width + 2);

  // This should be optimized to STS.64 but can not be STS.128 due to the bank
  // index.
  #pragma unroll
    for (int m_idx_thread = 0; m_idx_thread < thread_m; m_idx_thread++) {
  #pragma unroll
      for (int n_idx_thread = 0; n_idx_thread < thread_n; n_idx_thread++) {
        int m_idx =
            (lane_idx / 4) + (m_idx_thread % 2) * 8 + (m_idx_thread / 2) * 16;
        int n_idx =
            ((lane_idx % 4) * 2) + (n_idx_thread % 2) + (n_idx_thread / 2) * 8;
        smem_c[cosize_smem_c * warp_idx + smem_c_index_func(m_idx, n_idx)] =
            acc_reg_reorg[m_idx_thread][n_idx_thread];
      }
    }
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));

    if (warp_idx == 0) {
      constexpr int final_acc_reg_cnt = (tile_m * tile_n + 31) / 32;
      float acc_final[final_acc_reg_cnt]{};

  #pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        acc_final[reg_idx] +=
            smem_c[smem_c_index_func(m_idx, n_idx) + 0 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 1 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 2 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 3 * cosize_smem_c];
      }

  #pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        if (m_idx < tile_m && n_idx < gemm_n) {
          if constexpr (std::is_same_v<OutT, float>)
            gmem_c[n_idx * output_stride + m_idx] = acc_final[reg_idx];
          else
            gmem_c[n_idx * output_stride + m_idx] =
                __float2bfloat16(acc_final[reg_idx] * scales[0] * scales[1]);
        }
      }
    }
#endif
  }

  int output_stride;
  OutT* gmem_c;
  ProjectionScales scales;
  fp8_t* smem_a;
  fp8_t* smem_b;
  uint64_t* smem_barrier;
  int warp_idx;
  int gemm_n;
  int stage_idx = 0;
  int phase_bit = 0;
  int lane_idx = threadIdx.x % 32;
  int warp_k_offset_in_tile_k = warp_idx * per_warp_tile_k;

  int a_smem_offsets[m_iter_cnt][k_phase_cnt];
  int b_smem_offsets[n_iter_cnt][k_phase_cnt];

  fp8_t a_reg[m_iter_cnt][k_phase_cnt][16];
  fp8_t b_reg[n_iter_cnt][k_phase_cnt][8];
  float acc_reg[m_iter_cnt][n_iter_cnt][4]{};
};

// Each block owns either QKVZ or BA output columns. Separate weight and
// activation loader warps feed four tensor-core compute warps. No cross terms.
template <int TM, int TT, int TK, int ST, int S, int QuantMode>
__global__ __launch_bounds__(256, 1) void paired_gemm(
    const fp8_t* qkv_x, const fp8_t* ba_x, const bf16_t* x, const fp8_t* qkv_w,
    const fp8_t* ba_w, bf16_t* qkv_y, bf16_t* ba_y, float* temp,
    const float* input_scale, const float* qkv_scale, const float* ba_scale,
    int mode, int rows) {
  if (mode == 1) cudaGridDependencySynchronize();
  constexpr int K = 8192 / S;
  using OutT = std::conditional_t<S == 1, bf16_t, float>;
  using InputT = std::conditional_t<(QuantMode != 0), bf16_t, fp8_t>;
  const int token = blockIdx.y * TT;
  const int valid_tokens = min(TT, rows - token);
  const bool ba = blockIdx.x >= 4608 / TM;
  int column = (ba ? blockIdx.x - 4608 / TM : blockIdx.x) * TM;
  int split_offset = blockIdx.z * K;
  const fp8_t* weight = (ba ? ba_w : qkv_w) + column * 8192 + split_offset;
  const InputT* activation;
  if constexpr (QuantMode)
    activation = x + token * 8192 + split_offset;
  else
    activation = (ba ? ba_x : qkv_x) + token * 8192 + split_offset;
  ProjectionScales local_scales{input_scale, ba ? ba_scale : qkv_scale};
  OutT* output;
  int output_stride;
  if constexpr (S == 1) {
    output = (ba ? ba_y : qkv_y) + token * (ba ? 32 : 4608) + column;
    output_stride = ba ? 32 : 4608;
  } else {
    output =
        temp + (blockIdx.z * rows + token) * 4640 + (ba ? 4608 : 0) + column;
    output_stride = 4640;
  }
  extern __shared__ char smem[];
  auto barrier = reinterpret_cast<uint64_t*>(smem);
  auto wa = reinterpret_cast<fp8_t*>(smem + 1024);
  auto ax = wa + TM * TK * ST;
  int warp = threadIdx.x / 32;
  if (warp == 4) {
    for (int i = 0; i < ST; ++i) {
      initialize_barrier(barrier + i * 2, 128);
      initialize_barrier(barrier + i * 2 + 1, 128);
    }
  }
  __syncthreads();
  if (warp < 2) {
    GmemLoaderA<K, TM, TK, ST, S> loader(weight, wa, barrier);
    loader.prepare();
    loader.issue_mainloop();
  } else if (warp < 4) {
    GmemLoaderB<K, TT, TK, ST, S, QuantMode> loader(activation, ax, barrier,
                                                    valid_tokens, local_scales);
    loader.prepare();
    loader.issue_mainloop(mode);
  } else {
    MmaComputer<4640, K, TM, TT, TK, ST, OutT> compute(
        output, wa, ax, barrier, warp, valid_tokens, local_scales,
        output_stride);
    compute.prepare();
    compute.issue_mainloop();
    compute.epi();
  }
  if (mode) cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace paired_native
