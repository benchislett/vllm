// SPDX-License-Identifier: Apache-2.0
#include "paired_native.cuh"
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

template <int Tokens>
void launch(const Tensor& x, const Tensor& qkvw, const Tensor& baw,
            const Tensor& xs, const Tensor& qs, const Tensor& bs, Tensor& qkv,
            Tensor& ba, int mode, cudaStream_t stream) {
  constexpr int Shared = 1024 + (32 + Tokens) * 512 * 8;
  auto fn = paired_native::paired_gemm<32, Tokens, 512, 8, 1, 0>;
  auto error = cudaFuncSetAttribute(
      fn, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared);
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(145, (x.size(0) + Tokens - 1) / Tokens, 1);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = Shared;
  cfg.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = mode != 0;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  using paired_native::bf16_t;
  using paired_native::fp8_t;
  error = cudaLaunchKernelEx(
      &cfg, fn, static_cast<const fp8_t*>(x.data_ptr()),
      static_cast<const fp8_t*>(x.data_ptr()),
      static_cast<const bf16_t*>(nullptr),
      static_cast<const fp8_t*>(qkvw.data_ptr()),
      static_cast<const fp8_t*>(baw.data_ptr()),
      static_cast<bf16_t*>(qkv.data_ptr()), static_cast<bf16_t*>(ba.data_ptr()),
      static_cast<float*>(nullptr), static_cast<const float*>(xs.data_ptr()),
      static_cast<const float*>(qs.data_ptr()),
      static_cast<const float*>(bs.data_ptr()), mode,
      static_cast<int>(x.size(0)));
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

void gemms_out(const Tensor& x, const Tensor& qkvw, const Tensor& baw,
               const Tensor& xs, const Tensor& qs, const Tensor& bs,
               Tensor& qkv, Tensor& ba, int64_t pdl_mode) {
  STD_TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == ScalarType::Float8_e4m3fn &&
          x.dim() == 2 && x.size(0) > 0 && x.size(0) <= 16 &&
          x.size(1) == 8192 && x.is_contiguous(),
      "Custom GDN GEMM requires contiguous FP8 [1..16, 8192] input");
  const auto device = x.get_device_index();
  for (const auto* tensor : std::initializer_list<const Tensor*>{
           &qkvw, &baw, &xs, &qs, &bs, &qkv, &ba}) {
    STD_TORCH_CHECK(tensor->is_cuda() && tensor->get_device_index() == device,
                    "All custom GDN GEMM tensors must share the CUDA device");
  }
  for (const auto* weight : {&qkvw, &baw}) {
    const int n = weight == &qkvw ? 4608 : 32;
    STD_TORCH_CHECK(
        weight->scalar_type() == ScalarType::Float8_e4m3fn &&
            weight->dim() == 2 && weight->size(0) == 8192 &&
            weight->size(1) == n && weight->stride(0) == 1 &&
            weight->stride(1) == 8192 &&
            reinterpret_cast<uintptr_t>(weight->data_ptr()) % 16 == 0,
        "Custom GDN weights must be column-major FP8 [8192, 4608/32]");
  }
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                  "Custom GDN input must be 16-byte aligned");
  for (const auto* scale : {&xs, &qs, &bs}) {
    STD_TORCH_CHECK(
        scale->scalar_type() == ScalarType::Float && scale->numel() == 1,
        "Custom GDN scales must be scalar float32 tensors");
  }
  for (const auto* output : {&qkv, &ba}) {
    const int n = output == &qkv ? 4608 : 32;
    STD_TORCH_CHECK(output->scalar_type() == ScalarType::BFloat16 &&
                        output->dim() == 2 && output->size(0) == x.size(0) &&
                        output->size(1) == n && output->is_contiguous(),
                    "Custom GDN outputs must be contiguous BF16 [M, 4608/32]");
  }
  STD_TORCH_CHECK(pdl_mode >= 0 && pdl_mode <= 2, "PDL mode must be 0, 1 or 2");
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream));
  if (x.size(0) <= 8) {
    launch<8>(x, qkvw, baw, xs, qs, bs, qkv, ba, pdl_mode,
              static_cast<cudaStream_t>(stream));
  } else {
    launch<16>(x, qkvw, baw, xs, qs, bs, qkv, ba, pdl_mode,
               static_cast<cudaStream_t>(stream));
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(_qwen_gdn_custom, ops) {
  ops.def(
      "gemms_out(Tensor x, Tensor qkvw, Tensor baw, Tensor xs, Tensor qs, "
      "Tensor bs, Tensor! qkv, Tensor! ba, int pdl_mode=2) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_qwen_gdn_custom, CUDA, ops) {
  ops.impl("gemms_out", TORCH_BOX(&gemms_out));
}
