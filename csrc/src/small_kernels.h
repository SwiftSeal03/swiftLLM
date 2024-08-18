# pragma once

#include <torch/extension.h>
#include <cstdint>
#include <cuda_fp16.h>

void fused_add_rmsnorm_inplace(
  torch::Tensor buffer,
  torch::Tensor residual,
  torch::Tensor weight,
  const float epsilon
);

void silu_and_mul_inplace(
  torch::Tensor buffer
);

void rotary_embedding_inplace(
  torch::Tensor q,
  torch::Tensor k,
  torch::Tensor sin_table,
  torch::Tensor cos_table
);