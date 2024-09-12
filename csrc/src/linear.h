#pragma once

#include <torch/extension.h>

torch::Tensor linear(
  torch::Tensor a,
  torch::Tensor w
);