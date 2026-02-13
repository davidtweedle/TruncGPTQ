#pragma once
#include <torch/extension.h>

void gptq_fused_cuda(
        torch::Tensor W,
        torch::Tensor Err,
        torch::Tensor H,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        int col_offset,
        float qmin,
        float qmax
        );
