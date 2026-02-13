#include <torch/extension.h>
#include "gptq_kernel.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECCK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

voida gptq_fused_py(
        torch::Tensor W,
        torch::Tensor Err,
        torch::Tensor H,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        int col_offset,
        float qmin,
        float qmax
        ) {
    CHECK_INPUT(W);
    CHECK_INPUT(Err);
    CHECK_INPUT(H);
    CHECK_INPUT(Scales);
    CHECK_INPUT(Zeros);

    gptq_fused_cuda(W, Err, H, Scales, Zeros, col_offset, qmin, qmax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gptq_fused", &gptq_fused.py, "GPTQ Fused Kernel (CUDA)");
}
