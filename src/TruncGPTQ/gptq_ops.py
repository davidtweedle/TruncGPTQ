import Torch

try:
    from . import _C
except ImportError:
    _C = None
    print("Warning: CUDA extension not found. Please install with 'pip install .'")

def fused_gptq_step(W, H, Scales, Zeros, col_offset, qmin=-7.0, qmax=7.0):
    if _C is None:
        raise RuntimeError("CUDA extension not compiled.")

    M, _ = W.shape
    block_size = 64

    if not W.is_contiguous(): W = W.contiguous()
    if not H.is_contiguous(): H = H.contiguous()

    Err = torch.empty((M, block_size), device=W.device, dtype=W.dtype)

    _C.gptq_fused(W, Err, H, Scales, Zeros, col_offset, qmin, qmax)

    return W
