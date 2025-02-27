import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr,
                     stride_b, stride_n, stride_h,
                     N, H, eps,
                     BLOCK_SIZE: tl.constexpr):
    """Layer normalization kernel."""
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    x_offset = pid_batch * stride_b + pid_n * stride_n
    h_offsets = tl.arange(0, BLOCK_SIZE)
    mask = h_offsets < H
    x = tl.load(x_ptr + x_offset + h_offsets * stride_h, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / H
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / H
    rstd = 1 / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + h_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + h_offsets, mask=mask, other=0.0)
    y = x_centered * rstd * gamma + beta
    tl.store(out_ptr + x_offset + h_offsets * stride_h, y, mask=mask)

def triton_layernorm(x, gamma, beta, eps=1e-5):
    """Layer normalization using Triton."""
    B, N, H = x.shape
    output = torch.empty_like(x)
    stride_b, stride_n, stride_h = x.stride()
    BLOCK_SIZE = min(triton.next_power_of_2(H), 1024)
    grid = (B, N)
    layernorm_kernel[grid](x, gamma, beta, output,
                           stride_b, stride_n, stride_h,
                           N, H, eps,
                           BLOCK_SIZE=BLOCK_SIZE)
    return output 