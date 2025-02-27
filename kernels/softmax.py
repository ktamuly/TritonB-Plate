import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(x_ptr, out_ptr,
                   stride_b, stride_h, stride_s,
                   B, H, S,
                   BLOCK_SIZE: tl.constexpr):
    """Softmax kernel for attention scores."""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    offset = pid_batch * stride_b + pid_head * stride_h + pid_seq * stride_s
    s_offsets = tl.arange(0, BLOCK_SIZE)
    mask = s_offsets < S
    x = tl.load(x_ptr + offset + s_offsets, mask=mask, other=float("-inf"))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum
    tl.store(out_ptr + offset + s_offsets, result, mask=mask)

def triton_softmax(x, dim=-1):
    """Softmax function using Triton."""
    if dim < 0:
        dim = x.dim() + dim
    output = torch.empty_like(x)
    if dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(*perm).contiguous()
        output = output.permute(*perm).contiguous()
    shape = x.shape
    S = shape[-1]
    if len(shape) == 1:
        B, H = 1, 1
        stride_b = stride_h = 0
        stride_s = x.stride(0)
    elif len(shape) == 2:
        B, H = shape[0], 1
        stride_b = x.stride(0)
        stride_h = 0
        stride_s = x.stride(1)
    elif len(shape) == 3:
        B, H, S = shape
        stride_b, stride_h, stride_s = x.stride()
    else:
        B = shape[0]
        H = shape[1]
        S = shape[2]
        for d in shape[3:]:
            S *= d
        x = x.reshape(B, H, S)
        output = output.reshape(B, H, S)
        stride_b, stride_h, stride_s = x.stride()
    BLOCK_SIZE = min(triton.next_power_of_2(S), 1024)
    grid = (B, H, 1)
    softmax_kernel[grid](x, output, stride_b, stride_h, stride_s, B, H, S, BLOCK_SIZE=BLOCK_SIZE)
    if dim != x.dim() - 1:
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        output = output.permute(*inv_perm).contiguous()
    return output 