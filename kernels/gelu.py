import triton
import triton.language as tl
import torch


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU activation kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid_input = 1.702 * x
    sigmoid = 1.0 / (1.0 + tl.exp(-sigmoid_input))
    result = x * sigmoid
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_gelu(x):
    """GELU activation using Triton."""
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](
        x.reshape(-1), output.reshape(-1), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return output
