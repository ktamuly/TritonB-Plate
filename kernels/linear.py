import triton
import triton.language as tl
import torch
from .matmul import triton_matmul


@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B,
    N,
    in_features,
    out_features,
    stride_xb,
    stride_xn,
    stride_xh,
    stride_wo,
    stride_wi,
    stride_ob,
    stride_on,
    stride_oh,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    """Linear layer kernel: out = x @ weight.T + bias."""
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_o = tl.program_id(2)
    b_start = pid_batch * BLOCK_B
    n_start = pid_n * BLOCK_N
    o_start = pid_o * BLOCK_O
    b_offsets = b_start + tl.arange(0, BLOCK_B)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    o_offsets = o_start + tl.arange(0, BLOCK_O)
    b_mask = b_offsets < B
    n_mask = n_offsets < N
    o_mask = o_offsets < out_features
    acc = tl.zeros((BLOCK_B, BLOCK_N, BLOCK_O), dtype=tl.float32)
    for i_idx in range(0, in_features, 32):
        i_block = i_idx + tl.arange(0, 32)
        i_mask = i_block < in_features
        x_ptrs = (
            x_ptr
            + b_offsets[:, None, None] * stride_xb
            + n_offsets[None, :, None] * stride_xn
            + i_block[None, None, :] * stride_xh
        )
        x = tl.load(
            x_ptrs,
            mask=b_mask[:, None, None] & n_mask[None, :, None] & i_mask[None, None, :],
            other=0.0,
        )
        w_ptrs = (
            weight_ptr + o_offsets[:, None] * stride_wo + i_block[None, :] * stride_wi
        )
        w = tl.load(w_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        acc += tl.dot(x, w)
    bias = tl.load(bias_ptr + o_offsets, mask=o_mask, other=0.0)
    result = acc + bias
    out_ptrs = (
        out_ptr
        + b_offsets[:, None, None] * stride_ob
        + n_offsets[None, :, None] * stride_on
        + o_offsets[None, None, :] * stride_oh
    )
    tl.store(
        out_ptrs,
        result,
        mask=b_mask[:, None, None] & n_mask[None, :, None] & o_mask[None, None, :],
    )


def triton_linear(x, weight, bias=None):
    """Linear layer using Triton."""
    B, N, in_features = x.shape
    out_features, in_features_w = weight.shape
    assert (
        in_features == in_features_w
    ), f"Input features {in_features} doesn't match weight shape {in_features_w}"
    x_reshaped = x.view(-1, in_features)
    out = triton_matmul(x_reshaped, weight.T)
    if bias is not None:
        out += bias
    out = out.view(B, N, out_features)
    return out
