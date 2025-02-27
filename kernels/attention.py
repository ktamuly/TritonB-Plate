import triton
import triton.language as tl
import torch
from .softmax import triton_softmax
from .matmul import triton_matmul

@triton.jit
def scaled_dot_product_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                                          B, H, N, d,
                                          scale,
                                          stride_qb, stride_qh, stride_qn, stride_qd,
                                          stride_kb, stride_kh, stride_kn, stride_kd,
                                          stride_vb, stride_vh, stride_vn, stride_vd,
                                          stride_ob, stride_oh, stride_on, stride_od,
                                          BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_SEQLEN: tl.constexpr):
    """Compute attention scores and weighted values."""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_n = tl.program_id(2)
    q_start = q_ptr + pid_batch * stride_qb + pid_head * stride_qh
    k_start = k_ptr + pid_batch * stride_kb + pid_head * stride_kh
    v_start = v_ptr + pid_batch * stride_vb + pid_head * stride_vh
    out_start = out_ptr + pid_batch * stride_ob + pid_head * stride_oh
    block_n_start = pid_n * BLOCK_N
    n_offsets = block_n_start + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_D)
    q = tl.load(q_start + n_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd,
                mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < d), other=0.0)
    output = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for k_idx in range(0, N, BLOCK_SEQLEN):
        k_offsets = k_idx + tl.arange(0, BLOCK_SEQLEN)
        mask = (k_offsets[:, None] < N) & (d_offsets[None, :] < d)
        k = tl.load(k_start + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd,
                    mask=mask, other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale
        scores_max = tl.max(scores, axis=1)[:, None]
        scores_exp = tl.exp(scores - scores_max)
        v = tl.load(v_start + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd,
                    mask=mask, other=0.0)
        output += tl.dot(scores_exp, v)
    tl.store(out_start + n_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od,
             output, mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < d))

def triton_attention(q, k, v, scale=None):
    """
    Multi-head attention using Triton.
    
    Args:
        q: Query tensor [B, H, N, d]
        k: Key tensor [B, H, N, d]
        v: Value tensor [B, H, N, d]
        scale: Scaling factor (default: 1/sqrt(d))
        
    Returns:
        Attention output tensor [B, H, N, d]
    """
    B, H, N, d = q.shape
    assert k.shape == (B, H, N, d)
    assert v.shape == (B, H, N, d)
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    output = torch.empty_like(q)
    stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = v.stride()
    stride_ob, stride_oh, stride_on, stride_od = output.stride()
    BLOCK_N = min(triton.next_power_of_2(N), 32)
    BLOCK_D = min(triton.next_power_of_2(d), 32)
    BLOCK_SEQLEN = min(triton.next_power_of_2(N), 64)
    grid = (B, H, triton.cdiv(N, BLOCK_N))
    scaled_dot_product_attention_kernel[grid](
        q, k, v, output,
        B, H, N, d,
        scale,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_SEQLEN=BLOCK_SEQLEN,
    )
    return output