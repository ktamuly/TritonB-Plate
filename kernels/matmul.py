import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Matrix multiplication kernel: C = A @ B"""
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Offsets for A and B matrices
    offs_am = m_start + tl.arange(0, BLOCK_M)
    offs_bn = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B
        a_block_ptr = a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_block_ptr = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        
        # Create masks for valid indices
        a_mask = (offs_am[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
        
        # Load blocks with masks
        a = tl.load(a_block_ptr, mask=a_mask, other=0.0)
        b = tl.load(b_block_ptr, mask=b_mask, other=0.0)
        
        # Compute and accumulate matrix product
        acc += tl.dot(a, b)
    
    # Store output with mask for valid indices
    c_block_ptr = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_block_ptr, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def triton_matmul(a, b):
    """Wrapper for matmul kernel that handles dimensions and launches the kernel"""
    # Extract dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} and {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    BLOCK_M = min(triton.next_power_of_2(M), 64)  
    BLOCK_N = min(triton.next_power_of_2(N), 64)  
    BLOCK_K = min(triton.next_power_of_2(K), 32)  
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c 