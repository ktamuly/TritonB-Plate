import triton
import triton.language as tl
import torch


@triton.jit
def patch_embedding_kernel(
    img_ptr,
    weight_ptr,
    out_ptr,
    batch_size,
    in_channels,
    height,
    width,
    patch_size,
    embedding_dim,
    img_stride_b,
    img_stride_c,
    img_stride_h,
    img_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_h,
    weight_stride_w,
    out_stride_b,
    out_stride_p,
    out_stride_e,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_patch = tl.program_id(1)
    pid_embed = tl.program_id(2)

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    patch_h = (pid_patch // num_patches_w) * patch_size
    patch_w = (pid_patch % num_patches_w) * patch_size

    e_start = pid_embed * BLOCK_E
    e_offsets = e_start + tl.arange(0, BLOCK_E)
    e_mask = e_offsets < embedding_dim

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)

    for h_offset in range(patch_size):
        for w_offset in range(patch_size):
            h_pos = patch_h + h_offset
            w_pos = patch_w + w_offset

            for c in range(in_channels):
                pixel_ptr = (
                    img_ptr
                    + pid_batch * img_stride_b
                    + c * img_stride_c
                    + h_pos * img_stride_h
                    + w_pos * img_stride_w
                )
                pixel = tl.load(pixel_ptr)

                weight_ptrs = (
                    weight_ptr
                    + e_offsets * weight_stride_o
                    + c * weight_stride_i
                    + h_offset * weight_stride_h
                    + w_offset * weight_stride_w
                )
                weights = tl.load(weight_ptrs, mask=e_mask, other=0.0)

                acc += tl.where(e_mask, pixel * weights, 0.0)

    out_ptr_base = (
        out_ptr
        + pid_batch * out_stride_b
        + pid_patch * out_stride_p
        + e_start * out_stride_e
    )
    out_ptrs = out_ptr_base + tl.arange(0, BLOCK_E) * out_stride_e
    tl.store(out_ptrs, acc, mask=e_mask)


def triton_patch_embedding(img, patch_size, embedding_dim):
    batch_size, in_channels, height, width = img.shape

    assert (
        height % patch_size == 0
    ), f"Image height {height} is not divisible by patch size {patch_size}"
    assert (
        width % patch_size == 0
    ), f"Image width {width} is not divisible by patch size {patch_size}"

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    # Create random weights for visualization/testing
    weight = (
        torch.randn(
            embedding_dim,
            in_channels,
            patch_size,
            patch_size,
            device=img.device,
            dtype=img.dtype,
        )
        * 0.02
    )

    out = torch.empty(
        (batch_size, num_patches, embedding_dim), device=img.device, dtype=img.dtype
    )

    # Get strides for tensor memory layout
    img_stride_b, img_stride_c, img_stride_h, img_stride_w = img.stride()
    weight_stride_o, weight_stride_i, weight_stride_h, weight_stride_w = weight.stride()
    out_stride_b, out_stride_p, out_stride_e = out.stride()

    BLOCK_SIZE = min(triton.next_power_of_2(patch_size * patch_size), 256)
    BLOCK_E = min(triton.next_power_of_2(embedding_dim), 128)

    grid = (batch_size, num_patches, triton.cdiv(embedding_dim, BLOCK_E))

    patch_embedding_kernel[grid](
        img,
        weight,
        out,
        batch_size,
        in_channels,
        height,
        width,
        patch_size,
        embedding_dim,
        img_stride_b,
        img_stride_c,
        img_stride_h,
        img_stride_w,
        weight_stride_o,
        weight_stride_i,
        weight_stride_h,
        weight_stride_w,
        out_stride_b,
        out_stride_p,
        out_stride_e,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_E=BLOCK_E,
    )

    return out
