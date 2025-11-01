#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import triton
import triton.language as tl

# The @triton.autotune decorator should be applied directly to the JIT kernel function.
# It will test different configurations for this specific kernel.
# The @triton.jit decorator must come AFTER @triton.autotune.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_R': 16, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_R': 32, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_R': 64, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_R': 16, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_N': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_R': 8, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_R': 256, 'BLOCK_SIZE_N': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _grouped_sum_kernel(
    # Pointers to matrices
    in_ptr,
    out_ptr,
    # Matrix dimensions
    M,
    N,
    # Strides
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    # Meta-parameters from autotuner
    BLOCK_SIZE_R: tl.constexpr, # Block size for the output rows
    BLOCK_SIZE_N: tl.constexpr, # Block size for the columns
    # --- FIX ---
    # Add the data type as a constexpr argument. This provides essential type
    # information to the compiler when using raw data pointers.
    DTYPE: tl.constexpr,
):
    """
    Triton kernel for grouped sum reduction.
    Each program instance computes a BLOCK_SIZE_R x BLOCK_SIZE_N tile of the output matrix.
    """
    # tl.static_print(f"in_ptr", in_ptr)
    # tl.static_print(f"out_ptr", out_ptr)
    in_ptr = tl.cast(in_ptr, out_ptr.dtype)
    # tl.static_print(f"in_ptr", in_ptr)
    pid_r = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_N), dtype=tl.float32)
    
    M_DIV_8 = M // 8

    for k in range(8):
        current_in_r = offs_r + k * M_DIV_8
        in_ptrs = in_ptr + (current_in_r[:, None] * stride_in_m + offs_n[None, :] * stride_in_n)
        mask = (current_in_r[:, None] < M) & (offs_n[None, :] < N)
        block = tl.load(in_ptrs, mask=mask, other=0.0)
        accumulator += block
    
    # --- FIX ---
    # Cast the accumulator to the correct output type passed as a constexpr.
    # Do not use `out_ptr.dtype.element_ty` as `out_ptr` is just an integer.
    c = accumulator.to(DTYPE)
    c = accumulator
    out_ptrs = out_ptr + (offs_r[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_r[:, None] < M_DIV_8) & (offs_n[None, :] < N)
    tl.store(out_ptrs, c, mask=out_mask)


# This is now a standard Python function, not decorated with autotune.
# It acts as a launcher for the autotuned kernel.
def grouped_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Performs a grouped sum reduction on the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (M, N). M must be divisible by 8.

    Returns:
        torch.Tensor: The output tensor of shape (M/8, N).
    """
    M, N = x.shape
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be a contiguous CUDA tensor"
    assert M % 8 == 0, "The size of the first dimension must be divisible by 8"

    out = torch.empty((M // 8, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M // 8, META['BLOCK_SIZE_R']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # --- FIX ---
    # Convert the torch dtype to a triton dtype to be passed to the kernel.
    # This is necessary for the compiler to correctly infer pointer types.
    dtype = getattr(tl, str(x.dtype).split('.')[-1])
    
    _grouped_sum_kernel[grid](
        x.data_ptr(),
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        # Pass the triton dtype as a constexpr argument.
        DTYPE=dtype,
    )
    return out

# --- Verification ---
if __name__ == '__main__':
    # Test case
    M, N = 512, 4096
    
    # Create a random tensor on GPU
    input_tensor = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)

    # PyTorch implementation for reference
    def torch_grouped_sum(t):
        M_dim, N_dim = t.shape
        output = t.clone() 
        output = output.reshape(8, -1)
        output = torch.sum(output, dim=0)
        output = output.reshape(M_dim // 8, N_dim)
        return output

    # Calculate results
    torch_result = torch_grouped_sum(input_tensor)
    triton_result = grouped_sum(input_tensor) # This call will trigger autotuning on the first run

    # Check if the results are close
    print("PyTorch and Triton results match:", torch.allclose(torch_result, triton_result, atol=1e-2, rtol=1e-2))
    
    # --- Benchmarking ---
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg='provider',
            line_vals=['torch', 'triton'],
            line_names=['PyTorch', 'Triton'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='grouped-sum-performance',
            args={'M': 1024},
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn((M, N), device='cuda', dtype=torch.float16)
        quantiles = [0.2, 0.5, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_grouped_sum(x), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_sum(x), quantiles=quantiles)
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)

