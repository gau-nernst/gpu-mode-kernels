# pytorch_all2all.py
import os
import torch
import torch.distributed as dist
import dataclasses
from task import input_t, output_t
import numpy as np
import pickle
def log(*msg, **kwargs) -> None:
    import time
    import os
    import sys
    import torch.distributed as dist
    if dist.is_initialized():
        rank = dist.get_rank()
    else: 
        rank = 0

    try:
        raise Exception
    except:
        linenum = sys.exc_info()[2].tb_frame.f_back.f_lineno
        filename = sys.exc_info()[2].tb_frame.f_back.f_code.co_filename
    if int(os.environ.get("RANK", "0")) > 0:
        return
    # ANSI color codes
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    filename_only = filename.split("/")[-1]
    current_time = time.strftime("%H:%M:%S", time.localtime())
    milliseconds = int((time.time() % 1) * 1000)
    time_with_ms = f"{current_time}.{milliseconds:03d}"
    print(
        f"{time_with_ms} {YELLOW}RANK-{rank}{YELLOW} {BLUE}{filename_only}{RESET}:{YELLOW}{linenum}{RESET}:",
        *msg,**kwargs
    )
    print("", end="", flush=True)


# ---------------- MoE config ----------------
@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16

def analyze_error_pattern(error_mask, tensor_name="tensor"):
    """
    分析错误模式的tensor，提供简洁的错误描述
    
    Args:
        error_mask: torch.Tensor, 只包含0和1的tensor，1表示错误位置
        tensor_name: str, tensor的名称用于描述
    """
    if error_mask.sum() == 0:
        log(f"✓ {tensor_name}: No errors found")
        return
    
    total_elements = error_mask.numel()
    error_count = error_mask.sum().item()
    error_rate = error_count / total_elements
    
    log(f"✗ {tensor_name}: {error_count}/{total_elements} ({error_rate:.2%}) elements have errors")
    log(f"  Shape: {list(error_mask.shape)}")
    
    # 分析每个维度的错误分布
    for dim in range(error_mask.ndim):
        dim_errors = error_mask.sum(dim=tuple(i for i in range(error_mask.ndim) if i != dim))
        if dim_errors.sum() > 0:
            error_indices = (dim_errors > 0).nonzero(as_tuple=True)[0]
            if len(error_indices) > 0:
                ranges = find_consecutive_ranges(error_indices.tolist())
                log(f"  Dim {dim} errors: {format_ranges(ranges)} (total: {len(error_indices)})")

def find_consecutive_ranges(indices):
    """找到连续的索引范围"""
    if not indices:
        return []
    
    ranges = []
    start = indices[0]
    end = indices[0]
    
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append((start, end))
            start = end = i
    ranges.append((start, end))
    
    return ranges

def format_ranges(ranges, max_ranges=50):
    """格式化范围列表为可读字符串"""
    if not ranges:
        return "none"
    
    formatted = []
    for start, end in ranges[:max_ranges]:
        if start == end:
            formatted.append(f"{start}")
        else:
            formatted.append(f"{start}:{end+1}")
    
    if len(ranges) > max_ranges:
        formatted.append(f"... and {len(ranges) - max_ranges} more ranges")
    
    return ", ".join(formatted)

def find_error_slices(error_mask, min_consecutive=3):
    """
    找到错误的连续slice区域
    
    Args:
        error_mask: torch.Tensor, 错误mask
        min_consecutive: int, 最小连续长度才报告
    
    Returns:
        list: 错误slice的描述
    """
    slices = []
    ndim = error_mask.ndim
    
    if ndim == 1:
        # 1D tensor: 检查连续错误区域
        if error_mask.any():
            error_indices = error_mask.nonzero(as_tuple=True)[0]
            ranges = find_consecutive_ranges(error_indices.tolist())
            for start, end in ranges:
                if end - start + 1 >= min_consecutive:
                    slices.append(f"[{start}:{end+1}]")
    
    elif ndim == 2:
        # 检查整行错误
        row_errors = error_mask.all(dim=1)
        if row_errors.any():
            error_rows = row_errors.nonzero(as_tuple=True)[0]
            ranges = find_consecutive_ranges(error_rows.tolist())
            for start, end in ranges:
                if end - start + 1 >= min_consecutive:
                    slices.append(f"[{start}:{end+1}, :] (entire rows)")
        
        # 检查整列错误  
        col_errors = error_mask.all(dim=0)
        if col_errors.any():
            error_cols = col_errors.nonzero(as_tuple=True)[0]
            ranges = find_consecutive_ranges(error_cols.tolist())
            for start, end in ranges:
                if end - start + 1 >= min_consecutive:
                    slices.append(f"[:, {start}:{end+1}] (entire columns)")
    
    elif ndim >= 3:
        # 高维tensor: 检查各个维度的整个超平面
        for dim in range(ndim):
            # 创建除了当前维度外的所有维度列表
            other_dims = [i for i in range(ndim) if i != dim]
            # 检查在其他所有维度上都有错误的slice
            dim_errors = error_mask.all(dim=tuple(other_dims))
            if dim_errors.any():
                error_positions = dim_errors.nonzero(as_tuple=True)[0]
                ranges = find_consecutive_ranges(error_positions.tolist())
                for start, end in ranges:
                    if end - start + 1 >= min_consecutive:
                        slice_str = [":"] * ndim
                        slice_str[dim] = f"{start}:{end+1}"
                        slices.append(f"[{', '.join(slice_str)}] (dim {dim} hyperplane)")
    
    return slices

def summarize_errors(error_mask, tensor_name="tensor", show_slices=True):
    """
    综合错误分析函数
    """
    log(f"\n=== Error Analysis for {tensor_name} ===")
    analyze_error_pattern(error_mask, tensor_name)
    
    if show_slices and error_mask.sum() > 0:
        slices = find_error_slices(error_mask)
        if slices:
            log("  Problematic slices:")
            for slice_desc in slices:
                log(f"    {slice_desc}")
        
        # 显示一些典型的错误位置
        error_positions = error_mask.nonzero(as_tuple=False)
        if len(error_positions) > 0:
            log("  Sample error positions:")
            sample_size = min(100, len(error_positions))
            for i in range(sample_size):
                pos = error_positions[i].tolist()
                pos_str = "[" + ", ".join(map(str, pos)) + "]"
                log(f"    {pos_str}")
            if len(error_positions) > sample_size:
                log(f"    ... and {len(error_positions) - sample_size} more positions")

# 在现有的代码中使用这些函数
def enhanced_tolerance_check(expected, actual, rtol=1e-5, atol=1e-8, tensor_name="comparison"):
    """增强的容忍度检查，包含错误分析"""
    abs_diff = torch.abs(expected - actual)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-10)
    
    tol_exceeded = (abs_diff > atol) & (rel_diff > rtol)
    
    # 使用新的分析函数
    summarize_errors(tol_exceeded, tensor_name)
    
    return tol_exceeded
def detailed_compare(arr1, arr2, tensor_name, rtol=1e-2, atol=5e-3, top_n=10, fast_log=False, back = None):
    """Compare two arrays and provide detailed information about
    differences."""
    # arr1.sort()
    # arr2.sort()
    arr1 = np.array(arr1.cpu().numpy())
    arr2 = np.array(arr2.cpu().numpy())
    
    import random
    with open(f"output_{dist.get_rank()}.pkl", "wb") as f:
        pickle.dump((arr1, arr2, back), f)

    # Try to squeeze arrays to remove dimensions of size 1
    arr1_squeezed = np.squeeze(arr1)
    arr2_squeezed = np.squeeze(arr2)

    # Use squeezed arrays for comparison

    arr1 = arr1_squeezed
    assert arr1.shape == arr2.shape, f"{arr1.shape} != {arr2.shape}"
    if arr2.size == arr1.size:
        arr2 = arr2_squeezed.reshape(arr1.shape)

    if arr1.shape != arr2.shape:
        log(f"✗ {tensor_name}: Shapes are not equal {arr1.shape} != {arr2.shape}")
        log(f"  After squeeze: {arr1_squeezed.shape} != {arr2_squeezed.shape}")
        return False

    # Handle boolean arrays by converting to int
    if arr1.dtype == bool or arr2.dtype == bool:
        arr1 = arr1.astype(int)
        arr2 = arr2.astype(int)

    # Calculate absolute and relative differences
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = np.abs((arr1 - arr2) / (np.abs(arr2) + 1e-10))  # Add small epsilon to avoid division by zero

    # Find indices where the difference exceeds tolerance
    tol_exceeded = (abs_diff > atol) & (rel_diff > rtol)
    
    # 使用新的错误分析函数 (转换为torch tensor)
    summarize_errors(torch.from_numpy(tol_exceeded), "_tolerance_check")

    if not np.any(tol_exceeded):
        # log(f"✓ {tensor_name}: All values are within tolerance")
        return True

    log(f"✗ {tensor_name}: Found {np.sum(tol_exceeded)} values exceeding tolerance")
    log(f"  Array shapes: {arr1.shape} vs {arr2.shape}")
    log(f"  Tolerance: rtol={rtol}, atol={atol}")

    # Flatten arrays for sorting but keep track of original indices
    abs_diff_flat = abs_diff.flatten()
    rel_diff_flat = rel_diff.flatten()
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()
    if fast_log:
        return

    # Get indices of largest absolute differences
    largest_abs_diff_flat_indices = np.argsort(abs_diff_flat)[-top_n:][::-1]

    log(f"\n  Top {min(top_n, len(largest_abs_diff_flat_indices))} largest absolute differences:")
    for i, flat_idx in enumerate(largest_abs_diff_flat_indices):
        if abs_diff_flat[flat_idx] > 0:  # Only show non-zero differences
            # Convert flat index to multi-dimensional index
            multi_idx = np.unravel_index(flat_idx, arr1.shape)
            log(
                f"    #{i + 1}: index={multi_idx}, arr1={arr1_flat[flat_idx]:.6f}, arr2={arr2_flat[flat_idx]:.6f}, "
                f"abs_diff={abs_diff_flat[flat_idx]:.6f}, rel_diff={rel_diff_flat[flat_idx]:.6f}"
            )

    # Get indices of largest relative differences
    largest_rel_diff_flat_indices = np.argsort(rel_diff_flat)[-top_n:][::-1]

    log(f"\n  Top {min(top_n, len(largest_rel_diff_flat_indices))} largest relative differences:")
    for i, flat_idx in enumerate(largest_rel_diff_flat_indices):
        if rel_diff_flat[flat_idx] > 0:  # Only show non-zero differences
            # Convert flat index to multi-dimensional index
            multi_idx = np.unravel_index(flat_idx, arr1.shape)
            log(
                f"    #{i + 1}: index={multi_idx}, arr1={arr1_flat[flat_idx]:.6f}, arr2={arr2_flat[flat_idx]:.6f}, "
                f"abs_diff={abs_diff_flat[flat_idx]:.6f}, rel_diff={rel_diff_flat[flat_idx]:.6f}"
            )

    # Statistical summary
    log("\n  Statistical summary of differences:")
    log(f"    Max absolute difference: {np.max(abs_diff):.6f}")
    log(f"    Mean absolute difference: {np.mean(abs_diff):.6f}")
    log(f"    Max relative difference: {np.max(rel_diff):.6f}")
    log(f"    Mean relative difference: {np.mean(rel_diff):.6f}")
    log(f"    Percentage of values exceeding tolerance: {100 * np.sum(tol_exceeded) / arr1.size:.2f}%")

    log("=" * 100)
    return False
# ---------------- data per dp rank ----------------
CNT = 0
class RankTestData:
    def __init__(self, cfg: MoEConfig, rng: torch.Generator, rank: int):
        global CNT
        device = torch.device(f"cuda:{rank}")
        self.num_tokens = int(
            torch.randint(
                1, cfg.max_num_tokens, [1], generator=rng, device=device
            ).item()
        )
        if os.environ.get("FULL_TOKEN", "") == "1":
            self.num_tokens = cfg.max_num_tokens
        # token expert map
        self.indices = torch.empty(
            self.num_tokens, cfg.experts_per_token, dtype=torch.int32, device=device
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(cfg.num_experts, generator=rng, device=device)
            self.indices[i] = perm[: cfg.experts_per_token]
        # topk weights
        self.weights = torch.rand(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.float32,
            generator=rng,
            device=device,
        )
        # dp tokens, input of dispatch
        self.x = torch.randn(
            self.num_tokens,
            cfg.hidden_dim,
            dtype=cfg.in_dtype,
            generator=rng,
            device=device,
        )
        if os.environ.get("HAND_DATA", ""):
            self.weights = self.weights*0.0 + 0.01 * CNT
            self.x = self.x*0.0 + 0.01 * CNT





from task import input_t, output_t
import torch


def generate_input(rank: int, world_size: int, m: int, n: int, k: int, has_bias: bool, seed: int, is_bench=False) -> input_t:
    """
    Generate random input and weights for the Allgather-Gemm operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [local_M, k]
            weight: torch.Tensor,  # [local_N, K]
            bias: Optional[torch.Tensor],  # [local_N] or None
        )
    """
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert n % world_size == 0, "n must be divisible by world_size"
    if not is_bench:
        local_m = m // world_size
    else:
        local_m = m
    local_n = n // world_size

    # Generate random inputs and weights
    input = (torch.rand((local_m, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    weight = (torch.rand((local_n, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    # input.data_ptr()
    if os.environ.get("HAND_DATA", "") == "1":
        input = input*0.0 + 1
        weight = weight*0.0 + 1

    bias = None
    if has_bias:
        bias = (torch.rand((local_n,), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    return (input, weight, bias)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for AG-GEMM operation.
    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [local_M, K].
            - weight: Weight tensor of shape [local_N, K].
            - bias: Optional bias tensor of shape [local_N] or None.
    Returns:
        output: Resulting tensor of shape [local_M * world_size, local_N].
    """
    input, weight, bias = data
    local_M, K = input.shape
    world_size = torch.distributed.get_world_size()
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    # allgather
    torch.distributed.all_gather_into_tensor(full_input, input)
    # torch.cuda.synchronize()
    # matmul
    output = torch.matmul(full_input, weight.T)

    if bias is not None:
        output = output + bias

    return output

import time
def check_implementation(data: input_t, output: output_t, ref_result = None):
    global CNT
    if ref_result is None:
        expected = ref_kernel(data)
    else:
        expected = ref_result

    if output.device != expected.device:
        return False, f"Output device mismatch: {output.device} != {expected.device}"
    if os.environ.get("SKIP_CHECK", ""):
        log("SKIP_CHECK is set, skipping check")
        return True, ""
    torch.cuda.synchronize()
    if CNT < 4: 
        CNT += 1
        log(f"{CNT=}, skip check")
        return True, ""
    res = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    if not res:
        max_diff = (output - expected).abs().max().item()
        zero_count_output = (output == 0).sum().item()
        zero_count_expected = (expected == 0).sum().item()
        log(f"Zero count - output: {zero_count_output=}, expected: {zero_count_expected=}")
        log(f"WRONG result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {output.shape}")
        log(f"max_diff: {max_diff}")
        time.sleep(1)
        # log(data[0], data[1])
        # log(output, expected)
        
        log(f"{CNT=} detailed_compare ", detailed_compare(output.to(torch.float32), expected.to(torch.float32), "output", rtol=1e-3, atol=1e-3))
        exit(-1)
        return False, f"{CNT=} Output values mismatch, {output} != {expected}"
    CNT += 1
    return True, ""
