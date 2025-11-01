import torch
from task import input_t, output_t
from torch.nn import functional as F
from log_utils import log, log_first
TMP = True

def generate_input(rank: int, world_size: int, m: int, n: int, k: int, has_bias: bool, seed: int) -> input_t:
    """
    Generate random input and weights for the Gemm-ReduceScatter operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [M, local_K]
            weight: torch.Tensor,  # [N, local_K]
            bias: Optional[torch.Tensor],  # [N] or None
        )
    """
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert k % world_size == 0, "k must be divisible by world_size"
    local_k = k // world_size

    # Generate random inputs and weights
    input = (torch.rand((m, local_k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    weight = (torch.rand((n, local_k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    if TMP: 
        input = input * 0.0 + (rank + 1)
        weight = weight * 0.0 + (rank + 1)
        has_bias = False

    bias = None
    if has_bias:
        gen.manual_seed(seed)
        bias = (torch.rand((n,), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01

    return (input, weight, bias)


def ref_kernel(data: input_t, local_bench: bool = False) -> output_t: 
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    output = F.linear(input, weight, bias)
    if local_bench: 
        # output = torch.ones(16, 2)
        M, N= output.shape
        output = output.reshape(8, -1)
        output = torch.sum(output, dim=0)
        output = output.reshape(M // 8, N)
        return output
    world_size = torch.distributed.get_world_size()
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output

def check_implementation(expected: input_t, output: output_t):
    rtol = 1e-3 
    atol = 1e-3
    if TMP: atol = 1000
    res = torch.allclose(output, expected, rtol=rtol, atol=atol)
    if not res: 

        log("NOTPASS!!!!!!!!!!!!!!\n")
        log("max_diff: ", torch.max(torch.abs(output - expected)))
        log("excepted\n", expected, "output\n", output, "excepted.shape", expected.shape, "output.shape", output.shape)
        return False
    # log_first(f"PASS for shape: {expected.shape}")
    # import traceback
    # traceback.print_stack()
    return True