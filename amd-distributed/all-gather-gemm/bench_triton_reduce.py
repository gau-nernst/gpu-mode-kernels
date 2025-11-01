#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from reference import generate_input, ref_kernel, check_implementation
from run import config
import importlib
import ref10_first
importlib.reload(ref10_first)
from ref10_first import custom_kernel, _kernel2, get_torch_prof_ctx
from log_utils import log
import triton
import triton.language as tl
import torch
from triton.testing import do_bench
def launch_triton_kernel(M, N, input_ptr):
    BLOCK_SIZE_R = 8 
    BLOCK_SIZE_N = 256
    # time.sleep(rank * 500)
 
    grid = lambda META: (
        triton.cdiv(M // 8, BLOCK_SIZE_R),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    dtype = getattr(tl, str(input_tensor.dtype).split('.')[-1])
    if False:
        # log(f"grouped_sum {M=}, {N=}, {input.shape=}, {rank=}, {out.data_ptr()=} load_heap_base_ptr()[rank]= {load_heap_base_ptr()[rank]}")    
        pass
    # triton.compile()
    out = torch.empty((M // 8, N), device=input_tensor.device, dtype=input_tensor.dtype)
    _kernel2[grid](
        input_ptr, # one h2d here?
        0,
        out,
        M,
        N,
        out.stride(0),
        out.stride(1),
        out.stride(0),
        out.stride(1),
        DTYPE=dtype,
        BLOCK_SIZE_R = BLOCK_SIZE_R,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
    )
    return out

ctx = get_torch_prof_ctx()
with ctx:
    for i in config[-6:]: 
        log(f"{i}")
        input_tensor = torch.randn(i["m"], i["n"], device="cuda", dtype=torch.bfloat16)
        input_ptr = torch.empty(8, dtype=torch.uint64, device="cuda")
        input_ptr[0] = input_tensor.data_ptr()
        output = launch_triton_kernel(i["m"], i["n"], input_ptr)
        def ref_kernel(data): 
            # print(f"{data.shape=}")
            now = torch.reshape(data, (8, -1))
            # print(f"{now.shape=}")
            now = torch.sum(now, dim=0)
            # print(f"2:: {now.shape=}")
            return torch.reshape(now, (-1, data.shape[1]))
        output2 = ref_kernel(input_tensor)
        # log(f"{output.shape=} {output2.shape=}")
         
        assert check_implementation(output2, output), "NOTPASS!!!!!!!!!!!!!!\n"

        latency_tf = do_bench(lambda: launch_triton_kernel(i["m"], i["n"], input_ptr), warmup=100, rep=500) 
        latency_torch = do_bench(lambda: ref_kernel(input_tensor))
        
        log(f"{latency_tf=} {latency_torch=}, ratio: {latency_tf/latency_torch * 100:.2f}%")
        

        log("passed one")
ctx.export_chrome_trace(f"bench_triton_reduce.json.gz")