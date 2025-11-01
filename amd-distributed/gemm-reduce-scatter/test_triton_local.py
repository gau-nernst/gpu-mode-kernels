#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from IPython import embed
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from reference import generate_input, ref_kernel, check_implementation
from run import config
import importlib
import ref10_first
importlib.reload(ref10_first)
from ref10_first import custom_kernel, _triton_mm_kernel_autotune
from log_utils import log
import torch
from triton.testing import do_bench

def get_local_out_ptr(M, N, input):
    emp = []
    for i in range(8):
        emp.append(torch.zeros(M // 8, N, dtype=input.dtype, device=input.device))
    local_base_ptr = torch.zeros(8, dtype=torch.uint64, device=input.device)
    for i in range(8):
        local_base_ptr[i] = emp[i].data_ptr()
    return local_base_ptr, emp
def get_torch_prof_ctx():
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) 
    return ctx 
ctx = get_torch_prof_ctx()
with ctx:
    for i in config[-6:]:   
        log("start", i)
        # if i['m'] < 513:
        #     continue
        for j in range(3):
            # log(f"{i}")
            data = generate_input(0, 8, i["m"], i["n"], i["k"], i["has_bias"], i["seed"] + j)   

            # log(f"emp[0].shape: {emp[0].shape}")
                
            ret = get_local_out_ptr(i["m"], i["n"], data[0])
            
            output = custom_kernel(data, True, local_ret = ret)
            output2 = ref_kernel(data, True)
            # log(f"{output.shape=} {output2.shape=}")
            # embed()
            custom_output = torch.sum(torch.stack(ret[1]), dim=0).to(torch.device("cuda:0")).reshape(i["m"]//8, i["n"])
            # log(output2.shape, custom_output.shape)
            
            assert check_implementation(output2, custom_output), "NOTPASS!!!!!!!!!!!!!!\n"  
        # log("passed one")
        
        for k, v in _triton_mm_kernel_autotune.cache.items():
            print(f"{k}: {v.all_kwargs()}")

        # continue 

        def bench_custom():
            custom_kernel(data, True, local_ret = ret)
            torch.cuda.synchronize()
            
        def bench_ref():
            ref_kernel(data, True)
            torch.cuda.synchronize()
        
        with torch.autograd.profiler.record_function("triton_gpu"):
            triton_time = do_bench(lambda: custom_kernel(data, True, local_ret = ret))
        with torch.autograd.profiler.record_function("torch_gpu"):
            torch_time = do_bench(lambda: ref_kernel(data, True))
        
        log(f"M: {i['m']:>5}, N: {i['n']:>5}, K: {i['k']:>5}, triton_time: {triton_time:.2f}, \ttorch_time: {torch_time:.2f},\t ratio: {torch_time/triton_time * 100:.2f}%")
        
        with torch.autograd.profiler.record_function("triton_cpu"):
            triton_time = do_bench(bench_custom, bench_cpu=True)
        with torch.autograd.profiler.record_function("torch_cpu"):
            torch_time = do_bench(bench_ref, bench_cpu=True)
        
        log(f"M: {i['m']:>5}, N: {i['n']:>5}, K: {i['k']:>5}, triton_time: {triton_time:.2f}, \ttorch_time: {torch_time:.2f},\t ratio: {torch_time/triton_time * 100:.2f}%")

ctx.export_chrome_trace("local_matmul.json")
    # %