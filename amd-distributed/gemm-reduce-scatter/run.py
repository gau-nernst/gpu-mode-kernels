import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
os.environ["ZZ"] = "1"

import torch
import torch.multiprocessing as mp
from triton.testing import do_bench
import time
from task import input_t, output_t
from log_utils import log, log_first
from reference import generate_input, ref_kernel, check_implementation
import typer
import torch.distributed as dist


from ref10_first import custom_kernel, get_torch_prof_ctx


import os
from log_utils import log
config = [
  {"world_size": 8, "m": 64, "n": 32, "k": 128, "has_bias": True, "seed": 2035},
  {"world_size": 8, "m": 64, "n": 2880, "k": 2880, "has_bias": True, "seed": 2035},
  {"world_size": 8, "m": 64, "n": 3584, "k": 14336, "has_bias": True, "seed": 13},
  {"world_size": 8, "m": 64, "n": 16, "k": 16, "has_bias": True, "seed": 4297},
  {"world_size": 8, "m": 512, "n": 4608, "k": 36864, "has_bias": False, "seed": 1597},
  {"world_size": 8, "m": 2048, "n": 4096, "k": 7168, "has_bias": False, "seed": 716},
  {"world_size": 8, "m": 2048, "n": 8192, "k": 30720, "has_bias": False, "seed": 20201},
  {"world_size": 8, "m": 4096, "n": 2880, "k": 2880, "has_bias": True, "seed": 136},
  {"world_size": 8, "m": 4096, "n": 8192, "k": 2048, "has_bias": True, "seed": 138},
  {"world_size": 8, "m": 8192, "n": 3584, "k": 14336, "has_bias": True, "seed": 748},
  {"world_size": 8, "m": 8192, "n": 4608, "k": 36864, "has_bias": True, "seed": 4422},
  {"world_size": 8, "m": 8192, "n": 8192, "k": 28672, "has_bias": False, "seed": 1536},
  {"world_size": 8, "m": 64, "n": 7168, "k": 18432, "has_bias": False, "seed": 1234},
  {"world_size": 8, "m": 512, "n": 4096, "k": 12288, "has_bias": True, "seed": 663},
  {"world_size": 8, "m": 2048, "n": 2880, "k": 2880, "has_bias": True, "seed": 166},
  {"world_size": 8, "m": 4096, "n": 4096, "k": 4096, "has_bias": False, "seed": 1371},
  {"world_size": 8, "m": 8192, "n": 4096, "k": 14336, "has_bias": True, "seed": 7168},
  {"world_size": 8, "m": 8192, "n": 8192, "k": 29568, "has_bias": False, "seed": 42},
]

def _worker(rank, world_size, init_url, last_rocm_perf, skip_check):
    global config
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method=init_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()


    if last_rocm_perf:
        config = [config[-1]]

    ctx = get_torch_prof_ctx() 
    with ctx:
        for i in config[:5]: 
            for j in range(5):
                data = generate_input(rank, world_size, i["m"], i["n"], i["k"], i["has_bias"], i["seed"] + j)
                output = custom_kernel(data)
                output2 = ref_kernel(data)
                torch.cuda.synchronize()
                if last_rocm_perf:
                    time.sleep(1)
                    os._exit(0)
                if not skip_check:
                    assert check_implementation(output2, output) 
            log_first("passed one config,", i)
    ctx.export_chrome_trace(f"trace_rank{rank}.json.gz")
    log_first("finished one")
    if rank == 0:
        os.system("touch finish.txt")
    return

    # 1/0
    # def detailed_bench(data): 
    #     input, weight, bias = data
    #     flops = input.shape[0] * input.shape[1] * weight.shape[0] * 2
    #     toc = do_bench(lambda: ref_kernel(data), warmup=2, rep=300, return_mode="min")
    #     tic = do_bench(lambda: custom_kernel(data), warmup=2, rep=300, return_mode="min")
    #     print(f"percent {toc/tic*100:.2f}%", f"my_kernel: {tic=:.2f}s tflops: {flops/tic*1e-9:.2f}", f"ref_kernel: {toc=:.2f}s tflops: {flops/toc*1e-9:.2f}")

    # from contextlib import contextmanager, nullcontext, redirect_stdout
    # def get_torch_prof_ctx(do_prof: bool):
    #     ctx = (torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #         ],
    #         record_shapes=True,
    #         with_stack=False,
    #     ) if do_prof else nullcontext())
    #     return ctx

    # ctx = get_torch_prof_ctx(True)

    # with ctx:
    #     for i in config[:-6]:  
    #         data = generate_input(0, i["world_size"], i["m"], i["n"], i["k"], i["has_bias"], i["seed"])
    #         detailed_bench(data)
    # ctx.export_chrome_trace(f"trace_gemm.json.gz")

app = typer.Typer(help="解析tf.log文件并运行相关逻辑")

@app.command()
def run(
    world_size: int = typer.Option(8, help="全局进程数"),
    last_rocm_perf: bool = typer.Option(False, help="是否只运行最后一个配置"),
    skip_check: bool = typer.Option(False, help="是否跳过检查"),
):
    # os.system(f"nohup python create_shemem.py {world_size} &")
    init_url = "tcp://127.0.0.1:29500"
    mp.spawn(
        fn=_worker,
        args=(world_size, init_url, last_rocm_perf, skip_check),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    app()

