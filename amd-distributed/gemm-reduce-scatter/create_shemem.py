import os
# for i in os.environ: 
#     print(f"{i=}, {os.environ[i]=}")
import torch 
import iris
try:
    from log_utils import log, log_first
except Exception:
    log = print
    log_first = print
import sys
rank = int(sys.argv[1])
import time
import ctypes
import os

log(f"rank = {rank}")
heap_size = 1 << 32
# num_gpus = iris.hip.count_devices()
heap_bases = []
import pickle
zeros = []
gpu_id = rank 
# iris.hip.set_device(gpu_id)
ipc_handle = iris.hip.hipIpcMemHandle_t()
mem = torch.zeros(heap_size, device=f"cuda:{0}", dtype=torch.bfloat16)
print(f"mem.device = {mem.device}")
zeros.append(mem)
heap_base = mem.data_ptr()
heap_base_ptr = ctypes.c_void_p(heap_base)
ipc_ptr = iris.hip.get_ipc_handle(heap_base_ptr, None)
torch.cuda.synchronize()
heap_bases.append(ipc_ptr)
pickle.dump(heap_bases, open(f"heap_bases_{rank}.pkl", "wb"))
 
iris.Iris 

torch.set_printoptions(threshold=float("inf"))
torch.set_printoptions(linewidth=2000000)
log("created, sleep forever wait for a.txt")
for i in range(30): 
    if os.path.exists(f"a.txt"): 
        os.system("rm -rf a.txt")
        for j in range(8):
            now = zeros[j][:64*16].reshape(64, 16)
            log(j, "\n", now)
    if os.path.exists(f"finish.txt"):
        log("finish here")
        # os.system("rm -rf finish.txt")
        break
    if i % 5 == 0: 
        log("no finish.txt here")
    time.sleep(1)
