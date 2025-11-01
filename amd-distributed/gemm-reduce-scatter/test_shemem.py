#%%
import torch
import os
import time
import pickle
import iris
import ctypes
import numpy as np 
def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    iris.hip.hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        iris.hip.hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = iris.hip.hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    iris.hip.hip_try(
        iris.hip.hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value

# %% load iris origin
# Load IPC handles from binary files written by make_share.hip
ipc = []
for i in range(8):
    filename = f"ipc_handles_rank{i}.bin"
    with open(filename, "rb") as f:
        handle_data = f.read(64)  # hipIpcMemHandle_t is 64 bytes
        ipc.append(np.frombuffer(handle_data, dtype=np.uint8))
ret = open_ipc_handle(ipc[0], 0)
print(ret)



#%% load my handle
import numpy as np
for i in range(8):
    print(f"rank = {i}")
    handle_0 = np.asarray(np.frombuffer(ipc[i], dtype=np.uint8))
    ret = open_ipc_handle(handle_0, 0)
    print(ret)


#%%
handle_0 = np.asarray(np.frombuffer(ipc[0], dtype=np.uint8))
ret = open_ipc_handle(handle_0, 0)
print(6)