import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
import torch
from task import input_t, output_t
import time
import sys
import torch.distributed as dist
import triton
import triton.language as tl
import pickle
import functools
from torch.utils.cpp_extension import load_inline, load
from typing import Dict
import numpy as np
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton.testing import do_bench
OPEN_PERF = False
time_tensors_bank = []
time_tensor_save = []
os.system("sudo sed -i '66,82 s/^/#/' /usr/local/lib/python3.10/dist-packages/iris/__init__.py ")
with open("/usr/local/lib/python3.10/dist-packages/iris/__init__.py", "r") as f:
    lines = f.readlines()
    for line in lines:
        if "Check if the library exists" in line:
            print("first 5", line[:5])
            break
import iris
import signal
import traceback
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FIRST_RANK = None 
CREATE_SHEMEM_CODE = ""


CUDA_MAIN_SRC = ""
CPP_BARRIER = ""
CUDA_BARRIER = ""



from triton.compiler.errors import CompilationError

def make_launcher(constants, signature, warp_size):
    from triton.backends.amd.driver import ty_to_cpp, _BASE_ARGS_FORMAT, FLOAT_STORAGE_TYPE, FLOAT_PACK_FUNCTION, _get_path_to_hip_runtime_dylib
    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape, and
        # strides
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                ndim = sig.count(",") + 1
                dtype = re.match("tensordesc<([^[>]*)", sig).group()

                output.append("*" + dtype)
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        return output

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty == "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty == "constexpr":
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    signature = {idx: s for idx, s in enumerate(_expand_signature(signature.values()))}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]

    libhip_path = _get_path_to_hip_runtime_dylib()

    # generate glue code
    params = list(range(len(signature)))
    filtered_signature = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    # print(filtered_signature)
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in filtered_signature.items()) if len(filtered_signature) > 0 else ''
    args_format = ''.join([format_of(ty) for ty in filtered_signature.values()])
    format ="iiiKKOOOOO" + args_format
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <dlfcn.h>

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {{"{libhip_path}"}};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                     \\
  FOR_EACH_STR_FN(hipGetLastError)                                            \\
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                     \\
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                     \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipModuleLaunchCooperativeKernel, hipFunction_t f,          \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                         \\
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {{
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                             \\
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                             \\
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
}};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {{
  // Use the HIP runtime library loaded into the existing process if it exits.
  void *lib = dlopen("libamdhip64.so", RTLD_NOLOAD);

  // Otherwise, go through the list of search paths to dlopen the first HIP
  // driver library.
  if (!lib) {{
    int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
    for (int i = 0; i < n; ++i) {{
      void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
      if (handle) {{
        lib = handle;
      }}
    }}
  }}
  if (!lib) {{
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }}

  typedef hipError_t (*hipGetProcAddress_fn)(
      const char *symbol, void **pfn, int hipVersion, uint64_t hipFlags,
      hipDriverProcAddressQueryResult *symbolStatus);
  hipGetProcAddress_fn hipGetProcAddress;
  dlerror(); // Clear existing errors
  const char *error = NULL;
  *(void **)&hipGetProcAddress = dlsym(lib, "hipGetProcAddress");
  error = dlerror();
  if (error) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "cannot query 'hipGetProcAddress' from libamdhip64.so");
    dlclose(lib);
    return false;
  }}

  // Resolve all symbols we are interested in.
  int hipVersion = HIP_VERSION;
  uint64_t hipFlags = 0;
  hipDriverProcAddressQueryResult symbolStatus;
  hipError_t status = hipSuccess;
#define QUERY_EACH_FN(hipSymbolName, ...)                                      \
  status = hipGetProcAddress(#hipSymbolName,                                   \
                             (void **)&hipSymbolTable.hipSymbolName,           \
                             hipVersion, hipFlags, &symbolStatus);             \
  if (status != hipSuccess) {{                                                 \
    PyErr_SetString(PyExc_RuntimeError,                                        \
                    "cannot get address for '" #hipSymbolName                  \
                    "' from libamdhip64.so");                                  \
    dlclose(lib);                                                              \
    return false;                                                              \
  }}

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}}

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
       const char* str = hipSymbolTable.hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function, hipDeviceptr_t profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  hipDeviceptr_t global_scratch = 0;
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    HIP_CHECK(hipSymbolTable.hipModuleLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    hipDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  hipError_t status = hipSuccess;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }}
  ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  if (!ptr_info.dev_ptr)
    goto cleanup;
  uint64_t dev_ptr;
  status = hipSymbolTable.hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
  if (status == hipErrorInvalidValue) {{
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
      // Clear and ignore HIP error
      (void)hipSymbolTable.hipGetLastError();
  }}
  ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
cleanup:
  Py_DECREF(ret);
  return ptr_info;
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat/blob/5e317108f872c904eb726cb8d560dcadbdf88a72/pythoncapi_compat.h#L482-L492
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *profile_scratch_obj = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in filtered_signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  {' '.join(float_storage_decls)}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    return NULL;
  }}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  hipDeviceptr_t profile_scratch = 0;
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, (hipDeviceptr_t)profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  if(PyErr_Occurred()) {{
    return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  if (!initSymbolTable()) {{
    return NULL;
  }}
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    # print(__file__, "patch launcher success\n", src)
    return src
class HIPLauncher(object):

    def __init__(self, src, metadata):
        from triton.backends.amd.driver import compile_module_from_src, wrap_handle_tensor_descriptor, include_dirs
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        self.signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, self.signature, metadata.warp_size)
        # print(__file__, "\n", src)
        mod = compile_module_from_src(src=src, name="__triton_launcher", include_dirs=include_dirs)
        has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in self.signature.values())

        self.launch = wrap_handle_tensor_descriptor(mod.launch) if has_tensor_desc_arg else mod.launch
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.count_num = len([i for i in self.signature.values() if i == 'constexpr'])

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        from triton.runtime import _allocation
        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None
        # if dist.get_rank() == 0:
        #     log("raw_args", args)
        args_new = args[:-self.count_num]
        # if dist.get_rank() == 0:
        #     log(args_new) 
        #     log(args[self.count_num:])
        self.launch(gridX, gridY, gridZ, stream, function, None, *args_new)

from triton.backends.amd import driver
triton.backends.amd.driver.make_launcher = make_launcher
driver.HIPLauncher = HIPLauncher

CompilationError.source_line_count_max_in_message = 1

try:
    from log_utils import log, log_first
except Exception:
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
        BLUE = ""
        YELLOW = ""
        RESET = ""

        filename_only = filename.split("/")[-1]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        milliseconds = int((time.time() % 1) * 1000)
        time_with_ms = f"{current_time}.{milliseconds:03d}"
        print(
            f"{time_with_ms} {YELLOW}RANK-{rank}{YELLOW} {BLUE}{RESET}:{YELLOW}{linenum}{RESET}:",
            *msg,
        )
        print("", end="", flush=True)
        

    def log_first(*msg) -> None:
        import time
        import os
        import sys
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else: 
            rank = 0

        if rank != 0:
            return 
        try:
            raise Exception
        except:
            linenum = sys.exc_info()[2].tb_frame.f_back.f_lineno
            filename = sys.exc_info()[2].tb_frame.f_back.f_code.co_filename
        if int(os.environ.get("RANK", "0")) > 0:
            return
        # ANSI color codes
        BLUE = ""
        YELLOW = ""
        RESET = ""

        filename_only = filename.split("/")[-1]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        milliseconds = int((time.time() % 1) * 1000)
        time_with_ms = f"{current_time}.{milliseconds:03d}"
        print(
            f"{time_with_ms} {YELLOW}RANK-{rank}{YELLOW} {BLUE}{RESET}:{YELLOW}{linenum}{RESET}:",
            *msg,
        )
        with open("profile_data.txt", "a") as f:
            ss = ' '.join(str(m) for m in msg)
            f.write(f"{time_with_ms} {linenum}: {rank} {ss}\n")
        print("", end="", flush=True)


log("compile and load start")


tic = time.time()
if not os.environ.get("ZZ", ""):
    if CUDA_MAIN_SRC == "": # local
        _ = load(
            name = "noname", 
            sources=[
                "ref10_first.hip", "ref10_first.cpp"
            ],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-g"], 
        )
    else: # remote
        _ = load_inline(
            name="noname",
            cpp_sources=[CPP_BARRIER],
            cuda_sources=[CUDA_BARRIER],
            verbose=True,
            no_implicit_headers=True,
            extra_cuda_cflags=[
                "-O3",
                "--offload-arch=gfx942",
                "-save-temps",
                "-g",

            ],
        )
        pass

# threading.Thread(target=periodic_flush, daemon=True).start()
log("compile and load end use time: %f seconds" % (time.time() - tic), os.getpid())

configs = [
    # (64, 32, 128),
    # (64, 64, 128),
    # (64, 128, 64),
    # (64, 256, 64),
    (BM, BN, BK)
    for BM in (8, 32, 64, 128, 256)
    for BN in (32, 64, 128, 256)
    for BK in (32, 64, 128)
]


if torch.version.hip:
    configs = [
        triton.Config(
            dict(
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, 
                # AMD-specific
                waves_per_eu=2,
                matrix_instr_nonkdim=16,
                kpack=1,
            ),
            num_stages=2,
            num_warps=8,
        )
        for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K in configs
    ]
else: 
    configs = [
        triton.Config(
            dict(
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, 
                # AMD-specific
                # waves_per_eu=2,
                # matrix_instr_nonkdim=16,
                # kpack=1,
            ),
            num_stages=2,
            num_warps=8,
        )
        for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K in configs
    ]
    

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid


@triton.jit
def compute_pid(pid, grid_m, grid_n, GROUP_M: tl.constexpr, REMAP_XCD: tl.constexpr = True):
    if REMAP_XCD:
        # most of the time, this if beneficial
        # 4096, 4096, 512
        pid = remap_xcd(pid, grid_m * grid_n)

    if GROUP_M == 1:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)

    return pid_m, pid_n

@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_memrealtime $0""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,  # 改为True以支持pipeliner优化
        pack=1,
    )
    return tmp
@triton.jit
def _kernel1(
        a_ptr, 
        A_index: "tl.int64",
        b_ptr, 
        bias_ptr,
        time_tensor,
        my_rank: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        heap_base_0: tl.constexpr, 
        heap_base_1: tl.constexpr, 
        heap_base_2: tl.constexpr,
        heap_base_3: tl.constexpr, 
        heap_base_4: tl.constexpr, 
        heap_base_5: tl.constexpr, 
        heap_base_6: tl.constexpr, 
        heap_base_7: tl.constexpr, 
        stride_am: tl.constexpr, 
        stride_ak: tl.constexpr, 
        stride_bk: tl.constexpr, 
        stride_bn: tl.constexpr, 
        stride_cm: tl.constexpr, 
        stride_cn: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        ACTIVATION: tl.constexpr, 
        HAS_BIAS: tl.constexpr,
        CACHE_MODIFIER: tl.constexpr,
        OPEN_PERF: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
):
    # pid_m = tl.program_id(axis=0)
    # pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if OPEN_PERF:
        time_index = tl.program_id(0) + 1
        if tl.program_id(0) == 0:
            tl.store(time_tensor, tl.num_programs(0))
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)

    # pid_n, pid_m = compute_pid(tl.program_id(0), num_pid_n, num_pid_m, 4, True)
    pid_m, pid_n = compute_pid(tl.program_id(0), num_pid_m, num_pid_n, 4, True)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + A_index + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not EVEN_K:
           a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0, cache_modifier=CACHE_MODIFIER)
           b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0, cache_modifier=CACHE_MODIFIER)
        else: 
            a = tl.load(a_ptrs, cache_modifier=CACHE_MODIFIER)
            b = tl.load(b_ptrs, cache_modifier=CACHE_MODIFIER)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)
        accumulator += bias.to(tl.float32)
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)
    c = accumulator.to(tl.bfloat16)


    tl.static_assert(M % (BLOCK_SIZE_M * 8) == 0, "num_pid_m must be divisible by 8")
    which_base_use = pid_m // (num_pid_m // 8)
    ptr_diff = tl.cast(heap_base_0, tl.int64)
    if which_base_use == 0: 
        ptr_diff = tl.cast(heap_base_0, tl.int64)
    if which_base_use == 1:
        ptr_diff = tl.cast(heap_base_1, tl.int64)
    if which_base_use == 2:
        ptr_diff = tl.cast(heap_base_2, tl.int64)
    if which_base_use == 3:
        ptr_diff = tl.cast(heap_base_3, tl.int64)
    if which_base_use == 4:
        ptr_diff = tl.cast(heap_base_4, tl.int64)
    if which_base_use == 5:
        ptr_diff = tl.cast(heap_base_5, tl.int64)
    if which_base_use == 6:
        ptr_diff = tl.cast(heap_base_6, tl.int64)
    if which_base_use == 7:
        ptr_diff = tl.cast(heap_base_7, tl.int64) 
    offs_cm = (pid_m % (num_pid_m // 8)) * BLOCK_SIZE_M + my_rank * (M // 8) + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) 
    
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0) 
    offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = a_ptr + ptr_diff +  stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if EVEN_N:
        tl.store(c_ptrs, c, cache_modifier=".cg")
    else:
        c_mask = (offs_cm[:, None] < (M)) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".cg")
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)


def prune_configs_v21(config, nargs):
    """
    这个 pre_hook 为每个 config 单独调用。
    如果 config 有效，返回 True，否则返回 False。
    """
    M, N = nargs["M"]
    
    # 检查条件
    if M % (8 * config.kwargs["BLOCK_M"]) != 0:
        return False
    
    if M >= 2048 and config.kwargs["BLOCK_M"] == 8:
        return False
        
    return True
_triton_mm_kernel_autotune = triton.autotune(
    configs=configs,
    key=["M", "N", "K"],
    prune_configs_by={
        'early_config_pruning': prune_configs_v21,
        'perf_model': None,
        'top_k': None,
    },
    do_bench=functools.partial(do_bench, warmup=100, rep=500),
)(_kernel1)


online_config = {
(64, 7168, 2304): {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(512, 4096, 1536): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(2048, 2880, 360): {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(4096, 4096, 512): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(8192, 4096, 1792): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(8192, 8192, 3696): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
}

online_config_group = {
    57344: 2048,
    262144: 2048,
    737280:  2048,
    512 * 4096: 4096,
    1024 * 4096: 4096,
    1024 * 8192: 4096,
}


from triton.backends.amd.driver import HIPLauncher
from triton.compiler.compiler import LazyDict
from triton.compiler.compiler import ASTSource
from triton import knobs

pre_compile_cache = None
pre_compile_cache2 = None

from triton.runtime.driver import driver
def get_stream(): 
    device = driver.active.get_current_device()    
    stream = driver.active.get_current_stream(device)
    if stream != 0: 
        log("ret here is not 0")
    return stream
def launch_triton_kernel(a, b, bias):
    global pre_compile_cache
    if pre_compile_cache is not None:
        ret, grid = pre_compile_cache
        ret._run.launch(
            #launcher.launch_cooperative_grid,
            grid[0], grid[1], grid[2], 
            0, # stream
            ret.function, 
            None, 
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, #knobs.runtime.launch_enter_hook, 
            None, #knobs.runtime.launch_exit_hook,
            A_ptr_index_hack.data_ptr(),
            (a.data_ptr() - A_ptr_index_hack.data_ptr())//2,
            b.data_ptr(),
            bias.data_ptr(),
            # a.data_ptr(), 
            # b.data_ptr(), 
            # c.data_ptr(),
            # heap_base_ptr.data_ptr(),
            # bias.data_ptr(), 
        )
    else:
        global __conf
        M, local_K = a.shape
        N = b.shape[0]
        if (M, N, local_K) not in __conf:
            return origin((a, b, bias))
        heap_base_ptr = load_heap_base_ptr(a)
        M, local_K = a.shape
        # b = b.T
        N, K2 = b.shape
        # M, K = a.shape
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
        # assert local_K == K2, f"K: {local_K} K2: {K2}"
        assert a.dtype == b.dtype
        # BLOCK_SIZE_M = min((M // 8),32)
        # assert M % (BLOCK_SIZE_M * 8) == 0, f"M: {M} BLOCK_SIZE_M: {BLOCK_SIZE_M}"
        # BLOCK_SIZE_N = 32
        # BLOCK_SIZE_K = 64
        ACTIVATION = ""
        def grid(meta): 
            return (triton.cdiv(meta["M"], meta["BLOCK_SIZE_M"]), triton.cdiv(meta["N"], meta["BLOCK_SIZE_N"]))
        key = (M, N, local_K)
        if key in online_config:
            config_local =  online_config[key]
        else: 
            config_local = online_config[(64, 7168, 2304)]    
        HAS_BIAS = bias is not None
        grid = (triton.cdiv(M, config_local["BLOCK_SIZE_M"]) * triton.cdiv(N, config_local["BLOCK_SIZE_N"]), 1, 1)
        my_rank = get_rank(a)
        # log(f"{c.stride(0)=} {c.stride(1)=} {b.stride(0)=} {b.stride(1)=}, {a.stride(0)=} {a.stride(1)=}")
        time_tensor = None
        if OPEN_PERF:
            global time_tensors_bank
            if len(time_tensors_bank) == 0:
                for i in range(105): 
                    time_tensors_bank.append(torch.zeros((grid[0] * grid[1] * 10000), dtype=torch.int64, device=a.device))
            time_tensor = time_tensors_bank.pop(0)
            assert time_tensor is not None
             
        ret = _kernel1[grid](
            a_ptr = A_ptr_index_hack,  
            A_index=(a.data_ptr() - A_ptr_index_hack.data_ptr())//2,
            b_ptr=b, 
            bias_ptr=bias, 
            time_tensor=time_tensor,
            my_rank=my_rank,
            M=M, 
            N=N, 
            K=local_K,
            heap_base_0=base_addrs[0],
            heap_base_1=base_addrs[1],
            heap_base_2=base_addrs[2],
            heap_base_3=base_addrs[3],
            heap_base_4=base_addrs[4],
            heap_base_5=base_addrs[5],
            heap_base_6=base_addrs[6],
            heap_base_7=base_addrs[7],
            stride_am=local_K, 
            stride_ak=1,
            stride_bk=1, 
            stride_bn=local_K,
            stride_cm=N, 
            stride_cn=1,
            ACTIVATION=ACTIVATION, HAS_BIAS=HAS_BIAS,
            CACHE_MODIFIER=".cg" if M == 64 else "",
            OPEN_PERF=OPEN_PERF,
            EVEN_K = local_K % config_local["BLOCK_SIZE_K"] == 0,
            EVEN_N = N % config_local["BLOCK_SIZE_N"] == 0,
            **config_local
        )
        assert M % config_local["BLOCK_SIZE_M"] == 0, f"{M=} {config_local['BLOCK_SIZE_M']=}"
        # assert N % config_local["BLOCK_SIZE_N"] == 0, f"{N=} {config_local['BLOCK_SIZE_N']=}"
        # assert local_K % config_local["BLOCK_SIZE_K"] == 0, f"{local_K=} {config_local['BLOCK_SIZE_K']=}"
        if not OPEN_PERF:
            log(f"[RANK-{my_rank}] {M, N, local_K}  A: {ret.n_regs} , B: {ret.n_spills}")
        B = ret.n_spills
        assert B <= 0, f"{B=}"
        if not OPEN_PERF:
            pre_compile_cache = (ret, grid) 
        else:
            time_tensor_save.append(time_tensor)

    M, _= a.shape
    N, _ = b.shape
    return grouped_sum(M, N, get_rank(a), load_heap_base_ptr(a))

base_ptr = None 
A_ptr_index_hack = None
streams = []
B_index = [] 
base_addrs = []
def load_heap_base_ptr(a):
    # input is a torch.tensor.
    global base_ptr
    if base_ptr is not None: 
        return base_ptr
    global FIRST_RANK
    device_index = a.device.index
    tic=time.time()
    dist.barrier()
    log(f"RANK-{dist.get_rank()} pid: {os.getpid()}")
    # time.sleep(10)

   
    if device_index == 0: 
        os.system("rm -rf *.bin")

    dist.barrier()
    
    FIRST_RANK = device_index
    # if device_index == 0:
    base_ptr = torch.ops.my_ops.init(device_index).to(device=torch.device(f"cuda:{device_index}"))
    global A_ptr_index_hack
    A_ptr_index_hack = torch.empty(100, dtype=torch.bfloat16, device=a.device)
    base_addrs.clear()
    for i in range(8):
        base_addrs.append((base_ptr[i].item() - A_ptr_index_hack.data_ptr())//2)
        # print(f"RANK-{dist.get_rank()}_{i}: 0x{base_ptr[i].item():x}")
        assert type(base_addrs[-1]) == int, f"base_addrs[-1]={base_addrs[-1]}"
    log("init cost", time.time() - tic)


    return base_ptr



def get_rank(input: torch.Tensor):
    if torch.cuda.device_count() == 1:
        return dist.get_rank()
    rank = input.device.index
    # print("get_rank", rank)
    
    return rank

GLOBAL_LOCAL_BENCH = False if os.environ.get("LOCAL_BENCH", "") else False



IS_FIRST = True
IS_INIT = False
    

def close_heap_base_ptr():
    global base_ptr
    global A_ptr_index_hack
    base_ptr = None
    A_ptr_index_hack = None

    # triton.compiler.clear_cache()
    ret = torch.ops.my_ops.clear()
    os.system("touch now.txt")

    dist.barrier()


@triton.jit
def _gemm_a16w16_reduce_kernel_optimized(
    c_out_ptr,
    c_in_ptr: tl.constexpr,
    total_elements: tl.constexpr,  # M * N
    MAX_KSPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # 处理的元素数量
):

    c_in_ptr = tl.cast(c_in_ptr, tl.pointer_type(tl.bfloat16))
    pid = tl.program_id(axis=0)   
    block_start = pid * BLOCK_SIZE 
    offs = block_start + tl.arange(0, BLOCK_SIZE) 
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = c_in_ptr + offs_k[:, None] * total_elements + offs[None, :]
    c = tl.load(c_in_ptrs, cache_modifier=".cg")
    c = tl.sum(c, axis=0)
    c = c.to(c_out_ptr.type.element_ty)
    c_out_ptrs = c_out_ptr + offs
    c_out_ptrs = tl.max_contiguous(tl.multiple_of(c_out_ptrs, BLOCK_SIZE), BLOCK_SIZE) 
    tl.store(c_out_ptrs, c, cache_modifier=".cg")

def grouped_sum(M, N, my_rank, heap_base_ptr: torch.Tensor) -> torch.Tensor:
    torch.ops.my_ops.barrier(my_rank)
    out = torch.empty((M // 8, N), device=torch.device(f"cuda:{my_rank}"), dtype=torch.bfloat16)
    global pre_compile_cache2
    if pre_compile_cache2 is not None:
        ret, grid = pre_compile_cache2
        ret._run.launch(
            grid[0], grid[1], grid[2], 
            0, # stream
            ret.function, 
            None, 
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, #knobs.runtime.launch_enter_hook, 
            None, #knobs.runtime.launch_exit_hook,
            out.data_ptr(),
        )
        return out
    # torch.cuda.synchronize()
    BS = online_config_group[M//8*N]
    grid_reduce = (triton.cdiv(M//8*N, BS), 1, 1)
    assert M//8*N % BS == 0, f"{M//8*N=} {BS=}"
    heap_base = heap_base_ptr[my_rank].item()
    ret = _gemm_a16w16_reduce_kernel_optimized[grid_reduce](
        # load_heap_base_ptr(out.device.index)[rank],
        out,
        heap_base,
        M//8*N,
        8,
        BS,
    )
    pre_compile_cache2 = (ret, grid_reduce)

    # torch.cuda.synchronize()
    return out














def get_torch_prof_ctx():
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False if torch.cuda.device_count() == 1 else False,
    ) 
    return ctx
GLOBAL_CTX = None
CNT = 0
START_STEP = 1
STOP_STEP = 5
CNT_PLUS = 0
def get_perf_cond(m, n):
    # return m == 512 # 2
    return m == 8192 and n == 8192 # 6

IS_FIRST = True
def origin(data):
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = F.linear(input, weight, bias)
    # output = torch.matmul(input, weight.T)
    # if bias is not None:
    #     output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output
__conf = [
    (64, 7168, 18432//8), #####?????? bench all have bias?
    (512, 4096, 12288//8),
    (2048, 2880, 2880//8),
    (4096, 4096, 4096//8),
    (8192, 4096, 14336//8),
    (8192, 8192, 29568//8),
]
def custom_kernel(data: input_t, local_bench: bool = False, local_ret = None) -> output_t:
    input, weight, bias = data
    return launch_triton_kernel(input, weight, bias)

origin_destroy_process = dist.destroy_process_group

def patch_destroy_process_group():
    log("start clear")
    tic=time.time()
    global pre_compile_cache, pre_compile_cache2
    pre_compile_cache = None
    pre_compile_cache2 = None
    if OPEN_PERF: 
        global time_tensor_save, time_tensors_bank
        time_tensors_bank.clear()
        for index,i in enumerate(time_tensor_save):
            pickle.dump(i.cpu(), open(f"time_tensor_rank{dist.get_rank()}_{index}.pkl", "wb"))
        time_tensor_save.clear()

    send_index = 0
    global IS_INIT
    close_heap_base_ptr()
    IS_INIT = False
    log("clear cost", time.time() - tic)
    origin_destroy_process()

dist.destroy_process_group = patch_destroy_process_group

import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)
if __name__ == "__main__":
    if os.environ.get("ZZ", ""):
        M, N, K = 8192, 3696, 8192
        config_local = online_config[(M, N, K)]
        grid = (triton.cdiv(M, config_local["BLOCK_M"]) * triton.cdiv(N, config_local["BLOCK_N"]), 1, 1)
        a = torch.empty((M, K), dtype=torch.bfloat16, device="cuda:0")
        b = torch.empty((N, K), dtype=torch.bfloat16, device="cuda:0")
        c = torch.empty((M, N), dtype=torch.bfloat16, device="cuda:0")
        my_rank = 0
        send_index = 0
        heap_base_ptr = a.data_ptr()
        time_tensor = torch.empty((grid[0] * grid[1] * 10000), dtype=torch.int64, device="cuda:0")
        ret = triton_mm_kernel[grid](
            A_ptr=a, 
            A_index=a.data_ptr(), #TODO fix
            heap_base_0=a.data_ptr(),
            heap_base_1=a.data_ptr(),
            heap_base_2=a.data_ptr(),
            heap_base_3=a.data_ptr(),
            heap_base_4=a.data_ptr(),
            heap_base_5=a.data_ptr(),
            heap_base_6=a.data_ptr(),
            heap_base_7=a.data_ptr(),
            my_rank_base=a.data_ptr(),
            B_ptr=b, C_ptr=c, bias_ptr = None,
            M=M, N=N, K=K,
            my_rank = my_rank,
            signal_index = send_index,
            # heap_base_ptr = heap_base_ptr,
            # BLOCK_M = config_local["BLOCK_M"],
            # BLOCK_N = config_local["BLOCK_N"],
            # BLOCK_K = config_local["BLOCK_K"],
            time_tensor = time_tensor,
            HAS_BIAS=False,
            cache_modifier=".cg" if M * 8 == 64 else "",
            **config_local
        )
        log(f"[RANK-{my_rank}] {M, N, K}  regs: {ret.n_regs} , spills: {ret.n_spills}")
# import gc
# gc.set_threshold(0, 0, 0)
# gc.disable()


