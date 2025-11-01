GLOBAL_BASE_PTR = None
GLOBAL_MY_RANK = None
GLOBAL_SEND_INDEX = 0
GLOBAL_A_ptr_index_hack = None
GLOBAL_IS_INIT = False



OPEN_PERF = True if os.environ.get("OPEN_PERF", "") else False
SEND_CTA_PER_DEVICE = 8 ############ tmp.......... TODO:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
# os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"]="150000"
# os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"]="150000"
# os.environ['TORCH_NCCL_ENABLE_TIMING'] = '0'
# os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
# os.environ['TORCH_NCCL_DUMP_ON_TIMEOUT'] = '0'
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
import triton
import triton.language as tl
from torch import Tensor
from triton.testing import do_bench
import torch.nn.functional as F

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

        args_new = args[:-self.count_num]
        # print(self.launch_cooperative_grid, gridX, gridY, gridZ, stream, function, args_new)
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
    if CPP_BARRIER == "": # local
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
                f"-DSEND_CTA_PER_DEVICE={SEND_CTA_PER_DEVICE}",
            ],
        )
        pass

# threading.Thread(target=periodic_flush, daemon=True).start()
log("compile and load end use time: %f seconds" % (time.time() - tic), os.getpid())
    

@triton.jit
def get_xcd_id(pid, my_rank, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    num_pid_m = M // BM
    num_pid_n = N // BN
    if pid < (num_pid_m * num_pid_n) // 8: 
        result = (pid // num_pid_n) + my_rank * (num_pid_m // 8), pid % num_pid_n
    else:
        pid_left = pid - (num_pid_m * num_pid_n) // 8
        dest_rank = pid_left % 7
        if dest_rank >= my_rank:
            dest_rank += 1
        dest_pid = pid_left // 7
        # print(f"pid_left: {pid_left}, dest_rank: {dest_rank}, dest_pid: {dest_pid}")
        result = dest_rank * (num_pid_m // 8) + (dest_pid // num_pid_n), dest_pid % num_pid_n
    return result


@triton.jit
def device_sleep():
    tmp = tl.inline_asm_elementwise(
        asm="""s_sleep 1024""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp
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
def fast_load(ptr, mask, other=0.0):
    """使用 global_load_dword 加载数据（示例）
    
    Args:
        ptr: 要加载的地址（tensor）
        mask: 加载掩码
        other: mask为False时的默认值
    """
    # 方法1: 基本的 global_load_dword
    result = tl.inline_asm_elementwise(
        asm="""global_load_dword $0, $1, off sc0 sc1
               s_waitcnt vmcnt(0)
               buffer_inv sc0 sc1""",
        constraints="=v,v",  # $0=输出(VGPR), $1=输入地址(VGPR)
        args=[ptr],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    
    # 注意：inline_asm 不直接支持 mask，如果需要 mask，
    # 应该在外面用 tl.where 处理：
    # result = tl.where(mask, result, other)
    
    return result

@triton.jit
def fast_load_with_cache_modifier(ptr, use_cg: tl.constexpr = True):
    """使用不同的 cache modifier 加载数据
    
    Args:
        ptr: 要加载的地址
        use_cg: 是否使用 .cg (cache global) modifier
    """
    if use_cg:
        # 使用 sc0 sc1 (streaming cache)
        result = tl.inline_asm_elementwise(
            asm="""global_load_dword $0, $1, off sc0 sc1
                   s_waitcnt vmcnt(0)""",
            constraints="=v,v",
            args=[ptr],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
    else:
        # 默认 cache 行为
        result = tl.inline_asm_elementwise(
            asm="""global_load_dword $0, $1, off
                   s_waitcnt vmcnt(0)""",
            constraints="=v,v",
            args=[ptr],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
    return result

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
def compute_pid_old(pid, grid_m: tl.constexpr, grid_n: tl.constexpr, GROUP_M: tl.constexpr, REMAP_XCD: tl.constexpr = True):
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
#
@triton.jit
def compute_pid(pid, 
    grid_m:  tl.constexpr, 
    grid_n:  tl.constexpr, 
    GROUP_M: tl.constexpr, 
    my_rank: tl.constexpr, 
    REMAP_XCD: tl.constexpr=True):
    GROUP_M = 1 
    # grid_mn_const:tl.constexpr = grid_m * grid_n
    if pid < (grid_m * grid_n // 8): 
        if REMAP_XCD:
            pid = remap_xcd(pid, grid_m // 8 * grid_n)

        if GROUP_M == 1:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
        else:
            width = GROUP_M * grid_n
            group_id = pid // width
            group_size = min(grid_m//8 - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_size)
            pid_n = (pid % width) // (group_size)
        pid_m = (pid_m + grid_m // 8 * my_rank) % grid_m
        return pid_m, pid_n
    else:
        pid -= (grid_m * grid_n) // 8
        if REMAP_XCD:
            which_xcd = pid % 8
            xcd_local_index = pid // 8
            local_xcd_row, local_xcd_col = xcd_local_index // grid_n, xcd_local_index % grid_n
            
            id = local_xcd_row * 8 + which_xcd
            which_group = id % 7
            group_pos = id // 7
            if group_pos == grid_m//8:
                which_group += 3
                group_pos -=1
                local_xcd_col += grid_n // 2
            # grid_mn_const:tl.constexpr = grid_m * grid_n
            if grid_m * grid_n >= 416:
                if pid >= 416: ######### TMP HACK here.
                    local_xcd_col += grid_n // 2
            # assert group_pos < grid_m//8, f"which_group = {which_group} is out of range {id} {group_pos}"
            final_pos_row = which_group * (grid_m//8) + group_pos 
            # print(id, which_group, group_pos, final_pos_row)
            pid_m = final_pos_row
            pid_n = local_xcd_col
        pid_m = (pid_m + (grid_m // 8) * (my_rank + 1)) % grid_m

    return pid_m, pid_n

@triton.jit
def compute_pid_pure(pid, 
    grid_m:  tl.constexpr, 
    grid_n:  tl.constexpr, 
    GROUP_M: tl.constexpr, 
    my_rank: tl.constexpr, 
    REMAP_XCD: tl.constexpr=True):
    GROUP_M = 1 
    # grid_mn_const:tl.constexpr = grid_m * grid_n
    if pid < (grid_m * grid_n // 8): 
        if REMAP_XCD:
            pid = remap_xcd(pid, grid_m // 8 * grid_n)

        if GROUP_M == 1:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
        else:
            width = GROUP_M * grid_n
            group_id = pid // width
            group_size = min(grid_m//8 - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_size)
            pid_n = (pid % width) // (group_size)
        pid_m = (pid_m + grid_m // 8 * my_rank) % grid_m
        return pid_m, pid_n
    else:
        pid -= (grid_m * grid_n) // 8
        if REMAP_XCD:
            which_xcd = pid % 8
            xcd_local_index = pid // 8
            local_xcd_row, local_xcd_col = xcd_local_index // grid_n, xcd_local_index % grid_n
            
            id = local_xcd_row * 8 + which_xcd
            which_group = id % 7
            group_pos = id // 7
            # if group_pos == grid_m//8:
            #     which_group += 3
            #     group_pos -=1
            #     local_xcd_col += grid_n // 2
            # # grid_mn_const:tl.constexpr = grid_m * grid_n
            # if grid_m * grid_n >= 416:
            #     if pid >= 416: ######### TMP HACK here.
            #         local_xcd_col += grid_n // 2
            # assert group_pos < grid_m//8, f"which_group = {which_group} is out of range {id} {group_pos}"
            final_pos_row = which_group * (grid_m//8) + group_pos 
            # print(id, which_group, group_pos, final_pos_row)
            pid_m = final_pos_row
            pid_n = local_xcd_col
        pid_m = (pid_m + (grid_m // 8) * (my_rank + 1)) % grid_m

    return pid_m, pid_n
@triton.jit
def triton_mm_kernel(
    A_ptr, # fake fp16 tensor, 用来获取属性的..
    A_index: "tl.int64", # 真正的 A_index, 相对于 A_ptr 来说....
    B_ptr,
    C_ptr,
    bias_ptr,
    signal_index,
    time_tensor,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    my_rank: tl.constexpr,
    heap_base_0: tl.constexpr, # 每一个 heap base 的地址, 相对于真正的 A_ptr 的偏移
    heap_base_1: tl.constexpr,
    heap_base_2: tl.constexpr,
    heap_base_3: tl.constexpr,
    heap_base_4: tl.constexpr,
    heap_base_5: tl.constexpr,
    heap_base_6: tl.constexpr,
    heap_base_7: tl.constexpr,
    my_rank_base: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SEND_CTA_NUM: tl.constexpr,
    GROUP_M: tl.constexpr = 4,
    HAS_BIAS: tl.constexpr = False,
    REMAP_XCD: tl.constexpr = True,
    cache_modifier: tl.constexpr = "",
    OPEN_PERF: tl.constexpr = False,
    SEND_THREAD_NUM: tl.constexpr = 8192 * 2, #8192 * 2,
    SLEEP_CYCLE: tl.constexpr = 100,
    SPLIT_BLOCK: tl.constexpr = 2
):
    tl.static_print("S_BLOCK", SPLIT_BLOCK)
    COMM_PIDs = 7 * SEND_CTA_NUM

    SPLIT_SEND: tl.constexpr =  M >= 2048
    # tl.static_print("SPLIT_SEND", SPLIT_SEND)

    if tl.program_id(axis=0) < COMM_PIDs:
        if OPEN_PERF:
            time_index = tl.program_id(0) + 1
            if tl.program_id(0) == 0:
                tl.store(time_tensor, tl.num_programs(0))
            tl.store(time_tensor + time_index, read_realtime())
            time_index += tl.num_programs(0)
        dest_rank = tl.program_id(0) // SEND_CTA_NUM
        ptr_diff = tl.cast(heap_base_0, tl.int64)
        if dest_rank >= my_rank: dest_rank+=1
        if dest_rank == 0: 
            ptr_diff = tl.cast(heap_base_0, tl.int64)
        if dest_rank == 1:
            ptr_diff = tl.cast(heap_base_1, tl.int64)
        if dest_rank == 2:
            ptr_diff = tl.cast(heap_base_2, tl.int64)
        if dest_rank == 3:
            ptr_diff = tl.cast(heap_base_3, tl.int64)
        if dest_rank == 4:
            ptr_diff = tl.cast(heap_base_4, tl.int64)
        if dest_rank == 5:
            ptr_diff = tl.cast(heap_base_5, tl.int64)
        if dest_rank == 6:
            ptr_diff = tl.cast(heap_base_6, tl.int64)
        if dest_rank == 7:
            ptr_diff = tl.cast(heap_base_7, tl.int64)
        SIGNAL_POS= 2 * 8192 * 8192 // 2
        offset_am = tl.arange(0, SEND_THREAD_NUM)
        if not SPLIT_SEND:
            if OPEN_PERF:
                tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                time_index += tl.num_programs(0)  
            for i in range(SEND_THREAD_NUM * (tl.program_id(0) % SEND_CTA_NUM), K * M // 8, SEND_THREAD_NUM * SEND_CTA_NUM):
                val = tl.load(tl.multiple_of(A_ptr + A_index, [16]) + i + offset_am, cache_modifier=".cv")
                tl.store(tl.multiple_of(A_ptr + ptr_diff + i + my_rank * K * (M // 8), [16]) + offset_am, val, cache_modifier=".wt")
            if OPEN_PERF:
                tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                time_index += tl.num_programs(0)  
            tl.atomic_add(tl.cast(A_ptr + ptr_diff, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + my_rank * 32, 1, sem="release", scope="sys")
            if OPEN_PERF:
                tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                time_index += tl.num_programs(0)  
        else:
            SPLIT_LEN: tl.constexpr= BLOCK_M * SPLIT_BLOCK * K
            LEN: tl.constexpr = M * K // 8 // SPLIT_LEN
            # tl.static_print("LEN", LEN)
            if OPEN_PERF:
                tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                time_index += tl.num_programs(0)  
            for j in range(LEN):
                for i in range(SEND_THREAD_NUM * (tl.program_id(0) % SEND_CTA_NUM), 
                               SPLIT_LEN, 
                               SEND_THREAD_NUM * SEND_CTA_NUM):
                    val = tl.load(tl.multiple_of(A_ptr + j * SPLIT_LEN + A_index, [64]) + i + offset_am, cache_modifier=".cv")
                    mask = i + offset_am < SPLIT_LEN
                    tl.store(tl.multiple_of(A_ptr + j * SPLIT_LEN + ptr_diff + i + my_rank * K * (M // 8), [64]) + offset_am, val, mask=mask, cache_modifier=".wt")
                # A_ptr += j * (K * BLOCK_M // 8)
                # if my_rank == 0:
                    # tl.device_print(f"RANK={my_rank} send_for j, ", j)
                if OPEN_PERF:
                    tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                    time_index += tl.num_programs(0)  
                tl.atomic_add(tl.cast(A_ptr + ptr_diff, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + my_rank * 32 + j, 1, sem="release", scope="sys")
                if OPEN_PERF:
                    tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                    time_index += tl.num_programs(0)  
        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)  
                        
            
        return

        

    else: 
        if OPEN_PERF:
            time_index = tl.program_id(0) + 1
            if tl.program_id(0) == 0:
                tl.store(time_tensor, tl.num_programs(0))
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)

        
        # based on triton.ops.matmul
        pid = tl.program_id(0) - COMM_PIDs

        # re-order program ID for better L2 performance
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        tl.static_assert(GROUP_M == 4, "GROUP_M must be 4")
        if SPLIT_SEND:
            
            if M == 4096:
                pid_m, pid_n = compute_pid_pure(pid, grid_m, grid_n, GROUP_M, my_rank) ############ tmp here......
            else:
                pid_m, pid_n = compute_pid(pid, grid_m, grid_n, GROUP_M, my_rank) ############ tmp here......
        else:
            pid_m, pid_n = compute_pid_old(pid, grid_m, grid_n, GROUP_M, my_rank)
        # tl.device_assert(pid_m < grid_m, "pid_m < grid_m")
        # tl.device_assert(pid_m >=0, "pid_m < grid_m")
        # tl.device_assert(pid_n < grid_n, "pid_n < grid_n")
        # tl.device_assert(pid_n >=0, "pid_m < grid_m")
        # pid_m = (pid_m + my_rank * (grid_m // 8)) % grid_m
        # pid_m, pid_n = pid // grid_n, pid % grid_n

        stride_am, stride_ak = K, 1
        stride_bn, stride_bk = K, 1
        
        IS_LOCAL = (pid_m // (grid_m//8)) == my_rank
        SIGNAL_POS = 2 * 8192 * 8192 // 2
        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)  
        USE_FAST_LOAD: tl.constexpr = True ############################# TODO:
        if not IS_LOCAL: # IS_LOCAL
            dest_rank = pid_m // (grid_m//8)
            A_ptr += my_rank_base 
            flag_ptr = tl.cast(A_ptr, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + dest_rank * 32
            if SPLIT_SEND:
                flag_ptr = tl.cast(A_ptr, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + dest_rank * 32 + (pid_m % (grid_m//8) // SPLIT_BLOCK)
            else:
                flag_ptr = tl.cast(A_ptr, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + dest_rank * 32
                
            if USE_FAST_LOAD:
                result = tl.load(flag_ptr, cache_modifier=".cv")
            else:
                result = tl.atomic_add(flag_ptr, 0,  sem="acquire") #, scope="gpu", cache_modifier=".cg")
            i = 0
            while result != SEND_CTA_NUM:
                i += 1
                for j in range(SLEEP_CYCLE):
                    device_sleep()
                if USE_FAST_LOAD:
                    result = tl.load(flag_ptr, cache_modifier=".cv")
                else:
                    result = tl.atomic_add(flag_ptr, 0,  sem="acquire") #, scope="gpu", cache_modifier=".cg")
                if i < 2000:
                    if OPEN_PERF:
                        if time_index < 20000:
                            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
                            time_index += tl.num_programs(0)  
                if i % 10000000 == 0: 
                    tl.device_print(f"RANK={my_rank} flag result=", result)
            
        if IS_LOCAL:
            A_ptr += A_index - my_rank * K * (M // 8)
        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)  
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)  
        EVEN_K: tl.constexpr = K % BLOCK_K == 0
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B, cache_modifier=cache_modifier)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0, cache_modifier=cache_modifier)
            acc = tl.dot(a, b, acc)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk
            # if COLLECT_TIME:
            #     tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            #     time_index += tl.num_programs(0)  

        # rematerialize rm and rn to save registers
        idx_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)  
        if HAS_BIAS:
            bias = tl.load(bias_ptr + idx_n, mask=idx_n < N) # TODO: cg?
            acc += bias.to(tl.float32)
        if OPEN_PERF:
            tl.store(time_tensor + time_index, read_realtime(), cache_modifier=".cg")
            time_index += tl.num_programs(0)

        # inductor generates a suffix
        stride_cm, stride_cn = N, 1
        xindex = idx_m * stride_cm + idx_n * stride_cn
        tl.store(C_ptr + xindex, acc, mask=(idx_m < M) & (idx_n < N), cache_modifier=".cg")
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
# _triton_mm_kernel_autotune = triton.autotune(
#     configs=configs,
#     key=["M", "N", "K"],
#     prune_configs_by={
#         'early_config_pruning': prune_configs_v21,
#         'perf_model': None,
#         'top_k': None,
#     },
#     do_bench=functools.partial(do_bench, warmup=100, rep=500, return_mode="median"),
#     cache_results=True,
# )(_kernel1)


online_config = {
(64, 2304, 7168): {}, ########## tmp not doing M=64 calculation
(512, 1536, 4096): {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 2, "SPLIT_BLOCK": 2}, # try test it's time.
(2048, 360, 2880): {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 2, "SPLIT_BLOCK": 2},
(4096, 512, 4096): {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 2, "SPLIT_BLOCK": 4}, 
(8192, 1792, 4096): {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 10, "SPLIT_BLOCK": 2},
(8192, 3696, 8192): {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 20, "SPLIT_BLOCK": 2}, #'NUM_KSPLIT': 1,  'SPLITK_BLOCK_SIZE': 8192},
}
online_config_keys = set(online_config.keys())


from triton.backends.amd.driver import HIPLauncher
from triton.compiler.compiler import LazyDict
from triton.compiler.compiler import ASTSource
from triton import knobs
log_cache = {}
pre_compile_cache = None 

from triton.runtime.driver import driver
def get_stream(): 
    device = driver.active.get_current_device()    
    stream = driver.active.get_current_stream(device)
    if stream != 0: 
        log("ret here is not 0")
    return stream

log(f"OPEN_PERF: {OPEN_PERF}")
time_tensors_bank = []
time_tensors_save = [] 


# def launch_triton_kernel(M, N, K, a, b, bias, my_rank, heap_base_ptr=None, do_bench=False, use_cache_opt = True):
def custom_kernel(data):
    input, weight, bias = data 
    local_M, K = input.shape
    N = weight.shape[0]
    if local_M == 8 and K == 7168:
        return launch_triton_split_k(input, weight, bias, local_M, K)
    c = torch.empty((local_M * 8, N), dtype=torch.bfloat16, device=input.device)
    global GLOBAL_SEND_INDEX
    global pre_compile_cache
    if pre_compile_cache is not None: ## TODO opt
        ret, grid = pre_compile_cache

        ret._run.launch(
            grid[0], grid[1], grid[2], 
            0, 
            ret.function, 
            None, 
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, 
            None, 
            GLOBAL_A_ptr_index_hack.data_ptr(), 
            (input.data_ptr() - GLOBAL_A_ptr_index_hack.data_ptr())//2,
            weight.data_ptr(), 
            c.data_ptr(), 
            bias.data_ptr(),  
            GLOBAL_SEND_INDEX,
        )
        GLOBAL_SEND_INDEX += 256
        return c
    else:
        global base_addrs
        HAS_BIAS = bias is not None
        # key = (local_M * 8, N, K, GLOBAL_MY_RANK, HAS_BIAS)
        key = (local_M * 8, N, K)
        if key in online_config:
            config_local =  online_config[key]
        else: 
            return origin(data)
        # log("key", key)
        grid = (SEND_CTA_PER_DEVICE * 7 + triton.cdiv(local_M * 8, config_local["BLOCK_M"]) * triton.cdiv(N, config_local["BLOCK_N"]), 1, 1)
        if True:
            load_heap_base_ptr()
            time_tensor = None
            if OPEN_PERF:
                global time_tensors_bank
                if len(time_tensors_bank) == 0:
                    for i in range(105): 
                        time_tensors_bank.append(torch.zeros((grid[0] * grid[1] * 10000), dtype=torch.int64, device=weight.device))
                time_tensor = time_tensors_bank.pop(0)
            ret = triton_mm_kernel[grid](
                A_ptr=GLOBAL_A_ptr_index_hack, 
                B_ptr=weight, 
                C_ptr=c, 
                bias_ptr=bias,
                signal_index=GLOBAL_SEND_INDEX,
                time_tensor=time_tensor,
                A_index=(input.data_ptr() - GLOBAL_A_ptr_index_hack.data_ptr())//2, #TODO fix
                heap_base_0=base_addrs[0],
                heap_base_1=base_addrs[1],
                heap_base_2=base_addrs[2],
                heap_base_3=base_addrs[3],
                heap_base_4=base_addrs[4],
                heap_base_5=base_addrs[5],
                heap_base_6=base_addrs[6],
                heap_base_7=base_addrs[7],
                my_rank_base=base_addrs[GLOBAL_MY_RANK],
                M=local_M * 8, N=N, K=K,
                my_rank = GLOBAL_MY_RANK,
                HAS_BIAS=HAS_BIAS,
                cache_modifier="",
                SEND_CTA_NUM = SEND_CTA_PER_DEVICE,
                OPEN_PERF=OPEN_PERF,
                **online_config[key]
            )
            log(f"OPEN_PERF: {OPEN_PERF}")
            if OPEN_PERF:
                time_tensors_save.append(time_tensor)
            # if not OPEN_PERF:
            #     log(f"[RANK-{GLOBAL_MY_RANK}] {key}  A: {ret.n_regs} , B: {ret.n_spills}", flush=True)
            B = ret.n_spills
            assert B == 0, f"B={B}"
            GLOBAL_SEND_INDEX += 256
            if not OPEN_PERF:
                pre_compile_cache = (ret, grid)
        return c

streams = []
B_index = [] 
base_addrs = []
def load_heap_base_ptr(a=None):
    global GLOBAL_BASE_PTR

    if GLOBAL_BASE_PTR is not None: 
        return GLOBAL_BASE_PTR
    global GLOBAL_MY_RANK
    GLOBAL_MY_RANK = dist.get_rank()
    device_index = GLOBAL_MY_RANK
    tic=time.time()
    dist.barrier()
    # log(f"RANK-{dist.get_rank()} pid: {os.getpid()}")
    if device_index == 0: 
        os.system("rm -rf *.bin")
    dist.barrier() 
    # if device_index == 0:
    GLOBAL_BASE_PTR = torch.ops.my_ops.init(device_index).to(device=torch.device(f"cuda:{device_index}"))
    global GLOBAL_A_ptr_index_hack
    GLOBAL_A_ptr_index_hack = torch.empty(100, dtype=torch.bfloat16, device=torch.device(f"cuda:{device_index}"))
    base_addrs.clear()
    for i in range(8):
        base_addrs.append((GLOBAL_BASE_PTR[i].item() - GLOBAL_A_ptr_index_hack.data_ptr())//2)
        assert type(base_addrs[-1]) == int, f"base_addrs[-1]={base_addrs[-1]}"
    # log("init cost", time.time() - tic)
    return GLOBAL_BASE_PTR



def get_rank(input: torch.Tensor):
    if torch.cuda.device_count() == 1:
        return dist.get_rank()
    rank = input.device.index
    # print("get_rank", rank)
    
    return rank

GLOBAL_LOCAL_BENCH = False if os.environ.get("LOCAL_BENCH", "") else False



def get_torch_prof_ctx():
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False if torch.cuda.device_count() == 1 else False,
    ) 
    return ctx

def get_perf_cond(m, n):
    # return m == 512 # 2
    return m == 8192 and n == 8192 # 6


def close_heap_base_ptr():
    global GLOBAL_BASE_PTR
    global GLOBAL_A_ptr_index_hack
    GLOBAL_BASE_PTR = None
    GLOBAL_A_ptr_index_hack = None

    # triton.compiler.clear_cache()
    ret = torch.ops.my_ops.clear()
    os.system("touch now.txt")

    dist.barrier()
__conf = [
    (4096//8, 4096//8, 4096),
]
def origin(data):
    input, weight, bias = data
    local_M, K = input.shape
    world_size = torch.distributed.get_world_size()
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    # allgather
    torch.distributed.all_gather_into_tensor(full_input, input)
    # matmul
    output = torch.matmul(full_input, weight.T)

    if bias is not None:
        output = output + bias

    return output


@triton.jit
def _gemm_a16w16_reduce_kernel_optimized(
    c_in_ptr,
    c_out_ptr,
    total_elements: tl.constexpr,  # M * N
    MAX_KSPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # 处理的元素数量
):

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

@triton.jit
def _gemm_a16_w16_split_kernel(
    A_ptr, 
    A_index: "tl.int64", 
    b_ptr, 
    c_ptr, 
    signal_index: "tl.int64", 
    bias_ptr,
    time_tensor,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    my_rank: tl.constexpr, 
    HAS_BIAS: tl.constexpr,
    heap_base_0: tl.constexpr,  
    heap_base_1: tl.constexpr,
    heap_base_2: tl.constexpr,
    heap_base_3: tl.constexpr,
    heap_base_4: tl.constexpr,
    heap_base_5: tl.constexpr,
    heap_base_6: tl.constexpr,
    heap_base_7: tl.constexpr,
    my_rank_base: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    SEND_CTA_NUM: tl.constexpr,
    GROUP_M: tl.constexpr = 4,
    REMAP_XCD: tl.constexpr = True,
    cache_modifier: tl.constexpr = "",
    COLLECT_TIME: tl.constexpr = False,
    SEND_THREAD_NUM: tl.constexpr = 8192 * 2, #8192 * 2,
    SLEEP_CYCLE: tl.constexpr = 2, 
    COMM_PIDs: tl.constexpr = 8 * 3,
):
    if tl.program_id(axis=0) < COMM_PIDs:
        # tl.static_print("COMM_PIDs", COMM_PIDs)
        # tl.static_print("SEND_CTA_NUM", SEND_CTA_NUM)
        # tl.static_assert(SEND_CTA_NUM == 3, "SEND_CTA_NUM must be 3")
        # tl.static_assert(COMM_PIDs == 8 * 3, "COMM_PIDs must be 8 * 3")
        # tl.static_print("HAS_BIAS", HAS_BIAS)
        dest_rank = tl.program_id(axis=0) // SEND_CTA_NUM
        ptr_diff = tl.cast(heap_base_0, tl.int64)
        if dest_rank == 0: 
            ptr_diff = tl.cast(heap_base_0, tl.int64)
        if dest_rank == 1:
            ptr_diff = tl.cast(heap_base_1, tl.int64)
        if dest_rank == 2:
            ptr_diff = tl.cast(heap_base_2, tl.int64)
        if dest_rank == 3:
            ptr_diff = tl.cast(heap_base_3, tl.int64)
        if dest_rank == 4:
            ptr_diff = tl.cast(heap_base_4, tl.int64)
        if dest_rank == 5:
            ptr_diff = tl.cast(heap_base_5, tl.int64)
        if dest_rank == 6:
            ptr_diff = tl.cast(heap_base_6, tl.int64)
        if dest_rank == 7:
            ptr_diff = tl.cast(heap_base_7, tl.int64)
        SIGNAL_POS= 2 * 8192 * 8192 // 2
        offset_am = tl.arange(0, SEND_THREAD_NUM)
        # tl.static_print(f"K*M", K*M, K, M, SEND_CTA_NUM, SEND_THREAD_NUM)
        for i in range(SEND_THREAD_NUM * (tl.program_id(0) % SEND_CTA_NUM), K * M // 8, SEND_THREAD_NUM * SEND_CTA_NUM):
            # tl.device_print(f"RANK={my_rank} load one", i)
            val = tl.load(tl.multiple_of(A_ptr + A_index, [16]) + i + offset_am, cache_modifier=".cg")
            # tl.device_print(f"RANK={my_rank} store one", i)
            tl.store(tl.multiple_of(A_ptr + ptr_diff + i + my_rank * K * (M // 8), [16]) + offset_am, val, cache_modifier=".wt")
        # tl.device_print(f"RANK={my_rank} flag add=",SIGNAL_POS + signal_index + my_rank * 32)

        tl.atomic_add(tl.cast(A_ptr + ptr_diff, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + my_rank * 32, 1, sem="release", scope="sys")

        # tl.device_print(f"RANK={my_rank} set flag result=", ret)
        return
    else: 
        # pid = tl.program_id(0) - COMM_PIDs
        tl.static_assert(K % SPLITK_BLOCK_SIZE == 0, "K must be divisible by SPLITK_BLOCK_SIZE")
        tl.static_assert(SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0, "SPLITK_BLOCK_SIZE must be divisible by BLOCK_K")
        GRID_MN = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
        
        pid_unified = tl.program_id(axis=0) - COMM_PIDs
        # pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
        pid_k = pid_unified % NUM_KSPLIT # 0
        # tl.device_assert(pid_k == 0, "pid_k must be 0")
        pid = pid_unified // NUM_KSPLIT # equal to pid
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(pid_k >= 0)
        USE_FAST_LOAD: tl.constexpr = True

        # tl.static_print(f"SPLITK_BLOCK_SIZE: ", SPLITK_BLOCK_SIZE)
        split_k_start = pid_k * SPLITK_BLOCK_SIZE # 0
        SIGNAL_POS= 2 * 8192 * 8192 // 2
        if split_k_start < K: 
            for rank in range(8):
                flag_ptr = tl.cast(A_ptr + my_rank_base, tl.pointer_type(tl.int32)) + SIGNAL_POS + signal_index + rank * 32
                if USE_FAST_LOAD:
                    result = tl.load(flag_ptr, cache_modifier=".cv")
                else:
                    result = tl.atomic_add(flag_ptr, 0, sem="release", scope="sys")
                i = 0
                while result != SEND_CTA_NUM:
                    i += 1
                    for j in range(SLEEP_CYCLE):
                        device_sleep()
                    if USE_FAST_LOAD:
                        result = tl.load(flag_ptr, cache_modifier=".cv")
                    else:
                        result = tl.atomic_add(flag_ptr, 0, sem="release", scope="sys")
                    if i % 1000000000 == 0: 
                        tl.device_print(f"RANK={my_rank} flag result=", result, rank)
            A_ptr += my_rank_base
                        

            # Create pointers for first block of A and B input matrices
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            offs_k_split = split_k_start + offs_k
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N # here should have an even k condition to optimize that right ?

            a_ptrs = A_ptr + (offs_am[:, None] * K + offs_k_split[None, :] * 1)
            b_ptrs = b_ptr + (offs_k_split[:, None] * 1 + offs_bn[None, :] * K)

            acc_dtype = tl.float32
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

            split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
            k_span = split_k_end - split_k_start
            num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

            for k in range(num_k_iter):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier) # ".cg" when M = 64...
                accumulator += tl.dot(a, b, input_precision="ieee")
                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * 1
                b_ptrs += BLOCK_SIZE_K * 1

            if HAS_BIAS:
                if pid_k == 0:
                    idx_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
                    bias = tl.load(bias_ptr + idx_n, mask = idx_n < N)
                    accumulator += bias.to(tl.float32)

            c = accumulator.to(c_ptr.type.element_ty)

            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + N * offs_cm[:, None] + 1 * offs_cn[None, :] + pid_k * (M*N)
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".cg")
    
def launch_triton_split_k(input, weight, bias, M, K): 
    NUM_SPLIT = 8
    global base_addrs, GLOBAL_SEND_INDEX, GLOBAL_A_ptr_index_hack
    # if False:
    if launch_triton_split_k.cache is not None:
        # log("fast one here...")
        ret, grid, ret1, grid1, size = launch_triton_split_k.cache
        y_pp = torch.empty(size, dtype=torch.bfloat16, device=input.device) 
        ret._run.launch(
            grid[0], grid[1], grid[2], 
            0,
            ret.function,
            None,
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, 
            None, 
            GLOBAL_A_ptr_index_hack.data_ptr(),
            (input.data_ptr() - GLOBAL_A_ptr_index_hack.data_ptr())//2,
            weight.data_ptr(),
            y_pp.data_ptr(),
            GLOBAL_SEND_INDEX,
            bias.data_ptr(),
        )
        GLOBAL_SEND_INDEX += 256
        # log("fast one passed here..", GLOBAL_SEND_INDEX)
        y = torch.empty((size[1], size[2]), dtype=torch.bfloat16, device=input.device)
        ret1._run.launch(
            grid1[0], grid1[1], grid1[2], 
            0,
            ret1.function,
            None,
            ret1.packed_metadata, 
            LazyDict({"name": ret1.name, "function": ret1.function, "stream": 0}), 
            None, 
            None, 
            y_pp.data_ptr(),
            y.data_ptr(), 
        )
        return y

    load_heap_base_ptr()
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_M': 1, 'num_warps': 4, 'num_stages': 2, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': NUM_SPLIT, 'kpack': 1, 'SPLITK_BLOCK_SIZE': K//NUM_SPLIT} # (64,2304,7168):
    M = M * 8
    N = weight.shape[0]
    y_pp = torch.empty((config['NUM_KSPLIT'], M, N), dtype=torch.bfloat16, device=input.device) 

    LOCAL_SEND_CTA_PER_DEVICE = 3


    grid = (8*LOCAL_SEND_CTA_PER_DEVICE + config["NUM_KSPLIT"] * triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]), 1, 1)
    ret = _gemm_a16_w16_split_kernel[grid](
        A_ptr = GLOBAL_A_ptr_index_hack, 
        A_index = (input.data_ptr() - GLOBAL_A_ptr_index_hack.data_ptr())//2,
        b_ptr = weight,
        c_ptr = y_pp,
        bias_ptr = bias,
        M=M,
        N=N,
        K=K,
        heap_base_0=base_addrs[0],
        heap_base_1=base_addrs[1],
        heap_base_2=base_addrs[2],
        heap_base_3=base_addrs[3],
        heap_base_4=base_addrs[4],
        heap_base_5=base_addrs[5],
        heap_base_6=base_addrs[6],
        heap_base_7=base_addrs[7],
        my_rank_base=base_addrs[GLOBAL_MY_RANK],
        my_rank = GLOBAL_MY_RANK,
        SEND_CTA_NUM = LOCAL_SEND_CTA_PER_DEVICE,
        time_tensor=None,
        signal_index=GLOBAL_SEND_INDEX,
        HAS_BIAS=bias is not None,
        SEND_THREAD_NUM=2048,
        **config
    )
    
    torch.cuda.synchronize()
    
    GLOBAL_SEND_INDEX += 256
    y = torch.empty((M, N), dtype=torch.bfloat16, device=input.device)
    
    BS = 512
    # print("y_pp", y_pp)
    grid_reduce = (triton.cdiv(M*N, BS), 1,  1)
    ret2 = _gemm_a16w16_reduce_kernel_optimized[grid_reduce](
        y_pp,
        y, 
        M*N,
        config['NUM_KSPLIT'],
        BS, 
    ) 
    launch_triton_split_k.cache = (ret, grid, ret2, grid_reduce, (NUM_SPLIT, M, N))
    return y
launch_triton_split_k.cache = None
# def custom_kernel(data: input_t, local_bench: bool = False, local_ret = None) -> output_t:
#     if True:
#         input, weight, bias = data
#         local_M, K = input.shape
#         N = weight.shape[0]
#         M = local_M * 8
#         if (M, N, K) not in online_config_keys:
#             return origin(input, weight, bias, M, K)
#         if not GLOBAL_IS_INIT:
#             global_init()
#         if M == 64:
#             ret = launch_triton_split_k(input, weight, bias, M, N, K)
#             return ret

        
#         ret = launch_triton_kernel(input, weight, bias,  get_rank(input))
        
#         return ret
# origin_create_process = dist.init_process_group
def global_init():
    log("11 here")
    # origin_create_process(*args, **kwargs)
    global GLOBAL_BASE_PTR, GLOBAL_MY_RANK, GLOBAL_SEND_INDEX, GLOBAL_IS_INIT
    GLOBAL_MY_RANK = dist.get_rank()
    GLOBAL_BASE_PTR = load_heap_base_ptr(GLOBAL_MY_RANK)
    GLOBAL_SEND_INDEX = 0
    GLOBAL_IS_INIT = True
    log("22 here")
    launch_triton_split_k.cache=None
    

# dist.init_process_group = patch_init_process_group

origin_destroy_process = dist.destroy_process_group

def patch_destroy_process_group():
    global GLOBAL_A_ptr_index_hack, GLOBAL_IS_INIT
    GLOBAL_IS_INIT = False
    tic=time.time()
    global pre_compile_cache
    pre_compile_cache = None
    launch_triton_split_k.cache=None
    if OPEN_PERF: 
        global time_tensors_save, time_tensors_bank
        time_tensors_bank.clear()
        for index,i in enumerate(time_tensors_save):
            pickle.dump(i.cpu(), open(f"time_tensor_rank{dist.get_rank()}_{index}.pkl", "wb"))
    
    GLOBAL_A_ptr_index_hack = None

    close_heap_base_ptr()
    log("clear cost", time.time() - tic)
    origin_destroy_process()

dist.destroy_process_group = patch_destroy_process_group
