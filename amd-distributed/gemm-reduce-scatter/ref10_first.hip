// #define ZZ_DEBUG
// #define USE_PERF 1
//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#pragma once
#define DEBUG_COMBINE 0
#define DEBUG_DISPATCH_READY 0
#define DEBUG_DISPATCH_READY2 0
#define DEBUG_COMBINE_READY 0
#define DEBUG_RAW_RET 0
#define CALL_PRINT 0
#define DEBUG_VALUE 0
#define DEBUG_LOCAL_DISPATCH_START_END 0

#ifdef __HIPCC__
// AMD ROCm HIP 平台
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define uni(func) hip##func
#define WARP_SYNC __builtin_amdgcn_wave_barrier();
#elif defined(__CUDACC__)
// NVIDIA CUDA 平台
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define uni(func) cuda##func
#define WARP_SYNC __syncwarp();
#else
#error "Unknown GPU platform"
#endif
namespace zz{

#define _(a, b, c) (a) * (c) + (b)
#define DIV_UP(x, y) (x + y - 1) / y
constexpr int BLOCKSIZE = 128;

#define DEBUG_PRINT 1
#ifndef ZZ
#define ZZ
#endif

static int MYRANK = -1;

#define UNI_CHECK(err)                                                         \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << std::endl;                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define UNI_CHECK_RANK(err, local_rank)                                        \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << " rank: " << local_rank        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
const char *IPC_HANDLE_FILENAME_RANK[] = {
    "ipc_handles_rank0.bin", "ipc_handles_rank1.bin", "ipc_handles_rank2.bin",
    "ipc_handles_rank3.bin", "ipc_handles_rank4.bin", "ipc_handles_rank5.bin",
    "ipc_handles_rank6.bin", "ipc_handles_rank7.bin"};

enum class FLAG {
  WAITING = 0,
  READY = 1,
};
enum class DATA_FLAG {
  WAITING = 0,
  READY = 1,
};



constexpr int RANK_SIZE = 8;


struct IpcCommBlock {
    int barrier[1000000];
};

// maybe we need a double buffer here later?? for the memset to execute in the
// background.

IpcCommBlock *h_remote_data[RANK_SIZE]; // Host端指针数组，固定大小避免动态分配问题
static IpcCommBlock **d_remote_data = nullptr; // Device端指针，指向device内存中的指针数组


template <typename U> __device__ static U load_volatile(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) u1 = __hip_atomic_load( (uint8_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) u2 = __hip_atomic_load((uint16_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) u4 = __hip_atomic_load((uint32_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else                     u8 = __hip_atomic_load((uint64_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return elt;
}

template <typename U> __device__ static void store_volatile(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) __hip_atomic_store( (uint8_t *)dst, u1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) __hip_atomic_store((uint16_t *)dst, u2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) __hip_atomic_store((uint32_t *)dst, u4, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); 
  else                     __hip_atomic_store((uint64_t *)dst, u8, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <typename U> __device__ static void add_volatile(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) __hip_atomic_fetch_add( (uint8_t *)dst, u1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) __hip_atomic_fetch_add((uint16_t *)dst, u2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) __hip_atomic_fetch_add((uint32_t *)dst, u4, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); 
  else                     __hip_atomic_fetch_add((uint64_t *)dst, u8, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static int num_gpus = 0;
int inited = -1;


void* d_local_data = nullptr;

void init_shmem(int my_rank, uint64_t* d_tensor) {
  // fprintf(stderr, "init_shmem for rank: %d\n", my_rank);
  if(inited != -1){
    for(int rank = 0; rank < 8; rank++){
      if(rank == inited) continue;
      UNI_CHECK(hipIpcCloseMemHandle(h_remote_data[rank]));
    }
    UNI_CHECK(hipFree(d_remote_data));
    UNI_CHECK(hipFree(d_local_data));
    // fprintf(stderr, "relased here...\n");
  }
  if(my_rank == -1){
    inited = -1; // clear
    return;
  }
  inited = my_rank;
  MYRANK = my_rank;
  UNI_CHECK(hipGetDeviceCount(&num_gpus));
  UNI_CHECK(hipSetDevice(my_rank));
  constexpr int  alloc_size = 1<< 30;
  UNI_CHECK(hipExtMallocWithFlags(&d_local_data, alloc_size, hipDeviceMallocFinegrained));
  //UNI_CHECK(hipMalloc(&d_local_data, alloc_size));
  UNI_CHECK(hipMemset(d_local_data, 0, alloc_size));
  hipIpcMemHandle_t ipc_handle;
  UNI_CHECK(hipIpcGetMemHandle(&ipc_handle, d_local_data));

  std::string s = std::string(IPC_HANDLE_FILENAME_RANK[my_rank]);
  std::ofstream handle_file(s.c_str(), std::ios::binary);
  handle_file.write(reinterpret_cast<char *>(&ipc_handle),
                    sizeof(ipc_handle));
  handle_file.close();
  // fprintf(stderr, "write ipc handle to %s\n", s.c_str());

  for (int rank = 0; rank < 8; rank++) {
    if(rank == my_rank) {
      h_remote_data[rank] = reinterpret_cast<IpcCommBlock *>(d_local_data);
      continue;
    }
    auto ipc_file_name = IPC_HANDLE_FILENAME_RANK[rank];
    int cnt = 0;
    while (access(ipc_file_name, F_OK) == -1) {
      usleep(100000);
      if (++cnt % 100 == 0)
        fprintf(stderr, "waiting for ipc handle file %s\n", ipc_file_name);
    }
    usleep(500000);
    uni(IpcMemHandle_t) ipc_handle;
    std::ifstream handle_file(ipc_file_name, std::ios::binary);
    handle_file.read(reinterpret_cast<char *>(&ipc_handle),
                     sizeof(ipc_handle));
    handle_file.close();

    UNI_CHECK_RANK(hipIpcOpenMemHandle(reinterpret_cast<void **>(&h_remote_data[rank]), ipc_handle, hipIpcMemLazyEnablePeerAccess), my_rank);
    // Check if h_remote_data[kk][rank] is 4-byte aligned
    if (reinterpret_cast<uintptr_t>(h_remote_data[rank]) % 16 != 0) {
      fprintf(stderr, "Error: h_remote_data[%d][%d] = %p is not 4-byte aligned\n", 
              rank, rank, h_remote_data[rank]);
      abort();
    }
    // if(my_rank == 0){
    //   fprintf(stderr, "h_remote_data[%d][%d]: %p\n", kk, rank, h_remote_data[kk][rank]);

    // }
  }
  UNI_CHECK(hipMalloc(&d_remote_data, sizeof(IpcCommBlock *) * 16));
  UNI_CHECK(hipMemcpy(d_remote_data, &h_remote_data, sizeof(IpcCommBlock *) * 8, hipMemcpyHostToDevice));
  UNI_CHECK(hipMemcpy(d_tensor, &h_remote_data, sizeof(IpcCommBlock *) * 8, hipMemcpyHostToDevice));
}

__global__ void dist_barrier(IpcCommBlock **d_remote_data, int index, int offset) {
  int dst_rank = threadIdx.x / 64;
  if (threadIdx.x % 64 != 0 || dst_rank == index) { return; }

  store_volatile(&d_remote_data[dst_rank]->barrier[offset + index], 1);
  while (true) {
    if (load_volatile(&d_remote_data[index]->barrier[offset + dst_rank]) == 1) {
      break;
    }
  }
}

static int send_kernel_index = 0;
constexpr int start_offset = 2 * 8192 * 8192 / 2  + 10; // TODO: need adjust later...
static int offset = start_offset;

void barrier(int id) {
  UNI_CHECK(uni(SetDevice)(id % num_gpus)); 
  dist_barrier<<<1, 64 * 8>>>(d_remote_data, id, offset);
  offset+=8;// opt later... index can add much.
  UNI_CHECK(uni(GetLastError)());
}


// K is divisible by 64
// only the large 3 kernel is support now

constexpr int NOTI_START_OFFSET = 2 * 8192 * 8192 / 2;

hipStream_t stream[8] = {};

int32_t* d_one_ptr = nullptr;
void init_stream(int my_rank){
  if(my_rank == -1){ 
    send_kernel_index = 0;
    if(d_one_ptr != nullptr){
      UNI_CHECK(hipFree(d_one_ptr));
      d_one_ptr = nullptr;
    }
    for(int i = 0; i < 8; i++){
      if(stream[i] != nullptr){
        // fprintf(stderr, "destroy stream[%d] %p\n", i, stream[i]);
        UNI_CHECK(hipStreamDestroy(stream[i]));
        stream[i] = nullptr;
      }else{
        // fprintf(stderr, "stream[%d] is nullptr\n", i);
      }
    }
    return;
  };
  int leastPriority, greatestPriority;
  UNI_CHECK(hipMalloc(&d_one_ptr, 4));
  int now = 1; 
  UNI_CHECK(hipMemcpy(d_one_ptr, &now, 4, hipMemcpyHostToDevice));
  UNI_CHECK(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  // fprintf(stderr, "Priority range: %d (highest) to %d (lowest)\n", 
  //        greatestPriority, leastPriority);
  UNI_CHECK(hipSetDevice(my_rank));
  for(int i = 0; i < 8; i++){
    if(i == my_rank) continue;
    UNI_CHECK(hipStreamCreate(&stream[i]));
    // if(i == 0) fprintf(stderr, "rank-%d, create stream[%d] %p\n", my_rank, i, stream[i]);
  }
}


void put_kernel(float* source_tensor, int M, int K, int my_rank){

  #if DEBUG_COMM_TIME == 1
    hipEvent_t start_event, end_event;
    UNI_CHECK(hipEventCreate(&start_event));
    UNI_CHECK(hipEventCreate(&end_event));
    UNI_CHECK(hipEventRecord(start_event, stream[my_rank]));
  #endif

  // send_kernel<<<7 * SEND_CTA_PER_DEVICE, SEND_THREAD_NUM_SIZE, 0, stream[my_rank]>>>(reinterpret_cast<float4*>(source_tensor), M, K/8, my_rank, d_remote_data, send_kernel_index, d_noti_ptr[my_rank]);
  // UNI_CHECK(hipEventRecord(end_event, stream[my_rank]));
  // UNI_CHECK(hipEventSynchronize(start_event));
  // UNI_CHECK(hipEventSynchronize(end_event));
   
  // fprintf(stderr, "rank: %d, put_kernel start here...\n", my_rank);
  for(int i = 0; i < 8; i++){
    if(i == my_rank) continue;
    // fprintf(stderr, "rank: %d-%d, memcpy start here... %p\n", my_rank, i, stream[i]);
    UNI_CHECK(hipMemcpyAsync((short*)(h_remote_data[i]) + my_rank * (M * K), source_tensor, M * K * 2, hipMemcpyDeviceToDeviceNoCU, stream[i]));  
    // fprintf(stderr, "rank: %d-%d, memcpy2 start here...\n", my_rank, i);
    UNI_CHECK(hipMemcpyAsync((int*)h_remote_data[i] + NOTI_START_OFFSET + 32 * my_rank  + send_kernel_index, d_one_ptr, 4, hipMemcpyDeviceToDeviceNoCU, stream[i]));  
  }
  // fprintf(stderr, "rank: %d, put_kernel end here...\n", my_rank);
  send_kernel_index+=256;

  #if DEBUG_COMM_TIME == 1
  float time; 
  UNI_CHECK(hipEventDestroy(start_event));
  #endif
  UNI_CHECK(hipGetLastError());
  
  // simple_kernel<<<1, 1, 0, 0>>>(d_noti_ptr[my_rank], send_kernel_index++);
  // Wait for the send_kernel to complete on stream[my_rank]
  // UNI_CHECK(hipEventRecord(end_event, stream[my_rank]));
  // UNI_CHECK(hipStreamWaitEvent(0, start_event, 0));
  // // UNI_CHECK(hipEventSynchronize(end_event));
  // UNI_CHECK(hipEventDestroy(end_event));
  // UNI_CHECK(hipStreamSynchronize(stream[my_rank])); // debug时候用一下...
  // for(int _ =0 ; _ < 100; _++){
  //   auto &stream = stream_list[my_rank];
  //   for(int i = 0; i < 8; i++){
  //     UNI_CHECK(hipStreamSynchronize(stream[i]));
  //     UNI_CHECK(hipMemcpyAsync(h_remote_data[my_rank][i], source_tensor, M * K * 2 /*bf16*/, hipMemcpyDeviceToDeviceNoCU, stream[i]));

  //   }
  // }
  // UNI_CHECK(hipDeviceSynchronize()); // debug时候用一下...
  // sleep(2);
  // std::vector<int> now(NOTI_START_OFFSET + 2000, -1 );
  // sleep(2);
  // if(my_rank == 0){
  //   fprintf(stderr, "M:%d K/8:%d BM: 256\n", M, K/8);
  //   fprintf(stderr, "==============NOTI_START_OFFSET: %d %p\n", NOTI_START_OFFSET, h_remote_data[my_rank][my_rank]);
    
  //   UNI_CHECK(hipMemcpy(now.data(), reinterpret_cast<int*>(h_remote_data[my_rank][my_rank]), now.size() * sizeof(int), hipMemcpyDeviceToHost));
  //   long long is_one_start_pos = -1;
  //   for(long long i = 0; i < now.size();i++){
  //     if(now[i]!=0){
  //       if(is_one_start_pos == -1) is_one_start_pos = i;
  //     }else if(now[i] == 0){
  //       if(is_one_start_pos >=0) fprintf(stderr, "not zero: %lld-%lld\n", i, is_one_start_pos); 
  //       is_one_start_pos = -1;
  //     }
  //     if(i < 3){
  //       fprintf(stderr, "now[%lld]: %d\n", i, now[i]);
  //     }
  //   }
  //   fprintf(stderr, "============NOTI_START_OFFSET: %d\n", NOTI_START_OFFSET);
  // }
  // sleep(1000);
  // fprintf(stderr, "[RANK %d] put_kernel done\n", my_rank);
}
    
void init(int id, uint64_t* tensor) {

    fprintf(stderr, "rank %d init here...\n", id);
    init_shmem(id, tensor);
    // init_stream(id);
}
void clear(){
  init_shmem(-1, nullptr);  
  // init_stream(-1);
}
}