// #include "ref10_first.hip"
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h> // for sleep()
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
bool isFileOlderThan_posix(const char *filePath, double seconds_threshold) {
  struct stat fileStat;

  // 2. 调用 stat 获取文件信息
  if (stat(filePath, &fileStat) != 0) {
    // 如果 stat 返回非0值，表示出错（如文件不存在）
    perror("stat error");
    return false;
  }

  // 1. 获取当前时间
  time_t currentTime = time(nullptr);

  // 3. 计算时间差
  double diff_seconds = difftime(currentTime, fileStat.st_mtime);

  // 4. 比较差值
  if (diff_seconds > seconds_threshold) {
    std::cout << "文件 '" << filePath << "' 的修改时间已超过 "
              << seconds_threshold << " 秒. (实际差距: " << diff_seconds
              << "s)\n";
    return true;
  } else {
    std::cout << "文件 '" << filePath << "' 的修改时间在 " << seconds_threshold
              << " 秒内. (实际差距: " << diff_seconds << "s)\n";
    return false;
  }
}
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
int main() {
  for (int i = 0; i < 8; i++) {
    void *d_local_data = nullptr;
    int device_count;
    UNI_CHECK(hipGetDeviceCount(&device_count));
    int device_id = i % device_count;
    UNI_CHECK(uni(SetDevice)(device_id));
    printf("set device %d\n", device_id);
    constexpr int  alloc_size = 1<< 30;
    // UNI_CHECK(hipMalloc(&d_local_data, 1 << 30));
    UNI_CHECK(hipExtMallocWithFlags(&d_local_data, alloc_size, hipDeviceMallocFinegrained));
    // UNI_CHECK(hipMallocManaged(&d_local_data, 1 << 30, hipMemAttachGlobal));
    UNI_CHECK(uni(Memset)(d_local_data, 0, alloc_size));
    uni(IpcMemHandle_t) ipc_handle;
    UNI_CHECK(uni(IpcGetMemHandle)(&ipc_handle, d_local_data));
    std::string s = std::string(IPC_HANDLE_FILENAME_RANK[i]);
    std::ofstream handle_file(s.c_str(), std::ios::binary);
    handle_file.write(reinterpret_cast<char *>(&ipc_handle),
                      sizeof(ipc_handle));
    handle_file.close();
    fprintf(stderr, "write ipc handle to %s\n", s.c_str());
  }
  fprintf(stderr, "start to sleep now\n");
  const char *filename = "now.txt";
  for (int i = 1; i < 9; i++) {
    sleep(5);
    if (isFileOlderThan_posix(filename, 10.0)) {
      fprintf(stderr, "file is older than 20s, break\n");
      break;
    }
    fprintf(stderr, "%d s sleeped\n ", i * 10);
  }
  puts("end of sleep");
}
