// Copyright 2026 Parth
// Thin layer over the CUDA driver API and NVRTC: device setup, device memory,
// runtime kernel compilation, kernel launches and event timing.

#ifndef SRC_CUDA_CONTEXT_H_
#define SRC_CUDA_CONTEXT_H_

#include <cuda.h>
#include <nvrtc.h>

#include <cstddef>
#include <string>
#include <vector>

namespace fluid {

// Print the failing call and return false so callers can bail out.
bool CuOk(CUresult result, const char* what);
bool NvrtcOk(nvrtcResult result, const char* what);

// Owns the primary context of one device for the life of the program.
class CudaContext {
 public:
  CudaContext() = default;
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  // Picks the device, makes its primary context current and prints it.
  bool Init(int device);

  int cc_major() const { return cc_major_; }
  int cc_minor() const { return cc_minor_; }
  const std::string& name() const { return name_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  int cc_major_ = 0;
  int cc_minor_ = 0;
  std::string name_;
};

// cuMemAlloc'd device memory, freed when it goes out of scope.
class DeviceArray {
 public:
  explicit DeviceArray(size_t bytes);
  ~DeviceArray();

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  bool ok() const { return ptr_ != 0; }
  CUdeviceptr get() const { return ptr_; }
  size_t bytes() const { return bytes_; }

  // Swaps the buffers (used for ping-pong between two fields).
  void Swap(DeviceArray* other);

  bool Upload(const void* host);
  bool Download(void* host) const;
  bool Zero();

 private:
  CUdeviceptr ptr_ = 0;
  size_t bytes_ = 0;
};

// Kernels compiled at runtime from a .cu file with NVRTC.
class KernelModule {
 public:
  KernelModule() = default;
  ~KernelModule();

  KernelModule(const KernelModule&) = delete;
  KernelModule& operator=(const KernelModule&) = delete;

  // Compiles source_path for sm_<major><minor> and loads the cubin.
  bool Build(const std::string& source_path, int cc_major, int cc_minor);

  // Looks a kernel up by its extern "C" name, nullptr if missing.
  CUfunction Get(const char* name) const;

  double compile_ms() const { return compile_ms_; }
  size_t cubin_bytes() const { return cubin_bytes_; }

 private:
  CUmodule module_ = nullptr;
  double compile_ms_ = 0.0;
  size_t cubin_bytes_ = 0;
};

constexpr int kBlock = 16;

// Launches fn over an n x n grid with 16x16 blocks on the default stream.
// Each argument is copied, and cuLaunchKernel reads them through pointers,
// so the types must match the kernel signature exactly (CUdeviceptr for
// pointers, int, float, structs by value).
template <typename... Args>
bool Launch2D(CUfunction fn, int n, Args... args) {
  void* params[] = {&args...};
  const unsigned int blocks = (n + kBlock - 1) / kBlock;
  return CuOk(cuLaunchKernel(fn, blocks, blocks, 1, kBlock, kBlock, 1, 0,
                             nullptr, params, nullptr),
              "cuLaunchKernel");
}

// A row of CUDA events; Mark(i) records event i, Ms(a, b) gives the time
// between two of them once Sync() has returned.
class EventTimeline {
 public:
  explicit EventTimeline(int marks);
  ~EventTimeline();

  EventTimeline(const EventTimeline&) = delete;
  EventTimeline& operator=(const EventTimeline&) = delete;

  bool Mark(int index);
  bool Sync();
  float Ms(int from, int to) const;

 private:
  std::vector<CUevent> events_;
};

}  // namespace fluid

#endif  // SRC_CUDA_CONTEXT_H_
