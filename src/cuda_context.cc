// Copyright 2026 Parth
#include "src/cuda_context.h"

#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fluid {

bool CuOk(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return true;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  std::printf("CUDA error in %s: %s (%s)\n", what, name ? name : "?",
              text ? text : "?");
  return false;
}

bool NvrtcOk(nvrtcResult result, const char* what) {
  if (result == NVRTC_SUCCESS) return true;
  std::printf("NVRTC error in %s: %s\n", what, nvrtcGetErrorString(result));
  return false;
}

CudaContext::~CudaContext() {
  if (context_ != nullptr) cuDevicePrimaryCtxRelease(device_);
}

bool CudaContext::Init(int device) {
  if (!CuOk(cuInit(0), "cuInit")) return false;
  int count = 0;
  if (!CuOk(cuDeviceGetCount(&count), "cuDeviceGetCount")) return false;
  if (device >= count) {
    std::printf("error: device %d asked for but only %d found\n", device,
                count);
    return false;
  }
  if (!CuOk(cuDeviceGet(&device_, device), "cuDeviceGet") ||
      !CuOk(cuDevicePrimaryCtxRetain(&context_, device_), "ctx retain") ||
      !CuOk(cuCtxSetCurrent(context_), "cuCtxSetCurrent")) {
    return false;
  }

  char name[256] = {};
  int sms = 0;
  int l2_bytes = 0;
  int driver = 0;
  size_t memory = 0;
  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  cuDeviceGetName(name, sizeof(name), device_);
  cuDeviceGetAttribute(&cc_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       device_);
  cuDeviceGetAttribute(&cc_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       device_);
  cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_);
  cuDeviceGetAttribute(&l2_bytes, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_);
  cuDeviceTotalMem(&memory, device_);
  cuDriverGetVersion(&driver);
  nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  name_ = name;

  std::printf("GPU            : %s (sm_%d%d, %.1f GB, %d SMs, %d MB L2)\n",
              name, cc_major_, cc_minor_, memory / (1024.0 * 1024.0 * 1024.0),
              sms, l2_bytes / (1024 * 1024));
  std::printf("CUDA driver    : %d.%d\n", driver / 1000, (driver % 1000) / 10);
  std::printf("NVRTC          : %d.%d\n", nvrtc_major, nvrtc_minor);
  return true;
}

DeviceArray::DeviceArray(size_t bytes) : bytes_(bytes) {
  if (!CuOk(cuMemAlloc(&ptr_, bytes), "cuMemAlloc")) ptr_ = 0;
}

DeviceArray::~DeviceArray() {
  if (ptr_ != 0) cuMemFree(ptr_);
}

void DeviceArray::Swap(DeviceArray* other) {
  const CUdeviceptr ptr = ptr_;
  const size_t bytes = bytes_;
  ptr_ = other->ptr_;
  bytes_ = other->bytes_;
  other->ptr_ = ptr;
  other->bytes_ = bytes;
}

bool DeviceArray::Upload(const void* host) {
  return CuOk(cuMemcpyHtoD(ptr_, host, bytes_), "cuMemcpyHtoD");
}

bool DeviceArray::Download(void* host) const {
  return CuOk(cuMemcpyDtoH(host, ptr_, bytes_), "cuMemcpyDtoH");
}

bool DeviceArray::Zero() {
  return CuOk(cuMemsetD8(ptr_, 0, bytes_), "cuMemsetD8");
}

KernelModule::~KernelModule() {
  if (module_ != nullptr) cuModuleUnload(module_);
}

bool KernelModule::Build(const std::string& source_path, int cc_major,
                         int cc_minor) {
  std::ifstream file(source_path);
  if (!file) {
    std::printf(
        "error: cannot open kernel file %s (run from the repo root "
        "or pass --kernel_file)\n",
        source_path.c_str());
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string source = buffer.str();

  const auto start = std::chrono::steady_clock::now();
  nvrtcProgram program = nullptr;
  if (!NvrtcOk(nvrtcCreateProgram(&program, source.c_str(), "fluid.cu", 0,
                                  nullptr, nullptr),
               "nvrtcCreateProgram")) {
    return false;
  }

  // Real cubin for this exact GPU (no PTX JIT at load time). FMA contraction
  // is off so the GPU does the same float ops in the same order as the CPU
  // reference, which keeps the GPU/CPU comparison tight.
  const std::string arch = "--gpu-architecture=sm_" + std::to_string(cc_major) +
                           std::to_string(cc_minor);
  const char* options[] = {arch.c_str(), "--std=c++17", "--fmad=false"};
  const nvrtcResult compiled = nvrtcCompileProgram(program, 3, options);

  size_t log_size = 0;
  nvrtcGetProgramLogSize(program, &log_size);
  if (log_size > 1) {
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(program, &log[0]);
    std::printf("NVRTC log:\n%s\n", log.c_str());
  }
  if (!NvrtcOk(compiled, "nvrtcCompileProgram")) {
    nvrtcDestroyProgram(&program);
    return false;
  }

  nvrtcGetCUBINSize(program, &cubin_bytes_);
  std::vector<char> cubin(cubin_bytes_);
  const bool got = NvrtcOk(nvrtcGetCUBIN(program, cubin.data()), "GetCUBIN");
  nvrtcDestroyProgram(&program);
  if (!got) return false;

  if (!CuOk(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData")) {
    return false;
  }
  compile_ms_ = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
  std::printf("kernels        : %s -> %s cubin, %zu bytes, %.0f ms\n",
              source_path.c_str(), arch.substr(arch.find('=') + 1).c_str(),
              cubin_bytes_, compile_ms_);
  return true;
}

CUfunction KernelModule::Get(const char* name) const {
  CUfunction fn = nullptr;
  if (!CuOk(cuModuleGetFunction(&fn, module_, name), name)) return nullptr;
  return fn;
}

EventTimeline::EventTimeline(int marks) : events_(marks, nullptr) {
  for (CUevent& event : events_) {
    CuOk(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
  }
}

EventTimeline::~EventTimeline() {
  for (CUevent event : events_) {
    if (event != nullptr) cuEventDestroy(event);
  }
}

bool EventTimeline::Mark(int index) {
  return CuOk(cuEventRecord(events_[index], nullptr), "cuEventRecord");
}

bool EventTimeline::Sync() {
  return CuOk(cuEventSynchronize(events_.back()), "cuEventSynchronize");
}

float EventTimeline::Ms(int from, int to) const {
  float ms = 0.0f;
  cuEventElapsedTime(&ms, events_[from], events_[to]);
  return ms;
}

}  // namespace fluid
