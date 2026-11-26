// Copyright 2026 Parth
#include "src/image_writer.h"

#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17): portable mkdir -p
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace fluid {

bool SavePng(const std::string& path, int width, int height,
             const std::vector<unsigned char>& rgb) {
  const int ok =
      stbi_write_png(path.c_str(), width, height, 3, rgb.data(), width * 3);
  if (!ok) std::printf("cannot write %s\n", path.c_str());
  return ok != 0;
}

bool MakeDirs(const std::string& dir) {
  std::error_code err;
  std::filesystem::create_directories(dir, err);
  if (err) {
    std::printf("cannot create %s: %s\n", dir.c_str(), err.message().c_str());
    return false;
  }
  return true;
}

FrameWriter::FrameWriter(int threads, size_t max_queued)
    : max_queued_(max_queued) {
  for (int i = 0; i < threads; ++i) {
    workers_.emplace_back(&FrameWriter::WorkerLoop, this);
  }
}

FrameWriter::~FrameWriter() { Finish(); }

void FrameWriter::Write(const std::string& path, int width, int height,
                        std::vector<unsigned char> rgb) {
  std::unique_lock<std::mutex> lock(mutex_);
  has_room_.wait(lock, [this] { return queue_.size() < max_queued_; });
  queue_.push_back(Job{path, width, height, std::move(rgb)});
  has_work_.notify_one();
}

bool FrameWriter::Finish() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
  }
  has_work_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  return !failed_;
}

void FrameWriter::WorkerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      has_work_.wait(lock, [this] { return closing_ || !queue_.empty(); });
      if (queue_.empty()) return;  // closing and nothing left
      job = std::move(queue_.front());
      queue_.pop_front();
    }
    has_room_.notify_one();
    if (!SavePng(job.path, job.width, job.height, job.rgb)) {
      std::lock_guard<std::mutex> lock(mutex_);
      failed_ = true;
    }
  }
}

}  // namespace fluid
