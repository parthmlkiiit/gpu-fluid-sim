// Copyright 2026 Parth
// PNG output (stb_image_write), a background frame writer and folder
// creation.

#ifndef SRC_IMAGE_WRITER_H_
#define SRC_IMAGE_WRITER_H_

#include <condition_variable>  // NOLINT(build/c++11)
#include <cstddef>
#include <deque>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

namespace fluid {

// rgb holds width*height*3 bytes, rows top to bottom.
bool SavePng(const std::string& path, int width, int height,
             const std::vector<unsigned char>& rgb);

bool MakeDirs(const std::string& dir);

// Encodes PNGs on worker threads so the simulation loop (and with it the
// GPU) never sits idle waiting for the encoder.
class FrameWriter {
 public:
  // max_queued bounds memory: Write() blocks while that many frames wait.
  FrameWriter(int threads, size_t max_queued);
  ~FrameWriter();

  FrameWriter(const FrameWriter&) = delete;
  FrameWriter& operator=(const FrameWriter&) = delete;

  // Takes ownership of the pixels and queues the frame.
  void Write(const std::string& path, int width, int height,
             std::vector<unsigned char> rgb);

  // Writes everything still queued and stops the workers. Returns false if
  // any frame failed. Called by the destructor if not called before.
  bool Finish();

  int threads() const { return static_cast<int>(workers_.size()); }

 private:
  struct Job {
    std::string path;
    int width;
    int height;
    std::vector<unsigned char> rgb;
  };

  void WorkerLoop();

  std::mutex mutex_;
  std::condition_variable has_work_;
  std::condition_variable has_room_;
  std::deque<Job> queue_;
  std::vector<std::thread> workers_;
  size_t max_queued_;
  bool closing_ = false;
  bool failed_ = false;
};

}  // namespace fluid

#endif  // SRC_IMAGE_WRITER_H_
