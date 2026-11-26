# Build for Linux / the Coursera lab.
# The CUDA kernels in src/kernels/fluid.cu are compiled at runtime by NVRTC,
# so the program itself is plain C++ linked to libcuda (driver API) and
# libnvrtc - no nvcc step is needed.
#   make            build bin/gpu_fluid
#   make run        all experiments (same as ./run.sh)
#   make demo       quick 512x512 run with frames
#   make lint       cpplint (Google C++ style) on src/
#   make format     clang-format -i on src/
#   make clean

CUDA_PATH ?= /usr/local/cuda
CXX       ?= g++

CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -ffp-contract=off
INCLUDES  := -I. -isystem third_party -I$(CUDA_PATH)/include
# libcuda.so comes with the driver; the stubs folder lets it link on machines
# where only the toolkit is installed.
LDFLAGS   := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs -lnvrtc -lcuda

SRCS      := $(wildcard src/*.cc)
OBJS      := $(patsubst src/%.cc,build/%.o,$(SRCS))
TARGET    := bin/gpu_fluid

.PHONY: all run demo lint format clean

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/%.o: src/%.cc src/*.h | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin build:
	mkdir -p $@

run: $(TARGET)
	./run.sh

demo: $(TARGET)
	./$(TARGET) --grid=512 --steps=600 --frame_every=4 --output_dir=output/quick

lint:
	cpplint --recursive src

format:
	clang-format -i src/*.cc src/*.h src/kernels/*.cu

clean:
	rm -rf build bin
