CUDA_BIN ?= /usr/local/cuda-13.2/bin
HOST_CXX ?= /usr/bin/g++-14
NVCC = $(CUDA_BIN)/nvcc -ccbin $(HOST_CXX)

.PHONY: all clean
all: message_passing_device

message_passing_device: message_passing_device.cu
	$(NVCC) -std=c++17 -O3 -arch=sm_86 -lineinfo -o $@ $<

clean:
	rm -f message_passing_device
