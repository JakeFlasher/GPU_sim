CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: task_done_boundary task_done_boundary_trace

task_done_boundary: task_done_boundary.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

task_done_boundary_trace: task_done_boundary.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f task_done_boundary task_done_boundary_trace
