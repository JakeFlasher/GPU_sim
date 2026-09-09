CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: l2_queue_frontier l2_queue_frontier_trace

l2_queue_frontier: l2_queue_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Keep a bounded-generation twin for NVBit capture.  The source and SASS
# remain identical apart from the compile-time default generation bound.
l2_queue_frontier_trace: l2_queue_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -DL2_QUEUE_TRACE_BUILD=1 -o $@ $<

clean:
	rm -f l2_queue_frontier l2_queue_frontier_trace
