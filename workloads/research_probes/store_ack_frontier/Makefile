CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: store_ack_frontier store_ack_frontier_trace

store_ack_frontier: store_ack_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# The trace binary is compiled from the identical translation unit. The
# separate name keeps capture commands and content identities explicit.
store_ack_frontier_trace: store_ack_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f store_ack_frontier store_ack_frontier_trace
