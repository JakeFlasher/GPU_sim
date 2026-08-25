CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: l1tex_arrival_replay_service l1tex_arrival_replay_service_trace

l1tex_arrival_replay_service: l1tex_arrival_replay_service.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

l1tex_arrival_replay_service_trace: l1tex_arrival_replay_service.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f l1tex_arrival_replay_service l1tex_arrival_replay_service_trace

