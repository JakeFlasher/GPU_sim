CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: barrier_pipeline barrier_pipeline_trace multi_cta_barrier multi_cta_barrier_trace

barrier_pipeline: barrier_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

barrier_pipeline_trace: barrier_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -DBARRIER_PIPELINE_GENERATIONS=64 -o $@ $<

multi_cta_barrier: multi_cta_barrier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

multi_cta_barrier_trace: multi_cta_barrier.cu
	$(NVCC) $(NVCC_FLAGS) -DMULTI_CTA_BARRIER_GENERATIONS=64 -o $@ $<

clean:
	rm -f barrier_pipeline barrier_pipeline_trace multi_cta_barrier multi_cta_barrier_trace
