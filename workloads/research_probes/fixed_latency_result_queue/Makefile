CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

.PHONY: all clean

all: fixed_latency_result_queue fixed_latency_result_queue.cubin \
	fixed_latency_result_queue_driver \
	fixed_latency_result_queue_collision.cubin \
	fixed_latency_result_queue_separated.cubin

fixed_latency_result_queue: fixed_latency_result_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

fixed_latency_result_queue.cubin: fixed_latency_result_queue.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_result_queue_driver: fixed_latency_result_queue_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_result_queue.control_bits.json: fixed_latency_result_queue.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

fixed_latency_result_queue_collision.cubin \
fixed_latency_result_queue_separated.cubin &: \
		fixed_latency_result_queue.cubin fixed_latency_result_queue.control_bits.json \
		patch_sm86_result_pairs.py
	$(PYTHON) patch_sm86_result_pairs.py fixed_latency_result_queue.cubin \
		fixed_latency_result_queue.control_bits.json --prefix fixed_latency_result_queue \
		--report fixed_latency_result_queue.patch.json

clean:
	rm -f fixed_latency_result_queue \
		fixed_latency_result_queue.cubin \
		fixed_latency_result_queue_driver fixed_latency_result_queue_*collision.cubin \
		fixed_latency_result_queue_*separated.cubin \
		*.control_bits.json *.patch.json
