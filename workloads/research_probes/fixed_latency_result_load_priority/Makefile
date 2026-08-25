CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

LONG_VARIANTS = \
	fixed_latency_result_load_priority_same_noncollision.cubin \
	fixed_latency_result_load_priority_same_collision.cubin \
	fixed_latency_result_load_priority_opposite_noncollision.cubin \
	fixed_latency_result_load_priority_opposite_shifted.cubin
TRACE_VARIANTS = \
	fixed_latency_result_load_priority_trace_same_noncollision.cubin \
	fixed_latency_result_load_priority_trace_same_collision.cubin \
	fixed_latency_result_load_priority_trace_opposite_noncollision.cubin \
	fixed_latency_result_load_priority_trace_opposite_shifted.cubin

.PHONY: all clean

all: fixed_latency_result_load_priority_driver $(LONG_VARIANTS) $(TRACE_VARIANTS)

fixed_latency_result_load_priority_native.cubin: fixed_latency_result_load_priority.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_result_load_priority_trace_native.cubin: fixed_latency_result_load_priority.cu
	$(NVCC) $(NVCC_FLAGS) -DRESULT_LOAD_PRIORITY_GENERATIONS=64 -cubin -o $@ $<

fixed_latency_result_load_priority_driver: fixed_latency_result_load_priority_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_result_load_priority_native.control_bits.json: fixed_latency_result_load_priority_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

fixed_latency_result_load_priority_trace_native.control_bits.json: fixed_latency_result_load_priority_trace_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

$(LONG_VARIANTS) fixed_latency_result_load_priority.patch.json &: \
		fixed_latency_result_load_priority_native.cubin \
		fixed_latency_result_load_priority_native.control_bits.json \
		patch_sm86_result_load_priority.py
	$(PYTHON) patch_sm86_result_load_priority.py $< \
		fixed_latency_result_load_priority_native.control_bits.json \
		--prefix fixed_latency_result_load_priority \
		--report fixed_latency_result_load_priority.patch.json

$(TRACE_VARIANTS) fixed_latency_result_load_priority_trace.patch.json &: \
		fixed_latency_result_load_priority_trace_native.cubin \
		fixed_latency_result_load_priority_trace_native.control_bits.json \
		patch_sm86_result_load_priority.py
	$(PYTHON) patch_sm86_result_load_priority.py $< \
		fixed_latency_result_load_priority_trace_native.control_bits.json \
		--prefix fixed_latency_result_load_priority_trace \
		--report fixed_latency_result_load_priority_trace.patch.json

clean:
	rm -f fixed_latency_result_load_priority_driver \
		fixed_latency_result_load_priority_native.cubin \
		fixed_latency_result_load_priority_trace_native.cubin \
		$(LONG_VARIANTS) $(TRACE_VARIANTS) *.control_bits.json *.patch.json
