CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

.PHONY: all clean

all: fixed_latency_completion_driver \
	fixed_latency_completion_split.cubin fixed_latency_completion_bank3.cubin \
	fixed_latency_completion_trace_split.cubin fixed_latency_completion_trace_bank3.cubin

fixed_latency_completion_driver: fixed_latency_completion_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_completion_native.cubin: fixed_latency_completion.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_completion_trace_native.cubin: fixed_latency_completion.cu
	$(NVCC) $(NVCC_FLAGS) -DFIXED_COMPLETION_GENERATIONS=64 -cubin -o $@ $<

fixed_latency_completion_native.control_bits.json: fixed_latency_completion_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

fixed_latency_completion_trace_native.control_bits.json: fixed_latency_completion_trace_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

fixed_latency_completion_split.cubin fixed_latency_completion_bank3.cubin &: \
		fixed_latency_completion_native.cubin fixed_latency_completion_native.control_bits.json \
		patch_sm86_completion_layouts.py
	$(PYTHON) patch_sm86_completion_layouts.py \
		fixed_latency_completion_native.cubin \
		fixed_latency_completion_native.control_bits.json \
		fixed_latency_completion_split.cubin \
		fixed_latency_completion_bank3.cubin \
		--report fixed_latency_completion.patch.json

fixed_latency_completion_trace_split.cubin fixed_latency_completion_trace_bank3.cubin &: \
		fixed_latency_completion_trace_native.cubin fixed_latency_completion_trace_native.control_bits.json \
		patch_sm86_completion_layouts.py
	$(PYTHON) patch_sm86_completion_layouts.py \
		fixed_latency_completion_trace_native.cubin \
		fixed_latency_completion_trace_native.control_bits.json \
		fixed_latency_completion_trace_split.cubin \
		fixed_latency_completion_trace_bank3.cubin \
		--report fixed_latency_completion_trace.patch.json

clean:
	rm -f fixed_latency_completion_driver \
		fixed_latency_completion_native.cubin fixed_latency_completion_trace_native.cubin \
		fixed_latency_completion_split.cubin fixed_latency_completion_bank3.cubin \
		fixed_latency_completion_trace_split.cubin fixed_latency_completion_trace_bank3.cubin \
		*.control_bits.json *.patch.json

