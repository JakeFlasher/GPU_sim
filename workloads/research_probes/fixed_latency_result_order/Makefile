CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC_CCBIN ?= /usr/bin/g++-14
NVCC := $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

.PHONY: all clean

all: fixed_latency_result_order_driver fixed_latency_result_order.cubin

fixed_latency_result_order_native.cubin: fixed_latency_result_order.cu
	$(NVCC) -ccbin $(NVCC_CCBIN) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_result_order_driver: fixed_latency_result_order_driver.cpp
	$(NVCC) -ccbin $(NVCC_CCBIN) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_result_order_native.control_bits.json: fixed_latency_result_order_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

fixed_latency_result_order.cubin: fixed_latency_result_order_native.cubin \
		fixed_latency_result_order_native.control_bits.json patch_sm86_result_order.py
	$(PYTHON) patch_sm86_result_order.py $< \
		fixed_latency_result_order_native.control_bits.json $@ \
		--report fixed_latency_result_order.patch.json

clean:
	rm -f fixed_latency_result_order_driver fixed_latency_result_order_native.cubin \
		fixed_latency_result_order.cubin *.control_bits.json *.patch.json
