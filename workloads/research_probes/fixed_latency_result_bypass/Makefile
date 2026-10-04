CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

LONG_VARIANTS = \
	fixed_latency_result_bypass_fixed_bypass.cubin \
	fixed_latency_result_bypass_fixed_retired.cubin \
	fixed_latency_result_bypass_global_frontier.cubin \
	fixed_latency_result_bypass_global_late.cubin
CALIBRATION_VARIANTS = \
	fixed_latency_result_bypass_fixed_invalid.cubin \
	fixed_latency_result_bypass_canary_d4.cubin \
	fixed_latency_result_bypass_canary_d5.cubin

.PHONY: all clean

all: fixed_latency_result_bypass_driver fixed_latency_result_bypass_canary_driver \
	$(LONG_VARIANTS) $(CALIBRATION_VARIANTS)

fixed_latency_result_bypass_native.cubin: fixed_latency_result_bypass.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_result_bypass_driver: fixed_latency_result_bypass_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_result_bypass_canary_driver: fixed_latency_result_bypass_canary_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_result_bypass_native.control_bits.json: fixed_latency_result_bypass_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ --toolkit-bindir $(CUDA_BIN)

$(LONG_VARIANTS) fixed_latency_result_bypass.patch.json &: \
		fixed_latency_result_bypass_native.cubin \
		fixed_latency_result_bypass_native.control_bits.json \
		patch_sm86_bypass_pairs.py
	$(PYTHON) patch_sm86_bypass_pairs.py $< \
		fixed_latency_result_bypass_native.control_bits.json \
		--prefix fixed_latency_result_bypass \
		--report fixed_latency_result_bypass.patch.json

$(CALIBRATION_VARIANTS) fixed_latency_result_bypass_calibration.patch.json &: \
		fixed_latency_result_bypass_native.cubin \
		fixed_latency_result_bypass_native.control_bits.json \
		patch_sm86_bypass_calibration.py
	$(PYTHON) patch_sm86_bypass_calibration.py $< \
		fixed_latency_result_bypass_native.control_bits.json \
		--prefix fixed_latency_result_bypass \
		--report fixed_latency_result_bypass_calibration.patch.json

clean:
	rm -f fixed_latency_result_bypass_driver fixed_latency_result_bypass_canary_driver \
		fixed_latency_result_bypass_native.cubin \
		$(LONG_VARIANTS) $(CALIBRATION_VARIANTS) \
		*.control_bits.json *.patch.json
