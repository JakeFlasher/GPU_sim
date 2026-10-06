CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC_CCBIN ?= /usr/bin/g++-14
NVCC := $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py
VARIANTS = baseline_s0 \
	stall4_s0_i0 stall4_s0_i1 stall4_s0_i2 stall4_s0_i3 \
	stall4_s0_i4 stall4_s0_i5 stall4_s0_i6 stall4_s0_i7 \
	yield_s0_i0 yield_s0_i1 yield_s0_i2 yield_s0_i3 \
	yield_s0_i4 yield_s0_i5 yield_s0_i6 yield_s0_i7 \
	baseline_s1 \
	stall4_s1_i0 stall4_s1_i1 stall4_s1_i2 stall4_s1_i3 \
	stall4_s1_i4 stall4_s1_i5 stall4_s1_i6 stall4_s1_i7 \
	yield_s1_i0 yield_s1_i1 yield_s1_i2 yield_s1_i3 \
	yield_s1_i4 yield_s1_i5 yield_s1_i6 yield_s1_i7
VARIANT_CUBINS = $(foreach v,$(VARIANTS),front_end_ibuffer_depth_$(v).cubin)

.PHONY: all clean

all: front_end_ibuffer_depth_driver front_end_ibuffer_depth_native.cubin \
	front_end_ibuffer_depth_native.control_bits.json $(VARIANT_CUBINS)

front_end_ibuffer_depth_driver: front_end_ibuffer_depth_driver.cpp
	$(NVCC) -ccbin $(NVCC_CCBIN) -std=c++17 -O3 -o $@ $< -lcuda

front_end_ibuffer_depth_native.cubin: front_end_ibuffer_depth.cu
	$(NVCC) -ccbin $(NVCC_CCBIN) $(NVCC_FLAGS) -cubin -o $@ $<

front_end_ibuffer_depth_native.control_bits.json: front_end_ibuffer_depth_native.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ \
		--toolkit-bindir $(CUDA_BIN)

$(VARIANT_CUBINS) front_end_ibuffer_depth.patch.json &: \
		front_end_ibuffer_depth_native.cubin \
		front_end_ibuffer_depth_native.control_bits.json \
		patch_sm86_ibuffer_controls.py
	$(PYTHON) patch_sm86_ibuffer_controls.py \
		front_end_ibuffer_depth_native.cubin \
		front_end_ibuffer_depth_native.control_bits.json \
		--prefix front_end_ibuffer_depth \
		--report front_end_ibuffer_depth.patch.json

clean:
	rm -f front_end_ibuffer_depth_driver front_end_ibuffer_depth_native.cubin \
		front_end_ibuffer_depth_native.control_bits.json $(VARIANT_CUBINS) \
		front_end_ibuffer_depth_zero.cubin \
		front_end_ibuffer_depth_stall4_once.cubin \
		front_end_ibuffer_depth_yield_once.cubin \
		front_end_ibuffer_depth_stall4_repeat.cubin \
		front_end_ibuffer_depth_yield_repeat.cubin \
		front_end_ibuffer_depth_stall4.cubin \
		front_end_ibuffer_depth_yield.cubin *.patch.json
