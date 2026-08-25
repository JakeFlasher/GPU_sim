CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

UNROLLS = 1 2 4 8
LONG_NATIVES = $(foreach u,$(UNROLLS),fixed_latency_front_end_footprint_u$(u)_native.cubin)
TRACE_NATIVES = $(foreach u,$(UNROLLS),fixed_latency_front_end_footprint_u$(u)_trace_native.cubin)
NATIVE_SIDECARS = $(LONG_NATIVES:.cubin=.control_bits.json) \
	$(TRACE_NATIVES:.cubin=.control_bits.json)
VARIANTS = fixed_bypass fixed_retired global_frontier global_late
LONG_VARIANTS = $(foreach u,$(UNROLLS),$(foreach v,$(VARIANTS),fixed_latency_front_end_footprint_u$(u)_$(v).cubin))
TRACE_VARIANTS = $(foreach u,$(UNROLLS),$(foreach v,$(VARIANTS),fixed_latency_front_end_footprint_u$(u)_trace_$(v).cubin))

.PHONY: all native clean

all: native fixed_latency_front_end_footprint_driver $(LONG_VARIANTS) \
	$(TRACE_VARIANTS) \
	fixed_latency_front_end_footprint_u1_calibration_fixed_invalid.cubin \
	fixed_latency_front_end_footprint_u1_calibration_global_invalid.cubin

native: $(LONG_NATIVES) $(TRACE_NATIVES) $(NATIVE_SIDECARS)

fixed_latency_front_end_footprint_driver: fixed_latency_front_end_footprint_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

fixed_latency_front_end_footprint_u%_native.cubin: fixed_latency_front_end_footprint.cu
	$(NVCC) $(NVCC_FLAGS) -DFRONT_END_UNROLL=$* -cubin -o $@ $<

fixed_latency_front_end_footprint_u%_trace_native.cubin: fixed_latency_front_end_footprint.cu
	$(NVCC) $(NVCC_FLAGS) -DFRONT_END_UNROLL=$* \
		-DFRONT_END_GENERATIONS=64 -cubin -o $@ $<

%.control_bits.json: %.cubin
	$(PYTHON) $(CONTROL_TOOL) --binary $< --arch sm_86 --out $@ \
		--toolkit-bindir $(CUDA_BIN)

define LONG_PAIR_RULE
fixed_latency_front_end_footprint_u$(1)_fixed_bypass.cubin \
fixed_latency_front_end_footprint_u$(1)_fixed_retired.cubin \
fixed_latency_front_end_footprint_u$(1)_global_frontier.cubin \
fixed_latency_front_end_footprint_u$(1)_global_late.cubin \
fixed_latency_front_end_footprint_u$(1).patch.json &: \
		fixed_latency_front_end_footprint_u$(1)_native.cubin \
		fixed_latency_front_end_footprint_u$(1)_native.control_bits.json \
		patch_sm86_front_end_pairs.py
	$$(PYTHON) patch_sm86_front_end_pairs.py \
		fixed_latency_front_end_footprint_u$(1)_native.cubin \
		fixed_latency_front_end_footprint_u$(1)_native.control_bits.json \
		--unroll $(1) --prefix fixed_latency_front_end_footprint_u$(1) \
		--report fixed_latency_front_end_footprint_u$(1).patch.json
endef

define TRACE_PAIR_RULE
fixed_latency_front_end_footprint_u$(1)_trace_fixed_bypass.cubin \
fixed_latency_front_end_footprint_u$(1)_trace_fixed_retired.cubin \
fixed_latency_front_end_footprint_u$(1)_trace_global_frontier.cubin \
fixed_latency_front_end_footprint_u$(1)_trace_global_late.cubin \
fixed_latency_front_end_footprint_u$(1)_trace.patch.json &: \
		fixed_latency_front_end_footprint_u$(1)_trace_native.cubin \
		fixed_latency_front_end_footprint_u$(1)_trace_native.control_bits.json \
		patch_sm86_front_end_pairs.py
	$$(PYTHON) patch_sm86_front_end_pairs.py \
		fixed_latency_front_end_footprint_u$(1)_trace_native.cubin \
		fixed_latency_front_end_footprint_u$(1)_trace_native.control_bits.json \
		--unroll $(1) --prefix fixed_latency_front_end_footprint_u$(1)_trace \
		--report fixed_latency_front_end_footprint_u$(1)_trace.patch.json
endef

$(foreach u,$(UNROLLS),$(eval $(call LONG_PAIR_RULE,$(u))))
$(foreach u,$(UNROLLS),$(eval $(call TRACE_PAIR_RULE,$(u))))

fixed_latency_front_end_footprint_u1_calibration_fixed_invalid.cubin \
fixed_latency_front_end_footprint_u1_calibration_global_invalid.cubin \
fixed_latency_front_end_footprint_u1_calibration.patch.json &: \
		fixed_latency_front_end_footprint_u1_native.cubin \
		fixed_latency_front_end_footprint_u1_native.control_bits.json \
		patch_sm86_front_end_pairs.py
	$(PYTHON) patch_sm86_front_end_pairs.py \
		fixed_latency_front_end_footprint_u1_native.cubin \
		fixed_latency_front_end_footprint_u1_native.control_bits.json \
		--unroll 1 --prefix fixed_latency_front_end_footprint_u1_calibration \
		--report fixed_latency_front_end_footprint_u1_calibration.patch.json \
		--calibration

clean:
	rm -f fixed_latency_front_end_footprint_driver \
		$(LONG_NATIVES) $(TRACE_NATIVES) $(NATIVE_SIDECARS) \
		$(LONG_VARIANTS) $(TRACE_VARIANTS) \
		fixed_latency_front_end_footprint_u1_calibration_*.cubin \
		*.patch.json *.sass.txt *.resources.txt
