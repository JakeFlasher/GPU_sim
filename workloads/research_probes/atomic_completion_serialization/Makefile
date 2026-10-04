CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PYTHON ?= ../../../.venv/bin/python
CONTROL_TOOL = ../../../tools/sass_control/emit_control_bits_sidecar.py

VARIANTS = \
	ret_lane_line ret_lane_word ret_warp_word ret_global_word \
	red_lane_line red_lane_word red_warp_word red_global_word
LONG_CUBINS = $(foreach v,$(VARIANTS),atomic_completion_serialization_$(v).cubin)

.PHONY: all clean

all: atomic_completion_serialization_driver $(LONG_CUBINS)

atomic_completion_serialization_driver: atomic_completion_serialization_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

define LONG_RULE
atomic_completion_serialization_$(2).cubin: atomic_completion_serialization.cu
	$$(NVCC) $$(NVCC_FLAGS) -DATOMIC_VARIANT=$(1) -cubin -o $$@ $$<
endef

$(eval $(call LONG_RULE,0,ret_lane_line))
$(eval $(call LONG_RULE,1,ret_lane_word))
$(eval $(call LONG_RULE,2,ret_warp_word))
$(eval $(call LONG_RULE,3,ret_global_word))
$(eval $(call LONG_RULE,4,red_lane_line))
$(eval $(call LONG_RULE,5,red_lane_word))
$(eval $(call LONG_RULE,6,red_warp_word))
$(eval $(call LONG_RULE,7,red_global_word))
clean:
	rm -f atomic_completion_serialization_driver $(LONG_CUBINS) \
		*.control_bits.json
