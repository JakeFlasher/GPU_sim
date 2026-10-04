CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
VARIANTS = ret32 red32
LONG_CUBINS = $(foreach v,$(VARIANTS),atomic_multism_service_$(v).cubin)

.PHONY: all clean
all: atomic_multism_service_driver $(LONG_CUBINS)

atomic_multism_service_driver: atomic_multism_service_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

define CUBIN_RULES
atomic_multism_service_$(1).cubin: atomic_multism_service.cu
	$$(NVCC) $$(NVCC_FLAGS) -DATOMIC_RETURNING=$(2) -cubin -o $$@ $$<

endef

$(eval $(call CUBIN_RULES,ret32,1))
$(eval $(call CUBIN_RULES,red32,0))

clean:
	rm -f atomic_multism_service_driver $(LONG_CUBINS)
