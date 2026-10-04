CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

VARIANTS = ret32 red32 ret64 red64
LONG_CUBINS = $(foreach v,$(VARIANTS),atomic_alias_adversarial_$(v).cubin)

.PHONY: all clean

all: atomic_alias_adversarial_driver $(LONG_CUBINS)

atomic_alias_adversarial_driver: atomic_alias_adversarial_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

define CUBIN_RULES
atomic_alias_adversarial_$(1).cubin: atomic_alias_adversarial.cu
	$$(NVCC) $$(NVCC_FLAGS) -DATOMIC_RETURNING=$(2) -DATOMIC_WIDTH=$(3) \
		-cubin -o $$@ $$<

endef

$(eval $(call CUBIN_RULES,ret32,1,32))
$(eval $(call CUBIN_RULES,red32,0,32))
$(eval $(call CUBIN_RULES,ret64,1,64))
$(eval $(call CUBIN_RULES,red64,0,64))

clean:
	rm -f atomic_alias_adversarial_driver $(LONG_CUBINS)
