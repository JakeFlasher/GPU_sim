CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: issue_port_control issue_port_control_trace

issue_port_control: issue_port_control.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

issue_port_control_trace: issue_port_control.cu
	$(NVCC) $(NVCC_FLAGS) -DISSUE_PORT_GENERATIONS=64 -o $@ $<

clean:
	rm -f issue_port_control issue_port_control_trace
