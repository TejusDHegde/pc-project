# Build the CUDA program. Needs the CUDA toolkit (nvcc) and an NVIDIA GPU to run.
#   make                     -> bin/vector_add, compiled for the GPU in this machine
#   make ARCH=-arch=sm_75    -> compile for a specific GPU (sm_75 = Tesla T4 on Colab)
NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS  = -O3 -std=c++17 -Xcompiler -fopenmp -lgomp

bin/vector_add: src/vector_add.cu
	mkdir -p bin
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

clean:
	rm -rf bin

.PHONY: clean
