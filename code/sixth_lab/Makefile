# Week 7 Lab.
#   make ARCH=sm_86     RTX 30 series
#   make ARCH=sm_89     RTX 40 series
#   make ARCH=sm_75     Colab / Kaggle T4
ARCH ?= sm_86
NVCC ?= nvcc

reduction: reduction.cu
	$(NVCC) -O3 -arch=$(ARCH) $< -o $@

run: reduction
	./reduction

clean:
	rm -f reduction *.o
	rm -rf __pycache__

.PHONY: run clean
