# Week 6 Lab kit.
#   make ARCH=sm_86     RTX 30 series
#   make ARCH=sm_89     RTX 40 series
#   make ARCH=sm_75     Colab / Kaggle T4
ARCH ?= sm_86
NVCC ?= nvcc
FLAGS = -O3 -arch=$(ARCH)

all: conv_basic conv_tiled conv_layer

conv_basic: conv_basic.cu common.cuh
	$(NVCC) $(FLAGS) $< -o $@
conv_tiled: conv_tiled.cu common.cuh
	$(NVCC) $(FLAGS) $< -o $@
conv_layer: conv_layer.cu common.cuh
	$(NVCC) $(FLAGS) $< -o $@

# Everything a submission needs, in one go.
run: all
	./conv_basic          > conv_basic.csv
	./conv_tiled          > conv_tiled.csv
	./conv_layer          > conv_layer.csv
	./conv_basic break    > break_A.txt
	./conv_tiled break1   > break_B.txt
	./conv_tiled break2   > break_C.txt
	./conv_layer break    > break_D.txt
	python3 plot_layer.py conv_layer.csv layer_plot.png
	@echo "Now fill in answers.txt, then run: python3 selfcheck.py"

clean:
	rm -f conv_basic conv_tiled conv_layer *.csv break_*.txt layer_plot.png
	rm -rf __pycache__

.PHONY: all run clean
