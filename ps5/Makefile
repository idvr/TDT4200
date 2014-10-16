#Phonies
.PHONY: all
.PHONY: run
.PHONY: clean
.PHONY: install

all: clean make_all

#Whatever is currently being worked with
run: gpu
	./run_gpu

make_all: cpu gpu gpu_shared gpu_texture gpu_combined
	./run_cpu
	./run_gpu
	./run_gpu_shared
	./run_gpu_texture
	./run_gpu_combined

cpu: clean bmp.cpp raycast_support.cu raycast_cpu.cu
	nvcc bmp.cpp raycast_support.cu raycast_cpu.cu -lcudart -g -lm -o run_cpu

gpu: clean bmp.cpp raycast_support.cu raycast_gpu.cu
	nvcc bmp.cpp raycast_support.cu raycast_gpu.cu -lcudart -g -lm -o run_gpu

gpu_shared: clean bmp.cpp raycast_support.cu raycast_gpu_shared.cu
	nvcc bmp.cpp raycast_support.cu raycast_gpu_shared.cu -lcudart -g -lm -o run_gpu_shared

gpu_texture: clean bmp.cpp raycast_support.cu raycast_gpu_texture.cu
	nvcc bmp.cpp raycast_support.cu raycast_gpu_texture.cu -lcudart -g -lm -o run_gpu_texture

gpu_combined: clean bmp.cpp raycast_support.cu raycast_gpu_combined.cu
	nvcc bmp.cpp raycast_support.cu raycast_gpu_combined.cu -lcudart -g -lm -o run_gpu_combined

clean:
	rm -f run_*
