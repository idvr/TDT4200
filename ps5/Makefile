all: clean run_all

#Whatever is currently being worked ons
make:

run_all: run_cpu run_gpu run_gpu_shared run_gpu_texture run_gpu_combined
	./run_cpu
	./run_gpu
	./run_gpu_shared
	./run_gpu_texture
	./run_gpu_combined

run_cpu: clean bmp.cpp raycast_cpu.cu
	nvcc bmp.cpp raycast_cpu.cu -lcudart -lm -o run_cpu

run_gpu: clean bmp.cpp raycast_gpu.cu
	nvcc bmp.cpp raycast_gpu.cu -lcudart -lm -o run_gpu

run_gpu_shared: clean bmp.cpp raycast_gpu_shared.cu
	nvcc bmp.cpp raycast_gpu_shared.cu -lcudart -lm -o run_gpu_shared

run_gpu_texture: clean bmp.cpp raycast_gpu_texture.cu
	nvcc bmp.cpp raycast_gpu_texture.cu -lcudart -lm -o run_gpu_texture

run_gpu_combined: clean bmp.cpp raycast_gpu_combined.cu
	nvcc bmp.cpp raycast_gpu_combined.cu -lcudart -lm -o run_gpu_combined

clean: rm -f run_*
