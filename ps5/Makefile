#Phonies, making sure that Make does not mistake any file with below
#command names for something else than what the commands intend
.PHONY: all
.PHONY: run
.PHONY: clean
.PHONY: install

#Whatever is currently being worked with
target:= gpu #Edit this as project progresses

all: cpu gpu gpu_shared gpu_texture gpu_combined

memcheck: $(target)
	cuda-memcheck optirun ./run_$(target)

run_local: $(target)
	optirun ./run_$(target)
#Shorthand for above
rl: run_local

run_school: $(target)
	./run_$(target)
#Shorthand for above
rs: run_school

all_school: clean all
	./run_cpu
	./run_gpu
	./run_gpu_shared
	./run_gpu_texture
	./run_gpu_combined

all_local: clean all
	optirun ./run_cpu
	optirun ./run_gpu
	optirun ./run_gpu_shared
	optirun ./run_gpu_texture
	optirun ./run_gpu_combined

cpu: clean bmp.cpp raycast_common.cu raycast_cpu.cu
	nvcc bmp.cpp raycast_common.cu raycast_cpu.cu -lcudart -g -G -use_fast_math -lm -arch=sm_20 -o run_cpu

gpu: clean bmp.cpp raycast_common.cu raycast_gpu.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu.cu -lcudart -g -G -use_fast_math -lm -arch=sm_20 -o run_gpu

gpu_shared: clean bmp.cpp raycast_common.cu raycast_gpu_shared.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_shared.cu -lcudart -g -G -use_fast_math -lm -arch=sm_20 -o run_gpu_shared

gpu_texture: clean bmp.cpp raycast_common.cu raycast_gpu_texture.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_texture.cu -lcudart -g -G -use_fast_math -lm -arch=sm_20 -o run_gpu_texture

gpu_combined: clean bmp.cpp raycast_common.cu raycast_gpu_combined.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_combined.cu -lcudart -g -G -use_fast_math -lm -arch=sm_20 -o run_gpu_combined

clean:
	rm -f run_* *.o
