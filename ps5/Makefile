#Phonies, making sure that Make does not mistake any file with below
#command names for something else than what the commands intend
.PHONY: all
.PHONY: run
.PHONY: clean
.PHONY: install

#Whatever is currently being worked with
target:=gpu_texture

all: cpu gpu gpu_shared gpu_texture gpu_combined

get_result:
	scp gpu:~/tdt4200/ps5/raycast_$(target)_out.bmp ./

memcheck: $(target)
	cuda-memcheck optirun ./run_$(target) --no-xorg --no-failsafe

run_local: $(target)
	optirun ./run_$(target) --no-xorg --no-failsafe
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
	optirun ./run_gpu --no-xorg --no-failsafe
	optirun ./run_gpu_shared --no-xorg --no-failsafe
	optirun ./run_gpu_texture --no-xorg --no-failsafe
	optirun ./run_gpu_combined --no-xorg --no-failsafe

cpu: bmp.cpp raycast_cpu.cu
	nvcc bmp.cpp raycast_cpu.cu -o run_cpu \
	-lcudart -g -G -use_fast_math -lm -arch=sm_20 \

gpu: bmp.cpp raycast_common.cu raycast_gpu.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu.cu \
	-lcudart -g -G -use_fast_math -lm -arch=sm_20 \
	-o run_gpu

gpu_shared: bmp.cpp raycast_common.cu raycast_gpu_shared.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_shared.cu \
	 -lcudart -g -G -use_fast_math -lm -arch=sm_20 \
	 -o run_gpu_shared

gpu_texture: bmp.cpp raycast_common.cu raycast_gpu_texture.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_texture.cu \
	 -lcudart -g -G -use_fast_math -lm -arch=sm_20 \
	 -o run_gpu_texture

gpu_combined: bmp.cpp raycast_common.cu raycast_gpu_combined.cu
	nvcc bmp.cpp raycast_common.cu raycast_gpu_combined.cu \
	 -lcudart -g -G -use_fast_math -lm -arch=sm_20 \
	 -o run_gpu_combined

clean:
	rm -f run_* *.o
