#Phonies, making sure that Make does not mistake any file with below
#command names for something else than what the commands intend
.PHONY: all
.PHONY: run
.PHONY: clean
.PHONY: install

target:= raycast_debug

all: cpu gpu gpu_shared gpu_texture gpu_combined

get_result:
	scp gpu-lab:~/tdt4200/ps5/raycast_$(target)_out.bmp ./

memcheck: $(target)
	cuda-memcheck optirun ./run_$(target) --no-xorg --no-failsafe

run: $(target)
	./run_$(target)

raycast_debug: bmp.c clutil.c raycast.c
	gcc -std=c99 bmp.c clutil.c raycast.c -o raycast \
	-lOpenCL -lm -Wextra -Wall -pedantic -g -O3

raycast_release: bmp.c clutil.c raycast.c
	gcc -std=c99 bmp.c clutil.c raycast.c -o raycast \
	-lOpenCL -lm -O3

clean:
	rm -f run_* *.o
