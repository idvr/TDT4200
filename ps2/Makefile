all : region

clustis : region
	qrsh -cwd mpirun -n 1 region pic1.bmp

run : region
	mpirun -n 4 region pic1.bmp

val : region
	valgrind mpirun -n 4 region pic1.bmp

region : region.c
	mpicc -std=c99 region.c bmp.c -o region -lm -Wall -Wextra -pedantic -g

clean :
	rm -f region
