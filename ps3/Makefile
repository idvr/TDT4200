all: spmv

clustis: spmv-release
	qrsh -cwd ./spmv 100000 401 200 100 200 10

run: spmv-debug
	./spmv 1000 21 4 20 3 20

valgrind: spmv-debug
	valgrind ./spmv 5 3 0 1 0 1

fast: spmv-release
	./spmv 100 11 4 5 3 4

spmv-release: spmv.c
	gcc -std=c99 -O3 -msse3 spmv.c -o spmv -lm

spmv-debug: spmv.c
	gcc -Wall -std=c99 -O3 -msse3 spmv.c -o spmv -lm -Wall -Wextra -pedantic -g

clean:
	rm -f spmv-debug spmv-release
