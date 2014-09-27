all: spmv

clustis: spmv-release
	qrsh -cwd ./spmv 100000 401 200 100 200 10

run: spmv-debug
	./spmv 5 1 1 1 1 1

valgrind: spmv-debug
	valgrind ./spmv 5 1 1 1 1 1

fast: spmv-release
	./spmv 25 5 2 3 2 3

spmv-release: spmv.c
	gcc -std=c99 -O3 -msse3 spmv.c -o spmv -lm

spmv-debug: spmv.c
	gcc -Wall -std=c99 -O3 -msse3 spmv.c -o spmv -lm -Wall -Wextra -pedantic -g

clean:
	rm -f spmv-debug spmv-release
