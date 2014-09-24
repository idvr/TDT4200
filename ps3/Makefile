all: spmv

clustis: spmv
	qrsh -cwd ./spmv 100000 401 200 100 200 10

run: spmv
	./spmv 20 5 2 3 2 2

spmv: spmv.c
	gcc -Wall -std=c99 -O3 -msse3 spmv.c -o spmv -lm -Wall -Wextra -pedantic -g

clean:
	rm spmv -f
