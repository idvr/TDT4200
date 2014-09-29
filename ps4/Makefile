all: clean histogram_serial histogram_omp histogram_pthreads run_serial run_omp run_pthreads

input:= dark.bmp
t_count:= 4

run_serial: histogram_serial
	./histogram_serial $(input) $(t_count)

run_omp: histogram_omp
	./histogram_omp $(input) $(t_count)

run_pthreads: histogram_pthreads
	./histogram_pthreads $(input) $(t_count)

histogram_serial: histogram_serial.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_serial.c bmp.c -o histogram_serial

histogram_omp: histogram_omp.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_omp.c bmp.c -o histogram_omp -fopenmp

histogram_pthreads: histogram_pthreads.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_pthreads.c bmp.c -o histogram_pthreads -pthread

clean:
	rm -f histogram_omp histogram_serial histogram_pthreads *out.bmp
