all: clean serial omp pthreads run_serial run_omp run_pthreads

input:= light.bmp
t_count:= 4

run_serial: serial
	./serial $(input) $(t_count)

run_omp: omp
	./omp $(input) $(t_count)

run_pthreads: pthreads
	./pthreads $(input) $(t_count)

serial: histogram_serial.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_serial.c bmp.c -o serial

omp: histogram_omp.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_omp.c bmp.c -o omp -fopenmp

pthreads: histogram_pthreads.c
	gcc -std=c99 -Wall -Wextra -pedantic -g histogram_pthreads.c bmp.c -o pthreads -pthread

clean:
	rm -f omp serial pthreads *out.bmp
