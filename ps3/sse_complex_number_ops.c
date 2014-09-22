#include <stdio.h>
#include <tmmintrin.h>
#include <complex.h>

void complex_mult(complex double x, complex double y, complex double *z){
	__m128d num1, num2, num3;
	num1 = _mm_set_pd(creal(x), creal(x));
	num2 = _mm_set_pd(cimag(y), creal(y));
	num3 = _mm_mul_pd(num2, num1);
	
	num1 = _mm_set_pd(cimag(x), cimag(x));
	num2 = _mm_shuffle_pd(num2, num2, 1);
	
	num2 = _mm_mul_pd(num2, num1);
	num3 = _mm_addsub_pd(num3, num2);
	
	
	_mm_storeu_pd((double *)z, num3);
}

void complex_sub(complex double x, complex double y, complex double *z){
	__m128d num1, num2, num3;
	num1 = _mm_set_pd(cimag(x), creal(x));
	num2 = _mm_set_pd(cimag(y), creal(y));
	num3 = _mm_sub_pd(num1, num2);
	//num3 = _mm_shuffle_pd(num3, num3, 1);
	_mm_storeu_pd((double *)z, num3);
	
}


void complex_add(complex double x, complex double y, complex double *z){
	__m128d num1, num2, num3;
	num1 = _mm_set_pd(cimag(x), creal(x));
	num2 = _mm_set_pd(cimag(y), creal(y));
	num3 = _mm_add_pd(num1, num2);
	//num3 = _mm_shuffle_pd(num3, num3, 1);
	_mm_store_pd((double *)z, num3);
}


int main(int argc, char *argv[]) {
	
	complex double x, y, z;
	x = 3 + 4 * I;
	y = 0 - 2 * I;
	
	complex_add(x, y, &z);
	//complex_sub(x, y, &z);
	//complex_mult(x, y, &z);
	printf("(%f, %f)\n", creal(z), cimag(z));

}