#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;

int main(int argc, char** argv){
    if(argc != 3){
        printf("Useage: %s image n_threads\n", argv[0]);
        exit(-1);
    }

    int n_threads = atoi(argv[2]);
    omp_set_num_threads(n_threads);

    unsigned char* image = read_bmp(argv[1]);
    unsigned char* output_image = malloc(sizeof(unsigned char)*image_size);


    int* histogram = (int*)calloc(sizeof(int), color_depth);
    #pragma omp parallel for shared(histogram)
     for(int i = 0; i < image_size; i++){
        histogram[image[i]]++;
    }

    float* transfer_function = (float*)calloc(sizeof(float), color_depth);
    #pragma omp parallel for shared(transfer_function)
    for(int i = 0; i < color_depth; i++){
        for(int j = 0; j < i+1; j++){
            transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < image_size; i++){
        output_image[i] = transfer_function[image[i]];
    }

    write_bmp(output_image, image_width, image_height, "omp_out.bmp");
}
