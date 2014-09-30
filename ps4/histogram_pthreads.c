#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;

typedef struct{
    int start, stop;
    int *array;
} histogram_t;

typedef struct{
    int start, stop;
    float *array;
} transferFunction_t;

typedef struct{
    int start, stop;
    unsigned char *array;
} image_t;

typedef image_t* InputImage;
typedef image_t* OutputImage;
typedef histogram_t* Histogram;
typedef transferFunction_t* TransferFunction;

typedef struct{
    Histogram histogram;
    InputImage input_image;
    OutputImage output_image;
    TransferFunction transfer_function;
} threadData_t;

typedef threadData_t* ThreadData;

void* work(void* td){
    printf("Thread entered work()!!\n");
    ThreadData data = (ThreadData) td;

    //Making shorthand pointers to values for simplicity
    Histogram hist = (Histogram) data->histogram;
    InputImage input = (InputImage) data->input_image;
    OutputImage output = (OutputImage) data->output_image;
    TransferFunction transfer = (TransferFunction) data->transfer_function;

    //
    //Making of histogram
    for (int i = input->start; i < input->stop; ++i){
        hist->array[input->array[i]];
    }

    //Making of transfer function
    for(int i = transfer->start; i < transfer->stop; i++){
        for(int j = transfer->start; j < i+1; j++){
            transfer->array[i] += color_depth*((float)hist->array[j])/(image_size);
        }
    }

    //Making of output image
    for(int i = input->start; i < input->stop; i++){
        output->array[i] = transfer->array[input->array[i]];
    }
    return NULL;
}

int main(int argc, char** argv){
    if(argc != 3){
        printf("Useage: %s image n_threads\n", argv[0]);
        exit(-1);
    }
    int n_threads = atoi(argv[2]);

    ThreadData data[n_threads];
    pthread_t thread[n_threads];
    unsigned char* image = read_bmp(argv[1]);
    int image_per_thread = (image_size-1)/n_threads;
    int color_per_thread = (color_depth-1)/n_threads;

    int* histogram = (int*)calloc(sizeof(int), color_depth);
    float* transfer_function = (float*)calloc(sizeof(float), color_depth);
    unsigned char* output_image = (unsigned char*) malloc(sizeof(unsigned
        char)*image_size);

    //Initialize all thread structures sharing the work
    for (int i = 0; i < n_threads; ++i){
        //Malloc'ing all structures
        data[i] = (ThreadData) malloc(sizeof(threadData_t));
        data[i]->input_image = (InputImage) malloc(sizeof(image_t));
        data[i]->histogram = (Histogram) malloc(sizeof(histogram_t));
        data[i]->output_image = (OutputImage) malloc(sizeof(image_t));
        data[i]->transfer_function = (TransferFunction) malloc(sizeof(
            transferFunction_t));

        //Input Image Data
        data[i]->input_image->array = image;
        data[i]->input_image->start = i*image_per_thread;
        data[i]->input_image->stop = (i+1)*image_per_thread;

        //Histogram Data
        data[i]->histogram->array = histogram;
        data[i]->histogram->start = i*color_per_thread;
        data[i]->histogram->stop = (i+1)*color_per_thread;

        //Output Image Data
        data[i]->output_image->array = output_image;
        data[i]->output_image->start = i*image_per_thread;
        data[i]->output_image->stop = (i+1)*image_per_thread;

        //Transfer Function Data
        data[i]->transfer_function->array = transfer_function;
        data[i]->transfer_function->start = i*color_per_thread;
        data[i]->transfer_function->stop = (i+1)*color_per_thread;
    }
    /*Make sure that the last thread does not work on elements outside
        of the arrays if work%n_threads != 0*/
    data[n_threads-1]->input_image->stop = image_size;
    data[n_threads-1]->histogram->stop = color_depth;
    data[n_threads-1]->output_image->stop = image_size;
    data[n_threads-1]->transfer_function->stop = color_depth;

    //Launch threads
    for (int i = 0; i < n_threads; ++i){
        pthread_create(&thread[i], NULL, work, &data[i]);
    }

    //Wait for threads to finish and close them
    for (int i = 0; i < n_threads; ++i){
        pthread_join(thread[i], NULL);
    }

    write_bmp(output_image, image_width, image_height, "pthreads_out.bmp");

    return 0;
}
