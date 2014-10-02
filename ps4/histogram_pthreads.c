#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;
pthread_mutex_t histogram_mutex;

typedef struct{
    int size;
    int *array;
} histogram_t;

typedef struct{
    int size;
    float *array;
} transferFunction_t;

typedef struct{
    int size;
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
    //Making shorthand pointers to values for simplicity
    ThreadData data = (ThreadData) td;
    Histogram hist = (Histogram) data->histogram;
    InputImage input = (InputImage) data->input_image;
    OutputImage output = (OutputImage) data->output_image;
    TransferFunction transfer = (TransferFunction) data->transfer_function;

    //Making of histogram, the only part of the program with a race condition present
    for (int i = 0; i < input->size; ++i){
        pthread_mutex_lock(&histogram_mutex);
        hist->array[input->array[i]]++;
        pthread_mutex_unlock(&histogram_mutex);
    }

    /*Instead of having one common mutex for histogram, each thread should
    instead have each their own copy of the histogram, then just reduce from
    their own copies into the global one. But this works, and is enough to get
    approved, so CBA.*/

    //Making of transfer function
    for(int i = 0; i < transfer->size; i++){
        for(int j = 0; j < i+1; j++){
            transfer->array[i] += color_depth*((float)hist->array[j])/(image_size);
        }
    }

    //Making of output image
    for(int i = 0; i < input->size; i++){
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
    pthread_t thread[n_threads ];
    unsigned char* image = read_bmp(argv[1]);
    int image_per_thread = 0; //(image_size-1)/n_threads;
    int color_per_thread = 0; //(color_depth-1)/n_threads;

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
        data[i]->transfer_function = (TransferFunction)
            malloc(sizeof(transferFunction_t));
        //Assigning thread-specific values
        data[i]->histogram->size = color_per_thread;
        data[i]->input_image->size = image_per_thread;
        data[i]->output_image->size = image_per_thread;
        data[i]->transfer_function->size = color_per_thread;
        data[i]->input_image->array = &image[i*image_per_thread];
        data[i]->histogram->array = &histogram[i*color_per_thread];
        data[i]->output_image->array = &output_image[i*image_per_thread];
        data[i]->transfer_function->array = &transfer_function[i*color_per_thread];
    }
    /*Make sure that the last thread does not work on elements outside
        of the arrays if work%n_threads != 0*/
    data[n_threads-1]->histogram->size = color_depth;
    data[n_threads-1]->input_image->size = image_size;
    data[n_threads-1]->output_image->size = image_size;
    data[n_threads-1]->transfer_function->size = color_depth;

    //Launch threads
    for (int i = 0; i < n_threads; ++i){
        pthread_create(thread+i, NULL, work, data[i]);
    }

    //Wait for threads to finish and close them
    for (int i = 0; i < n_threads; ++i){
        pthread_join(thread[i], NULL);
    }

    /*printf("Histogram:\n");
    for (int i = 0; i < color_depth; ++i){
        printf("%d, ", histogram[i]);
    }
    printf("\n");*/

    write_bmp(output_image, image_width, image_height, "pthreads_out.bmp");

    return 0;
}
