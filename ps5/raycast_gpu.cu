#include "raycast.h"

unsigned char* grow_region_gpu(unsigned char* data){

    return NULL;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){

    return NULL;
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){
    //blah
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    //blah
}

int main(int argc, char** argv){
    float ms_time;
    cudaEvent_t start, end;
    print_properties();

    printf("Done printing properties\n");

    printf("size of data: %zd\n", sizeof(unsigned char)*DATA_DIM*DATA_DIM*DATA_DIM/(1024*1024));

    unsigned char* data = create_data();

    printf("Done creating data\n");

    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventRecord(start, 0));
    unsigned char* region = grow_region_gpu(data);
    gpuErrorCheck(cudaEventCreate(&end));
    gpuErrorCheck(cudaEventRecord(end, 0));
    gpuErrorCheck(cudaEventElapsedTime(&ms_time, start, end));
    printf("grow_region_gpu() took %f ms\n", ms_time);

    printf("Done creating region\n");

    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventRecord(start, 0));
    unsigned char* image = raycast_gpu(data, region);
    gpuErrorCheck(cudaEventCreate(&end));
    gpuErrorCheck(cudaEventRecord(end, 0));
    gpuErrorCheck(cudaEventElapsedTime(&ms_time, start, end));
    printf("raycast_gpu() took %f ms\n", ms_time);

    printf("Done creating image\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_out.bmp");

    printf("Done with program\n");
    return 0;
}
