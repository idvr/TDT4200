#include "raycast.h"

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){
    //blah
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    //blah
}

unsigned char* grow_region_gpu(unsigned char* data){
    int finished = 0; float ms_time = -1.0;
    cudaEvent_t start, end;
    unsigned char* cudaImage, *cudaRegion,
        *region = (unsigned char*) malloc(sizeof(unsigned char)*IMAGE_SIZE);
    dim3 blockDim, gridDim;
    blockDim.x = 512, gridDim.x = 512, gridDim.y = 512;

    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaImage, sizeof(unsigned char)*IMAGE_SIZE));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, sizeof(unsigned char)*IMAGE_SIZE));

    //Copy image over to device
    gEC(cudaMemcpy(cudaImage, data, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice));

    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventRecord(start, 0));

    while(!finished){
        region_grow_kernel<<<gridDim, blockDim>>>(data, region, &finished);
    }

    gpuErrorCheck(cudaEventCreate(&end));
    gpuErrorCheck(cudaEventRecord(end, 0));
    gpuErrorCheck(cudaEventElapsedTime(&ms_time, start, end));

    //Copy region from device
    gEC(cudaMemcpy(region, cudaRegion, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyDeviceToHost));

    return region;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){

    return NULL;
}

int main(int argc, char** argv){
    float ms_time;
    cudaEvent_t start, end;
    print_properties();

    printf("Done printing properties\n");

    printf("size of data: %zd\n", sizeof(unsigned char)*DATA_DIM*DATA_DIM*DATA_DIM/(1024*1024));

    unsigned char* data = create_data();

    printf("Done creating data\n");


    unsigned char* region = grow_region_gpu(data);
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
