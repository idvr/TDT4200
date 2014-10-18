#include "raycast.h"

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){
    *finished = 1;
    return;
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    //blah
}

unsigned char* grow_region_gpu(unsigned char* data){
    int finished = 0;
    dim3 blockDim, gridDim;
    cudaEvent_t start, end;
    unsigned char* cudaImage, *cudaRegion, *region;
    blockDim.x = 512, gridDim.x = 512, gridDim.y = 512;
    region = (unsigned char*) calloc(IMAGE_SIZE, sizeof(unsigned char));

    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaImage, sizeof(unsigned char)*IMAGE_SIZE));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, sizeof(unsigned char)*IMAGE_SIZE));

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaImage, data, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying image and region to device took %f ms\n",
        getCudaEventTime(start, end));


    //while(!finished){
        //region_grow_kernel<<<gridDim, blockDim>>>(data, region, &finished);
    //}



    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying region from device took %f ms\n", getCudaEventTime(start, end));

    return region;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){

    return NULL;
}

int main(int argc, char** argv){
    //float ms_time;
    print_properties();

    printf("Done printing properties\n");

    printf("size of data: %zd\n", sizeof(unsigned char)*DATA_DIM*DATA_DIM*DATA_DIM/(1024*1024));

    unsigned char* data = create_data();

    printf("Done creating data\n");


    unsigned char* region = grow_region_gpu(data);
    //printf("grow_region_gpu() took %f ms\n", ms_time);

    printf("Done creating region\n");

    unsigned char* image = raycast_gpu(data, region);
    /*printf("raycast_gpu() took %f ms\n", ms_time);*/

    printf("Done creating image\n");

    //write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_out.bmp");

    printf("Done with program\n");
    return 0;
}
