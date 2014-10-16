#include "raycast.h"

unsigned char* grow_region_gpu(unsigned char* data){
    return NULL;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    return NULL;
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){

}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){

}

int main(int argc, char** argv){
    print_properties();

    printf("Done printing properties\n");

    unsigned char* data = create_data();

    printf("Done creating data\n");

    unsigned char* region = grow_region_gpu(data);

    printf("Done creating region\n");

    unsigned char* image = raycast_gpu(data, region);

    printf("Done creating image\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_out.bmp");

    printf("Done with program\n");
    return 0;
}
