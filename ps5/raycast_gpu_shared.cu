#include "raycast.h"

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){

}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    return NULL;
}

__global__ void region_grow_kernel_shared(unsigned char* data, unsigned char* region, int* finished){

}

unsigned char* grow_region_gpu_shared(unsigned char* data){
    return NULL;
}


int main(int argc, char** argv){

    print_properties();

    unsigned char* data = create_data();

    /* //Serial version
    unsigned char* region = grow_region_serial(data);

    unsigned char* image = raycast_serial(data, region);*/

    //write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycase_gpu_shared_out.bmp");
}
