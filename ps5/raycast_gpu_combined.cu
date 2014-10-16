#include "raycast.h"

__global__ void raycast_kernel_texture(unsigned char* image){

}

unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region){
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

    //write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_combined_out.bmp");
}
