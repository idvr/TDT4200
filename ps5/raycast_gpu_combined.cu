#include "raycast.h"

__global__ void raycast_kernel_texture(uchar* image){

}

uchar* raycast_gpu_texture(uchar* data, uchar* region){
    return NULL;
}


__global__ void region_grow_kernel_shared(uchar* data, uchar* region, int* finished){

}

uchar* grow_region_gpu_shared(uchar* data){
    return NULL;
}


int main(int argc, char** argv){

    print_properties();

    uchar* data = create_data();

    /* //Serial version
    uchar* region = grow_region_serial(data);

    uchar* image = raycast_serial(data, region);*/

    //write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_combined_out.bmp");
}
