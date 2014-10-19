#include "raycast.h"

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* changed){
    *changed = 0;
    int3 pixel = {
        .x = threadIdx.x,
        .y = blockIdx.x*IMAGE_DIM,
        .z = blockIdx.y*IMAGE_SIZE};
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    int tid = pixel.z + pixel.y + pixel.x;

    if (/*50 == pixel.x && 300 == (pixel.y&pixel.z)*/
        (50*300*300) == tid){
        printf("\nI EXIST!!!\n\n");
    }

    if (0 > tid || (512*512*512) <= tid){
        printf("We have tid out of boundary: %d\n", tid);
    }
    //printf("NEW_VOX: %u\n", VISITED);

    if(NEW_VOX == region[tid]){
        printf("Entered first if!\n");
        printf("tid: .x=%d, .y=%d, .z=%d\n", pixel.x, pixel.y, pixel.z);
        int3 pos;
        region[tid] = VISITED;
        for (int i = 0; i < 6; ++i){
            pos = pixel;
            pos.x += dx[i];
            pos.y += dy[i];
            pos.z += dz[i];
            if (//Check that pos pixel is inside image/region
                ((pos.x >= 0 && pos.x < DATA_DIM-1) &&
                (pos.y >= 0 && pos.y < DATA_DIM-1) &&
                (pos.z >= 0 && pos.z < DATA_DIM-1)) &&
                //Check that it's not already been "discovered"
                !region[tid] &&
                //Check that the corresponding color values actually match
                (abs(data[tid] - data[pos.z + pos.y + pos.x]) < 1)){
                //then
                printf("Found neighbour!\n");
                region[tid] = NEW_VOX;
                atomicAdd(changed, 1);
            }
        }
    }

    //*changed = 1;
    return;
}

unsigned char* grow_region_gpu(unsigned char* data){
    printf("Entered grow_region_gpu!\n");
    int finished = 0, *gpu_finished;
    cudaEvent_t start, end;
    dim3 blockDim, gridDim;
    stack_t* stack = new_stack();
    int3 seed = {.x = 50, .y = 300, .z = 300};
    unsigned char *cudaImage, *cudaRegion, *region;
    blockDim.x = 512, gridDim.x = 512, gridDim.y = 512;
    region = (unsigned char*) calloc(DATA_SIZE, sizeof(unsigned char));

    push(stack, seed);
    region[seed.z*IMAGE_SIZE + seed.y*DATA_DIM + seed.x] = NEW_VOX;

    printf("Done preparing variables!\n");

    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaImage, sizeof(unsigned char)*DATA_SIZE));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, sizeof(unsigned char)*DATA_SIZE));
    gEC(cudaMalloc(&gpu_finished, sizeof(int)));

    printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaImage, data, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying image and region to device took %f ms\n",
        getCudaEventTime(start, end));

    int cntr = 0;
    //while(!finished || 9 < cntr){
        cntr += 1;
        printf("\nEntered while-loop\n");
        gEC(cudaMemcpy(gpu_finished, &finished, sizeof(int), cudaMemcpyHostToDevice));
        printf("Finished first while-loop memcpy!\n");
        region_grow_kernel<<<gridDim, blockDim>>>(data, region, gpu_finished);
        printf("Finished kernel call!\n");
        gEC(cudaMemcpy(&finished, gpu_finished, sizeof(int), cudaMemcpyDeviceToHost));
    //}


    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying region from device took %f ms\n", getCudaEventTime(start, end));

    return region;
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    //blah
    return;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){

    return NULL;
}

int main(int argc, char** argv){
    //float ms_time;
    /*print_properties();

    printf("Done printing properties\n");

    printf("size of data: %zd\n", sizeof(unsigned char)*DATA_DIM*DATA_DIM*DATA_DIM/(1024*1024));*/

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
