#include "raycast.h"

__device__ int getBlockId_3D(){
    return blockIdx.x + (blockIdx.y*gridDim.x)
            + (blockIdx.z*gridDim.x*gridDim.y);
}

__device__ int gpu_getDataIndex(int3 pos){
    return pos.z*IMAGE_SIZE
        + pos.y*DATA_DIM + pos.x;
}

__device__ int gpu_isPosInside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__device__ int getBlockThreadId_3D(){
    return threadIdx.x + (threadIdx.y*blockDim.x)
            + (threadIdx.z*blockDim.x*blockDim.y);
}

__device__ int getGlobalIdx_3D_3D(){
    int blockId = getBlockId_3D();
    int threadId = getBlockThreadId_3D() +
            blockId*(blockDim.x*blockDim.y*blockDim.z);
    return threadId;
}

__device__ int3 getGlobalPos(int globalThreadId){
    int3 pos = {
        .x = globalThreadId,
            .y = 0, .z = 0};

    //Check if x > (512^2 - 1)
    if ((IMAGE_SIZE-1) < pos.x){
        pos.z = pos.x/IMAGE_SIZE;
        pos.x -= pos.z*IMAGE_SIZE;
    }

    //Check if x > (512 - 1)
    if ((IMAGE_DIM-1) < pos.x){
        pos.y = pos.x/IMAGE_DIM;
        pos.x -= pos.y*IMAGE_DIM;
    }

    return pos;
}

__device__ int gpu_similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[gpu_getDataIndex(a)];
    unsigned char vb = data[gpu_getDataIndex(b)];
    return (abs(va-vb) < 1);
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* changed){
    *changed = 0;
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    int3 pixel = {.x = threadIdx.x,
        .y = blockIdx.x, .z = blockIdx.y};
    int tid = getGlobalIdx_3D_3D();

    /*if(pixel.y == 0 && pixel.x == 0 && pixel.z == 0){
        int x = 2147483648/2;
        printf("Value: 1073741824\n");
        printf("Ints : %d\n", x);
    }*/

    if (0 > tid || DATA_SIZE <= tid){
        printf("We have tid out of boundary: %d\n", tid);
    }
    printf("tid: %d: .x = %d, .y = %d, .z = %d\n", tid, pixel.x, pixel.y, pixel.z);

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
                gpu_isPosInside(pos) &&
                //Check that it's not already been "discovered"
                !region[tid] &&
                //Check that the corresponding color values actually match
                abs(data[tid] - data[gpu_getDataIndex(pos)]) < 1){
                //then
                printf("Found neighbour!\n");
                region[tid] = NEW_VOX;
                atomicAdd(changed, 1);
            }
        }
    }

    //__syncthreads();
    *changed = 1;
    return;
}

unsigned char* grow_region_gpu(unsigned char* data){
    printf("\nEntered grow_region_gpu!\n");

    cudaEvent_t start, end;
    dim3 **sizes = getGridAndBlockSize(0);
    int changed = 1, *gpu_changed, rounds = 1;
    int3 seed = {.x = 50, .y = 300, .z = 300};
    unsigned char *cudaImage, *cudaRegion, *region;
    region = (unsigned char*) calloc(DATA_SIZE, sizeof(unsigned char));
    region[seed.z*IMAGE_SIZE + seed.y*DATA_DIM + seed.x] = NEW_VOX;
    printf("Done instantiating variables...\n");

    gEC(cudaMalloc(&gpu_changed, sizeof(int)));
    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaImage, sizeof(unsigned char)*DATA_SIZE));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, sizeof(unsigned char)*DATA_SIZE));

    printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaImage, data, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying image and region to device took %f ms\n",
        getCudaEventTime(start, end));

    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", sizes[0]->x, sizes[0]->y, sizes[0]->z);
    printf("block.x: %d, block.y: %d, block.z: %d\n", sizes[1]->x, sizes[1]->y, sizes[1]->z);

    int roundsSize = DATA_SIZE/rounds;
    for (int i = 0; (i < 3)/* && (changed)*/; ++i){
        printf("\nEntered #%d kernel outer-loop\n", i);
        for (int j = 0; j < rounds; ++j){
            gEC(cudaMemcpy(gpu_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));
            printf("Finished changed memcpy to device!\n");
            int tmp = changed;
            //region_grow_kernel<<<*sizes[0], *sizes[1]>>>(&cudaImage[roundsSize*j], &cudaRegion[roundsSize*j], gpu_changed);
            gEC(cudaMemcpy(&changed, gpu_changed, sizeof(int), cudaMemcpyDeviceToHost));
            tmp += changed;
            changed = tmp;
        }
        printf("Finished iteration %d of kernel outer-loop!\n", i);
    }

    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, sizeof(unsigned char)*DATA_SIZE, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("\nCopying region from device took %f ms\n", getCudaEventTime(start, end));

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
