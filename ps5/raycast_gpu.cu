#include "raycast.cuh"

__device__ int getBlockId_3D(){
    return blockIdx.x + (blockIdx.y*gridDim.x)
            + (blockIdx.z*gridDim.x*gridDim.y);
}

__device__ int getBlockThreadId_3D(){
    return threadIdx.x + (threadIdx.y*blockDim.x)
            + (threadIdx.z*blockDim.x*blockDim.y);
}

__device__ int gpu_getIndex(int3 pos){
    return pos.z*IMAGE_SIZE
        + pos.y*DATA_DIM + pos.x;
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
    unsigned char va = data[gpu_getIndex(a)];
    unsigned char vb = data[gpu_getIndex(b)];
    return (abs(va-vb) < 1);
}

__device__ int gpu_isPosInside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* changed){
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    int tid = getGlobalIdx_3D_3D();
    int3 pixel = getGlobalPos(tid);

    if(NEW_VOX == region[tid]){
        //printf("Entered first if!\n");
        //printf("tid: .x=%d, .y=%d, .z=%d\n", pixel.x, pixel.y, pixel.z);
        int3 pos; int pos_id;
        region[tid] = VISITED;
        for (int i = 0; i < 6; ++i){
            pos = pixel;
            pos.x += dx[i];
            pos.y += dy[i];
            pos.z += dz[i];
            pos_id = gpu_getIndex(pos);
            if (//Check that pos pixel is inside image/region
                gpu_isPosInside(pos) &&
                //Check that it's not already been "discovered"
                !region[pos_id] &&
                //Check that the corresponding color values actually match
                abs(data[tid] - data[pos_id]) < 1){
                //printf("Found neighbour! changed: %d\n", (*changed)+1);
                region[pos_id] = NEW_VOX;
                //atomicAdd(changed, 1);
                *changed = 1;
            }
        }
    }

    //__syncthreads();
    //*changed = 1;
    return;
}

unsigned char* grow_region_gpu(unsigned char* data){
    printf("\nEntered grow_region_gpu!\n");

    cudaEvent_t start, end;
    int changed = 1, *gpu_changed;
    dim3 **sizes = getGridAndBlockSize(0);
    int3 seed = {.x = 50, .y = 300, .z = 300};
    unsigned char *cudaImage, *cudaRegion, *region;

    region = (unsigned char*) malloc(dataSize);
    region[seed.z*IMAGE_SIZE + seed.y*DATA_DIM + seed.x] = NEW_VOX;
    //printf("Done instantiating variables...\n");

    gEC(cudaMalloc(&gpu_changed, sizeof(int)));
    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaImage, dataSize));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, dataSize));
    gEC(cudaMemset(cudaRegion, 0, dataSize));

    //printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaImage, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying image and region to device took %f ms\n\n",
        getCudaEventTime(start, end));

    //printf("grid.x: %d, grid.y: %d, grid.z: %d\n", sizes[0]->x, sizes[0]->y, sizes[0]->z);
    //printf("block.x: %d, block.y: %d, block.z: %d\n", sizes[1]->x, sizes[1]->y, sizes[1]->z);

    createCudaEvent(&start);
    for (int i = 0; changed && (175 > i); ++i){
        //printf("\nEntered #%d kernel outer-loop\n", i+1);
        gEC(cudaMemset(gpu_changed, 0, sizeof(int)));
        //printf("Finished changed memcpy to device!\n");
        region_grow_kernel<<<*sizes[0], *sizes[1]>>>(&cudaImage[0], &cudaRegion[0], gpu_changed);
        gEC(cudaMemcpy(&changed, gpu_changed, sizeof(int), cudaMemcpyDeviceToHost));
        //printf("changed: %d\n", *changed);
        //printf("Finished iteration %d of kernel outer-loop!\n", i+1);
    }
    createCudaEvent(&end);
    printf("Calling kernel x amount of times took %f ms\n\n", getCudaEventTime(start, end));

    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, dataSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("\nCopying region from device took %f ms\n\n", getCudaEventTime(start, end));

    gEC(cudaFree(cudaImage));
    gEC(cudaFree(cudaRegion));
    gEC(cudaFree(gpu_changed));

    return region;
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    //blah
    return;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    cudaEvent_t start, end;
    dim3 **sizes = getGridAndBlockSize(0);
    unsigned char *cudaImage, *cudaRegion, *cudaData;
    unsigned char *image = (unsigned char*) malloc(imageSize);
    //printf("Done instantiating variables...\n");

    //Malloc image++ on cuda device
    gEC(cudaMalloc(&cudaData, dataSize));
    gEC(cudaMalloc(&cudaImage, dataSize));
    gEC(cudaMalloc(&cudaRegion, dataSize));
    gEC(cudaMemset(cudaData, 0, dataSize));
    //printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaData, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying data and region to device took %f ms\n\n",
        getCudaEventTime(start, end));

    return image;
}

int main(int argc, char** argv){
    /*print_properties();

    printf("Done printing properties\n");*/

    unsigned char* data = create_data();

    printf("Done creating data\n");


    unsigned char* region = grow_region_gpu(data);
    //printf("grow_region_gpu() took %f ms\n", ms_time);

    printf("Done creating region\n");

    unsigned char* image = raycast_gpu(data, region);
    /*printf("raycast_gpu() took %f ms\n", ms_time);*/

    printf("Done creating image\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_out.bmp");

    printf("Done with program\n");
    return 0;
}
