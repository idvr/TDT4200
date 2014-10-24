#include "raycast.cuh"

// float3 utilities
__host__ __device__ float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;
    return c;
}

__host__ __device__ float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;
    return v;
}

__host__ __device__ float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

// Trilinear interpolation
__device__ float value_at(float3 pos, unsigned char* data){
    if(!gpu_isPosInside(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx*data[gpu_getIndex(z,y,x)] + (1-rx)*data[gpu_getIndex(z,y,x_u)];
    float a1 = rx*data[gpu_getIndex(z,y_u,x)] + (1-rx)*data[gpu_getIndex(z,y_u,x_u)];
    float a2 = rx*data[gpu_getIndex(z_u,y,x)] + (1-rx)*data[gpu_getIndex(z_u,y,x_u)];
    float a3 = rx*data[gpu_getIndex(z_u,y_u,x)] + (1-rx)*data[gpu_getIndex(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;

    return c0;
}

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

__device__ int gpu_getIndex(int z, int y, int x){
    return z*IMAGE_SIZE
        + y*DATA_DIM + x;
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

__device__ int gpu_isPosInside(float3 pos){
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
    dim3 **sizes = getGridsBlocksGrowRegion(0);
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
    int tid = getGlobalIdx_3D_3D();
    int y = getBlockId_3D() - (IMAGE_DIM/2);
    int x = getBlockThreadId_3D() - (IMAGE_DIM/2);
    float3 z_axis = {.x=0, .y=0, .z = 1};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 camera = {.x=1000, .y=1000, .z=1000};

    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    up = normalize(up);
    right = normalize(right);
    forward = normalize(forward);

    float fov = 3.14/4;
    float step_size = 0.5;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);

    //Do the raycasting
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;


    for (int i = 0; 255 > image[tid] && (5000 > i); ++i){
        pos = add(pos, scale(ray, step_size));
        int r = value_at(pos, region);
        image[tid] += value_at(pos, data)*(0.01+r);
    }

    return;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    cudaEvent_t start, end;
    dim3 **sizes = getGridsBlocksRaycasting(0);
    unsigned char *cudaImage, *cudaRegion, *cudaData;
    unsigned char *image = (unsigned char*) malloc(imageSize);
    //printf("Done instantiating variables...\n");

    //Malloc image++ on cuda device
    gEC(cudaMalloc(&cudaData, dataSize));
    gEC(cudaMalloc(&cudaImage, dataSize));
    gEC(cudaMalloc(&cudaRegion, dataSize));
    gEC(cudaMemset(cudaData, 0, dataSize));
    //printf("Done mallocing on CUDA device!\n");

    //Copy data and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaData, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying data and region to device took %f ms\n\n",
        getCudaEventTime(start, end));

    raycast_kernel<<<*sizes[0], *sizes[1]>>>(cudaData, cudaImage, cudaRegion);

    //Copy image back from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(image, cudaImage, imageSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying image from device took %f ms\n\n",
        getCudaEventTime(start, end));

    gEC(cudaFree(cudaData));
    gEC(cudaFree(cudaImage));
    gEC(cudaFree(cudaRegion));

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
