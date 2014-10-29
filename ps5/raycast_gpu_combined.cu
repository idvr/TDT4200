#include "raycast.cuh"

__device__ int getBlockId(){
    return blockIdx.x + (blockIdx.y*gridDim.x)
            + (blockIdx.z*gridDim.x*gridDim.y);
}

__device__ int getGlobalIdx(){
    int blockId = getBlockId();
    int threadId = getThreadInBlockId() +
            blockId*(blockDim.x*blockDim.y*blockDim.z);
    return threadId;
}

__device__ int getThreadInBlockId(){
    return threadIdx.x + (threadIdx.y*blockDim.x)
            + (threadIdx.z*blockDim.x*blockDim.y);
}

__device__ float valueAtData(float3 pos){
    return tex3D(data_texture,
        pos.x, pos.y, pos.z)*255.f;
}

__device__ float valueAtRegion(float3 pos){
    return tex3D(region_texture,
        pos.x, pos.y, pos.z)*255.f;
}

__device__ int insideThreadBlock(int3 pos){
    int x = (pos.x >= 0 && pos.x < blockIdx.x);
    int y = (pos.y >= 0 && pos.y < blockIdx.y);
    int z = (pos.z >= 0 && pos.z < blockIdx.z);
    return x && y && z;
}

__device__ int3 getThreadInBlockPos(int tid){
    int3 pos = {.y = 0, .z = 0,
        .x = getThreadInBlockId()};
    int zd = gridDim.y*gridDim.z;
    int yd = gridDim.y;
    if ((zd-1) > pos.x){
        pos.z = pos.x/zd;
        pos.x -= pos.z*zd;
    }
    if ((yd-1) > pos.x){
        pos.y = pos.x/yd;
        pos.x -= pos.y*yd;
    }
    return pos;
}

__device__ int getThreadInBlockIndex(int3 pos){
    int tid = pos.x;
    tid += pos.y*gridDim.y;
    tid += pos.z*gridDim.y*gridDim.z;
    return tid;
}

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

__host__ __device__ int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__global__ void raycast_kernel_texture(uchar* image){
    int tid = getGlobalIdx();
    int y = getBlockId() - (IMAGE_DIM/2);
    int x = getThreadInBlockId() - (IMAGE_DIM/2);
    float step_size = 0.5, fov = 3.14/4, color = 0,
            pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float3 z_axis = {.x=0, .y=0, .z = 1};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 camera = {.x=1000, .y=1000, .z=1000};

    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    up = normalize(up);
    right = normalize(right);
    forward = normalize(forward);

    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center,
        scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    for (int i = 0; 255 > color && 5000 > i; ++i){
        pos = add(pos, scale(ray, step_size));
        if(!inside(pos)){
            continue;
        }
        int r = valueAtRegion(pos);
        color += valueAtData(pos)*(0.01+r);
    }
    image[tid] = min(color, 255.f);
}

uchar* raycast_gpu_texture(uchar* data, uchar* region){
    cudaEvent_t start, end;
    uchar *cudaImage;
    cudaArray *cudaData, *cudaRegion;
    dim3 **sizes = getGridsBlocksRaycasting(0);
    uchar *image = (uchar*) malloc(imageSize);
    cudaMemcpy3DParms copyData = {0}, copyRegion = {0};
    const cudaExtent volumeSize = make_cudaExtent(DATA_DIM, DATA_DIM, DATA_DIM);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    //printf("Finished creating variables.\n");

    gEC(cudaMalloc(&cudaImage, imageSize));
    gEC(cudaMemset(cudaImage, 0, imageSize));
    gEC(cudaMalloc3DArray(&cudaData, &channelDesc, volumeSize));
    gEC(cudaMalloc3DArray(&cudaRegion, &channelDesc, volumeSize));

    //For data
    copyData.dstArray = cudaData;
    copyData.extent = volumeSize;
    copyData.kind = cudaMemcpyHostToDevice;
    copyData.srcPtr = make_cudaPitchedPtr(data,
        volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);

    //For region
    copyRegion.extent = volumeSize;
    copyRegion.dstArray = cudaRegion;
    copyRegion.kind = cudaMemcpyHostToDevice;
    copyRegion.srcPtr = make_cudaPitchedPtr(region,
        volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);

    data_texture.normalized = false;
    data_texture.filterMode = cudaFilterModeLinear;
    data_texture.addressMode[0] = cudaAddressModeBorder;
    data_texture.addressMode[1] = cudaAddressModeBorder;
    data_texture.addressMode[2] = cudaAddressModeBorder;

    region_texture.normalized = false;
    region_texture.filterMode = cudaFilterModeLinear;
    region_texture.addressMode[0] = cudaAddressModeBorder;
    region_texture.addressMode[1] = cudaAddressModeBorder;
    region_texture.addressMode[2] = cudaAddressModeBorder;
    //printf("Texture variables/structs set up.\n");

    createCudaEvent(&start);
    gEC(cudaMemcpy3D(&copyData));
    gEC(cudaMemcpy3D(&copyRegion));
    gEC(cudaBindTextureToArray(data_texture, cudaData, channelDesc));
    gEC(cudaBindTextureToArray(region_texture, cudaRegion, channelDesc));
    createCudaEvent(&end);
    printf("Copying and binding data and region to textures took %.4f ms\n",
        getCudaEventTime(start, end));

    createCudaEvent(&start);
    raycast_kernel_texture<<<*sizes[0], *sizes[1]>>>(cudaImage);
    createCudaEvent(&end);
    printf("Calling kernel took %.4f ms\n", getCudaEventTime(start, end));

    //Copy image back from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(image, cudaImage, imageSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying image from device took %.4f ms\n",
        getCudaEventTime(start, end));

    gEC(cudaFree(cudaImage));
    gEC(cudaFreeArray(cudaData));
    gEC(cudaFreeArray(cudaRegion));
    return image;
}

__global__ void region_grow_kernel_shared(uchar* data, uchar* region, int* changed){
    extern __shared__ unsigned char sdata[];
    //Load into shared memory
    __shared__ int3 pos, pixel;
    __shared__ unsigned int pos_id, bid, tid;
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    bid = getBlockId();
    tid = getThreadInBlockId();
    pixel = getThreadInBlockPos(tid);
    sdata[tid] = data[tid+bid];
    //Constant factor with 512 threads per block of shared memory used:
    //3*6 (dx,dy,dz) + 8*512 (thread specific helpers) = 18 + 4096 = 4114
    //sdata size = (x)(y)(z) = 8^3 with 8 == (x&y&z)
    __syncthreads();

    //Process shared memory (non-synchronizing version)
    if(NEW_VOX == region[tid+bid]){
        region[tid+bid] = VISITED;
        for (int i = 0; i < 6; ++i){
            pos = pixel;
            pos.x += dx[i];
            pos.y += dy[i];
            pos.z += dz[i];
            pos_id = getThreadInBlockIndex(pos);
            if (insideThreadBlock(pos)  &&
                !region[pos_id+bid]     &&
                abs(sdata[tid] - sdata[pos_id]) < 1){
                //Write results
                region[pos_id+bid] = NEW_VOX;
                *changed = 1;
            }
        }
    }
}

uchar* grow_region_gpu_shared(uchar* data){
    //8 rows 8 heigh of 8 depth threads per block
    cudaEvent_t start, end;
    int changed = 1, *gpu_changed;
    stack2_t *time_stack = new_time_stack(256);
    dim3 **sizes = getGridsBlocksGrowRegion(0);


    int3 seed = {.x = 50, .y = 300, .z = 300};
    uchar *cudaData, *cudaRegion, *region;

    region = (uchar*) calloc(sizeof(uchar), DATA_SIZE);
    region[seed.z*IMAGE_SIZE + seed.y*DATA_DIM + seed.x] = NEW_VOX;
    //printf("Done instantiating variables...\n");

    printf("%s\n", cudaGetLastError());
    gEC(cudaMalloc(&gpu_changed, sizeof(int)));
    printf("%s\n", cudaGetLastError());
    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaData, dataSize));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, dataSize));
    gEC(cudaMemset(cudaRegion, 0, dataSize));
    //printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaData, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying data and region to device took %.4f ms\n",
        getCudaEventTime(start, end));

    for (int i = 0; changed && (256 > i); ++i){
        gEC(cudaMemset(gpu_changed, 0, sizeof(int)));
        createCudaEvent(&start);
        region_grow_kernel_shared<<<*sizes[0], *sizes[1]>>>(&cudaData[0], &cudaRegion[0], gpu_changed);
        createCudaEvent(&end);
        push(time_stack, getCudaEventTime(start, end));
        gEC(cudaMemcpy(&changed, gpu_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }

    float sum = 0;
    for (int i = 0; i < time_stack->size; ++i){
        sum += peek(time_stack, i);
    }
    printf("%d kernel calls took a sum total of %.4f ms\n", time_stack->size, sum);
    destroy(time_stack);

    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, dataSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("\nCopying region from device took %.4f ms\n", getCudaEventTime(start, end));

    gEC(cudaFree(cudaData));
    gEC(cudaFree(cudaRegion));
    gEC(cudaFree(gpu_changed));

    return region;
}

int main(int argc, char** argv){
    printf("\nStarting program...\n\n");
    //print_properties();

    uchar* data = create_data();
    printf("Done creating data\n\n");

    uchar* region = grow_region_gpu_shared(data);
    printf("Done creating region\n\n");

    uchar* image = raycast_gpu_texture(data, region);
    printf("Done creating image\n\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_combined_out.bmp");
    printf("Done with program\n\n");

    return 0;
}
