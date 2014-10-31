#include "raycast.cuh"

// float3 utilities
__host__ __device__
float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;
    return c;
}

__host__ __device__
float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;
    return v;
}

__host__ __device__
float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__
float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

// Trilinear interpolation
__host__ __device__
float value_at(float3 pos, uchar* data){
    if(!inside(pos)){
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

    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;

    return c0;
}

__device__
int getBlockId(){
    return (blockIdx.x +
        (blockIdx.y*gridDim.x) +
        (blockIdx.z*gridDim.x*gridDim.y)) /*
        (blockDim.x*blockDim.y*blockDim.z)*/;
}

__device__
int getThreadId(){
    return threadIdx.x +
        (threadIdx.y*blockDim.x) +
        (threadIdx.z*blockDim.x*blockDim.y);
}

__device__
int insideThreadBlock(int3 pos){
    int x = (pos.x >= 0 && pos.x < blockDim.x);
    int y = (pos.y >= 0 && pos.y < blockDim.y);
    int z = (pos.z >= 0 && pos.z < blockDim.z);
    return x && y && z;
}

__device__
int3 getThreadPosInBlock(){
    int3 pos = {
        .x = threadIdx.x,
        .y = threadIdx.y,
        .z = threadIdx.z};
    return pos;
}

__device__
int getThreadInBlockIndex(int3 pos){
    if (!insideThreadBlock(pos)){
        return 0;
    }
    return pos.x +
        (pos.y*blockDim.x) +
        (pos.z*blockDim.x*blockDim.y);
}

__host__ __device__
int index(int3 pos){
    return pos.z*IMAGE_SIZE
        + pos.y*DATA_DIM + pos.x;
}

__host__ __device__
int index(int z, int y, int x){
    return z*IMAGE_SIZE
        + y*DATA_DIM + x;
}

__device__
int getGlobalIdx(){
    return getThreadId() +
        (getBlockId() *
        (blockDim.x*blockDim.y*blockDim.z));
}

__device__
int isThreadOnBlockEdge(int3 voxel){
    //Check if thread is along one border-edge of the cube or another
    if (0 == voxel.x || //if along plane x == 0
        0 == voxel.y || //if along plane y == 0
        0 == voxel.z || //if along plane z == 0
        (blockDim.x-1 == voxel.x)|| //if along plane x == max value
        (blockDim.y-1 == voxel.y)|| //if along plane y == max value
        (blockDim.z-1 == voxel.z)){ //if along plane z == max value
        return 1;
    }
    return 0;
}

__host__ __device__
int similar(uchar* data, int3 a, int3 b){
    uchar va = data[index(a)];
    uchar vb = data[index(b)];
    return (abs(va-vb) < 1);
}

__device__
int similar(uchar* data, int idx, int idy){
    uchar va = data[idx];
    uchar vb = data[idy];
    return (abs(va-vb) < 1);
}

__host__ __device__
int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__host__ __device__
int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__global__
void raycast_kernel(uchar* data, uchar* image, uchar* region){
    int tid = getGlobalIdx();
    int y = getBlockId() - (IMAGE_DIM/2);
    int x = getThreadId() - (IMAGE_DIM/2);
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

    float color = 0;
    for (int i = 0; 255 > color && (5000 > i); ++i){
        pos = add(pos, scale(ray, step_size));
        int r = value_at(pos, region);
        color += value_at(pos, data)*(0.01+r);
    }
    image[tid] = min(color, 255.f);

    return;
}

uchar* raycast_gpu(uchar* data, uchar* region){
    cudaEvent_t start, end;
    dim3 **sizes = getGridsBlocksRaycasting(0);
    uchar *cudaImage, *cudaRegion, *cudaData;
    uchar *image = (uchar*) malloc(imageSize);
    //printf("Done instantiating variables...\n");

    //Malloc image++ on cuda device
    gEC(cudaMalloc(&cudaData, dataSize));
    gEC(cudaMalloc(&cudaImage, imageSize));
    gEC(cudaMalloc(&cudaRegion, dataSize));
    gEC(cudaMemset(cudaImage, 0, imageSize));
    //printf("Done mallocing on CUDA device!\n");

    //Copy data and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaData, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying data and region to device took %.4f ms\n",
        getCudaEventTime(start, end));

    createCudaEvent(&start);
    raycast_kernel<<<*sizes[0], *sizes[1]>>>(cudaData, cudaImage, cudaRegion);
    createCudaEvent(&end);
    printf("Calling kernel took %.4f ms\n", getCudaEventTime(start, end));

    //Copy image back from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(image, cudaImage, imageSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying image from device took %.4f ms\n",
        getCudaEventTime(start, end));

    gEC(cudaFree(cudaData));
    gEC(cudaFree(cudaImage));
    gEC(cudaFree(cudaRegion));

    return image;
}

__host__ __device__
int3 getGlobalPos(int globalThreadId){
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

__global__
void region_grow_kernel_shared(uchar* data, uchar* region, int* changed){
    __shared__ uchar sdata[1000];
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    unsigned int tid = getThreadId();
    int3 blockVox = getThreadPosInBlock();
    unsigned int globalIdx = getGlobalIdx();
    int3 globalVox = getGlobalPos(globalIdx);

    //Load into shared memory
    sdata[tid] = data[globalIdx];
    __syncthreads();

    //If already discovered or not yet (maybe never) reached; skip it
    if (!inside(globalVox) || NEW_VOX != region[globalIdx]){
        return;
    }
    region[globalIdx] = VISITED;

    for (int i = 0; i < 6; ++i){
        int3 curPos = blockVox;
        int3 globalPos = globalVox;
        curPos.x += dx[i];
        curPos.y += dy[i];
        curPos.z += dz[i];
        globalPos.x += dx[i];
        globalPos.y += dy[i];
        globalPos.z += dz[i];

        int curIndex = getThreadInBlockIndex(curPos);
        unsigned int globalIndex = index(globalPos);

        //If outside or region != 0; skip it
        if (!inside(globalPos) || region[globalIndex]){
            continue;
        }

        //if curPos is a voxel on cube outermost edge(s) and similar == 0
        if (isThreadOnBlockEdge(curPos)){
            if(!similar(data, globalIdx, globalIndex)){
                continue;
            }
        } else if (!similar(sdata, tid, curIndex)){
            //If curPos not a voxel on cube outermost edge(s)
            continue;
        }

        region[globalIndex] = NEW_VOX;
        *changed = 1;
    }
}

uchar* grow_region_gpu_shared(uchar* data){
    cudaEvent_t start, end;
    int changed = 1, *gpu_changed, itrs = 256;
    dim3 **sizes = getGridsBlocksGrowRegion(0);
    stack2_t *time_stack = new_time_stack(itrs);
    uchar *cudaData, *cudaRegion, *region;

    region = (uchar*) calloc(sizeof(uchar), DATA_SIZE);
    region[300*IMAGE_SIZE + 300*DATA_DIM + 50] = NEW_VOX;
    //printf("Done instantiating variables...\n");

    gEC(cudaMalloc(&gpu_changed, sizeof(int)));
    //Malloc image on cuda device
    gEC(cudaMalloc(&cudaData, dataSize));
    //Malloc region on cuda device
    gEC(cudaMalloc(&cudaRegion, dataSize));
    //printf("Done mallocing on CUDA device!\n");

    //Copy image and region over to device
    createCudaEvent(&start);
    gEC(cudaMemcpy(cudaData, data, dataSize, cudaMemcpyHostToDevice));
    gEC(cudaMemcpy(cudaRegion, region, dataSize, cudaMemcpyHostToDevice));
    createCudaEvent(&end);
    printf("Copying data and region to device took %.4f ms\n",
        getCudaEventTime(start, end));

    for (int i = 0; changed && (itrs > i); ++i){
        gEC(cudaMemset(gpu_changed, 0, sizeof(int)));
        createCudaEvent(&start);
        region_grow_kernel_shared<<<*sizes[0], *sizes[1]>>>(
            cudaData, cudaRegion, gpu_changed);
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

    uchar* image = raycast_gpu(data, region);
    printf("Done creating image\n\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_shared_out.bmp");
    printf("Done with program\n\n");

    return 0;
}
