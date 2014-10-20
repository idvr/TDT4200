#include "raycast.h"

stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (int3*)malloc(sizeof(int3)*1024);

    return stack;
}

void push(stack_t* stack, int3 p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        int3* temp = stack->pixels;
        stack->pixels = (int3*)malloc(sizeof(int3)*stack->buffer_size);
        memcpy(stack->pixels, temp, sizeof(int3)*stack->buffer_size/2);
        free(temp);

    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}

int3 pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}

// float3 utilities
float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
}

void setCudaDevice(cudaDeviceProp* p, int device){
    gEC(cudaSetDevice(device));
    gEC(cudaGetDeviceProperties(p, device));
    gEC(cudaDeviceSynchronize());
}

int getAmountOfSMs(int device){
    cudaDeviceProp p;
    setCudaDevice(&p, device);
    return p.multiProcessorCount;
}

int getThreadsPerBlock(int device){
    cudaDeviceProp p;
    setCudaDevice(&p, device);
    return p.maxThreadsDim[0];
}

int getBlocksPerSM(int device, int dim){
    if(2 < dim){
        fprintf(stderr, "Not enough block dimensions in SM!! dim: %d\n", dim);
        exit(-1);
    }
    cudaDeviceProp p;
    setCudaDevice(&p, device);
    return p.maxGridSize[dim];
}

int getMaxThreadsPerSM(int device){
    cudaDeviceProp p;
    setCudaDevice(&p, device);
    return p.maxThreadsPerMultiProcessor;
}

// Prints CUDA device properties
void print_properties(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n\n", deviceCount);

    cudaDeviceProp p;
    for (int i = 0; i < deviceCount; ++i){
        int cudaReturnStatus = cudaSetDevice(i);
        if (cudaSuccess != cudaReturnStatus){
            printf("cudaSetDevice(%d) returned error\n", i);
            continue;
        }

        cudaReturnStatus = cudaGetDeviceProperties (&p, i);
        if (cudaSuccess != cudaReturnStatus){
            printf("cudaGetDeviceProperties(&p, %d) returned error\n", i);
            continue;
        }

        //If all went well, print info:
        printf("Device #%d, Name: %s\n" , (i+1), p.name);
        printf("Compute capability: %d.%d\n", p.major, p.minor);

        printf("#Threads per Warp: %d\n", p.warpSize);

        printf("Multiprocessor (SM/SMX) count: %d\n", p.multiProcessorCount);

        printf("Max #threads per SM/SMX: %d\n", p.maxThreadsPerMultiProcessor);

        printf("Total memory: %zd MiB \nShared memory per thread block in a SM/SMX: %zd KiB\n",
            p.totalGlobalMem/(1024*1024), p.sharedMemPerBlock/1024);

        printf("Max #registers per Block: %d\n", p.regsPerBlock);

        printf("Max threads per Blocks: ");
        for (int j = 0; j < 2; ++j){
            printf("%d, ", p.maxThreadsDim[j]);
        }printf("%d\n", p.maxThreadsDim[2]);

        printf("Max Grid Size: ");
        for (int j = 0; j < 2; ++j){
            printf("%d, ", p.maxGridSize[j]);
        }printf("%d\n", p.maxGridSize[2]);

        printf("Max Threads per Block: %d\n", p.maxThreadsPerBlock);

        printf("Are concurrent kernels supported?: %s\n",
            p.concurrentKernels ? "yes" : "no");

        if (p.asyncEngineCount){
            printf("Device can transfer data %s host and device while executing a kernel\n",
                p.asyncEngineCount == 2 ? "both ways between" : "one way between");
        } else{
            printf("Device cannot transfer data between host/device while a kernel is running\n");
        }

        printf("\n");
    }
}

// Fills data with values
unsigned char func(int x, int y, int z){
    unsigned char value = rand() % 20;

    int x1 = 300;
    int y1 = 400;
    int z1 = 100;
    float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 100){
        value  = 30;
    }

    x1 = 100;
    y1 = 200;
    z1 = 400;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 50){
        value = 50;
    }

    if(x > 200 && x < 300 && y > 300 && y < 500 && z > 200 && z < 300){
        value = 45;
    }
    if(x > 0 && x < 100 && y > 250 && y < 400 && z > 250 && z < 400){
        value =35;
    }
    return value;
}

unsigned char* create_data(){
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM);

    for(int i = 0; i < DATA_DIM; i++){
        for(int j = 0; j < DATA_DIM; j++){
            for(int k = 0; k < DATA_DIM; k++){
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM + k]= func(k,j,i);
            }
        }
    }

    return data;
}

// Checks if position is inside the volume (float3 and int3 versions)
int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

int index(int3 pos){
    return pos.z*DATA_DIM*DATA_DIM
            + pos.y*DATA_DIM + pos.x;
}

// Indexing function (note the argument order)
int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
float value_at(float3 pos, unsigned char* data){
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

// Check if two values are similar, threshold can be changed.
int similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[index(a)];
    unsigned char vb = data[index(b)];

    int i = abs(va-vb) < 1;
    return i;
}

void gpuAssert(cudaError_t code, const char *file, int line, int abort){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s, @%s:%d\n", cudaGetErrorString(code), file, line);
      if(abort){
        exit(code);
      }
   }
}

void createCudaEvent(cudaEvent_t* event){
    gEC(cudaEventCreate(event)); gEC(cudaEventRecord(*event, 0));
    gEC(cudaEventSynchronize(*event)); return;
}

float getCudaEventTime(cudaEvent_t start, cudaEvent_t end){
    float result = -1;
    gEC(cudaEventElapsedTime(&result, start, end));
    return result;
}

dim3** getGridAndBlockSize(int device){
    dim3 grid, block;
    int SMs = getAmountOfSMs(device),
        totThreads = getMaxThreadsPerSM(device)*SMs,
        totBlocks = totThreads/512; //Ref amount threads per block below
    dim3 **sizes = (dim3**) malloc(sizeof(dim3*)*2);
    sizes[0] = (dim3*) malloc(sizeof(dim3));
    sizes[1] = (dim3*) malloc(sizeof(dim3));
    printf("Done assigning to size!\n");

    //Hardcoding blockdim values (8^3 = 512 = DATA_DIM)
    block.x = 8; grid.x = 1;
    block.y = 8; grid.y = 1;
    block.z = 8; grid.z = 1;

    while(8 <= totBlocks){
        grid.x += 2;
        grid.y += 2;
        grid.z += 2;
        totBlocks /= 8;
    }
    /*if (totBlocks){
        printf("\t\t%d threads left, making %d blocks\n", totBlocks*512, totBlocks);
    }*/
    printf("Done assigning grid and block values!\n");

    printf("Before memcpy's!\n");
    memcpy(sizes[0], &grid, sizeof(dim3));
    memcpy(sizes[1], &block, sizeof(dim3));
    printf("Done with memcpy!\n");
    return sizes;
}

//################# Functions accessible by kernels ##############
__device__ int gpu_index(int3 pos){
    return pos.z*IMAGE_SIZE
        + pos.y*DATA_DIM + pos.x;
}

__device__ int gpu_inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__device__ int getGlobalThreadId_3D_3D(){
    return 0;
}

__device__ int getGlobalBlockId_3D_3D(){
    return blockIdx.x + (blockIdx.y*gridDim.x)
            + (blockIdx.z*gridDim.x*gridDim.y);
}

__device__ int getGlobalIdx_3D_3D(){
    int blockId = getGlobalBlockId_3D_3D();
    int threadId = threadIdx.x + (threadIdx.y*blockDim.x)
                    + (threadIdx.z*blockDim.x*blockDim.y)
                    + blockId*(blockDim.x*blockDim.y*blockDim.z);
    return threadId;
}

__host__ __device__ int3 getGlobalPos(int globalThreadId){
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
    unsigned char va = data[gpu_index(a)];
    unsigned char vb = data[gpu_index(b)];
    return (abs(va-vb) < 1);
}
