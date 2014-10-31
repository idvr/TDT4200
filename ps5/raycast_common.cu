#include "raycast.cuh"

size_t dataDim = sizeof(uchar)*DATA_DIM;
size_t dataSize = sizeof(uchar)*DATA_SIZE;
size_t imageDim = sizeof(uchar)*IMAGE_DIM;
size_t imageSize = sizeof(uchar)*IMAGE_SIZE;

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

stack2_t* new_time_stack(int start_size){
    stack2_t* stack = (stack2_t*) malloc(sizeof(stack2_t));
    stack->size = 0;
    stack->buffer_size = start_size;
    stack->buffer = (float*) malloc(sizeof(float)*start_size);
    return stack;
}

int push(stack2_t* stack, float input){
    if(stack->buffer_size == stack->size){
        stack->buffer_size *= 2;
        float* temp = stack->buffer;
        stack->buffer = (float*) malloc(sizeof(float)*stack->buffer_size);
        memcpy(stack->buffer, temp, sizeof(sizeof(float)*stack->buffer_size/2));
        free(temp);
    }
    stack->buffer[stack->size] = input;
    stack->size += 1;
    return stack->size-1;
}

float pop(stack2_t* stack){
    stack->size -= 1;
    return stack->buffer[stack->size];
}

float peek(stack2_t* stack, int pos){
    if(0 <= pos && pos < stack->size){
        return stack->buffer[pos];
    } else{
        return 0.0/0.0;
    }
}

void destroy(stack2_t *stack){
    free(stack->buffer);
    free(stack);
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
        printf("# of blocks needed when using shared memory(no borders): %zd\n", sizeof(uchar)*((int) ceil(DATA_SIZE/p.sharedMemPerBlock)));

        printf("# of blocks needed when using shared memory w/borders: %d\n", (int) ceil(DATA_SIZE/(DATA_DIM*93)));
        printf("# of rows with borders per block: %d, total #rows: %d, size per row: %d\n", 93, 95, 514);

        printf("\n");
    }
}

// Fills data with values
uchar func(int x, int y, int z){
    uchar value = rand() % 20;

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

uchar* create_data(){
    uchar* data = (uchar*)malloc(sizeof(uchar) * DATA_DIM*DATA_DIM*DATA_DIM);

    for(int i = 0; i < DATA_DIM; i++){
        for(int j = 0; j < DATA_DIM; j++){
            for(int k = 0; k < DATA_DIM; k++){
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM + k]= func(k,j,i);
            }
        }
    }

    return data;
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
    gEC(cudaEventSynchronize(*event));
}

float getCudaEventTime(cudaEvent_t start, cudaEvent_t end){
    float result = -1;
    gEC(cudaEventElapsedTime(&result, start, end));
    return result;
}

dim3** getGridsBlocksGrowRegion(int device){
    dim3 grid, block;
    dim3 **sizes = (dim3**) malloc(sizeof(dim3*)*2);
    sizes[0] = (dim3*) malloc(sizeof(dim3));
    sizes[1] = (dim3*) malloc(sizeof(dim3));

    //Hardcoding blockdim values (8^3 = 512 = DATA_DIM)
    block.z = 8; grid.z = (DATA_DIM/block.z);
    block.y = 8; grid.y = (DATA_DIM/block.y);
    block.x = 16; grid.x = (DATA_DIM/block.x);

    memcpy(sizes[0], &grid, sizeof(dim3));
    memcpy(sizes[1], &block, sizeof(dim3));
    return sizes;
}

dim3** getGridsBlocksRaycasting(int device){
    dim3 grid, block;
    dim3 **sizes = (dim3**) malloc(sizeof(dim3*)*2);
    sizes[0] = (dim3*) malloc(sizeof(dim3));
    sizes[1] = (dim3*) malloc(sizeof(dim3));

    //Hardcoding blockdim values (16^2 = 512 = DATA_DIM)
    block.z = 2; grid.z = 1;
    block.y = 16;
    block.x = 16;
    int rest = IMAGE_SIZE - (block.x*block.y*block.z);
    grid.x = (int) sqrt(rest);
    rest -= (int) sqrt(rest);
    grid.y = rest;

    memcpy(sizes[0], &grid, sizeof(dim3));
    memcpy(sizes[1], &block, sizeof(dim3));
    return sizes;
}
