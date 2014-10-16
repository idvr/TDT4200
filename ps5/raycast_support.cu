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

        printf("Total memory: %zdGiB \nShared memory per block: %zdKiB\n", p.totalGlobalMem/(1024*1024*1024), p.sharedMemPerBlock/1024);

        printf("#Threads per Warp: %d\n", p.warpSize);

        printf("Multiprocessor (SM/SMX) count: %d\n", p.multiProcessorCount);

        printf("Max threads per Blocks: ");
        for (int j = 0; j < 2; ++j){
            printf("%d, ", p.maxThreadsDim[j]);
        }printf("%d\n", p.maxThreadsDim[2]);

        printf("Max Grid Size: ");
        for (int j = 0; j < 2; ++j){
            printf("%d, ", p.maxGridSize[j]);
        }printf("%d\n", p.maxGridSize[2]);

        printf("Max Threads per Block: %d\n", p.maxThreadsPerBlock);

        printf("Are concurrent kernels supported?: %s\n", p.concurrentKernels ? "yes" : "no");

        printf("\n\n");
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
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM+ k]= func(k,j,i);
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
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

    int i = abs(va-vb) < 1;
    return i;
}
