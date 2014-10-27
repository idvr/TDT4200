#ifndef RAYCAST
#define RAYCAST
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include "bmp.h"

#define NEW_VOX 2
#define VISITED 1

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
extern size_t dataDim;
#define DATA_SIZE (DATA_DIM*DATA_DIM*DATA_DIM)
extern size_t dataSize;

// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
extern size_t imageDim;
#define IMAGE_SIZE (IMAGE_DIM*IMAGE_DIM)
extern size_t imageSize;

//Cuda texture
texture<int, cudaTextureType3D, cudaReadModelElementType> data_texture;

//Whether to abort when errors are found in above macro/function call
#define ERROR_ABORT 1
//For error-checking Nvidia CUDA calls
#define gEC(inpt) {gpuErrorCheck(inpt)}
#define gpuErrorCheck(inpt) {gpuAssert((inpt), __FILE__, __LINE__, ERROR_ABORT);}
//Function prototype for the above two defines
void gpuAssert(cudaError_t code, const char *file, int line, int abort);

// Stack for the serial region growing
typedef struct{
    int size;
    int buffer_size;
    int3* pixels;
} stack_t;

typedef struct{
    int size;
    int buffer_size;
    float* buffer;
} stack2_t;

//General support functions
stack_t* new_stack();
int3 pop(stack_t* stack);
void push(stack_t* stack, int3 p);
float pop(stack2_t* stack);
void destroy(stack2_t *stack);
float peek(stack2_t* stack, int pos);
int push(stack2_t* stack, float input);
stack2_t* new_time_stack(int start_size);

//Functions accessible by kernels
__device__ int getBlockId_3D();
__device__ int getGlobalIdx_3D_3D();
__device__ int getBlockThreadId_3D();
__host__ __device__ int index(int3 pos);
__host__ __device__ int inside(int3 pos);
__host__ __device__ int inside(float3 pos);
__host__ __device__ float3 normalize(float3 v);
__host__ __device__ int index(int z, int y, int x);
__host__ __device__ float3 add(float3 a, float3 b);
__host__ __device__ float3 scale(float3 a, float b);
__host__ __device__ float3 cross(float3 a, float3 b);
__host__ __device__ int3 getGlobalPos(int globalThreadId);
__host__ __device__ int similar(unsigned char* data, int3 a, int3 b);
//Trilinear interpolation
__host__ __device__ float value_at(float3 post, unsigned char* data);

//Print/get properties of Nvidia CUDA card
void print_properties();
int getAmountOfSMs(int device);
int getThreadsPerBlock(int device);
int getMaxThreadsPerSM(int device);
int getBlocksPerSM(int device, int dim);
int getKernelThreadAmount(dim3** sizes);
dim3** getGridsBlocksGrowRegion(int device);
dim3** getGridsBlocksRaycasting(int device);
void createCudaEvent(cudaEvent_t* cudaEvent);
void setCudaDevice(cudaDeviceProp* p, int device);
float getCudaEventTime(cudaEvent_t start, cudaEvent_t end);

//Generate input data for exercise
unsigned char func(int x, int y, int z);
unsigned char* create_data();

#endif
