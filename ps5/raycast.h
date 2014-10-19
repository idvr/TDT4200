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
#define DATA_SIZE (512*512*512)
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define IMAGE_SIZE (512*512)
//Whether to abort when errors are found in above macro/function call
#define ERROR_ABORT 0
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

//General support functions
stack_t* new_stack();
void push(stack_t* stack, int3 p);
int3 pop(stack_t* stack);
float3 cross(float3 a, float3 b);
float3 normalize(float3 v);
float3 add(float3 a, float3 b);
float3 scale(float3 a, float b);
int inside(int3 pos);
int inside(float3 post);
int index(int z, int y, int x);

//Functions accessible by kernels
/*__device__ int gpu_index(int3 pos);
__device__ int gpu_inside(int3 pos);
__device__ int gpu_similar(unsigned char* data, int3 a, int3 b);*/

//Print properties of Nvidia CUDA card
void print_properties();

//Generate input data for exercise
unsigned char func(int x, int y, int z);
unsigned char* create_data();
int similar(unsigned char* data, int3 a, int3 b);

//Trilinear interpolation
float value_at(float3 post, unsigned char* data);

//Common function calls for exercise
unsigned char* grow_region_serial(unsigned char* data, unsigned char* region);
unsigned char* raycast_serial(unsigned char* data, unsigned char* region);
void createCudaEvent(cudaEvent_t* cudaEvent);
float getCudaEventTime(cudaEvent_t start, cudaEvent_t end);

#endif
