#include <stdio.h>
#include <stdlib.h>
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

// Stack for the serial region growing
typedef struct{
    int size;
    int buffer_size;
    int3* pixels;
} stack_t;

//General support functions
int inside(int3 pos);
stack_t* new_stack();
int inside(float3 pos);
int3 pop(stack_t* stack);
float3 normalize(float3 v);
int index(int z, int y, int x);
float3 add(float3 a, float3 b);
float3 scale(float3 a, float b);
float3 cross(float3 a, float3 b);
void push(stack_t* stack, int3 p);
int getKernelThreadAmount(dim3** sizes);

//Trilinear interpolation
float value_at(float3 post, unsigned char* data);

//Generate input data for exercise
unsigned char func(int x, int y, int z);
unsigned char* create_data();
int similar(unsigned char* data, int3 a, int3 b);

//Common function calls for exercise
unsigned char* grow_region_serial(unsigned char* data, unsigned char* region);
unsigned char* raycast_serial(unsigned char* data, unsigned char* region);
