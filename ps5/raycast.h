#ifndef RAYCAST
#define RAYCAST
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512

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
int inside(float3 post);
int inside(int3 post);
//NOTE ARGUMENT ORDER FOR BELOW FUNCTION
int index(int z, int y, int x);

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

#endif
