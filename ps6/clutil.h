#ifndef CLUTIL_H
#define CLUTIL_H
#include <CL/cl.h>
#include <stdio.h>

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
#define DATA_SIZE (DATA_DIM*DATA_DIM*DATA_DIM)

// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define IMAGE_SIZE (IMAGE_DIM*IMAGE_DIM)

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
#define DATA_SIZE (DATA_DIM*DATA_DIM*DATA_DIM)
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define IMAGE_SIZE (IMAGE_DIM*IMAGE_DIM)

//Originals
const char *clErrorStr(cl_int err);
void clError(char *s, cl_int err);
void printPlatformInfo(cl_platform_id platform);
void printDeviceInfo(cl_device_id device);
cl_kernel buildKernel(char* sourceFile, char* kernelName, char* options, cl_context context, cl_device_id device);

#endif
