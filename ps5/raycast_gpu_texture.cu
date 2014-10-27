#include "raycast.cuh"

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

// Trilinear interpolation
__host__ __device__ float value_at(float3 pos, unsigned char* data){
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

__device__ int getBlockId_3D(){
    return blockIdx.x + (blockIdx.y*gridDim.x)
            + (blockIdx.z*gridDim.x*gridDim.y);
}

__device__ int getBlockThreadId_3D(){
    return threadIdx.x + (threadIdx.y*blockDim.x)
            + (threadIdx.z*blockDim.x*blockDim.y);
}

__host__ __device__ int index(int3 pos){
    return pos.z*IMAGE_SIZE
        + pos.y*DATA_DIM + pos.x;
}

__host__ __device__ int index(int z, int y, int x){
    return z*IMAGE_SIZE
        + y*DATA_DIM + x;
}

__device__ int getGlobalIdx_3D_3D(){
    int blockId = getBlockId_3D();
    int threadId = getBlockThreadId_3D() +
            blockId*(blockDim.x*blockDim.y*blockDim.z);
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

__host__ __device__ int similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[index(a)];
    unsigned char vb = data[index(b)];
    return (abs(va-vb) < 1);
}

__host__ __device__ int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

__host__ __device__ int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    return x && y && z;
}

// Serial ray casting
unsigned char* raycast_serial(unsigned char* data, unsigned char* region){
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_SIZE);
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    int cntr = 0;
    for(int y = -(IMAGE_DIM/2); y < (IMAGE_DIM/2); y++){
        for(int x = -(IMAGE_DIM/2); x < (IMAGE_DIM/2); x++){
            float3 screen_center = add(camera, forward);
            float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
            ray = add(ray, scale(camera, -1));
            ray = normalize(ray);
            float3 pos = camera;
            int i = 0;
            float color = 0;
            while(color < 255 && i < 5000){
                i++;
                pos = add(pos, scale(ray, step_size));   // Update position
                int r = value_at(pos, region);           // Check if we're in the region
                color += value_at(pos, data)*(0.01 + r) ;// Update the color based on data value, and if we're in the region
            }
            image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
        }
        cntr++;
        if (0 == cntr%10){
            printf("Done with image row #%d\n", y+(IMAGE_DIM/2));
        }
    }
    return image;
}

// Serial region growing, same algorithm as in assignment 2
unsigned char* grow_region_serial(unsigned char* data){
    unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);

    stack_t* stack = new_stack();

    int3 seed = {.x=50, .y=300, .z=300};
    push(stack, seed);
    region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 1;

    int dx[6] = {-1,1,0,0,0,0};
    int dy[6] = {0,0,-1,1,0,0};
    int dz[6] = {0,0,0,0,-1,1};

    while(stack->size > 0){
        int3 pixel = pop(stack);
        for(int n = 0; n < 6; n++){
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if(!inside(candidate)){
                continue;
            }

            if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
                continue;
            }

            if(similar(data, pixel, candidate)){
                push(stack, candidate);
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 1;
            }
        }
    }

    return region;
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* changed){
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    int tid = getGlobalIdx_3D_3D();
    int3 pixel = getGlobalPos(tid);

    if(NEW_VOX == region[tid]){
        int3 pos; int pos_id;
        region[tid] = VISITED;
        for (int i = 0; i < 6; ++i){
            pos = pixel;
            pos.x += dx[i];
            pos.y += dy[i];
            pos.z += dz[i];
            pos_id = index(pos);
            if (inside(pos)     &&
                !region[pos_id] &&
                abs(data[tid] - data[pos_id]) < 1){
                region[pos_id] = NEW_VOX;
                *changed = 1;
            }
        }
    }
    return;
}

unsigned char* grow_region_gpu(unsigned char* data){
    cudaEvent_t start, end;
    int changed = 1, *gpu_changed;
    stack2_t *time_stack = new_time_stack(175);
    dim3 **sizes = getGridsBlocksGrowRegion(0);
    int3 seed = {.x = 50, .y = 300, .z = 300};
    unsigned char *cudaData, *cudaRegion, *region;

    region = (unsigned char*) calloc(sizeof(unsigned char), DATA_SIZE);
    region[seed.z*IMAGE_SIZE + seed.y*DATA_DIM + seed.x] = NEW_VOX;
    //printf("Done instantiating variables...\n");

    gEC(cudaMalloc(&gpu_changed, sizeof(int)));
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
    printf("Copying data and region to device took %f ms\n\n",
        getCudaEventTime(start, end));

    for (int i = 0; changed && (175 > i); ++i){
        gEC(cudaMemset(gpu_changed, 0, sizeof(int)));
        createCudaEvent(&start);
        region_grow_kernel<<<*sizes[0], *sizes[1]>>>(&cudaData[0], &cudaRegion[0], gpu_changed);
        createCudaEvent(&end);
        push(time_stack, getCudaEventTime(start, end));
        gEC(cudaMemcpy(&changed, gpu_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }

    float sum = 0;
    for (int i = 0; i < time_stack->size; ++i){
        sum += peek(time_stack, i);
    }
    printf("%d kernel calls took a sum total of %f ms\n\n", time_stack->size, sum);
    destroy(time_stack);

    //Copy region from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(region, cudaRegion, dataSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("\nCopying region from device took %f ms\n\n", getCudaEventTime(start, end));

    gEC(cudaFree(cudaData));
    gEC(cudaFree(cudaRegion));
    gEC(cudaFree(gpu_changed));

    return region;
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    int tid = getGlobalIdx_3D_3D();
    int y = getBlockId_3D() - (IMAGE_DIM/2);
    int x = getBlockThreadId_3D() - (IMAGE_DIM/2);
    float3 z_axis = {.x=0, .y=0, .z = 1};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 camera = {.x=1000, .y=1000, .z=1000};
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    float fov = 3.14/4;
    up = normalize(up);
    float step_size = 0.5;
    right = normalize(right);
    forward = normalize(forward);
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

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    cudaEvent_t start, end;
    dim3 **sizes = getGridsBlocksRaycasting(0);
    unsigned char *cudaImage, *cudaRegion, *cudaData;
    unsigned char *image = (unsigned char*) malloc(imageSize);

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
    printf("Copying data and region to device took %f ms\n\n",
        getCudaEventTime(start, end));

    createCudaEvent(&start);
    raycast_kernel<<<*sizes[0], *sizes[1]>>>(cudaData, cudaImage, cudaRegion);
    createCudaEvent(&end);
    printf("Calling kernel took %f ms\n", getCudaEventTime(start, end));

    //Copy image back from device
    createCudaEvent(&start);
    gEC(cudaMemcpy(image, cudaImage, imageSize, cudaMemcpyDeviceToHost));
    createCudaEvent(&end);
    printf("Copying image from device took %f ms\n\n",
        getCudaEventTime(start, end));

    gEC(cudaFree(cudaData));
    gEC(cudaFree(cudaImage));
    gEC(cudaFree(cudaRegion));
    return image;
}

__global__ void raycast_kernel_texture(unsigned char* image){

}

unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region){
    return NULL;
}

int main(int argc, char** argv){
    //print_properties();

    unsigned char* data = create_data();
    printf("Done creating data\n");

    //Serial version
    //unsigned char* region = grow_region_serial(data);
    unsigned char* region = grow_region_gpu(data);
    printf("Done creating region\n");

    //Serial version
    //unsigned char* image = raycast_serial(data, region);
    unsigned char* image = raycast_gpu_texture(data, region);
    //unsigned char* image = raycast_gpu(data, region);
    printf("Done creating image\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM, "raycast_gpu_texture_out.bmp");
    printf("Done with program\n");

    return 0;
}
