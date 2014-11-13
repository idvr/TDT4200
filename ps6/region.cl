// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
#define DATA_SIZE (DATA_DIM*DATA_DIM*DATA_DIM)

// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define IMAGE_SIZE (IMAGE_DIM*IMAGE_DIM)

typedef struct{
    int x;
    int y;
    int z;
} int3;

typedef barrier(CLK_LOCAL_MEM_FENCE) blockBarrier;

// Indexing function (note the argument order)
int index(int z, int y, int x){
    return x + y*DATA_DIM +
        z*DATA_DIM*DATA_DIM;
}

int index(int3 pos){
    return pos.x +
        pos.y*DATA_DIM +
        pos.z*DATA_DIM*DATA_DIM;
}

int inside_int(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);
    return x && y && z;
}

// Check if two values are similar, threshold can be changed.
int similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[a.z*DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z*DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];
    return (int) abs(va-vb) < 1;
}

__kernel
void region(const __global unsigned char* data, __global unsigned char* region, __global int* changed){
    __local int isEmpty = 1;
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int globalId = idz*(IMAGE_SIZE) + idy*DATA_DIM + idx;
    blockBarrier;

    //Check if region is empty for this block, if so, exit block
    if (region[globalId]){
        isEmpty = 0;
    }
    blockBarrier;
    if (isEmpty){
        return;
    }

    const int3 globalVox = {.x=idx, .y=idy, .z=idz};
    if (!inside(globalVox) || 2 != region[globalId]){
        return;
    }
    region[globalId] = 1;

    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};
    for (int i = 0; i < 6; ++i){
        int3 curPosGlob = globalVox;
        curPosGlob.x += dx[i];
        curPosGlob.y += dy[i];
        curPosGlob.z += dz[i];
        int curGlobIdx = index(curPosGlob);

        if (!inside(curPosGlob)) || region[curGlobIdx]){
            continue;
        }

        if (!similar(data, globalId, curPosGlob)){
            continue;
        }

        region[curGlobIdx] = 2;
        *changed = 1;
    }

    return;
}
