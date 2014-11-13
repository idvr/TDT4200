__kernel
void region(__global unsigned char* data, __global unsigned char* region, __global int* changed){
    int isEmpty = 0;
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int tid = idz*(512*512) + idy*512 + idx;
    const int dx[6] = {-1,1,0,0,0,0};
    const int dy[6] = {0,0,-1,1,0,0};
    const int dz[6] = {0,0,0,0,-1,1};

    /*
    unsigned int tid = getThreadId();
    int3 blockVox = getThreadPosInBlock();
    unsigned int globalIdx = getGlobalIdx();
    int3 globalVox = getGlobalPos(globalIdx);

    //Load into shared memory
    sdata[tid] = data[globalIdx];
    __syncthreads();

    //If already discovered or not yet (maybe never) reached; skip it
    if (!inside(globalVox) || NEW_VOX != region[globalIdx]){
        return;
    }
    region[globalIdx] = VISITED;

    for (int i = 0; i < 6; ++i){
        int3 curPos = blockVox;
        int3 globalPos = globalVox;
        curPos.x += dx[i];
        curPos.y += dy[i];
        curPos.z += dz[i];
        globalPos.x += dx[i];
        globalPos.y += dy[i];
        globalPos.z += dz[i];

        int curIndex = getThreadInBlockIndex(curPos);
        unsigned int globalIndex = index(globalPos);

        //If outside or region != 0; skip it
        if (!inside(globalPos) || region[globalIndex]){
            continue;
        }

        //if curPos is a voxel on cube outermost edge(s) and similar == 0
        if (isOnEdgeOfThreadBlock(curPos)){
            if(!similar(data, globalIdx, globalIndex)){
                continue;
            }
        } else if (!similar(sdata, tid, curIndex)){
            //If curPos not a voxel on cube outermost edge(s)
            continue;
        }

        region[globalIndex] = NEW_VOX;
        *changed = 1;
    }*/
    return;
}
