#include "bmp.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct{
    int x;
    int y;
} pixel_t;

typedef struct{
    int size;
    int buffer_size;
    pixel_t* pixels;
} stack_t;

// Global variables
int rank,                       // MPI rank
    size,                       // Number of MPI processes
    irow,                       // Inner row length in local char array
    icol,                       // Inner col length in local char array
    orow,                       // Outer row length in local char array
    ocol,                       // Outer col length in local char array
    lsize,                      // Inner size of local image char array
    bsize,                      // Outer size of local image char array
    totSize,                    // Total size of global image char array
    dims[2],                    // Dimensions of MPI grid
    coords[2],                  // Coordinate of this rank in MPI grid
    periods[2] = {0,0},         // Periodicity of grid
    north,south,east,west,      // Four neighbouring MPI ranks
    image_size[2] = {512,512},  // Hard coded image size
    local_image_size[2],        // Size of local part of image (not including border)
    *displs;                    // List with displacements that go along with sendcounts

MPI_Comm cart_comm;             // Cartesian communicator

// MPI datatypes, you may have to add more.
MPI_Datatype border_col_t;

// For MPI_Recv() and MPI_Wait()
MPI_Status status;

unsigned char *ptr,             // Shorthand for the below ptrs
              *ptr2,            // Same as above
              *image,           // Entire image, only on rank 0
              *region,          // Region bitmap. 1 if in region, 0 elsewise
              *local_image,     // Local part of image
              *local_region;    // Local part of region bitmap

// Create new pixel stack
stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (pixel_t*)malloc(sizeof(pixel_t)*1024);
    return stack;
}

// Push on pixel stack
void push(stack_t* stack, pixel_t p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        stack->pixels = realloc(stack->pixels, sizeof(pixel_t)*stack->buffer_size);
    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}

// Pop from pixel stack
pixel_t pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}

// Check if two pixels are similar. The hardcoded threshold can be changed.
// More advanced similarity checks could have been used.
int similar(unsigned char* im, pixel_t p, pixel_t q){
    int a = im[p.x +  (p.y * orow) + 1];
    int b = im[q.x +  (q.y * orow) + 1];
    int diff = abs(a-b);
    return diff < 2;
}

// Create and commit MPI datatypes
void create_types(){
    //For coloumns that neighbour other processes
    MPI_Type_vector(icol, 1, orow, MPI_UNSIGNED_CHAR, &border_col_t);
    //Commit the above
    MPI_Type_commit(&border_col_t);
}

// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
    ptr = image;
    unsigned char *ptr2 = local_image;
    if (0 == rank){ //Send to all other processes but self
        int globRowSz = image_size[1];

        for (int i = 1; i < size; ++i){ //For each process except root (0)
            for (int j = 0; j < icol; ++j){ //For each contigous row in image
                MPI_Send(&(ptr[displs[i] + (globRowSz*j)]), irow,
                            MPI_UNSIGNED_CHAR, i, 51, cart_comm);
            }
        }

        //Unpack and send to local_image on rank 0 without MPI
        for (int i = 0; i < icol; ++i){
            memcpy(&(ptr2[(orow*(i+1))+1]), &(ptr[globRowSz*i]), irow);
        }
    } else{
        for (int i = 0; i < icol; ++i){
            MPI_Recv(&(ptr2[(orow*(i+1))+1]), irow, MPI_UNSIGNED_CHAR, 0, 51, cart_comm, &status);
        }
    }
}

int pixelInStack(stack_t* stack, pixel_t p){
    int inStack = 0;
    for (int i = 0; i < stack->size; ++i){
        if (p.x == stack->pixels[i].x &&
            p.y == stack->pixels[i].y){
            inStack = 1;
            break;
        }
    }
    return inStack;
}

void popPixel(stack_t* stack, pixel_t p){
    int pos;
    if (pixelInStack(stack, p)){
        for (int i = 0; i < stack->size; ++i){
            if (p.x == stack->pixels[i].x &&
                p.y == stack->pixels[i].y){
                pos = i;
                break;
            }
        }

        for (int i = pos; i < stack->size-1; ++i){
            stack->pixels[i] = stack->pixels[i+1];
        }
        stack->size -= 1;
    }
}

// Check if pixel is inside local image
int inside(pixel_t p){
    return (p.x >= 0 && p.x < orow &&
            p.y >= 0 && p.y < ocol);
}

// Exchange borders with neighbour ranks
void exchange(stack_t* stck, stack_t* h_stck){
    ptr = local_region;
    //Send north and receive from south
    MPI_Send(&(ptr[orow+1]), irow, MPI_UNSIGNED_CHAR, north, 0, cart_comm);
    MPI_Recv(&(ptr[bsize-orow+1]), irow, MPI_UNSIGNED_CHAR, south, 0, cart_comm, &status);

    //Send east and receive from west
    MPI_Send(&(ptr[(2*orow)-2]), 1, border_col_t, east, 1, cart_comm);
    MPI_Recv(&(ptr[orow]), 1, border_col_t, west, 1, cart_comm, &status);

    //Send south and receive from north
    MPI_Send(&(ptr[bsize-(2*orow)+1]), irow, MPI_UNSIGNED_CHAR, south, 2, cart_comm);
    MPI_Recv(&(ptr[1]), irow, MPI_UNSIGNED_CHAR, north, 2, cart_comm, &status);

    //Send west and receive from east
    MPI_Send(&(ptr[orow]), 1, border_col_t, west, 3, cart_comm);
    MPI_Recv(&(ptr[(orow*2)-1]), 1, border_col_t, east, 3, cart_comm, &status);

    //Then check if the halo pixels have been added to the stack already
    int cntr = 0, test = 0;
    while(cntr < h_stck->size){
        pixel_t p = h_stck->pixels[cntr];
        if(ptr[p.x + (p.y*orow)]){
            push(stck, p);
            popPixel(h_stck, p);
            test += 1;
        }
        ++cntr;
    }
    printf("Rank %d added %d pixels to stack after exchange()!\n", rank, test);
}

// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(){
    ptr = region;
    unsigned char *ptr2 = local_region;
    if (0 == rank){
        int globRowSz = image_size[1];
        for (int i = 1; i < size; ++i){ //For each process except root (0)
            for (int j = 0; j < icol; ++j){ //For each contigous row in image
                MPI_Recv(&(ptr[displs[i] + (globRowSz*j)]), irow,
                         MPI_UNSIGNED_CHAR, i, 37, cart_comm, &status);
            }
        }

        //Then transfer memory locally without MPI to region
        for (int i = 0; i < icol; ++i){ //Transfer locally first
            memcpy(&(ptr[globRowSz*i]), &(ptr2[(orow*(i+1))+1]), irow);
        }
    } else{
        for (int i = 0; i < icol; ++i){
            MPI_Send(&(ptr2[(orow*(i+1))+1]), irow, MPI_UNSIGNED_CHAR, 0, 37,
                     cart_comm);
        }
    }
}

// Adding seeds in corners.
void add_seeds(stack_t* stack){
    int seeds [8];
    seeds[0] = 5;
    seeds[1] = 5;
    seeds[2] = orow-5;
    seeds[3] = 5;
    seeds[4] = orow-5;
    seeds[5] = ocol-5;
    seeds[6] = 5;
    seeds[7] = ocol-5;

    for(int i = 0; i < 4; i++){
        pixel_t seed;
        seed.x = abs(seeds[i*2] - orow);
        seed.y = abs(seeds[i*2+1] - ocol);

        // Below if-check unnecessary?
        if(inside(seed)){
            push(stack, seed);
        }
    }
}

// Region growing, parallell implementation
int grow_region(stack_t* stack){
    ptr = local_region;
    int stackNotEmpty = 0;

    /*for (int i = 0; i < bsize; ++i){
        if (150 > local_image[i]){
            ptr[i] = 1;
        }
    }

    return 0;*/

    while(stack->size > 0){
        stackNotEmpty = 1;
        pixel_t pixel = pop(stack);
        ptr[(pixel.y*orow) + pixel.x] = 1;

        int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
        for(int c = 0; c < 4; c++){
            pixel_t candidate;
            candidate.x = pixel.x + dx[c];
            candidate.y = pixel.y + dy[c];

            if(!inside(candidate)){
                continue;
            }

            // If pixel in region already has value set to 1 like on line 202
            if(ptr[(candidate.y*orow) + candidate.x]){
                continue;
            }

            if(similar(local_image, pixel, candidate)){
                ptr[candidate.x + (candidate.y*orow)] = 1;
                push(stack,candidate);
            }
        }
    }
    return stackNotEmpty;
}

// MPI initialization, setting up cartesian communicator
void init_mpi(int *argc, char*** argv){
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if ((size & (size-1)) != 0){
        printf("Need number of processes to be a power of 2!\nExiting program.\n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
}

void load_and_allocate_images(int argc, char** argv){
    if(argc != 2){
        printf("Usage: region file\n");
        exit(-1);
    }

    if(rank == 0){
        image = read_bmp(argv[1]);
        region = (unsigned char*)calloc(sizeof(unsigned char),totSize);
    }

    local_image_size[0] = image_size[0]/dims[0];
    local_image_size[1] = image_size[1]/dims[1];
    icol = local_image_size[0];
    irow = local_image_size[1];

    lsize = icol*irow;
    ocol = icol + 2;
    orow = irow + 2;
    bsize = ocol*orow;
    local_image = (unsigned char*)calloc(sizeof(unsigned char),bsize);
    local_region = (unsigned char*)calloc(sizeof(unsigned char),bsize);
}

void write_image(){
    if(rank==0){
        for(int i = 0; i < totSize; i++){
            image[i] *= (region[i] == 0);
        }
        write_bmp(image, image_size[0], image_size[1]);
    }
}

int main(int argc, char** argv){
    totSize = image_size[0]*image_size[1];
    init_mpi( &argc, &argv);
    //printf("Done with init_mpi()\n");

    load_and_allocate_images(argc, argv);
    //printf("Done with load_and_allocate_images()\n");

    create_types();
    //printf("Done with create_types()\n");

    displs = (int*) malloc(sizeof(int)*size);
    int image_tot_row_length = image_size[1];
    //Set displs for where to start sending data to each rank from,
    // in distribute_image() and gather_region()
    for (int i = 0; i < dims[0]; ++i){
        for (int j = 0; j < dims[1]; ++j){
            displs[(i*dims[1]) + j] = (i*icol*image_tot_row_length) + (j*irow);
        }
    }

    distribute_image();
    //printf("After distribute_image() in main()\n");

    pixel_t p1, p2;
    stack_t* stack = new_stack();
    stack_t* halo_stack = new_stack();
    add_seeds(stack);
    // Fill halo_stack with the pixel coordinates making up the halo
    for (int i = 0; i < orow; ++i){ //First rows
        p1.x = i; p2.x = i;
        p1.y = 0; p2.y = ocol-1;
        push(halo_stack, p1); push(halo_stack, p2);
    }
    for (int i = 0; i < ocol; ++i){ //Then cols
        p1.y = i; p2.y = i;
        p1.x = 0; p2.x = orow-1;
        push(halo_stack, p1); push(halo_stack, p2);
    }

    if (1 == rank){
        int cntr = 0; pixel_t p;
        for (int i = 0; i < halo_stack->size; ++i){
            p = halo_stack->pixels[i];
            if(!inside(p)){
                cntr += 1;
            }
        }
        printf("This many pixels were out of bounds on rank %d: %d\n", rank, cntr);
    }

    // Run while-loop to empty stack
    int filledStack = grow_region(stack), recvbuf = 1;
    while(!MPI_Allreduce(&filledStack, &recvbuf, 1, MPI_INT, MPI_SUM, cart_comm) && (recvbuf != 0)){
        exchange(stack, halo_stack);
        filledStack = grow_region(stack);
    }
    printf("After grow_region() in main()\n");

    gather_region();
    //printf("After gather_region() in main()\n");

    MPI_Finalize();
    //printf("After MPI_Finalize() in main()\n");

    write_image();

    printf("Rank %d: Program successfully completed!\n", rank);
    exit(0);
}
