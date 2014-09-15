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
int similar(unsigned char* im, pixel_t p, pixel_t q, int row_stride){
    int a = im[p.x +  (p.y * row_stride)];
    int b = im[q.x +  (q.y * row_stride)];
    int diff = abs(a-b);
    return diff < 2;
}

// Create and commit MPI datatypes
void create_types(MPI_Datatype* newtype, int block_count, int block_length, int stride){
    //For coloumns that neighbour other processes
    MPI_Type_vector(block_count, block_length, stride, MPI_UNSIGNED_CHAR, newtype);
    //Commit the above
    MPI_Type_commit(newtype);
}

// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(unsigned char* src, unsigned char* dest, int* image_size,
    int icol, int irow, int orow, int* displs, int rank, int size, MPI_Comm comm, MPI_Status status){
    if (0 == rank){ //Send to all other processes but self
        int globRowSz = image_size[1];
        for (int i = 1; i < size; ++i){ //For each process except root (0)
            for (int j = 0; j < icol; ++j){ //For each contiguous row in image
                MPI_Send(&(src[displs[i] + (globRowSz*j)]), irow,
                            MPI_UNSIGNED_CHAR, i, 51, comm);
            }
        }

        //Unpack and send to local_image on rank 0 without MPI
        for (int i = 0; i < icol; ++i){
            memcpy(&(dest[(orow*(i+1))+1]), &(src[globRowSz*i]), irow);
        }
    } else{
        for (int i = 0; i < icol; ++i){
            MPI_Recv(&(dest[(orow*(i+1))+1]), irow, MPI_UNSIGNED_CHAR, 0, 51, comm, &status);
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

void popPixel(stack_t* stack, pixel_t p, int rank){
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
    if (2 == rank){
        printf("Rank %d popped pixel x: %d y: %d\n", rank, p.x, p.y);
    }
}

// Check if pixel is inside local image
int inside(pixel_t p, int max_x, int max_y){
    return (p.x >= 0 && p.x < max_x &&
            p.y >= 0 && p.y < max_y);
}

// Exchange the borders for local_image so that the halo works
void distribute_halo(unsigned char* img, int irow, int orow, int bsize, int north,
    int south, int east, int west, MPI_Datatype datatype, MPI_Comm comm, MPI_Status status){
    //Send north and receive from south
    MPI_Send(&(img[orow+1]), irow, MPI_UNSIGNED_CHAR, north, 0, comm);
    MPI_Recv(&(img[bsize-orow+1]), irow, MPI_UNSIGNED_CHAR, south, 0, comm, &status);

    //Send east and receive from west
    MPI_Send(&(img[(2*orow)-2]), 1, datatype, east, 1, comm);
    MPI_Recv(&(img[orow]), 1, datatype, west, 1, comm, &status);

    //Send south and receive from north
    MPI_Send(&(img[bsize-(2*orow)+1]), irow, MPI_UNSIGNED_CHAR, south, 2, comm);
    MPI_Recv(&(img[1]), irow, MPI_UNSIGNED_CHAR, north, 2, comm, &status);

    //Send west and receive from east
    MPI_Send(&(img[orow+1]), 1, datatype, west, 3, comm);
    MPI_Recv(&(img[(orow*2)-1]), 1, datatype, east, 3, comm, &status);
}

// Exchange borders with neighbour ranks
void exchange(stack_t* stck, stack_t* h_stck, unsigned char* img, int irow, int orow,
    int bsize, int north, int south, int east, int west, int rank, MPI_Datatype datatype,
    MPI_Comm comm, MPI_Status status){
    //Send north and receive from south
    MPI_Send(&(img[orow+1]), irow, MPI_UNSIGNED_CHAR, north, 0, comm);
    MPI_Recv(&(img[bsize-orow+1]), irow, MPI_UNSIGNED_CHAR, south, 0, comm, &status);

    //Send east and receive from west
    MPI_Send(&(img[(2*orow)-2]), 1, datatype, east, 1, comm);
    MPI_Recv(&(img[orow]), 1, datatype, west, 1, comm, &status);

    //Send south and receive from north
    MPI_Send(&(img[bsize-(2*orow)+1]), irow, MPI_UNSIGNED_CHAR, south, 2, comm);
    MPI_Recv(&(img[1]), irow, MPI_UNSIGNED_CHAR, north, 2, comm, &status);

    //Send west and receive from east
    MPI_Send(&(img[orow+1]), 1, datatype, west, 3, comm);
    MPI_Recv(&(img[(orow*2)-1]), 1, datatype, east, 3, comm, &status);

    //Then check if the halo pixels have been added to the stack already
    int cntr = 0, test = 0;
    while(cntr < h_stck->size){
        pixel_t p = h_stck->pixels[cntr];
        if(img[p.x + (p.y*orow)]){
            push(stck, p);
            popPixel(h_stck, p, rank);
            test += 1;
        } else{
            ++cntr;
        }
    }
    printf("Rank %d added %d pixels to stack after exchange()!\n", rank, test);
}

// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(unsigned char* src, unsigned char* dest, int* image_size,
    int icol, int irow, int orow, int* displs, int rank, int size, MPI_Comm comm, MPI_Status status){
    if (0 == rank){
        int globRowSz = image_size[1];
        for (int i = 1; i < size; ++i){ //For each process except root (0)
            for (int j = 0; j < icol; ++j){ //For each contigous row in image
                MPI_Recv(&(dest[displs[i] + (globRowSz*j)]), irow,
                         MPI_UNSIGNED_CHAR, i, 37, comm, &status);
            }
        }

        //Then transfer memory locally without MPI to region
        for (int i = 0; i < icol; ++i){
            memcpy(&(dest[globRowSz*i]), &(src[(orow*(i+1))+1]), irow);
        }
    } else{
        for (int i = 0; i < icol; ++i){
            MPI_Send(&(src[(orow*(i+1))+1]), irow, MPI_UNSIGNED_CHAR, 0, 37, comm);
        }
    }
}

// Adding seeds in corners.
void add_seeds(stack_t* stack, int* coords, int* dims, int orow, int ocol){
    pixel_t seed;
    int x = coords[1], y = coords[0],
    max_x = dims[1]-1, max_y = dims[0]-1;
    if (0 == x && 0 == y){
        seed.x = 5; seed.y = 5;
        push(stack, seed);
    }
    if(0 == x && max_y == y){
        seed.x = 5; seed.y = ocol - 5;
        push(stack, seed);
    }
    if(max_x == x && 0 == y){
        seed.x = orow - 5; seed.y = 5;
        push(stack, seed);
    }
    if(max_x == x && max_y == y){
        seed.x = orow - 5; seed.y = ocol - 5;
        push(stack, seed);
    }
    /*seed.x = 256; seed.y = 170;
    push(stack, seed);
    printf("Inside call for seed: %d\n", inside(seed, orow, ocol));*/
}

// Region growing, parallell implementation
int grow_region(stack_t* stack, unsigned char* img, unsigned char* region, int orow, int ocol){
    int stackNotEmpty = 0;

    while(stack->size > 0){
        stackNotEmpty = 1;
        pixel_t pixel = pop(stack);
        region[(pixel.y*orow) + pixel.x] = 1;

        int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
        for(int c = 0; c < 4; c++){
            pixel_t candidate;
            candidate.x = pixel.x + dx[c];
            candidate.y = pixel.y + dy[c];

            if(!inside(candidate, orow, ocol)){
                continue;
            }

            // If pixel in region already has value set to 1 like on line 202
            if(region[(candidate.y*orow) + candidate.x]){
                continue;
            }

            if(similar(img, pixel, candidate, orow)){
                region[candidate.x + (candidate.y*orow)] = 1;
                push(stack,candidate);
            }
        }
    }
    return stackNotEmpty;
}

// MPI initialization, setting up cartesian communicator
void init_mpi(int *argc, char*** argv, int* size, int* rank, int** dims,
    int** coords, int** periods, int* north, int* south, int* east, int* west,
    MPI_Comm* cart_comm){
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    if ((*size & (*size-1)) != 0){
        printf("Need number of processes to be a power of 2!\nExiting program.\n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Dims_create(*size, 2, *dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, *dims, *periods, 0, cart_comm);
    MPI_Cart_coords(*cart_comm, *rank, 2, *coords);

    MPI_Cart_shift(*cart_comm, 0, 1, north, south);
    MPI_Cart_shift(*cart_comm, 1, 1, west, east);
}

void load_and_allocate_images(int argc, char** argv, unsigned char** global_image,
    unsigned char** global_region, unsigned char** local_image, unsigned char** local_region,
    int* global_image_size, int** local_image_size, int* dims, int* icol, int* irow, int* lsize,
    int* orow, int* ocol, int* bsize, int rank){
    if(argc != 2){
        printf("Usage: region file\n");
        exit(-1);
    }

    if(rank == 0){
        *global_image = read_bmp(argv[1]);
        *global_region = (unsigned char*)calloc(sizeof(unsigned char),
            global_image_size[0]*global_image_size[1]);
    }

    (*local_image_size)[0] = global_image_size[0]/dims[0];
    (*local_image_size)[1] = global_image_size[1]/dims[1];

    *icol = (*local_image_size)[0];
    *irow = (*local_image_size)[1];
    *lsize = (*icol)*(*irow);

    *ocol = *icol + 2;
    *orow = *irow + 2;
    *bsize = (*ocol)*(*orow);

    *local_image = (unsigned char*)calloc(sizeof(unsigned char),*bsize);
    *local_region = (unsigned char*)calloc(sizeof(unsigned char),*bsize);
}

void write_image(int rank, unsigned char* image, int* image_size, unsigned char* region){
    if(rank==0){
        int length = image_size[0]*image_size[1];
        for(int i = 0; i < length; i++){
            image[i] *= (region[i] == 0);
        }
        write_bmp(image, image_size[0], image_size[1]);
    }
}

int main(int argc, char** argv){
    int rank, size,                                 // Number of MPI processes
        irow, icol,                                 // Inner row/col lengths/sizes in local char array
        orow, ocol,                                 // Outer row/col lengths/sizes in local char array
        lsize,                                      // Inner size of local image char array
        bsize,                                      // Outer size of local image char array
        *dims = calloc(sizeof(int),2),              // Dimensions of MPI grid
        *coords = malloc(sizeof(int)*2),            // Coordinate of this rank in MPI grid
        *periods = calloc(sizeof(int),2),           // Periodicity of grid
        north,south,east,west,                      // Four neighbouring MPI ranks
        image_size[2] = {512,512},                  // Hard coded image size
        *local_image_size = malloc(sizeof(int)*2),  // Size of local part of image (not including border)
        *displs;                                    // List with displacements that go along with sendcounts

    // For MPI_Recv and MPI_Wait
    MPI_Status status;
    // MPI Cartesian communicator
    MPI_Comm cart_comm;
    // MPI datatype for non-contiguous coloumns in memory
    MPI_Datatype border_col_t;

    unsigned char *image,           // Entire image, only on rank 0
                  *region,          // Region bitmap. 1 if in region, 0 elsewise
                  *local_image,     // Local part of image
                  *local_region;    // Local part of region bitmap

    init_mpi(&argc, &argv, &size, &rank, &dims, &coords, &periods, &north, &south,
        &east, &west, &cart_comm);
        //printf("Done with init_mpi()\n");

    load_and_allocate_images(argc, argv, &image, &region, &local_image, &local_region,
        image_size, &local_image_size, dims, &icol, &irow, &lsize, &orow, &ocol, &bsize, rank);
        //printf("Done with load_and_allocate_images()\n");

    create_types(&border_col_t, irow, 1, orow);
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

    distribute_image(image, local_image, image_size, icol, irow, orow, displs, rank,
        size, cart_comm, status);
        //printf("After distribute_image() in main()\n");

    distribute_halo(local_image, irow, orow, bsize, north, south, east, west, border_col_t,
        cart_comm, status);
        //printf("After distribute_halo() in main()\n");

    pixel_t p1, p2;
    stack_t* stack = new_stack();
    stack_t* halo_stack = new_stack();
    add_seeds(stack, coords, dims, orow, ocol);
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

    //Check if all halo coordinates are inside or not. (At this point in development they are)
    if (1 == rank){
        int cntr = 0; pixel_t p;
        for (int i = 0; i < halo_stack->size; ++i){
            p = halo_stack->pixels[i];
            if(!inside(p, orow, ocol)){
                cntr += 1;
            }
        }
        printf("This many pixels were out of bounds on rank %d: %d\n", rank, cntr);
    }

    // Run while-loop to empty stack
    int filledStack = grow_region(stack, local_image, local_region, orow, ocol), recvbuf = 1;
    while(!MPI_Allreduce(&filledStack, &recvbuf, 1, MPI_INT, MPI_SUM, cart_comm) && (recvbuf != 0)){
        exchange(stack, halo_stack, local_region, irow, orow, bsize, north, south,
            east, west, rank, border_col_t, cart_comm, status);
        filledStack = grow_region(stack, local_image, local_region, orow, ocol);
    }
    printf("After grow_region() in main()\n");

    gather_region(local_region, region, image_size, icol, irow, orow, displs, rank,
        size, cart_comm, status);
        //printf("After gather_region() in main()\n");

    MPI_Finalize();
    //printf("After MPI_Finalize() in main()\n");

    write_image(rank, image, image_size, region);

    printf("Rank %d: Program successfully completed!\n", rank);
    exit(0);
}
