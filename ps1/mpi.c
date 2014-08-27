#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int size, rank, msg;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank > 0 && rank < (size-1)){
        //Send the integer "Forwards"
        MPI_Recv(&msg, 1, MPI_INT, rank-1, msg, MPI_COMM_WORLD, &status);
        printf("Rank %d received %d\n", rank, msg);
        msg += 1;
        MPI_Send(&msg, 1, MPI_INT, rank+1, msg, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, msg);

        //Send the integer "Backwards"
        MPI_Recv(&msg, 1, MPI_INT, rank+1, msg, MPI_COMM_WORLD, &status);
        printf("Rank %d received %d\n", rank, msg);
        msg += 1;
        MPI_Send(&msg, 1, MPI_INT, rank-1, msg, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, msg);
    } else if(rank == 0){
        //Start the transmission of the integer through the ranks/nodes
        msg = 0;
        MPI_Send(&msg, 1, MPI_INT, rank+1, msg, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, msg);

        //Receive the integer after having been through all the other nodes but one twice.
        MPI_Recv(&msg, 1, MPI_INT, rank+1, msg, MPI_COMM_WORLD, &status);
        printf("Rank %d received %d\n", rank, msg);
    } else{ //If rank == size -1, aka final/highest rank/node.
        //Receive the integer "msg"
        MPI_Recv(&msg, 1, MPI_INT, rank-1, msg, MPI_COMM_WORLD, &status);
        printf("Rank %d received %d\n", rank, msg);

        //Increment and send integer back.
        msg += 1;
        MPI_Send(&msg, 1, MPI_INT, rank-1, msg, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, msg);
    }

    MPI_Finalize();
}
