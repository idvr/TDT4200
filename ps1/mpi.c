#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG 1047

void send1(int rank, int *msg, int direction){
    *msg += 1;
    MPI_Send(msg, 1, MPI_INT, rank+direction, TAG, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n", rank, *msg);
}

void recv1(int rank, int *msg, int direction, MPI_Status status){
    MPI_Recv(msg, 1, MPI_INT, rank+direction, TAG, MPI_COMM_WORLD, &status);
    printf("Rank %d received %d\n", rank, *msg);
}

int main(int argc, char** argv) {
    MPI_Status status;
    int size, rank, msg;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0){
        //Start the transmission of the integer through the ranks/nodes
        msg = -1;
        send1(rank, &msg, 1);

        //Receive the integer after having been through all the other nodes but one twice.
        recv1(rank, &msg, 1, status);
    } else if (rank < (size-1)){ //If rank is not zero or size-1 (AKA first or last)
        //send1 the integer "Forwards"
        recv1(rank, &msg, -1, status);
        send1(rank, &msg, 1);

        //send1 the integer "Backwards"
        recv1(rank, &msg, 1, status);
        send1(rank, &msg, -1);
    } else{//If rank == size -1, aka final/highest rank/node.
        //Receive the integer "msg"
        recv1(rank, &msg, -1, status);
        //Increment and send1 integer back.
        send1(rank, &msg, -1);
    }

    MPI_Finalize();
}
