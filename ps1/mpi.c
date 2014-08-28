#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG 1047

void send(int rank, int *msg, int direction){
    *msg += 1;
    MPI_Send(msg, 1, MPI_INT, rank+direction, TAG, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n", rank, *msg);
}

void recv(int rank, int *msg, int direction, MPI_Status status){
    MPI_Recv(msg, 1, MPI_INT, rank+direction, TAG, MPI_COMM_WORLD, &status);
    printf("Rank %d received %d\n", rank, *msg);
}

int main(int argc, char** argv) {
    int size, rank, msg;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank > 0 && rank < (size-1)){ //If rank is not zero or size-1 (AKA first or last)
        //Send the integer "Forwards"
        recv(rank, &msg, -1, status);
        send(rank, &msg, 1);

        //Send the integer "Backwards"
        recv(rank, &msg, 1, status);
        send(rank, &msg, -1);
    } else if(rank == 0){
        //Start the transmission of the integer through the ranks/nodes
        msg = -1;
        send(rank, &msg, 1);

        //Receive the integer after having been through all the other nodes but one twice.
        recv(rank, &msg, 1, status);
    } else{ //If rank == size -1, aka final/highest rank/node.
        //Receive the integer "msg"
        recv(rank, &msg, -1, status);

        //Increment and send integer back.
        send(rank, &msg, -1);
    }

    MPI_Finalize();
}
