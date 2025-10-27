#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    int me, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Codice principale del programma
    // ...

    MPI_Finalize();
    return 0;
}