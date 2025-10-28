/*
Creare una griglia cartesiana 3 x 3, non periodica. Ogni processo Pij della griglia assegna alla variabile intera a i seguenti valori:
a = i*i se i=j
a = i+2*j se i!=j
Ogni processo stampa il proprio identificativo, le proprie coordinate e la variabile a.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Scopo: definizione di una topologia a griglia bidimensionale nproc=row*col */
int main(int argc, char **argv){

    int menum, nproc, menum_grid;
    int dim, *ndim, reorder, *period; 
    int result = 0;
    int coordinate[2];
    // definizione del tipo di contesto di comunicazione
    MPI_Comm comm_grid;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&menum);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    // vettore contenente le lunghezze di ciascuna dimensione
    dim = 2; // Numero di dimensioni della griglia (rows, cols)
    ndim = (int*)calloc(dim, sizeof(int));
    ndim[0] = 3; // numero di righe
    ndim[1] = 3; // numero di colonne

    // vettore contenente la periodicità delle dimensioni. In questo caso non periodica
    period = (int*)calloc(dim,sizeof(int));
    period[0] = period[1] = 0;
    reorder = 0;

    // Definizione della griglia bidimensionale di dimensione 3*3
    MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder, &comm_grid);
    MPI_Comm_rank(comm_grid, &menum_grid); // id del processore nella griglia
    MPI_Cart_coords(comm_grid, menum_grid, dim, coordinate); // coordinate cartesiane di ciascun processo nella griglia

    if(coordinate[0] == coordinate[1]){
        result = coordinate[0] * coordinate[0];
    } else {
        result = coordinate[0] + 2 * coordinate[1];
    }

    printf("Processore %d (%d,%d) = %d \n", menum, coordinate[0], coordinate[1], result);
    
    MPI_Finalize();
    return 0; 
}