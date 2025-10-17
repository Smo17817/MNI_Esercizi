/*
Implementazione del prodotto matrice vettore utilizzando la strategia 2: 
ogni processo riceve un blocco di colonne della matrice e una parte del vettore della stessa dimensione del numero di colonne e ne calcola il prodotto.
Il processo root raccoglie i risultati parziali dai processi figli e li somma. Per la somma verrà urilizzata la strategia 2 dell'Esercitazione 1.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

int main(int argc, char **argv) {
    int nproc; // Numero di processi totale
    int me; // Il mio id
    int m,n; // Dimensione della matrice
    int local_n; // Dimensione dei dati locali
    int i,j; // Iteratori vari
    double time_start, time_end; // Variabili per il calcolo del tempo

    /*
    A: matrice m x n
    AT: trasposta di A di dimensione n x m
    v: vettore di dimensione n
    localA: parte di matrice locale di dimensione m x local_n
    local_w: vettore locale di dimensione m
    w: vettore risultato di dimensione m
    */
    double *A, *AT, *v, *localA,*local_w, *w;

    /*Attiva MPI*/
    MPI_Init(&argc, &argv);
    /*Trova il numero totale dei processi*/
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    /*Da ad ogni processo il proprio numero identificativo*/
    MPI_Comm_rank (MPI_COMM_WORLD, &me);

    if(me == 0) {
        printf("inserire numero di righe m: ");
        fflush(stdout);
        scanf("%d",&m);

        printf("inserire numero di colonne n: ");
        fflush(stdout);
        scanf("%d",&n);

        // Numero di colonne da processare
        local_n = n/nproc;

        // Alloco spazio di memoria
        A = malloc(m * n * sizeof(double));
        v = malloc(sizeof(double)*n);
        w =  malloc(sizeof(double)*m); 
        AT = malloc(n * m * sizeof(double));
        
        // Inizializza vettore con valori 0,1,2,...
        for (j=0;j<n;j++)
                v[j]=j; 

        // Inizializza matrice A
        for (i=0;i<m;i++) {
            for(j=0;j<n;j++) {
                if (j==0)
                    A[i*n+j]= 1.0/(i+1)-1;
                else
                    A[i*n+j]= 1.0/(i+1)-pow(1.0/2.0,j); 
            }
        }

        // Calcola la trasposta di A
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                AT[j * m + i] = A[i * n + j];
            }
        }

        // Stampa matrice e vettore se piccoli
        if (n < 11 && m < 11){  
            printf("Vettore v: ");   
                for (j=0;j<n;j++)
                    printf("%f ", v[j]);
                printf("\n");
        
            printf("Matrice A\n"); 
            for (i=0;i<m;i++) {
                for(j=0;j<n;j++)
                        printf("%f ", A[i*n+j] );
                printf("\n");
            }
        } 
    }

    fflush(stdout);

    // Spedisco m, n, local_n
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Alloca memoria
    localA = malloc(local_n * m * sizeof(double)); // blocco di righe di AT
    local_w = malloc(m * sizeof(double));
    double *local_v = malloc(local_n * sizeof(double));
    w = malloc(m * sizeof(double));
    v = malloc(n * sizeof(double));

    // Invia la parte di vettore v
    MPI_Scatter(v, local_n, MPI_DOUBLE,
                local_v, local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Invia le righe (blocchi di colonne originali)
    MPI_Scatter(AT, local_n * m, MPI_DOUBLE,
                localA, local_n * m, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}