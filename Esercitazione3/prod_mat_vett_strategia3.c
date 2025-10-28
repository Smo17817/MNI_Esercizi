#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    int nproc; // Numero di processi totale
    int me; // Il mio id
    int m,n; // Dimensione della matrice
    int local_m; // Dimensione dei dati locali
    int i,j; // Iteratori vari 
    double T_inizio,T_fine;

    /*
    A: matrice m x n
    v: vettore di dimensione n
    localA: parte di matrice locale di dimensione m x local_n
    local_w: vettore locale di dimensione m
    w: vettore risultato di dimensione m
    */
    double *A, *v, *localA,*local_w, *w;
    
    // definizione della griglia
    MPI_Comm comm_grid;
    int dim = 2, *ndim, reorder, *period, rows, cols; 
    int coordinate[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if(me == 0) {
        printf("inserire numero di righe e colonne (separate da spazio): ");
        fflush(stdout);
        scanf("%d",&m);
        fflush(stdout);
        scanf("%d",&n);

        // Alloco spazio di memoria
        A = malloc(m * n * sizeof(double));
        v = malloc(sizeof(double)*n);
        w =  malloc(sizeof(double)*m); 

        // Definisce la griglia
        rows = 0; // TODO stabilire i valori che devono assumere 
        cols = 0;
        ndim = (int*)calloc(dim, sizeof(int));
        ndim[0] = 3; // numero di righe
        ndim[1] = 3; // numero di colonne
        
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

        // Stampa matrice e vettore se piccoli
        if (n < 11 && m < 11){  
            printf("Vettore v: ");   
                for (j=0;j<n;j++)
                    printf("%.2f ", v[j]);
                printf("\n");
        
            printf("Matrice A\n"); 
            for (i=0;i<m;i++) {
                for(j=0;j<n;j++)
                        printf("%.2f ", A[i*n+j] );
                printf("\n");
            }

            printf("\n");
        } 
    }

    fflush(stdout);

    

    MPI_Finalize();
    return 0;
}