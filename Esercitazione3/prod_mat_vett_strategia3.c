#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

// Funzione per calcolare la griglia ottimale
void optimal_grid(int nproc, int rows, int cols, int dim, int *ndim, int verbose);

int main(int argc, char *argv[]) {
    int nproc; // Numero di processi totale
    int me; // Il mio id
    int m, n; // Dimensione della matrice
    int local_n, local_m; // Dimensione dei dati locali
    int i, j; // Iteratori vari 
    double T_inizio, T_fine;

    /*
    A: matrice m x n
    v: vettore di dimensione n
    localA: parte di matrice locale di dimensione local_m x n
    localAT: matrice locale trasposta di dimensione n x local_m
    local_v: vettore locale di dimensione local_n
    w: vettore risultato di dimensione m
    finalAT: parte di matrice trasposta locale di dimensione local_m x local_n
    */
    double *A, *v, *localA, *localAT, *local_v, *w, *finalAT;
    
    // definizione della griglia
    MPI_Comm comm_grid;
    int dim = 2, *ndim, reorder, *period;
    int coordinate[2];
    int menum_grid; // id del processo nella griglia

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

    // Spedisco m, n, local_n da P0 a tutti i processi
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*--- Creazione Contesto ---*/
    // vettore contenente le lunghezze di ciascuna dimensione
    ndim = (int*)calloc(dim, sizeof(int));

    // Calcolo la griglia ottimale
    optimal_grid(nproc, m, n, dim, ndim, 0);
    local_n = n/ndim[1]; // dimensione locale del vettore

    // Ogni processo alloca local_v
    local_v = malloc(local_n * sizeof(double));

    // vettore contenente la periodicità delle dimensioni. In questo caso non periodica
    period = (int*)calloc(dim,sizeof(int));
    period[0] = period[1] = 0;
    reorder = 0;

    // Definizione della griglia bidimensionale
    MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder, &comm_grid);

    // Ogni processore conosce la propria posizione nella griglia
    MPI_Comm_rank(comm_grid, &menum_grid); // id del processore nella griglia
    MPI_Cart_coords(comm_grid, menum_grid, dim, coordinate); // coordinate cartesiane di ciascun processo nella griglia    

    // Sottogriglie per comuinicazioni tra righe e tra colonne
    MPI_Comm row_comm, col_comm;
    int remain_dims_row[2] = {0,1}; // tieni solo la dimensione colonne
    int remain_dims_col[2] = {1,0}; // tieni solo la dimensione righe
    MPI_Cart_sub(comm_grid, remain_dims_row, &row_comm);
    MPI_Cart_sub(comm_grid, remain_dims_col, &col_comm);

    /*--- Invio del vettore v a tutti i processi ---*/
    // Fase 1: processi della prima riga
    if(coordinate[0] == 0) {
        MPI_Scatter(v, local_n, MPI_DOUBLE, 
            local_v, local_n, MPI_DOUBLE,
            0, row_comm);
    }
    
    // Fase 2: tutti i processi della prima riga inviano il vettore ai processi delle altre righe
    // In ogni riga il processo root ha rank 0
    int root_col = 0;
    MPI_Bcast(local_v, local_n, MPI_DOUBLE, root_col, col_comm);
    
    /*--- Invio della Matrice A ad ogni processo ---*/
    // Fase 1: P00 invia blocchi di rhighe ai processi della prima colonna
    local_m = m/ndim[0]; // numero di righe locale
    localA = malloc(local_m * n * sizeof(double)); // ogni processo alloca localA
    if(coordinate[1] == 0) {
        MPI_Scatter(A, local_m * n, MPI_DOUBLE,
            localA, local_m * n, MPI_DOUBLE,
            0, col_comm); 
    }

    // Fase 2: ogni processo della prima colonna invia il blocco di righe ai processi delle altre colonne
    // Inizialmente è necessario trasporre localA per inviare blocchi di colonne
    localAT = malloc(n * local_m * sizeof(double));
    for(int i=0; i<local_m; i++){
        for(int j=0; j<n; j++){
            localAT[j * local_m + i] = localA[i * n + j];
        }
    }

    // In ogni colonna il processo root ha rank 0
    int root_row = 0;
    finalAT = malloc(local_m * local_n * sizeof(double));
    MPI_Scatter(localAT, local_m * local_n, MPI_DOUBLE,
        finalAT, local_m * local_n, MPI_DOUBLE,
        root_row, row_comm);

    // Stampa di debug
    if(n < 11 && m < 11){
        printf("\nProcesso %d (%d,%d)\n", me, coordinate[0], coordinate[1]);
        
        printf("local_v: ");
        for(int i=0; i<local_n; i++)
            printf("%.2f ", local_v[i]);
        printf("\n");

        printf("finalAT\n"); 
        for(int i=0; i<local_m; i++){
            for(int j=0; j<local_n; j++){
                printf("%.2f ", finalAT[j * local_m + i]);
            }
            printf("\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}

void optimal_grid(int nproc, int rows, int cols, int dim, int *ndim, int verbose){
    int fattori[nproc];
    int nfattori = 0;
    double ratio = (double)rows / cols; // Rapporto desiderato tra righe e colonne
    double best_diff = rows; // Inizializza con un valore grande

    // Trova i fattori di nproc
    for(int i = 1; i <= (nproc / 2); i++){
        if(nproc % i == 0){
            fattori[nfattori] = i;
            nfattori++;
        }
    }
     // Aggiungi nproc stesso come fattore
    fattori[nfattori]=nproc;
    nfattori++;

    // Cerca la coppia di fattori che meglio approssima il rapporto desiderato
    for(int i=0; i < nfattori; i++){
        int f1 = fattori[i];
        int f2 = nproc / f1;

        // Il rapporto di tra le dimensioni di ogni sottogriglia deve essere vicino a quello della griglia
        double current_ratio = (double)(rows/f1) / (cols/f2);
        double diff = abs(current_ratio - ratio);

        if(diff < best_diff){
            best_diff = diff;
            ndim[0] = f1;
            ndim[1] = f2;
        }
    }  

    if(verbose)
        printf("Griglia ottimale: %d x %d\n", ndim[0], ndim[1]);
}