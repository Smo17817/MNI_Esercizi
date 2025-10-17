#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int main(int argc, char *argv[]) {

    int n, m;
    int *matrice, *vettore, *risultato;
    double T_inizio, T_fine;

    MPI_Init(&argc, &argv);

    printf("Inserire il numero di righe e di colonne della matrice separate da spazio: "); 
    fflush(stdout);
    scanf("%d %d", &n, &m);

    matrice = (int*)calloc(n * m, sizeof(int));
    vettore = (int*)calloc(m, sizeof(int));
    risultato = (int*)calloc(n, sizeof(int));

    // Riempimento della matrice con valori di esempio
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            matrice[i * m + j] = (int)rand() % 5 - 2;
        }
    }

    // Riempimento del vettore con valori di esempio
    for(int i = 0; i < m; i++) {
        vettore[i] = (int)rand() % 5 - 2;
    }

    // stampa nel caso di dimensione della matrice e del vettore <= 100
    if(n * m <= 100){
        printf("\nMatrice:\n");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                printf("%d\t", matrice[i * m + j]);
            }
            printf("\n");
        }

        printf("\nVettore: ");
        for(int i = 0; i < m; i++) {
            printf("%d ", vettore[i]);
        }
        printf("\n\n");
    }

    T_inizio = MPI_Wtime();

    // Prodotto matrice-vettore
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            risultato[i] += matrice[i * m + j] * vettore[j];
        }
    }

    T_fine = MPI_Wtime() - T_inizio;

    // Stampa del risultato se m <= 10
    if (n <= 10){
        printf("\nRisultato: ");
        for(int i = 0; i < n; i++) {
            printf("%d ", risultato[i]);
        }
        printf("\n");
    }
    
    printf("\nTempo calcolo locale: %lf\n", T_fine);

    MPI_Finalize();

    return 0;
}