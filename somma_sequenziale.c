/*
Sviluppare e implementare in linguaggio C un algoritmo seriale per il calcolo della somma di n numeri reali. 
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

int main(int argc, char **argv){

  int *vett;
  int n, sum, i;
  double T_inizio, T_fine;
    
  /*Inizializzazione dell'ambiente di calcolo MPI*/
  MPI_Init(&argc,&argv);
    
  printf("Inserire il numero di elementi da sommare: ");
  fflush(stdout);
  scanf("%d", &n);

  vett = (int*)calloc(n, sizeof(int));

  // Generazione numeri casuali
  for(i=0; i<n; i++){
	  /*creazione del vettore contenente numeri casuali */
	  *(vett+i)=(int)rand()%5-2;
	}
		
   	// Stampa del vettore che contiene i dati da sommare, se sono meno di 100 
	if (n<100){
	  for (i=0; i<n; i++){
			printf("\nvett[%d]=%d ",i,*(vett+i));
	  }
  }

  T_inizio = MPI_Wtime();

  // Somma dei valori casuali
  sum = 0;
  for(i=0; i < n; i++){
    sum += vett[i];
  }

  T_fine = MPI_Wtime() - T_inizio;

  printf("\nLa somma e': %d\n", sum);
  printf("\nTempo calcolo locale: %lf\n", T_fine);

  MPI_Finalize();
}