/*
Implementazione del prodotto matrice vettore utilizzando la strategia 2: 
ogni processo riceve un blocco di colonne della matrice e una parte del vettore della stessa dimensione del numero di colonne e ne calcola il prodotto.
Il processo root raccoglie i risultati parziali dai processi figli e li somma. Per la somma verrà urilizzata la strategia 2 dell'Esercitazione 1.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

// Funzione che esegue il prodotto matrice trasversa vettore
void prod_mat_trasv_vett(double w[], double *localAT, int m, int local_n, double v[]) {
    int i, j;

    // localAT ha dimensione local_n x m (memorizzata per righe)
    for (i = 0; i < m; i++)
        w[i] = 0.0;

    for (i = 0; i < local_n; i++) {
        for (j = 0; j < m; j++) {
            w[j] += localAT[i*m + j] * v[i];
        }
    }
}

// Funzione che esegue il prodotto locale (su localAT x local_v) e poi effettua la somma distribuita
// Risultato finale è presente in 'w' solo sul processo 0.
void prod_mat_trasv_vett_distr(double w[], double *localAT, int m, int local_n, double v[]) {
    int i, j;
    int me, nproc;
    int tag, sendTo, recvBy;
    int p, resto, *potenze, passi = 0;
    MPI_Status info;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Inizializzazione del vettore risultato locale
    for (i = 0; i < m; i++) 
        w[i] = 0.0;

    // esecuzione del prodotto locale
    for (i = 0; i < local_n; i++) {
        double vi = v[i]; // elemento corrispondente del blocco di vettore
        for (j = 0; j < m; j++) {
            w[j] += localAT[i * m + j] * vi;
        }
    }

    /* SOMMA STRATEGIA 2 */

    // vettore temporaneo per ricezione dati
    double *tmp = (double *) malloc(m * sizeof(double));

    // calcolo di p=log_2 (nproc) attraverso uno shift a destra bit a bit
	p = nproc;

	while(p!=1) {
		p = p>>1; //shift
		passi++;
	}

	// creazione del vettore potenze, che contiene le potenze di 2
	potenze=(int*)calloc(passi+1,sizeof(int));
		
	for(i = 0; i <= passi; i++) {
		potenze[i] = p<<i;
	}

    for(i=0; i < passi; i++){
        // Calcolo identificativo del processore: resto(me, 2^(k+1))
        resto = me % potenze[i+1];

        // Se il resto(me, 2^(k+1))= 2^k, il processore me invia
        if (resto == potenze[i]) {
            // calcolo dell'id del processore a cui spedire la somma locale: me - DIST
			sendTo=me - potenze[i];
            tag = sendTo;
            MPI_Send(w, m, MPI_DOUBLE, sendTo, tag, MPI_COMM_WORLD);
        } 
        // Se il resto e' uguale a 0, il processore me riceve
        else if (resto == 0) {
            // ricevo da me + DIST
			recvBy=me + potenze[i];
            tag = me;
            MPI_Recv(tmp, m, MPI_DOUBLE, recvBy, tag, MPI_COMM_WORLD, &info);

            // calcolo della somma parziale solo nel ricevente
            for (j = 0; j < m; j++) 
                w[j] += tmp[j];
        }
    }
}

int main(int argc, char **argv) {
    int nproc; // Numero di processi totale
    int me; // Il mio id
    int m,n; // Dimensione della matrice
    int local_n; // Dimensione dei dati locali
    int i,j; // Iteratori vari
    double T_inizio, T_fine, T_max; // Variabili per il calcolo del tempo

    /*
    A: matrice m x n
    AT: trasposta di A di dimensione n x m
    v: vettore di dimensione n
    localA: parte di matrice locale di dimensione m x local_n
    local_v: parte di vettore locale di dimensione local_n
    local_w: vettore locale di dimensione m
    w: vettore risultato di dimensione m
    */
    double *A, *AT, *v, *localAT, *local_v, *local_w, *w;

    // Attiva MPI
    MPI_Init(&argc, &argv);
    // Trova il numero totale dei processi
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    // Da ad ogni processo il proprio numero identificativo
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

            printf("\n");
        } 
    }

    fflush(stdout);

    // Spedisco m, n, local_n a tutti i processi
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Se sono un figlio alloco local_v
    local_v = malloc(local_n * sizeof(double));

    // Alloco localAT e local_w per tutti i processi
    localAT = malloc(local_n * m * sizeof(double));
    local_w = malloc(m * sizeof(double));

    // Invia la parte di vettore v
    MPI_Scatter(v, local_n, MPI_DOUBLE,
                local_v, local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Invia le righe (blocchi di colonne originali)
    MPI_Scatter(AT, local_n * m, MPI_DOUBLE,
                localAT, local_n * m, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Scriviamo la matrice locale e il vettore locale ricevuti
    if (n<11 && m<11)
    {
        printf("local_v %d: ", me);
        for(i = 0; i < local_n; i++){
            printf("%lf\t", local_v[i]);
        }
        printf("\n");

        printf("localAT %d\n", me); 
        for(i = 0; i < local_n; i++){
            for(j = 0; j < m; j++)
                printf("%lf\t", localAT[i*m+j]);
            printf("\n");
        } 
        printf("\n");
    }

    // sincronizzazione dei processori del contesto MPI_COMM_WORLD
	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_inizio=MPI_Wtime(); // calcolo del tempo di inizio

    double *result = (me==0) ? w : local_w;
    prod_mat_trasv_vett_distr(result, localAT, m, local_n, local_v);

    MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine = MPI_Wtime()-T_inizio; // calcolo del tempo di fine

    // calcolo del tempo totale di esecuzione
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    // stampa a video dei risultati finali
	if(me==0) {
        printf("\n--- STATISTICHE ---\n");
		printf("Processori impegnati: %d\n", nproc);
		printf("Tempo calcolo locale: %lf\n", T_fine);
		printf("MPI_Reduce max time: %f\n",T_max);

        if(n < 11 && m < 11) {
            printf("\nRisultato finale w: ");
            for (i = 0; i < m; i++)
                printf("%.4f ", w[i]);
            printf("\n");
        }
	}
    printf("\n");

    MPI_Finalize();
    return 0;
}