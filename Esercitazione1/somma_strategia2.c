/*
Sviluppare e implementare in linguaggio C--MPI un algoritmo parallelo per il 
calcolo della somma di n numeri reali, che utilizzi la II strategia di 
parallelizzazione. 
*/ 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char **argv)
{
	// Id processore, numero processori, tag messaggio
    int menum, nproc, tag; 
    // numero addendi, addendi locali, indice, somma totale, resto divisione, addendi locali generali
    int n, nloc, i, somma, resto, nlocgen; 
    // indice, log_2(nproc), resto, invio a, ricevo da, variabile temporanea
    int ind, p, r, sendTo, recvBy, tmp; 
    // vettore potenze di 2, vettore globale, vettore locale, numero passi
    int *potenze, *vett, *vett_loc, passi = 0; 
	int sommaloc = 0;
	double T_inizio,T_fine,T_max;

	MPI_Status info;
	
	// Inizializzazione dell'ambiente di calcolo MPI
	MPI_Init(&argc,&argv);
	// assegnazione IdProcessore a menum
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	// assegna numero processori a nproc
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	// lettura e inserimento dati
	if (menum==0)
	{
		printf("Inserire il numero di elementi da sommare: \n");
		fflush(stdout);
		scanf("%d",&n);
		
        vett=(int*)calloc(n,sizeof(int));
	}

	// invio del valore di n a tutti i processori appartenenti a MPI_COMM_WORLD
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    // numero di addendi da assegnare a ciascun processore
	nlocgen=n/nproc; // divisione intera
    resto=n%nproc; // resto della divisione

	// Se resto è non nullo, i primi resto processi ricevono un addento in più
	if(menum<resto) {
		nloc = nlocgen + 1;
	}
	else {
		nloc = nlocgen;
	}
	
    // allocazione di memoria del vettore per le somme parziali
	vett_loc=(int*)calloc(nloc, sizeof(int));

    /* 
    Il primo processore P0 (menum==0) inizializza il vettore con i numeri casuali e distribuisce i vettori locali agli altri processori.
    Gli altri processori (menum!=0) ricevono il vettore locale da P0.
    */
	if (menum==0)
	{
        // Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema              
        srand((unsigned int) time(0)); 
		
        for(i = 0; i < n; i++)
		{
			// creazione del vettore contenente numeri casuali 
			*(vett+i) = (int)rand()%5-2;
		}
		
   		// Stampa del vettore che contiene i dati da sommare, se sono meno di 100 
		if (n < 100)
		{
			for (i=0; i<n; i++)
			{
				printf("\nvett[%d] = %d ",i,*(vett+i));
			}
        }

		// assegnazione dei primi addendi a P0
        for(i = 0; i < nloc; i++)
		{
			*(vett_loc+i) = *(vett+i);
		}

  		// ind è il numero di addendi già assegnati     
		ind = nloc;
        
		// P0 assegna i restanti addendi agli altri processori 
		for(i = 1; i < nproc; i++)
		{
			// il tag del messaggio è uguale all'id del processo che riceve
			tag = i;
			// Ripartisce tra i processori gli addendi in sovrannumero (se ci sono)
            if (i<resto) 
			{
				// il processore P0 gli invia il corrispondete vettore locale considerando un addendo in piu'
				MPI_Send(vett+ind,nloc,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind=ind+nloc;
			} 
			else 
			{
				// il processore P0 gli invia il corrispondete vettore locale considerando un addendo in piu'
				MPI_Send(vett+ind,nlocgen,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind = ind + nlocgen;
			}
		}
	} else {
		// tag è uguale numero di processore che riceve
		tag=menum;
		// fase di ricezione
		MPI_Recv(vett_loc,nloc,MPI_INT,0,tag,MPI_COMM_WORLD,&info);
	}
	
	// sincronizzazione dei processori del contesto MPI_COMM_WORLD
	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_inizio=MPI_Wtime(); // calcolo del tempo di inizio

	// ogni processore effettua la somma parziale
	for(i = 0; i < nloc; i++) {
		sommaloc = sommaloc+*(vett_loc+i);
	}

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

	// Calcolo delle altre somme parziali e combinazione dei risultati parziali
	for(i=0;i<passi;i++){
		// Calcolo identificativo del processore: resto(menum, 2^(k+1))
		r=menum%potenze[i+1];
		
		// Se il resto(menum, 2^(k+1))= 2^k, il processore menum invia
		if(r==potenze[i]) {
			// calcolo dell'id del processore a cui spedire la somma locale: menum - DIST
			sendTo=menum-potenze[i];
			tag=sendTo;
			MPI_Send(&sommaloc,1,MPI_INT,sendTo,tag,MPI_COMM_WORLD);
		} 
		// Se il resto e' uguale a 0, il processore menum riceve
		else if(r == 0) {
			// ricevo da menum + DIST
			recvBy=menum+potenze[i];
			tag=menum;
			MPI_Recv(&tmp,1,MPI_INT,recvBy,tag,MPI_COMM_WORLD,&info);
			// calcolo della somma parziale solo nel ricevente
			sommaloc=sommaloc+tmp;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine = MPI_Wtime()-T_inizio; // calcolo del tempo di fine

	// calcolo del tempo totale di esecuzione
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	// stampa a video dei risultati finali
	if(menum==0)
	{
		printf("\nProcessori impegnati: %d\n", nproc);
		printf("\nLa somma e': %d\n", sommaloc);
		printf("\nTempo calcolo locale: %lf\n", T_fine);
		printf("\nMPI_Reduce max time: %f\n",T_max);
	}

	// chiusura ambiente MPI
	MPI_Finalize();
	return 0;
}
