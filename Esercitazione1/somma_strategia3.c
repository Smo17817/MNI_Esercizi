/*
Sviluppare e implementare in linguaggio C--MPI un algoritmo parallelo per il 
calcolo della somma di n numeri reali, che utilizzi la III strategia di 
parallelizzazione.
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

int main(int argc, char **argv){

    // Id processore, numero processori, tag messaggio
    int menum, nproc, tag; 
    // numero addendi, addendi locali, indice, somma totale, resto divisione, addendi locali generali
    int n, nloc, i, somma, resto, nlocgen; 
    // indice, log_2(nproc), resto, invio a, ricevo da, variabile temporanea
    int ind, p, r, sendTo, recvBy, tmp; 
    // vettore potenze di 2, vettore globale, vettore locale, numero passi
    int *potenze, *vett, *vett_loc, passi=0; 
    int sommaloc=0;
    double T_inizio, T_fine, T_max;

    MPI_Status info;

    // Inizializzazione ambiente MPI
    MPI_Init(&argc, &argv);
    // Assegnazione IdProcessore a menum
    MPI_Comm_rank(MPI_COMM_WORLD, &menum);
    // Assegna numero processori a nproc
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Lettura e inserimento dati
    if(menum==0){
        printf("Inserire il numero di elementi da sommare: \n");
        fflush(stdout);
        scanf("%d", &n);

        vett=(int*)calloc(n, sizeof(int));
        for(i=0; i<n; i++){
            vett[i]=i+1;
        }
    }

    // Invio del valore di n a tutti i processori appartenenti a MPI_COMM_WORLD
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Numero di addendi da assegnare a ciascun processore
    nlocgen=n/nproc;
    // Resto della divisione
    resto=n%nproc;

    // Se resto è non nullo, i primi processi ricevono un addendo in più
    if(menum < resto){
        nloc = nlocgen + 1;
    } else {
        nloc = nlocgen;
    }

    // Allocazione di memoria del vettore per le somme parziali
    vett_loc=(int*)calloc(nloc, sizeof(int));

    /* 
    Il primo processore (menum==0) inizializza il vettore con i numeri casuali e distribuisce i vettori locali agli altri processori.
    Gli altri processori (menum!=0) ricevono il vettore locale dal processore 0.
    */
    if (menum==0)
    {
        /*Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema*/                
        srand((unsigned int) time(0)); 
		
        for(i=0; i<n; i++)
		{
			/*creazione del vettore contenente numeri casuali */
			*(vett+i)=(int)rand()%5-2;
		}
		
   		// Stampa del vettore che contiene i dati da sommare, se sono meno di 100 
		if (n<100)
		{
			for (i=0; i<n; i++)
			{
				printf("\n vett[%d] = %d ", i, *(vett+i));
			}
        }

        // assegnazione dei primi addendi a P0  
        for(i = 0; i < nloc; i++)
		{
			*(vett_loc + i) = *(vett + i);
		}

        // ind è il numero di addendi già assegnati     
		ind=nloc;

        for(i = 1; i < nproc; i++)
		{
            // il tag del messaggio è uguale all'id del processo che riceve
			tag = i;
			// Ripartisce tra i processori gli addendi in sovrannumero (se ci sono)
            if (i < resto) 
			{
				// il processore P0 gli invia il corrispondete vettore locale considerando un addendo in piu'
				MPI_Send(vett+ind,nloc,MPI_INT,i,tag,MPI_COMM_WORLD);
                // aggiorna l'indice degli addendi già assegnati
				ind = ind + nloc;
			} 
			else {
				// il processore P0 gli invia il corrispondete vettore locale
				MPI_Send(vett+ind,nlocgen,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind = ind + nlocgen;
			}
		}   
    } else {
        // tag è uguale numero di processore che riceve
        tag = menum;
        // fase di ricezione
        MPI_Recv(vett_loc,nloc,MPI_INT,0,tag,MPI_COMM_WORLD,&info);
    }

    // Sincronizzazione di tutti i processi prima di iniziare il calcolo del tempo
    MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
    T_inizio = MPI_Wtime(); // calcolo del tempo di inizio

    // Calcolo iniziale (passo 0) della somma locale da parte di ogni processore
    for(i = 0; i < nloc; i++) {
		sommaloc=sommaloc + *(vett_loc + i);
	}

    // calcolo di p=log_2 (nproc) attraverso uno shift a destra bit a bit
    p = nproc;

    while(p != 1) {
        p = p>>1;
        passi++;
    }   
    
    // creazione del vettore potenze, che contiene le potenze di 2
    potenze = (int*)calloc(passi+1,sizeof(int));

    for(i = 0; i <= passi; i++) {
        potenze[i] = p<<i;
    }

    // Calcolo delle altre somme parziali e combinazione dei risultati parziali
    for(i = 0; i < passi; i++) {
        // Calcolo identificativo del processore: resto(menum, 2^(k+1))
        r=menum%potenze[i+1];
        // resto(menum, 2^(k+1)) < DIST, dove DIST = 2^k
        if(r<potenze[i]) {
            // Invio e ricevo da menum + DIST
            sendTo = menum + potenze[i];
            recvBy = sendTo;
            tag = sendTo;
            MPI_Sendrecv(&sommaloc, 1, MPI_INT, sendTo, tag, 
                &tmp, 1, MPI_INT, recvBy, tag, MPI_COMM_WORLD, &info);
        } else {
            // Invio e ricevo da menum - DIST
            sendTo = menum - potenze[i];
            recvBy = sendTo;
            tag = menum;
            MPI_Sendrecv(&sommaloc, 1, MPI_INT, sendTo, tag, 
                &tmp, 1, MPI_INT, recvBy, tag, MPI_COMM_WORLD, &info);       
        }
        // Calcolo della somma parziale al passo i
        sommaloc = sommaloc + tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine = MPI_Wtime()-T_inizio; // calcolo del tempo di fine

    /* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    /*stampa a video dei risultati finali*/
	if(menum == 0)
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