/*
Prodotto scalare di due vettori - Strategia 3
La GPU calcola le somme parziali e le combina secondo una strategia ottimizzata.
Non viene generata divergenza tra i thread durante la riduzione.

Dall'analisi dell'hardware della GPU, è emerso che:
- Max threads per SM = 1536
- Max blocks per SM = 16

Ponendo la blockDim.x = 512 (equivalente al numero di thread per blocco):
- Numero di blocchi per SM = 1536 / 512 = 3 con una occupancy del 100%
*/
#include <assert.h>
#include <stdio.h>
#include<cuda.h>
#include <time.h>

float sommaCPU(float *, int);
__global__ void ProdGPU_Reduction(float *, const float *, const float *, int N);

int main() {
    float *h_a, *h_b, *h_v, h_res;
    float *d_a, *d_b, *d_v;
    int N, nBytes, nBytesV, i; 
    dim3 gridDim, blockDim;
    size_t sBytes;
    float elapsed;

    printf("***\t PRODOTTO SCALARE STRATEGIA 2 \t***\n");
    printf("Inserire la dimensione degli array: ");
    scanf("%d", &N);

    // Definisco la configurazione della griglia
    blockDim.x = 512; // 128 thread in una sola riga
    gridDim.x = N / blockDim.x + (( N % blockDim.x)== 0? 0 : 1);

    // Dimensione dei vetori di input
    nBytes = N * sizeof(float);
    // Dimensione del vettore delle somme parziali (uno per blocco)
    nBytesV = gridDim.x * sizeof(float);
    // dimensione della shared memory dinamica: un float per ogni thread del blocco
    sBytes = blockDim.x * sizeof(float);

    // alloco i vettori in host e device
    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    h_v = (float*)malloc(nBytesV);
    cudaMalloc((void**)&d_a, nBytes);
    cudaMalloc((void**)&d_b, nBytes);
    cudaMalloc((void**)&d_v, nBytesV);

    // Inizializzazione vettori a e b (assumono valore casuale tra -2 e 2)
    srand((unsigned)time(NULL));
    for(i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 5 - 2);
        h_b[i] = (float)(rand() % 5 - 2);
    }

    // Copio i vettori dall'host al device
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // Inizializzo il vettore dei risultati a zero
    memset(h_v, 0, nBytesV);
    cudaMemset(d_v, 0, nBytesV);
    
    // calcolo del tempo
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    // lancio del kernel, viene inclusa anche la dimensione della shared memory dinamica
    ProdGPU_Reduction<<<gridDim, blockDim, sBytes>>>(d_v, d_a, d_b, N);

    // viene calcolato il tempo di fine esecuzione
    cudaEventRecord(stop);
    
    // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventSynchronize(stop);

    // tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("tempo GPU=%f\n", elapsed);

    // copio il vettore dei risultati dal device all'host
    cudaMemcpy(h_v, d_v, nBytesV, cudaMemcpyDeviceToHost);
    h_res = sommaCPU(h_v, gridDim.x);

    if(N < 20){
        printf("Vettore A: ");
        for(i = 0; i < N; i++) {
            printf("%.2f ", h_a[i]);
        }
        printf("\nVettore B: ");
        for(i = 0; i < N; i++) {
            printf("%.2f ", h_b[i]);
        }
        printf("\nVettore V: ");
        for(i = 0; i < gridDim.x; i++) {
            printf("%.2f ", h_v[i]);
        }
        printf("\n");
    }

    printf("Risultato: %.2f\n", h_res);

    free(h_a); free(h_b); free(h_v);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_v);

    return 0;
}

// Seriale - somma tutti i prodotti calcolati dalla GPU, producendo uno scalare
float sommaCPU(float *a, int n){
	float sum = 0;
    for(int i=0; i < n; i++)
        sum += a[i];
    return sum;
}

__global__ void ProdGPU_Reduction(float *v, const float *a, const float *b, int N) {
    int recvBy;
    // nthrd sostinuisce nproc
    int nthrd = blockDim.x;
    // l'identificativo del thread nel blocco
    int menum = threadIdx.x; 
    
    // Memoria condivisa dinamica
    extern __shared__ float s[];
    
    // indice globale del thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ogni thread calcola il prodotto delle sue componenti
    float w = 0.0f;
    if (idx < N) {
        w = a[idx] * b[idx];
    }

    // Carico il valore calcolato in memoria condivisa
    s[threadIdx.x] = w;
    __syncthreads(); // sincronizzo i thread del blocco

    /* Riduzione in memoria condivisa - Strategia Ottimizzata per la somma */
    for(int p = nthrd / 2; p > 0; p /= 2){
        if(menum < p){
            recvBy = menum + p;
            s[menum] += s[recvBy];
        }
        __syncthreads();
    }

    // Dopo l'ultimo passo, il thread 0 del blocco ha la somma del blocco
    if (threadIdx.x == 0) {
        v[blockIdx.x] = s[0]; 
    }
}
