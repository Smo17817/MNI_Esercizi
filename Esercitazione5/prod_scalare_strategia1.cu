/*
Prodotto scalare di due vettori - Strategia 1
La GPU calcola le somme parziali, che verranno combinate dalla CPU

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
__global__ void dotProdGPU(float *, const float *, const float *, int);

int main() {
    float *h_a, *h_b, *h_v, h_res;
    float *d_a, *d_b, *d_v;
    int N, nBytes, i; 
    dim3 gridDim, blockDim;
    float elapsed;

    printf("***\t PRODOTTO SCALARE STRATEGIA 1 \t***\n");
    printf("Inserire la dimensione degli array: ");
    scanf("%d", &N);

    nBytes = N * sizeof(float);

    // alloco i vettori in host e device
    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    h_v = (float*)malloc(nBytes);
    cudaMalloc((void**)&d_a, nBytes);
    cudaMalloc((void**)&d_b, nBytes);
    cudaMalloc((void**)&d_v, nBytes);

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
    memset(h_v, 0, nBytes);
    cudaMemset(d_v, 0, nBytes);

    // Definisco la configurazione della griglia
    blockDim.x = 512; // numero di thread in una sola riga
    gridDim.x = N / blockDim.x + (( N % blockDim.x)== 0? 0 : 1);

    // calcolo del tempo
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    // lancio del kernel
    dotProdGPU<<<gridDim, blockDim>>>(d_v, d_a, d_b, N);

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
    cudaMemcpy(h_v, d_v, nBytes, cudaMemcpyDeviceToHost);

    // calcolo su CPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    
    h_res = sommaCPU(h_v, N);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("tempo CPU=%f\n", elapsed);

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
        for(i = 0; i < N; i++) {
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

__global__ void dotProdGPU(float *v, const float *a, const float *b, int N) {
    // indice globale del thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        v[idx] = a[idx] * b[idx];
    }
}