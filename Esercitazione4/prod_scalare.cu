/*
Prodotto scalare di due vettori
1. La GPU calcola il prodotto elemento per elemento;
2. La CPU somma tutti i prodotti calcolati dalla GPU, producendo uno scalare.

Eseguire: pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
Si consiglia di usare CUDA Toolkit fino a 12.6
*/ 
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>
#include <math.h>

float sommaCPU(float *a, int n);
__global__ void prodottoGPU(float* a, float* b, float* c, int n);

int main (){
    float *h_u, *h_v, *h_res, h_res2;
    float *d_u, *d_v, *d_res;
    int N, nBytes;
    dim3 gridDim, blockDim;
    float elapsed; // calcolo del tempo

    printf("***\t PRODOTTO SCALARE DI DUE VETTORI \t***\n");
	printf("Inserisci N: ");
    scanf(" %d",&N);

    nBytes = N * sizeof(float);
    h_u = (float *)malloc(nBytes);
    h_v = (float *)malloc(nBytes);
    h_res = (float *)malloc(nBytes);
    cudaMalloc((void **) &d_u, nBytes);
    cudaMalloc((void **) &d_v, nBytes);
    cudaMalloc((void **) &d_res, nBytes);

    // la generazione randomica dei vettori segue l'ora attuale del sistema
    srand((unsigned int) time(0));

    for (int i = 0; i < N; i++) {
        h_u[i] = rand()%5-2;
        h_v[i] = rand()%5-2;
    }

    // i vettori inizializzati vengono copiati dall'host al device
    cudaMemcpy(d_u, h_u, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v, nBytes, cudaMemcpyHostToDevice);

    // il contenuto del vettore res viene posto a 0 sia in host che device
	memset(h_res, 0, nBytes);
	cudaMemset(d_res, 0, nBytes);

    //configurazione del kernel
	blockDim.x=128;

    // la griglia risultante è 1D - se il resto !=0 viene distribuito un addendo aggiuntivo
	gridDim.x= N / blockDim.x + (( N % blockDim.x)== 0? 0 : 1);

    // calcolo del tempo
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    //invocazione del kernel
	prodottoGPU<<<gridDim, blockDim>>>(d_u, d_v, d_res, N);

	cudaEventRecord(stop);
    // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventSynchronize(stop);


	// tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("tempo GPU=%f\n", elapsed);

    // copia nuovamente il risultato sull'host
	cudaMemcpy(h_res, d_res, nBytes, cudaMemcpyDeviceToHost);

	// calcolo su CPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// calcolo somma seriale
	h_res2 = sommaCPU(h_res, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("tempo CPU=%f\n", elapsed);

	if (N < 20) {
        // stampa vettore u
		for(int i = 0; i < N; i++)
			printf("h_u[%d]=%6.2f ", i, h_u[i]);
		printf("\n");
        // stampa vettore v
		for(int i = 0; i < N; i++)
			printf("h_v[%d]=%6.2f ",i, h_v[i]);    
		printf("\n");
		// stampa vettore res
		for(int i = 0; i < N; i++)
			printf("h_res[%d]=%6.2f ", i, h_res[i]);
		printf("\n");
	}

	printf("Risultato: %6.2f\n", h_res2);

	free(h_u); free(h_v); free(h_res);
	cudaFree(d_u); cudaFree(d_v); cudaFree(d_res);
	return 0;
}

// Seriale - somma tutti i prodotti calcolati dalla GPU, producendo uno scalare
float sommaCPU(float *a, int n){
	float sum = 0;
    for(int i=0; i < n; i++)
        sum += a[i];
    return sum;
}

// Parallelo - La GPU calcola il prodotto elemento per elemento
__global__ void prodottoGPU (float* a, float * b, float* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n)
		c[index] = a[index] * b[index];
}

