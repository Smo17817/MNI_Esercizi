#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float* h_a = 0;     // Host array a
    float* d_a;         // Device array a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float result = 0;   // Risultato finale
    int N, i;
    float elapsed;
    cudaEvent_t start, stop;
	
	printf("***\t PRODOTTO SCALARE CUBLAS \t***\n");
    printf("Inserisci N: ");
    scanf(" %d", &N);

    // Allocazione di h_a
    h_a = (float *)malloc (N * sizeof (*h_a)); 
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Allocazione di h_b
    h_b = (float *)malloc (N * sizeof (*h_b)); 
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Inizializzazione di h_a e h_b
    srand((unsigned int) time(0));
    for (i = 0; i < N; i++) {
        h_a[i] = rand() % 5 - 2;
        h_b[i] = rand() % 5 - 2;
    }
    
    // Allocazione di d_a
    cudaStat = cudaMalloc ((void**)&d_a, N*sizeof(*h_a)); 
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    // Allocazione di d_b
    cudaStat = cudaMalloc ((void**)&d_b, N*sizeof(*h_b));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    // Creo l'handle per cublas - racchiude lo stato
    stat = cublasCreate(&handle);               
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    // Copio il vettore a da host (h_a) a device (d_a)
    stat = cublasSetVector(N,sizeof(float),h_a,1,d_a,1); 
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    // Copio il vettore b da host (h_b) a device (d_b)
    stat = cublasSetVector(N,sizeof(float),h_b,1,d_b,1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Eseguo il calcolo del prodotto scalare su device
    stat = cublasSdot(handle, N, d_a, 1, d_b, 1, &result);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
        // Errore
        printf ("data download failed cublasSdot");
        cudaFree (d_a);
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Tempo: %f\n", elapsed);

    if(N < 20){
        printf("Vettore A: ");
        for(i = 0; i < N; i++) printf("%.2f ", h_a[i]);
        printf("\nVettore B: ");
        for(i = 0; i < N; i++) printf("%.2f ", h_b[i]);
        printf("\n");
    }
    
    printf("Risultato cuBLAS: %.2f", result);
    
    cudaFree (d_a);     // Dealloco d_a
    cudaFree (d_b);     // Dealloco d_b
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b    
    return EXIT_SUCCESS;
}