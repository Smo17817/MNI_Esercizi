#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *h_A = 0;     // Host matrice a (M x N)
    float *d_A;         // Device matrice a
    float *h_b = 0;     // Host array b (N)
    float *d_b;         // Device array b
    float *h_res = 0;   // Host risultato (M)
    float *d_res;       // Device risultato
    int M, N, i, j;     // M righe ed N colonne
    float elapsed;
    cudaEvent_t start, stop;
	
	printf("***\t PRODOTTO MATRICE-VETTORE CUBLAS \t***\n");
    printf("Inserisci M ed N (separati da spazio): ");
    scanf(" %d", &M);
    scanf(" %d", &N);

    // Allocazione di h_a
    h_A = (float *)malloc (M * N * sizeof (*h_A)); 
    if (!h_A) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Allocazione di h_b
    h_b = (float *)malloc (N * sizeof (*h_b)); 
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Allocazione di h_res
    h_res = (float *)malloc (M * sizeof (*h_res)); 
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Inizializzazione di h_A in column-major
    srand((unsigned int) time(0));
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            h_A[i + j*M] = rand() % 5 - 2;
        }
    }


    // Inizializzazione di h_b
    for (i = 0; i < N; i++) {
        h_b[i] = i;
    }
    
    // Allocazione di d_a
    cudaStat = cudaMalloc ((void**)&d_A, M*N*sizeof(*h_A)); 
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

    // Allocazione di d_res
    cudaStat = cudaMalloc ((void**)&d_res, M*sizeof(*h_res));
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
    stat = cublasSetMatrix(M, N, sizeof(float), h_A, M, d_A, M); 
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_A);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    // Copio il vettore b da host (h_b) a device (d_b)
    stat = cublasSetVector(N,sizeof(float), h_b, 1, d_b,1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    stat = cublasSetVector(M,sizeof(float), h_res, 1, d_res,1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_res);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Eseguo il calcolo del prodotto scalare su device
    float alpha = 1.0f, beta=0.0f;
    stat = cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_b, 1, &beta, d_res, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
        // Errore
        printf ("data download failed cublasSdot");
        cudaFree (d_A);
        cudaFree (d_b);
        cudaFree (d_res);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Tempo: %f\n", elapsed);

    // Copia del risultato su host
    stat = cublasGetVector(M, sizeof(float), d_res, 1, h_res, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed (y)\n");
        cublasDestroy(handle);
        cudaFree(d_A); 
        cudaFree(d_b); 
        cudaFree(d_res);    
        return EXIT_FAILURE;
    }

    if(N < 10 && M < 10){
        printf("\nMatrice A:\n");
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                printf("%.2f ", h_A[i + j*M]);
            }
            printf("\n");
        }
        printf("\nVettore B: ");
        for(i = 0; i < N; i++) printf("%.2f ", h_b[i]);
        printf("\nVettore Risultato: ");
        for(i = 0; i < M; i++) printf("%.2f ", h_res[i]);
        printf("\n");
    }
    
    cudaFree(d_A); 
    cudaFree(d_res); 
    cudaFree(d_res);
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_A); 
    free(h_res); 
    free(h_res);  

    return EXIT_SUCCESS;
}