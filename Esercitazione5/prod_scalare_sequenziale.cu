#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

float prodScalareCPU(float *, float *, int);

int main (){
    float *a, *b, res;
    int N, i;
    float elapsed;
    cudaEvent_t start, stop;

    printf("***\t PRODOTTO SCALARE SEQUENZIALE \t***\n");
    printf("Inserisci N: ");
    scanf(" %d", &N);

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    srand((unsigned int) time(0));
    for (i = 0; i < N; i++) {
        a[i] = rand() % 5 - 2;
        b[i] = rand() % 5 - 2;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    res = prodScalareCPU(a, b, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("tempo CPU = %f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if(N < 20){
        printf("Vettore A: ");
        for(i = 0; i < N; i++) printf("%.2f ", a[i]);
        printf("\nVettore B: ");
        for(i = 0; i < N; i++) printf("%.2f ", b[i]);
        printf("\n");
    }

    printf("Risultato: %.2f\n", res);

    free(a);
    free(b);

    return 0;
}

float prodScalareCPU(float *a, float *b, int N){
    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += a[i] * b[i];
    }
    return sum;
}
