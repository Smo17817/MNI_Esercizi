/*
Calcola le statistiche della GPU
*/
#include <assert.h>
#include <stdio.h>
#include<cuda.h>
#include <time.h>

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Shared mem per SM: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("Shared mem per block: %zu\n", prop.sharedMemPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("SM count: %d\n", prop.multiProcessorCount);

    return 0;
}