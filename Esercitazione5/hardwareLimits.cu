/*
    Stampa le principali proprietà della GPU
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== SPECIFICHE GPU ===\n");

    printf("Nome GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    printf("\n--- Parallelismo ---\n");
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    printf("\n--- Memoria condivisa e registri ---\n");
    printf("Shared memory per SM: %zu B\n", prop.sharedMemPerMultiprocessor);
    printf("Shared memory per block: %zu B\n", prop.sharedMemPerBlock);
    printf("Max shared memory opt-in per block: %zu B\n", prop.sharedMemPerBlockOptin);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block: %d\n", prop.regsPerBlock);

    printf("\n--- Limiti griglia ---\n");
    printf("Max grid size: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max block dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    printf("\n--- Frequenze e memoria globale ---\n");
    printf("Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory clock rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory bus width: %d-bit\n", prop.memoryBusWidth);

    // Bandwidth stimata
    double bandwidthGBs = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    printf("Memory bandwidth teorica: %.2f GB/s\n", bandwidthGBs);

    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("L2 cache size: %d KB\n", prop.l2CacheSize / 1024);

    printf("\n--- Altro ---\n");
    printf("Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("Can map host memory: %d\n", prop.canMapHostMemory);
    printf("Unified addressing: %d\n", prop.unifiedAddressing);

    return 0;
}
