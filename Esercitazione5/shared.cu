#include<cuda.h>
#include<stdio.h>
#include <assert.h>

__global__ void staticReverse(int *d, int n);
__global__ void dynamicReverse(int *d, int n);

int main(void) {
  const int n = 64;
  int i, a[n], r[n], d[n];
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 

// run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  
  printf("Static reverse check \n");
  for (i=0; i< n; i++) 
    assert( d[i] == r[i] );

  /*for(i=0;i<n;i++) 
    printf("d[%d]=%d ",i, d[i]);
  printf("\n");
	for(i=0;i<n;i++)
    printf("r[%d]=%d ",i, r[i]);
	printf("\n\n");*/

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  
  printf("Dynamic reverse check\n");
  for (i=0; i< n; i++) 
    assert( d[i] == r[i] );

  /*for(i=0;i<n;i++)
    printf("d[%d]=%d ",i, d[i]);
  printf("\n");
	for(i=0;i<n;i++)
		printf("r[%d]=%d ",i, r[i]);
	printf("\n");*/
  
  cudaFree(d_d);
  return 0;
}

__global__ void staticReverse(int *d, int n) {
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n) {
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

