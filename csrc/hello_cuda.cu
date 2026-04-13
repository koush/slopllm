#include "hello_cuda.h"
#include <cstdio>

__global__ void hello_kernel() {
}

void launch_hello_kernel() {
  hello_kernel<<<1, 1>>>();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
  } else {
    printf("CUDA kernel launched and completed successfully!\n");
  }
}
