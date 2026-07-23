#include <stdio.h>
#include <cuda_runtime.h>
int main(void) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    printf("%d\n", p.l2CacheSize);
    return 0;
}
