#include <stdio.h>

__global__ void hello() {
    printf("Hello from my GPU\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Hello from my CPU\n");

    return 0;
}