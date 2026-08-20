#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU kernel
__global__ void vectorAdd(const float *x,
                          const float *y,
                          float *z,
                          int N)
{
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

int main()
{

    const int N = 1000;

    size_t size = N * sizeof(float);

    // -----------------------------
    // 1. Allocate HOST memory
    // -----------------------------
    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);
    float *z = (float *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(2 * i);
    }

    // -----------------------------
    // 2. Allocate DEVICE memory
    // -----------------------------
    float *x_d;
    float *y_d;
    float *z_d;

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);
    cudaMalloc((void **)&z_d, size);

    // -----------------------------
    // 3. Copy HOST -> DEVICE
    // -----------------------------
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    // -----------------------------
    // 4. Configure GPU execution
    // -----------------------------
    int threadsPerBlock = 256;

    int numBlocks =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Vector size       : %d\n", N);
    printf("Threads per block : %d\n", threadsPerBlock);
    printf("Number of blocks  : %d\n", numBlocks);
    printf("Total GPU threads : %d\n\n",
           numBlocks * threadsPerBlock);

    // -----------------------------
    // 5. Launch GPU kernel
    // -----------------------------
    vectorAdd<<<numBlocks, threadsPerBlock>>>(
        x_d, y_d, z_d, N
    );

    // Check kernel launch
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n",
               cudaGetErrorString(error));
        return 1;
    }

    // Wait until GPU computation finishes
    cudaDeviceSynchronize();

    // -----------------------------
    // 6. Copy DEVICE -> HOST
    // -----------------------------
    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    // -----------------------------
    // 7. Check results
    // -----------------------------
    bool correct = true;

    for (int i = 0; i < N; i++) {

        float expected = x[i] + y[i];

        if (z[i] != expected) {
            printf("Error at index %d\n", i);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Vector addition completed successfully!\n\n");
    }

    // Print first 10 results
    printf("First 10 results:\n");

    for (int i = 0; i < 10; i++) {
        printf(
            "Thread %d: %.1f + %.1f = %.1f\n",
            i,
            x[i],
            y[i],
            z[i]
        );
    }

    // -----------------------------
    // 8. Free DEVICE memory
    // -----------------------------
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    // -----------------------------
    // 9. Free HOST memory
    // -----------------------------
    free(x);
    free(y);
    free(z);

    return 0;
}