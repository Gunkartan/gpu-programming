#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void mat_mul_kernel(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }
}

void mat_mul_cpu(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;

            for (int k = 0; k < n; k++) {
                sum += a[row * n + k] * b[k * n + col];
            }

            c[row * n + col] = sum;
        }
    }
}

int main(
    int argc,
    char **argv
) {
    if (argc != 2) {
        printf(
            "Usage, %s N\n",
            argv[0]
        );

        return 1;
    }

    int n = atoi(argv[1]);
    int matches = 0;
    size_t size = n * n * sizeof(float);
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);
    
    for (int i = 0; i < n * n; i++) {
        h_a[i] = (float)(i + 1);
        h_b[i] = (float)(i + 1);
    }
    
    mat_mul_cpu(
        h_a,
        h_b,
        h_c_cpu,
        n
    );
    float *d_a, *d_b, *d_c;
    cudaMalloc(
        (void **)&d_a,
        size
    );
    cudaMalloc(
        (void **)&d_b,
        size
    );
    cudaMalloc(
        (void **)&d_c,
        size
    );
    cudaMemcpy(
        d_a,
        h_a,
        size,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_b,
        h_b,
        size,
        cudaMemcpyHostToDevice
    );
    dim3 block(
        16,
        16
    );
    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );
    mat_mul_kernel<<<grid, block>>>(
        d_a,
        d_b,
        d_c,
        n
    );
    cudaMemcpy(
        h_c,
        d_c,
        size,
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float cpu_value = h_c_cpu[i * n + j];
            float gpu_value = h_c[i * n + j];
            float difference = fabsf(cpu_value - gpu_value);
            float tolerance = 1e-5f * fmaxf(
                1.0f,
                fabsf(cpu_value)
            );

            if (difference <= tolerance) {
                matches++;
            }
        }
    }

    if (matches == n * n) {
        printf("Pass\n");
    } else {
        printf("Fail\n");
    }

    if (n <= 4) {
        printf("Matrix A\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%6.1f", h_a[i * n + j]);
            }

            printf("\n");
        }

        printf("Matrix B\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%6.1f", h_b[i * n + j]);
            }

            printf("\n");
        }

        printf("Matrix C\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%6.1f", h_c[i * n + j]);
            }

            printf("\n");
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    return 0;
}