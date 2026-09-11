#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;

            for (int k = 0; k < N; ++k) {
                sum += (double)A[i * N + k] * (double)B[k * N + j];
            }

            C[i * N + j] = (float)sum;
        }
    }
}

bool verify(
    const float* cpu,
    const float* gpu,
    int N,
    float tol
) {
    float max_rel_err = 0.0f;
    int max_index = 0;

    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float denom = fabsf(cpu[i]) + 1e-6f;
        float rel = diff / denom;

        if (rel > max_rel_err) {
            max_rel_err = rel;
            max_index = i;
        }
    }

    printf(
        "The maximum relative error is %e\n",
        max_rel_err
    );
    printf(
        "The CPU value at the maximum relative error is %e\n",
        cpu[max_index]
    );
    printf(
        "The GPU value at the maximum relative error is %e\n",
        gpu[max_index]
    );
    printf(
        "The absolute error at the maximum relative error is %e\n",
        fabsf(cpu[max_index] - gpu[max_index])
    );

    return max_rel_err < tol;
}

int main() {
    int N = 2048;
    const size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        fprintf(
            stderr,
            "Host memory allocation failed\n"
        );

        return 1;
    }

    srand(42);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(
        &d_A,
        bytes
    );
    cudaMalloc(
        &d_B,
        bytes
    );
    cudaMalloc(
        &d_C,
        bytes
    );
    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );
    dim3 block(
        BLOCK_SIZE,
        BLOCK_SIZE
    );
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    matmul_naive<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Kernel launch failed, %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Warm-up execution failed, %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int trial = 0; trial < 5; ++trial) {
        matmul_naive<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            N
        );
    }

    cudaEventRecord(stop);
    err = cudaEventSynchronize(stop);

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Kernel execution failed, %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    float ms;
    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );
    float ms_per_trial = ms / 5.0f;
    cudaMemcpy(
        h_C_gpu,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );
    matmul_cpu(
        h_A,
        h_B,
        h_C_cpu,
        N
    );
    printf(
        "The first GPU result is %f\n",
        h_C_gpu[0]
    );
    printf(
        "The first CPU result is %f\n",
        h_C_cpu[0]
    );
    float max_abs_err = 0.0f;

    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);

        if (diff > max_abs_err) {
            max_abs_err = diff;
        }
    }

    printf(
        "The maximum absolute error is %e\n",
        max_abs_err
    );
    bool correct = verify(
        h_C_cpu,
        h_C_gpu,
        N,
        1e-3f
    );
    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms_per_trial * 1e-3) / 1e9;
    printf(
        "N is %d\n",
        N
    );
    printf(
        "The block size is %d by %d\n",
        BLOCK_SIZE,
        BLOCK_SIZE
    );
    printf(
        "The grid size is %d by %d\n",
        grid.x,
        grid.y
    );
    printf(
        "The kernel time is %.3f ms\n",
        ms_per_trial
    );
    printf(
        "The achieved GFLOPS is %.2f\n",
        gflops
    );
    printf(
        "The verification result is a %s\n",
        correct ? "pass" : "fail"
    );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return correct ? 0 : 1;
}