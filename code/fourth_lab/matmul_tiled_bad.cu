#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template <int TILE_WIDTH>
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float sub_tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_tile_b[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float p_value = 0.0f;

    for (int m = 0; m < N / TILE_WIDTH; m++) {
        sub_tile_a[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        sub_tile_b[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            p_value += sub_tile_a[ty][k] * sub_tile_b[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = p_value;
}

template <int TILE_WIDTH>
__global__ void matmul_tiled_bad(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float sub_tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_tile_b[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float p_value = 0.0f;

    for (int m = 0; m < N / TILE_WIDTH; m++) {
        sub_tile_a[tx][ty] = A[(by * TILE_WIDTH + tx) * N + m * TILE_WIDTH + ty];
        sub_tile_b[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            p_value += sub_tile_a[ty][k] * sub_tile_b[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = p_value;
}

template <int TILE_WIDTH>
void run_coalesced(int N) {
    const size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
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
    srand(42);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

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
        TILE_WIDTH,
        TILE_WIDTH
    );
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );
    matmul_tiled<TILE_WIDTH><<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int trial = 0; trial < 5; ++trial) {
        matmul_tiled<TILE_WIDTH><<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            N
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );
    float ms_per_trial = ms / 5.0f;
    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms_per_trial * 1.0e6);
    printf(
        "The coalesced kernel time is %.3f ms\n",
        ms_per_trial
    );
    printf(
        "The achieved GFLOPS is %.2f\n",
        gflops
    );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}

template <int TILE_WIDTH>
void run_uncoalesced(int N) {
    const size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
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
    srand(42);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

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
        TILE_WIDTH,
        TILE_WIDTH
    );
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );
    matmul_tiled_bad<TILE_WIDTH><<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int trial = 0; trial < 5; ++trial) {
        matmul_tiled_bad<TILE_WIDTH><<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            N
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );
    float ms_per_trial = ms / 5.0f;
    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms_per_trial * 1.0e6);
    printf(
        "The uncoalesced kernel time is %.3f ms\n",
        ms_per_trial
    );
    printf(
        "The achieved GFLOPS is %.2f\n",
        gflops
    );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}

int main() {
    const int N = 2048;
    printf("Coalescing comparison\n");
    run_coalesced<16>(N);
    run_uncoalesced<16>(N);

    return 0;
}