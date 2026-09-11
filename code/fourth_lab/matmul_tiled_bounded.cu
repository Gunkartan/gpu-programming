#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
    int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < num_tiles; m++) {
        if (row < N && m * TILE_WIDTH + tx < N) {
            sub_tile_a[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        } else {
            sub_tile_a[ty][tx] = 0.0f;
        }

        if (m * TILE_WIDTH + ty < N && col < N) {
            sub_tile_b[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        } else {
            sub_tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            p_value += sub_tile_a[ty][k] * sub_tile_b[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = p_value;
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

    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float denom = fabsf(cpu[i]) + 1e-6f;
        float rel = diff / denom;

        if (rel > max_rel_err) {
            max_rel_err = rel;
        }
    }

    printf(
        "The maximum relative error is %e\n",
        max_rel_err
    );

    return max_rel_err < tol;
}

template <int TILE_WIDTH>
void run_experiment(
    int N,
    int do_verify
) {
    const size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
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

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX * 2 - 1;
        h_B[i] = (float)rand() / RAND_MAX * 2 - 1;
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
        "The tile width is %d\n",
        TILE_WIDTH
    );
    printf(
        "N is %d\n",
        N
    );
    printf(
        "The block size is %d by %d\n",
        block.x,
        block.y
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

    if (do_verify) {
        matmul_cpu(
            h_A,
            h_B,
            h_C_cpu,
            N
        );
        cudaMemcpy(
            h_C,
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        );
        int passed = verify(
            h_C_cpu,
            h_C,
            N,
            1e-3f
        );
        printf(
            "The verification result is a %s\n",
            passed ? "pass" : "fail"
        );
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
}

int main() {
    printf("Boundary handling experiment\n");
    run_experiment<16>(
        1000,
        1
    );
    run_experiment<16>(
        1023,
        1
    );
    run_experiment<16>(
        1025,
        1
    );
    printf("Timing comparison\n");
    run_experiment<16>(
        1000,
        0
    );
    run_experiment<16>(
        1024,
        0
    );

    return 0;
}