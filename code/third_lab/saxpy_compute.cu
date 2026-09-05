#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

void check_cuda(const char* message) {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf(
            "Error after %s with a message %s\n",
            message,
            cudaGetErrorString(error)
        );
        exit(EXIT_FAILURE);
    }
}

__global__ void saxpy_compute(
    float a,
    const float* x,
    float* y,
    int N,
    int NITER
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float xi = x[i];
        float yi = y[i];
        
        for (int k = 0; k < NITER; ++k) {
            yi = a * xi + yi;
            xi = xi * 1.001f + 0.001f;
        }
        
        y[i] = yi;
    }
}

int main() {
    const int N = 1 << 24;
    const int NITER = 32;
    const float a = 2.0f;
    int block_sizes[] = {
        32,
        64,
        128,
        256,
        512,
        1024
    };
    int num_block_sizes = 6;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(
        &prop,
        0
    );
    check_cuda("Get device properties");
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    printf("Block-size occupancy study with SAXPY\n");
    printf(
        "The value of N is %d\n",
        N
    );
    printf(
        "The value of NITER is %d\n",
        NITER
    );
    printf(
        "The GPU is %s\n",
        prop.name
    );
    printf(
        "The maximum warps per SM is %d\n",
        max_warps_per_sm
    );
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_result = new float[N];

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 1.0f;
    }

    float* d_x;
    float* d_y;
    cudaMalloc(
        &d_x,
        N * sizeof(float)
    );
    check_cuda("Memory allocation for d_x");
    cudaMalloc(
        &d_y,
        N * sizeof(float)
    );
    check_cuda("Memory allocation for d_y");
    cudaMemcpy(
        d_x,
        h_x,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy h_x to d_x");
    cudaMemcpy(
        d_y,
        h_y,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy h_y to d_y");
    int check_block = 256;
    int check_grid = (N + check_block - 1) / check_block;
    saxpy_compute<<<check_grid, check_block>>>(
        a,
        d_x,
        d_y,
        N,
        NITER
    );
    check_cuda("Kernel launch test");
    cudaDeviceSynchronize();
    check_cuda("Kernel synchronization test");
    cudaMemcpy(
        h_result,
        d_y,
        N * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    check_cuda("Copy d_y to h_result");
    float expected_x = 1.0f;
    float expected_y = 1.0f;

    for (int k = 0; k < NITER; ++k) {
        expected_y = a * expected_x + expected_y;
        expected_x = expected_x * 1.001f + 0.001f;
    }

    bool correct = true;

    for (int i = 0; i < N; ++i) {
        float difference = fabsf(h_result[i] - expected_y);
        float tolerance = 1e-4f * fabsf(expected_y) + 1e-5f;

        if (difference > tolerance) {
            correct = false;
            printf(
                "Verification failed at index %d, expected %.6f, and got %.6f\n",
                i,
                expected_y,
                h_result[i]
            );

            break;
        }
    }

    if (correct) {
        printf("Passed\n");
    } else {
        printf("Failed\n");
    }

    printf(
        "%-12s %-14s %-13s %-13s %-15s %-17s\n",
        "block_size",
        "active_blocks",
        "active_warps",
        "occupancy_%",
        "ms_per_trial",
        "bandwidth_GB_s"
    );

    for (int j = 0; j < num_block_sizes; ++j) {
        int block = block_sizes[j];
        int grid = (N + block - 1) / block;
        cudaMemcpy(
            d_y,
            h_y,
            N * sizeof(float),
            cudaMemcpyHostToDevice
        );
        check_cuda("Reset d_y");
        saxpy_compute<<<grid, block>>>(
            a,
            d_x,
            d_y,
            N,
            NITER
        );
        check_cuda("Warm-up kernel launch");
        cudaDeviceSynchronize();
        check_cuda("Warm-up kernel synchronization");
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        check_cuda("Create event start");
        cudaEventCreate(&stop);
        check_cuda("Create event stop");
        cudaEventRecord(start);
        check_cuda("Start event recording");

        for (int trial = 0; trial < 10; ++trial) {
            saxpy_compute<<<grid, block>>>(
                a,
                d_x,
                d_y,
                N,
                NITER
            );
        }

        check_cuda("Timed kernel launches");
        cudaEventRecord(stop);
        check_cuda("Stop event recording");
        cudaEventSynchronize(stop);
        check_cuda("Synchronization after stop event recording");
        float ms;
        cudaEventElapsedTime(
            &ms,
            start,
            stop
        );
        check_cuda("Event elapsed time");
        float ms_per_trial = ms / 10.0f;
        int active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            saxpy_compute,
            block,
            0
        );
        check_cuda("Occupancy");
        int active_warps = active_blocks * (block / prop.warpSize);
        float occupancy = static_cast<float>(active_warps) / max_warps_per_sm * 100.0f;
        size_t bytes = 2 * static_cast<size_t>(N) * sizeof(float);
        double bandwidth_gb_s = bytes / (ms_per_trial * 1e-3) / 1e9;
        printf(
            "%-12d %-14d %-13d %-13.2f %-15.4f %-17.2f\n",
            block,
            active_blocks,
            active_warps,
            occupancy,
            ms_per_trial,
            bandwidth_gb_s
        );
        cudaEventDestroy(start);
        check_cuda("Start event destroy");
        cudaEventDestroy(stop);
        check_cuda("Stop event destroy");
    }

    cudaFree(d_x);
    check_cuda("Free d_x");
    cudaFree(d_y);
    check_cuda("Free d_y");
    delete[] h_x;
    delete[] h_y;
    delete[] h_result;

    return 0;
}