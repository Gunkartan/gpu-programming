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

__device__ float heavy_compute(int x) {
    float f = (float)x;

    for (int k = 0; k < 1000; ++k) {
        f = f * 1.0001f + 0.001f;
    }

    return f;
}

__device__ float light_compute(int x) {
    float f = (float)x;

    for (int k = 0; k < 100; ++k) {
        f = f * 1.0001f + 0.001f;
    }

    return f;
}

__global__ void variant_a(
    const int* input,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (input[i] < 16) {
            output[i] = heavy_compute(input[i]);
        } else {
            output[i] = light_compute(input[i]);
        }
    }
}

__global__ void variant_b(
    const int* input,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (input[i] < 16) {
            output[i] = heavy_compute(input[i]);
        } else {
            output[i] = light_compute(input[i]);
        }
    }
}

__global__ void variant_c(
    const int* input,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (input[i] < 16) {
            output[i] = heavy_compute(input[i]);
        } else {
            output[i] = light_compute(input[i]);
        }
    }
}

bool check_results(
    const float* result,
    const int* input,
    int N
) {
    float expected_x = 0.0f;
    float expected_y = 0.0f;

    for (int i = 0; i < N; ++i) {
        expected_x = (float)input[i];

        if (input[i] < 16) {
            expected_y = expected_x;

            for (int k = 0; k < 1000; ++k) {
                expected_y = expected_y * 1.0001f + 0.001f;
            }
        } else {
            expected_y = expected_x;

            for (int k = 0; k < 100; ++k) {
                expected_y = expected_y * 1.0001f + 0.001f;
            }
        }

        float difference = fabsf(result[i] - expected_y);
        float tolerance = 1e-4f * fabsf(expected_y) + 1e-5f;

        if (difference > tolerance) {
            printf(
                "Verification failed at index %d, expected %.6f, and got %.6f\n",
                i,
                expected_y,
                result[i]
            );

            return false;
        }
    }

    return true;
}

void initialize_variant_a(
    int* input,
    int N
) {
    for (int i = 0; i < N; ++i) {
        input[i] = i % 32;
    }
}

void initialize_variant_b(
    int* input,
    int N
) {
    for (int i = 0; i < N; ++i) {
        input[i] = (i / 32) % 32;
    }
}

void initialize_variant_c(
    int* input,
    int N
) {
    srand(42);

    for (int i = 0; i < N; ++i) {
        input[i] = rand() % 32;
    }
}

int main() {
    const int N = 1 << 24;
    const int BLOCK_SIZE = 256;
    const int NUM_TRIALS = 10;
    printf("Warp divergence experiment\n");
    printf(
        "The value of N is %d\n",
        N
    );
    printf(
        "The block size is %d\n",
        BLOCK_SIZE
    );
    float* h_output = new float[N];
    int* h_input_a = new int[N];
    int* h_input_b = new int[N];
    int* h_input_c = new int[N];
    int* d_input = nullptr;
    float* d_output = nullptr;
    size_t input_bytes = (size_t)N * sizeof(int);
    size_t output_bytes = (size_t)N * sizeof(float);
    cudaMalloc(
        &d_input,
        input_bytes
    );
    check_cuda("Allocation 1");
    cudaMalloc(
        &d_output,
        output_bytes
    );
    check_cuda("Allocation 2");
    initialize_variant_a(
        h_input_a,
        N
    );
    initialize_variant_b(
        h_input_b,
        N
    );
    initialize_variant_c(
        h_input_c,
        N
    );
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemcpy(
        d_input,
        h_input_a,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 1");
    variant_a<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 1");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 1");
    cudaMemcpy(
        h_output,
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    );
    check_cuda("Copy 2");
    bool correct_a = check_results(
        h_output,
        h_input_a,
        N
    );
    cudaMemcpy(
        d_input,
        h_input_a,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 3");
    variant_b<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 2");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 2");
    cudaMemcpy(
        h_output,
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    );
    check_cuda("Copy 4");
    bool correct_b = check_results(
        h_output,
        h_input_a,
        N
    );
    cudaMemcpy(
        d_input,
        h_input_a,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 5");
    variant_c<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 3");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 3");
    cudaMemcpy(
        h_output,
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    );
    check_cuda("Copy 6");
    bool correct_c = check_results(
        h_output,
        h_input_a,
        N
    );
    printf(
        "A, %s\n",
        correct_a ? "passed" : "failed"
    );
    printf(
        "B, %s\n",
        correct_b ? "passed" : "failed"
    );
    printf(
        "C, %s\n",
        correct_c ? "passed" : "failed"
    );

    if (!correct_a || !correct_b || !correct_c) {
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_output;
        delete[] h_input_a;
        delete[] h_input_b;
        delete[] h_input_c;

        return 1;
    }

    cudaMemcpy(
        d_input,
        h_input_a,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 7");
    variant_a<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 4");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 4");
    cudaEvent_t start_a;
    cudaEvent_t stop_a;
    cudaEventCreate(&start_a);
    check_cuda("Event 1");
    cudaEventCreate(&stop_a);
    check_cuda("Event 2");
    cudaEventRecord(start_a);
    check_cuda("Recording 1");

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        variant_a<<<grid, BLOCK_SIZE>>>(
            d_input,
            d_output,
            N
        );
    }

    check_cuda("Launching 5");
    cudaEventRecord(stop_a);
    check_cuda("Recording 2");
    cudaEventSynchronize(stop_a);
    check_cuda("Stop 1");
    float total_a;
    cudaEventElapsedTime(
        &total_a,
        start_a,
        stop_a
    );
    check_cuda("Duration 1");
    float time_a = total_a / NUM_TRIALS;
    cudaEventDestroy(start_a);
    check_cuda("Destroy 1");
    cudaEventDestroy(stop_a);
    check_cuda("Destroy 2");
    cudaMemcpy(
        d_input,
        h_input_b,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 8");
    variant_b<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 6");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 5");
    cudaEvent_t start_b;
    cudaEvent_t stop_b;
    cudaEventCreate(&start_b);
    check_cuda("Event 3");
    cudaEventCreate(&stop_b);
    check_cuda("Event 4");
    cudaEventRecord(start_b);
    check_cuda("Recording 3");

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        variant_b<<<grid, BLOCK_SIZE>>>(
            d_input,
            d_output,
            N
        );
    }

    check_cuda("Launching 7");
    cudaEventRecord(stop_b);
    check_cuda("Recording 4");
    cudaEventSynchronize(stop_b);
    check_cuda("Stop 2");
    float total_b;
    cudaEventElapsedTime(
        &total_b,
        start_b,
        stop_b
    );
    check_cuda("Duration 2");
    float time_b = total_b / NUM_TRIALS;
    cudaEventDestroy(start_b);
    check_cuda("Destroy 3");
    cudaEventDestroy(stop_b);
    check_cuda("Destroy 4");
    cudaMemcpy(
        d_input,
        h_input_c,
        input_bytes,
        cudaMemcpyHostToDevice
    );
    check_cuda("Copy 9");
    variant_c<<<grid, BLOCK_SIZE>>>(
        d_input,
        d_output,
        N
    );
    check_cuda("Launching 8");
    cudaDeviceSynchronize();
    check_cuda("Synchronization 6");
    cudaEvent_t start_c;
    cudaEvent_t stop_c;
    cudaEventCreate(&start_c);
    check_cuda("Event 5");
    cudaEventCreate(&stop_c);
    check_cuda("Event 6");
    cudaEventRecord(start_c);
    check_cuda("Recording 5");

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        variant_c<<<grid, BLOCK_SIZE>>>(
            d_input,
            d_output,
            N
        );
    }

    check_cuda("Launching 9");
    cudaEventRecord(stop_c);
    check_cuda("Recording 6");
    cudaEventSynchronize(stop_c);
    check_cuda("Stop 3");
    float total_c;
    cudaEventElapsedTime(
        &total_c,
        start_c,
        stop_c
    );
    check_cuda("Duration 3");
    float time_c = total_c / NUM_TRIALS;
    cudaEventDestroy(start_c);
    check_cuda("Destroy 5");
    cudaEventDestroy(stop_c);
    check_cuda("Destroy 6");
    double throughput_a = (double)N / (time_a * 1e-3);
    double throughput_b = (double)N / (time_b * 1e-3);
    double throughput_c = (double)N / (time_c * 1e-3);
    double speedup_a = 1.0;
    double speedup_b = time_a / time_b;
    double speedup_c = time_a / time_c;
    printf(
        "%-12s %-15s %-20s %-15s\n",
        "variant",
        "time_ms",
        "elements_per_sec",
        "speedup"
    );
    printf(
        "%-12s %-15.4f %-20.4e %-15.4f\n",
        "A",
        time_a,
        throughput_a,
        speedup_a
    );
    printf(
        "%-12s %-15.4f %-20.4e %-15.4f\n",
        "B",
        time_b,
        throughput_b,
        speedup_b
    );
    printf(
        "%-12s %-15.4f %-20.4e %-15.4f\n",
        "C",
        time_c,
        throughput_c,
        speedup_c
    );
    FILE* csv = fopen(
        "divergence.csv",
        "w"
    );

    if (csv == nullptr) {
        fprintf(
            stderr,
            "Failed to create the CSV file\n"
        );
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_output;
        delete[] h_input_a;
        delete[] h_input_b;
        delete[] h_input_c;

        return 1;
    }

    fprintf(
        csv,
        "variant,kernel_time_ms,elements_per_second,speedup_vs_a\n"
    );
    fprintf(
        csv,
        "A,%.4f,%.4e,%.4f\n",
        time_a,
        throughput_a,
        speedup_a
    );
    fprintf(
        csv,
        "B,%.4f,%.4e,%.4f\n",
        time_b,
        throughput_b,
        speedup_b
    );
    fprintf(
        csv,
        "C,%.4f,%.4e,%.4f\n",
        time_c,
        throughput_c,
        speedup_c
    );
    fclose(csv);
    cudaFree(d_input);
    check_cuda("Free 1");
    cudaFree(d_output);
    check_cuda("Free 2");
    delete[] h_output;
    delete[] h_input_a;
    delete[] h_input_b;
    delete[] h_input_c;

    return 0;
}