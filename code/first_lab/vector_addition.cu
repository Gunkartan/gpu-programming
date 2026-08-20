#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

inline void check_cuda(
    cudaError_t err,
    const char* action,
    const char* file,
    int line
) {
    if (err != cudaSuccess) {
        std::cerr << "Error during " << action << ", with the content being " << cudaGetErrorString(err) << ", occurring at " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add_vectors(
    const float* a,
    const float* b,
    const float* c,
    float* d,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d[idx] = a[idx] + b[idx] + c[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    float *h_d = new float[n];
    float *h_expected = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 3.0f;
        h_expected[i] = h_a[i] + h_b[i] + h_c[i];
    }

    float *d_a, *d_b, *d_c, *d_d;
    check_cuda(
        cudaMalloc(
            &d_a,
            bytes
        ),
        "Memory allocation for d_a",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMalloc(
            &d_b,
            bytes
        ),
        "Memory allocation for d_b",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMalloc(
            &d_c,
            bytes
        ),
        "Memory allocation for d_c",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMalloc(
            &d_d,
            bytes
        ),
        "Memory allocation for d_d",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMemcpy(
            d_a,
            h_a,
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Memory copy h_a to d_a",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMemcpy(
            d_b,
            h_b,
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Memory copy h_b to d_b",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMemcpy(
            d_c,
            h_c,
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Memory copy h_c to d_c",
        __FILE__,
        __LINE__
    );
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_vectors<<<blocks_per_grid, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        d_d,
        n
    );
    check_cuda(
        cudaGetLastError(),
        "Vector-adding kernel launch",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "Vector-adding kernel execution",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaMemcpy(
            h_d,
            d_d,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Memory copy d_d to h_d",
        __FILE__,
        __LINE__
    );
    int mismatches = 0;
    float tolerance = 1e-5f;
    int max_printed_mismatches = 10;

    for (int i = 0; i < n; i++) {
        if (std::abs(h_d[i] - h_expected[i]) > tolerance) {
            if (mismatches < max_printed_mismatches) {
                std::cout << "Mismatch at index " << i << ", with " << h_d[i] << ", expecting " << h_expected[i] << std::endl;
            }

            mismatches++;
        }
    }

    std::cout << "There are " << mismatches << " mismatches" << std::endl;

    if (mismatches == 0) {
        std::cout << "Pass with all " << n << " elements matched" << std::endl;
    } else {
        std::cout << "Fail with " << mismatches << " out of " << n << " mismatched" << std::endl;
    }

    check_cuda(
        cudaFree(d_a),
        "Free d_a",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaFree(d_b),
        "Free d_b",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaFree(d_c),
        "Free d_c",
        __FILE__,
        __LINE__
    );
    check_cuda(
        cudaFree(d_d),
        "Free d_d",
        __FILE__,
        __LINE__
    );
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d;
    delete[] h_expected;

    return 0;
}