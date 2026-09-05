#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(
        &prop,
        device
    );
    printf("Device characterization\n");
    printf(
        "The device name is %s\n",
        prop.name
    );
    printf(
        "The compute capability is %d.%d\n",
        prop.major,
        prop.minor
    );
    printf("Streaming multiprocessors and thread resources\n");
    printf(
        "The number of SMs is %d\n",
        prop.multiProcessorCount
    );
    printf(
        "The maximum threads per SM is %d\n",
        prop.maxThreadsPerMultiProcessor
    );
    printf(
        "The maximum threads per block is %d\n",
        prop.maxThreadsPerBlock
    );
    printf(
        "The maximum warps per SM is %d\n",
        prop.maxThreadsPerMultiProcessor / prop.warpSize
    );
    printf(
        "The maximum blocks per SM are %d\n",
        prop.maxBlocksPerMultiProcessor
    );
    printf(
        "The warp size is %d\n",
        prop.warpSize
    );
    printf("Register resources\n");
    printf(
        "The registers per SM are %d\n",
        prop.regsPerMultiprocessor
    );
    printf(
        "The registers per block is %d\n",
        prop.regsPerBlock
    );
    printf("Shared memory resources\n");
    printf(
        "The shared memory per SM is %zu bytes\n",
        prop.sharedMemPerMultiprocessor
    );
    printf(
        "The shared memory per block is %zu bytes\n",
        prop.sharedMemPerBlock
    );
    printf("Memory\n");
    printf(
        "The L2 cache size is %d bytes\n",
        prop.l2CacheSize
    );
    printf(
        "The total global memory is %zu bytes\n",
        prop.totalGlobalMem
    );
    printf(
        "The memory clock rate is %d kHz\n",
        prop.memoryClockRate
    );
    printf(
        "The memory bus width is %d bits\n",
        prop.memoryBusWidth
    );
    double bandwidth = 2.0 * prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8.0;
    printf(
        "The peak memory bandwidth is %.2f GB per second\n",
        bandwidth / 1e9
    );

    return 0;
}