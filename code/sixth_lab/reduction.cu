// =====================================================================
// Week 7 Lab - Parallel Reduction
//
//   nvcc -O3 -arch=sm_86 reduction.cu -o reduction     (sm_89 / sm_75)
//   ./reduction
//
// ONE program, one CONTROL PANEL. Every experiment in the lab is:
//   change one setting below  ->  rebuild  ->  run  ->  copy the RESULT line.
//
// The input is 2^N_LOG2 copies of 1.0f, so the correct answer is just the
// number of elements. That matters: if a run prints a smaller sum, the
// shortfall tells you EXACTLY how many values fell out of the reduction.
//
// There are 4 TODOs in this file - search for "TODO". Fill them in BEFORE you
// touch the control panel: with the default settings the answer must come out
// exactly right, and the program checks that for you and stops if it does
// not.
// =====================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// =====================================================================
//                          CONTROL PANEL
//        The lab instructions tell you which line to change.
// =====================================================================
#define N_LOG2         26     // input size = 2^26 = 67,108,864 floats
#define BLOCK_SIZE    256     // threads per block
#define COARSE          8     // elements each thread sums before the tree
#define INDEXING        2     // 1 = interleaved (divergent), 2 = sequential
#define USE_SYNC        1     // 1 = keep the barrier, 0 = remove it
#define GRID_PERCENT  100     // how much of the input the grid covers

#define PEAK_BW_GBS   716.86   // <<< YOUR GPU's peak bandwidth, from your
                              //     Week 4 device_query output
// =====================================================================

#define N ((unsigned)1 << N_LOG2)
#define TRIALS 10

// True only while the control panel is untouched. With the defaults the
// answer MUST come out exactly right, so the program can tell the
// difference between "a TODO is wrong" and "you broke it on purpose".
#if (BLOCK_SIZE == 256) && (COARSE == 4) && (INDEXING == 2) && \
    (USE_SYNC == 1) && (GRID_PERCENT == 100)
#define DEFAULT_SETTINGS 1
#else
#define DEFAULT_SETTINGS 0
#endif

#define CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

static void check_kernel(const char* what)
{
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "kernel %s failed -> %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static void require(bool ok, const char* msg)
{
    if (!ok) { fprintf(stderr, "\n  STOP: %s\n\n", msg); exit(1); }
}

// ---------------------------------------------------------------------
// One block reduces BLOCK_SIZE * COARSE elements to a single float.
// ---------------------------------------------------------------------
__global__ void reduce_block(const float* in, float* out, unsigned n)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned tid  = threadIdx.x;
    unsigned base = blockIdx.x * (BLOCK_SIZE * COARSE) + tid;

    // -- Phase 1: each thread adds up COARSE elements of its own ------
    float sum = 0.0f;
    // ---- TODO 1 ----------------------------------------------------------
    // Each thread adds up COARSE elements of its own before the tree starts.
    // Thread tid takes element 'base', then base + BLOCK_SIZE, then base +
    // 2*BLOCK_SIZE, and so on - that stride keeps neighbouring threads
    // reading neighbouring addresses. Skip any index that is past n.
    for (unsigned i = 0; i < COARSE; ++i) {
        unsigned idx = base + i * BLOCK_SIZE;

        if (idx < n) {
            sum += in[idx];
        }
    }
    // --------------------------------------------------------------------
    sdata[tid] = sum;
    __syncthreads();

    // -- Phase 2: the block reduces its BLOCK_SIZE partials to one ----
#if INDEXING == 1
    // Interleaved addressing: active threads spread across the warp.
    // (given to you - this is the version Exercise 3 compares against)
    for (unsigned s = 1; s < BLOCK_SIZE; s *= 2) {
        if (tid % (2 * s) == 0 && tid + s < BLOCK_SIZE)
            sdata[tid] += sdata[tid + s];
        if (USE_SYNC) __syncthreads();
    }
#else
    // Sequential addressing: the active threads are the low ones.
    for (unsigned s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        // ---- TODO 2 ------------------------------------------------------
        // One step of the tree: the lower half of the still-active threads
        // absorbs the upper half. s is the current half-width, and it halves
        // on every pass.
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // ----------------------------------------------------------------
        if (USE_SYNC) __syncthreads();
    }
#endif

    // -- Phase 3: one value per block ---------------------------------
    // ---- TODO 4 ----------------------------------------------------------
    // sdata[0] now holds this block's total. Exactly one thread should write
    // it into this block's slot of out[].
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
    // --------------------------------------------------------------------
}

// How many blocks are needed so that every one of the n elements is read?
// Each block consumes BLOCK_SIZE * COARSE elements.
static unsigned grid_blocks(unsigned n)
{
    // ---- TODO 3 ----------------------------------------------------------
    // Each block consumes BLOCK_SIZE * COARSE elements, so work out how many
    // blocks it takes to cover n of them. Round UP - otherwise the last few
    // elements get no block at all, and the sum comes out short.
    return (n + BLOCK_SIZE * COARSE - 1) / (BLOCK_SIZE * COARSE);
    // --------------------------------------------------------------------
}

// ---------------------------------------------------------------------
int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# GPU: %s (CC %d.%d), %d SMs\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const unsigned n = N;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    require(h_in != NULL, "out of host memory - lower N_LOG2");
    for (unsigned i = 0; i < n; ++i) h_in[i] = 1.0f;

    // ---- how do you add n ones? three answers --------------------
    // cpu_float is volatile on purpose: it forces a genuine one-at-a-time
    // float accumulation. Without it an optimising host compiler would be
    // free to vectorise the loop into several partial sums, which changes
    // the very result Exercise 1 is about.
    volatile float cpu_float = 0.0f;
    double cpu_double = 0.0;
    for (unsigned i = 0; i < n; ++i) { cpu_float += h_in[i]; cpu_double += h_in[i]; }

    float *d_in, *d_partial;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    unsigned full = grid_blocks(n);
    require(full > 0, "check TODO 3 - grid_blocks() returned 0 blocks");
    unsigned blocks = (unsigned)((unsigned long long)full * GRID_PERCENT / 100);
    if (blocks < 1) blocks = 1;

    CHECK(cudaMalloc(&d_partial, (size_t)full * sizeof(float)));
    float* h_partial = (float*)malloc((size_t)full * sizeof(float));

    // ---- warm up, then time TRIALS launches ------------------------
    CHECK(cudaMemset(d_partial, 0, (size_t)full * sizeof(float)));
    reduce_block<<<blocks, BLOCK_SIZE>>>(d_in, d_partial, n);
    check_kernel("reduce_block");

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int t = 0; t < TRIALS; ++t)
        reduce_block<<<blocks, BLOCK_SIZE>>>(d_in, d_partial, n);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms_total;
    CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
    float ms = ms_total / TRIALS;
    check_kernel("reduce_block (timed)");

    // ---- combine the per-block partials on the host ----------------
    // Done in double on purpose: this lab measures the BLOCK-level
    // reduction, so the final combine must not add error of its own.
    CHECK(cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    double gpu_sum = 0.0;
    for (unsigned b = 0; b < blocks; ++b) gpu_sum += h_partial[b];

    if (gpu_sum == 0.0)
        require(false, "every block returned 0 - check TODO 1 (the load) "
                       "and TODO 4 (writing the block result)");

    if (DEFAULT_SETTINGS && gpu_sum != (double)n) {
        fprintf(stderr, "  sum = %.0f, expected %u (short by %.0f)\n",
                gpu_sum, n, (double)n - gpu_sum);
        require(false, "the control panel is still at its DEFAULT settings, "
                       "so this run must be exactly right - check TODO 1, 2, "
                       "3 and 4 before you start changing anything");
    }

    // ---- report -----------------------------------------------------
    double covered = (double)blocks * BLOCK_SIZE * COARSE;
    if (covered > (double)n) covered = (double)n;
    double gbs = (double)bytes / (ms * 1e-3) / 1e9;

    printf("\n--- how do you add %u ones? ---\n", n);
    printf("   CPU loop, float      : %.0f\n", (double)cpu_float);
    printf("   CPU loop, double     : %.0f\n", cpu_double);
    printf("   GPU tree (this run)  : %.0f\n", gpu_sum);
    printf("   correct answer       : %u\n\n", n);

    printf("RESULT block=%d coarse=%d indexing=%d sync=%d grid=%d%% | "
           "sum=%.0f expected=%u %s | %.3f ms | %.1f GB/s | %.1f %% of peak\n",
           BLOCK_SIZE, COARSE, INDEXING, USE_SYNC, GRID_PERCENT,
           gpu_sum, n, (gpu_sum == (double)n) ? "MATCH" : "MISMATCH",
           ms, gbs, 100.0 * gbs / PEAK_BW_GBS);

    if (gpu_sum != (double)n)
        printf("         the sum is short by %.0f, i.e. %.0f of the %u "
               "values never reached it\n",
               (double)n - gpu_sum, (double)n - gpu_sum, n);
    printf("         (blocks launched = %u of %u, covering %.0f elements)\n",
           blocks, full, covered);

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_partial));
    free(h_in); free(h_partial);
    return 0;
}
