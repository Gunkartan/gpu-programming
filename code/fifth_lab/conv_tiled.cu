// =====================================================================
// Week 6 Lab - Exercise 2: constant memory and shared-memory tiling
//
//   nvcc -O3 -arch=sm_86 conv_tiled.cu -o conv_tiled
//
//   ./conv_tiled          V1 / V2 / V3 + tile sweep    -> CSV on stdout
//   ./conv_tiled break1   BREAK B: grid sized by IN_TILE_DIM, 128x128
//   ./conv_tiled break2   BREAK C: __syncthreads() removed, 3 runs
//
// r is fixed at 2 (a 5x5 mask) throughout this exercise.
//
// There are 2 TODOs in this file - search for "TODO". Each is a line or two,
// and each is one idea from the Week 6 code lecture. The program checks your
// TODOs before it measures anything and tells you which one is wrong.
// =====================================================================
#include "common.cuh"
#include <cstring>

#define R 2                      // filter radius
__constant__ float F_c[2 * R + 1][2 * R + 1];

// ---------------- V1: basic, mask in GLOBAL memory --------------------
__global__ void conv_v1(const float* N, const float* F, float* P,
                        int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;
    const int m = 2 * R + 1;
    float v = 0.0f;
    for (int fy = 0; fy < m; ++fy)
        for (int fx = 0; fx < m; ++fx) {
            int iy = outRow - R + fy, ix = outCol - R + fx;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                v += F[fy * m + fx] * N[iy * width + ix];
        }
    P[outRow * width + outCol] = v;
}

// ---------------- V2: basic, mask in CONSTANT memory ------------------
__global__ void conv_v2(const float* N, float* P, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;
    float v = 0.0f;
    for (int fy = 0; fy < 2 * R + 1; ++fy)
        for (int fx = 0; fx < 2 * R + 1; ++fx) {
            int iy = outRow - R + fy, ix = outCol - R + fx;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                v += F_c[fy][fx] * N[iy * width + ix];
        }
    P[outRow * width + outCol] = v;
}

// ---------------- V3: tiled, shared input tile + halo -----------------
// SYNC=false is BREAK C: the barrier after the cooperative load is gone.
template <int IN_T, bool SYNC>
__global__ void conv_v3(const float* N, float* P, int width, int height)
{
    constexpr int OUT_T = IN_T - 2 * R;

    int col = blockIdx.x * OUT_T + threadIdx.x - R;
    int row = blockIdx.y * OUT_T + threadIdx.y - R;

    // Phase 1: every thread loads ONE element of the input tile.
    __shared__ float N_s[IN_T][IN_T];
    if (row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;          // zero padding

    // ---- TODO 4 ----------------------------------------------------------
    // Each thread has loaded ONE element of N_s, but Phase 2 reads 25 of them
    // - most loaded by OTHER threads. Make the whole block wait here until
    // the tile is complete, but only when SYNC is true: BREAK C builds this
    // kernel with SYNC = false on purpose.   [Week 6 code lecture: 'Every
    // thread loads one tile value or zero']
    if (SYNC) {
        __syncthreads();
    }
    // --------------------------------------------------------------------

    // Phase 2: interior threads compute one output each, reading 25
    // elements of N_s - most of them loaded by OTHER threads.
    int tileRow = (int)threadIdx.y - R;
    int tileCol = (int)threadIdx.x - R;
    if (tileRow >= 0 && tileRow < OUT_T && tileCol >= 0 && tileCol < OUT_T &&
        row >= 0 && row < height && col >= 0 && col < width) {
        float v = 0.0f;
        for (int fy = 0; fy < 2 * R + 1; ++fy)
            for (int fx = 0; fx < 2 * R + 1; ++fx)
                v += F_c[fy][fx] * N_s[tileRow + fy][tileCol + fx];
        P[row * width + col] = v;
    }
}

// launch helper: correct grid is sized by the OUTPUT tile
template <int IN_T, bool SYNC>
static void launch_v3(const float* d_N, float* d_P, int W, int H,
                      bool broken_grid = false)
{
    constexpr int OUT_T = IN_T - 2 * R;
    int step = broken_grid ? IN_T : OUT_T;          // BREAK B uses IN_T
    dim3 block(IN_T, IN_T);
    dim3 grid((W + step - 1) / step, (H + step - 1) / step);
    conv_v3<IN_T, SYNC><<<grid, block>>>(d_N, d_P, W, H);
}

// Copy the 5x5 mask from host memory into the __constant__ array F_c.
static void upload_mask(const float* h_F)
{
    // ---- TODO 3 ----------------------------------------------------------
    // F_c is a __constant__ array on the GPU, so an ordinary cudaMemcpy to a
    // pointer cannot reach it. Copy all (2R+1)*(2R+1) floats of h_F into F_c,
    // and wrap the call in CHECK( ... ).   [Week 6 code lecture: 'Host code
    // copies the mask into constant memory']
    CHECK(cudaMemcpyToSymbol(
        F_c,
        h_F,
        (2 * R + 1) * (2 * R + 1) * sizeof(float)
    ));
    // --------------------------------------------------------------------
}

// ---------------------------------------------------------------------
// Built-in check, run before anything is measured. V2 tests the mask
// upload; V3 tests the tile load and the barrier.
// ---------------------------------------------------------------------
template <int IN_T>
static long check_v3(const float* d_N, float* d_P, float* h_out,
                     const float* h_ref, int W, int H)
{
    size_t bytes = (size_t)W * H * sizeof(float);
    CHECK(cudaMemset(d_P, 0, bytes));
    launch_v3<IN_T, true>(d_N, d_P, W, H);
    check_kernel("conv_v3 (verify)");
    CHECK(cudaMemcpy(h_out, d_P, bytes, cudaMemcpyDeviceToHost));
    return count_diffs(h_out, h_ref, W * H, 1e-3f, nullptr);
}

static void verify_or_stop()
{
    const int W = 100, H = 100, m = 2 * R + 1;
    const int n = W * H;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_N   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float h_F[25];
    fill_image(h_N, n);
    fill_mask(h_F, m);
    cpu_conv2D(h_N, h_F, h_ref, R, W, H, 0);
    upload_mask(h_F);

    float *d_N, *d_P;
    CHECK(cudaMalloc(&d_N, bytes));
    CHECK(cudaMalloc(&d_P, bytes));
    CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));

    // V2 reads only the constant mask, so it isolates TODO 3
    CHECK(cudaMemset(d_P, 0, bytes));
    dim3 b16(16, 16), g16((W + 15) / 16, (H + 15) / 16);
    conv_v2<<<g16, b16>>>(d_N, d_P, W, H);
    check_kernel("conv_v2 (verify)");
    CHECK(cudaMemcpy(h_out, d_P, bytes, cudaMemcpyDeviceToHost));
    long bad = count_diffs(h_out, h_ref, n, 1e-3f, nullptr);
    if (bad > 0) {
        fprintf(stderr, "  V2: %ld of %d outputs wrong.\n", bad, n);
        require(false, "check TODO 3 - the constant-memory mask is not "
                       "being filled (conv_tiled.cu, upload_mask)");
    }

    // V3 at every tile size; V2 already passed, so the mask is fine
    long b8  = check_v3<8>(d_N, d_P, h_out, h_ref, W, H);
    long b16v = check_v3<16>(d_N, d_P, h_out, h_ref, W, H);
    long b32 = check_v3<32>(d_N, d_P, h_out, h_ref, W, H);
    if (b8 + b16v + b32 > 0) {
        fprintf(stderr, "  V3 wrong outputs: IN=8 %ld, IN=16 %ld, IN=32 %ld\n",
                b8, b16v, b32);
        require(false, "check TODO 4 - threads are reading the shared tile "
                       "before the rest of the block has finished loading it");
    }
    fprintf(stderr, "# verify: V2 and V3 (all tile sizes) match the CPU "
                    "reference\n");

    CHECK(cudaFree(d_N)); CHECK(cudaFree(d_P));
    free(h_N); free(h_ref); free(h_out);
}

// =====================================================================
static void run_sweep()
{
    const int DIM = 4096, TRIALS = 10, m = 2 * R + 1;
    const int n = DIM * DIM;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_N = (float*)malloc(bytes);
    float h_F[25];
    fill_image(h_N, n);
    fill_mask(h_F, m);

    float *d_N, *d_P, *d_F;
    CHECK(cudaMalloc(&d_N, bytes));
    CHECK(cudaMalloc(&d_P, bytes));
    CHECK(cudaMalloc(&d_F, 25 * sizeof(float)));
    CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_F, h_F, 25 * sizeof(float), cudaMemcpyHostToDevice));
    upload_mask(h_F);

    double ops = (double)n * (2.0 * m * m - 1.0);
    dim3 b16(16, 16), g16((DIM + 15) / 16, (DIM + 15) / 16);
    float ms;

    printf("variant,in_tile,ms_per_trial,gflop_s\n");

    TIME_KERNEL(ms, TRIALS, (conv_v1<<<g16, b16>>>(d_N, d_F, d_P, DIM, DIM)));
    check_kernel("v1");
    printf("V1_global_mask,-,%.4f,%.1f\n", ms, ops / (ms * 1e-3) / 1e9);

    TIME_KERNEL(ms, TRIALS, (conv_v2<<<g16, b16>>>(d_N, d_P, DIM, DIM)));
    check_kernel("v2");
    printf("V2_const_mask,-,%.4f,%.1f\n", ms, ops / (ms * 1e-3) / 1e9);

    TIME_KERNEL(ms, TRIALS, (launch_v3<8,  true>(d_N, d_P, DIM, DIM)));
    check_kernel("v3_8");
    printf("V3_tiled,8,%.4f,%.1f\n", ms, ops / (ms * 1e-3) / 1e9);

    TIME_KERNEL(ms, TRIALS, (launch_v3<16, true>(d_N, d_P, DIM, DIM)));
    check_kernel("v3_16");
    printf("V3_tiled,16,%.4f,%.1f\n", ms, ops / (ms * 1e-3) / 1e9);

    TIME_KERNEL(ms, TRIALS, (launch_v3<32, true>(d_N, d_P, DIM, DIM)));
    check_kernel("v3_32");
    printf("V3_tiled,32,%.4f,%.1f\n", ms, ops / (ms * 1e-3) / 1e9);
    fflush(stdout);

    float first;
    CHECK(cudaMemcpy(&first, d_P, sizeof(float), cudaMemcpyDeviceToHost));
    fprintf(stderr, "# sanity: P[0] = %f\n", first);

    CHECK(cudaFree(d_N)); CHECK(cudaFree(d_P)); CHECK(cudaFree(d_F));
    free(h_N);
}

// ---------------------------------------------------------------------
struct Small {
    int DIM, n; size_t bytes;
    float *h_N, *h_ref, *h_a, *h_b;
    float *d_N, *d_P; float h_F[25];
};

static void small_setup(Small& s, int DIM)
{
    s.DIM = DIM; s.n = DIM * DIM; s.bytes = (size_t)s.n * sizeof(float);
    s.h_N   = (float*)malloc(s.bytes);
    s.h_ref = (float*)malloc(s.bytes);
    s.h_a   = (float*)malloc(s.bytes);
    s.h_b   = (float*)malloc(s.bytes);
    fill_image(s.h_N, s.n);
    fill_mask(s.h_F, 2 * R + 1);
    CHECK(cudaMalloc(&s.d_N, s.bytes));
    CHECK(cudaMalloc(&s.d_P, s.bytes));
    CHECK(cudaMemcpy(s.d_N, s.h_N, s.bytes, cudaMemcpyHostToDevice));
    upload_mask(s.h_F);
    cpu_conv2D(s.h_N, s.h_F, s.h_ref, R, DIM, DIM, 0);
}

// BREAK B: grid sized by IN_TILE_DIM instead of OUT_TILE_DIM
static void run_break1()
{
    Small s; small_setup(s, 128);
    const int IN_T = 32, OUT_T = IN_T - 2 * R;

    // sentinel, so untouched outputs are visible rather than stale
    CHECK(cudaMemset(s.d_P, 0, s.bytes));
    launch_v3<IN_T, true>(s.d_N, s.d_P, s.DIM, s.DIM, /*broken_grid=*/true);
    check_kernel("v3 broken grid");
    CHECK(cudaMemcpy(s.h_a, s.d_P, s.bytes, cudaMemcpyDeviceToHost));

    long bad = count_diffs(s.h_a, s.h_ref, s.n, 1e-3f, nullptr);

    printf("BREAK B - grid sized by IN_TILE_DIM (%d) instead of "
           "OUT_TILE_DIM (%d)\n", IN_T, OUT_T);
    printf("     image %dx%d\n", s.DIM, s.DIM);
    printf("     correct grid would be ceil(%d/%d) = %d blocks per axis\n",
           s.DIM, OUT_T, (s.DIM + OUT_T - 1) / OUT_T);
    printf("     broken  grid is    ceil(%d/%d) = %d blocks per axis\n",
           s.DIM, IN_T, (s.DIM + IN_T - 1) / IN_T);
    printf("E2_BREAKB_DIFFS = %ld\n\n", bad);
    print_diff_map(s.h_a, s.h_ref, s.DIM, s.DIM, 1e-3f, 4);
    printf("\n     Record E2_BREAKB_DIFFS, paste the map, and explain the\n"
           "     SHAPE of the damaged region.\n");
}

// BREAK C: the barrier after the cooperative load is removed
static void run_break2()
{
    Small s; small_setup(s, 128);
    printf("BREAK C - __syncthreads() removed after the tile load "
           "(IN_TILE_DIM = 32)\n");
    printf("     image %dx%d, three consecutive runs\n\n", s.DIM, s.DIM);

    long bad[3]; float mx[3];
    for (int run = 0; run < 3; ++run) {
        CHECK(cudaMemset(s.d_P, 0, s.bytes));
        launch_v3<32, false>(s.d_N, s.d_P, s.DIM, s.DIM);
        check_kernel("v3 no-sync");
        float* dst = (run == 0) ? s.h_a : s.h_b;
        CHECK(cudaMemcpy(dst, s.d_P, s.bytes, cudaMemcpyDeviceToHost));
        bad[run] = count_diffs(dst, s.h_ref, s.n, 1e-3f, &mx[run]);
        printf("     run %d: wrong elements = %7ld   max error = %.3e\n",
               run + 1, bad[run], mx[run]);
        if (run > 0) {
            long d = count_diffs(s.h_a, s.h_b, s.n, 1e-3f, nullptr);
            printf("             differs from run 1 in %ld elements\n", d);
        }
    }
    int same = (bad[0] == bad[1] && bad[1] == bad[2]);
    printf("\nE2_BREAKC_WRONG = %ld        (from run 1)\n", bad[0]);
    printf("E2_BREAKC_RUNS_AGREE = %s\n", same ? "yes" : "no");
    printf("\n     If the three runs disagree you have observed a genuine race.\n"
           "     If they happen to agree, say so honestly and explain why an\n"
           "     unsynchronized read can still LOOK stable on some hardware.\n");
}

int main(int argc, char** argv)
{
    print_gpu_banner();
    const char* mode = (argc > 1) ? argv[1] : "";
    if (mode[0] != 0 && strcmp(mode, "break1") != 0 &&
        strcmp(mode, "break2") != 0) {
        fprintf(stderr, "usage: %s [break1|break2]\n", argv[0]);
        return 2;
    }
    verify_or_stop();
    if      (strcmp(mode, "break1") == 0) run_break1();
    else if (strcmp(mode, "break2") == 0) run_break2();
    else                                  run_sweep();
    return 0;
}
