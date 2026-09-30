// =====================================================================
// Week 6 Lab - Exercise 1: baseline convolution and arithmetic intensity
//
//   nvcc -O3 -arch=sm_86 conv_basic.cu -o conv_basic
//
//   ./conv_basic          sweep r = 1,2,3 on 4096x4096   -> CSV on stdout
//   ./conv_basic break    BREAK A: clamp instead of zero padding, 128x128
//
// There are 2 TODOs in this file - search for "TODO". Each is a line or two,
// and each is one idea from the Week 6 code lecture. The program checks your
// TODOs before it measures anything and tells you which one is wrong.
// =====================================================================
#include "common.cuh"

__global__ void conv2D_basic(const float* N, const float* F, float* P,
                             int r, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;

    int m = 2 * r + 1;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < m; ++fRow) {
        for (int fCol = 0; fCol < m; ++fCol) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            // ---- TODO 1 --------------------------------------------------
            // Zero padding: a filter tap that lands OUTSIDE the image must
            // add nothing. Set 'inside' to true only when (inRow, inCol) is a
            // real pixel of the width x height image.   [Week 6 code lecture:
            // 'The inner if implements zero-padding safely']
            bool inside = inRow >= 0 && inRow < height && inCol >= 0 && inCol < width;
            // ------------------------------------------------------------
            if (inside)
                Pvalue += F[fRow * m + fCol] * N[inRow * width + inCol];
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// BREAK A: identical, except the boundary rule is CLAMP instead of zero
// padding. Everything else is untouched.
__global__ void conv2D_clamp(const float* N, const float* F, float* P,
                             int r, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;

    int m = 2 * r + 1;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < m; ++fRow) {
        for (int fCol = 0; fCol < m; ++fCol) {
            int iy = outRow - r + fRow;
            int ix = outCol - r + fCol;
            iy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);   // clamp
            ix = ix < 0 ? 0 : (ix >= width  ? width  - 1 : ix);
            Pvalue += F[fRow * m + fCol] * N[iy * width + ix];
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// How many blocks does it take to cover a width x height image?
static dim3 grid_for(int width, int height, dim3 block)
{
    // ---- TODO 2 ----------------------------------------------------------
    // Return enough blocks to cover every pixel. Divide, but round UP: the
    // last block may hang off the edge of the image, and the kernel's 'if
    // (outRow >= height || ...) return;' switches those extra threads off.
    // [Week 6 code lecture: 'Host code configures and launches a 2D
    // convolution grid']
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    // --------------------------------------------------------------------
}

// ---------------------------------------------------------------------
// Built-in check. Runs before anything is measured. The image is 100x100
// on purpose: 100 is not a multiple of 16, so a grid that does not round
// up leaves pixels uncomputed and fails here.
// ---------------------------------------------------------------------
static void verify_or_stop()
{
    const int W = 100, H = 100, r = 2, m = 5;
    const int n = W * H;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_N   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_gpu = (float*)malloc(bytes);
    float h_F[25];
    fill_image(h_N, n);
    fill_mask(h_F, m);
    cpu_conv2D(h_N, h_F, h_ref, r, W, H, /*zero padding*/ 0);

    float *d_N, *d_P, *d_F;
    CHECK(cudaMalloc(&d_N, bytes));
    CHECK(cudaMalloc(&d_P, bytes));
    CHECK(cudaMalloc(&d_F, 25 * sizeof(float)));
    CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_F, h_F, 25 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_P, 0, bytes));

    dim3 block(16, 16);
    conv2D_basic<<<grid_for(W, H, block), block>>>(d_N, d_F, d_P, r, W, H);
    check_kernel("conv2D_basic (verify)");
    CHECK(cudaMemcpy(h_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

    float mx = 0.0f;
    long bad = count_diffs(h_gpu, h_ref, n, 1e-3f, &mx);
    if (bad > 0) {
        fprintf(stderr, "  %ld of %d outputs disagree with the CPU "
                        "reference.\n", bad, n);
        require(false, "check TODO 1 (zero-padding condition) and "
                       "TODO 2 (grid size) in conv_basic.cu");
    }
    fprintf(stderr, "# verify: conv2D_basic matches the CPU reference "
                    "(max error %.2e)\n", mx);

    CHECK(cudaFree(d_N)); CHECK(cudaFree(d_P)); CHECK(cudaFree(d_F));
    free(h_N); free(h_ref); free(h_gpu);
}

static void run_sweep()
{
    const int DIM = 4096, TRIALS = 10;
    const int n = DIM * DIM;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_N = (float*)malloc(bytes), *h_P = (float*)malloc(bytes);
    fill_image(h_N, n);

    float *d_N, *d_P, *d_F;
    CHECK(cudaMalloc(&d_N, bytes));
    CHECK(cudaMalloc(&d_P, bytes));
    CHECK(cudaMalloc(&d_F, 49 * sizeof(float)));
    CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = grid_for(DIM, DIM, block);

    printf("r,mask_dim,ms_per_trial,gflop_s,eff_bandwidth_GB_s\n");
    for (int r = 1; r <= 3; ++r) {
        int m = 2 * r + 1;
        float h_F[49];
        fill_mask(h_F, m);
        CHECK(cudaMemcpy(d_F, h_F, m * m * sizeof(float),
                         cudaMemcpyHostToDevice));

        float ms;
        TIME_KERNEL(ms, TRIALS,
                    (conv2D_basic<<<grid, block>>>(d_N, d_F, d_P, r, DIM, DIM)));
        check_kernel("conv2D_basic");

        double ops    = (double)n * (2.0 * m * m - 1.0);
        double gflops = ops / (ms * 1e-3) / 1e9;
        // unique DRAM traffic: read the image once, write it once
        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;
        printf("%d,%d,%.4f,%.1f,%.1f\n", r, m, ms, gflops, bw);
        fflush(stdout);
    }

    // read one value back so nothing can be optimized away
    CHECK(cudaMemcpy(h_P, d_P, sizeof(float), cudaMemcpyDeviceToHost));
    fprintf(stderr, "# sanity: P[0] = %f\n", h_P[0]);

    CHECK(cudaFree(d_N)); CHECK(cudaFree(d_P)); CHECK(cudaFree(d_F));
    free(h_N); free(h_P);
}

// ---------------------------------------------------------------------
// BREAK A: same image through the zero-padding kernel and the clamp
// kernel, and count what changed.
// ---------------------------------------------------------------------
static void run_break()
{
    const int DIM = 128, r = 2, m = 5;
    const int n = DIM * DIM;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_N   = (float*)malloc(bytes);
    float *h_gpu = (float*)malloc(bytes);
    float *h_brk = (float*)malloc(bytes);
    float h_F[25];
    fill_image(h_N, n);
    fill_mask(h_F, m);

    float *d_N, *d_P, *d_F;
    CHECK(cudaMalloc(&d_N, bytes));
    CHECK(cudaMalloc(&d_P, bytes));
    CHECK(cudaMalloc(&d_F, 25 * sizeof(float)));
    CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_F, h_F, 25 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = grid_for(DIM, DIM, block);

    conv2D_basic<<<grid, block>>>(d_N, d_F, d_P, r, DIM, DIM);
    check_kernel("conv2D_basic");
    CHECK(cudaMemcpy(h_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

    conv2D_clamp<<<grid, block>>>(d_N, d_F, d_P, r, DIM, DIM);
    check_kernel("conv2D_clamp");
    CHECK(cudaMemcpy(h_brk, d_P, bytes, cudaMemcpyDeviceToHost));

    float mxb = 0.0f;
    long badb = count_diffs(h_brk, h_gpu, n, 1e-3f, &mxb);

    printf("BREAK A - boundary rule changed from zero padding to clamp\n");
    printf("     image %dx%d, r = %d\n", DIM, DIM, r);
    printf("E1_BREAK_DIFFS = %ld\n", badb);
    printf("E1_BREAK_MAXERR = %.3e\n\n", mxb);
    print_diff_map(h_brk, h_gpu, DIM, DIM, 1e-3f, 4);
    printf("\n     Record E1_BREAK_DIFFS in your answer sheet, paste the map\n"
           "     into your report, and answer: why exactly these elements?\n");

    CHECK(cudaFree(d_N)); CHECK(cudaFree(d_P)); CHECK(cudaFree(d_F));
    free(h_N); free(h_gpu); free(h_brk);
}

int main(int argc, char** argv)
{
    print_gpu_banner();
    verify_or_stop();
    if (argc > 1 && argv[1][0] == 'b') run_break();
    else                               run_sweep();
    return 0;
}
