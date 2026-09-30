// =====================================================================
// Week 6 Lab - Exercise 3: from one filter to a CNN layer
//
//   nvcc -O3 -arch=sm_86 conv_layer.cu -o conv_layer
//
//   ./conv_layer          channel sweep C = 1..64 at 512x512 -> CSV
//   ./conv_layer break    BREAK D: 8 stacked layers, in place vs ping-pong
//
// A convolution LAYER convolves C_in input channels with C_out filters:
//   Y[oc][y][x] = sum over ic, fy, fx of
//                    W[oc][ic][fy][fx] * X[ic][y+fy-R][x+fx-R]
//
// There are 2 TODOs in this file - search for "TODO". Each is a line or two,
// and each is one idea from the Week 6 code lecture. The program checks your
// TODOs before it measures anything and tells you which one is wrong.
// =====================================================================
#include "common.cuh"

#define K      3                  // 3x3 filters
#define LR     1                  // filter radius
#define IN_T   18                 // block is IN_T x IN_T
#define OUT_T  (IN_T - 2 * LR)    // 16

__global__ void conv_layer(const float* X, const float* Wt, float* Y,
                           int Cin, int Cout, int H, int W)
{
    int oc  = blockIdx.z;                     // one output channel per z
    int col = blockIdx.x * OUT_T + (int)threadIdx.x - LR;
    int row = blockIdx.y * OUT_T + (int)threadIdx.y - LR;

    __shared__ float Xs[IN_T][IN_T];

    int tileRow = (int)threadIdx.y - LR;
    int tileCol = (int)threadIdx.x - LR;
    bool active = (tileRow >= 0 && tileRow < OUT_T &&
                   tileCol >= 0 && tileCol < OUT_T &&
                   row >= 0 && row < H && col >= 0 && col < W);
    bool inImage = (row >= 0 && row < H && col >= 0 && col < W);

    float acc = 0.0f;
    for (int ic = 0; ic < Cin; ++ic) {
        Xs[threadIdx.y][threadIdx.x] =
            inImage ? X[((size_t)ic * H + row) * W + col] : 0.0f;
        __syncthreads();                       // read-after-write

        if (active) {
            // ---- TODO 5 --------------------------------------------------
            // The weights are stored flattened as W[oc][ic][fy][fx]. Point w
            // at the first of the 9 weights that connect input channel ic to
            // output channel oc. How many floats come before that filter?
            const float* w = Wt + ((size_t)oc * Cin + ic) * K * K;
            // ------------------------------------------------------------
            for (int fy = 0; fy < K; ++fy)
                for (int fx = 0; fx < K; ++fx)
                    acc += w[fy * K + fx] * Xs[tileRow + fy][tileCol + fx];
        }
        __syncthreads();                       // write-after-read
    }
    if (active) Y[((size_t)oc * H + row) * W + col] = acc;
}

static void launch_layer(const float* dX, const float* dW, float* dY,
                         int Cin, int Cout, int H, int W)
{
    dim3 block(IN_T, IN_T);
    dim3 grid((W + OUT_T - 1) / OUT_T, (H + OUT_T - 1) / OUT_T, Cout);
    conv_layer<<<grid, block>>>(dX, dW, dY, Cin, Cout, H, W);
}

// Apply the layer L times, like stacking L layers of a CNN.
// dA holds the input on entry. Returns the buffer holding the final output.
static float* run_layers(float* dA, float* dB, const float* dW,
                         int C, int H, int W, int L)
{
    float* in  = dA;
    float* out = dB;
    for (int l = 0; l < L; ++l) {
        launch_layer(in, dW, out, C, C, H, W);
        // ---- TODO 6 ------------------------------------------------------
        // The next layer must READ what this layer just wrote, and must WRITE
        // somewhere else. Swap the two pointers.   [Week 6 code lecture:
        // 'Never update an iterative stencil in place']
        float* tmp = in;
        in = out;
        out = tmp;
        // ----------------------------------------------------------------
    }
    return in;              // after the last swap, 'in' is the newest result
}

// ---------------------------------------------------------------- CPU
static void cpu_layer(const float* X, const float* Wt, float* Y,
                      int Cin, int Cout, int H, int W)
{
    for (int oc = 0; oc < Cout; ++oc)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float acc = 0.0f;
                for (int ic = 0; ic < Cin; ++ic)
                    for (int fy = 0; fy < K; ++fy)
                        for (int fx = 0; fx < K; ++fx) {
                            int iy = y + fy - LR, ix = x + fx - LR;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                acc += Wt[((size_t)oc * Cin + ic) * K * K
                                          + fy * K + fx]
                                     * X[((size_t)ic * H + iy) * W + ix];
                        }
                Y[((size_t)oc * H + y) * W + x] = acc;
            }
}

// Every (oc, ic) pair gets DIFFERENT weights, so a wrong index shows up.
static void fill_weights(float* w, int n)
{
    for (int i = 0; i < n; ++i) w[i] = 0.02f + 0.001f * (float)(i % 53);
}

// Uniform weights with gain ~1 per layer, so 8 stacked layers stay finite.
static void fill_stable_weights(float* w, int C)
{
    float v = 1.0f / (float)(C * K * K);
    for (int i = 0; i < C * C * K * K; ++i) w[i] = v;
}

// ---------------------------------------------------------------------
// Built-in check, run before anything is measured.
//   1) one layer at C = 2 on an odd-sized image   -> tests TODO 5
//   2) eight stacked layers at C = 4               -> tests TODO 6
// ---------------------------------------------------------------------
static void verify_or_stop()
{
    // ---- 1) a single layer ------------------------------------------
    {
        const int C = 2, H = 33, W = 31;
        int n = C * H * W;
        size_t xb = (size_t)n * sizeof(float);
        size_t wb = (size_t)C * C * K * K * sizeof(float);
        float* hX = (float*)malloc(xb);
        float* hW = (float*)malloc(wb);
        float* hR = (float*)malloc(xb);
        float* hG = (float*)malloc(xb);
        fill_image(hX, n);
        fill_weights(hW, C * C * K * K);
        cpu_layer(hX, hW, hR, C, C, H, W);

        float *dX, *dW, *dY;
        CHECK(cudaMalloc(&dX, xb)); CHECK(cudaMalloc(&dW, wb));
        CHECK(cudaMalloc(&dY, xb));
        CHECK(cudaMemcpy(dX, hX, xb, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dY, 0, xb));
        launch_layer(dX, dW, dY, C, C, H, W);
        check_kernel("conv_layer (verify)");
        CHECK(cudaMemcpy(hG, dY, xb, cudaMemcpyDeviceToHost));

        float mx = 0.0f;
        long bad = count_diffs(hG, hR, n, 1e-3f, &mx);
        if (bad > 0) {
            fprintf(stderr, "  one layer, C = 2: %ld of %d outputs wrong.\n",
                    bad, n);
            require(false, "check TODO 5 - the weight pointer does not select "
                           "the filter for this (oc, ic) pair");
        }
        fprintf(stderr, "# verify: one layer matches the CPU reference "
                        "(max error %.2e)\n", mx);
        CHECK(cudaFree(dX)); CHECK(cudaFree(dW)); CHECK(cudaFree(dY));
        free(hX); free(hW); free(hR); free(hG);
    }

    // ---- 2) eight stacked layers ------------------------------------
    {
        const int C = 4, H = 24, W = 24, L = 8;
        int n = C * H * W;
        size_t xb = (size_t)n * sizeof(float);
        size_t wb = (size_t)C * C * K * K * sizeof(float);
        float* hX = (float*)malloc(xb);
        float* hW = (float*)malloc(wb);
        float* hA = (float*)malloc(xb);
        float* hB = (float*)malloc(xb);
        float* hG = (float*)malloc(xb);
        fill_image(hX, n);
        fill_stable_weights(hW, C);

        // CPU: apply the layer L times, copying each result back as the
        // next input (slow but obviously correct)
        for (int i = 0; i < n; ++i) hA[i] = hX[i];
        for (int l = 0; l < L; ++l) {
            cpu_layer(hA, hW, hB, C, C, H, W);
            for (int i = 0; i < n; ++i) hA[i] = hB[i];
        }

        float *dA, *dB, *dW;
        CHECK(cudaMalloc(&dA, xb)); CHECK(cudaMalloc(&dB, xb));
        CHECK(cudaMalloc(&dW, wb));
        CHECK(cudaMemcpy(dA, hX, xb, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dB, 0, xb));
        CHECK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
        float* dRes = run_layers(dA, dB, dW, C, H, W, L);
        check_kernel("run_layers (verify)");
        CHECK(cudaMemcpy(hG, dRes, xb, cudaMemcpyDeviceToHost));

        float mx = 0.0f;
        long bad = count_diffs(hG, hA, n, 1e-3f, &mx);
        if (bad > 0) {
            fprintf(stderr, "  %d stacked layers: %ld of %d outputs wrong.\n",
                    L, bad, n);
            require(false, "check TODO 6 - each layer must read what the "
                           "previous layer wrote");
        }
        fprintf(stderr, "# verify: %d stacked layers match the CPU reference "
                        "(max error %.2e)\n", L, mx);
        CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dW));
        free(hX); free(hW); free(hA); free(hB); free(hG);
    }
}

// ------------------------------------------------------- channel sweep
static void run_sweep()
{
    const int H = 512, W = 512, TRIALS = 10;
    const int Cs[] = {1, 4, 8, 16, 32, 64};

    printf("C,ms_per_trial,gflop_s,ai_ideal\n");
    for (int i = 0; i < 6; ++i) {
        int C = Cs[i];
        size_t xb = (size_t)C * H * W * sizeof(float);
        size_t wb = (size_t)C * C * K * K * sizeof(float);

        float* hX = (float*)malloc(xb);
        float* hW = (float*)malloc(wb);
        fill_image(hX, C * H * W);
        fill_weights(hW, C * C * K * K);

        float *dX, *dW, *dY;
        CHECK(cudaMalloc(&dX, xb)); CHECK(cudaMalloc(&dW, wb));
        CHECK(cudaMalloc(&dY, xb));
        CHECK(cudaMemcpy(dX, hX, xb, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));

        float ms;
        TIME_KERNEL(ms, TRIALS, (launch_layer(dX, dW, dY, C, C, H, W)));
        check_kernel("conv_layer sweep");

        double ops = (double)H * W * C * C * K * K * 2.0;
        double gflops = ops / (ms * 1e-3) / 1e9;
        double ai = (double)C * K * K / 4.0;      // ideal-reuse model
        printf("%d,%.4f,%.1f,%.2f\n", C, ms, gflops, ai);
        fflush(stdout);

        CHECK(cudaFree(dX)); CHECK(cudaFree(dW)); CHECK(cudaFree(dY));
        free(hX); free(hW);
    }
}

// --------------------------------------- BREAK D: in place vs ping-pong
static void run_break()
{
    const int C = 16, H = 256, W = 256, L = 8;
    size_t xb = (size_t)C * H * W * sizeof(float);
    size_t wb = (size_t)C * C * K * K * sizeof(float);
    int n = C * H * W;

    float* hX = (float*)malloc(xb);
    float* hW = (float*)malloc(wb);
    float* hA = (float*)malloc(xb);
    float* hB = (float*)malloc(xb);
    float* hGood = (float*)malloc(xb);
    fill_image(hX, n);
    fill_stable_weights(hW, C);

    float *dA, *dB, *dW;
    CHECK(cudaMalloc(&dA, xb)); CHECK(cudaMalloc(&dB, xb));
    CHECK(cudaMalloc(&dW, wb));
    CHECK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));

    printf("BREAK D - %d stacked layers, C = %d, %dx%d\n", L, C, H, W);

    // ---- correct: ping-pong -----------------------------------------
    CHECK(cudaMemcpy(dA, hX, xb, cudaMemcpyHostToDevice));
    float* dGood = run_layers(dA, dB, dW, C, H, W, L);
    check_kernel("ping-pong");
    CHECK(cudaMemcpy(hGood, dGood, xb, cudaMemcpyDeviceToHost));
    printf("     ping-pong (correct): done, sample value %.6f\n", hGood[0]);

    // ---- wrong: in place --------------------------------------------
    long diffs[2]; float mx[2];
    for (int run = 0; run < 2; ++run) {
        CHECK(cudaMemcpy(dA, hX, xb, cudaMemcpyHostToDevice));
        for (int l = 0; l < L; ++l)
            launch_layer(dA, dW, dA, C, C, H, W);   // SAME buffer both ways
        check_kernel("in-place");
        float* dst = run == 0 ? hA : hB;
        CHECK(cudaMemcpy(dst, dA, xb, cudaMemcpyDeviceToHost));
        diffs[run] = count_diffs(dst, hGood, n, 1e-3f, &mx[run]);
        printf("     in-place run %d: %ld of %d elements differ from the "
               "correct result (max error %.3e)\n",
               run + 1, diffs[run], n, mx[run]);
    }
    long between = count_diffs(hA, hB, n, 1e-3f, nullptr);

    printf("\nE3_INPLACE_DIFFS = %ld\n", diffs[0]);
    printf("E3_INPLACE_RUNS_AGREE = %s\n", between == 0 ? "yes" : "no");
    printf("     (the two in-place runs differ from EACH OTHER in %ld "
           "elements)\n", between);
    printf("\n     Record both values. Then explain WHICH thread read data\n"
           "     written by WHICH other thread, and why __syncthreads()\n"
           "     inside a block does not prevent it.\n");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dW));
    free(hX); free(hW); free(hA); free(hB); free(hGood);
}

int main(int argc, char** argv)
{
    print_gpu_banner();
    verify_or_stop();
    if (argc > 1 && argv[1][0] == 'b') run_break();
    else                               run_sweep();
    return 0;
}
