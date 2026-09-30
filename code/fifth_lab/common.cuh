// =====================================================================
// Week 6 Lab kit - shared helpers.
// You do not need to modify this file.
// =====================================================================
#ifndef W6_COMMON_CUH
#define W6_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

// Call after a kernel launch to catch both launch and execution errors.
static inline void check_kernel(const char* what)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "launch error in %s -> %s\n", what,
                cudaGetErrorString(e));
        exit(1);
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "execution error in %s -> %s\n", what,
                cudaGetErrorString(e));
        exit(1);
    }
}

// Reproducible, non-constant input. A constant image hides indexing bugs.
static inline void fill_image(float* h, int n)
{
    for (int i = 0; i < n; ++i) h[i] = (float)(i % 97) * 0.01f;
}

static inline void fill_mask(float* h, int m)
{
    // A mask that is not symmetric, so a transposed index shows up.
    for (int i = 0; i < m * m; ++i) h[i] = 0.1f + 0.01f * (float)i;
}

// ---------------------------------------------------------------------
// CPU reference convolution. BOUNDARY: 0 = zero padding, 1 = clamp.
// ---------------------------------------------------------------------
static inline void cpu_conv2D(const float* N, const float* F, float* P,
                              int r, int width, int height, int boundary)
{
    int m = 2 * r + 1;
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            float acc = 0.0f;
            for (int fy = 0; fy < m; ++fy) {
                for (int fx = 0; fx < m; ++fx) {
                    int iy = oy - r + fy;
                    int ix = ox - r + fx;
                    float v;
                    if (boundary == 1) {                  // clamp to edge
                        int cy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);
                        int cx = ix < 0 ? 0 : (ix >= width  ? width  - 1 : ix);
                        v = N[cy * width + cx];
                    } else {                              // zero padding
                        v = (iy >= 0 && iy < height && ix >= 0 && ix < width)
                            ? N[iy * width + ix] : 0.0f;
                    }
                    acc += F[fy * m + fx] * v;
                }
            }
            P[oy * width + ox] = acc;
        }
    }
}

// Count elements differing by more than tol.
static inline long count_diffs(const float* a, const float* b, int n,
                               float tol, float* max_err_out)
{
    long bad = 0; float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
        if (e > tol) ++bad;
    }
    if (max_err_out) *max_err_out = mx;
    return bad;
}

// ---------------------------------------------------------------------
// ASCII difference map. Each character covers a BLK x BLK region:
//   '.' every element matches      'X' at least one differs
// Paste this straight into your report.
// ---------------------------------------------------------------------
static inline void print_diff_map(const float* a, const float* b,
                                  int width, int height, float tol, int blk)
{
    printf("     difference map (each char = %dx%d pixels, "
           "'.' match, 'X' differ)\n", blk, blk);
    for (int by = 0; by < height; by += blk) {
        printf("     ");
        for (int bx = 0; bx < width; bx += blk) {
            int bad = 0;
            for (int y = by; y < by + blk && y < height && !bad; ++y)
                for (int x = bx; x < bx + blk && x < width; ++x)
                    if (fabsf(a[y * width + x] - b[y * width + x]) > tol) {
                        bad = 1; break;
                    }
            putchar(bad ? 'X' : '.');
        }
        putchar('\n');
    }
}

// ---------------------------------------------------------------------
// Timing: TRIALS back-to-back launches inside one CUDA event pair.
// Usage:  TIME_KERNEL(ms, 10, myKernel<<<g,b>>>(args));
// ---------------------------------------------------------------------
#define TIME_KERNEL(ms_out, TRIALS, LAUNCH) do {                           \
    cudaEvent_t _start, _end;                                                    \
    CHECK(cudaEventCreate(&_start)); CHECK(cudaEventCreate(&_end));              \
    LAUNCH; CHECK(cudaDeviceSynchronize());          /* warm-up */         \
    CHECK(cudaEventRecord(_start));                                            \
    for (int _t = 0; _t < (TRIALS); ++_t) { LAUNCH; }                      \
    CHECK(cudaEventRecord(_end));                                            \
    CHECK(cudaEventSynchronize(_end));                                       \
    float _m; CHECK(cudaEventElapsedTime(&_m, _start, _end));                    \
    (ms_out) = _m / (float)(TRIALS);                                       \
    CHECK(cudaEventDestroy(_start)); CHECK(cudaEventDestroy(_end));              \
} while (0)

static inline void print_gpu_banner()
{
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "# GPU: %s (CC %d.%d), %d SMs, %d threads/SM, "
                    "%d blocks/SM\n",
            p.name, p.major, p.minor, p.multiProcessorCount,
            p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor);
}

// Stop with a clear message. Used by the built-in checks so a wrong TODO
// is reported before any timing is done.
static inline void require(bool ok, const char* msg)
{
    if (!ok) {
        fprintf(stderr, "\n  STOP: %s\n\n", msg);
        exit(1);
    }
}

#endif
