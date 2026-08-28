#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define BLUR_SIZE 2

__global__ void gray_kernel(
    const unsigned char *rgb,
    unsigned char *gray,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        int rgb_idx = idx * 3;
        gray[idx] = (unsigned char)(0.21f * rgb[rgb_idx + 0] + 0.71f * rgb[rgb_idx + 1] + 0.07f * rgb[rgb_idx + 2]);
    }
}

__global__ void blur_kernel(
    const unsigned char *in,
    unsigned char *out,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int sum = 0;
        int count = 0;

        for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
            for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
                int r = row + dy;
                int c = col + dx;

                if (r >= 0 && r < height && c >= 0 && c < width) {
                    sum += in[r * width + c];
                    count++;
                }
            }
        }

        out[row * width + col] = (unsigned char)(sum / count);
    }
}

void make_image(
    unsigned char *rgb,
    int width,
    int height
) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = (row * width + col) * 3;
            rgb[i + 0] = (unsigned char)((col * 255) / width);
            rgb[i + 1] = (unsigned char)((row * 255) / height);
            rgb[i + 2] = (((row / 32) + (col / 32)) % 2) ? 220 : 40;
        }
    }
}

void cpu_gray(
    const unsigned char *rgb,
    unsigned char *gray,
    int width,
    int height
) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int g0 = row * width + col;
            int r0 = g0 * 3;
            gray[g0] = (unsigned char)(0.21f * rgb[r0 + 0] + 0.71f * rgb[r0 + 1] + 0.07f * rgb[r0 + 2]);
        }
    }
}

void cpu_blur(
    const unsigned char *in,
    unsigned char *out,
    int width,
    int height
) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int sum = 0, count = 0;

            for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
                for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
                    int r = row + dy;
                    int c = col + dx;

                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        sum += in[r * width + c];
                        count++;
                    }
                }
            }

            out[row * width + col] = (unsigned char)(sum / count);
        }
    }
}

int compare(
    const char *label,
    const unsigned char *got,
    const unsigned char *want,
    int n,
    int tol
) {
    int bad = 0;

    for (int i = 0; i < n; i++) {
        int d = (int)got[i] - (int)want[i];

        if (d < 0) {
            d = -d;
        }

        if (d > tol) {
            if (bad < 5) {
                printf(
                    " %s mismatch at i=%d: got %d, expected %d\n",
                    label,
                    i,
                    (int)got[i],
                    (int)want[i]
                );
            }

            bad++;
        }
    }

    if (bad == 0) {
        printf(
            "%s PASS\n",
            label
        );
    } else {
        printf(
            "%s FAIL: %d of %d pixels wrong\n",
            label,
            bad,
            n
        );
    }

    return bad;
}

float time_blur(
    const unsigned char *d_in,
    unsigned char *d_out,
    int width,
    int height,
    int bx,
    int by
) {
    dim3 block(
        bx,
        by
    );
    dim3 grid(
        (width + bx - 1) / bx,
        (height + by - 1) / by
    );
    blur_kernel<<<grid, block>>>(
        d_in,
        d_out,
        width,
        height
    );
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 10; i++) {
        blur_kernel<<<grid, block>>>(
            d_in,
            d_out,
            width,
            height
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 10.0f;
}

int main(
    int argc,
    char **argv
) {
    int width = 1024;

    if (argc > 1) {
        width = atoi(argv[1]);
    }

    int height = width;
    int n = width * height;
    size_t rgb_bytes = (size_t)n * 3 * sizeof(unsigned char);
    size_t gray_bytes = (size_t)n * sizeof(unsigned char);
    unsigned char *rgb = (unsigned char *)malloc(rgb_bytes);
    unsigned char *gray_cpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *blur_cpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *gray_gpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *blur_gpu = (unsigned char *)malloc(gray_bytes);
    make_image(
        rgb,
        width,
        height
    );
    cpu_gray(
        rgb,
        gray_cpu,
        width,
        height
    );
    cpu_blur(
        gray_cpu,
        blur_cpu,
        width,
        height
    );
    unsigned char *d_rgb;
    unsigned char *d_gray;
    unsigned char *d_blur;
    cudaMalloc(
        &d_rgb,
        rgb_bytes
    );
    cudaMalloc(
        &d_gray,
        gray_bytes
    );
    cudaMalloc(
        &d_blur,
        gray_bytes
    );
    cudaMemcpy(
        d_rgb,
        rgb,
        rgb_bytes,
        cudaMemcpyHostToDevice
    );
    dim3 gray_block(
        16,
        16
    );
    dim3 gray_grid(
        (width + gray_block.x - 1) / gray_block.x,
        (height + gray_block.y - 1) / gray_block.y
    );
    gray_kernel<<<gray_grid, gray_block>>>(
        d_rgb,
        d_gray,
        width,
        height
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        gray_gpu,
        d_gray,
        gray_bytes,
        cudaMemcpyDeviceToHost
    );
    compare(
        "Gray",
        gray_gpu,
        gray_cpu,
        n,
        0
    );
    dim3 blur_block(
        16,
        16
    );
    dim3 blur_grid(
        (width + blur_block.x - 1) / blur_block.x,
        (height + blur_block.y - 1) / blur_block.y
    );
    blur_kernel<<<blur_grid, blur_block>>>(
        d_gray,
        d_blur,
        width,
        height
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        blur_gpu,
        d_blur,
        gray_bytes,
        cudaMemcpyDeviceToHost
    );
    compare(
        "Blur",
        blur_gpu,
        blur_cpu,
        n,
        0
    );

    if (width == 4096) {
        printf("Block size timing\n");
        printf(
            "Size 1, 8 by 8, %.4f ms\n",
            time_blur(
                d_gray,
                d_blur,
                width,
                height,
                8,
                8
            )
        );
        printf(
            "Size 2, 16 by 16, %.4f ms\n",
            time_blur(
                d_gray,
                d_blur,
                width,
                height,
                16,
                16
            )
        );
        printf(
            "Size 3, 32 by 8, %.4f ms \n",
            time_blur(
                d_gray,
                d_blur,
                width,
                height,
                32,
                8
            )
        );
        printf(
            "Size 4, 8 by 32, %.4f ms \n",
            time_blur(
                d_gray,
                d_blur,
                width,
                height,
                8,
                32
            )
        );
    }

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    free(rgb);
    free(gray_cpu);
    free(blur_cpu);
    free(gray_gpu);
    free(blur_gpu);

    return 0;
}