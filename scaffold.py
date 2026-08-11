"""
Flash Attention in CUDA from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

/*
 * scaffold.cu - End-to-end driver for the Flash Attention project.
 * Exercises elementary kernels, the naive attention pipeline, the tiled
 * Flash Attention launcher, and the causal variant on small toy inputs.
 */

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA err %s at %d: %s\n", #call, __LINE__,      \
                cudaGetErrorString(err));                                \
        exit(1);                                                         \
    }                                                                    \
} while (0)

static void print_row(const char* tag, const float* h, int row, int cols) {
    printf("%s row %d: ", tag, row);
    for (int i = 0; i < cols; ++i) printf("%7.4f ", h[row * cols + i]);
    printf("\n");
}

int main() {
    srand(0);
    const int seq_len  = 8;
    const int head_dim = 4;
    const int tile_q   = 4;
    const int tile_k   = 4;
    const int qkv_n    = seq_len * head_dim;
    const float scale  = 1.0f / sqrtf((float)head_dim);

    std::vector<float> h_q(qkv_n), h_k(qkv_n), h_v(qkv_n);
    for (int i = 0; i < qkv_n; ++i) {
        h_q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_k[i] = (float)rand() / RAND_MAX - 0.5f;
        h_v[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    float *d_q, *d_k, *d_v, *d_out_naive, *d_out_flash, *d_out_causal;
    CUDA_CHECK(cudaMalloc(&d_q, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_naive,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_flash,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_causal, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Elementary kernel sanity checks ---
    const int en = 16;
    std::vector<float> h_a(en), h_b(en);
    for (int i = 0; i < en; ++i) { h_a[i] = (float)i; h_b[i] = (float)(en - i); }
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, en * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, en * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, en * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), en * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), en * sizeof(float), cudaMemcpyHostToDevice));
    vector_add<<<1, 64>>>(d_a, d_b, d_c, en);
    scale_array<<<1, 64>>>(d_c, 0.5f, en);
    elementwise_exp<<<1, 64>>>(d_a, en);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_c(en);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, en * sizeof(float), cudaMemcpyDeviceToHost));
    printf("vector_add+scale_array[0..3]: %.2f %.2f %.2f %.2f\n", h_c[0], h_c[1], h_c[2], h_c[3]);

    // --- Row reductions on Q for demonstration ---
    float *d_rmax, *d_rsum;
    CUDA_CHECK(cudaMalloc(&d_rmax, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rsum, seq_len * sizeof(float)));
    row_max<<<seq_len, 32>>>(d_q, d_rmax, seq_len, head_dim);
    row_sum<<<seq_len, 32, 32 * sizeof(float)>>>(d_q, d_rsum, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_rmax(seq_len), h_rsum(seq_len);
    CUDA_CHECK(cudaMemcpy(h_rmax.data(), d_rmax, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rsum.data(), d_rsum, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Q row0 max=%.4f sum=%.4f\n", h_rmax[0], h_rsum[0]);

    // --- matmul + transpose sanity ---
    float *d_kt;
    CUDA_CHECK(cudaMalloc(&d_kt, qkv_n * sizeof(float)));
    dim3 tBlock(8, 8), tGrid((head_dim + 7) / 8, (seq_len + 7) / 8);
    transpose<<<tGrid, tBlock>>>(d_k, d_kt, seq_len, head_dim);
    float *d_qk;
    CUDA_CHECK(cudaMalloc(&d_qk, seq_len * seq_len * sizeof(float)));
    dim3 mBlock(8, 8), mGrid((seq_len + 7) / 8, (seq_len + 7) / 8);
    matmul<<<mGrid, mBlock>>>(d_q, d_kt, d_qk, seq_len, head_dim, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Naive attention baseline ---
    naive_attention(d_q, d_k, d_v, d_out_naive, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_naive(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), d_out_naive, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Flash Attention ---
    flash_attention_launcher(d_q, d_k, d_v, d_out_flash, seq_len, head_dim, tile_q, tile_k);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_flash(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_flash.data(), d_out_flash, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Causal Flash Attention ---
    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    size_t shmem = (tile_q * head_dim + tile_k * head_dim + tile_k * head_dim
                    + tile_q * tile_k + tile_q * head_dim + 3 * tile_q) * sizeof(float);
    flash_attention_causal_kernel<<<num_q_tiles, 128, shmem>>>(
        d_q, d_k, d_v, d_out_causal, seq_len, head_dim, tile_q, tile_k, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_causal(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_causal.data(), d_out_causal, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n--- Attention outputs (seq_len=%d, head_dim=%d) ---\n", seq_len, head_dim);
    print_row("naive ",  h_out_naive.data(),  0, head_dim);
    print_row("flash ",  h_out_flash.data(),  0, head_dim);
    print_row("causal", h_out_causal.data(), 0, head_dim);
    print_row("naive ",  h_out_naive.data(),  seq_len - 1, head_dim);
    print_row("flash ",  h_out_flash.data(),  seq_len - 1, head_dim);
    print_row("causal", h_out_causal.data(), seq_len - 1, head_dim);

    float max_diff = 0.0f;
    for (int i = 0; i < qkv_n; ++i)
        max_diff = fmaxf(max_diff, fabsf(h_out_naive[i] - h_out_flash[i]));
    printf("\nmax|naive - flash| = %.6e\n", max_diff);

    // --- Why Flash Attention matters: memory, not FLOPs ---
    // Naive attention stores the whole seq_len x seq_len score matrix in global
    // memory before softmax; Flash Attention streams over key/value tiles and
    // keeps only a per-row running max and sum, so it never allocates that
    // matrix. Same result (see max|naive - flash| above), very different memory.
    printf("\n--- Memory: naive O(N^2) scores vs flash O(1) global scratch ---\n");
    printf("  this run (seq_len=%d): naive scores = %.0f bytes, flash global scratch = 0\n",
           seq_len, (double)seq_len * (double)seq_len * sizeof(float));
    printf("  %12s %18s %18s\n", "seq_len", "naive scores", "flash scratch");
    long long demo_lens[4] = {1024LL, 8192LL, 32768LL, 131072LL};
    for (int i = 0; i < 4; ++i) {
        long long N = demo_lens[i];
        double naive_mb = (double)N * (double)N * (double)sizeof(float) / 1.0e6;
        printf("  %12lld %15.1f MB %18s\n", N, naive_mb, "~0 (tiles only)");
    }
    printf("  Flash keeps only a tile in shared memory (tens of KB per block), so it runs\n");
    printf("  at sequence lengths where the naive score matrix would not fit in GPU memory.\n");
    printf("  (This from-scratch kernel favors clarity over speed; it is not throughput-\n");
    printf("   optimized like production FlashAttention -- the win here is memory scaling.)\n");


    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_out_naive); cudaFree(d_out_flash); cudaFree(d_out_causal);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_rmax); cudaFree(d_rsum);
    cudaFree(d_kt); cudaFree(d_qk);
    return 0;
}
