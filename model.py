"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // implement elementwise c[i] = a[i] + b[i]
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // multiply each element of a by scalar in place
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // replace each a[i] with expf(a[i])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = expf(a[idx]);
    }
}

# Step 4 - row_max
__global__ void row_max (const float* matrix, float* out, int rows, int cols) {
    // compute the max of each row and write it to out[r].
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < rows) {
        float m = matrix[idx * cols];
        for (int col = 1; col < cols; col++) {
            m = fmaxf(m, matrix[idx * cols + col]);
        }
        out[idx] = m;
    }
}

# Step 5 - row_sum
__global__ void row_sum (const float* matrix, float* out, int rows, int cols) {
    // write out[r] = sum of matrix row r
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[idx * cols + col];
        }
        out[idx] = sum;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float res = 0.0f;
    for (int i = 0; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

# Step 7 - matmul
__global__ void matmul (const float* a, const float* b, float* c, int m, int k, int n) {
    // compute C = A * B for row-major matrices
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = acc;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // write out[c*rows + r] = in[r*cols + c]
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void
qk_scores (const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        scores[i * seq_len + j] = dot_product(q + i * head_dim, k + j * head_dim, head_dim) / sqrtf(head_dim);
    }
}

# Step 10 - softmax_rows
#include <cfloat>

__device__ float warp_reduce_max (float val) {
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val = fmaxf (val, __shfl_down_sync (mask, val, off));
    return val; // only laneId == 0
}

__device__ float warp_reduce_sum (float val) {
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    return val; // only laneId == 0
}

__device__ float block_reduce_max (float val, float* shared) {
    const int tid     = threadIdx.x;
    const int laneId  = tid % warpSize;
    const int warpId  = tid / warpSize;
    const int warpNum = blockDim.x / warpSize;

    val = warp_reduce_max (val);
    if (laneId == 0)
        shared[warpId] = val;
    __syncthreads ();

    if (warpId == 0) {
        val = (laneId < warpNum) ? shared[laneId] : -FLT_MAX;
        val = warp_reduce_max (val);
    }
    return val; // only tid == 0
}

__device__ float block_reduce_sum (float val, float* shared) {
    const int tid     = threadIdx.x;
    const int laneId  = tid % warpSize;
    const int warpId  = tid / warpSize;
    const int warpNum = blockDim.x / warpSize;

    val = warp_reduce_sum (val);
    if (laneId == 0)
        shared[warpId] = val;
    __syncthreads ();

    if (warpId == 0) {
        val = (laneId < warpNum) ? shared[laneId] : 0.0f;
        val = warp_reduce_sum (val);
    }
    return val; // only tid == 0
}


__global__ void softmax_rows (float* matrix, int rows, int cols) {
    // implement numerically stable row-wise softmax in place
    const int tid    = threadIdx.x;
    const int rowIdx = blockIdx.x;
    float* row       = matrix + rowIdx * cols;

    __shared__ float smem[32];
    float maxVal;
    float val;
    float sum;
    float inv_sum;

    maxVal = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        maxVal = fmaxf(maxVal, row[i]);
    maxVal = block_reduce_max(maxVal, smem);
    if (tid == 0)
        smem[0] = maxVal;
    __syncthreads();

    maxVal = smem[0];

    val = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        val += expf(row[i] - maxVal);
    sum = block_reduce_sum(val, smem);
    if (tid == 0)
        smem[0] = 1 / sum;
    __syncthreads();

    inv_sum = smem[0];

    for (int i = tid; i < cols; i += blockDim.x)
        row[i] = expf(row[i] - maxVal) * inv_sum;
}

# Step 11 - pv_matmul
__global__ void
pv_matmul (const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // compute out[i, d] = sum_j p[i, j] * v[j, d]
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int d = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += p[i * seq_len + j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = acc;
    }
}

# Step 12 - naive_attention
void naive_attention (const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // allocate scratch, launch qk_scores -> softmax_rows -> pv_matmul, free scratch
    float* d_scores;
    float* d_p;
    cudaMalloc (&d_scores, seq_len * seq_len * sizeof (float));
    d_p = d_scores;

    dim3 blockDim;
    dim3 gridDim;

    blockDim = dim3 (16, 16);
    gridDim  = dim3 ((seq_len + blockDim.x - 1) / blockDim.x,
     (seq_len + blockDim.y - 1) / blockDim.y);

    qk_scores<<<gridDim, blockDim>>> (d_q, d_k, d_scores, seq_len, head_dim);

    blockDim = dim3 (256);
    gridDim  = dim3 (seq_len);

    softmax_rows<<<gridDim, blockDim>>> (d_scores, seq_len, seq_len);

    blockDim = dim3 (16, 16);
    gridDim  = dim3 ((head_dim + blockDim.x - 1) / blockDim.x,
     (seq_len + blockDim.y - 1) / blockDim.y);

    pv_matmul<<<gridDim, blockDim>>> (d_p, d_v, d_out, seq_len, head_dim);

    // cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    cudaFree (d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor (float old_max, float new_max) {
    // return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum (float old_sum, float correction, float block_sum) {
    // combine the rescaled old sum with the new block sum
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // multiply each of the head_dim entries of out_row by correction in place
    for (int i = 0; i < head_dim; i++) {
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
// clang-format off
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // clang-format on
    // cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    for (int i = thread_id; i < tile_rows; i += num_threads) {
        int row = src_row_start + i;
        for (int j = 0; j < tile_cols; j++) {
            int col = src_col_start + j;
            shared_dst[i * tile_cols + j] = (row < src_rows && col < src_cols) ? src[row * src_cols + col] : 0.0f;
        }
    }
}

# Step 18 - tile_scores
// clang-format off
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;
        float acc = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            acc += q_tile[i * head_dim + k] * k_tile[j * head_dim + k];
        }
        s_tile[idx] = acc * scale;
    }
}

# Step 19 - tile_rowmax
__device__ float safe_warp_reduce_max(float val) {
    int mask = __activemask();
    int active = __popc(mask);          // 活跃线程数
    int max_off = 1;
    while (max_off < active) max_off <<= 1;
    max_off >>= 1;                      // 小于等于 active 的最大 2 的幂
    for (int off = max_off; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, off));
    return val;
}

__device__ void
tile_rowmax (const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // write row_max_out[r] = max over c of s_tile[r, c]
    const int numWarps = (num_threads + warpSize - 1) / warpSize;
    const int warpId   = thread_id / warpSize;
    const int laneId   = thread_id % warpSize;

    // one warp per row
    const int iterations = (tile_k + warpSize - 1) / warpSize;
    for (int r = warpId; r < tile_q; r += numWarps) {
        float maxVal = -FLT_MAX;
        for (int it = 0; it < iterations; it++) {
            int c     = it * warpSize + laneId;
            float val = (c < tile_k) ? s_tile[r * tile_k + c] : -FLT_MAX;
            float warpMaxVal = safe_warp_reduce_max(val);
            if (laneId == 0) maxVal = fmaxf(maxVal, warpMaxVal);
        }
        if (laneId == 0)
            row_max_out[r] = maxVal;
    }
}

# Step 20 - tile_exp
// clang-format off
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // clang-format on
    // for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int r = idx / tile_k;
        // int c = idx % tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
// clang-format off
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill row_sum_out[r] with the sum of p_tile row r

    // one thread per row
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float row_sum = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            row_sum += p_tile[r * tile_k + c];
        }
        row_sum_out[r] = row_sum;
    }
}

# Step 22 - accumulate_pv
// clang-format off
__device__ void accumulate_pv (const float* p_tile, const float* v_tile, float* out_acc, int tile_q,
                               int tile_k, int head_dim, int thread_id, int num_threads) {
    // clang-format on
    // cooperatively add P_tile * V_tile into out_acc
    for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        float acc = 0;
        for (int k = 0; k < tile_k; k++)
            acc += p_tile[i * tile_k + k] * v_tile[k * head_dim + j];
        out_acc[idx] += acc;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    // TODO: tiled fused attention using shared memory and online softmax.
}

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

