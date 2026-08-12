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
// clang-format off
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    // tiled fused attention using shared memory and online softmax.
    const int thread_id   = threadIdx.x;
    const int num_threads = blockDim.x;
    const int block_id    = blockIdx.x;
    const int num_blocks  = gridDim.x;

    // one dynamic shared segment carved into all working tiles.
    // (multiple `extern __shared__` declarations would alias the same region.)
    extern __shared__ float smem[];
    float* q_tile   = smem;                       // tile_q * head_dim
    float* k_tile   = q_tile + tile_q * head_dim; // tile_k * head_dim
    float* v_tile   = k_tile + tile_k * head_dim; // tile_k * head_dim
    float* sp_tile  = v_tile + tile_k * head_dim; // tile_q * tile_k
    float* row_max1 = sp_tile + tile_q * tile_k;  // tile_q
    float* row_max2 = row_max1 + tile_q;          // tile_q
    float* row_sum1 = row_max2 + tile_q;          // tile_q
    float* row_sum2 = row_sum1 + tile_q;          // tile_q

    float* old_row_max = row_max1;
    float* new_row_max = row_max2;
    float* old_row_sum = row_sum1;
    float* new_row_sum = row_sum2;

    for (int q_row_start = block_id * tile_q; q_row_start < seq_len; q_row_start += num_blocks * tile_q) {
        float* out_acc = out + q_row_start * head_dim;

        // zero the output accumulator and initialise running statistics.
        for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads)
            out_acc[idx] = 0.0f;
        for (int r = thread_id; r < tile_q; r += num_threads) {
            old_row_max[r] = -FLT_MAX;
            old_row_sum[r] = 0.0f;
        }
        __syncthreads();

        // load this block's Q tile once, reused across all KV tiles.
        load_tile (q, q_tile, q_row_start, 0, seq_len, head_dim, tile_q, head_dim, thread_id, num_threads);
        __syncthreads();

        for (int kv_row_start = 0; kv_row_start < seq_len; kv_row_start += tile_k) {
            load_tile (k, k_tile, kv_row_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
            load_tile (v, v_tile, kv_row_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
            __syncthreads();

            // S = scale * Q @ K^T  (tile_q x tile_k)
            tile_scores (q_tile, k_tile, sp_tile, tile_q, tile_k, head_dim, scale, thread_id, num_threads);
            __syncthreads();

            // mask K positions past seq_len to -inf: load_tile zero-pads OOB K
            // rows, which would otherwise give S=0 and contaminate the softmax
            // (exp(0-max) > 0). -inf -> exp(-inf) = 0, a clean zero contribution.
            for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
                int j = idx % tile_k;
                if (kv_row_start + j >= seq_len) sp_tile[idx] = -INFINITY;
            }
            __syncthreads();

            // block_max = rowmax(S)
            tile_rowmax(sp_tile, new_row_max, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // merge running max and rescale the running output to the new max.
            // corr = exp(old_max - new_max) brings old O / old l onto the new_max scale.
            for (int r = thread_id; r < tile_q; r += num_threads) {
                new_row_max[r] = online_max(old_row_max[r], new_row_max[r]);
                float corr = correction_factor(old_row_max[r], new_row_max[r]);
                rescale_output(out_acc + r * head_dim, head_dim, corr);
            }
            __syncthreads();

            // P = exp(S - new_max)  (already in the new running-max scale)
            tile_exp(sp_tile, new_row_max, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // block_sum = rowsum(P)
            tile_rowsum(sp_tile, new_row_sum, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // l_new = l_old * corr + block_sum
            for (int r = thread_id; r < tile_q; r += num_threads) {
                float corr = correction_factor(old_row_max[r], new_row_max[r]);
                new_row_sum[r] = update_running_sum(old_row_sum[r], corr, new_row_sum[r]);
            }
            __syncthreads();

            // O += P @ V   (P and O now share the new_max scale)
            accumulate_pv(sp_tile, v_tile, out_acc, tile_q, tile_k, head_dim, thread_id, num_threads);
            __syncthreads();

            // ping-pong: current new stats become next iteration's old stats.
            float* tmp;
            tmp         = old_row_max;
            old_row_max = new_row_max;
            new_row_max = tmp;
            tmp         = old_row_sum;
            old_row_sum = new_row_sum;
            new_row_sum = tmp;
        }

        // finalise: divide by the running sum.
        for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads) {
            int r = idx / head_dim;
            out_acc[idx] /= old_row_sum[r];
        }
        // leave out_acc fully written before the next Q tile reuses old_row_*.
        __syncthreads ();
    }
}
// clang-format on

# Step 24 - flash_attention_launcher
// clang-format off
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    // clang-format on
    // configure grid/block/shared memory and launch flash_attention_kernel
    dim3 blockDim(256);
    dim3 gridDim((seq_len + tile_q - 1) / tile_q);

    float scale = 1 / sqrtf(head_dim);
    int sharedMemSize = (tile_q * tile_k + tile_q * head_dim + 2 * tile_k * head_dim + 4 * tile_q) * sizeof(float);

    flash_attention_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);
}

# Step 25 - causal_mask
// clang-format off
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    // clang-format on
    // write -INFINITY into entries where the global key index exceeds the global query index.
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;
        int global_i = q_row_start + i;
        int global_j = k_col_start + j;
        if (global_i < global_j) s_tile[idx] = -INFINITY;
    }
}

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

