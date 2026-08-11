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

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

