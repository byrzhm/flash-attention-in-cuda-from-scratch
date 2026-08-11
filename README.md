# flash-attention-in-cuda-from-scratch
Build a tiled, IO-aware Flash Attention implementation in CUDA, starting from elementary GPU primitives and progressing to a fused online-softmax attention kernel. Along the way you implement a naive attention baseline, the online softmax math, and finish with a causal variant suitable for autoregressive models.
