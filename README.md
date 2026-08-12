# Flash Attention in CUDA from Scratch

Build a tiled, IO-aware Flash Attention implementation in CUDA, starting from elementary GPU primitives and progressing to a fused online-softmax attention kernel. Along the way you implement a naive attention baseline, the online softmax math, and finish with a causal variant suitable for autoregressive models.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** vector_add
- [x] **2.** scale_array
- [x] **3.** elementwise_exp
- [x] **4.** row_max
- [x] **5.** row_sum
- [x] **6.** dot_product
- [x] **7.** matmul
- [x] **8.** transpose
- [x] **9.** qk_scores
- [x] **10.** softmax_rows
- [x] **11.** pv_matmul
- [x] **12.** naive_attention
- [x] **13.** online_max
- [ ] **14.** correction_factor
- [ ] **15.** update_running_sum
- [ ] **16.** rescale_output
- [ ] **17.** load_tile
- [ ] **18.** tile_scores
- [ ] **19.** tile_rowmax
- [ ] **20.** tile_exp
- [ ] **21.** tile_rowsum
- [ ] **22.** accumulate_pv
- [ ] **23.** flash_attention_kernel
- [ ] **24.** flash_attention_launcher
- [ ] **25.** causal_mask
- [ ] **26.** flash_attention_causal_kernel

---

Built on Deep-ML.
