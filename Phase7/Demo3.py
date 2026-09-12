def kv_cache_size_gb(n_layers, n_kv_heads, head_dim, seq_len,
                     batch_size, dtype_bytes=2):
    """
    KV cache = 2 (K và V) × layers × kv_heads × head_dim × seq_len × batch × bytes
    Đây thường là thứ ĂN HẾT VRAM, không phải trọng số model.
    """
    return (2 * n_layers * n_kv_heads * head_dim *
            seq_len * batch_size * dtype_bytes) / 1e9


# Llama-3-8B: 32 layer, 8 KV head (GQA), head_dim 128
for seq, bs in [(2048, 1), (8192, 1), (32768, 1), (8192, 16)]:
    gb = kv_cache_size_gb(32, 8, 128, seq, bs)
    print(f"seq={seq:6d} batch={bs:3d} -> KV cache = {gb:6.2f} GB")

# seq=  2048 batch=  1 ->   0.27 GB
# seq=  8192 batch=  1 ->   1.07 GB
# seq= 32768 batch=  1 ->   4.29 GB
# seq=  8192 batch= 16 ->  17.18 GB   <- lớn hơn cả trọng số model 4-bit!

# CÁCH GIẢM KV CACHE:
# 1. GQA/MQA        -> giảm số KV head (model hiện đại đã có sẵn)
# 2. KV cache quant -> fp8 thay bf16, giảm 50%
# 3. PagedAttention -> giảm lãng phí do phân mảnh (vLLM tự làm)
# 4. Prefix caching -> chia sẻ KV của phần prompt chung
# 5. Sliding window -> chỉ giữ N token gần nhất