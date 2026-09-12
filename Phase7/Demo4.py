"""
Nguyên lý: model NHỎ (draft) đoán trước k token,
model LỚN (target) xác minh TẤT CẢ cùng lúc trong 1 lượt forward.

Vì sao nhanh hơn? Sinh token là bài toán memory-bound,
không phải compute-bound. Xác minh 5 token tốn gần bằng sinh 1 token.
"""

def speculative_step(draft_model, target_model, prefix, k=5):
    # B1: draft sinh k token (nhanh, model nhỏ)
    draft_tokens, draft_probs = [], []
    ctx = prefix
    for _ in range(k):
        p = draft_model(ctx)
        t = sample(p)
        draft_tokens.append(t); draft_probs.append(p[t])
        ctx = ctx + [t]

    # B2: target xác minh TẤT CẢ k token trong MỘT lượt forward
    target_probs = target_model(prefix + draft_tokens)   # <- chỉ 1 lần gọi

    # B3: chấp nhận từng token theo xác suất (rejection sampling)
    accepted = []
    for i, t in enumerate(draft_tokens):
        r = uniform(0, 1)
        if r < min(1, target_probs[i][t] / draft_probs[i]):
            accepted.append(t)
        else:
            # Từ chối -> lấy mẫu lại từ phân phối điều chỉnh, dừng tại đây
            accepted.append(sample(normalize(target_probs[i] - draft_probs[i])))
            break
    return accepted

# TỈ LỆ CHẤP NHẬN quyết định tất cả:
#   >= 0.7  -> tăng tốc 1.3-2x
#   <  0.5  -> KHÔNG nhanh hơn, thậm chí chậm hơn do overhead
#
# Các biến thể:
#   EAGLE-3   : draft head huấn luyện riêng, tỉ lệ chấp nhận cao
#   Medusa    : nhiều đầu dự đoán song song
#   N-gram    : tra cứu từ chính prompt, ZERO overhead tính toán
#               -> rất hiệu quả cho tác vụ có nhiều lặp lại (sửa code, tóm tắt)