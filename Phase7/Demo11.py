import torch, torch.nn as nn

class SimpleVLM(nn.Module):
    """
    Công thức chung của mọi VLM hiện đại (LLaVA, Qwen-VL, InternVL):
      Vision Encoder → Projector → ghép vào chuỗi token của LLM
    Chìa khoá: chiếu feature ảnh vào CÙNG KHÔNG GIAN với token embedding.
    """
    def __init__(self, vision_encoder, llm, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.vision = vision_encoder     # ví dụ SigLIP / CLIP ViT
        self.llm = llm
        # Projector: thường chỉ là MLP 2 tầng — đơn giản đến bất ngờ
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, images, input_ids, attention_mask):
        # 1) Ảnh -> patch feature
        with torch.no_grad():
            img_feat = self.vision(images)             # (B, n_patch, vision_dim)

        # 2) Chiếu sang không gian của LLM
        img_tokens = self.projector(img_feat)          # (B, n_patch, llm_dim)

        # 3) Ghép trước token văn bản
        txt_tokens = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([img_tokens, txt_tokens], dim=1)

        img_mask = torch.ones(img_tokens.shape[:2], device=images.device)
        mask = torch.cat([img_mask, attention_mask], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=mask)


# HUẤN LUYỆN 2 GIAI ĐOẠN (chuẩn của LLaVA):
#   GD1 — Pretrain align : ĐÓNG BĂNG vision + LLM, chỉ train projector
#                          (dữ liệu: cặp ảnh-caption, số lượng lớn)
#   GD2 — Instruction FT : mở băng LLM (hoặc LoRA), train cùng projector
#                          (dữ liệu: hội thoại về ảnh, chất lượng cao)