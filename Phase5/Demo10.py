import torch, open_clip
from PIL import Image
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval().cuda()

# --- Zero-shot classification: không cần train gì cả ---
labels = ["một con mèo", "một con chó", "một chiếc xe hơi", "một bát phở"]
image = preprocess(Image.open("test.jpg")).unsqueeze(0).cuda()
text  = tokenizer(labels).cuda()

with torch.no_grad():
    img_f = model.encode_image(image)
    txt_f = model.encode_text(text)
    img_f /= img_f.norm(dim=-1, keepdim=True)     # chuẩn hoá -> cosine
    txt_f /= txt_f.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_f @ txt_f.T).softmax(dim=-1)

for label, p in zip(labels, probs[0].tolist()):
    print(f"{p:.2%}  {label}")


# --- Tìm kiếm ảnh bằng ngôn ngữ tự nhiên ---
def build_index(image_paths):
    feats = []
    for p in image_paths:
        im = preprocess(Image.open(p)).unsqueeze(0).cuda()
        with torch.no_grad():
            f = model.encode_image(im)
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0])
    return np.array(feats)

def search(query: str, index, paths, top_k=5):
    with torch.no_grad():
        q = model.encode_text(tokenizer([query]).cuda())
        q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    scores = index @ q
    return [(paths[i], float(scores[i])) for i in np.argsort(-scores)[:top_k]]

# search("người đội mũ bảo hiểm màu đỏ", index, paths)