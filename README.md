# AI Engineering Review

Kho lưu trữ mã nguồn thực hành cho lộ trình **AI Engineer Roadmap** gồm **7 phase** — từ nền tảng toán & Python, qua Machine Learning, Deep Learning, NLP/LLM, Computer Vision, MLOps, đến AI Engineering nâng cao.

📘 **Lộ trình chi tiết trên Notion:** [AI Engineer Roadmap — Lộ trình học đầy đủ](https://ai-engineer-roadmap.notion.site/AI-Engineer-Roadmap-L-tr-nh-h-c-y-3d10743dbce580f68924cd33d22ec22a?source=copy_link)

---

## Tổng quan 7 Phase

| Phase | Chủ đề | Thời gian ước tính | Trạng thái code |
|-------|--------|--------------------|-----------------|
| **Phase 1** | Nền tảng (Math + Python) | 4–6 tuần | ✅ Demo + dự án cuối phase |
| **Phase 2** | Machine Learning cơ bản | 6–8 tuần | ✅ Demo + dự án cuối phase |
| **Phase 3** | Deep Learning & Transformer | 6–8 tuần | ✅ Demo + dự án cuối phase |
| **Phase 4** | NLP & Large Language Models | 6–8 tuần | ✅ Demo 1–19 |
| **Phase 5** | Computer Vision | 4–6 tuần | ✅ Demo 1–12 + dự án cuối phase |
| **Phase 6** | MLOps & Triển khai Production | 4–6 tuần | 🚧 Đang chuẩn bị |
| **Phase 7** | AI Engineering nâng cao | On-going | 🚧 Đang chuẩn bị |

---

## Cấu trúc dự án

```
AI-Engineering-Review/
├── Phase1/          # Nền tảng: NumPy, toán ML, autograd, Python nâng cao
├── Phase2/          # Classical ML: sklearn, tuning, explainability
├── Phase3/          # Deep Learning: PyTorch, CNN, RNN/LSTM, Transformer, MiniGPT
├── Phase4/          # NLP, LLM, RAG & AI Agents
├── Phase5/          # Computer Vision: YOLO, tracking, SAM, CLIP, ONNX
├── Phase6/          # MLOps & triển khai production
├── Phase7/          # AI Engineering nâng cao
└── ai-roadmap/      # Môi trường Python (uv) & dependencies dùng chung
```

Mỗi phase gồm các file `Demo1.py` → `DemoN.py` (bài thực hành ngắn, chạy độc lập) và một file tổng hợp cuối phase.

| Phase | File tổng hợp | Chủ đề chính |
|-------|---------------|--------------|
| Phase 1 | `Phase1/final_phase1.py` | Mini Vector DB (TF-IDF + cosine search) |
| Phase 2 | `Phase2/final_phase2.py` | Pipeline dự đoán churn (LightGBM + Optuna) |
| Phase 3 | `Phase3/final_phase3.py` | MiniGPT — BPE tokenizer + RoPE |
| Phase 4 | *(chưa có final)* | Tokenizer → RAG → Agents → Fine-tune → Eval/Obs |
| Phase 5 | `Phase5/final_phase5.py` | Video analytics — detect, track, zone, alert |
| Phase 6 | — | MLOps, deployment, monitoring |
| Phase 7 | — | Model optimization, system design, multimodal AI |

---

## Yêu cầu hệ thống

- **Python** 3.12+
- **GPU** (khuyến nghị từ Phase 3 trở đi) — PyTorch cấu hình CUDA 12.8
- **[uv](https://docs.astral.sh/uv/)** — quản lý môi trường & cài đặt dependencies
- **OpenAI API key** — bắt buộc cho nhiều demo Phase 4 (structured output, RAG, agents, Langfuse)
- **Anthropic API key** — tùy chọn: `Demo6-anthropic.py`, `Demo19.py` (LLM-as-judge)
- **Langfuse keys** — tùy chọn: `Demo18.py`, `Demo19-main.py` (observability)
- **GPU** — khuyến nghị mạnh cho Demo16/17 (QLoRA / DPO)

---

## Cài đặt

```bash
git clone https://github.com/tientho201/AI-Engineer-Roadmap.git
cd AI-Engineer-Roadmap

cd ai-roadmap && uv sync && cd ..
source ai-roadmap/.venv/bin/activate   # Linux / macOS
# ai-roadmap\.venv\Scripts\activate    # Windows
```

### Cấu hình API keys (`.env`)

Tạo file `.env` ở thư mục gốc repo:

```bash
cp .env.example .env
```

Điền key:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...          # tùy chọn
LANGFUSE_PUBLIC_KEY=pk-lf-...         # tùy chọn — Demo18 / Demo19-main
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

| Biến | Lấy key tại | Dùng cho |
|------|-------------|----------|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Demo4–7, 6-openai, 11–12, 14–15, 18, 19-main |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Demo6-anthropic, Demo19 |
| `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` | [cloud.langfuse.com](https://cloud.langfuse.com) → Settings → API Keys | Demo18, Demo19-main |

**Không commit** `.env` (đã có trong `.gitignore`). Chạy demo từ thư mục gốc repo để `load_dotenv()` tìm được `.env`.

```bash
python Phase1/Demo1.py
python Phase3/final_phase3.py
python Phase4/Demo1.py
python Phase4/Demo18.py   # cần OpenAI + Langfuse
python Phase4/Demo19.py   # cần Anthropic
```

---

## Phase 1 — Nền tảng AI / ML

Xây dựng trực giác toán học và kỹ năng Python cần thiết trước khi vào ML/DL.

| Demo | Chủ đề |
|------|--------|
| Demo1 | Vector & Cosine Similarity (nền tảng vector database) |
| Demo2 | Nhân ma trận & Broadcasting |
| Demo3 | PCA — tự cài đặt bằng SVD, so sánh sklearn |
| Demo4 | Gradient Descent — hồi quy tuyến tính từ dữ liệu |
| Demo5 | Autograd thuần Python — chain rule, backprop |
| Demo6 | Softmax & Cross-Entropy Loss |
| Demo7 | Sampling — temperature, top-p |
| Demo8 | Bayes — xác suất có điều kiện (bài toán xét nghiệm y tế) |
| Demo9 | Python nâng cao — dataclass, generator, decorator, async |
| Demo10-a/b | NumPy vectorization vs vòng lặp Python |
| Demo11 | Pandas — groupby, phân tích dữ liệu |
| Demo12 | Confusion matrix & metrics phân loại |
| Demo13 | sklearn Pipeline — preprocessing + Logistic Regression |

**Dự án cuối phase:** `MiniVectorDB` — cài đặt vector search đơn giản với TF-IDF embedding, hỗ trợ search đơn và batch.

---

## Phase 2 — Classical Machine Learning

Áp dụng scikit-learn và các thư viện ML cổ điển vào bài toán thực tế.

| Demo | Chủ đề |
|------|--------|
| Demo1 | Bias–Variance tradeoff (polynomial regression) |
| Demo2 | Normal equation, Ridge vs Lasso |
| Demo3 | Train/validation split, data leakage |
| Demo4 | So sánh Decision Tree, Random Forest, XGBoost |
| Demo5 | Accuracy vs Precision/Recall — dữ liệu mất cân bằng |
| Demo6 | Precision-Recall curve, chọn ngưỡng tối ưu |
| Demo7 | K-Means — chọn số cụm bằng Silhouette score |
| Demo8 | K-Means vs DBSCAN — cụm phi hình cầu |
| Demo9 | Feature engineering — encoding thời gian, log transform, tương tác |
| Demo10 | Cross-validation — KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit |
| Demo11 | Hyperparameter tuning với Optuna (TPE sampler) |
| Demo12 | Average Precision Score — metric cho dữ liệu lệch |
| Demo13 | SHAP — giải thích mô hình (summary plot, waterfall plot) |

**Dataset:** Telco Customer Churn (`Phase2/WA_Fn-UseC_-Telco-Customer-Churn.csv`)

**Dự án cuối phase:** Pipeline end-to-end — preprocessing (ColumnTransformer) + LightGBM + Optuna tuning, đánh giá bằng Average Precision.

---

## Phase 3 — Deep Learning & Transformer

PyTorch từ autograd đến kiến trúc GPT mini và các kỹ thuật tối ưu GPU.

| Demo | Chủ đề |
|------|--------|
| Demo1 | PyTorch autograd — `requires_grad`, `backward`, `no_grad`, `detach` |
| Demo2 | einops — reshape tensor dễ đọc |
| Demo3 | Training loop hoàn chỉnh — DataLoader, early stopping, checkpoint |
| Demo4 | MLP — xây dựng mạng fully-connected |
| Demo5 | CNN — công thức output size convolution |
| Demo6 | Transfer learning — fine-tuning 2 giai đoạn, discriminative LR |
| Demo7 | RNN vs LSTM — vanishing gradient trên chuỗi dài |
| Demo8 | Multi-Head Attention — scaled dot-product attention |
| Demo9 | Causal mask & padding mask (GPT vs BERT) |
| Demo10 | Positional encoding — sinusoidal vs RoPE |
| Demo11 | MiniGPT — Transformer block, weight tying |
| Demo12 | Tối ưu GPU — `torch.compile`, gradient accumulation, checkpointing |

**Dự án cuối phase:** train MiniGPT trên Truyện Kiều (`kieu.txt`).

| File | Vai trò |
|------|---------|
| `mini_gpt_v2.py` | MiniGPT với RoPE (không dùng positional embedding học được) |
| `final_phase3.py` | Training loop: BPE, val split, early stopping, TensorBoard, attention map |
| `kieu.txt` | Corpus huấn luyện |
| `bpe_tokenizer.json` | Tokenizer BPE (tự tạo lần chạy đầu) |

Nâng cấp so với MiniGPT ở Demo 11: tokenizer BPE (`tokenizers`), RoPE, validation + early stopping, TensorBoard, visualize attention, so sánh temperature khi sinh văn.

```bash
python Phase3/final_phase3.py
tensorboard --logdir Phase3/runs
```

Tùy chọn: `--epochs`, `--batch-size`, `--block-size`, `--patience`, `--lr`. GPU khuyến nghị; CPU vẫn chạy được (tắt `torch.compile` và autocast). Checkpoint `best_minigpt.pt` không commit lên git. 

---

## Phase 4 — NLP & Large Language Models

Tokenizer, retrieval/RAG, LLM API, agents, fine-tuning (QLoRA/DPO), observability và LLM-as-judge.

| Demo | Chủ đề | API / yêu cầu |
|------|--------|---------------|
| Demo1 | Tokenizer GPT-2 — đo token (tiếng Việt vs tiếng Anh) | — |
| Demo2 | Tự train BPE — hiểu merge pair | — |
| Demo3 | BM25 vs embedding — hybrid search cơ bản | — |
| Demo4 | Structured output — `instructor` + Pydantic + OpenAI | OpenAI |
| Demo5 | Tool-calling agent — vòng lặp OpenAI Responses API | OpenAI |
| Demo6-openai | Prompt caching OpenAI — prefix tĩnh + handbook | OpenAI |
| Demo6-anthropic | Prompt caching Anthropic — `cache_control` | Anthropic |
| Demo7 | Prompt injection — 4 lớp phòng thủ | OpenAI |
| Demo8 | Semantic chunking — cắt theo chỗ đổi nghĩa | — |
| Demo9 | HybridRetriever — BM25 + embedding + RRF | — |
| Demo10 | Cross-encoder rerank — retrieve 20 → rerank 5 | — |
| Demo11 | RAG prompt — chỉ trả lời theo context + citation | OpenAI |
| Demo12 | Đánh giá RAG bằng RAGAS (faithfulness, recall, …) | OpenAI |
| Demo13 | MCP server tối giản — tool + resource (stdio) | — |
| Demo14 | LangGraph agent — tools, checkpointer, human-in-the-loop | OpenAI |
| Demo15 | LangGraph router — phân tuyến multi-expert | OpenAI |
| Demo16 | QLoRA SFT — fine-tune Qwen3-4B (4-bit + LoRA) | GPU |
| Demo17 | DPO — alignment không cần reward model | GPU (+ Demo16) |
| Demo18 | Langfuse tracing — observe RAG pipeline | OpenAI + Langfuse |
| Demo19 | LLM-as-judge — chấm điểm RAG + Cohen's kappa | Anthropic |
| Demo19-main | Judge + Langfuse + golden set (bản mở rộng) | OpenAI + Langfuse |

**File / thư mục kèm theo**

| Path | Vai trò |
|------|---------|
| `company_handbook.txt` | Corpus dài cho demo prompt caching / judge |
| `mcp-demo/` | Ví dụ MCP client + server |
| `train.jsonl` | Dataset fine-tune (Demo16) |
| `gen_dataset.py` | Sinh dataset train (local) |

```bash
python Phase4/Demo1.py
python Phase4/Demo12.py
python Phase4/Demo16.py   # cần GPU
python Phase4/Demo18.py   # OpenAI + Langfuse
python Phase4/Demo19.py   # Anthropic
```

---

## Phase 5 — Computer Vision

Từ xử lý ảnh cơ bản đến detection, tracking, segmentation, VLM và deploy ONNX.

| Demo | Chủ đề | Ghi chú |
|------|--------|---------|
| Demo1 | OpenCV pitfalls — BGR↔RGB, (H,W,C)→(C,H,W), letterbox | Cần `photo.jpg` |
| Demo2 | Albumentations — augment ảnh + bbox đồng bộ | — |
| Demo3 | `timm` pretrained + fine-tune 2 giai đoạn | — |
| Demo4 | IoU & NMS — tự cài đặt | — |
| Demo5 | Average Precision / mAP@0.5 | Dùng `iou` từ Demo4 |
| Demo6 | Train YOLO11 — dataset YAML, mosaic, export | Cần dataset YOLO |
| Demo7 | ByteTrack + LineZone — đếm người qua đường kẻ | Cần `traffic.mp4` |
| Demo8 | UNet + DiceBCELoss — segmentation | — |
| Demo9 | SAM (Segment Anything) — box → mask, auto-label | GPU khuyến nghị |
| Demo10 | OpenCLIP — zero-shot classify + tìm ảnh bằng text | — |
| Demo11 | Vision-Language Model (Claude) — OCR/layout bằng prompt | Anthropic |
| Demo12 | Export ONNX + benchmark vs PyTorch | GPU / onnxruntime |

**Dự án cuối phase:** `final_phase5.py` — pipeline video analytics:

`Detect (YOLO) → Track (ByteTrack) → ước tốc độ → cảnh báo vùng cấm → stream webcam`

```bash
python Phase5/final_phase5.py   # mặc định dùng webcam (VideoCapture(0))
```

---

## Phase 6 — MLOps & Triển khai Production

Đưa mô hình ML/DL từ notebook ra production.

| Chủ đề | Nội dung |
|--------|----------|
| Experiment tracking | MLflow, versioning model & data |
| Model deployment | Serving, REST API (FastAPI) |
| Cloud platforms | Triển khai trên cloud |
| Monitoring | Observability, drift detection |
| CI/CD cho ML | Pipeline automation |

**Trạng thái:** thư mục `Phase6/` đã tạo — demo code sẽ được bổ sung theo lộ trình Notion.

---

## Phase 7 — AI Engineering nâng cao

Chuyên sâu và học liên tục — cập nhật theo xu hướng mới.

| Chủ đề | Nội dung |
|--------|----------|
| Model optimization | Quantization, distillation, pruning |
| AI system design | Thiết kế hệ thống AI quy mô lớn |
| Multimodal AI | Kết hợp text, image, audio |

**Trạng thái:** thư mục `Phase7/` đã tạo — demo code sẽ được bổ sung theo lộ trình Notion.

---

## Dự án portfolio gợi ý (từ lộ trình)

| # | Dự án |
|---|-------|
| 1 | Chatbot RAG với tài liệu PDF |
| 2 | Fine-tune LLM cho classification / summarization |
| 3 | End-to-end ML pipeline (MLflow + FastAPI + Docker) |
| 4 | Image classification app với transfer learning |
| 5 | AI Agent tự động hóa tác vụ với tool calling |
| 6 | Real-time model serving với monitoring dashboard |

---

## Tech stack

| Thư viện | Mục đích |
|----------|----------|
| NumPy, Pandas, SciPy | Tính toán số, xử lý dữ liệu |
| scikit-learn | Classical ML, preprocessing, metrics |
| LightGBM, XGBoost, CatBoost | Gradient boosting |
| Optuna | Hyperparameter optimization |
| SHAP | Model explainability |
| PyTorch, torchvision, torchaudio | Deep Learning |
| tokenizers, transformers | Tokenizer / Hugging Face models |
| sentence-transformers, rank-bm25 | Embedding & lexical retrieval (Phase 4) |
| openai, instructor, pydantic | LLM API & structured output |
| anthropic | Claude API — prompt caching (Demo6-anthropic) |
| langchain, langgraph | Agents, router, human-in-the-loop |
| peft, trl, bitsandbytes, datasets | QLoRA SFT / DPO fine-tuning |
| ragas | Đánh giá pipeline RAG |
| langfuse | Observability — tracing, cost, latency |
| mcp, fastmcp | Model Context Protocol |
| opencv-python, pillow, albumentations | Xử lý ảnh & augmentation (Phase 5) |
| timm, ultralytics, supervision | Classification, YOLO, tracking |
| open-clip-torch | Zero-shot / image-text search |
| onnxruntime-gpu | Deploy & benchmark ONNX |
| dotenv | Load API keys từ `.env` |
| einops | Tensor manipulation |
| Matplotlib, Seaborn | Visualization |
| Jupyter | Notebook (Phase 2: `final_phase2.ipynb`) |
| TensorBoard | Training monitoring |

Danh sách đầy đủ trong [`ai-roadmap/pyproject.toml`](ai-roadmap/pyproject.toml).

---

## Ghi chú

- Các file demo được thiết kế **chạy độc lập**, có comment giải thích bằng tiếng Việt.
- Một số demo Phase 2/3 cần dataset hoặc GPU — kiểm tra comment đầu file trước khi chạy.
- Phase 4: nhiều demo cần `.env` (`OPENAI_API_KEY`, tùy chọn `ANTHROPIC_API_KEY`, `LANGFUSE_*`).
- Demo16/17 cần GPU (QLoRA 4-bit); Demo17 import model từ Demo16.
- Phase 5: một số demo cần file media (`photo.jpg`, `traffic.mp4`) hoặc webcam; Demo11 cần Anthropic.
- Phase 6–7 hiện mới có cấu trúc thư mục; nội dung demo sẽ được cập nhật dần theo [Notion roadmap](https://ai-engineer-roadmap.notion.site/AI-Engineer-Roadmap-L-tr-nh-h-c-y-3d10743dbce580f68924cd33d22ec22a?source=copy_link).
- Đã ignore: `ai-roadmap/.venv`, `.env`, `*/data/`, `*/__pycache__/`, `*/runs/`, `*/attention_maps/`, `Phase3/best_minigpt.pt` (xem `.gitignore`).

---

## Tác giả

**tientho201** — [tientho2012004@gmail.com](mailto:tientho2012004@gmail.com)
