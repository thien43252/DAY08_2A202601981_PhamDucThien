# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**

- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**

```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework                                           | Cài đặt               | Đặc điểm                                      |
| --------------------------------------------------- | ------------------------ | ------------------------------------------------- |
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas`    | Chuẩn industry cho RAG eval, 3 trục chính      |
| [TruLens](https://github.com/truera/trulens)         | `pip install trulens`  | Dashboard UI, feedback functions mạnh            |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

**Sơ đồ luồng xử lý chi tiết (ASCII Flowchart):**

```
+-----------------------------------------------------------------------------------+
|                            DATA INGESTION & PIPELINE                              |
+-----------------------------------------------------------------------------------+
|  [PDF/DOCX Legal Docs]  +  [News JSON/HTML]                                       |
|            │                     │                                                |
|            ▼                     ▼                                                |
|     (Task 1 & Task 2)  ──>  [MarkItDown Conversion] (Task 3)                      |
|                                  │                                                |
|                                  ▼                                                |
|                   [data/standardized/*.md Files]                                  |
+----------------------------------┬------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------------------+
|                            INDEXING & STORAGE MODULE                              |
+-----------------------------------------------------------------------------------+
|   [RecursiveCharacterTextSplitter] (chunk_size=800, overlap=100)                  |
|            ├───────────────────────────────┐                                      |
|            ▼                               ▼                                      |
|   [BAAI/bge-m3 Embeddings]        [BM25 Tokenizer]                                |
|            │                               │                                      |
|            ▼                               ▼                                      |
|  [(ChromaDB Vector Store)]       [BM25 Lexical Corpus Index]                      |
+------------┬───────────────────────────────┬--------------------------------------+
             │                               │
             └───────────────────────┬───────┘
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                        HYBRID RETRIEVAL & RERANKING                               |
+-----------------------------------------------------------------------------------+
|  User Query ─────────┬───────────────────────────────┐                            |
|                      ▼                               ▼                            |
|            (Task 5: Semantic Search)      (Task 6: Lexical Search BM25)           |
|                      │                               │                            |
|                      └───────────────┬───────────────┘                            |
|                                      ▼                                            |
|                        (Task 7: RRF Hybrid Fusion / Reranker)                     |
|                                      │                                            |
|                                      ▼                                            |
|                         [Top Similarity Score Check]                              |
|                          /                        \                               |
|              (Score >= Threshold)            (Score < Threshold)                  |
|                        │                              │                           |
|                        ▼                              ▼                           |
|               [Hybrid Candidate Chunks]     (Task 8: PageIndex Fallback)          |
|                        │                              │                           |
|                        └───────────────┬──────────────┘                           |
+--------------------------------────────┼------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                      GENERATION & USER INTERFACE                                  |
+-----------------------------------------------------------------------------------+
|  (Task 10: Reorder Chunks for 'Lost in the Middle' Mitigation)                    |
|                        │                                                          |
|                        ▼                                                          |
|  [Prompt Engineering + System Instructions for Citations]                         |
|                        │                                                          |
|                        ▼                                                          |
|  [LLM Generation (OpenRouter / Gemini API)]                                       |
|                        │                                                          |
|                        ▼                                                          |
|  [Streamlit Chatbot UI (app.py)] <──> [RAGAS Evaluation (results.md)]             |
+-----------------------------------------------------------------------------------+
```

---

## Phân Công Công Việc

| Thành viên                   | MSSV        | Nhiệm vụ                                                                                                 | Trạng thái                  |
| :----------------------------- | :---------- | :--------------------------------------------------------------------------------------------------------- | :---------------------------- |
| **Trần Công Chiến**   | 2A202601981 | Điều phối tiến độ, ghép code tổng hợp (`supervisor.py` & Task 9).                               | **Hoàn thành (100%)** |
| **Phạm Khắc Duy**      | 2A202600002 | Tạo`golden_dataset.json` (15 câu hỏi), thực thi RAGAS `eval_pipeline.py` và viết `results.md`. | **Hoàn thành (100%)** |
| **Nguyễn Ngọc Thuận** | 2A202600003 | Phụ trách thu thập, chuẩn hoá dữ liệu (Task 1–3) và xây dựng ChromaDB (Task 4–5).              | **Hoàn thành (100%)** |
| **Phạm Đức Thiện**   | 2A202600004 | Xây dựng giao diện Streamlit`app.py` và nối LLM Generation (Task 10).                               | **Hoàn thành (100%)** |

---

## Hướng Dẫn Chạy

### 1. Môi Trường & Dependencies

```bash
# Khởi tạo môi trường ảo Python
python -m venv .venv
# Kích hoạt trên Windows PowerShell:
.venv\Scripts\Activate.ps1

# Cài đặt tất cả thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu Hình API Key

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Điền các API Key trong file `.env`:

- `OPENROUTER_API_KEY`: Dùng cho mô hình LLM Generation
- `JINA_API_KEY`: (Tùy chọn) Dùng cho Reranking
- `PAGEINDEX_API_KEY`: (Tùy chọn) Dùng cho Vectorless Fallback

### 3. Chạy Chatbot Streamlit (Yêu cầu 1)

```bash
streamlit run app.py
```

### 4. Chạy Đánh Giá RAG Evaluation (Yêu cầu 2)

```bash
# Chạy đánh giá offline (BM25 vs TF-IDF A/B Testing):
python group_project/evaluation/eval_pipeline_offline.py

# Chạy đánh giá qua RAGAS / DeepEval API:
python group_project/evaluation/eval_pipeline_ragas.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
