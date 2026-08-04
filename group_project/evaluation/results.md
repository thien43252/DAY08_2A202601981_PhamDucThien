# RAG Evaluation Results

## Run Settings

- Backend: fallback_local
- Sample count: 5
- Full dataset enabled: False
- Pipeline passthrough enabled: False

## Overall Scores

| Metric | Score |
|---|---:|
| faithfulness | 0.017 |
| answer_relevancy | 0.000 |
| context_recall | 0.992 |
| context_precision | 0.022 |

## A/B Comparison

| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |
|---|---:|---:|---:|---:|
| bm25 | 0.017 | 0.000 | 0.992 | 0.022 |
| tfidf | 0.025 | 0.000 | 0.992 | 0.024 |

## Worst Performers

- Sinh viên phải trả học phí theo hình thức nào?
  - Faithfulness: 0.017
  - Answer relevance: 0.000
  - Context recall: 0.958
  - Context precision: 0.012
- Sinh viên nội trú có những quyền gì?
  - Faithfulness: 0.006
  - Answer relevance: 0.000
  - Context recall: 1.000
  - Context precision: 0.010
- Mỗi học phần có khối lượng bao nhiêu tín chỉ?
  - Faithfulness: 0.019
  - Answer relevance: 0.000
  - Context recall: 1.000
  - Context precision: 0.015
- Học bổng khuyến khích học tập dành cho sinh viên đại học hệ chính quy có tiêu chí gì?
  - Faithfulness: 0.022
  - Answer relevance: 0.000
  - Context recall: 1.000
  - Context precision: 0.035
- Ký túc xá của ĐHQGHN có những khu nào và khu nào gần UET?
  - Faithfulness: 0.021
  - Answer relevance: 0.000
  - Context recall: 1.000
  - Context precision: 0.039

## Recommendations

- Giữ sample size nhỏ khi demo nếu dùng OpenRouter free để tránh rate limit.
- Nếu cần chạy full dataset, tăng dần `RAGAS_SAMPLE_LIMIT` sau khi xác nhận quota.
- Context precision thấp thường là tín hiệu cần chỉnh retrieval trước generation.
