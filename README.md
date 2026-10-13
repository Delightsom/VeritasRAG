# VeritasRAG

Audits a generated summary against ingested source documents. Each atomic claim is retrieved independently, judged against those chunks, and scored in code.

## What it does

- Chunks sources (~180 words, 40-word overlap) and upserts them into an in-memory Chroma collection

- Deletes leftover chunks when a document is shortened or cleared

- Extracts claims with structured OpenAI output, then retrieves and judges each claim

- Rejects a `Grounded` verdict whose citation is not one of the retrieved labels

- Computes `overall_faithfulness_score` from claim statuses `1.0` / `0.5` / `0.0`)

## Stack

- Python

- ChromaDB

- sentence-transformers `all-MiniLM-L6-v2`)

- OpenAI structured outputs `gpt-4o`)

## Layout

```text

├── backend/

│   ├── rag_[engine.py](http://engine.py)

│   └── [auditor.py](http://auditor.py)

├── tests/

└── [app.py](http://app.py)