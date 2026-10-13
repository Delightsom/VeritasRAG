* **`backend/rag_engine.py`**:
```python
import chromadb
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("veritas_docs")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_text(self, text, chunk_size=180, overlap=40):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def ingest_document(self, base_id: str, text: str):
        # Delete existing chunks for this doc to prevent stale data
        existing = self.collection.get(where={"source_doc": base_id})
        if existing and existing['ids']:
            self.collection.delete(ids=existing['ids'])

        chunks = self.chunk_text(text)
        if not chunks:
            return

        ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        metadatas = [{"source_doc": base_id, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def retrieve(self, query: str, n_results: int = 3):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results