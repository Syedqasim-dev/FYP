# RAG Module (From Scratch)

This module adds a lightweight Retrieval-Augmented Generation (RAG) pipeline to the project.

## What it includes

- Document ingestion from files/directories (`.txt`, `.md`, `.rst`, `.py`, `.json`, `.csv`)
- Word-based chunking with overlap
- TF-IDF vector index built from scratch (no external vector DB)
- Cosine-similarity retrieval
- Extractive answer generation grounded in retrieved chunks
- Index save/load support (`JSON`)
- CLI for indexing and querying

## Quick start

From repository root:

```bash
python -m rag_module.cli index --inputs "Facial module" Audio_module --output rag_module/index.json
python -m rag_module.cli ask --index rag_module/index.json --query "How is facial empathy score calculated?" --pretty
```

## Programmatic usage

```python
from rag_module import RAGPipeline

rag = RAGPipeline(chunk_size=120, chunk_overlap=30)
rag.ingest_paths(["Facial module", "Audio_module"])
rag.save_index("rag_module/index.json")

rag.load_index("rag_module/index.json")
result = rag.answer("How is facial empathy score calculated?", top_k=3)
print(result["answer"])
```
