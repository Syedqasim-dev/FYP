from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


@dataclass
class SearchResult:
    source: str
    chunk_id: int
    score: float
    text: str


class TfIdfIndex:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.vectors: list[list[float]] = []
        self.chunks: list[Chunk] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.vocabulary = {}
            self.idf = []
            self.vectors = []
            return

        doc_freq: Counter[str] = Counter()
        tokenized_chunks: list[list[str]] = []

        for chunk in self.chunks:
            tokens = self.tokenize(chunk.text)
            tokenized_chunks.append(tokens)
            doc_freq.update(set(tokens))

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(doc_freq))}
        total_docs = len(self.chunks)
        self.idf = [0.0] * len(self.vocabulary)

        for term, idx in self.vocabulary.items():
            # Smoothed idf to avoid divide-by-zero and keep stable for tiny corpora.
            self.idf[idx] = math.log((1 + total_docs) / (1 + doc_freq[term])) + 1

        self.vectors = [self._to_tfidf_vector(tokens) for tokens in tokenized_chunks]

    def _to_tfidf_vector(self, tokens: Sequence[str]) -> list[float]:
        if not tokens or not self.vocabulary:
            return [0.0] * len(self.vocabulary)

        counts = Counter(tokens)
        max_tf = max(counts.values())
        vector = [0.0] * len(self.vocabulary)

        for term, term_count in counts.items():
            idx = self.vocabulary.get(term)
            if idx is None:
                continue
            tf = term_count / max_tf
            vector[idx] = tf * self.idf[idx]

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def query(self, text: str, top_k: int = 3) -> list[SearchResult]:
        if not self.chunks:
            return []

        query_vector = self._to_tfidf_vector(self.tokenize(text))
        scored: list[tuple[float, Chunk]] = []

        for vector, chunk in zip(self.vectors, self.chunks):
            score = sum(v1 * v2 for v1, v2 in zip(query_vector, vector))
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[SearchResult] = []

        for score, chunk in scored[: max(top_k, 0)]:
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    score=round(score, 4),
                    text=chunk.text,
                )
            )
        return results

    def to_json(self) -> dict:
        return {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "vectors": self.vectors,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "TfIdfIndex":
        index = cls()
        index.vocabulary = {str(k): int(v) for k, v in payload.get("vocabulary", {}).items()}
        index.idf = [float(v) for v in payload.get("idf", [])]
        index.vectors = [[float(x) for x in vector] for vector in payload.get("vectors", [])]
        index.chunks = [Chunk(**chunk) for chunk in payload.get("chunks", [])]
        return index


class RAGPipeline:
    """A lightweight Retrieval-Augmented Generation pipeline using TF-IDF retrieval."""

    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 30) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = TfIdfIndex()

    @staticmethod
    def _read_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []

        for chunk_id, start in enumerate(range(0, len(words), step)):
            segment = words[start : start + self.chunk_size]
            if not segment:
                continue
            chunks.append(Chunk(source=source, chunk_id=chunk_id, text=" ".join(segment)))

        return chunks

    def ingest_paths(self, paths: Iterable[str | Path]) -> int:
        collected_chunks: list[Chunk] = []

        for input_path in paths:
            path = Path(input_path)
            if not path.exists():
                continue

            if path.is_dir():
                files = sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".txt", ".md", ".rst", ".py", ".json", ".csv"}
                )
            else:
                files = [path]

            for file in files:
                text = self._read_text_file(file)
                if not text:
                    continue
                collected_chunks.extend(self._chunk_text(text, str(file)))

        self.index.fit(collected_chunks)
        return len(collected_chunks)

    @staticmethod
    def _tokenize_query_terms(query: str) -> set[str]:
        return {token.lower() for token in TOKEN_PATTERN.findall(query)}

    def _generate_answer(self, query: str, retrieved: Sequence[SearchResult]) -> str:
        if not retrieved:
            return "I could not find enough relevant context to answer that from the indexed documents."

        query_terms = self._tokenize_query_terms(query)
        sentence_pool: list[tuple[int, str]] = []

        for item in retrieved:
            sentences = SENTENCE_SPLIT_PATTERN.split(item.text)
            for sentence in sentences:
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                sentence_terms = {token.lower() for token in TOKEN_PATTERN.findall(cleaned)}
                overlap = len(query_terms & sentence_terms)
                sentence_pool.append((overlap, cleaned))

        sentence_pool.sort(key=lambda pair: pair[0], reverse=True)

        selected: list[str] = []
        seen: set[str] = set()
        for overlap, sentence in sentence_pool:
            if overlap == 0 and selected:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            selected.append(sentence)
            seen.add(key)
            if len(selected) >= 4:
                break

        if not selected:
            selected = [retrieved[0].text]

        return " ".join(selected)

    def answer(self, query: str, top_k: int = 3) -> dict:
        results = self.index.query(query, top_k=top_k)
        answer = self._generate_answer(query, results)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "source": result.source,
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "preview": result.text[:240],
                }
                for result in results
            ],
        }

    def save_index(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.index.to_json(), indent=2), encoding="utf-8")

    def load_index(self, input_path: str | Path) -> None:
        path = Path(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.index = TfIdfIndex.from_json(payload)
