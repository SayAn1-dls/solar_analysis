"""Retrieval helpers for the grid-guideline knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunk_text(text: str, chunk_size: int = 550, overlap: int = 110) -> List[str]:
    clean = " ".join(text.split())
    if len(clean) <= chunk_size:
        return [clean]

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def build_retrieval_query(question: str, forecast_summary: Dict[str, Any], risk_analysis: Dict[str, Any]) -> str:
    deficit_hours = ", ".join(str(hour) for hour in forecast_summary.get("deficit_hours", [])) or "none"
    surplus_hours = ", ".join(str(hour) for hour in forecast_summary.get("surplus_hours", [])) or "none"
    risk_level = risk_analysis.get("risk_level", "pending")
    return (
        f"{question.strip()} solar grid optimization battery dispatch load shifting reserve margin "
        f"risk level {risk_level} deficit hours {deficit_hours} surplus hours {surplus_hours}"
    )


class GuidelineRetriever:
    """Small local retriever with optional FAISS acceleration."""

    def __init__(self, docs_dir: str = "knowledge_base/grid_guidelines") -> None:
        self.docs_dir = Path(docs_dir)
        self.chunks = self._load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk["content"] for chunk in self.chunks])
        self.backend = "tfidf-cosine"
        self.index = None

        try:
            import faiss  # type: ignore

            dense_matrix = self.matrix.astype(np.float32).toarray()
            faiss.normalize_L2(dense_matrix)
            index = faiss.IndexFlatIP(dense_matrix.shape[1])
            index.add(dense_matrix)
            self.index = index
            self.backend = "faiss"
        except Exception:
            self.index = None

    def _load_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for path in sorted(self.docs_dir.glob("*.md")):
            title = path.stem.replace("_", " ").title()
            text = path.read_text(encoding="utf-8")
            for index, chunk in enumerate(_chunk_text(text), start=1):
                chunks.append(
                    {
                        "id": f"{path.stem}-chunk-{index}",
                        "title": title,
                        "source": str(path),
                        "content": chunk,
                    }
                )
        if not chunks:
            raise FileNotFoundError(f"No knowledge-base documents found in {self.docs_dir}")
        return chunks

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])

        if self.backend == "faiss" and self.index is not None:
            import faiss  # type: ignore

            dense_query = query_vector.astype(np.float32).toarray()
            faiss.normalize_L2(dense_query)
            scores, indices = self.index.search(dense_query, top_k)
            ranked = [
                (int(indices[0][i]), float(scores[0][i]))
                for i in range(len(indices[0]))
                if int(indices[0][i]) >= 0
            ]
        else:
            similarities = cosine_similarity(query_vector, self.matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            ranked = [(int(index), float(similarities[index])) for index in top_indices]

        results: List[Dict[str, Any]] = []
        for index, score in ranked:
            chunk = dict(self.chunks[index])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results
