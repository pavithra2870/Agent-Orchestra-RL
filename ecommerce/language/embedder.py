"""Text -> vector. Uses a local sentence-transformer; falls back to character TF-IDF if it cannot load.

Vectors are normalised, so a dot product is the cosine similarity (1.0 = same direction/meaning).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("ecommerce.language")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER: Optional["Embedder"] = None


class Embedder:
    name = "base"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = MODEL_NAME):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer

        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name)
        self._cache: dict = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self.model.encode(todo, normalize_embeddings=True, show_progress_bar=False)
            self._cache.update(zip(todo, vecs))
        return np.stack([self._cache[t] for t in texts]).astype(np.float32)


class TfidfEmbedder(Embedder):
    """Fallback. Fitted on the router's example sentences so it works without any download."""

    name = "tfidf-char"

    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True).fit(corpus)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = self.vec.transform(list(texts)).toarray().astype(np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(n, 1e-9)


def get_embedder(fallback_corpus: Optional[List[str]] = None, force_fallback: bool = False) -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None and not force_fallback:
        return _EMBEDDER
    if not force_fallback:
        try:
            _EMBEDDER = SentenceTransformerEmbedder()
            return _EMBEDDER
        except Exception as exc:  # no network / model missing
            log.warning("sentence-transformer unavailable (%s); using TF-IDF fallback", exc)
    from ecommerce.sim.text import PHRASES

    corpus = fallback_corpus or [p for goals in PHRASES["A"].values() for p in goals]
    _EMBEDDER = TfidfEmbedder(corpus)
    return _EMBEDDER
