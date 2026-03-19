"""
DualRAG: Dual-Path Retrieval-Augmented Generation for RS-Agent Knowledge Space.

Algorithm (from the paper):
1. Extract keywords from user query with weights
2. Path 1 (Semantic): Dense vector similarity search
3. Path 2 (Keyword): Sparse keyword-weighted BM25 search
4. Weighted combination of both paths
5. Generate answer using retrieved context

DualRAG improves over single-path RAG by:
- Semantic path captures conceptual similarity
- Keyword path captures exact domain terminology
- Weighted combination balances both signals
"""

import logging
import re
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extract and weight keywords from a query."""

    # RS-domain specific stop words to boost
    DOMAIN_KEYWORDS = {
        "aircraft", "satellite", "radar", "sar", "optical", "sensor", "image",
        "remote", "sensing", "detection", "classification", "segmentation",
        "extraction", "resolution", "multispectral", "hyperspectral", "lidar",
        "ship", "vehicle", "building", "road", "vegetation", "cloud", "haze",
        "fighter", "bomber", "transport", "helicopter", "reconnaissance",
        "c-band", "l-band", "x-band", "sentinel", "landsat", "modis",
    }

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "able", "not", "no", "and", "or", "but", "if", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "up", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "then", "once",
        "how", "what", "when", "where", "which", "who", "why", "that", "this",
        "these", "those", "i", "me", "my", "we", "our", "you", "your", "it",
    }

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract keywords with weights.
        Domain-specific terms get higher weights.
        """
        words = re.findall(r"\b[a-z0-9\-]+\b", text.lower())
        keyword_weights: Dict[str, float] = {}

        for word in words:
            if word in self.STOP_WORDS or len(word) < 2:
                continue
            weight = 2.0 if word in self.DOMAIN_KEYWORDS else 1.0
            keyword_weights[word] = keyword_weights.get(word, 0) + weight

        # Normalize
        total = sum(keyword_weights.values())
        if total > 0:
            keyword_weights = {k: v / total for k, v in keyword_weights.items()}

        return keyword_weights


class BM25:
    """
    BM25 sparse retrieval for keyword-based search path.
    Implements BM25+ with term frequency saturation and document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, documents: List[str]):
        """Build BM25 index from document texts."""
        self.documents = [self._tokenize(doc) for doc in documents]
        N = len(self.documents)
        if N == 0:
            return
        self.avgdl = sum(len(d) for d in self.documents) / N

        df: Dict[str, int] = {}
        for doc in self.documents:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z0-9\-]+\b", text.lower())

    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for all documents."""
        query_terms = self._tokenize(query)
        scores = np.zeros(len(self.documents))

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.documents):
                tf = doc.count(term)
                dl = len(doc)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                scores[i] += idf * (numerator / denominator)

        return scores

    def get_scores_weighted(self, keyword_weights: Dict[str, float]) -> np.ndarray:
        """Compute BM25 scores with keyword weights."""
        scores = np.zeros(len(self.documents))

        for term, weight in keyword_weights.items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.documents):
                tf = doc.count(term)
                dl = len(doc)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                scores[i] += weight * idf * (numerator / denominator)

        return scores


class DualRAG:
    """
    DualRAG: Weighted dual-path retrieval for domain-specific knowledge.

    Combines:
    - Path 1 (Semantic): Dense vector similarity for conceptual relevance
    - Path 2 (Keyword): Weighted BM25 for exact term matching

    The dual-path approach is especially effective for remote sensing,
    where domain-specific terminology is critical.
    """

    def __init__(
        self,
        knowledge_db,
        embedding_model: str = "all-MiniLM-L6-v2",
        local_model_path: str = "",
        top_k_semantic: int = 5,
        top_k_keyword: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        self.knowledge_db = knowledge_db
        self.top_k_semantic = top_k_semantic
        self.top_k_keyword = top_k_keyword
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.keyword_extractor = KeywordExtractor()
        self.bm25 = BM25()
        self.semantic_model = None
        self.doc_embeddings: Optional[np.ndarray] = None

        # 폐쇄망: local_model_path가 지정되면 해당 경로에서 로드, 아니면 HuggingFace 다운로드
        model_to_load = local_model_path.strip() if local_model_path else embedding_model
        self._build_index(model_to_load)

    def _build_index(self, model_name: str):
        """Build both semantic and keyword indices."""
        docs = self.knowledge_db.get_all()
        if not docs:
            logger.warning("Knowledge database is empty. DualRAG index not built.")
            return

        # Build keyword index
        doc_texts = [f"{d.title} {d.content}" for d in docs]
        self.bm25.fit(doc_texts)
        logger.info(f"BM25 keyword index built for {len(docs)} documents")

        # Build semantic index
        try:
            import os
            from sentence_transformers import SentenceTransformer

            # 폐쇄망: local path가 지정된 경우 HuggingFace Hub 네트워크 접근을 완전 차단
            is_local_path = os.path.isdir(model_name)
            if is_local_path:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self.semantic_model = SentenceTransformer(
                    model_name, local_files_only=True
                )
            else:
                self.semantic_model = SentenceTransformer(model_name)

            self.doc_embeddings = self.semantic_model.encode(
                doc_texts, normalize_embeddings=True
            )
            logger.info(f"Semantic embedding index built for {len(docs)} documents")
        except ImportError:
            logger.warning("sentence-transformers not available. Semantic path disabled.")
        except Exception as e:
            logger.warning(f"Could not build semantic index: {e}")

    def query(self, query: str) -> Dict[str, Any]:
        """
        Execute DualRAG retrieval and return answer context.

        Algorithm:
        1. Extract weighted keywords from query
        2. Path 1: Semantic retrieval (if model available)
        3. Path 2: Weighted keyword BM25 retrieval
        4. Combine results with weighted fusion
        5. Return ranked documents as context
        """
        docs = self.knowledge_db.get_all()
        if not docs:
            return {
                "query": query,
                "answer": "Knowledge base is empty.",
                "sources": [],
                "context": "",
            }

        # Step 1: Extract weighted keywords
        keyword_weights = self.keyword_extractor.extract(query)

        # Step 2 & 3: Dual-path retrieval
        combined_scores = np.zeros(len(docs))

        # Path 2: Keyword-weighted BM25
        keyword_scores = self.bm25.get_scores_weighted(keyword_weights)
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        combined_scores += self.keyword_weight * keyword_scores

        # Path 1: Semantic similarity
        if self.semantic_model is not None and self.doc_embeddings is not None:
            query_emb = self.semantic_model.encode([query], normalize_embeddings=True)
            semantic_scores = np.dot(self.doc_embeddings, query_emb.T).flatten()
            if semantic_scores.max() > 0:
                semantic_scores = (semantic_scores - semantic_scores.min()) / (
                    semantic_scores.max() - semantic_scores.min() + 1e-8
                )
            combined_scores += self.semantic_weight * semantic_scores

        # Step 4: Rank and select top results
        top_k = max(self.top_k_semantic, self.top_k_keyword)
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        retrieved_docs = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                doc = docs[idx]
                retrieved_docs.append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "keywords": doc.keywords,
                    "score": float(combined_scores[idx]),
                })

        # Step 5: Format context
        context = self._format_context(retrieved_docs)
        answer = self._synthesize_answer(query, retrieved_docs)

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_docs,
            "context": context,
            "keywords_used": keyword_weights,
        }

    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as LLM context."""
        if not docs:
            return ""
        parts = ["[Retrieved Knowledge]\n"]
        for i, doc in enumerate(docs[:5], 1):
            parts.append(f"[{i}] {doc['title']} (Category: {doc['category']})")
            parts.append(doc["content"])
            parts.append("")
        return "\n".join(parts)

    def _synthesize_answer(self, query: str, docs: List[Dict]) -> str:
        """Synthesize a concise answer from retrieved documents."""
        if not docs:
            return f"No relevant information found for: {query}"
        top_doc = docs[0]
        return (
            f"Based on the knowledge base: {top_doc['content'][:500]}"
            f"{'...' if len(top_doc['content']) > 500 else ''}"
            f"\n\nSource: {top_doc['title']}"
        )

    def rebuild_index(self):
        """Rebuild the index after adding new documents."""
        self._build_index(
            getattr(self.semantic_model, "model_name_or_path", "all-MiniLM-L6-v2")
        )
