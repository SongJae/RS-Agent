"""
Task-Aware Retrieval for RS-Agent Solution Space.

Retrieves relevant expert solutions based on user task descriptions,
enabling accurate tool selection and step-by-step task decomposition.

Algorithm:
1. Encode user query using sentence transformer
2. Compute cosine similarity against solution embeddings
3. Retrieve top-k most relevant solutions
4. Format retrieved solutions as context for the LLM
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TaskAwareRetrieval:
    """
    Retrieves expert solutions relevant to a given remote sensing task.

    Uses semantic similarity to match user queries with stored solution templates,
    providing the Central Controller with expert guidance for tool selection
    and task planning.
    """

    def __init__(
        self,
        solution_db,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.3,
    ):
        self.solution_db = solution_db
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self._init_embeddings(embedding_model)

    def _init_embeddings(self, model_name: str):
        """Initialize the sentence transformer and pre-compute solution embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._build_index()
            logger.info(f"Task-Aware Retrieval initialized with {len(self.solution_db)} solutions")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to keyword matching."
            )
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}. Using keyword fallback.")

    def _build_index(self):
        """Build the embedding index from the solution database."""
        solutions = self.solution_db.get_all()
        if not solutions:
            logger.warning("Solution database is empty.")
            return

        texts = [
            f"{s.get('task_type', '')} {s.get('task_description', '')}"
            for s in solutions
        ]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        logger.info(f"Built embedding index for {len(solutions)} solutions")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant solutions for the given query.

        Args:
            query: User's task description

        Returns:
            List of relevant solution dicts, ordered by relevance
        """
        solutions = self.solution_db.get_all()
        if not solutions:
            return []

        if self.model is not None and self.embeddings is not None:
            return self._semantic_retrieve(query, solutions)
        else:
            return self._keyword_retrieve(query, solutions)

    def _semantic_retrieve(self, query: str, solutions: List[Dict]) -> List[Dict]:
        """Semantic similarity-based retrieval."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()

        top_indices = np.argsort(similarities)[::-1][: self.top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                solution = solutions[idx].copy()
                solution["_score"] = float(similarities[idx])
                results.append(solution)
        return results

    def _keyword_retrieve(self, query: str, solutions: List[Dict]) -> List[Dict]:
        """Simple keyword-based retrieval as fallback."""
        query_words = set(query.lower().split())
        scored = []
        for sol in solutions:
            text = f"{sol.get('task_type', '')} {sol.get('task_description', '')}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, sol))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, sol in scored[: self.top_k]:
            sol = sol.copy()
            sol["_score"] = score
            results.append(sol)
        return results

    def format_as_context(self, solutions: List[Dict]) -> str:
        """
        Format retrieved solutions as a context string for the LLM.

        This forms the core of Task-Aware Retrieval — providing the
        Central Controller with expert guidance.
        """
        if not solutions:
            return ""

        lines = ["[Expert Solutions Retrieved]\n"]
        for i, sol in enumerate(solutions, 1):
            lines.append(f"Solution {i}: {sol.get('task_type', 'Unknown Task')}")
            lines.append(f"  Description: {sol.get('task_description', '')}")
            lines.append(f"  Tools Required: {', '.join(sol.get('tools_used', []))}")
            lines.append("  Steps:")
            for step in sol.get("solution_steps", []):
                lines.append(f"    - {step}")
            if sol.get("tips"):
                lines.append(f"  Expert Tips: {sol['tips']}")
            lines.append("")
        return "\n".join(lines)

    def get_relevant_tools(self, query: str) -> List[str]:
        """Return list of tool names relevant to the query."""
        solutions = self.retrieve(query)
        tools = set()
        for sol in solutions:
            tools.update(sol.get("tools_used", []))
        return list(tools)

    def rebuild_index(self):
        """Rebuild embedding index after adding new solutions."""
        if self.model is not None:
            self._build_index()
