"""
Knowledge Database for RS-Agent Knowledge Space.

Stores domain-specific remote sensing knowledge documents for DualRAG retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class KnowledgeDocument:
    """A single knowledge document entry."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str,
        keywords: List[str] = None,
        metadata: Dict = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.category = category
        self.keywords = keywords or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeDocument":
        return cls(
            doc_id=data.get("doc_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
        )


class KnowledgeDatabase:
    """
    Database storing domain-specific remote sensing knowledge.

    Documents are organized by category:
    - aircraft: Aircraft specifications and recognition features
    - sensors: Satellite sensors and imaging systems
    - techniques: Remote sensing methods and algorithms
    - targets: Ground object characteristics
    - applications: Remote sensing application domains
    """

    def __init__(self, data_dir: str = "data/knowledge"):
        self.data_dir = data_dir
        self.documents: List[KnowledgeDocument] = []
        self._load_documents()

    def _load_documents(self):
        """Load knowledge documents from JSON files."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Knowledge directory created: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.data_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        entries = data if isinstance(data, list) else [data]
                        for entry in entries:
                            self.documents.append(KnowledgeDocument.from_dict(entry))
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(self.documents)} knowledge documents from {self.data_dir}")

    def add_document(self, doc: KnowledgeDocument):
        """Add a document to the knowledge base."""
        self.documents.append(doc)

    def get_all(self) -> List[KnowledgeDocument]:
        """Return all knowledge documents."""
        return self.documents

    def get_by_category(self, category: str) -> List[KnowledgeDocument]:
        """Return documents in a specific category."""
        return [d for d in self.documents if d.category == category]

    def get_by_keyword(self, keyword: str) -> List[KnowledgeDocument]:
        """Return documents containing a specific keyword."""
        keyword_lower = keyword.lower()
        return [
            d for d in self.documents
            if any(k.lower() == keyword_lower for k in d.keywords)
        ]

    def __len__(self):
        return len(self.documents)
