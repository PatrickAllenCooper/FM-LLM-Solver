"""
Knowledge base service for FM-LLM Solver.

Handles storage and retrieval of knowledge documents for RAG.
"""

from typing import List, Optional, Dict, Any

from fm_llm_solver.core.interfaces import KnowledgeStore
from fm_llm_solver.core.types import RAGDocument
from fm_llm_solver.core.config import Config
from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.exceptions import KnowledgeBaseError


class KnowledgeBase(KnowledgeStore):
    """
    Knowledge base implementation for document storage and retrieval.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the knowledge base.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.documents = []  # Simple in-memory storage for now
        
    def add_document(self, document: RAGDocument) -> None:
        """
        Add a document to the knowledge base.
        
        Args:
            document: Document to add
        """
        self.documents.append(document)
        self.logger.info(f"Added document: {document.title}")
        
    def add_documents(self, documents: List[RAGDocument]) -> None:
        """
        Add multiple documents to the knowledge base.
        
        Args:
            documents: List of documents to add
        """
        for doc in documents:
            self.add_document(doc)
            
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RAGDocument]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            filters: Optional filters to apply
            
        Returns:
            List of relevant documents
        """
        # Simple keyword-based search for now
        # In a real implementation, this would use embeddings
        
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.documents:
            # Simple scoring based on keyword matches
            score = 0.0
            if query_lower in doc.content.lower():
                score += 0.5
            if query_lower in doc.title.lower():
                score += 0.3
                
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if key in doc.metadata and doc.metadata[key] == value:
                        score += 0.2
                        
            if score > 0:
                doc.score = score
                relevant_docs.append(doc)
        
        # Sort by score and return top k
        relevant_docs.sort(key=lambda x: x.score, reverse=True)
        return relevant_docs[:k]
    
    def get_document(self, doc_id: str) -> Optional[RAGDocument]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the knowledge base.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed, False if not found
        """
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                del self.documents[i]
                self.logger.info(f"Removed document: {doc_id}")
                return True
        return False
    
    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.documents.clear()
        self.logger.info("Cleared knowledge base")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "total_size": sum(len(doc.content) for doc in self.documents)
        } 