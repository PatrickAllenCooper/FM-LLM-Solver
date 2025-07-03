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
    Knowledge base implementation for document storage and retrieval using FAISS.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the knowledge base.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.index = None
        self.documents = []
        self.embedding_model = None
        self.metadata = {}
        
        # Initialize if path exists
        if hasattr(config.paths, 'kb_output_dir') and config.paths.kb_output_dir:
            self._load_index()
        
    def _load_index(self):
        """Load FAISS index and documents."""
        try:
            import faiss
            import json
            from pathlib import Path
            
            kb_dir = Path(self.config.paths.kb_output_dir)
            index_path = kb_dir / "knowledge_base.faiss"
            metadata_path = kb_dir / "metadata.json"
            
            if not index_path.exists():
                self.logger.warning(f"Knowledge base index not found at {index_path}")
                return
            
            # Load FAISS index
            self.logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                self.documents = self.metadata.get('chunks', [])
                self.logger.info(f"Loaded {len(self.documents)} document chunks")
            
            # Initialize embedding model
            self._load_embedding_model()
            
            self.logger.info("Knowledge base loaded successfully")
            
        except ImportError:
            self.logger.error("FAISS not available - install with: pip install faiss-cpu")
            raise KnowledgeBaseError("FAISS not available")
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            raise KnowledgeBaseError(f"Failed to load knowledge base: {e}")
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.metadata.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info(f"Loading embedding model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Embedding model loaded successfully")
            
        except ImportError:
            self.logger.error("sentence-transformers not available")
            raise KnowledgeBaseError("sentence-transformers not available")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise KnowledgeBaseError(f"Failed to load embedding model: {e}")
        
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
        Search for relevant documents using FAISS.
        
        Args:
            query: Search query
            k: Number of documents to return
            filters: Optional filters to apply
            
        Returns:
            List of relevant documents
        """
        if not self.index or not self.embedding_model:
            self.logger.warning("Knowledge base not properly loaded, falling back to keyword search")
            return self._keyword_search(query, k, filters)
        
        try:
            import numpy as np
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert results to RAGDocument objects
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                if idx < len(self.documents):
                    doc_metadata = self.documents[idx]
                    
                    # Apply filters if provided
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if doc_metadata.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    # Create RAGDocument
                    doc = RAGDocument(
                        id=str(idx),
                        content=f"Document chunk {idx}",  # Would need to store actual content
                        title=doc_metadata.get('source', 'Unknown'),
                        metadata=doc_metadata,
                        score=float(score)
                    )
                    results.append(doc)
            
            self.logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}, falling back to keyword search")
            return self._keyword_search(query, k, filters)
    
    def _keyword_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RAGDocument]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        relevant_docs = []
        
        for i, doc_metadata in enumerate(self.documents):
            # Simple scoring based on keyword matches
            score = 0.0
            content = str(doc_metadata)  # Simple string conversion
            
            if query_lower in content.lower():
                score += 0.5
                
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if doc_metadata.get(key) == value:
                        score += 0.2
                        
            if score > 0:
                doc = RAGDocument(
                    id=str(i),
                    content=f"Document chunk {i}",
                    title=doc_metadata.get('source', 'Unknown'),
                    metadata=doc_metadata,
                    score=score
                )
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