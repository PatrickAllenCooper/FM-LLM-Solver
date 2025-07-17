"""
Abstract interfaces for FM-LLM Solver components.

Defines contracts that concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from fm_llm_solver.core.types import (
    SystemDescription,
    BarrierCertificate,
    GenerationResult,
    VerificationResult,
    RAGDocument,
    ModelConfig,
    VerificationMethod,
)


class Generator(ABC):
    """Abstract interface for certificate generators."""

    @abstractmethod
    def generate(
        self, system: SystemDescription, context: Optional[List[RAGDocument]] = None, **kwargs
    ) -> GenerationResult:
        """
        Generate a barrier certificate for the given system.

        Args:
            system: System description
            context: Optional RAG context documents
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult containing the certificate or error
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the generator is ready to use."""


class Verifier(ABC):
    """Abstract interface for certificate verifiers."""

    @abstractmethod
    def verify(
        self,
        system: SystemDescription,
        certificate: BarrierCertificate,
        method: VerificationMethod = VerificationMethod.ALL,
        **kwargs,
    ) -> VerificationResult:
        """
        Verify a barrier certificate for the given system.

        Args:
            system: System description
            certificate: Barrier certificate to verify
            method: Verification method(s) to use
            **kwargs: Additional verification parameters

        Returns:
            VerificationResult with detailed check results
        """

    @abstractmethod
    def supports_method(self, method: VerificationMethod) -> bool:
        """Check if the verifier supports a given method."""


class KnowledgeStore(ABC):
    """Abstract interface for knowledge base storage."""

    @abstractmethod
    def search(
        self, query: str, k: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> List[RAGDocument]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filters: Optional metadata filters

        Returns:
            List of relevant documents
        """

    @abstractmethod
    def add_document(
        self, content: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            content: Document content
            metadata: Document metadata
            embedding: Optional pre-computed embedding

        Returns:
            Document ID
        """

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""

    @abstractmethod
    def update_document(
        self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing document."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""


class ModelProvider(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def load_model(self, config: ModelConfig) -> Any:
        """Load a language model."""

    @abstractmethod
    def generate_text(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate text from the model."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get text embedding from the model."""

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free resources."""


class Parser(ABC):
    """Abstract interface for parsing system descriptions and certificates."""

    @abstractmethod
    def parse_system(self, text: str) -> SystemDescription:
        """Parse a system description from text."""

    @abstractmethod
    def parse_certificate(self, text: str, variables: List[str]) -> BarrierCertificate:
        """Parse a barrier certificate from text."""

    @abstractmethod
    def extract_variables(self, dynamics: Dict[str, str]) -> List[str]:
        """Extract variable names from dynamics."""


class Trainer(ABC):
    """Abstract interface for model training."""

    @abstractmethod
    def train(
        self, dataset_path: str, base_model: str, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Train or fine-tune a model.

        Args:
            dataset_path: Path to training data
            base_model: Base model to fine-tune
            output_dir: Directory for saving results
            **kwargs: Additional training parameters

        Returns:
            Training metrics and results
        """

    @abstractmethod
    def evaluate(self, model_path: str, eval_dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a trained model."""


class Monitor(ABC):
    """Abstract interface for system monitoring."""

    @abstractmethod
    def log_query(self, query_log: Any) -> None:
        """Log a query."""

    @abstractmethod
    def get_metrics(
        self, time_range: Optional[Tuple[str, str]] = None, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get system metrics."""

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""


class Cache(ABC):
    """Abstract interface for caching."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
