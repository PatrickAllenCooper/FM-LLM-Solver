"""
Certificate generator service for FM-LLM Solver.

Handles generation of barrier certificates using LLMs with RAG.
"""

import time
from typing import Optional, List, Dict, Any

from fm_llm_solver.core.interfaces import Generator, KnowledgeStore, ModelProvider, Cache
from fm_llm_solver.core.types import (
    SystemDescription,
    BarrierCertificate,
    GenerationResult,
    RAGDocument,
)
from fm_llm_solver.core.exceptions import GenerationError, ModelError
from fm_llm_solver.core.logging import get_logger, log_performance
from fm_llm_solver.core.config import Config
from fm_llm_solver.services.parser import SystemParser
from fm_llm_solver.services.prompt_builder import PromptBuilder


class CertificateGenerator(Generator):
    """
    Main certificate generator implementation.

    Generates barrier certificates using LLMs enhanced with RAG.
    """

    def __init__(
        self,
        config: Config,
        model_provider: ModelProvider,
        knowledge_store: Optional[KnowledgeStore] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Initialize the certificate generator.

        Args:
            config: Configuration object
            model_provider: Model provider instance
            knowledge_store: Optional knowledge store for RAG
            cache: Optional cache service
        """
        self.config = config
        self.model_provider = model_provider
        self.knowledge_store = knowledge_store
        self.cache = cache
        self.parser = SystemParser()
        self.prompt_builder = PromptBuilder()
        self.logger = get_logger(__name__)

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the language model."""
        try:
            self.logger.info(f"Loading model: {self.config.model.name}")
            self.model_provider.load_model(self.config.model)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}", model_name=self.config.model.name)

    @log_performance(get_logger(__name__), "certificate_generation")
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
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._get_cache_key(system)
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    self.logger.info("Retrieved certificate from cache")
                    return GenerationResult(**cached)

            # Retrieve context if not provided
            if context is None and self.knowledge_store and self.config.rag.enabled:
                context = self._retrieve_context(system)

            # Build prompt
            prompt = self.prompt_builder.build_generation_prompt(
                system=system, context=context, examples=kwargs.get("examples", [])
            )

            # Generate certificate
            self.logger.info("Generating certificate")
            response = self.model_provider.generate_text(
                prompt=prompt,
                max_tokens=self.config.model.max_new_tokens,
                temperature=kwargs.get("temperature", self.config.model.temperature),
                top_p=kwargs.get("top_p", self.config.model.top_p),
            )

            # Parse certificate
            try:
                certificate = self.parser.parse_certificate(
                    response, self.parser.extract_variables(system.dynamics)
                )
            except Exception as e:
                self.logger.error(f"Failed to parse certificate: {e}")
                raise GenerationError(f"Failed to parse certificate: {e}")

            # Create result
            result = GenerationResult(
                certificate=certificate,
                confidence=self._estimate_confidence(certificate, context),
                rag_context=[doc.__dict__ for doc in (context or [])],
                generation_time=time.time() - start_time,
                model_name=self.config.model.name,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "system_type": system.system_type.value,
                },
            )

            # Cache result
            if self.cache and result.success:
                self.cache.set(cache_key, result.__dict__, ttl=3600)

            self.logger.info(f"Certificate generated successfully in {result.generation_time:.2f}s")
            return result

        except GenerationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during generation: {e}")
            return GenerationResult(
                certificate=None,
                confidence=0.0,
                error=str(e),
                generation_time=time.time() - start_time,
                model_name=self.config.model.name,
            )

    def _retrieve_context(self, system: SystemDescription) -> List[RAGDocument]:
        """Retrieve relevant context from knowledge store."""
        if not self.knowledge_store:
            return []

        query = system.to_text()
        self.logger.info(f"Retrieving context for query: {query[:100]}...")

        try:
            documents = self.knowledge_store.search(
                query=query,
                k=self.config.rag.k_retrieved,
                filters={"system_type": system.system_type.value},
            )

            # Filter by minimum score
            documents = [doc for doc in documents if doc.score >= self.config.rag.min_score]

            self.logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents

        except Exception as e:
            self.logger.warning(f"Failed to retrieve context: {e}")
            return []

    def _estimate_confidence(
        self, certificate: BarrierCertificate, context: Optional[List[RAGDocument]]
    ) -> float:
        """Estimate confidence in the generated certificate."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on context
        if context:
            avg_score = sum(doc.score for doc in context) / len(context)
            confidence += 0.3 * avg_score

        # Increase confidence based on certificate complexity
        if len(certificate.variables) > 2:
            confidence += 0.1

        # Cap at 0.95 (never 100% confident without verification)
        return min(confidence, 0.95)

    def _get_cache_key(self, system: SystemDescription) -> str:
        """Generate cache key for a system."""
        import hashlib
        import json

        # Create deterministic representation
        system_dict = {
            "dynamics": sorted(system.dynamics.items()),
            "initial_set": system.initial_set,
            "unsafe_set": system.unsafe_set,
            "system_type": system.system_type.value,
        }

        system_str = json.dumps(system_dict, sort_keys=True)
        return f"cert:{hashlib.sha256(system_str.encode()).hexdigest()}"

    def is_ready(self) -> bool:
        """Check if the generator is ready to use."""
        try:
            # Check if model is loaded
            if not hasattr(self, "model_provider"):
                return False

            # Test with a simple generation
            test_result = self.model_provider.generate_text("Test", max_tokens=1)

            return test_result is not None

        except Exception as e:
            self.logger.warning(f"Generator not ready: {e}")
            return False

    def __del__(self):
        """Cleanup when generator is destroyed."""
        if hasattr(self, "model_provider"):
            try:
                self.model_provider.unload_model()
            except Exception as e:
                self.logger.warning(f"Failed to unload model: {e}")
