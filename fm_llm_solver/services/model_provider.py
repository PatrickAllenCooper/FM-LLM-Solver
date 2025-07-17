"""
Model provider service for FM-LLM Solver.

Handles loading and interfacing with different language models.
"""

from typing import Any, Optional

from fm_llm_solver.core.exceptions import ModelError
from fm_llm_solver.core.interfaces import ModelProvider
from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.types import ModelConfig


class BaseModelProvider(ModelProvider):
    """Base class for model providers."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.config = None

    def load_model(self, config: ModelConfig) -> None:
        """Load the model with given configuration."""
        self.config = config
        self.logger.info(f"Loading model: {config.name}")
        # Stub implementation
        self.model = "loaded"

    def unload_model(self) -> None:
        """Unload the current model."""
        self.logger.info("Unloading model")
        self.model = None

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text using the loaded model."""
        if not self.model:
            raise ModelError("Model not loaded")

        # Stub implementation
        return f"Generated response for prompt: {prompt[:50]}..."

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


class TransformersModelProvider(BaseModelProvider):
    """Base provider for Transformers-based models."""

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self, config: ModelConfig) -> None:
        """Load Transformers model."""
        try:
            # Import here to make it optional
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            self.config = config

            # Determine device
            if torch.cuda.is_available() and config.device == "cuda":
                self.device = "cuda"
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                self.logger.warning("CUDA not available, using CPU")

            # Set up quantization if requested
            quantization_config = None
            if hasattr(config, "quantization") and config.quantization:
                if config.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    self.logger.info("Using 4-bit quantization")
                elif config.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.logger.info("Using 8-bit quantization")

            # Determine trust_remote_code setting
            trust_remote_code = getattr(config, "trust_remote_code", True)

            # Load tokenizer
            self.logger.info(f"Loading tokenizer: {config.name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name, trust_remote_code=trust_remote_code, use_fast=True
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.logger.info(f"Loading model: {config.name}")
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": (
                    torch.float16 if self.device == "cuda" else torch.float32
                ),
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None

            self.model = AutoModelForCausalLM.from_pretrained(
                config.name, **model_kwargs
            )

            # Move to device if not using device_map
            if not quantization_config and model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            raise ModelError(f"Required dependencies not installed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text using the loaded model."""
        if not self.model or not self.tokenizer:
            raise ModelError("Model or tokenizer not loaded")

        try:
            # Import here to make it optional
            import torch  # type: ignore

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            return generated_text

        except ImportError as e:
            raise ModelError(f"Required dependencies not installed: {e}")
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise ModelError(f"Text generation failed: {e}")

    def get_embedding(self, text: str) -> list:
        """Get text embedding from the model."""
        # For code generation models, we typically don't use their embeddings
        # This would require a separate embedding model
        raise NotImplementedError(
            "Embedding extraction not implemented for code generation models"
        )


class QwenProvider(TransformersModelProvider):
    """Provider for Qwen models using transformers."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized Qwen provider")


class DeepSeekProvider(TransformersModelProvider):
    """Provider for DeepSeek models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized DeepSeek provider")


class StarCoderProvider(TransformersModelProvider):
    """Provider for StarCoder models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized StarCoder provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load StarCoder model with specific configurations."""
        # StarCoder models don't need trust_remote_code
        config.trust_remote_code = False
        super().load_model(config)


class CodeLlamaProvider(TransformersModelProvider):
    """Provider for CodeLlama models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized CodeLlama provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load CodeLlama model with specific configurations."""
        # CodeLlama models don't need trust_remote_code
        config.trust_remote_code = False
        super().load_model(config)


class CodestralProvider(TransformersModelProvider):
    """Provider for Codestral models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized Codestral provider")


class OpenCoderProvider(TransformersModelProvider):
    """Provider for OpenCoder models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized OpenCoder provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load OpenCoder model with specific configurations."""
        # OpenCoder models don't need trust_remote_code
        config.trust_remote_code = False
        super().load_model(config)


class CodeGemmaProvider(TransformersModelProvider):
    """Provider for CodeGemma models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized CodeGemma provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load CodeGemma model with specific configurations."""
        # CodeGemma models don't need trust_remote_code
        config.trust_remote_code = False
        super().load_model(config)


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized OpenAI provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load OpenAI model."""
        super().load_model(config)
        self.logger.info(f"Loaded OpenAI model: {config.name}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI API."""
        try:
            # Import here to make it optional
            import openai  # type: ignore

            # This would require proper OpenAI API integration
            # For now, return a placeholder
            return f"OpenAI generated response for: {prompt[:50]}..."

        except ImportError:
            raise ModelError("OpenAI package not installed")
        except Exception as e:
            raise ModelError(f"OpenAI generation failed: {e}")

    def get_embedding(self, text: str) -> list:
        """Get text embedding from OpenAI API."""
        try:
            # Import here to make it optional
            import openai  # type: ignore

            # This would require proper OpenAI API integration
            # For now, return a placeholder
            return [0.0] * 1536  # OpenAI embedding dimension

        except ImportError:
            raise ModelError("OpenAI package not installed")
        except Exception as e:
            raise ModelError(f"OpenAI embedding failed: {e}")


class ModelProviderFactory:
    """Factory for creating model providers."""

    @staticmethod
    def create(provider: str, config: ModelConfig) -> ModelProvider:
        """
        Create a model provider instance.

        Args:
            provider: Provider name
            config: Model configuration

        Returns:
            Model provider instance
        """
        provider_lower = provider.lower()

        if provider_lower == "qwen":
            return QwenProvider()
        elif provider_lower == "deepseek":
            return DeepSeekProvider()
        elif provider_lower == "starcoder":
            return StarCoderProvider()
        elif provider_lower == "codellama":
            return CodeLlamaProvider()
        elif provider_lower == "codestral":
            return CodestralProvider()
        elif provider_lower == "opencoder":
            return OpenCoderProvider()
        elif provider_lower == "codegemma":
            return CodeGemmaProvider()
        elif provider_lower == "openai":
            return OpenAIProvider()
        else:
            raise ModelError(f"Unknown provider: {provider}")

    @staticmethod
    def list_providers() -> list:
        """List available providers."""
        return [
            "qwen",
            "deepseek",
            "starcoder",
            "codellama",
            "codestral",
            "opencoder",
            "codegemma",
            "openai",
        ]

    @staticmethod
    def get_provider_info(provider: str) -> dict:
        """Get information about a specific provider."""
        provider_info = {
            "qwen": {
                "description": "Qwen family code generation models",
                "supports_quantization": True,
                "trust_remote_code": True,
                "context_lengths": [32000, 128000],
            },
            "deepseek": {
                "description": "DeepSeek Coder V2 models with MoE architecture",
                "supports_quantization": True,
                "trust_remote_code": True,
                "context_lengths": [128000],
            },
            "starcoder": {
                "description": "StarCoder family models for code completion",
                "supports_quantization": True,
                "trust_remote_code": False,
                "context_lengths": [16000],
            },
            "codellama": {
                "description": "Meta's CodeLlama models for code generation",
                "supports_quantization": True,
                "trust_remote_code": False,
                "context_lengths": [16000],
            },
            "codestral": {
                "description": "Mistral's Codestral models for multi-language coding",
                "supports_quantization": True,
                "trust_remote_code": True,
                "context_lengths": [32000],
            },
            "opencoder": {
                "description": "Open reproducible code generation models",
                "supports_quantization": True,
                "trust_remote_code": False,
                "context_lengths": [16000],
            },
            "codegemma": {
                "description": "Google's CodeGemma models for code tasks",
                "supports_quantization": True,
                "trust_remote_code": False,
                "context_lengths": [8000],
            },
            "openai": {
                "description": "OpenAI API models",
                "supports_quantization": False,
                "trust_remote_code": False,
                "context_lengths": [4000, 8000, 16000, 32000],
            },
        }

        return provider_info.get(provider.lower(), {})
