"""
Model provider service for FM-LLM Solver.

Handles loading and interfacing with different language models.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from fm_llm_solver.core.interfaces import ModelProvider
from fm_llm_solver.core.types import ModelConfig
from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.exceptions import ModelError


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


class QwenProvider(BaseModelProvider):
    """Provider for Qwen models using transformers."""

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.device = None
        self.logger.info("Initialized Qwen provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load Qwen model."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

            # Load tokenizer
            self.logger.info(f"Loading tokenizer: {config.name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name, trust_remote_code=True, use_fast=True
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.logger.info(f"Loading model: {config.name}")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None

            self.model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)

            # Move to device if not using device_map
            if not quantization_config and model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            raise ModelError(f"Required dependencies not installed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load Qwen model: {e}")
            raise ModelError(f"Failed to load model: {e}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text using the loaded Qwen model."""
        if not self.model or not self.tokenizer:
            raise ModelError("Model not loaded")

        try:
            import torch

            # Tokenize input
            # Use reasonable max_length, ensuring we have space for generation
            max_total_length = getattr(
                self.config, "max_length", getattr(self.config, "max_tokens", 4096)
            )
            max_input_length = max(
                512, max_total_length - max_tokens
            )  # Ensure at least 512 for input

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=False,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                # Ensure we have valid input
                if inputs["input_ids"].shape[1] == 0:
                    raise ModelError("Empty input after tokenization")

                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.7,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs,
                )

            # Decode response (excluding input)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise ModelError(f"Text generation failed: {e}")

    def get_embedding(self, text: str) -> list:
        """Get text embedding from the model."""
        # Qwen is primarily a generative model, so embeddings would need special handling
        # For now, return a placeholder
        self.logger.warning("Embedding generation not implemented for Qwen")
        return [0.0] * 768  # Placeholder embedding

    def unload_model(self) -> None:
        """Unload the current model."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache")
        except Exception:
            pass

        self.logger.info("Model unloaded")


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI models."""

    def __init__(self):
        super().__init__()
        self.logger.info("Initialized OpenAI provider")

    def load_model(self, config: ModelConfig) -> None:
        """Load OpenAI model."""
        super().load_model(config)
        self.logger.info(f"Loaded OpenAI model: {config.name}")


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
        if provider.lower() == "qwen":
            return QwenProvider()
        elif provider.lower() == "openai":
            return OpenAIProvider()
        else:
            raise ModelError(f"Unknown provider: {provider}")

    @staticmethod
    def list_providers() -> list:
        """List available providers."""
        return ["qwen", "openai"]
