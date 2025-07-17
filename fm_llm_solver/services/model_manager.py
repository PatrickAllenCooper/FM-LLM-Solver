"""
Model Manager for dynamic model switching in FM-LLM Solver.

Handles model loading, switching, and state management.
"""

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fm_llm_solver.core.exceptions import ModelError
from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.types import ModelConfig
from fm_llm_solver.core.types import ModelProvider as ModelProviderEnum
from fm_llm_solver.services.model_downloader import get_model_downloader
from fm_llm_solver.services.model_provider import ModelProvider, ModelProviderFactory
from utils.config_loader import load_config


@dataclass
class ModelState:
    """Current state of a loaded model."""

    model_id: str
    provider: str
    display_name: str
    loaded_at: datetime
    is_active: bool
    memory_usage_mb: Optional[float] = None
    last_used: Optional[datetime] = None


class ModelManager:
    """Manages model loading, switching, and state."""

    def __init__(self):
        """Initialize the model manager."""
        self.logger = get_logger(__name__)
        self._lock = threading.RLock()

        # Current model state
        self._current_model: Optional[str] = None
        self._model_provider: Optional[ModelProvider] = None
        self._model_states: Dict[str, ModelState] = {}

        # Configuration
        self._config = None
        self._available_models: Dict[str, Dict[str, Any]] = {}

        # Load configuration
        self._load_config()

        self.logger.info("Model Manager initialized")

    def _load_config(self):
        """Load model configuration."""
        try:
            self._config = load_config()

            if (
                "models" in self._config
                and "available_models" in self._config["models"]
            ):
                self._available_models = self._config["models"]["available_models"]
                self.logger.info(
                    f"Loaded {len(self._available_models)} model configurations"
                )
            else:
                self.logger.warning("No model configurations found in config")

        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise ModelError(f"Configuration loading failed: {e}")

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models."""
        return self._available_models.copy()

    def get_current_model(self) -> Optional[str]:
        """Get the currently active model ID."""
        with self._lock:
            return self._current_model

    def get_model_state(self, model_id: str) -> Optional[ModelState]:
        """Get state information for a specific model."""
        with self._lock:
            return self._model_states.get(model_id)

    def get_all_model_states(self) -> Dict[str, ModelState]:
        """Get state information for all models."""
        with self._lock:
            return self._model_states.copy()

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded and ready for use."""
        if model_id not in self._available_models:
            return False

        downloader = get_model_downloader()
        return downloader.is_model_downloaded(model_id)

    def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a specific model.

        Args:
            model_id: Model identifier to load
            force_reload: Force reload even if already loaded

        Returns:
            True if successfully loaded, False otherwise
        """
        with self._lock:
            try:
                # Check if model exists in configuration
                if model_id not in self._available_models:
                    raise ModelError(f"Model {model_id} not found in configuration")

                # Check if model is downloaded
                if not self.is_model_downloaded(model_id):
                    raise ModelError(f"Model {model_id} is not downloaded")

                # Check if already loaded and not forcing reload
                if not force_reload and model_id in self._model_states:
                    state = self._model_states[model_id]
                    if state.is_active:
                        self.logger.info(f"Model {model_id} is already loaded")
                        return True

                # Unload current model if different
                if self._current_model and self._current_model != model_id:
                    self._unload_current_model()

                # Load new model
                model_config = self._available_models[model_id]
                provider_name = model_config["provider"]

                self.logger.info(
                    f"Loading model {model_id} with provider {provider_name}"
                )

                # Create model configuration object
                provider_enum = ModelProviderEnum(provider_name.upper())

                model_cfg = ModelConfig(
                    provider=provider_enum,
                    name=model_config["name"],
                    trust_remote_code=model_config.get("trust_remote_code", True),
                    device=model_config.get("device", "cuda"),
                    quantization=model_config.get("quantization", None),
                )

                # Create and load model provider
                self._model_provider = ModelProviderFactory.create(
                    provider_name, model_cfg
                )
                self._model_provider.load_model(model_cfg)

                # Update state
                self._current_model = model_id
                self._model_states[model_id] = ModelState(
                    model_id=model_id,
                    provider=provider_name,
                    display_name=model_config["display_name"],
                    loaded_at=datetime.now(),
                    is_active=True,
                    last_used=datetime.now(),
                )

                # Deactivate other models
                for other_id, state in self._model_states.items():
                    if other_id != model_id:
                        state.is_active = False

                self.logger.info(f"Successfully loaded model {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {e}")
                # Clean up on failure
                self._model_provider = None
                self._current_model = None
                if model_id in self._model_states:
                    del self._model_states[model_id]
                raise ModelError(f"Model loading failed: {e}")

    def _unload_current_model(self):
        """Unload the currently loaded model."""
        if self._model_provider and self._current_model:
            try:
                self.logger.info(f"Unloading model {self._current_model}")
                self._model_provider.unload_model()

                # Update state
                if self._current_model in self._model_states:
                    self._model_states[self._current_model].is_active = False

            except Exception as e:
                self.logger.warning(f"Error unloading model {self._current_model}: {e}")
            finally:
                self._model_provider = None
                self._current_model = None

    def unload_model(self, model_id: Optional[str] = None):
        """
        Unload a specific model or the current model.

        Args:
            model_id: Model to unload (if None, unload current model)
        """
        with self._lock:
            if model_id is None:
                model_id = self._current_model

            if not model_id:
                self.logger.warning("No model to unload")
                return

            if model_id == self._current_model:
                self._unload_current_model()

            # Remove from states
            if model_id in self._model_states:
                del self._model_states[model_id]

            self.logger.info(f"Unloaded model {model_id}")

    def switch_model(self, model_id: str) -> bool:
        """
        Switch to a different model.

        Args:
            model_id: Model identifier to switch to

        Returns:
            True if successfully switched, False otherwise
        """
        try:
            if model_id == self._current_model:
                self.logger.info(f"Model {model_id} is already active")
                return True

            self.logger.info(f"Switching from {self._current_model} to {model_id}")
            success = self.load_model(model_id)

            if success:
                self.logger.info(f"Successfully switched to model {model_id}")

            return success

        except Exception as e:
            self.logger.error(f"Model switching failed: {e}")
            return False

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the current model.

        Args:
            prompt: Text prompt for generation
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        with self._lock:
            if not self._model_provider or not self._current_model:
                raise ModelError("No model currently loaded")

            try:
                # Update last used timestamp
                if self._current_model in self._model_states:
                    self._model_states[self._current_model].last_used = datetime.now()

                # Generate text
                result = self._model_provider.generate_text(prompt, **kwargs)

                return result

            except Exception as e:
                self.logger.error(f"Text generation failed: {e}")
                raise ModelError(f"Text generation failed: {e}")

    def get_model_info(
        self, model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.

        Args:
            model_id: Model identifier (if None, use current model)

        Returns:
            Model information dictionary
        """
        with self._lock:
            if model_id is None:
                model_id = self._current_model

            if not model_id or model_id not in self._available_models:
                return None

            model_config = self._available_models[model_id]
            state = self._model_states.get(model_id)

            return {
                "id": model_id,
                "config": model_config,
                "state": state,
                "is_downloaded": self.is_model_downloaded(model_id),
                "is_current": model_id == self._current_model,
            }

    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        with self._lock:
            return [
                model_id
                for model_id, state in self._model_states.items()
                if state.is_active
            ]

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for loaded models."""
        # This would require integration with system monitoring
        # For now, return placeholder data
        return {
            "total_memory_mb": 0.0,
            "available_memory_mb": 0.0,
            "model_memory_usage": {
                model_id: state.memory_usage_mb or 0.0
                for model_id, state in self._model_states.items()
            },
        }

    def cleanup_inactive_models(self, max_inactive_time_minutes: int = 30):
        """
        Clean up models that have been inactive for too long.

        Args:
            max_inactive_time_minutes: Maximum inactive time before cleanup
        """
        with self._lock:
            current_time = datetime.now()
            models_to_remove = []

            for model_id, state in self._model_states.items():
                if (
                    not state.is_active
                    and state.last_used
                    and (current_time - state.last_used).total_seconds()
                    > max_inactive_time_minutes * 60
                ):
                    models_to_remove.append(model_id)

            for model_id in models_to_remove:
                self.logger.info(f"Cleaning up inactive model {model_id}")
                self.unload_model(model_id)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the model manager."""
        with self._lock:
            return {
                "current_model": self._current_model,
                "loaded_models": len(self._model_states),
                "available_models": len(self._available_models),
                "model_states": {
                    model_id: {
                        "provider": state.provider,
                        "display_name": state.display_name,
                        "loaded_at": state.loaded_at.isoformat(),
                        "is_active": state.is_active,
                        "last_used": (
                            state.last_used.isoformat() if state.last_used else None
                        ),
                    }
                    for model_id, state in self._model_states.items()
                },
                "memory_usage": self.get_memory_usage(),
            }


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager


def switch_model(model_id: str) -> bool:
    """Convenience function to switch models."""
    manager = get_model_manager()
    return manager.switch_model(model_id)


def generate_text(prompt: str, **kwargs) -> str:
    """Convenience function to generate text with current model."""
    manager = get_model_manager()
    return manager.generate_text(prompt, **kwargs)


def get_current_model() -> Optional[str]:
    """Convenience function to get current model ID."""
    manager = get_model_manager()
    return manager.get_current_model()
