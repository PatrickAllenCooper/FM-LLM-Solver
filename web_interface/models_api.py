"""
Models API for web interface.

Handles model selection, downloading, and management.
"""

import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, current_app, jsonify, request  # type: ignore

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fm_llm_solver.core.exceptions import ModelError
from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.services.model_downloader import get_model_downloader
from fm_llm_solver.services.model_provider import ModelProviderFactory
from utils.config_loader import load_config

# Create blueprint
models_api = Blueprint("models_api", __name__, url_prefix="/api/models")
logger = get_logger(__name__)

# Global variables
_current_model = None
_model_provider = None
_config = None


def load_app_config():
    """Load application configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def handle_api_errors(f):
    """Decorator to handle API errors gracefully."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ModelError as e:
            logger.error(f"Model error in {f.__name__}: {e}")
            return (
                jsonify({"error": "Model Error", "message": str(e), "success": False}),
                400,
            )
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return (
                jsonify(
                    {
                        "error": "Internal Error",
                        "message": "An unexpected error occurred",
                        "success": False,
                    }
                ),
                500,
            )

    return decorated_function


@models_api.route("/available", methods=["GET"])
@handle_api_errors
def get_available_models():
    """Get list of available models with their configurations."""
    config = load_app_config()

    if "models" not in config or "available_models" not in config["models"]:
        return (
            jsonify(
                {
                    "error": "Configuration Error",
                    "message": "No models configured",
                    "success": False,
                }
            ),
            500,
        )

    # Get download status
    downloader = get_model_downloader()
    download_status = downloader.get_download_status()

    available_models = config["models"]["available_models"]

    # Add download status to each model
    models_with_status = {}
    for model_id, model_config in available_models.items():
        models_with_status[model_id] = {
            **model_config,
            "id": model_id,
            "download_status": download_status.get(
                model_id, {"downloaded": False, "downloading": False, "verified": False}
            ),
        }

    return jsonify(
        {
            "success": True,
            "models": models_with_status,
            "download_status": download_status,
            "total_models": len(available_models),
            "cache_size_gb": downloader.get_cache_size(),
        }
    )


@models_api.route("/status", methods=["GET"])
@handle_api_errors
def get_models_status():
    """Get current status of all models."""
    downloader = get_model_downloader()
    download_status = downloader.get_download_status()
    downloaded_models = downloader.list_downloaded_models()

    return jsonify(
        {
            "success": True,
            "download_status": download_status,
            "downloaded_count": len(downloaded_models),
            "cache_size_gb": downloader.get_cache_size(),
            "current_model": _current_model,
        }
    )


@models_api.route("/download/<model_id>", methods=["POST"])
@handle_api_errors
def download_model(model_id: str):
    """Download a specific model."""
    config = load_app_config()
    available_models = config["models"]["available_models"]

    if model_id not in available_models:
        return (
            jsonify(
                {
                    "error": "Model Not Found",
                    "message": f"Model {model_id} not found in configuration",
                    "success": False,
                }
            ),
            404,
        )

    downloader = get_model_downloader()

    # Check if already downloaded
    if downloader.is_model_downloaded(model_id):
        return jsonify(
            {
                "success": True,
                "message": f"Model {model_id} is already downloaded",
                "cache_path": downloader.get_model_path(model_id),
            }
        )

    try:
        # Start download
        model_config = available_models[model_id]
        cache_path = downloader.download_model(model_id, model_config)

        return jsonify(
            {
                "success": True,
                "message": f"Model {model_id} downloaded successfully",
                "cache_path": cache_path,
                "model_id": model_id,
            }
        )

    except Exception as e:
        logger.error(f"Download failed for model {model_id}: {e}")
        return (
            jsonify(
                {
                    "error": "Download Failed",
                    "message": str(e),
                    "success": False,
                    "model_id": model_id,
                }
            ),
            500,
        )


@models_api.route("/delete/<model_id>", methods=["DELETE"])
@handle_api_errors
def delete_model(model_id: str):
    """Delete a downloaded model."""
    downloader = get_model_downloader()

    if not downloader.is_model_downloaded(model_id):
        return (
            jsonify(
                {
                    "error": "Model Not Found",
                    "message": f"Model {model_id} is not downloaded",
                    "success": False,
                }
            ),
            404,
        )

    # Don't delete if it's the current model
    if _current_model == model_id:
        return (
            jsonify(
                {
                    "error": "Model In Use",
                    "message": f"Cannot delete model {model_id} as it's currently in use",
                    "success": False,
                }
            ),
            400,
        )

    success = downloader.delete_model(model_id)

    if success:
        return jsonify(
            {
                "success": True,
                "message": f"Model {model_id} deleted successfully",
                "model_id": model_id,
            }
        )
    else:
        return (
            jsonify(
                {
                    "error": "Delete Failed",
                    "message": f"Failed to delete model {model_id}",
                    "success": False,
                }
            ),
            500,
        )


@models_api.route("/select", methods=["POST"])
@handle_api_errors
def select_model():
    """Select a model for use."""
    global _current_model, _model_provider

    data = request.get_json()

    if not data or "model_id" not in data:
        return (
            jsonify(
                {
                    "error": "Invalid Request",
                    "message": "model_id is required",
                    "success": False,
                }
            ),
            400,
        )

    model_id = data["model_id"]
    config = load_app_config()
    available_models = config["models"]["available_models"]

    if model_id not in available_models:
        return (
            jsonify(
                {
                    "error": "Model Not Found",
                    "message": f"Model {model_id} not found",
                    "success": False,
                }
            ),
            404,
        )

    downloader = get_model_downloader()

    if not downloader.is_model_downloaded(model_id):
        return (
            jsonify(
                {
                    "error": "Model Not Downloaded",
                    "message": f"Model {model_id} must be downloaded first",
                    "success": False,
                }
            ),
            400,
        )

    try:
        # Unload current model if any
        if _model_provider:
            _model_provider.unload_model()
            _model_provider = None

        # Get model configuration
        model_config = available_models[model_id]
        provider = model_config["provider"]

        # Create model provider
        from fm_llm_solver.core.types import ModelConfig, ModelProvider

        # Convert string provider to enum
        provider_enum = ModelProvider(provider.upper())

        # Create model config object
        model_cfg = ModelConfig(
            provider=provider_enum,
            name=model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        # Create and load model provider
        _model_provider = ModelProviderFactory.create(provider, model_cfg)
        _model_provider.load_model(model_cfg)

        # Update current model
        _current_model = model_id

        return jsonify(
            {
                "success": True,
                "message": f"Model {model_id} selected successfully",
                "model_id": model_id,
                "display_name": model_config["display_name"],
                "provider": provider,
            }
        )

    except Exception as e:
        logger.error(f"Model selection failed for {model_id}: {e}")
        return (
            jsonify({"error": "Selection Failed", "message": str(e), "success": False}),
            500,
        )


@models_api.route("/generate", methods=["POST"])
@handle_api_errors
def generate_code():
    """Generate code using the selected model."""
    global _model_provider, _current_model

    if not _model_provider or not _current_model:
        return (
            jsonify(
                {
                    "error": "No Model Selected",
                    "message": "Please select a model first",
                    "success": False,
                }
            ),
            400,
        )

    data = request.get_json()

    if not data or "prompt" not in data:
        return (
            jsonify(
                {
                    "error": "Invalid Request",
                    "message": "prompt is required",
                    "success": False,
                }
            ),
            400,
        )

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)

    try:
        generated_text = _model_provider.generate_text(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )

        return jsonify(
            {
                "success": True,
                "generated_text": generated_text,
                "model_id": _current_model,
                "prompt_length": len(prompt),
                "generated_length": len(generated_text),
            }
        )

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return (
            jsonify(
                {"error": "Generation Failed", "message": str(e), "success": False}
            ),
            500,
        )


@models_api.route("/current", methods=["GET"])
@handle_api_errors
def get_current_model():
    """Get information about the currently selected model."""
    if not _current_model:
        return jsonify(
            {
                "success": True,
                "current_model": None,
                "message": "No model currently selected",
            }
        )

    config = load_app_config()
    available_models = config["models"]["available_models"]
    model_config = available_models.get(_current_model)

    if not model_config:
        return (
            jsonify(
                {
                    "error": "Model Configuration Error",
                    "message": "Current model not found in configuration",
                    "success": False,
                }
            ),
            500,
        )

    return jsonify(
        {
            "success": True,
            "current_model": {
                "id": _current_model,
                **model_config,
                "loaded": _model_provider is not None,
            },
        }
    )


@models_api.route("/compare", methods=["POST"])
@handle_api_errors
def compare_models():
    """Compare multiple models."""
    data = request.get_json()

    if not data or "model_ids" not in data:
        return (
            jsonify(
                {
                    "error": "Invalid Request",
                    "message": "model_ids array is required",
                    "success": False,
                }
            ),
            400,
        )

    model_ids = data["model_ids"]

    if len(model_ids) < 2:
        return (
            jsonify(
                {
                    "error": "Invalid Request",
                    "message": "At least 2 models are required for comparison",
                    "success": False,
                }
            ),
            400,
        )

    config = load_app_config()
    available_models = config["models"]["available_models"]
    downloader = get_model_downloader()

    comparison_data = []

    for model_id in model_ids:
        if model_id not in available_models:
            return (
                jsonify(
                    {
                        "error": "Model Not Found",
                        "message": f"Model {model_id} not found",
                        "success": False,
                    }
                ),
                404,
            )

        model_config = available_models[model_id]
        download_status = downloader.get_download_status().get(model_id, {})

        comparison_data.append(
            {
                "id": model_id,
                "display_name": model_config["display_name"],
                "provider": model_config["provider"],
                "parameters": model_config["parameters"],
                "context_length": model_config["context_length"],
                "specialization": model_config["specialization"],
                "recommended_gpu_memory": model_config["recommended_gpu_memory"],
                "quantization_support": model_config["quantization_support"],
                "downloaded": download_status.get("downloaded", False),
                "verified": download_status.get("verified", False),
            }
        )

    return jsonify(
        {
            "success": True,
            "comparison": comparison_data,
            "model_count": len(comparison_data),
        }
    )


@models_api.route("/cache/cleanup", methods=["POST"])
@handle_api_errors
def cleanup_cache():
    """Clean up model cache."""
    data = request.get_json() or {}
    max_size_gb = data.get("max_size_gb", 50.0)

    downloader = get_model_downloader()
    old_size = downloader.get_cache_size()

    downloader.cleanup_cache(max_size_gb)

    new_size = downloader.get_cache_size()
    freed_space = old_size - new_size

    return jsonify(
        {
            "success": True,
            "message": "Cache cleanup completed",
            "old_size_gb": round(old_size, 2),
            "new_size_gb": round(new_size, 2),
            "freed_space_gb": round(freed_space, 2),
        }
    )


@models_api.route("/verify/<model_id>", methods=["POST"])
@handle_api_errors
def verify_model(model_id: str):
    """Verify integrity of a downloaded model."""
    downloader = get_model_downloader()

    if not downloader.is_model_downloaded(model_id):
        return (
            jsonify(
                {
                    "error": "Model Not Downloaded",
                    "message": f"Model {model_id} is not downloaded",
                    "success": False,
                }
            ),
            404,
        )

    is_valid = downloader.verify_model_integrity(model_id)

    return jsonify(
        {
            "success": True,
            "model_id": model_id,
            "verified": is_valid,
            "message": f"Model {model_id} is {'valid' if is_valid else 'corrupted'}",
        }
    )


@models_api.route("/providers", methods=["GET"])
@handle_api_errors
def get_providers():
    """Get information about available model providers."""
    providers = ModelProviderFactory.list_providers()
    provider_info = {}

    for provider in providers:
        provider_info[provider] = ModelProviderFactory.get_provider_info(provider)

    return jsonify(
        {"success": True, "providers": provider_info, "available_providers": providers}
    )


# Error handlers
@models_api.errorhandler(404)
def not_found(error):
    return (
        jsonify(
            {
                "error": "Not Found",
                "message": "The requested resource was not found",
                "success": False,
            }
        ),
        404,
    )


@models_api.errorhandler(500)
def internal_error(error):
    return (
        jsonify(
            {
                "error": "Internal Server Error",
                "message": "An internal server error occurred",
                "success": False,
            }
        ),
        500,
    )
