"""
Model downloader service for FM-LLM Solver.

Handles downloading and caching of code generation models with progress tracking.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fm_llm_solver.core.exceptions import ModelError
from fm_llm_solver.core.logging import get_logger


@dataclass
class ModelDownloadInfo:
    """Information about a model download."""

    model_id: str
    provider: str
    name: str
    display_name: str
    size_gb: Optional[float] = None
    download_url: Optional[str] = None
    cache_path: Optional[str] = None
    downloaded: bool = False
    download_time: Optional[datetime] = None
    checksum: Optional[str] = None


class ModelDownloader:
    """Handles downloading and caching of code generation models."""

    def __init__(self, cache_dir: str = "models_cache"):
        """
        Initialize the model downloader.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model download info cache
        self.cache_info_file = self.cache_dir / "download_info.json"
        self.download_info: Dict[str, ModelDownloadInfo] = {}
        self._load_cache_info()

        # Progress callback
        self.progress_callback: Optional[Callable[[str, float], None]] = None

        self.logger.info(
            f"Model downloader initialized with cache dir: {self.cache_dir}"
        )

    def _load_cache_info(self) -> None:
        """Load cached download information."""
        if self.cache_info_file.exists():
            try:
                with open(self.cache_info_file, "r") as f:
                    data = json.load(f)

                self.download_info = {
                    k: ModelDownloadInfo(**v) for k, v in data.items()
                }
                self.logger.info(
                    f"Loaded {len(self.download_info)} cached model entries"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load cache info: {e}")
                self.download_info = {}

    def _save_cache_info(self) -> None:
        """Save download information to cache."""
        try:
            data = {
                k: {
                    "model_id": v.model_id,
                    "provider": v.provider,
                    "name": v.name,
                    "display_name": v.display_name,
                    "size_gb": v.size_gb,
                    "download_url": v.download_url,
                    "cache_path": v.cache_path,
                    "downloaded": v.downloaded,
                    "download_time": (
                        v.download_time.isoformat() if v.download_time else None
                    ),
                    "checksum": v.checksum,
                }
                for k, v in self.download_info.items()
            }

            with open(self.cache_info_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cache info: {e}")

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set progress callback for download tracking."""
        self.progress_callback = callback

    def _update_progress(self, model_id: str, progress: float) -> None:
        """Update download progress."""
        if self.progress_callback:
            self.progress_callback(model_id, progress)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def download_model(self, model_id: str, model_config: Dict[str, Any]) -> str:
        """
        Download a model and return the cache path.

        Args:
            model_id: Unique model identifier
            model_config: Model configuration from config.yaml

        Returns:
            Path to cached model
        """
        try:
            # Import here to make it optional
            from huggingface_hub import snapshot_download  # type: ignore

        except ImportError:
            raise ModelError(
                "huggingface_hub not installed. Please install with: pip install huggingface_hub"
            )

        # Check if already downloaded
        if model_id in self.download_info and self.download_info[model_id].downloaded:
            cache_path = self.download_info[model_id].cache_path
            if cache_path and Path(cache_path).exists():
                self.logger.info(f"Model {model_id} already cached at {cache_path}")
                return cache_path

        # Extract model information
        model_name = model_config["name"]
        provider = model_config["provider"]
        display_name = model_config["display_name"]

        self.logger.info(f"Downloading model {model_id}: {display_name}")

        # Create model-specific cache directory
        model_cache_dir = self.cache_dir / provider / model_id
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._update_progress(model_id, 0.0)

            # Download model using huggingface_hub
            cache_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(model_cache_dir),
                resume_download=True,
                local_files_only=False,
            )

            self._update_progress(model_id, 100.0)

            # Update download info
            download_info = ModelDownloadInfo(
                model_id=model_id,
                provider=provider,
                name=model_name,
                display_name=display_name,
                cache_path=cache_path,
                downloaded=True,
                download_time=datetime.now(),
            )

            self.download_info[model_id] = download_info
            self._save_cache_info()

            self.logger.info(f"Successfully downloaded {model_id} to {cache_path}")
            return cache_path

        except Exception as e:
            self.logger.error(f"Failed to download model {model_id}: {e}")
            self._update_progress(model_id, -1.0)  # Indicate error
            raise ModelError(f"Model download failed: {e}")

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        if model_id not in self.download_info:
            return False

        info = self.download_info[model_id]
        if not info.downloaded or not info.cache_path:
            return False

        return Path(info.cache_path).exists()

    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get the local path for a downloaded model."""
        if not self.is_model_downloaded(model_id):
            return None

        return self.download_info[model_id].cache_path

    def list_downloaded_models(self) -> List[ModelDownloadInfo]:
        """List all downloaded models."""
        return [
            info
            for info in self.download_info.values()
            if info.downloaded and info.cache_path and Path(info.cache_path).exists()
        ]

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a downloaded model from cache.

        Args:
            model_id: Model identifier to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        if model_id not in self.download_info:
            self.logger.warning(f"Model {model_id} not found in cache")
            return False

        info = self.download_info[model_id]
        if not info.cache_path:
            return False

        try:
            cache_path = Path(info.cache_path)
            if cache_path.exists():
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()

                self.logger.info(f"Deleted model {model_id} from {cache_path}")

            # Update download info
            info.downloaded = False
            info.cache_path = None
            self._save_cache_info()

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0

        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.exists():
                        total_size += file_path.stat().st_size

            return total_size / (1024**3)  # Convert to GB

        except Exception as e:
            self.logger.error(f"Failed to calculate cache size: {e}")
            return 0.0

    def cleanup_cache(self, max_size_gb: float = 50.0) -> None:
        """
        Clean up cache if it exceeds the maximum size.

        Args:
            max_size_gb: Maximum cache size in GB
        """
        current_size = self.get_cache_size()

        if current_size <= max_size_gb:
            return

        self.logger.info(
            f"Cache size {current_size:.2f}GB exceeds limit {max_size_gb}GB, cleaning up"
        )

        # Sort models by download time (oldest first)
        downloaded_models = [
            info
            for info in self.download_info.values()
            if info.downloaded and info.download_time
        ]

        downloaded_models.sort(key=lambda x: x.download_time or datetime.min)

        # Delete oldest models until under limit
        for info in downloaded_models:
            if self.get_cache_size() <= max_size_gb:
                break

            self.logger.info(f"Deleting old model {info.model_id} to free space")
            self.delete_model(info.model_id)

    def verify_model_integrity(self, model_id: str) -> bool:
        """
        Verify the integrity of a downloaded model.

        Args:
            model_id: Model identifier to verify

        Returns:
            True if model is valid, False otherwise
        """
        if not self.is_model_downloaded(model_id):
            return False

        info = self.download_info[model_id]
        if not info.cache_path:
            return False

        cache_path = Path(info.cache_path)

        try:
            # Check if the cache directory contains expected files
            expected_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]

            for expected_file in expected_files:
                file_path = cache_path / expected_file
                if not file_path.exists():
                    # Try alternative file names
                    if expected_file == "pytorch_model.bin":
                        # Check for safetensors or sharded models
                        safetensors_files = list(cache_path.glob("*.safetensors"))
                        model_files = list(cache_path.glob("pytorch_model-*.bin"))
                        if not safetensors_files and not model_files:
                            self.logger.warning(
                                f"No model weights found in {cache_path}"
                            )
                            return False
                    elif expected_file == "tokenizer.json":
                        # Some models might not have tokenizer.json
                        tokenizer_config = cache_path / "tokenizer_config.json"
                        if not tokenizer_config.exists():
                            self.logger.warning(
                                f"No tokenizer files found in {cache_path}"
                            )
                    else:
                        self.logger.warning(
                            f"Missing expected file {expected_file} in {cache_path}"
                        )
                        return False

            self.logger.info(f"Model {model_id} integrity verified")
            return True

        except Exception as e:
            self.logger.error(f"Failed to verify model {model_id}: {e}")
            return False

    def get_download_status(self) -> Dict[str, Dict[str, Any]]:
        """Get download status for all models."""
        status = {}

        for model_id, info in self.download_info.items():
            status[model_id] = {
                "provider": info.provider,
                "display_name": info.display_name,
                "downloaded": info.downloaded,
                "cache_path": info.cache_path,
                "download_time": (
                    info.download_time.isoformat() if info.download_time else None
                ),
                "size_gb": info.size_gb,
                "verified": (
                    self.verify_model_integrity(model_id) if info.downloaded else False
                ),
            }

        return status


# Global model downloader instance
_model_downloader: Optional[ModelDownloader] = None


def get_model_downloader(cache_dir: str = "models_cache") -> ModelDownloader:
    """Get the global model downloader instance."""
    global _model_downloader

    if _model_downloader is None:
        _model_downloader = ModelDownloader(cache_dir)

    return _model_downloader
