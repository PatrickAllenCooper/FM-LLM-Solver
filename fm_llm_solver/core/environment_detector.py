#!/usr/bin/env python3
"""
Environment Detection Module for FM-LLM Solver.

This module detects the current testing environment and its capabilities
to enable adaptive testing strategies.

Environment Types:
1. MacBook (local development, no GPU)
2. Desktop (local development, with GPU)  
3. Deployed (production/staging environment)
"""

import os
import sys
import platform
import subprocess
import psutil
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """Detects current environment and its capabilities."""
    
    def __init__(self):
        self.environment_info = {}
        self._detect_environment()
    
    def _detect_environment(self):
        """Detect comprehensive environment information."""
        self.environment_info = {
            "environment_type": self._detect_environment_type(),
            "hardware": self._detect_hardware(),
            "software": self._detect_software(),
            "network": self._detect_network(),
            "storage": self._detect_storage(),
            "testing_capabilities": {}
        }
        
        # Determine testing capabilities based on detected environment
        self.environment_info["testing_capabilities"] = self._determine_testing_capabilities()
    
    def _detect_environment_type(self) -> str:
        """
        Detect the primary environment type.
        
        Returns:
            str: 'macbook', 'desktop', or 'deployed'
        """
        # Check for deployment indicators first
        deployment_indicators = [
            os.getenv("KUBERNETES_SERVICE_HOST"),  # Running in Kubernetes
            os.getenv("DOCKER_CONTAINER"),         # Running in Docker
            os.getenv("CI"),                       # Running in CI/CD
            os.getenv("GITHUB_ACTIONS"),           # GitHub Actions
            os.getenv("JENKINS_URL"),              # Jenkins
            os.getenv("HEROKU_APP_NAME"),          # Heroku
            os.getenv("AWS_EXECUTION_ENV"),        # AWS Lambda/ECS
            os.getenv("AZURE_CLIENT_ID"),          # Azure
            os.getenv("GOOGLE_CLOUD_PROJECT"),     # Google Cloud
            os.getenv("VERCEL"),                   # Vercel
            os.getenv("NETLIFY"),                  # Netlify
            os.getenv("RAILWAY_ENVIRONMENT"),      # Railway
            os.getenv("RENDER"),                   # Render
        ]
        
        if any(deployment_indicators):
            return "deployed"
        
        # Check for cloud instance metadata
        if self._detect_cloud_environment():
            return "deployed"
        
        # Check for headless/server environment
        if self._is_headless_server():
            return "deployed"
        
        # Detect MacBook vs Desktop
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin":  # macOS
            if machine == "arm64":  # Apple Silicon
                return "macbook"
            else:  # Intel Mac (could be MacBook or iMac/Mac Pro)
                # Additional heuristics for Intel Macs
                try:
                    # Check for laptop battery
                    result = subprocess.run(
                        ["system_profiler", "SPPowerDataType"],
                        capture_output=True, text=True, timeout=10
                    )
                    if "Battery" in result.stdout:
                        return "macbook"
                    else:
                        return "desktop"  # iMac, Mac Pro, Mac Studio
                except:
                    # Default to macbook for Intel Macs if detection fails
                    return "macbook"
        
        # For non-macOS systems, use hardware heuristics
        gpu_info = self._detect_gpu()
        cpu_cores = os.cpu_count() or 1
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Desktop indicators: high-end GPU, many cores, lots of RAM
        is_desktop = (
            gpu_info["has_cuda_gpu"] or
            gpu_info["gpu_memory_gb"] > 8 or
            cpu_cores >= 16 or
            total_ram_gb >= 32
        )
        
        return "desktop" if is_desktop else "macbook"
    
    def _is_headless_server(self) -> bool:
        """Check if running in a headless server environment."""
        # Check for display
        if os.getenv("DISPLAY") is None and platform.system() == "Linux":
            return True
        
        # Check for SSH connection
        if os.getenv("SSH_CLIENT") or os.getenv("SSH_CONNECTION"):
            return True
        
        # Check for typical server hostnames
        hostname = socket.gethostname().lower()
        server_indicators = ["server", "node", "worker", "compute", "instance"]
        if any(indicator in hostname for indicator in server_indicators):
            return True
        
        return False
    
    def _detect_cloud_environment(self) -> bool:
        """Check if running in a cloud environment by detecting cloud metadata."""
        # Check for AWS EC2
        try:
            import urllib.request
            urllib.request.urlopen("http://169.254.169.254/latest/meta-data/", timeout=2)
            return True  # AWS EC2 instance
        except:
            pass
        
        # Check for Google Cloud metadata
        try:
            import urllib.request
            urllib.request.urlopen("http://metadata.google.internal/", timeout=2)
            return True  # Google Cloud instance
        except:
            pass
        
        # Check for Azure metadata
        try:
            import urllib.request
            req = urllib.request.Request("http://169.254.169.254/metadata/instance?api-version=2021-02-01")
            req.add_header("Metadata", "true")
            urllib.request.urlopen(req, timeout=2)
            return True  # Azure instance
        except:
            pass
        
        # Check for common cloud hostnames
        hostname = socket.gethostname().lower()
        cloud_patterns = [
            "ip-",           # AWS private hostnames
            "compute-",      # Google Cloud
            "vm-",           # Generic VM
            ".internal",     # Cloud internal domains
            "ec2-",          # AWS
            "gcp-",          # Google Cloud Platform
            "azure-",        # Azure
        ]
        
        if any(pattern in hostname for pattern in cloud_patterns):
            return True
        
        # Check for virtualization
        if self._is_virtualized():
            # If virtualized and has specific cloud indicators, likely deployed
            cloud_files = [
                "/sys/class/dmi/id/product_name",
                "/sys/class/dmi/id/sys_vendor"
            ]
            
            for cloud_file in cloud_files:
                if os.path.exists(cloud_file):
                    try:
                        with open(cloud_file, 'r') as f:
                            content = f.read().lower()
                            if any(vendor in content for vendor in ["amazon", "google", "microsoft", "digitalocean", "linode"]):
                                return True
                    except:
                        pass
        
        return False
    
    def _is_virtualized(self) -> bool:
        """Check if running in a virtualized environment."""
        # Check for common virtualization indicators
        virt_indicators = [
            "/proc/vz",           # OpenVZ
            "/proc/xen",          # Xen
            "/sys/bus/pci/devices/0000:00:01.1/vendor",  # VMware
        ]
        
        for indicator in virt_indicators:
            if os.path.exists(indicator):
                return True
        
        # Check systemd-detect-virt if available
        try:
            result = subprocess.run(
                ["systemd-detect-virt"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip() != "none":
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return False
    
    def _detect_hardware(self) -> Dict:
        """Detect hardware specifications."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_cores": os.cpu_count() or 1,
            "cpu_cores_physical": psutil.cpu_count(logical=False) or 1,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu": self._detect_gpu(),
            "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64"
        }
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities."""
        gpu_info = {
            "has_cuda_gpu": False,
            "has_mps": False,  # Metal Performance Shaders (Apple)
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory_gb": 0,
            "cuda_version": None,
            "torch_cuda_available": False
        }
        
        # Try to detect CUDA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info["has_cuda_gpu"] = len(lines) > 0 and lines[0].strip()
                gpu_info["gpu_count"] = len(lines)
                
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory_mb = float(parts[1].strip())
                            gpu_info["gpu_names"].append(name)
                            gpu_info["gpu_memory_gb"] = max(
                                gpu_info["gpu_memory_gb"], 
                                memory_mb / 1024
                            )
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
        
        # Check PyTorch CUDA availability
        try:
            import torch
            gpu_info["torch_cuda_available"] = torch.cuda.is_available()
            if gpu_info["torch_cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                if not gpu_info["has_cuda_gpu"]:  # nvidia-smi failed but torch sees GPU
                    gpu_info["has_cuda_gpu"] = True
                    gpu_info["gpu_count"] = torch.cuda.device_count()
                    for i in range(gpu_info["gpu_count"]):
                        gpu_info["gpu_names"].append(torch.cuda.get_device_name(i))
                        props = torch.cuda.get_device_properties(i)
                        gpu_info["gpu_memory_gb"] = max(
                            gpu_info["gpu_memory_gb"],
                            props.total_memory / (1024**3)
                        )
            
            # Check for MPS (Apple Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["has_mps"] = True
        except ImportError:
            pass
        
        return gpu_info
    
    def _detect_software(self) -> Dict:
        """Detect software environment."""
        software_info = {
            "python_version": platform.python_version(),
            "is_conda_env": "CONDA_DEFAULT_ENV" in os.environ,
            "is_virtual_env": hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            ),
            "environment_name": os.getenv("CONDA_DEFAULT_ENV") or os.getenv("VIRTUAL_ENV"),
            "packages": self._detect_key_packages()
        }
        
        return software_info
    
    def _detect_key_packages(self) -> Dict:
        """Detect availability of key packages."""
        packages = {
            "torch": False,
            "torch_version": None,
            "transformers": False,
            "flask": False,
            "pytest": False,
            "docker": False,
            "kubernetes": False
        }
        
        try:
            import torch
            packages["torch"] = True
            packages["torch_version"] = torch.__version__
        except ImportError:
            pass
        
        for package in ["transformers", "flask", "pytest"]:
            try:
                __import__(package)
                packages[package] = True
            except ImportError:
                pass
        
        # Check for Docker
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, timeout=5
            )
            packages["docker"] = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check for Kubernetes
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client"], 
                capture_output=True, timeout=5
            )
            packages["kubernetes"] = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return packages
    
    def _detect_network(self) -> Dict:
        """Detect network environment."""
        network_info = {
            "hostname": socket.gethostname(),
            "is_localhost": True,
            "external_connectivity": self._check_external_connectivity()
        }
        
        # Check if we're on localhost or have external IP
        try:
            # Connect to a remote server to get our external-facing IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                network_info["local_ip"] = local_ip
                network_info["is_localhost"] = local_ip.startswith("127.") or local_ip.startswith("192.168.")
        except Exception:
            network_info["local_ip"] = "unknown"
        
        return network_info
    
    def _check_external_connectivity(self) -> bool:
        """Check if we have external internet connectivity."""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "8.8.8.8"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Try using socket for Windows compatibility
                socket.create_connection(("8.8.8.8", 53), timeout=5).close()
                return True
            except (socket.error, OSError):
                return False
    
    def _detect_storage(self) -> Dict:
        """Detect storage information."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            disk_usage = psutil.disk_usage(str(project_root))
            return {
                "project_root": str(project_root),
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "used_gb": disk_usage.used / (1024**3)
            }
        except Exception:
            return {
                "project_root": str(project_root),
                "total_gb": "unknown",
                "free_gb": "unknown", 
                "used_gb": "unknown"
            }
    
    def _determine_testing_capabilities(self) -> Dict:
        """Determine what testing capabilities are available."""
        env_type = self.environment_info["environment_type"]
        hardware = self.environment_info["hardware"]
        software = self.environment_info["software"]
        
        capabilities = {
            "can_run_unit_tests": True,  # Always available
            "can_run_integration_tests": True,  # Always available
            "can_run_performance_tests": True,  # Always available
            "can_run_gpu_tests": hardware["gpu"]["has_cuda_gpu"] or hardware["gpu"]["has_mps"],
            "can_run_load_tests": env_type in ["desktop", "deployed"],
            "can_run_security_tests": True,  # Always available
            "can_run_deployment_tests": software["packages"]["docker"],
            "can_run_end_to_end_tests": env_type in ["desktop", "deployed"],
            "recommended_test_scope": self._get_recommended_scope(env_type, hardware),
            "max_parallel_jobs": self._get_max_parallel_jobs(env_type, hardware),
            "memory_constraints": self._get_memory_constraints(env_type, hardware),
            "timeout_multiplier": self._get_timeout_multiplier(env_type, hardware)
        }
        
        return capabilities
    
    def _get_recommended_scope(self, env_type: str, hardware: Dict) -> str:
        """Get recommended test scope based on environment."""
        if env_type == "macbook":
            return "essential"  # Run core tests quickly
        elif env_type == "desktop":
            return "comprehensive"  # Run all tests including GPU-intensive ones
        else:  # deployed
            return "production"  # Run production-focused tests
    
    def _get_max_parallel_jobs(self, env_type: str, hardware: Dict) -> int:
        """Get maximum recommended parallel test jobs."""
        cpu_cores = hardware["cpu_cores"]
        memory_gb = hardware["memory_total_gb"]
        
        if env_type == "macbook":
            # Conservative for laptops to avoid overheating/battery drain
            return min(cpu_cores // 2, 4)
        elif env_type == "desktop":
            # More aggressive for desktops
            return min(cpu_cores - 1, 8)
        else:  # deployed
            # Very conservative in production environments
            return min(cpu_cores // 4, 2)
    
    def _get_memory_constraints(self, env_type: str, hardware: Dict) -> Dict:
        """Get memory constraints for testing."""
        memory_gb = hardware["memory_total_gb"]
        
        if env_type == "macbook":
            return {
                "max_memory_per_test_gb": min(memory_gb * 0.3, 4),  # 30% or 4GB max
                "enable_memory_monitoring": True,
                "oom_protection": True
            }
        elif env_type == "desktop":
            return {
                "max_memory_per_test_gb": min(memory_gb * 0.5, 16),  # 50% or 16GB max
                "enable_memory_monitoring": False,
                "oom_protection": False
            }
        else:  # deployed
            return {
                "max_memory_per_test_gb": min(memory_gb * 0.2, 2),  # 20% or 2GB max
                "enable_memory_monitoring": True,
                "oom_protection": True
            }
    
    def _get_timeout_multiplier(self, env_type: str, hardware: Dict) -> float:
        """Get timeout multiplier based on expected performance."""
        gpu_available = hardware["gpu"]["has_cuda_gpu"] or hardware["gpu"]["has_mps"]
        
        if env_type == "macbook":
            return 3.0 if not gpu_available else 2.0  # Slower without GPU
        elif env_type == "desktop":
            return 1.0 if gpu_available else 1.5  # Baseline or slightly slower
        else:  # deployed
            return 2.0  # Conservative in production
    
    def get_environment_type(self) -> str:
        """Get the detected environment type."""
        return self.environment_info["environment_type"]
    
    def get_testing_capabilities(self) -> Dict:
        """Get testing capabilities for this environment."""
        return self.environment_info["testing_capabilities"]
    
    def get_full_info(self) -> Dict:
        """Get complete environment information."""
        return self.environment_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for testing."""
        return self.environment_info["testing_capabilities"]["can_run_gpu_tests"]
    
    def should_run_comprehensive_tests(self) -> bool:
        """Check if comprehensive tests should be run."""
        scope = self.environment_info["testing_capabilities"]["recommended_test_scope"]
        return scope in ["comprehensive", "production"]
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the environment."""
        info = self.environment_info
        env_type = info["environment_type"]
        hardware = info["hardware"]
        capabilities = info["testing_capabilities"]
        
        gpu_desc = "No GPU"
        if hardware["gpu"]["has_cuda_gpu"]:
            gpu_names = ", ".join(hardware["gpu"]["gpu_names"][:2])  # Show first 2 GPUs
            gpu_desc = f"CUDA GPU: {gpu_names} ({hardware['gpu']['gpu_memory_gb']:.1f}GB)"
        elif hardware["gpu"]["has_mps"]:
            gpu_desc = "Apple Metal Performance Shaders"
        
        return (
            f"Environment: {env_type.title()} | "
            f"CPU: {hardware['cpu_cores']} cores | "
            f"RAM: {hardware['memory_total_gb']:.1f}GB | "
            f"{gpu_desc} | "
            f"Scope: {capabilities['recommended_test_scope']} | "
            f"Parallel: {capabilities['max_parallel_jobs']} jobs"
        )


# Global detector instance
_detector = None

def get_environment_detector() -> EnvironmentDetector:
    """Get the global environment detector instance."""
    global _detector
    if _detector is None:
        _detector = EnvironmentDetector()
    return _detector

def detect_environment() -> str:
    """Quick function to get environment type."""
    return get_environment_detector().get_environment_type()

def get_testing_capabilities() -> Dict:
    """Quick function to get testing capabilities."""
    return get_environment_detector().get_testing_capabilities() 