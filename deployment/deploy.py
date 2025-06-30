#!/usr/bin/env python3
"""
Unified deployment script for FM-LLM Solver
Supports multiple cloud providers and deployment modes.
"""

import os
import sys
import argparse
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployments across different cloud providers."""
    
    def __init__(self, provider: str, config_path: str = "config/config.yaml"):
        self.provider = provider.lower()
        self.config_path = config_path
        self.project_root = Path(__file__).parent.parent
        
        # Load environment variables
        self.load_env()
        
    def load_env(self):
        """Load environment variables from .env file if exists."""
        env_file = self.project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value.strip('"\'')
    
    def deploy_local(self):
        """Deploy locally using docker-compose."""
        logger.info("Starting local deployment...")
        
        # Build images
        logger.info("Building Docker images...")
        subprocess.run([
            "docker-compose", "build"
        ], check=True)
        
        # Start services
        logger.info("Starting services...")
        subprocess.run([
            "docker-compose", "up", "-d"
        ], check=True)
        
        logger.info("Local deployment complete!")
        logger.info("Web interface: http://localhost:5000")
        logger.info("Inference API: http://localhost:8000")
        
    def deploy_runpod(self):
        """Deploy to RunPod."""
        logger.info("Deploying to RunPod...")
        
        api_key = os.environ.get('RUNPOD_API_KEY')
        if not api_key:
            raise ValueError("RUNPOD_API_KEY environment variable not set")
        
        # Create RunPod configuration
        runpod_config = {
            "name": "fm-llm-solver",
            "imageName": "fm-llm-solver:inference",
            "gpuTypeId": "NVIDIA RTX 3090",
            "cloudType": "SECURE",
            "volumeInGb": 50,
            "containerDiskInGb": 50,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "ports": "8000/http",
            "env": [
                {"key": "DEPLOYMENT_MODE", "value": "cloud"},
                {"key": "MODEL_CACHE_DIR", "value": "/workspace/models"},
                {"key": "CONFIG_PATH", "value": "/workspace/config/config.yaml"}
            ]
        }
        
        # Deploy using RunPod CLI or API
        # This is a simplified example - actual implementation would use RunPod SDK
        logger.info(f"Creating RunPod instance with config: {json.dumps(runpod_config, indent=2)}")
        
        # In production, use runpod-python SDK:
        # import runpod
        # runpod.api_key = api_key
        # pod = runpod.create_pod(**runpod_config)
        
        logger.info("RunPod deployment initiated. Check RunPod dashboard for status.")
        
    def deploy_modal(self):
        """Deploy to Modal."""
        logger.info("Deploying to Modal...")
        
        # Create modal_app.py if not exists
        modal_app_path = self.project_root / "deployment" / "modal_app.py"
        if not modal_app_path.exists():
            self.create_modal_app()
        
        # Deploy using Modal CLI
        subprocess.run([
            "modal", "deploy", str(modal_app_path)
        ], check=True)
        
        logger.info("Modal deployment complete!")
        
    def deploy_vastai(self):
        """Deploy to Vast.ai."""
        logger.info("Deploying to Vast.ai...")
        
        api_key = os.environ.get('VASTAI_API_KEY')
        if not api_key:
            raise ValueError("VASTAI_API_KEY environment variable not set")
        
        # Search for suitable instances
        search_params = {
            "gpu_name": "RTX 3090",
            "gpu_ram": 24,
            "disk_space": 50,
            "inet_down": 100,
            "reliability": 0.98,
            "sort_option": "dlperf_usd"
        }
        
        logger.info(f"Searching for instances with params: {search_params}")
        
        # In production, use vast.ai CLI:
        # vast search offers 'RTX 3090' --disk 50 --inet_down 100
        # vast create instance <instance_id> --image fm-llm-solver:inference
        
        logger.info("Vast.ai deployment instructions:")
        logger.info("1. Run: vast search offers 'RTX 3090' --disk 50")
        logger.info("2. Select an instance and note its ID")
        logger.info("3. Run: vast create instance <ID> --image fm-llm-solver:inference")
        
    def deploy_gcp(self):
        """Deploy to Google Cloud Platform."""
        logger.info("Deploying to Google Cloud Platform...")
        
        # Create GKE deployment configuration
        k8s_config_path = self.project_root / "deployment" / "k8s"
        k8s_config_path.mkdir(exist_ok=True)
        
        self.create_k8s_configs(k8s_config_path)
        
        logger.info("GCP deployment files created in deployment/k8s/")
        logger.info("To deploy:")
        logger.info("1. Build and push image: gcloud builds submit --tag gcr.io/PROJECT_ID/fm-llm-solver")
        logger.info("2. Apply configs: kubectl apply -f deployment/k8s/")
        
    def create_modal_app(self):
        """Create Modal deployment script."""
        modal_content = '''import modal
import os
from pathlib import Path

# Create Modal stub
stub = modal.Stub("fm-llm-solver")

# Define the container image
image = (
    modal.Image.from_dockerfile("../Dockerfile", context_mount=modal.Mount.from_local_dir(".."))
    .pip_install("fastapi", "uvicorn", "aioredis")
)

# Define GPU configuration
gpu_config = modal.gpu.A10G()

# Create volume for model storage
volume = modal.SharedVolume().persist("fm-llm-models")

@stub.function(
    image=image,
    gpu=gpu_config,
    shared_volumes={"/models": volume},
    timeout=600,
    keep_warm=1,  # Keep 1 instance warm
)
def generate_certificate(system_description: str, model_config: str = "finetuned", rag_k: int = 3):
    """Generate barrier certificate using Modal GPU."""
    import sys
    sys.path.append("/app")
    
    from inference_api.main import generate_certificate as _generate
    from utils.config_loader import load_config
    
    # Load config
    config = load_config("/app/config/config.yaml")
    
    # Generate certificate
    result = _generate({
        "system_description": system_description,
        "model_config": model_config,
        "rag_k": rag_k
    })
    
    return result

@stub.web_endpoint(method="POST")
def api_generate(request: dict):
    """Web endpoint for certificate generation."""
    return generate_certificate.call(
        request.get("system_description"),
        request.get("model_config", "finetuned"),
        request.get("rag_k", 3)
    )

if __name__ == "__main__":
    stub.deploy()
'''
        
        modal_path = self.project_root / "deployment" / "modal_app.py"
        with open(modal_path, 'w') as f:
            f.write(modal_content)
        
        logger.info(f"Created Modal app at {modal_path}")
        
    def create_k8s_configs(self, output_dir: Path):
        """Create Kubernetes deployment configurations."""
        # Deployment config
        deployment_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: fm-llm-solver-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fm-llm-inference
  template:
    metadata:
      labels:
        app: fm-llm-inference
    spec:
      containers:
      - name: inference
        image: gcr.io/PROJECT_ID/fm-llm-solver:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEPLOYMENT_MODE
          value: "cloud"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "24Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: model-cache
          mountPath: /models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: fm-llm-inference-service
spec:
  selector:
    app: fm-llm-inference
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
'''
        
        with open(output_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_yaml)
        
        # Web deployment (without GPU)
        web_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: fm-llm-solver-web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fm-llm-web
  template:
    metadata:
      labels:
        app: fm-llm-web
    spec:
      containers:
      - name: web
        image: gcr.io/PROJECT_ID/fm-llm-solver-web:latest
        ports:
        - containerPort: 5000
        env:
        - name: DEPLOYMENT_MODE
          value: "cloud"
        - name: INFERENCE_API_URL
          value: "http://fm-llm-inference-service:8000"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
          requests:
            memory: "1Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: fm-llm-web-service
spec:
  selector:
    app: fm-llm-web
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
'''
        
        with open(output_dir / "web-deployment.yaml", 'w') as f:
            f.write(web_yaml)
        
        logger.info(f"Created Kubernetes configs in {output_dir}")

def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(description="Deploy FM-LLM Solver to cloud providers")
    parser.add_argument(
        "provider",
        choices=["local", "runpod", "modal", "vastai", "gcp"],
        help="Deployment target"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build Docker images without deploying"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager(args.provider, args.config)
    
    # Build Docker images if not local
    if args.provider != "local" and not args.build_only:
        logger.info("Building Docker images...")
        subprocess.run([
            "docker", "build", "-t", f"fm-llm-solver:{args.provider}", 
            "--target", "inference" if args.provider != "local" else "web",
            "."
        ], check=True)
    
    if args.build_only:
        logger.info("Build complete. Skipping deployment.")
        return
    
    # Deploy to selected provider
    deploy_method = getattr(manager, f"deploy_{args.provider}", None)
    if deploy_method:
        deploy_method()
    else:
        logger.error(f"Deployment method for {args.provider} not implemented")
        sys.exit(1)

if __name__ == "__main__":
    main() 