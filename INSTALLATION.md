# 🛠️ FM-LLM Solver Installation Guide

**Complete setup instructions for FM-LLM Solver with GCP + Modal hybrid deployment support.**

---

## 📋 **System Requirements**

### **💻 Hardware Requirements**

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 4 cores | 8+ cores | Intel/AMD, modern architecture |
| **RAM** | 8GB | 16GB+ | 32GB for large models |
| **GPU** | Optional | NVIDIA RTX 3070+ | 6GB+ VRAM, CUDA support |
| **Storage** | 20GB | 50GB+ | SSD recommended |

### **🖥️ Operating System Support**

- ✅ **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- ✅ **macOS**: 10.15+ (Intel/Apple Silicon)
- ✅ **Windows**: 10/11 with WSL2 recommended

### **⚙️ Software Prerequisites**

- **Python**: 3.8+ (3.10 recommended)
- **Git**: Latest version
- **Docker**: 20.10+ (optional but recommended)
- **CUDA**: 11.7+ (for GPU support)

---

## 🚀 **Quick Installation (5 Minutes)**

### **🎯 Method 1: Unified Command (Recommended)**

```bash
# Clone repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Check system compatibility
./fm-llm status

# Automatic setup with environment detection
./fm-llm setup --auto

# Start web interface
./fm-llm start web
```

**✨ Done!** Access at http://localhost:5000

### **🐳 Method 2: Docker (Zero Configuration)**

```bash
# Clone repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Web interface only (CPU)
docker build --target web -t fm-llm:web .
docker run -p 5000:5000 fm-llm:web

# Full stack with GPU
docker build --target production -t fm-llm:full .
docker run --gpus all -p 5000:5000 -p 8000:8000 fm-llm:full

# Development environment
docker-compose --profile development up
```

---

## 🔧 **Manual Installation**

### **Step 1: Repository Setup**

```bash
# Clone the repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Verify repository structure
ls -la fm-llm config/ requirements/
```

### **Step 2: Python Environment**

#### **Option A: Conda (Recommended)**
```bash
# Create environment
conda create -n fmllm python=3.10
conda activate fmllm

# Verify Python version
python --version  # Should show Python 3.10.x
```

#### **Option B: venv**
```bash
# Create virtual environment
python -m venv fmllm-env

# Activate environment
source fmllm-env/bin/activate  # Linux/macOS
# or
fmllm-env\Scripts\activate     # Windows
```

### **Step 3: Dependencies Installation**

#### **🎯 Modular Installation (Choose Your Use Case)**

```bash
# Base installation (always required)
pip install -r requirements/base.txt

# Choose additional components:

# Web interface only
pip install -r requirements/web.txt

# ML/Inference capabilities  
pip install -r requirements/inference.txt

# Development tools
pip install -r requirements/dev.txt

# Production deployment
pip install -r requirements/production.txt

# Everything (development + ML)
pip install -r requirements/web.txt -r requirements/inference.txt -r requirements/dev.txt
```

#### **⚡ GPU Support (NVIDIA)**

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### **🍎 Apple Silicon (M1/M2/M3)**

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### **Step 4: Configuration Setup**

#### **Environment Variables**

```bash
# Copy environment template
cp config/env.example .env

# Edit .env file with your settings
nano .env  # or your preferred editor
```

**Required Variables:**
```bash
# Basic configuration
FM_LLM_ENV=development  # development, staging, production
SECRET_KEY=your-32-character-secret-key-here

# Database (optional for development)
DATABASE_URL=sqlite:///instance/fmllm.db

# External APIs (optional)
MATHPIX_APP_ID=your_mathpix_app_id
MATHPIX_APP_KEY=your_mathpix_app_key
UNPAYWALL_EMAIL=your-email@example.com
SEMANTIC_SCHOLAR_API_KEY=your_api_key
```

#### **Configuration Validation**

```bash
# Test configuration loading
./fm-llm status

# Validate all components
./fm-llm test --quick
```

---

## 🌐 **Deployment Options**

### **🏠 Local Development**

```bash
# Set development environment
export FM_LLM_ENV=development

# Start development server with auto-reload
./fm-llm start web --debug

# Full local stack (web + inference)
./fm-llm start full --debug
```

**Access Points:**
- Web Interface: http://localhost:5000
- API Documentation: http://localhost:5000/docs
- Health Check: http://localhost:5000/health

### **☁️ Hybrid Cloud (GCP + Modal)**

#### **Prerequisites**

```bash
# Install cloud dependencies
pip install -r requirements/production.txt

# Install Modal CLI
pip install modal

# Install Google Cloud SDK
# Follow: https://cloud.google.com/sdk/docs/install
```

#### **Authentication Setup**

```bash
# Modal authentication
modal token new

# Google Cloud authentication
gcloud auth login
gcloud config set project your-project-id
```

#### **Deploy Hybrid Architecture**

```bash
# Set production environment
export FM_LLM_ENV=production
export DEPLOYMENT_MODE=hybrid

# Deploy to hybrid cloud
./fm-llm deploy hybrid

# Check deployment status
./fm-llm status
```

**What gets deployed:**
- **GCP Kubernetes**: Web interface, PostgreSQL, Redis ($3-20/month)
- **Modal Serverless**: GPU inference with auto-scaling ($0-30/month)
- **Total Cost**: 80-95% savings vs dedicated GPU

### **🐳 Docker Deployment**

#### **Single Container**

```bash
# Build target-specific images
docker build --target web -t fm-llm:web .           # Web only
docker build --target inference -t fm-llm:inference . # Inference only
docker build --target production -t fm-llm:full .   # Full stack

# Run with appropriate configuration
docker run -p 5000:5000 \
  -e FM_LLM_ENV=production \
  -e SECRET_KEY=your-secret-key \
  fm-llm:web
```

#### **Docker Compose Orchestration**

```bash
# Development environment
docker-compose --profile development up

# Hybrid deployment (local web + services)
docker-compose --profile hybrid up

# Full production stack
docker-compose --profile production up

# Just supporting services
docker-compose --profile services up
```

---

## 🔍 **Verification & Testing**

### **🎯 Installation Verification**

```bash
# Quick system check
./fm-llm test --quick

# Expected output:
# ✅ Configuration System
# ✅ Core Services  
# ✅ Web Interface
# 🎮 GPU: [Available/Not available]
# 💻 Environment: [laptop/desktop/minimal]
```

### **🧪 Component Testing**

```bash
# Test individual components
./fm-llm test --unit           # Core functionality
./fm-llm test --integration    # Component integration
./fm-llm test --gpu            # GPU capabilities (if available)
./fm-llm test --performance    # Performance benchmarks

# Comprehensive test suite
./fm-llm test --all
```

### **🌐 Web Interface Verification**

```bash
# Start web interface
./fm-llm start web

# Test endpoints
curl http://localhost:5000/health
curl http://localhost:5000/api/status
```

**Manual Testing:**
1. Visit http://localhost:5000
2. Register/login user account
3. Try certificate generation
4. Check conversation mode

---

## 🚨 **Troubleshooting**

### **🐍 Python Issues**

```bash
# Check Python version
python --version  # Must be 3.8+

# Check virtual environment
which python
which pip

# Fix common issues
pip install --upgrade pip setuptools wheel
```

### **🎮 GPU Issues**

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Common fixes
# 1. Reinstall PyTorch with correct CUDA version
# 2. Update NVIDIA drivers
# 3. Check CUDA_VISIBLE_DEVICES environment variable
```

### **🔧 Configuration Issues**

```bash
# Check configuration loading
./fm-llm status

# Validate specific environment
FM_LLM_ENV=production ./fm-llm status

# Reset configuration
rm -f .env
cp config/env.example .env
# Edit .env with correct values
```

### **🐳 Docker Issues**

```bash
# Check Docker installation
docker --version
docker-compose --version

# GPU support in Docker
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi

# Build issues
docker system prune  # Clean up
docker build --no-cache --target web .
```

### **🌐 Network Issues**

```bash
# Check port availability
netstat -tulpn | grep :5000
lsof -i :5000

# Test API connectivity
curl -I http://localhost:5000/health

# DNS issues (for cloud deployment)
nslookup your-domain.com
dig your-domain.com
```

---

## 🎛️ **Advanced Configuration**

### **⚡ Performance Optimization**

```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU optimization
export OMP_NUM_THREADS=4

# Model caching
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### **🔒 Security Hardening**

```bash
# Generate secure secrets
./fm-llm generate-secrets

# Set production security
export FM_LLM_ENV=production
export FLASK_ENV=production
export DEPLOYMENT_MODE=hybrid
```

### **📊 Monitoring Setup**

```bash
# Enable metrics
export METRICS_ENABLED=true
export PROMETHEUS_PORT=9090

# Health check endpoints
curl http://localhost:5000/health
curl http://localhost:5000/metrics
```

---

## 📱 **Platform-Specific Notes**

### **🐧 Linux**

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-dev python3-pip git curl build-essential

# GPU drivers (if needed)
sudo apt install -y nvidia-driver-535
```

### **🍎 macOS**

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 git

# For Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### **🪟 Windows**

```powershell
# Install Python from python.org or Microsoft Store
# Install Git from git-scm.com

# Use WSL2 for best compatibility
wsl --install -d Ubuntu-22.04

# Or use Windows directly with PowerShell
python -m pip install --upgrade pip
```

---

## 🚀 **Quick Start Workflows**

### **🔬 Researcher Setup**

```bash
# Clone and setup
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver
pip install -r requirements/base.txt -r requirements/inference.txt

# Test with GPU
./fm-llm test --gpu

# Start generating certificates
./fm-llm start web
```

### **💻 Developer Setup**

```bash
# Full development environment
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver
pip install -r requirements/base.txt -r requirements/dev.txt

# Development with Docker
docker-compose --profile development up
```

### **🏢 Production Setup**

```bash
# Production deployment
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Setup environment
export FM_LLM_ENV=production
cp config/env.example .env
# Edit .env with production values

# Deploy hybrid cloud
./fm-llm deploy hybrid
```

---

## 📞 **Support & Next Steps**

### **✅ Installation Complete?**

1. **Test Your Setup**: `./fm-llm test --quick`
2. **Start Using**: `./fm-llm start web`
3. **Read User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
4. **Deploy Production**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### **🆘 Need Help?**

- **Common Issues**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Architecture Details**: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)

### **🤝 Contributing**

Ready to contribute? See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for development setup and contribution guidelines.

---

**🎉 Installation complete! Ready to generate barrier certificates with LLMs!** 🧠🚀 