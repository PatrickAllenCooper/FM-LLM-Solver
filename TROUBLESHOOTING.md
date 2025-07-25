# 🚨 FM-LLM Solver Troubleshooting Guide

**Common issues and solutions for FM-LLM Solver deployment and usage.**

---

## 📋 **Quick Diagnostic Commands**

Before diving into specific issues, run these diagnostic commands:

```bash
# System health check
./fm-llm status

# Quick validation
./fm-llm test --quick

# Check configuration
./fm-llm config validate

# View system logs
tail -f logs/fm-llm.log
```

---

## 🐍 **Installation & Setup Issues**

### **Python Environment Problems**

#### **Issue: Wrong Python Version**
```bash
# Check Python version
python --version

# Error: Python 3.7 or older
```

**Solution:**
```bash
# Install Python 3.8+ (recommended: 3.10)
# Ubuntu/Debian
sudo apt update && sudo apt install python3.10 python3.10-venv

# macOS with Homebrew
brew install python@3.10

# Windows: Download from python.org
```

#### **Issue: Virtual Environment Problems**
```bash
# Error: Command not found or import errors
```

**Solution:**
```bash
# Create new virtual environment
python3.10 -m venv fmllm-env
source fmllm-env/bin/activate  # Linux/macOS
# or
fmllm-env\Scripts\activate     # Windows

# Verify environment
which python
which pip
```

#### **Issue: Package Installation Fails**
```bash
# Error: Failed building wheel for torch
# Error: Microsoft Visual C++ 14.0 is required
```

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Linux: Install build dependencies
sudo apt install build-essential python3-dev

# macOS: Install Xcode Command Line Tools
xcode-select --install
```

### **Dependency Conflicts**

#### **Issue: PyTorch CUDA Mismatch**
```bash
# Error: CUDA version mismatch or PyTorch not detecting GPU
```

**Solution:**
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
# CUDA 11.8
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **Issue: Conflicting Package Versions**
```bash
# Error: ImportError or version conflicts
```

**Solution:**
```bash
# Create clean environment
conda create -n fmllm-clean python=3.10
conda activate fmllm-clean

# Install in order
pip install -r requirements/base.txt
pip install -r requirements/web.txt      # If needed
pip install -r requirements/inference.txt # If needed
```

---

## 🎮 **GPU & CUDA Issues**

### **GPU Not Detected**

#### **Issue: PyTorch Can't See GPU**
```bash
# Check GPU detection
python -c "import torch; print(torch.cuda.is_available())"
# Output: False
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# If nvidia-smi fails, install/update drivers
# Ubuntu
sudo apt update && sudo apt install nvidia-driver-535

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Issue: CUDA Out of Memory**
```bash
# Error: CUDA out of memory
# Error: CUDA error: out of memory
```

**Solution:**
```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use memory-efficient settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use smaller model or quantization
./fm-llm generate "system" --model base --quantization 4bit
```

### **Performance Issues**

#### **Issue: Slow Generation Times**
```bash
# Generation taking >60 seconds
```

**Solution:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# Optimize memory settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Use faster model
./fm-llm generate "system" --model base

# Reduce token count
./fm-llm generate "system" --max-tokens 256
```

#### **Issue: Model Loading Takes Forever**
```bash
# First model load taking >5 minutes
```

**Solution:**
```bash
# Check internet connection and disk space
df -h

# Use local model cache
export HF_HOME=/path/to/fast/storage

# Pre-download models
./fm-llm download-models

# Use smaller model
./fm-llm generate "system" --model small
```

---

## 🌐 **Web Interface Issues**

### **Server Startup Problems**

#### **Issue: Port Already in Use**
```bash
# Error: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find what's using port 5000
lsof -i :5000
netstat -tulpn | grep :5000

# Kill the process
sudo kill -9 PID

# Or use different port
./fm-llm start web --port 5001
```

#### **Issue: Web Interface Won't Start**
```bash
# Error: ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```bash
# Install web dependencies
pip install -r requirements/web.txt

# Check configuration
./fm-llm config validate

# Start with debug mode
./fm-llm start web --debug
```

### **Browser Connection Issues**

#### **Issue: Can't Connect to http://localhost:5000**
```bash
# Browser shows "This site can't be reached"
```

**Solution:**
```bash
# Check if service is running
curl http://localhost:5000/health

# Check firewall settings
sudo ufw status

# Try binding to all interfaces
./fm-llm start web --host 0.0.0.0

# Check logs
tail -f logs/web.log
```

#### **Issue: 500 Internal Server Error**
```bash
# Web interface loads but shows errors
```

**Solution:**
```bash
# Check detailed logs
tail -f logs/web.log
tail -f logs/fm-llm.log

# Check database connection
./fm-llm db status

# Reset database
./fm-llm db reset --confirm

# Restart web service
./fm-llm restart web
```

---

## 🧮 **Certificate Generation Issues**

### **Generation Failures**

#### **Issue: No Certificate Generated**
```bash
# Output: "Failed to generate certificate"
```

**Solution:**
```bash
# Test with simple system
./fm-llm generate "dx/dt = -x, dy/dt = -y"

# Check model status
./fm-llm status

# Try different model
./fm-llm generate "system" --model base

# Enable verbose logging
./fm-llm generate "system" --verbose
```

#### **Issue: Invalid Certificates**
```bash
# Output: "Certificate validation failed"
```

**Solution:**
```bash
# This is normal! Success rate is ~20%
# Try multiple generations
for i in {1..5}; do ./fm-llm generate "dx/dt = -x, dy/dt = -y"; done

# Use RAG for better results
./fm-llm generate "system" --rag --kb-docs 5

# Simplify system
./fm-llm generate "dx/dt = -x, dy/dt = -y"  # Very simple
```

### **Model Loading Issues**

#### **Issue: Model Download Fails**
```bash
# Error: Connection timeout or SSL errors
```

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Use different mirror
export HF_ENDPOINT=https://hf-mirror.com

# Manual download
git clone https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

# Use offline mode
./fm-llm generate "system" --offline
```

#### **Issue: Model Compatibility**
```bash
# Error: Unsupported model format
```

**Solution:**
```bash
# Update transformers library
pip install --upgrade transformers

# Clear model cache
rm -rf ~/.cache/huggingface/

# Use specific model version
./fm-llm generate "system" --model base --model-revision main
```

---

## 📚 **Knowledge Base Issues**

### **Build Failures**

#### **Issue: Knowledge Base Build Fails**
```bash
# Error during ./fm-llm kb build
```

**Solution:**
```bash
# Check disk space
df -h

# Check permissions
ls -la kb_data/

# Clean and rebuild
./fm-llm kb clean
./fm-llm kb build --rebuild

# Build with debug info
./fm-llm kb build --debug
```

#### **Issue: Mathpix API Errors**
```bash
# Error: Mathpix API key invalid or quota exceeded
```

**Solution:**
```bash
# Check API credentials
echo $MATHPIX_APP_ID
echo $MATHPIX_APP_KEY

# Check quota at https://accounts.mathpix.com/

# Build without Mathpix
./fm-llm kb build --no-mathpix

# Use alternative text extraction
./fm-llm kb build --text-only
```

### **RAG Performance Issues**

#### **Issue: RAG Retrieval Slow**
```bash
# Knowledge base search taking >10 seconds
```

**Solution:**
```bash
# Check vector store
./fm-llm kb validate

# Reduce retrieval count
./fm-llm generate "system" --rag-docs 3

# Update index
./fm-llm kb reindex

# Check disk I/O
iostat -x 1
```

#### **Issue: RAG Not Improving Results**
```bash
# RAG enabled but no improvement in generation quality
```

**Solution:**
```bash
# Check RAG status
./fm-llm kb status

# Verify retrieval
./fm-llm kb search "barrier certificate"

# Increase similarity threshold
./fm-llm generate "system" --rag-threshold 0.8

# Try different embedding model
./fm-llm kb rebuild --embedding all-MiniLM-L6-v2
```

---

## 🐳 **Docker Issues**

### **Build Problems**

#### **Issue: Docker Build Fails**
```bash
# Error during docker build
```

**Solution:**
```bash
# Check Docker version
docker --version  # Should be 20.10+

# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache --target web -t fm-llm:web .

# Check disk space
df -h
```

#### **Issue: GPU Docker Issues**
```bash
# Error: nvidia-docker not found
```

**Solution:**
```bash
# Install nvidia-docker2
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU in Docker
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi
```

### **Container Runtime Issues**

#### **Issue: Container Crashes on Startup**
```bash
# Container exits immediately
```

**Solution:**
```bash
# Check container logs
docker logs container-name

# Run interactively for debugging
docker run -it fm-llm:web /bin/bash

# Check environment variables
docker run fm-llm:web env

# Use debug entrypoint
docker run fm-llm:web /entrypoint.sh debug
```

#### **Issue: Permission Denied in Container**
```bash
# Error: Permission denied writing to /app
```

**Solution:**
```bash
# Check user in container
docker run fm-llm:web whoami

# Fix ownership
docker run --user root fm-llm:web chown -R fmllm:fmllm /app

# Mount with correct permissions
docker run -v $(pwd):/app:Z fm-llm:web

# Use rootless container
docker run --user $(id -u):$(id -g) fm-llm:web
```

---

## ☁️ **Deployment Issues**

### **GCP Deployment Problems**

#### **Issue: kubectl Connection Fails**
```bash
# Error: Unable to connect to cluster
```

**Solution:**
```bash
# Check authentication
gcloud auth list
gcloud auth login

# Get cluster credentials
gcloud container clusters get-credentials cluster-name --zone zone-name

# Check cluster status
kubectl cluster-info

# Test connection
kubectl get nodes
```

#### **Issue: Pod Stuck in Pending**
```bash
# Pod never starts, stays in Pending state
```

**Solution:**
```bash
# Check pod status
kubectl describe pod pod-name

# Check node resources
kubectl top nodes

# Check resource requests
kubectl describe deployment deployment-name

# Scale down resource requests if needed
kubectl edit deployment deployment-name
```

### **Modal Deployment Issues**

#### **Issue: Modal Authentication Fails**
```bash
# Error: Invalid token or unauthorized
```

**Solution:**
```bash
# Check token status
modal token current

# Create new token
modal token new

# Set token explicitly
export MODAL_TOKEN_ID=your-token-id
export MODAL_TOKEN_SECRET=your-token-secret

# Test connection
modal app list
```

#### **Issue: Modal Function Timeout**
```bash
# Functions timing out or cold start issues
```

**Solution:**
```bash
# Increase timeout
./fm-llm deploy modal --timeout 600

# Enable warm pool
./fm-llm deploy modal --keep-warm 2

# Check function logs
modal app logs app-name

# Optimize function startup
./fm-llm deploy modal --optimize-startup
```

---

## 🔧 **Configuration Issues**

### **Environment Variables**

#### **Issue: Configuration Not Loading**
```bash
# Error: Configuration file not found
```

**Solution:**
```bash
# Check environment variable
echo $FM_LLM_ENV

# Set environment explicitly
export FM_LLM_ENV=development

# Check config file paths
ls -la config/
ls -la config/environments/

# Validate configuration
./fm-llm config validate

# Use specific config file
./fm-llm --config /path/to/config.yaml status
```

#### **Issue: Secret Variables Missing**
```bash
# Error: SECRET_KEY is required
```

**Solution:**
```bash
# Create .env file
cp config/env.example .env

# Edit .env with your values
nano .env

# Generate secure secrets
./fm-llm generate-secrets

# Set environment variables
export SECRET_KEY=your-secret-key
export DATABASE_URL=your-database-url
```

### **Database Connection Issues**

#### **Issue: Database Connection Failed**
```bash
# Error: psycopg2.OperationalError: connection failed
```

**Solution:**
```bash
# Check database service
./fm-llm db status

# Test connection manually
psql $DATABASE_URL

# Reset database
./fm-llm db reset --confirm

# Use SQLite for development
export DATABASE_URL=sqlite:///instance/fmllm.db
```

---

## 📊 **Performance Troubleshooting**

### **Memory Issues**

#### **Issue: System Running Out of Memory**
```bash
# Error: Killed (OOM killer)
```

**Solution:**
```bash
# Check memory usage
free -h
htop

# Reduce model cache
export MODEL_CACHE_SIZE=1

# Use CPU instead of GPU
./fm-llm generate "system" --device cpu

# Enable model offloading
export ENABLE_MODEL_OFFLOAD=true

# Use smaller batch sizes
export BATCH_SIZE=1
```

#### **Issue: Disk Space Issues**
```bash
# Error: No space left on device
```

**Solution:**
```bash
# Check disk usage
df -h
du -sh ~/.cache/huggingface/

# Clean caches
./fm-llm clean-cache

# Move cache to larger disk
export HF_HOME=/path/to/larger/disk

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete
```

### **Network Issues**

#### **Issue: Slow API Responses**
```bash
# Web interface or API extremely slow
```

**Solution:**
```bash
# Check system resources
top
nvidia-smi

# Increase timeout
./fm-llm start web --timeout 300

# Check network latency
ping localhost

# Optimize database queries
./fm-llm db optimize

# Use connection pooling
export DB_POOL_SIZE=10
```

---

## 🆘 **Emergency Recovery**

### **Complete System Reset**

If everything is broken:

```bash
# 1. Stop all services
./fm-llm stop all

# 2. Reset configuration
rm -rf config/user/local.yaml
cp config/env.example .env

# 3. Clear caches
./fm-llm clean-cache
rm -rf ~/.cache/huggingface/

# 4. Reset database
./fm-llm db reset --confirm

# 5. Rebuild knowledge base
./fm-llm kb build --rebuild

# 6. Test system
./fm-llm test --quick

# 7. Restart services
./fm-llm start web
```

### **Backup and Restore**

#### **Create Backup**
```bash
# Backup configuration and data
./fm-llm backup create --name emergency-backup

# Manual backup
tar -czf backup-$(date +%Y%m%d).tar.gz \
  config/ kb_data/ instance/ logs/
```

#### **Restore Backup**
```bash
# Restore from backup
./fm-llm backup restore --name emergency-backup

# Manual restore
tar -xzf backup-20250101.tar.gz
```

---

## 📞 **Getting Additional Help**

### **Diagnostic Information**

When reporting issues, include:

```bash
# System information
./fm-llm diagnose --full > system-info.txt

# Configuration
./fm-llm config show --redacted > config-info.txt

# Recent logs
tail -100 logs/fm-llm.log > recent-logs.txt

# Environment
env | grep -E "(FM_LLM|CUDA|TORCH)" > env-vars.txt
```

### **Support Channels**

1. **Self-Service**: Check this troubleshooting guide first
2. **Documentation**: Review [INSTALLATION.md](INSTALLATION.md), [USER_GUIDE.md](USER_GUIDE.md), [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
3. **GitHub Issues**: Report bugs with diagnostic information
4. **Discussions**: Ask questions and share solutions
5. **Community**: Join discussions and help others

### **Bug Report Template**

When reporting bugs:

```markdown
## Bug Report

### Environment
- OS: [Ubuntu 22.04 / macOS 13.0 / Windows 11]
- Python: [3.10.8]
- CUDA: [11.8 / N/A]
- GPU: [RTX 4070 / N/A]

### Issue Description
[Describe the problem]

### Steps to Reproduce
1. [First step]
2. [Second step]
3. [Third step]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Error Messages
```
[Include error messages and logs]
```

### Diagnostic Information
[Include output from ./fm-llm diagnose]
```

---

**🎯 Most issues can be resolved by checking system status, validating configuration, and reviewing logs. Start with the quick diagnostic commands and work through the relevant sections above.** 