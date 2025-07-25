# 👤 FM-LLM Solver User Guide

**Complete guide to using FM-LLM Solver for barrier certificate generation with Large Language Models.**

---

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Web Interface](#web-interface)
3. [Barrier Certificate Generation](#barrier-certificate-generation)
4. [Knowledge Base Operations](#knowledge-base-operations)
5. [Command Line Interface](#command-line-interface)
6. [Configuration & Customization](#configuration--customization)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 **Quick Start**

### **First-Time Setup**

1. **Installation**: Follow [INSTALLATION.md](INSTALLATION.md) for complete setup
2. **System Check**: Verify your installation
   ```bash
   ./fm-llm status
   ```
3. **Start Web Interface**: Launch the user-friendly interface
   ```bash
   ./fm-llm start web
   ```
4. **Open Browser**: Visit http://localhost:5000

### **Generate Your First Certificate**

#### **Option A: Web Interface (Recommended)**
1. Navigate to http://localhost:5000
2. Register an account or use demo mode
3. Enter a simple system: `dx/dt = -x, dy/dt = -y`
4. Click "Generate Certificate"
5. View the mathematically verified result!

#### **Option B: Command Line**
```bash
./fm-llm generate "dx/dt = -x, dy/dt = -y"
```

### **What You'll See**
```
✅ Generated Certificate: x**2 + y**2 - 1.5
🔍 Verification: Mathematically valid (0 violations)
⏱️ Generation Time: 9.4s (RTX 4070)
📊 Success: Real LLM pipeline working!
```

---

## 🌐 **Web Interface**

### **Overview**

The FM-LLM Solver web interface provides a **Material Design 3** experience for barrier certificate generation with real-time results and conversation mode.

### **Main Features**

#### **🏠 Dashboard**
- **System Status**: View GPU availability, model status, system health
- **Recent Generations**: Access your certificate generation history
- **Quick Actions**: Generate certificates, manage settings
- **Statistics**: Success rates, generation times, system metrics

#### **🧮 Certificate Generation**
- **System Input**: Enter dynamical system descriptions
  - Linear systems: `dx/dt = -x, dy/dt = -y`
  - Nonlinear systems: `dx/dt = -x^3, dy/dt = x - y^3`
  - Discrete systems: `x[k+1] = 0.9*x[k] + 0.1*y[k]`
- **Domain Specification**: Define initial and unsafe sets
- **Real-Time Progress**: Live updates during generation
- **Results Display**: Mathematical verification, LaTeX rendering

#### **💬 Conversation Mode**
- **Interactive Generation**: Refine certificates through conversation
- **Follow-up Questions**: Ask for modifications or explanations
- **Context Preservation**: Maintains conversation history
- **Export Options**: Save conversations and results

#### **👤 User Management**
- **Account Registration**: Create personal accounts
- **Usage Tracking**: Monitor your generation history
- **API Key Management**: Generate keys for programmatic access
- **Subscription Tiers**: Free, Premium, Enterprise levels

### **Navigation Guide**

#### **Header Navigation**
- **FM-LLM Solver**: Home page with quick start
- **Generate**: Main certificate generation interface
- **History**: View past generations and results
- **Account**: Profile, settings, API keys
- **Help**: Documentation and support

#### **Generation Interface**
1. **System Description**: Enter your dynamical system
   ```
   System: dx/dt = -x^3 - y, dy/dt = x - y^3
   Initial Set: x^2 + y^2 <= 0.1
   Unsafe Set: x >= 1.5
   ```
2. **Advanced Options**: Model selection, RAG settings, verification options
3. **Generate**: Click to start the LLM generation process
4. **Results**: View generated certificate with verification

#### **Results Panel**
- **Generated Certificate**: Mathematical expression (e.g., `x^2 + y^2 - 1.5`)
- **Verification Status**: ✅ Valid / ❌ Invalid with details
- **Generation Metadata**: Model used, generation time, confidence
- **Export Options**: JSON, LaTeX, PDF formats
- **Actions**: Save, share, regenerate, discuss

---

## 🧮 **Barrier Certificate Generation**

### **Understanding Barrier Certificates**

A **barrier certificate** is a mathematical function that proves safety properties of dynamical systems by creating an "invisible barrier" between safe and unsafe regions.

#### **Mathematical Definition**
For a system `dx/dt = f(x)`:
- **Initial Set** I: Safe starting region
- **Unsafe Set** U: Regions to avoid
- **Barrier Certificate** B(x): Function satisfying:
  1. `B(x) ≤ 0` for all `x ∈ I` (negative on initial set)
  2. `B(x) > 0` for all `x ∈ U` (positive on unsafe set)
  3. `∇B(x) · f(x) ≤ 0` everywhere (barrier condition)

### **System Types Supported**

#### **Continuous-Time Systems**
```bash
# Linear system
./fm-llm generate "dx/dt = -x, dy/dt = -y"

# Nonlinear system  
./fm-llm generate "dx/dt = -x^3 - y, dy/dt = x - y^3"

# Polynomial system
./fm-llm generate "dx/dt = x - x^3, dy/dt = -y"
```

#### **Discrete-Time Systems**
```bash
# Linear discrete system
./fm-llm generate "x[k+1] = 0.9*x[k], y[k+1] = 0.8*y[k]"

# Nonlinear discrete system
./fm-llm generate "x[k+1] = x[k] - x[k]^3, y[k+1] = y[k] - x[k]*y[k]"
```

#### **Multi-Dimensional Systems**
```bash
# 3D system
./fm-llm generate "dx/dt = -x + y, dy/dt = -y + z, dz/dt = -z"

# Higher dimensional (up to ~6D)
./fm-llm generate "dx1/dt = -x1 + x2, dx2/dt = -x2 + x3, ..."
```

### **Specification Formats**

#### **Compact Format**
```bash
./fm-llm generate "System: dx/dt = -x^3, dy/dt = -y^3. Initial: x^2+y^2 <= 0.1. Unsafe: x >= 1.5"
```

#### **Structured Format**
```bash
./fm-llm generate "
Dynamics: {
  dx/dt = -x^3 - y,
  dy/dt = x - y^3
}
Initial: x^2 + y^2 <= 0.1
Unsafe: x >= 1.5 OR y >= 1.5
"
```

#### **LaTeX Format**
```bash
./fm-llm generate "
\begin{align}
\frac{dx}{dt} &= -x^3 - y \\
\frac{dy}{dt} &= x - y^3
\end{align}
Initial: $x^2 + y^2 \leq 0.1$
"
```

### **Generation Options**

#### **Model Selection**
```bash
# Use base model (default)
./fm-llm generate "system" --model base

# Use fine-tuned model (if available)
./fm-llm generate "system" --model finetuned

# Use specific model size
./fm-llm generate "system" --model large
```

#### **RAG Enhancement**
```bash
# Enable RAG with knowledge base
./fm-llm generate "system" --rag --kb-docs 5

# Disable RAG for pure LLM generation
./fm-llm generate "system" --no-rag
```

#### **Verification Options**
```bash
# Skip verification for speed
./fm-llm generate "system" --no-verify

# Enhanced verification with more samples
./fm-llm generate "system" --verify-samples 1000
```

### **Success Rates & Expectations**

#### **Current Performance**
- **Overall Success Rate**: 20% of generations produce mathematically valid certificates
- **Linear Systems**: ~40% success rate (higher than average)
- **Simple Nonlinear**: ~25% success rate
- **Complex Nonlinear**: ~10% success rate
- **Discrete Systems**: ~15% success rate

#### **Generation Times**
- **RTX 4070**: 9-10 seconds average
- **RTX 3070**: 12-15 seconds average
- **CPU Only**: 45-90 seconds
- **Apple Silicon**: 20-30 seconds

#### **Tips for Success**
1. **Start Simple**: Begin with linear or simple nonlinear systems
2. **Clear Specification**: Use precise mathematical notation
3. **Reasonable Domains**: Avoid extremely complex initial/unsafe sets
4. **Try Multiple Times**: Success rate improves with multiple attempts
5. **Use RAG**: Knowledge base context improves generation quality

---

## 📚 **Knowledge Base Operations**

### **Understanding the Knowledge Base**

The knowledge base contains research papers and mathematical knowledge about barrier certificates, control theory, and verification techniques. It enhances LLM generation through **Retrieval-Augmented Generation (RAG)**.

### **Building Knowledge Base**

#### **Automatic Build (Recommended)**
```bash
# Build unified knowledge base with all sources
./fm-llm kb build --type unified

# Build discrete-time specific knowledge base
./fm-llm kb build --type discrete

# Build continuous-time specific knowledge base  
./fm-llm kb build --type continuous
```

#### **Manual Paper Fetching**
```bash
# Set up environment
export UNPAYWALL_EMAIL="your-email@example.com"
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"  # Optional

# Fetch papers for specific authors
./fm-llm kb fetch --authors "author1,author2"

# Fetch papers by keywords
./fm-llm kb fetch --keywords "barrier certificate,control theory"
```

#### **Advanced Processing**
```bash
# Build with Mathpix OCR (requires API key)
export MATHPIX_APP_ID="your-app-id"
export MATHPIX_APP_KEY="your-app-key"
./fm-llm kb build --type unified --mathpix

# Rebuild from scratch
./fm-llm kb build --rebuild

# Debug build issues
./fm-llm kb debug
```

### **Knowledge Base Status**

#### **Check KB Health**
```bash
# View knowledge base status
./fm-llm kb status

# Validate KB integrity
./fm-llm kb validate
```

**Example Output:**
```
📚 Knowledge Base Status:
✅ Vector Store: 1,247 documents indexed
✅ Metadata: Complete for all documents
✅ Embedding Model: all-mpnet-base-v2
📊 Coverage:
  - Barrier Certificates: 342 papers
  - Control Theory: 451 papers
  - Verification: 298 papers
  - Discrete Systems: 156 papers
```

#### **Search Knowledge Base**
```bash
# Search for specific concepts
./fm-llm kb search "barrier certificate construction"

# Retrieve similar documents
./fm-llm kb similar "polynomial barrier functions" --k 5
```

### **Using RAG in Generation**

#### **Automatic RAG (Default)**
When generating certificates, the system automatically:
1. **Analyzes** your system description
2. **Retrieves** relevant knowledge base documents
3. **Enhances** the LLM prompt with research context
4. **Generates** certificates with improved accuracy

#### **Manual RAG Control**
```bash
# Generate with specific number of RAG documents
./fm-llm generate "system" --rag-docs 10

# Generate with minimum similarity threshold
./fm-llm generate "system" --rag-threshold 0.8

# Generate without RAG for comparison
./fm-llm generate "system" --no-rag
```

---

## 💻 **Command Line Interface**

### **Unified fm-llm Command**

All operations are available through the unified `./fm-llm` command:

#### **System Operations**
```bash
./fm-llm status                    # System status and health check
./fm-llm setup --auto              # Automatic environment setup
./fm-llm test --quick              # Quick system validation
./fm-llm test --all                # Comprehensive test suite
```

#### **Service Management**
```bash
./fm-llm start web                 # Web interface only
./fm-llm start inference           # Inference API only
./fm-llm start hybrid              # Hybrid mode (web + Modal)
./fm-llm start full                # Full local stack
```

#### **Certificate Generation**
```bash
./fm-llm generate "system description"
./fm-llm generate "system" --model finetuned --rag
./fm-llm generate "system" --output result.json
```

#### **Knowledge Base Management**
```bash
./fm-llm kb build --type unified   # Build knowledge base
./fm-llm kb status                 # Check KB status
./fm-llm kb search "query"         # Search knowledge base
```

#### **Testing & Validation**
```bash
./fm-llm test --unit               # Unit tests
./fm-llm test --integration        # Integration tests
./fm-llm test --gpu                # GPU tests (if available)
./fm-llm test --performance        # Performance benchmarks
```

#### **Deployment Operations**
```bash
./fm-llm deploy modal              # Deploy to Modal
./fm-llm deploy hybrid             # Deploy GCP + Modal hybrid
./fm-llm deploy gcp                # Deploy to GCP
```

### **Command Options**

#### **Global Options**
- `--config PATH`: Specify configuration file
- `--env ENV`: Override environment (development/staging/production)
- `--verbose`: Enable verbose logging
- `--quiet`: Suppress non-essential output

#### **Generation Options**
- `--model MODEL`: Model to use (base/finetuned/large)
- `--rag / --no-rag`: Enable/disable RAG
- `--rag-docs N`: Number of RAG documents to retrieve
- `--verify / --no-verify`: Enable/disable verification
- `--output PATH`: Save results to file
- `--format FORMAT`: Output format (json/latex/text)

#### **Testing Options**
- `--quick`: Quick validation tests only
- `--adaptive`: Adaptive testing based on environment
- `--gpu`: Include GPU-specific tests
- `--performance`: Include performance benchmarks
- `--coverage`: Generate coverage reports

---

## ⚙️ **Configuration & Customization**

### **Configuration System**

FM-LLM Solver uses a **hierarchical configuration system**:

1. **Base Configuration**: `config/base.yaml` (common settings)
2. **Environment Overrides**: `config/environments/{env}.yaml`
3. **User Overrides**: `config/user/local.yaml` (optional)

#### **Environment Detection**
```bash
# Set environment explicitly
export FM_LLM_ENV=production  # development, staging, production

# Configuration loads automatically:
# base.yaml → environments/production.yaml → user/local.yaml
```

### **Key Configuration Sections**

#### **Model Configuration**
```yaml
# config/base.yaml
models:
  base:
    name: "Qwen/Qwen2.5-14B-Instruct"
    max_new_tokens: 512
    temperature: 0.1
    device_map: auto
    
  finetuned:
    name: "qwen2.5-14b-barrier-ft"
    use_peft: true
    adapter_path: "output/finetuning_results/checkpoint-best"
```

#### **RAG Configuration**
```yaml
knowledge_base:
  embedding_model: "all-mpnet-base-v2"
  default_k: 3                    # Documents to retrieve
  similarity_threshold: 0.7       # Minimum similarity
  chunk_size: 1000               # Text chunk size
  chunk_overlap: 150             # Overlap between chunks
```

#### **Performance Configuration**
```yaml
performance:
  cuda:
    visible_devices: "0"
    memory_config: "max_split_size_mb:512"
  cpu_workers: 4
  model_cache_size: 2            # Models to keep in memory
```

### **User-Specific Configuration**

Create `config/user/local.yaml` for personal settings:
```yaml
# Personal model preferences
inference:
  default_model: "finetuned"

# Personal RAG settings
knowledge_base:
  default_k: 5
  
# Personal paths
paths:
  models_dir: "/path/to/your/models"
  output_dir: "/path/to/your/output"

# Development settings (if FM_LLM_ENV=development)
development:
  auto_reload: true
  enable_profiling: true
```

### **Environment Variables**

#### **Required Variables**
```bash
# Basic configuration
FM_LLM_ENV=development          # Environment mode
SECRET_KEY=your-secret-key      # Web interface security

# External APIs (optional)
MATHPIX_APP_ID=your-app-id
MATHPIX_APP_KEY=your-app-key
UNPAYWALL_EMAIL=your@email.com
```

#### **Performance Variables**
```bash
# GPU optimization
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model caching
HF_HOME=/path/to/cache
TRANSFORMERS_CACHE=/path/to/cache
```

#### **Deployment Variables**
```bash
# Database (production)
DATABASE_URL=postgresql://user:pass@host:port/db

# Cache (production)
REDIS_URL=redis://host:port/db

# Modal deployment
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

---

## 🔬 **Advanced Features**

### **Fine-Tuning Custom Models**

#### **Data Preparation**
```bash
# Create fine-tuning dataset
./fm-llm finetune prepare --type discrete

# Combine multiple datasets
./fm-llm finetune combine --manual --extracted --synthetic
```

#### **Model Training**
```bash
# Fine-tune on barrier certificate data
./fm-llm finetune train --model base --epochs 10 --lr 2e-5

# Resume training from checkpoint
./fm-llm finetune resume --checkpoint checkpoint-500
```

#### **Model Evaluation**
```bash
# Evaluate fine-tuned model
./fm-llm finetune evaluate --model finetuned --test-data test.jsonl
```

### **Batch Processing**

#### **Process Multiple Systems**
```bash
# Create batch file (batch.txt):
# dx/dt = -x, dy/dt = -y
# dx/dt = -x^3, dy/dt = -y^3  
# x[k+1] = 0.9*x[k], y[k+1] = 0.8*y[k]

./fm-llm batch generate batch.txt --output results/
```

#### **Batch Analysis**
```bash
# Analyze batch results
./fm-llm batch analyze results/ --report summary.json
```

### **Performance Monitoring**

#### **System Metrics**
```bash
# Performance benchmark
./fm-llm perf benchmark --gpu --duration 300

# Monitor resource usage
./fm-llm perf monitor --interval 5
```

#### **Generation Analytics**
```bash
# Analyze generation patterns
./fm-llm analyze generations --timeframe 7d

# Success rate analysis
./fm-llm analyze success --by-system-type
```

### **Integration & APIs**

#### **REST API**
```bash
# Start API server
./fm-llm start inference --port 8000

# API endpoints available:
# POST /generate - Generate certificates
# GET /health - Health check
# GET /models - Available models
```

#### **Python API**
```python
from fm_llm_solver import CertificateGenerator

generator = CertificateGenerator.from_config()
result = generator.generate("dx/dt = -x, dy/dt = -y")

print(f"Certificate: {result.certificate}")
print(f"Valid: {result.is_valid}")
print(f"Verification: {result.verification}")
```

#### **Webhook Integration**
```bash
# Configure webhooks for generation events
./fm-llm webhook add --url https://your-app.com/webhook --events generation.complete
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Generation Failures**
```bash
# Check system status
./fm-llm status

# Test with simple system
./fm-llm generate "dx/dt = -x, dy/dt = -y"

# Check logs
tail -f logs/fm-llm.log
```

#### **GPU Issues**
```bash
# Check GPU availability
./fm-llm test --gpu

# Monitor GPU usage
watch -n 1 nvidia-smi

# Clear GPU memory
./fm-llm gpu clear-cache
```

#### **Knowledge Base Issues**
```bash
# Validate knowledge base
./fm-llm kb validate

# Rebuild if corrupted
./fm-llm kb build --rebuild

# Check disk space
df -h
```

#### **Web Interface Issues**
```bash
# Check web service
curl http://localhost:5000/health

# Restart web interface
./fm-llm restart web

# Check configuration
./fm-llm config validate
```

### **Performance Optimization**

#### **Speed Up Generation**
```bash
# Use smaller model
./fm-llm generate "system" --model base

# Reduce token count
./fm-llm generate "system" --max-tokens 256

# Disable verification for speed
./fm-llm generate "system" --no-verify
```

#### **Improve Success Rate**
```bash
# Enable RAG
./fm-llm generate "system" --rag --rag-docs 10

# Use fine-tuned model
./fm-llm generate "system" --model finetuned

# Simplify system description
./fm-llm generate "dx/dt = -x, dy/dt = -y"
```

### **Getting Help**

#### **Built-in Help**
```bash
# Command help
./fm-llm --help
./fm-llm generate --help
./fm-llm kb --help

# System diagnostics
./fm-llm diagnose
```

#### **Documentation**
- **Installation Issues**: [INSTALLATION.md](INSTALLATION.md)
- **Development Setup**: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Deployment Problems**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Common Issues**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🎯 **Next Steps**

### **Learning Path**

1. **Beginner**: Start with web interface and simple linear systems
2. **Intermediate**: Try command line interface and nonlinear systems  
3. **Advanced**: Use RAG, fine-tuning, and batch processing
4. **Expert**: Deploy production systems and integrate APIs

### **Best Practices**

1. **Always Test**: Use `./fm-llm test --quick` before important work
2. **Monitor Resources**: Check GPU memory and disk space regularly
3. **Backup Results**: Save important certificates and configurations
4. **Stay Updated**: Check for model and knowledge base updates
5. **Contribute**: Share successful system examples and improvements

### **Community & Support**

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Share examples and ask questions
- **Documentation**: Contribute to guides and tutorials
- **Research**: Publish results and cite the project

---

**🎉 Ready to generate barrier certificates? Start with the web interface at http://localhost:5000 or try the CLI with simple systems!** 🧠🚀 