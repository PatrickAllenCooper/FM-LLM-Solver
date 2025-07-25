# 📁 Scripts Directory Consolidation Plan
**Phase 2: Organizing 100+ Scripts into Unified fm-llm Command**

*Generated: January 2025 | Status: ORGANIZATION PLAN*

---

## 📊 Current Scripts Chaos

### **Scripts Audit Results**

We have identified **100+ script files** scattered across the `scripts/` directory:

#### **Root Level Scripts (20+ files)**
- **Deployment**: `quick-deploy.sh`, `deploy-gcp.sh`, `deploy-full-stack.sh`, `deploy-via-cloud-build.sh`
- **Testing**: `pre-deployment-test.sh`, `pre-deployment-test-focused.sh`, `final-deployment-test.sh`
- **Validation**: `validate_capabilities.py`, `validate-cicd-setup.sh`, `deployment_check.py`
- **Security**: `security_audit.py`, `init_security.py`
- **Performance**: `performance_benchmark.py`, `complete_production_assessment.py`
- **Setup**: `setup-model-storage.sh`, `create-gpu-nodepool.sh`
- **Monitoring**: `gcp-cost-monitor.sh`
- **Legacy**: `fm-llm` (39 lines - old incomplete version)

#### **Batch Scripts (`scripts/batch/` - 19 files)**
- **Knowledge Base**: `run_kb_builder.bat/sh`, `debug_kb_build.bat`, `optimize_kb_build.bat`
- **Experiments**: `run_optimized_experiments.bat/sh`, `run_barrier_certificate_experiments.sh`
- **Model Testing**: `run_model_comparison.bat`, `run_qwen15b_comparison.bat`, `run_awq_experiment.bat`
- **GPU Testing**: `run_gpu_kb_build.bat`, `check_cuda.bat`
- **Inference**: `run_inference.bat/sh`

#### **Specialized Directories**
- **`knowledge_base/`** (17 files): KB building, debugging, optimization
- **`experiments/`** (10 files): Model experiments, barrier certificate testing
- **`comparison/`** (4 files): Model size and performance comparisons
- **`analysis/`** (4 files): Result analysis and comparison tools
- **`optimization/`** (5 files): Performance optimization scripts
- **`run/`** (6 files): Various runner scripts
- **`setup/`** (4 files): Environment and dependency setup

---

## 🎯 Consolidation Strategy

### **Integration into Unified fm-llm Command**

#### **1. Core Operations** (Integrate into main command)
**Scripts to integrate:**
- `validate_capabilities.py` → `./fm-llm validate`
- `security_audit.py` → `./fm-llm security audit`
- `performance_benchmark.py` → `./fm-llm benchmark`
- `deployment_check.py` → `./fm-llm deploy check`

#### **2. Knowledge Base Operations** (Enhance existing KB command)
**Scripts to integrate:**
- `scripts/knowledge_base/` → `./fm-llm kb` subcommands
- `scripts/batch/run_kb_builder.*` → `./fm-llm kb build`
- `scripts/batch/debug_kb_build.bat` → `./fm-llm kb debug`
- `scripts/batch/optimize_kb_build.bat` → `./fm-llm kb optimize`

#### **3. Experiment Management** (New experiment command)
**Scripts to integrate:**
- `scripts/experiments/` → `./fm-llm experiment`
- `scripts/batch/run_optimized_experiments.*` → `./fm-llm experiment run`
- `scripts/batch/run_barrier_certificate_experiments.sh` → `./fm-llm experiment barrier`
- `scripts/comparison/` → `./fm-llm experiment compare`

#### **4. Deployment Operations** (Enhance existing deploy command)
**Scripts to integrate:**
- `quick-deploy.sh` → `./fm-llm deploy quick`
- `deploy-gcp.sh` → `./fm-llm deploy gcp`
- `deploy-full-stack.sh` → `./fm-llm deploy stack`
- `pre-deployment-test.sh` → `./fm-llm deploy test`

#### **5. Performance Operations** (New performance command)
**Scripts to integrate:**
- `performance_benchmark.py` → `./fm-llm perf benchmark`
- `gcp-cost-monitor.sh` → `./fm-llm perf cost`
- `scripts/optimization/` → `./fm-llm perf optimize`

### **Archive Strategy**

#### **Scripts Archive** (`scripts/archive/`)
**Move these categories to archive:**
- **Legacy Batch Files**: Windows .bat files superseded by unified command
- **Duplicate Functionality**: Scripts that duplicate existing fm-llm features
- **Experimental Scripts**: One-off experiments no longer relevant
- **Platform-Specific**: Scripts for specific deprecated environments

#### **Keep in Place** 
**Scripts that serve specific purposes:**
- `scripts/setup/` - Environment setup utilities
- `scripts/run/` - Background job runners
- Essential deployment scripts not yet integrated

---

## 🔄 Integration Plan

### **Phase A: Core Command Enhancement**

```python
# Add to fm-llm main command
class ValidationCommand(FMLLMCommand):
    """Handle system validation operations."""
    
    def capabilities(self):
        """Validate system capabilities - from validate_capabilities.py"""
        
    def security(self):
        """Run security audit - from security_audit.py"""
        
    def deployment(self):
        """Check deployment readiness - from deployment_check.py"""

class BenchmarkCommand(FMLLMCommand):
    """Handle performance benchmarking."""
    
    def system(self):
        """System performance benchmark - from performance_benchmark.py"""
        
    def cost(self):
        """GCP cost monitoring - from gcp-cost-monitor.sh"""
```

### **Phase B: Knowledge Base Enhancement**

```python
# Enhance existing KnowledgeBaseCommand
class KnowledgeBaseCommand(FMLLMCommand):
    def build(self, kb_type: str = "unified", debug: bool = False, optimize: bool = False):
        """Enhanced KB building with debug and optimization options"""
        
    def debug(self):
        """Debug KB build issues - from debug_kb_build.bat"""
        
    def optimize(self):
        """Optimize KB build performance - from optimize_kb_build.bat"""
        
    def validate(self):
        """Validate KB integrity and performance"""
```

### **Phase C: New Experiment System**

```python
class ExperimentCommand(FMLLMCommand):
    """Handle experimental operations."""
    
    def run(self, experiment_type: str):
        """Run barrier certificate experiments"""
        
    def compare(self, models: List[str]):
        """Compare model performance"""
        
    def barrier(self, system_type: str = "all"):
        """Run barrier certificate experiments"""
```

### **Phase D: Enhanced Deployment**

```python
# Enhance existing DeployCommand
class DeployCommand(FMLLMCommand):
    def quick(self, environment: str = "staging"):
        """Quick deployment - from quick-deploy.sh"""
        
    def gcp(self, environment: str = "production"):
        """Full GCP deployment - from deploy-gcp.sh"""
        
    def stack(self, components: List[str] = None):
        """Deploy full stack - from deploy-full-stack.sh"""
        
    def test(self, environment: str = "staging"):
        """Test deployment - from pre-deployment-test.sh"""
        
    def check(self, environment: str = "staging"):
        """Check deployment status - from deployment_check.py"""
```

---

## 📁 New Command Structure

### **Enhanced fm-llm Command Tree**

```bash
./fm-llm
├── start
│   ├── web
│   ├── inference  
│   ├── hybrid
│   └── full
├── deploy
│   ├── modal
│   ├── hybrid
│   ├── gcp
│   ├── quick          # NEW: from quick-deploy.sh
│   ├── stack          # NEW: from deploy-full-stack.sh
│   ├── test           # NEW: from pre-deployment-test.sh
│   └── check          # NEW: from deployment_check.py
├── test
│   ├── --unit
│   ├── --integration
│   ├── --performance
│   ├── --gpu
│   ├── --quick
│   └── --all
├── kb (Knowledge Base)
│   ├── build
│   ├── debug          # NEW: from debug_kb_build.bat
│   ├── optimize       # NEW: from optimize_kb_build.bat
│   └── validate       # NEW
├── experiment         # NEW: from scripts/experiments/
│   ├── run
│   ├── compare
│   ├── barrier
│   └── analyze
├── validate           # NEW: from validate_capabilities.py
│   ├── capabilities
│   ├── security
│   └── deployment
├── benchmark          # NEW: from performance_benchmark.py
│   ├── system
│   ├── gpu
│   ├── cost
│   └── compare
├── generate
├── status
└── setup              # NEW: from scripts/setup/
    ├── environment
    ├── dependencies
    └── gpu
```

---

## 🗂️ Archive Organization

### **Archive Structure**

```
scripts/
├── archive/
│   ├── batch/              # All .bat files and duplicates
│   ├── legacy/             # Superseded scripts
│   ├── experiments/        # One-off experimental scripts
│   └── platform-specific/ # OS-specific scripts
├── active/
│   ├── setup/              # Keep: environment setup
│   ├── run/                # Keep: background runners
│   └── utilities/          # Keep: standalone utilities
└── integrated/             # Note which scripts were integrated
    └── integration-log.md  # Track what was integrated where
```

### **Integration Tracking**

Create `scripts/integrated/integration-log.md`:
```markdown
# Scripts Integration Log

## Integrated into fm-llm

### Core Operations
- `validate_capabilities.py` → `./fm-llm validate capabilities`
- `security_audit.py` → `./fm-llm validate security`
- `deployment_check.py` → `./fm-llm deploy check`

### Knowledge Base
- `scripts/batch/run_kb_builder.*` → `./fm-llm kb build`
- `scripts/batch/debug_kb_build.bat` → `./fm-llm kb debug`

### Experiments
- `scripts/experiments/` → `./fm-llm experiment`
- `scripts/comparison/` → `./fm-llm experiment compare`

[Continue tracking...]
```

---

## 🚀 Implementation Timeline

### **Day 1: Core Integration**
1. Integrate `validate_capabilities.py` → `./fm-llm validate`
2. Integrate `security_audit.py` → `./fm-llm validate security`
3. Integrate `performance_benchmark.py` → `./fm-llm benchmark`

### **Day 2: KB & Deployment Enhancement**
1. Enhance KB command with debug/optimize options
2. Integrate deployment scripts into deploy command
3. Add deployment testing capabilities

### **Day 3: Experiment System**
1. Create new experiment command system
2. Integrate experiment and comparison scripts
3. Add experiment result analysis

### **Day 4: Cleanup & Archive**
1. Create archive structure
2. Move legacy and duplicate scripts
3. Update documentation
4. Create integration tracking

---

## 📈 Expected Benefits

### **User Experience**
- **Single Interface**: All operations through unified `./fm-llm` command
- **Consistent API**: Same command patterns across all functionality
- **Better Help**: Integrated help system with examples
- **Reduced Confusion**: No more hunting through script directories

### **Maintainability**  
- **Code Deduplication**: Remove redundant batch/shell script variations
- **Single Language**: Python implementation instead of mixed bash/batch
- **Better Error Handling**: Consistent error reporting across all operations
- **Testing**: All functionality can be unit tested

### **Feature Enhancement**
- **Cross-Platform**: No more platform-specific script limitations
- **Configuration Integration**: Scripts respect hierarchical config system
- **Progress Reporting**: Unified progress and status reporting
- **Logging**: Consistent logging across all operations

---

## 💡 Migration Strategy

### **Backward Compatibility**
1. **Preserve Key Scripts**: Keep essential scripts until integration complete
2. **Symlinks**: Create symlinks from old script locations to new commands
3. **Deprecation Warnings**: Show migration hints when old scripts used
4. **Documentation**: Update all references to use new commands

### **Validation**
1. **Functional Testing**: Ensure all integrated functionality works
2. **Performance Testing**: Verify no performance regression
3. **Cross-Platform Testing**: Test on Windows, macOS, Linux
4. **User Acceptance**: Validate with existing users

---

**This consolidation will transform the scripts directory from a chaotic collection into a clean, integrated system accessible through the unified fm-llm command interface.** 