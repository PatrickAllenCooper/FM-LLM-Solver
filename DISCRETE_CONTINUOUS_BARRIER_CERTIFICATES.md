# Discrete and Continuous Barrier Certificates Support

This document explains how to use the FM-LLM-Solver's support for separate knowledge bases and models for discrete and continuous barrier certificates.

## Overview

The FM-LLM-Solver now supports three modes of operation:

1. **Unified Mode** (default): Uses a single knowledge base containing all documents
2. **Discrete Mode**: Uses a knowledge base filtered for discrete barrier certificate papers
3. **Continuous Mode**: Uses a knowledge base filtered for continuous barrier certificate papers

This separation allows for more targeted fine-tuning and RAG retrieval based on the specific type of barrier certificate problem you're working on.

## Configuration

### Basic Configuration

Update your `config.yaml` file to specify the barrier certificate type:

```yaml
knowledge_base:
  barrier_certificate_type: "discrete"  # Options: "unified", "discrete", "continuous"
  classification:
    enable_auto_classification: true
    confidence_threshold: 0.6

fine_tuning:
  barrier_certificate_type: "discrete"  # Should match knowledge_base type
```

### Classification Keywords

The system automatically classifies documents based on keywords. You can customize these in your config:

```yaml
knowledge_base:
  classification:
    discrete_keywords: 
      - "discrete"
      - "hybrid automata"
      - "symbolic"
      - "finite state"
      - "temporal logic"
      - "LTL"
      - "CTL"
      - "model checking"
      - "transition system"
      - "discrete dynamics"
    continuous_keywords:
      - "continuous"
      - "differential equation"
      - "control theory"
      - "Lyapunov"
      - "SOS"
      - "polynomial"
      - "semidefinite"
      - "continuous dynamics"
      - "flow"
      - "vector field"
```

## Workflow

### 1. Build Knowledge Bases

#### For Discrete Barrier Certificates:

```bash
# Update config.yaml to set barrier_certificate_type: "discrete"
python knowledge_base/knowledge_base_builder.py --config config.yaml
```

#### For Continuous Barrier Certificates:

```bash
# Update config.yaml to set barrier_certificate_type: "continuous"
python knowledge_base/knowledge_base_builder.py --config config.yaml
```

#### For Both Types (Run Separately):

```bash
# First, build discrete KB
# Set barrier_certificate_type: "discrete" in config.yaml
python knowledge_base/knowledge_base_builder.py --config config.yaml

# Then, build continuous KB
# Set barrier_certificate_type: "continuous" in config.yaml
python knowledge_base/knowledge_base_builder.py --config config.yaml
```

### 2. Prepare Fine-tuning Data

#### Create Type-Specific Training Data:

```bash
# First, create combined training data
python fine_tuning/combine_datasets.py --config config.yaml

# Then, create type-specific datasets
python fine_tuning/create_type_specific_data.py --config config.yaml --types discrete continuous
```

This will create:
- `data/ft_data_discrete.jsonl` - Training data for discrete barrier certificates
- `data/ft_data_continuous.jsonl` - Training data for continuous barrier certificates

### 3. Fine-tune Models

#### For Discrete Barrier Certificates:

```bash
# Update config.yaml to set both knowledge_base and fine_tuning barrier_certificate_type to "discrete"
python fine_tuning/finetune_llm.py --config config.yaml
```

#### For Continuous Barrier Certificates:

```bash
# Update config.yaml to set both knowledge_base and fine_tuning barrier_certificate_type to "continuous"
python fine_tuning/finetune_llm.py --config config.yaml
```

### 4. Run Inference

#### Generate Discrete Barrier Certificates:

```bash
# Set barrier_certificate_type: "discrete" in config.yaml
python inference/generate_certificate.py "System description here" --config config.yaml
```

#### Generate Continuous Barrier Certificates:

```bash
# Set barrier_certificate_type: "continuous" in config.yaml
python inference/generate_certificate.py "System description here" --config config.yaml
```

### 5. Run Experiments

```bash
# Set the desired barrier_certificate_type in config.yaml
python run_experiments.py --config config.yaml
```

## File Structure

The system creates separate directories and files for each type:

```
├── kb_data/                    # Unified knowledge base
│   ├── paper_index_mathpix.faiss
│   └── paper_metadata_mathpix.jsonl
├── kb_data_discrete/           # Discrete knowledge base
│   ├── paper_index_discrete.faiss
│   ├── paper_metadata_discrete.jsonl
│   └── classification_report_discrete.json
├── kb_data_continuous/         # Continuous knowledge base
│   ├── paper_index_continuous.faiss
│   ├── paper_metadata_continuous.jsonl
│   └── classification_report_continuous.json
└── data/
    ├── ft_data_combined.jsonl           # Combined training data
    ├── ft_data_discrete.jsonl           # Discrete training data
    └── ft_data_continuous.jsonl         # Continuous training data
```

## Classification Reports

The system generates classification reports showing how documents were categorized:

```bash
# View discrete classification report
cat kb_data_discrete/classification_report_discrete.json

# View continuous classification report  
cat kb_data_continuous/classification_report_continuous.json
```

## Utilities

### Check Available Knowledge Bases

```python
from knowledge_base.kb_utils import list_available_kbs
from utils.config_loader import load_config

cfg = load_config("config.yaml")
available_kbs = list_available_kbs(cfg)
print(f"Available knowledge bases: {available_kbs}")
```

### Validate Configuration

```python
from knowledge_base.kb_utils import validate_kb_config
from utils.config_loader import load_config

cfg = load_config("config.yaml")
is_valid = validate_kb_config(cfg)
print(f"Configuration is valid: {is_valid}")
```

## Advanced Usage

### Custom Classification

You can manually classify documents by creating a custom classifier:

```python
from knowledge_base.document_classifier import BarrierCertificateClassifier
from utils.config_loader import load_config

cfg = load_config("config.yaml")
classifier = BarrierCertificateClassifier(cfg)

# Classify a document
text = "Your document text here"
classification, confidence, details = classifier.classify_document(text)
print(f"Classification: {classification} (confidence: {confidence:.3f})")
```

### Building Custom Knowledge Bases

```python
from knowledge_base.knowledge_base_builder import build_single_knowledge_base
from knowledge_base.document_classifier import BarrierCertificateClassifier
from utils.config_loader import load_config

cfg = load_config("config.yaml")
classifier = BarrierCertificateClassifier(cfg)

# Build a specific type of knowledge base
pdf_files = ["path/to/paper1.pdf", "path/to/paper2.pdf"]
success = build_single_knowledge_base(cfg, "discrete", pdf_files, classifier)
```

## Troubleshooting

### Common Issues

1. **No documents classified as target type**:
   - Check your classification keywords
   - Lower the confidence threshold
   - Verify your input documents contain relevant terms

2. **Knowledge base not found**:
   - Ensure you've built the knowledge base for the specified type
   - Check the `barrier_certificate_type` setting in config

3. **Configuration validation errors**:
   - Ensure `barrier_certificate_type` matches between `knowledge_base` and `fine_tuning` sections
   - Verify all required path configurations are present

### Debug Mode

Enable debug logging to see classification details:

```yaml
# In config.yaml
evaluation:
  log_level: "DEBUG"
```

Or use:

```bash
python knowledge_base/knowledge_base_builder.py --config config.yaml --verbose
```

## Performance Considerations

1. **Discrete Mode**: 
   - Typically has fewer documents in the knowledge base
   - May have faster RAG retrieval
   - Better suited for symbolic/logical systems

2. **Continuous Mode**:
   - Usually has more mathematical content
   - May benefit from higher embedding dimensions
   - Better suited for differential equation systems

3. **Unified Mode**:
   - Largest knowledge base
   - May provide more diverse context
   - Good for general-purpose usage

## Migration from Previous Versions

If you have an existing unified knowledge base, you can:

1. Keep using unified mode (no changes needed)
2. Rebuild with classification to create type-specific knowledge bases
3. Use the `create_type_specific_data.py` script to split existing training data

The system maintains backward compatibility with existing configurations.