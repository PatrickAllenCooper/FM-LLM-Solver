# Stochastic Barrier Certificate Filtering

This document explains the stochastic barrier certificate filtering system implemented in the FM-LLM-Solver project. This feature allows you to include or exclude papers containing stochastic barrier certificate content from both knowledge base construction and fine-tuning data creation.

## Overview

Stochastic barrier certificates deal with probabilistic safety verification under uncertainty, noise, and random disturbances. These differ from deterministic barrier certificates (discrete or continuous) and may require different mathematical approaches and verification methods.

The filtering system automatically detects stochastic content in papers and provides flexible options to:
- **Exclude stochastic papers** (focus on deterministic methods)
- **Include only stochastic papers** (focus on probabilistic methods)
- **Apply filtering at different stages** (knowledge base, fine-tuning data)

## Architecture

### Components

1. **Configuration System** (`config/config.yaml`)
   - Centralized settings for stochastic filtering
   - Separate controls for knowledge base and fine-tuning

2. **Document Classifier** (`knowledge_base/document_classifier.py`)
   - Keyword-based stochastic content detection
   - Confidence scoring and thresholding
   - Integration with existing discrete/continuous classification

3. **Knowledge Base Builder** (`knowledge_base/knowledge_base_builder.py`)
   - Applies filtering during KB construction
   - Detailed logging of filter decisions
   - Classification reporting

4. **Fine-tuning Components** (`fine_tuning/`)
   - Filtering in LLM extraction (`extract_from_papers.py`)
   - Filtering in manual data creation (`create_finetuning_data.py`)
   - Support for synthetic data filtering

5. **Configuration Script** (`scripts/configure_stochastic_filter.py`)
   - Easy enable/disable functionality
   - Status reporting
   - Advanced configuration options

## Configuration

### Stochastic Keywords

The system uses a comprehensive set of keywords to detect stochastic content:

```yaml
stochastic_keywords: [
  "stochastic", "probabilistic", "random", "noise", "uncertainty",
  "martingale", "supermartingale", "submartingale", 
  "brownian motion", "wiener process", "stochastic differential", "SDE",
  "markov", "random walk", "monte carlo", "probabilistic safety",
  "almost surely", "probability", "stochastic reachability", "stochastic control"
]
```

### Knowledge Base Filtering

```yaml
knowledge_base:
  classification:
    stochastic_filter:
      enable: false  # Whether to apply stochastic filtering
      mode: "exclude"  # "include" or "exclude"
      min_stochastic_keywords: 2  # Minimum keywords for classification
      stochastic_confidence_threshold: 0.4  # Classification threshold
```

### Fine-tuning Filtering

```yaml
fine_tuning:
  stochastic_filter:
    enable: false  # Whether to apply stochastic filtering to training data
    mode: "exclude"  # "include" or "exclude" 
    apply_to_extracted_data: true   # Filter LLM-extracted data
    apply_to_manual_data: false     # Filter manually created data
    apply_to_synthetic_data: true   # Filter synthetically generated data
```

## Usage

### Quick Start

Check current status:
```bash
python scripts/configure_stochastic_filter.py --status
```

Enable filtering to exclude stochastic papers:
```bash
python scripts/configure_stochastic_filter.py --enable --mode exclude
```

Enable filtering to include only stochastic papers:
```bash
python scripts/configure_stochastic_filter.py --enable --mode include
```

Disable filtering:
```bash
python scripts/configure_stochastic_filter.py --disable
```

### Advanced Configuration

Enable with custom settings:
```bash
python scripts/configure_stochastic_filter.py --enable --mode exclude \
    --min-keywords 3 \
    --confidence-threshold 0.5 \
    --apply-to-manual \
    --no-apply-to-synthetic
```

### Rebuilding After Configuration Changes

After changing filter settings, rebuild affected components:

1. **Knowledge Base**: Run the knowledge base builder to apply filtering
   ```bash
   # For specific barrier certificate types
   python -m knowledge_base.knowledge_base_builder --config config.yaml
   
   # Or use the provided scripts
   scripts/knowledge_base/run_kb_builder.sh
   ```

2. **Fine-tuning Data**: Regenerate training data if filtering was applied
   ```bash
   # Regenerate LLM extraction prompts
   python -m fine_tuning.extract_from_papers --config config.yaml
   
   # Regenerate manual data with filtering
   python -m fine_tuning.create_finetuning_data --config config.yaml
   ```

## Classification Algorithm

### Stochastic Detection Logic

1. **Text Preprocessing**: Normalize text (lowercase, remove punctuation, normalize whitespace)

2. **Keyword Matching**: Count occurrences of stochastic keywords using regex patterns

3. **Classification Decision**:
   ```python
   if stochastic_count >= min_stochastic_keywords:
       is_stochastic = True
       confidence = min(stochastic_freq * 50 + (stochastic_count / min_keywords) * 0.5, 1.0)
   else:
       is_stochastic = False
       confidence = 1.0 - (stochastic_count / min_keywords)
   ```

4. **Confidence Thresholding**: Apply minimum confidence threshold for reliable classification

### Filter Decision Logic

- **Exclude Mode**: `include_paper = not is_stochastic`
- **Include Mode**: `include_paper = is_stochastic`

## Examples

### Example 1: Focus on Deterministic Methods

**Goal**: Build a knowledge base and training data focused only on deterministic barrier certificates.

**Configuration**:
```bash
python scripts/configure_stochastic_filter.py --enable --mode exclude
```

**Result**: Papers containing stochastic keywords like "brownian motion", "probability", "martingale" will be excluded from both knowledge base and training data.

### Example 2: Stochastic-Only Dataset

**Goal**: Create a specialized dataset containing only stochastic barrier certificate papers.

**Configuration**:
```bash
python scripts/configure_stochastic_filter.py --enable --mode include \
    --min-keywords 3 \
    --confidence-threshold 0.6
```

**Result**: Only papers with 3+ stochastic keywords and high confidence will be included.

### Example 3: Selective Filtering

**Goal**: Filter knowledge base but not manual training data.

**Configuration**:
1. Enable filtering for knowledge base:
   ```yaml
   knowledge_base:
     classification:
       stochastic_filter:
         enable: true
         mode: "exclude"
   ```

2. Disable filtering for manual fine-tuning data:
   ```yaml
   fine_tuning:
     stochastic_filter:
       enable: true
       apply_to_manual_data: false
   ```

## Monitoring and Debugging

### Classification Reports

The system generates detailed classification reports:
- **Location**: `{kb_output_dir}/classification_report_{kb_type}.json`
- **Contents**: Classification results, confidence scores, keyword matches, filter decisions

### Logging

Filter decisions are logged with detailed information:
```
INFO - Stochastic classification for paper.pdf: is_stochastic=True (confidence: 0.750, keywords: 5, required: 2)
INFO - Stochastic filter decision for paper.pdf: include=False, reason='Excluded stochastic content (confidence: 0.750)'
INFO - Excluding paper.pdf from discrete KB (stochastic: Excluded stochastic content)
```

### Status Checking

Monitor current configuration:
```bash
python scripts/configure_stochastic_filter.py --status
```

Output includes:
- Current enable/disable status
- Filter mode (include/exclude)
- Confidence thresholds
- Application settings for different data types
- Number of stochastic keywords

## Integration with Existing Systems

### Barrier Certificate Type Classification

Stochastic filtering works alongside existing discrete/continuous classification:

```python
# Both filters must pass for inclusion
type_include = classification in ["discrete", "both"]  # Type filter
stochastic_include = should_include_document(text)     # Stochastic filter
final_include = type_include and stochastic_include    # Combined decision
```

### Metadata Enhancement

Filtered documents include additional metadata:
```json
{
  "source": "manual",
  "stochastic_filtered": true,
  "stochastic_filter_reason": "Included non-stochastic content (confidence: 0.850)",
  "stochastic_filter": {
    "include": true,
    "reason": "Included non-stochastic content",
    "details": {
      "stochastic_count": 1,
      "stochastic_freq": 0.001,
      "stochastic_matches": ["probability"]
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **No papers after filtering**:
   - Check filter mode (include vs exclude)
   - Lower confidence threshold
   - Reduce minimum keywords required
   - Verify stochastic keywords are appropriate

2. **Unexpected classifications**:
   - Review keyword list for false positives
   - Adjust confidence thresholds
   - Check classification reports for details

3. **Configuration not taking effect**:
   - Ensure knowledge base is rebuilt after changes
   - Check that filtering is enabled in correct sections
   - Verify configuration file syntax

### Debug Commands

Show detailed status:
```bash
python scripts/configure_stochastic_filter.py --status
```

Test classification on sample text:
```python
from knowledge_base.document_classifier import BarrierCertificateClassifier
from utils.config_loader import load_config

cfg = load_config()
classifier = BarrierCertificateClassifier(cfg)
is_stochastic, confidence, details = classifier.classify_stochastic_content(sample_text)
print(f"Stochastic: {is_stochastic}, Confidence: {confidence:.3f}")
print(f"Matches: {details['stochastic_matches']}")
```

## Performance Considerations

- **Keyword matching**: O(n*k) where n=text length, k=number of keywords
- **Memory usage**: Minimal additional overhead
- **Build time impact**: ~5-10% increase due to additional classification
- **Storage**: Classification metadata adds ~1KB per document

## Future Enhancements

Potential improvements to the stochastic filtering system:

1. **Machine Learning Classification**: Replace keyword-based approach with trained models
2. **Semantic Similarity**: Use embeddings to detect stochastic concepts beyond keywords
3. **Mathematical Pattern Recognition**: Detect stochastic equations and notation
4. **Hierarchical Classification**: Distinguish between different types of stochastic methods
5. **Interactive Tuning**: Web interface for adjusting filter parameters
6. **Batch Re-classification**: Efficiently re-classify existing datasets

## API Reference

### BarrierCertificateClassifier Methods

#### `classify_stochastic_content(text, source_path=None)`
Classify text for stochastic content.

**Parameters**:
- `text` (str): Input text to analyze
- `source_path` (str, optional): Path for logging

**Returns**:
- `is_stochastic` (bool): Whether text contains stochastic content
- `confidence` (float): Classification confidence [0, 1]
- `details` (dict): Keyword counts and matches

#### `should_include_document(text, source_path=None)`
Determine if document should be included based on filter configuration.

**Parameters**:
- `text` (str): Input text to analyze  
- `source_path` (str, optional): Path for logging

**Returns**:
- `should_include` (bool): Whether to include document
- `reason` (str): Explanation of decision
- `stochastic_details` (dict): Classification details

### Configuration Functions

#### `load_config(config_path=DEFAULT_CONFIG_PATH)`
Load configuration from YAML file.

#### `save_config(cfg, config_path=DEFAULT_CONFIG_PATH)`  
Save configuration object to YAML file.

---

For questions or issues with stochastic filtering, please check the troubleshooting section or refer to the project documentation. 