# Knowledge Base Module

This module manages the creation and management of the knowledge base using either Mathpix API or an open-source PDF processing pipeline.

## Configuration

The pipeline can be configured in `config.yaml`:

```yaml
knowledge_base:
  pipeline: "mathpix"  # or "open_source"
```

## Pipelines

### Mathpix Pipeline

- Requires Mathpix API credentials in `.env`
- High-quality PDF to markdown conversion, especially for mathematical formulas
- Recommended for production use

### Open-Source Pipeline

- No API keys required
- Uses pdf2image, Tesseract OCR, and basic formula detection
- Hardware-aware: optimizes settings for M-series Macs
- Requires system dependencies: 
  - Poppler (`brew install poppler` on macOS)
  - Tesseract OCR (`brew install tesseract` on macOS)

## Quick Test

You can quickly test the open-source PDF processor on a single file:

```bash
python test_minimal_pdf_processor.py path/to/your/file.pdf
```

The output will be saved to `output/test_pipeline/` as Markdown.

## Rebuilding the Knowledge Base

To rebuild the knowledge base with the open-source pipeline:

1. Set `pipeline: "open_source"` in `config.yaml`
2. Run: `python run_experiments.py --only-kb-build`

## Hardware Compatibility

The open-source pipeline is designed to work on:

- Apple Silicon (M1/M2/M3) Macs (optimized)
- x86 systems with or without CUDA GPUs
- Linux systems (requires appropriate system dependencies)

## Troubleshooting

If you encounter NumPy compatibility issues on M-series Macs, try:

```bash
pip install "numpy<2.0.0"
```

For any other issues, check the system requirements and make sure all dependencies are installed. 