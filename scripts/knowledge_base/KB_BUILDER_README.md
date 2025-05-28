# FM-LLM-Solver Knowledge Base Builder

This README documents the consolidated knowledge base builder for FM-LLM-Solver.

## Overview

The Knowledge Base Builder processes scientific PDFs and creates a searchable vector database that allows semantic searching of their contents. This consolidated version includes multiple optimizations:

1. **Memory Management**: Prevents CUDA out-of-memory errors on GPUs with limited VRAM
2. **Batch Processing**: Processes PDFs in small batches with incremental saves
3. **Resume Capability**: Can continue from where it left off if interrupted
4. **Optimized Chunking**: Uses an improved text chunking algorithm
5. **Progress Monitoring**: Provides detailed real-time progress updates
6. **Nested Directory Support**: Handles PDFs organized in subdirectories
7. **Fallback Mechanisms**: Automatically falls back to CPU when GPU memory is exhausted

## Usage

### Running the Builder

The simplest way to run the knowledge base builder is with the batch script:

```
run_kb_builder.bat
```

This will process PDFs using the default settings (batch size of 3).

### Command-Line Options

You can customize the behavior with these command-line arguments:

```
run_kb_builder.bat --batch-size 5 --force
```

Available options:

- `--batch-size N`: Process N PDFs at a time (default: 3)
- `--force`: Force rebuild the entire knowledge base (discard existing progress)
- `--cpu-only`: Use CPU only (slower but more reliable)
- `--config PATH`: Specify a custom config file path
- `--skip-monitor`: Disable the progress monitor thread

### Examples

Process just 1 PDF at a time (safest but slowest):
```
run_kb_builder.bat --batch-size 1
```

Rebuild the entire knowledge base from scratch:
```
run_kb_builder.bat --force
```

Use CPU only (for machines with GPU issues):
```
run_kb_builder.bat --cpu-only
```

## Troubleshooting

If the process seems to hang:
1. Check the progress monitor output - it will warn you if no activity is detected
2. The embedding generation phase can be very slow, especially on CPU
3. Try reducing batch size to 1 using `--batch-size 1`
4. If CUDA memory errors persist, try `--cpu-only` mode

## Technical Details

The knowledge base builder performs these operations:

1. Searches for PDFs in `data/fetched_papers` (including subdirectories)
2. Processes PDFs in small batches:
   - Extracts text content using the optimized PDF processor
   - Chunks text into manageable segments
   - Generates vector embeddings for each chunk
   - Builds a FAISS vector index
3. Saves incremental results after each batch
4. Tracks which PDFs have been processed to avoid duplication

## Files

- `kb_builder.py`: Main Python script with all optimizations
- `run_kb_builder.bat`: Windows batch file to run the builder
- `knowledge_base/optimized_chunker.py`: Optimized chunking algorithm
- `output/knowledge_base/`: Location of the built knowledge base
- `output/logs/`: Directory containing detailed log files 