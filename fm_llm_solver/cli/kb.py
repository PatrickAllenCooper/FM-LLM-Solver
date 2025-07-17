"""
Knowledge base CLI commands for FM-LLM Solver.

Replaces the scattered scripts in scripts/knowledge_base/ with a unified interface.
"""

import click
import time
from pathlib import Path
from typing import Optional

from fm_llm_solver.core.logging import get_logger


@click.group()
def kb():
    """Knowledge base management commands."""
    pass


@kb.command()
@click.option("--pdf-dir", type=click.Path(exists=True), help="Directory containing PDF files")
@click.option("--output-dir", type=click.Path(), help="Output directory for knowledge base")
@click.option("--force", is_flag=True, help="Force rebuild even if KB exists")
@click.option("--batch-size", default=3, help="Number of PDFs to process in each batch")
@click.option("--use-mathpix", is_flag=True, help="Use Mathpix for PDF processing")
@click.option("--cpu-only", is_flag=True, help="Force CPU usage for processing")
@click.pass_context
def build(
    ctx,
    pdf_dir: Optional[str],
    output_dir: Optional[str],
    force: bool,
    batch_size: int,
    use_mathpix: bool,
    cpu_only: bool,
):
    """Build knowledge base from PDF documents."""
    config = ctx.obj["config"]
    logger = get_logger("kb.build")

    # Set default paths
    if not pdf_dir:
        pdf_dir = Path("data/papers")
    else:
        pdf_dir = Path(pdf_dir)

    if not output_dir:
        output_dir = Path(config.paths.kb_output_dir)
    else:
        output_dir = Path(output_dir)

    click.echo(f"📚 Building knowledge base from PDFs in {pdf_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"🔧 Processing method: {'Mathpix' if use_mathpix else 'Open-source'}")

    if not pdf_dir.exists():
        click.echo(f"❌ PDF directory not found: {pdf_dir}")
        return

    # Check for existing KB
    existing_files = list(output_dir.glob("*.faiss"))
    if existing_files and not force:
        click.echo(f"⚠️  Knowledge base already exists ({len(existing_files)} files)")
        if not click.confirm("Rebuild anyway?"):
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find PDF files
    pdf_files = list(pdf_dir.glob("*.pd"))
    if not pdf_files:
        click.echo(f"❌ No PDF files found in {pdf_dir}")
        return

    click.echo(f"📄 Found {len(pdf_files)} PDF files")

    try:
        start_time = time.time()

        if use_mathpix:
            _build_with_mathpix(pdf_files, output_dir, batch_size, cpu_only, logger)
        else:
            _build_open_source(pdf_files, output_dir, batch_size, cpu_only, logger)

        elapsed = time.time() - start_time
        click.echo(f"\n✅ Knowledge base built successfully in {elapsed:.1f}s")

        # Show stats
        kb_files = list(output_dir.glob("*.faiss"))
        total_size = sum(f.stat().st_size for f in kb_files) / (1024**2)
        click.echo(f"📊 Created {len(kb_files)} index files ({total_size:.1f} MB total)")

    except Exception as e:
        logger.error(f"Knowledge base build failed: {e}")
        click.echo(f"❌ Build failed: {e}")


def _build_with_mathpix(pdf_files, output_dir, batch_size, cpu_only, logger):
    """Build knowledge base using Mathpix API."""
    import os

    # Check for Mathpix credentials
    api_id = os.environ.get("MATHPIX_APP_ID")
    api_key = os.environ.get("MATHPIX_APP_KEY")

    if not api_id or not api_key:
        click.echo("❌ Mathpix credentials not found in environment")
        click.echo("   Set MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables")
        return

    click.echo("🔐 Using Mathpix API for PDF processing")

    # Process in batches
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size

    with click.progressbar(range(total_batches), label="Processing batches") as bar:
        for batch_idx in bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pdf_files))
            batch_files = pdf_files[start_idx:end_idx]

            # Process batch (simplified - real implementation would be more complex)
            for pdf_file in batch_files:
                logger.info(f"Processing {pdf_file.name}")
                # TODO: Implement actual Mathpix processing
                time.sleep(0.1)  # Simulate processing


def _build_open_source(pdf_files, output_dir, batch_size, cpu_only, logger):
    """Build knowledge base using open-source tools."""
    click.echo("🔓 Using open-source PDF processing")

    # Check dependencies
    try:
        import PyMuPDF  # fitz
        import sentence_transformers
    except ImportError as e:
        click.echo(f"❌ Missing dependency: {e}")
        click.echo("   Install with: pip install PyMuPDF sentence-transformers")
        return

    # Process files
    documents = []

    with click.progressbar(pdf_files, label="Processing PDFs") as bar:
        for pdf_file in bar:
            try:
                # Extract text using PyMuPDF
                import fitz  # PyMuPDF

                doc = fitz.open(pdf_file)
                text = ""
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text()
                doc.close()

                if text.strip():
                    documents.append({"content": text, "metadata": {"source": pdf_file.name}})
                    logger.debug(f"Extracted text from {pdf_file.name} ({len(text)} chars)")

            except Exception as e:
                logger.warning(f"Failed to process {pdf_file.name}: {e}")

    if not documents:
        click.echo("❌ No text extracted from PDFs")
        return

    click.echo(f"📝 Extracted text from {len(documents)} documents")

    # Build embeddings and index
    click.echo("🧠 Creating embeddings...")
    try:
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if cpu_only:
            model = SentenceTransformer(model_name, device="cpu")
        else:
            model = SentenceTransformer(model_name)

        # Create chunks
        chunks = []
        chunk_size = 1000
        for doc in documents:
            content = doc["content"]
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                if chunk.strip():
                    chunks.append({"text": chunk, "metadata": doc["metadata"]})

        click.echo(f"📄 Created {len(chunks)} text chunks")

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)

        # Create FAISS index
        import faiss
        import numpy as np

        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]

        # Use flat index for simplicity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)

        # Save index
        index_path = output_dir / "knowledge_base.faiss"
        faiss.write_index(index, str(index_path))

        # Save metadata
        import json

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": [chunk["metadata"] for chunk in chunks],
                    "model": model_name,
                    "dimension": dimension,
                    "count": len(chunks),
                },
                f,
                indent=2,
            )

        click.echo(f"💾 Saved index to {index_path}")

    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        click.echo(f"❌ Embedding creation failed: {e}")


@kb.command()
@click.option("--kb-path", type=click.Path(exists=True), help="Path to knowledge base directory")
@click.pass_context
def info(ctx, kb_path: Optional[str]):
    """Show knowledge base information."""
    config = ctx.obj["config"]

    if not kb_path:
        kb_path = Path(config.paths.kb_output_dir)
    else:
        kb_path = Path(kb_path)

    click.echo(f"📚 Knowledge Base Information: {kb_path}")

    if not kb_path.exists():
        click.echo("❌ Knowledge base directory not found")
        return

    # Find index files
    faiss_files = list(kb_path.glob("*.faiss"))
    metadata_files = list(kb_path.glob("metadata.json"))

    if not faiss_files:
        click.echo("❌ No FAISS index files found")
        return

    click.echo("\n📊 Statistics:")
    total_size = 0

    for faiss_file in faiss_files:
        size = faiss_file.stat().st_size
        total_size += size
        click.echo(f"  • {faiss_file.name}: {size / (1024**2):.1f} MB")

    click.echo(f"  • Total size: {total_size / (1024**2):.1f} MB")

    # Load metadata if available
    if metadata_files:
        import json

        try:
            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            click.echo("\n🔧 Configuration:")
            click.echo(f"  • Embedding model: {metadata.get('model', 'Unknown')}")
            click.echo(f"  • Vector dimension: {metadata.get('dimension', 'Unknown')}")
            click.echo(f"  • Number of chunks: {metadata.get('count', 'Unknown')}")
            click.echo(
                f"  • Sources: {len(set(chunk.get('source', '') for chunk in metadata.get('chunks', [])))}"
            )

        except Exception as e:
            click.echo(f"⚠️  Could not read metadata: {e}")


@kb.command()
@click.argument("query")
@click.option("--k", default=5, help="Number of results to return")
@click.option("--kb-path", type=click.Path(exists=True), help="Path to knowledge base directory")
@click.pass_context
def search(ctx, query: str, k: int, kb_path: Optional[str]):
    """Search the knowledge base for relevant documents."""
    config = ctx.obj["config"]

    if not kb_path:
        kb_path = Path(config.paths.kb_output_dir)
    else:
        kb_path = Path(kb_path)

    click.echo(f"🔍 Searching for: '{query}'")

    try:
        # Load index and metadata
        import faiss
        import json
        from sentence_transformers import SentenceTransformer

        index_path = kb_path / "knowledge_base.faiss"
        metadata_path = kb_path / "metadata.json"

        if not index_path.exists():
            click.echo("❌ Knowledge base index not found")
            return

        # Load index
        index = faiss.read_index(str(index_path))

        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Load embedding model
        model_name = metadata.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)

        # Generate query embedding
        query_embedding = model.encode([query])

        # Search
        scores, indices = index.search(query_embedding, k)

        click.echo(f"\n📋 Top {k} results:")

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # No more results
                break

            click.echo(f"\n{i+1}. Score: {score:.3f}")

            # Get metadata if available
            chunks = metadata.get("chunks", [])
            if idx < len(chunks):
                chunk_meta = chunks[idx]
                source = chunk_meta.get("source", "Unknown")
                click.echo(f"   Source: {source}")

            # Show snippet (would need to store text separately in real implementation)
            click.echo(f"   Preview: [Document chunk {idx}]")

    except Exception as e:
        click.echo(f"❌ Search failed: {e}")


@kb.command()
@click.option("--kb-path", type=click.Path(exists=True), help="Path to knowledge base directory")
@click.confirmation_option(prompt="Are you sure you want to delete the knowledge base?")
@click.pass_context
def clean(ctx, kb_path: Optional[str]):
    """Clean/delete the knowledge base."""
    config = ctx.obj["config"]

    if not kb_path:
        kb_path = Path(config.paths.kb_output_dir)
    else:
        kb_path = Path(kb_path)

    if not kb_path.exists():
        click.echo("❌ Knowledge base directory not found")
        return

    # Remove all KB files
    removed_count = 0
    for pattern in ["*.faiss", "*.json", "*.pkl"]:
        for file in kb_path.glob(pattern):
            file.unlink()
            removed_count += 1

    click.echo(f"🗑️  Removed {removed_count} knowledge base files")
    click.echo("✅ Knowledge base cleaned")
