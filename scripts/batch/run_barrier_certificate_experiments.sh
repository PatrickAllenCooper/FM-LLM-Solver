#!/bin/bash
# Barrier Certificate Experiment Runner
# This script provides convenient shortcuts for running various experiment configurations

set -e  # Exit on error

# Default values
EXPERIMENT_TYPE="full"
MODELS="Qwen/Qwen2.5-7B-Instruct"
RAG_K_VALUES="3,5,7"
EMBEDDING_MODELS="all-mpnet-base-v2"
SKIP_DATA_FETCHING=true
SKIP_KB_BUILDING=true
LIMIT=0
OUTPUT_DIR="experiments"
RANDOM_SAMPLE=false
NO_COMBINATIONS=false
DIMENSIONS=""
SKIP_FINETUNING=false

# Display help message
function show_help {
    echo "Barrier Certificate Experiment Runner"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -t, --type TYPE            Experiment type (full, quick, model, rag, system, verification)"
    echo "  -m, --models MODELS        Comma-separated list of models to test"
    echo "  -k, --rag-k VALUES         Comma-separated list of RAG k values to test"
    echo "  -e, --embeddings MODELS    Comma-separated list of embedding models to test"
    echo "  -d, --dimensions DIMS      Comma-separated list of specific dimensions to vary"
    echo "  -l, --limit N              Limit number of experiments to run"
    echo "  -o, --output-dir DIR       Output directory for experiment results"
    echo "  -r, --random-sample        Randomly sample experiments when using limit"
    echo "  -n, --no-combinations      Run each dimension independently (no cross-product)"
    echo "  -f, --fetch-data           Include data fetching step (normally skipped)"
    echo "  -b, --build-kb             Include knowledge base building step (normally skipped)"
    echo "  --finetune                 Include fine-tuning step (normally skipped)"
    echo ""
    echo "Predefined experiment types:"
    echo "  full         All combinations of parameters (may generate many experiments)"
    echo "  quick        Single dimension variation (model OR rag OR system OR verification)"
    echo "  model        Test different LLM models only"
    echo "  rag          Test different RAG settings only"
    echo "  system       Test different system types only"
    echo "  verification Test different verification methods only"
    echo ""
    echo "Examples:"
    echo "  $0 --type quick               Run a quick test varying one dimension at a time"
    echo "  $0 --type model --models 'Qwen/Qwen2.5-7B-Instruct,Llama-3-7b-instruct'   Test specific models"
    echo "  $0 --type rag --rag-k 3,5,7,10   Test with different RAG k values"
    echo "  $0 --dimensions model,rag --limit 10   Test model and RAG dimensions, limit to 10 experiments"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -k|--rag-k)
            RAG_K_VALUES="$2"
            shift 2
            ;;
        -e|--embeddings)
            EMBEDDING_MODELS="$2"
            shift 2
            ;;
        -d|--dimensions)
            DIMENSIONS="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--random-sample)
            RANDOM_SAMPLE=true
            shift
            ;;
        -n|--no-combinations)
            NO_COMBINATIONS=true
            shift
            ;;
        -f|--fetch-data)
            SKIP_DATA_FETCHING=false
            shift
            ;;
        -b|--build-kb)
            SKIP_KB_BUILDING=false
            shift
            ;;
        --finetune)
            SKIP_FINETUNING=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Construct base command
CMD="python run_parameterized_experiments.py"
CMD+=" --models \"$MODELS\""
CMD+=" --rag-k-values \"$RAG_K_VALUES\""
CMD+=" --embedding-models \"$EMBEDDING_MODELS\""
CMD+=" --output-dir \"$OUTPUT_DIR\""

# Add skip options
if $SKIP_DATA_FETCHING; then
    CMD+=" --skip-data-fetching"
fi

if $SKIP_KB_BUILDING; then
    CMD+=" --skip-kb-building"
fi

if $SKIP_FINETUNING; then
    CMD+=" --skip-finetuning"
fi

# Add other options
if $RANDOM_SAMPLE; then
    CMD+=" --random-sample"
fi

if $NO_COMBINATIONS; then
    CMD+=" --no-combinations"
fi

if [ $LIMIT -gt 0 ]; then
    CMD+=" --limit $LIMIT"
fi

# Configure experiment type
case $EXPERIMENT_TYPE in
    full)
        # Full experiment - all combinations
        echo "Running full experiment with all dimensions"
        # Keep dimensions empty to use all
        ;;
    quick)
        # Quick experiment - no combinations
        echo "Running quick experiment with single dimensions"
        CMD+=" --no-combinations"
        # Keep dimensions empty to use all individually
        ;;
    model)
        # Model-only experiment
        echo "Running model comparison experiment"
        DIMENSIONS="model"
        ;;
    rag)
        # RAG-only experiment
        echo "Running RAG parameter experiment"
        DIMENSIONS="knowledge_base"
        ;;
    system)
        # System-only experiment
        echo "Running system types experiment"
        DIMENSIONS="system"
        ;;
    verification)
        # Verification-only experiment
        echo "Running verification methods experiment"
        DIMENSIONS="verification"
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        show_help
        exit 1
        ;;
esac

# Add dimensions if specified
if [ -n "$DIMENSIONS" ]; then
    CMD+=" --dimensions $DIMENSIONS"
fi

# Print the command that will be executed
echo "Executing: $CMD"
echo "This may take a while to complete..."
echo ""

# Execute the command
eval $CMD

# Suggest next steps
echo ""
echo "Experiment completed. To analyze the results:"
echo "python analyze_experiment_results.py --experiment-dir $OUTPUT_DIR/<experiment_batch_dir>" 